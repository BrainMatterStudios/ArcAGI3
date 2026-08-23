#!/usr/bin/env python3
"""Stage 1 — bankruptcy-judge trigger tuning against pre-registered bars.

Pre-registered bars (task order of 2026-08-24, from the §B design):
  BAR-1  recall of labeled terminal spirals >= 7/8 of the label set
         (with N labels the equivalent bar is ceil(0.875*N)/N)
  BAR-2  <= 35% of all fires land on eventually-completed levels
  BAR-3  <= 3 fires/session (hard cap; also measured pre-cap)

Trigger (replayed over transcript blocks, i.e. per LLM call, firing BEFORE the call):
  state per session: actions_since_level_start (asl), blocks_since_level_start (bsl),
  both reset on level change and on a fire (the judge rebuilds the model, so the
  fresh model earns a fresh window).
  FIRE at a block iff  (asl >= A[archetype])  OR  (T_stall and bsl >= T_stall)
  subject to: >= cooldown actions AND >= cooldown_blocks blocks since last fire,
  < cap_level fires on this level, < cap_session fires this session, and a carried
  world model exists at the block (the judge needs a model to judge).

The OR-branch on blocks exists because measurement showed action-light,
turn-heavy spirals (digest1/sb26 L2: 43 stale turns over ~24 actions) that an
action-only window can never reach. A[.] starting point per the design:
A_click=45, A_avatar=30 (A_mixed tunable), cooldown 20 actions, cap 2/level 3/session.

Usage:
  .venv/bin/python stage1_tune.py            # sweep + report + stage1_results.json
  .venv/bin/python stage1_tune.py --confusion A_click=.. A_avatar=.. ...  # one config detail
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass

import corpus_lib as cl

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Params:
    A_click: int = 45
    A_avatar: int = 30
    A_mixed: int = 45
    T_stall: int | None = None      # blocks since level start; None disables the OR-branch
    cooldown: int = 20              # actions since last fire
    cooldown_blocks: int = 8        # blocks since last fire
    cap_level: int = 2
    cap_session: int = 3

    def window(self, arch: str) -> int:
        return {"CLICK": self.A_click, "AVATAR": self.A_avatar, "MIXED": self.A_mixed}[arch]


def replay_trigger(t: cl.Transcript, arch: str, p: Params):
    """Returns (fires, natural_fire_count). Each fire: dict(block, level, cum_actions, asl, bsl, reason)."""
    fires = []
    natural = 0
    prev_level = None
    level_start_action = 0
    level_start_block = 0
    last_fire_action = -(10 ** 9)
    last_fire_block = -(10 ** 9)
    fires_this_level = 0
    for b in t.blocks:
        if b.level != prev_level:
            prev_level = b.level
            level_start_action = b.cum_actions
            level_start_block = b.idx
            fires_this_level = 0
        asl = b.cum_actions - level_start_action
        bsl = b.idx - level_start_block
        act_hit = asl >= p.window(arch)
        stall_hit = p.T_stall is not None and bsl >= p.T_stall
        if not (act_hit or stall_hit):
            continue
        natural += 1
        if b.cum_actions - last_fire_action < p.cooldown:
            continue
        if b.idx - last_fire_block < p.cooldown_blocks:
            continue
        if fires_this_level >= p.cap_level or len(fires) >= p.cap_session:
            continue
        if b.wm_hash == "-":
            continue  # nothing carried to judge
        fires.append({
            "block": b.idx, "level": b.level, "cum_actions": b.cum_actions,
            "asl": asl, "bsl": bsl,
            "reason": "actions" if act_hit else "stall",
        })
        last_fire_action = b.cum_actions
        last_fire_block = b.idx
        fires_this_level += 1
        # judge rebuild resets the clocks
        level_start_action = b.cum_actions
        level_start_block = b.idx
    return fires, natural


MERGED_MIN_TURNS = 25  # pre-registered cut: the sorted merged-episode sizes are
# 43,39,35,35,33,32,31,29,25 | 20,17,16,15 — the largest tail gap (25 vs 20) sets
# the boundary, yielding 9 labeled terminal spirals (design §B estimated ~8).
# The sub-cut stuck levels remain never-completed, so fires there are NOT counted
# as false fires; they are just not required for recall.


def load_labels():
    with open(os.path.join(HERE, "labels_terminal_spirals.json")) as f:
        raw = json.load(f)["labels"]
    # merge episodes per (sid, level): one terminal spiral = one stuck level
    merged = {}
    for r in raw:
        k = (r["sid"], r["level"])
        m = merged.setdefault(k, dict(r, episodes=0, turns=0))
        m["episodes"] += 1
        m["turns"] += r["turns"]
        m["onset_block"] = min(m["onset_block"], r["onset_block"])
        m["onset_action"] = min(m["onset_action"], r["onset_action"])
        m["end_block"] = max(m["end_block"], r["end_block"])
        m["end_action"] = max(m["end_action"], r["end_action"])
    out = [m for m in merged.values() if m["turns"] >= MERGED_MIN_TURNS]
    return sorted(out, key=lambda r: -r["turns"])


def evaluate(p: Params, corpus, labels):
    label_keys = {(r["sid"], r["level"]) for r in labels}
    hit = set()
    hit_strict = set()
    all_fires = []
    per_session = {}
    for t, arch in corpus:
        fires, natural = replay_trigger(t, arch, p)
        per_session[t.sid] = {"fires": len(fires), "natural": natural}
        for f in fires:
            f2 = dict(f, sid=t.sid, arch=arch,
                      eventually_completed=t.level_eventually_completed(f["level"]))
            all_fires.append(f2)
            k = (t.sid, f["level"])
            if k in label_keys:
                hit.add(k)
                lab = next(r for r in labels if (r["sid"], r["level"]) == k)
                if f["block"] >= lab["onset_block"]:
                    hit_strict.add(k)
    n_fires = len(all_fires)
    n_completed = sum(1 for f in all_fires if f["eventually_completed"])
    max_fires = max((v["fires"] for v in per_session.values()), default=0)
    return {
        "params": p.__dict__,
        "recall": len(hit),
        "recall_strict": len(hit_strict),
        "n_labels": len(labels),
        "missed": sorted(f"{s}#L{l}" for (s, l) in label_keys - hit),
        "fires_total": n_fires,
        "fires_on_completed_levels": n_completed,
        "completed_share": (n_completed / n_fires) if n_fires else 0.0,
        "max_fires_per_session": max_fires,
        "mean_fires_per_session": n_fires / len(per_session),
        "all_fires": all_fires,
        "per_session": per_session,
    }


def bars(ev):
    need = math.ceil(0.875 * ev["n_labels"])
    return {
        "recall_bar": f">={need}/{ev['n_labels']}",
        "recall_pass": ev["recall"] >= need,
        "completed_share_pass": ev["completed_share"] <= 0.35,
        "cap_pass": ev["max_fires_per_session"] <= 3,
    }


def main():
    labels = load_labels()
    corpus = [(t, cl.archetype(t.run_dir, t.game_key)) for t in cl.list_corpus()]

    print(f"label set: {len(labels)} terminal spirals (merged per stuck level)")
    for r in labels:
        print(f"  {r['sid']:30s} L{r['level']} {r['archetype']:6s} stale_turns={r['turns']:3d} "
              f"blocks {r['onset_block']}..{r['end_block']} actions {r['onset_action']}..{r['end_action']}")

    grid = []
    for A_click, A_avatar, A_mixed, T_stall, cooldown in itertools.product(
        [30, 35, 40, 45, 55], [20, 25, 30, 40], [30, 40, 45, 55],
        [None, 12, 15, 18, 22], [15, 20, 25],
    ):
        grid.append(Params(A_click, A_avatar, A_mixed, T_stall, cooldown))

    rows = []
    for p in grid:
        ev = evaluate(p, corpus, labels)
        b = bars(ev)
        ev["bars"] = b
        ev["passes_all"] = all(b[k] for k in ("recall_pass", "completed_share_pass", "cap_pass"))
        rows.append(ev)

    passing = [r for r in rows if r["passes_all"]]

    def dist_from_start(r):
        p = r["params"]
        return (abs(p["A_click"] - 45) + abs(p["A_avatar"] - 30) + abs(p["A_mixed"] - 45)
                + abs(p["cooldown"] - 20))

    # among passing: max recall, then fewest fires (decode cost), then lowest
    # false-fire share, then closest to the pre-registered starting point
    passing.sort(key=lambda r: (
        -r["recall"], r["fires_total"], r["completed_share"], -r["recall_strict"],
        dist_from_start(r),
        tuple(sorted((k, str(v)) for k, v in r["params"].items())),
    ))

    print(f"\nswept {len(rows)} configs; {len(passing)} pass all three bars")

    def fmt(r, name=""):
        p = r["params"]
        return (f"{name}A_c={p['A_click']:2d} A_a={p['A_avatar']:2d} A_m={p['A_mixed']:2d} "
                f"T={str(p['T_stall']):>4s} cd={p['cooldown']:2d} | "
                f"recall={r['recall']}/{r['n_labels']} (strict {r['recall_strict']}) "
                f"fires={r['fires_total']:3d} on-completed={r['fires_on_completed_levels']:3d} "
                f"({r['completed_share']:.1%}) max/sess={r['max_fires_per_session']}")

    if passing:
        print("\ntop passing configs (fires ascending):")
        for r in passing[:10]:
            print("  " + fmt(r))
        best = passing[0]
    else:
        # Pareto frontier: maximize recall, minimize completed_share, minimize fires
        frontier = []
        for r in rows:
            dominated = any(
                (o["recall"] >= r["recall"] and o["completed_share"] <= r["completed_share"]
                 and o["fires_total"] <= r["fires_total"]
                 and (o["recall"], -o["completed_share"], -o["fires_total"])
                 != (r["recall"], -r["completed_share"], -r["fires_total"])
                 and (o["recall"] > r["recall"] or o["completed_share"] < r["completed_share"]
                      or o["fires_total"] < r["fires_total"]))
                for o in rows)
            if not dominated:
                frontier.append(r)
        frontier.sort(key=lambda r: (-r["recall"], r["completed_share"]))
        print("\nNO config passes all bars — Pareto frontier:")
        for r in frontier[:15]:
            print("  " + fmt(r))
        best = frontier[0]

    print("\nSELECTED: " + fmt(best))
    print("missed spirals:", best["missed"] or "none")
    print("\nconfusion detail (selected config):")
    label_keys = {(r["sid"], r["level"]) for r in labels}
    tp = [f for f in best["all_fires"] if (f["sid"], f["level"]) in label_keys]
    fp_completed = [f for f in best["all_fires"] if f["eventually_completed"]]
    other = [f for f in best["all_fires"]
             if (f["sid"], f["level"]) not in label_keys and not f["eventually_completed"]]
    print(f"  fires on labeled terminal spirals : {len(tp)}")
    print(f"  fires on eventually-completed lvls: {len(fp_completed)}  (bar <= 35% of {best['fires_total']})")
    print(f"  fires on other never-completed    : {len(other)}  (unlabeled stuck levels — not penalized)")
    for f in best["all_fires"]:
        tag = ("SPIRAL" if (f["sid"], f["level"]) in label_keys
               else "COMPL " if f["eventually_completed"] else "stuck ")
        print(f"   {tag} {f['sid']:30s} L{f['level']} block={f['block']:3d} "
              f"act={f['cum_actions']:3d} asl={f['asl']:3d} bsl={f['bsl']:3d} via={f['reason']}")

    out = {
        "labels": labels,
        "selected": {k: v for k, v in best.items() if k not in ("per_session",)},
        "passing_count": len(passing),
        "swept": len(rows),
        "top_passing": [
            {k: v for k, v in r.items() if k not in ("all_fires", "per_session")}
            for r in (passing[:10] if passing else [])
        ],
    }
    with open(os.path.join(HERE, "stage1_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {os.path.join(HERE, 'stage1_results.json')}")


if __name__ == "__main__":
    main()
