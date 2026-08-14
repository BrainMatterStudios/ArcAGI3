"""Stage 2b — THE FULL-25 EVALUATION (design doc §4 Stage-2 milestones b+c).

Engineered agent (effects ON: T0 + T1/T2 + move_blocked + win predicate +
audit share + commit mode) on all 25 public games, 4000-action budget,
scored on the TRUE objective and compared against the duck BASE_ENV banked
waves. Pre-registered Stage-2 criteria (design doc §4 + AMENDMENT 1):

  (b) full-25 true-objective mean >= duck BASE_ENV mean;
  (c) outright wins (engineered > duck) on >= 5 games;
  KILL if (b) misses by > 20% with (c) < 3.

Scoring provenance:
  * per-level score  = min(115, 100*(baseline/actions)^2) if completed
  * per-game         = sum(score_i * (i+1)) / sum(i+1), capped by the
                       completion share  (exact reimplementation of
                       submission/_ab_patch_closure/pc_driver.py::pc_env_score;
                       reimplemented because pc_driver imports heavy serving
                       deps at module top)
  * duck per game    = MAX over clones of the stored row score
                       (true_score.py convention), waves pc_base + w2_base
                       (scratchpad/banked_waves_20260809/)
  * mean             = over the 25 games.

Hybrid line: per-game max(engineered, duck) = the dispatch-floor projection,
plus the REALIZABLE hybrid under the pre-registered dispatch rule
(engineered on the 13 measured frame-Markov games minus the AMENDMENT-1
duck-keeps set {ka59, re86, sb26}; duck everywhere else — routable live via
battery archetype + early Markov-violation signal).

Writes scratchpad/engineered_stage2/stage2b_full25.json + .md.
Run: .venv/bin/python src/engineered/evaluate_stage2b.py [--workers N] [stems...]
"""
from __future__ import annotations

import glob as _glob
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.envs import repo_root
from engineered.evaluate_stage1 import (
    BUDGET,
    FRAME_MARKOV,
    WALL_S,
    human_medians,
)

ALL_25 = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]
# AMENDMENT 1: the hybrid dispatch floor stands — duck keeps these
DUCK_KEEPS = {"ka59", "re86", "sb26"}
ENGINEERED_ROUTE = [g for g in FRAME_MARKOV if g not in DUCK_KEEPS]


# --- true objective (pc_env_score reimplementation, provenance in docstring) ---

def true_score(levels_completed: int, actions_per_level: list[int],
               baselines: list[int]) -> float | None:
    n = len(baselines)
    if not n:
        return None
    total_w, total_score, max_w = 0, 0.0, 0
    for i in range(n):
        w = i + 1
        total_w += w
        a = actions_per_level[i] if i < len(actions_per_level) else 0
        completed = i < (levels_completed or 0)
        s = min(115.0, (baselines[i] / a) ** 2 * 100.0) \
            if (completed and a > 0) else 0.0
        if s > 0:
            max_w += w
        total_score += s * w
    return round(min(total_score / total_w, max_w / total_w * 100.0), 4)


def load_baselines() -> dict[str, list[int]]:
    env_dir = repo_root() / "environment_files"
    out: dict[str, list[int]] = {}
    for meta in (_glob.glob(f"{env_dir}/*/metadata.json")
                 + _glob.glob(f"{env_dir}/*/*/metadata.json")):
        try:
            d = json.loads(Path(meta).read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get("game_id") and d.get("baseline_actions"):
            out[d["game_id"].split("-")[0][:4]] = list(d["baseline_actions"])
    return out


def duck_banked_full() -> dict[str, dict[str, float]]:
    """Per game: max levels AND max true score over clones (pc_base+w2_base)."""
    out: dict[str, dict[str, float]] = {}
    for wave in ("pc_base", "w2_base"):
        path = (repo_root() / "scratchpad" / "banked_waves_20260809"
                / f"{wave}.json")
        for row in json.loads(path.read_text())["rows"]:
            g = row["source_game"]
            d = out.setdefault(g, {"levels": 0, "score": 0.0})
            d["levels"] = max(d["levels"], int(row["levels_completed"]))
            if row.get("score") is not None:
                d["score"] = max(d["score"], float(row["score"]))
    return out


# --- one game, one process ------------------------------------------------------

def run_game(stem: str) -> dict[str, Any]:
    from engineered.agent import AgentConfig, EngineeredAgent
    from engineered.envs import open_arcade, resolve_game_ids

    t0 = time.time()
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    env = arcade.make(game_id=gid_of[stem], scorecard_id=f"eng2b-{stem}")
    agent = EngineeredAgent(AgentConfig(budget=BUDGET, wall_s=WALL_S,
                                        effects_enabled=True))
    r = agent.play(env, gid_of[stem])
    return {
        "stem": stem,
        "levels": r.levels_completed,
        "win_levels": r.win_levels,
        "won": r.won,
        "actions_total": r.actions_total,
        "per_level_actions": {str(k): v for k, v in
                              sorted(r.per_level_actions.items())},
        "apl_mean": round(r.apl_mean, 1) if r.apl_mean is not None else None,
        "battery_actions": r.battery_actions,
        "plan_mismatches": r.plan_mismatches,
        "rule_mispredictions": r.rule_mispredictions,
        "predicted_steps_taken": r.predicted_steps_taken,
        "audit_plans": r.audit_plans,
        "win_pred_attempts": r.win_pred_attempts,
        "win_pred_hits": r.win_pred_hits,
        "violation_events": r.violation_events,
        "violated_states": r.violated_states,
        "archetype": r.archetype,
        "end_reason": r.end_reason,
        "wall_s": round(time.time() - t0, 1),
        "rules_gated": sorted(
            f"{x['family']}:{x['scope']}" for x in r.effect_stats
            if x.get("gated")),
    }


# --- aggregation ---------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = load_baselines()
    duck = duck_banked_full()
    human = human_medians()

    for row in rows:
        stem = row["stem"]
        bl = baselines[stem]
        apl_list = [row["per_level_actions"].get(str(i), 0)
                    for i in range(len(bl))]
        row["true_score"] = true_score(row["levels"], apl_list, bl)
        row["duck_levels"] = int(duck[stem]["levels"])
        row["duck_score"] = duck[stem]["score"]
        row["human_apl"] = human[stem]
        row["apl_ratio"] = (round(row["apl_mean"] / human[stem], 2)
                            if row["apl_mean"] else None)
        row["outright_win_score"] = row["true_score"] > row["duck_score"]
        row["outright_win_levels"] = row["levels"] > row["duck_levels"]
        row["route"] = ("engineered" if stem in ENGINEERED_ROUTE else "duck")
        row["hybrid_max"] = max(row["true_score"], row["duck_score"])
        row["hybrid_rule"] = (row["true_score"] if row["route"] == "engineered"
                              else row["duck_score"])

    n = len(rows)
    ours = sum(r["true_score"] for r in rows) / n
    duck_mean = sum(r["duck_score"] for r in rows) / n
    hybrid_max = sum(r["hybrid_max"] for r in rows) / n
    hybrid_rule = sum(r["hybrid_rule"] for r in rows) / n
    wins_score = sum(r["outright_win_score"] for r in rows)
    wins_levels = sum(r["outright_win_levels"] for r in rows)

    b_pass = ours >= duck_mean
    b_miss_frac = (duck_mean - ours) / duck_mean if duck_mean > 0 else 0.0
    c_pass = wins_score >= 5
    kill = (b_miss_frac > 0.20) and (wins_score < 3)
    verdict = ("PASS" if (b_pass and c_pass)
               else ("KILL" if kill else "MARGINAL"))

    return {
        "protocol": {
            "games": [r["stem"] for r in rows],
            "budget": BUDGET, "wall_s": WALL_S,
            "objective": "min(115,100*(b/a)^2) weighted (level+1), capped by "
                         "completion share; duck = max over clones "
                         "(pc_base+w2_base); mean over 25 games",
            "criteria": "(b) mean >= duck mean; (c) outright true-score wins "
                        ">= 5; KILL if (b) miss > 20% AND (c) < 3",
            "dispatch_rule": "engineered on measured frame-Markov 13 minus "
                             "duck-keeps {ka59,re86,sb26} (AMENDMENT 1); "
                             "duck elsewhere; routable live via battery "
                             "archetype + early Markov-violation signal",
        },
        "means": {
            "engineered": round(ours, 4),
            "duck_base_env": round(duck_mean, 4),
            "hybrid_max": round(hybrid_max, 4),
            "hybrid_rule": round(hybrid_rule, 4),
        },
        "milestone_b": {
            "pass": b_pass,
            "miss_fraction": round(b_miss_frac, 4),
        },
        "milestone_c": {
            "outright_wins_true_score": wins_score,
            "outright_wins_levels": wins_levels,
            "pass": c_pass,
        },
        "verdict": verdict,
        "rows": rows,
    }


def to_markdown(p: dict[str, Any]) -> str:
    m = p["means"]
    rows = p["rows"]
    lines = [
        "# Stage-2b — FULL-25 verdict: engineered agent vs duck BASE_ENV",
        "",
        f"Protocol: all 25 public games, budget {p['protocol']['budget']} "
        f"actions/game, wall cap {p['protocol']['wall_s']} s, effects ON "
        "(T0 + T1/T2 + move_blocked + win predicate + audit share). "
        "True objective throughout; duck = banked max-over-clones "
        "(pc_base + w2_base).",
        "",
        "## Table 1 — per-game levels, efficiency, true score",
        "",
        "| game | lv ours | lv duck | apl ours | apl human | x-human | "
        "score ours | score duck | hybrid | route | end |",
        "|------|---------|---------|----------|-----------|---------|"
        "-----------|------------|--------|-------|-----|",
    ]
    for r in rows:
        win_mark = " **W**" if r["outright_win_score"] else ""
        lines.append(
            f"| {r['stem']} | {r['levels']}/{r['win_levels']} | "
            f"{r['duck_levels']} | {r['apl_mean']} | {r['human_apl']} | "
            f"{r['apl_ratio']} | {r['true_score']}{win_mark} | "
            f"{r['duck_score']} | {r['hybrid_max']} | {r['route'][:4]} | "
            f"{r['end_reason']} |")
    b, c = p["milestone_b"], p["milestone_c"]
    lines += [
        "",
        "## Means (true objective, 25 games)",
        "",
        f"| engineered | duck BASE_ENV | hybrid (per-game max) | "
        f"hybrid (dispatch rule) |",
        f"|------------|---------------|----------------------|"
        f"------------------------|",
        f"| **{m['engineered']}** | {m['duck_base_env']} | "
        f"{m['hybrid_max']} | {m['hybrid_rule']} |",
        "",
        "## Stage-2 verdict (pre-registered criteria)",
        "",
        f"* (b) full-25 true mean >= duck mean: **{'PASS' if b['pass'] else 'FAIL'}** "
        f"(engineered {m['engineered']} vs duck {m['duck_base_env']}; "
        f"miss fraction {b['miss_fraction']:.1%})",
        f"* (c) outright true-score wins >= 5: "
        f"**{'PASS' if c['pass'] else 'FAIL'}** "
        f"({c['outright_wins_true_score']} score wins; "
        f"{c['outright_wins_levels']} level wins)",
        f"* KILL rule ((b) miss > 20% AND (c) < 3): "
        f"{'TRIGGERED' if p['verdict'] == 'KILL' else 'not triggered'}",
        "",
        f"**VERDICT: {p['verdict']}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    args = sys.argv[1:]
    workers = 5
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    stems = args or ALL_25

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_game, s): s for s in stems}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(f"[{time.time() - t0:7.1f}s] {row['stem']}: "
                  f"levels={row['levels']}/{row['win_levels']} "
                  f"actions={row['actions_total']} end={row['end_reason']} "
                  f"audits={row['audit_plans']} "
                  f"wp={row['win_pred_hits']}/{row['win_pred_attempts']} "
                  f"wall={row['wall_s']}s", flush=True)
    rows.sort(key=lambda r: r["stem"])

    payload = aggregate(rows)
    out_dir = repo_root() / "scratchpad" / "engineered_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2b_full25.json").write_text(json.dumps(payload, indent=1))
    md = to_markdown(payload)
    (out_dir / "stage2b_full25.md").write_text(md)
    print()
    print(md)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
