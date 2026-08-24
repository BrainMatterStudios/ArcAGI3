"""SearchCore falsifier harness — runs the nbfs+tiers suite over the 25 dev
fixtures at a parameterized per-game wall budget and emits the levels/cracks
table as JSON + markdown, side by side with the probe's measured baseline
(scratchpad/searchtheory_20260823/results, compiled 2026-08-23).

Usage:
  .venv/bin/python submission/_search_core/run_falsifier.py \
      --budget 120 --backend snapshot --jobs 5 [--games tu93,vc33] [--tag screen]

The 25-game list is the intersection of the offline environment_files/ dirs
with the win fixtures (submission/_explorer_floor/fixtures/ +
scratchpad/testing_20260822/human_fixtures/) — currently all 25.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ["ONLY_RESET_LEVELS"] = "true"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT_DIR = os.path.join(_HERE, "results")

# probe nbfs baseline (16-click cap, snapshot backend), compiled from
# scratchpad/searchtheory_20260823/results/*_nbfs*.json. 120 s/game entries
# sum to 29 (the doc §A head-to-head); three games only have longer-budget
# nbfs runs on file (dc22 3@240s, lp85 1@240s, r11l 1@180s -> union 32).
PROBE_NBFS_120 = {
    "ar25": 2, "bp35": 0, "cd82": 2, "cn04": 0, "dc22": 0, "ft09": 0,
    "g50t": 0, "ka59": 1, "lf52": 2, "lp85": 0, "ls20": 3, "m0r0": 2,
    "r11l": 0, "re86": 0, "s5i5": 2, "sb26": 0, "sc25": 0, "sk48": 1,
    "sp80": 1, "su15": 0, "tn36": 0, "tr87": 0, "tu93": 9, "vc33": 4,
    "wa30": 0,
}
PROBE_LONGER = {"dc22": "3@240s", "lp85": "1@240s", "r11l": "1@180s"}
PROBE_UNION = 36   # probe suite union incl. longer budgets + gex (doc §A)

# stage-2 falsifier baseline (nbfs+tiers, snapshot, 120 s/game, commit'd run
# results/falsifier_screen120final_20260823_232018.md): 30 levels, 1 crack.
STAGE2_120 = {
    "ar25": 2, "bp35": 0, "cd82": 2, "cn04": 0, "dc22": 2, "ft09": 1,
    "g50t": 0, "ka59": 1, "lf52": 1, "lp85": 1, "ls20": 1, "m0r0": 2,
    "r11l": 0, "re86": 0, "s5i5": 2, "sb26": 0, "sc25": 0, "sk48": 1,
    "sp80": 1, "su15": 0, "tn36": 0, "tr87": 0, "tu93": 9, "vc33": 4,
    "wa30": 0,
}

# live-transfer gate: reset-replay simulation at the shared gateway rate
GATEWAY_ACT_PER_S = 130.0
GRIND_ENVELOPE_S = 2700.0        # one grind run's wall budget


def live_sim_seconds(levels: list[dict]) -> float:
    """Simulated reset-replay cost of the found solution search:
    sum over solved levels of nodes * (depth/2 + 1) / 130 act/s
    (NOTES.md live-cost rule)."""
    total = 0.0
    for L in levels:
        if not L.get("solved") or "nodes" not in L:
            continue
        d = L.get("depth", 0) or 0
        total += L["nodes"] * (d / 2.0 + 1.0) / GATEWAY_ACT_PER_S
    return total


def fixture_games() -> list[str]:
    stems = set()
    for d in (os.path.join(ROOT, "submission/_explorer_floor/fixtures"),
              os.path.join(ROOT, "scratchpad/testing_20260822/human_fixtures")):
        if os.path.isdir(d):
            stems.update(f[:-5] for f in os.listdir(d) if f.endswith(".json")
                         and not f.startswith("_"))
    envs = set(os.listdir(os.path.join(ROOT, "environment_files")))
    return sorted(stems & envs)


def _worker(args: tuple) -> dict:
    stem, budget, backend, max_states, max_tier, algo = args
    os.environ["ONLY_RESET_LEVELS"] = "true"
    import logging

    logging.disable(logging.CRITICAL)
    from search_core import run_game

    try:
        return run_game(stem, budget, backend=backend, algo=algo,
                        max_states=max_states, max_tier=max_tier)
    except Exception as e:  # noqa: BLE001
        return dict(game=stem, error=repr(e), levels_won=0, cracked=False)


def markdown_table(results: list[dict], budget: int) -> str:
    lines = [
        "| game | levels | crack | stage2@120s | probe@120s | arch | lane |"
        " tiers | states(last) | live-sim | wall | notes |",
        "|------|--------|-------|-------------|------------|------|------|"
        "-------|--------------|----------|------|-------|",
    ]
    total = 0
    cracks = 0
    stage2_total = sum(STAGE2_120.values())
    probe_total = sum(PROBE_NBFS_120.values())
    for r in sorted(results, key=lambda x: x["game"]):
        g = r["game"]
        if "error" in r:
            lines.append(
                f"| {g} | ERR | | {STAGE2_120.get(g, '?')} "
                f"| {PROBE_NBFS_120.get(g, '?')} | | | | | | "
                f"| {r['error'][:40]} |")
            continue
        total += r["levels_won"]
        cracks += bool(r.get("cracked"))
        tiers = sorted({L.get("tier") for L in r["levels"] if L.get("tier")})
        last = next((L for L in reversed(r["levels"]) if "states" in L), {})
        note = PROBE_LONGER.get(g, "")
        unsolved = next((L for L in r["levels"] if not L.get("solved", True)
                         and "reason" in L), None)
        if unsolved:
            note = (note + " " if note else "") + unsolved["reason"]
        lanes = sorted({L.get("algo") for L in r["levels"]
                        if L.get("solved") and L.get("algo")})
        if any(L.get("ignition") for L in r["levels"]):
            note = (note + " " if note else "") + "ignition"
        sim = ""
        if r.get("cracked"):
            s = live_sim_seconds(r["levels"])
            gate = "PASS" if s <= GRIND_ENVELOPE_S else "FAIL"
            sim = f"{s / 60:.1f}m {gate}"
        lines.append(
            f"| {g} | {r['levels_won']} | {'WIN' if r.get('cracked') else ''} "
            f"| {STAGE2_120.get(g, '?')} | {PROBE_NBFS_120.get(g, '?')} "
            f"| {r.get('archetype', '')[:2]} | {'+'.join(lanes)} | {tiers} "
            f"| {last.get('states', '')} | {sim} | {r['wall']}s "
            f"| {note.strip()} |")
    lines.append(
        f"| **TOTAL** | **{total}** | **{cracks} crack(s)** "
        f"| **{stage2_total}** | **{probe_total}** | | | | | | "
        f"| budget {budget}s/game; probe union {PROBE_UNION} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--backend", default="snapshot",
                    choices=["snapshot", "reset_replay"])
    ap.add_argument("--algo", default="portfolio",
                    choices=["nbfs", "nbfs_macros", "goexplore", "portfolio"])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--games", default="")
    ap.add_argument("--max-states", type=int, default=20000)
    ap.add_argument("--max-tier", type=int, default=4)
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args()

    games = args.games.split(",") if args.games else fixture_games()
    os.makedirs(OUT_DIR, exist_ok=True)
    work = [(g, args.budget, args.backend, args.max_states, args.max_tier,
             args.algo)
            for g in games]
    t0 = time.time()
    if args.jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            results = []
            for res in pool.imap_unordered(_worker, work):
                results.append(res)
                print(json.dumps({k: v for k, v in res.items() if k != "levels"}),
                      flush=True)
    else:
        results = []
        for w in work:
            res = _worker(w)
            results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "levels"}),
                  flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = dict(tag=args.tag, budget_s=args.budget, backend=args.backend,
                   algo=args.algo,
                   jobs=args.jobs, wall=round(time.time() - t0, 1),
                   total_levels=sum(r.get("levels_won", 0) for r in results),
                   cracks=[r["game"] for r in results if r.get("cracked")],
                   results=sorted(results, key=lambda r: r["game"]))
    jpath = os.path.join(OUT_DIR, f"falsifier_{args.tag}_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=1)
    md = markdown_table(results, args.budget)
    mpath = os.path.join(OUT_DIR, f"falsifier_{args.tag}_{stamp}.md")
    with open(mpath, "w") as f:
        f.write(f"# SearchCore falsifier — {args.backend} @ {args.budget}s/game "
                f"({stamp})\n\n{md}\n")
    print(f"\nTOTAL levels: {payload['total_levels']}  cracks: {payload['cracks']}")
    print(f"wrote {jpath}\nwrote {mpath}")
    print(md)


if __name__ == "__main__":
    main()
