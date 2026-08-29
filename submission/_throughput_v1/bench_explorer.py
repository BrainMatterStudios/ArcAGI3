#!/usr/bin/env python3
"""CPU benchmark: the frontier explorer ALONE on the 25 public games.

Answers "how many levels does a model-free fallback clear inside our action
budgets?" — the number Pack 4 rests on. Runs the offline arcade
(reference/arc-agi-toolkit + environment_files), one play per game, up to
--budget actions, recording the action index of every level-up.

Usage:
  .venv/bin/python submission/_throughput_v1/bench_explorer.py --budget 800 [--games ls20,vc33] [--seed 0]
Writes submission/_throughput_v1/results/explorer_bench_<budget>_<seed>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "reference/arc-agi-toolkit"))
sys.path.insert(0, str(HERE))

import arc_agi  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

import frontier_explorer as fe  # noqa: E402


def game_names(only: list[str] | None) -> list[str]:
    names = []
    for game_dir in sorted(p for p in (REPO / "environment_files").iterdir() if p.is_dir()):
        if only and game_dir.name not in only:
            continue
        versions = sorted(p.name for p in game_dir.iterdir() if p.is_dir())
        names.append(f"{game_dir.name}-{versions[0]}")
    return names


def as_grid(frame_data) -> tuple:
    import numpy as np

    arr = np.asarray(frame_data.frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return tuple(tuple(int(v) for v in row) for row in arr.tolist())


def play(arcade, name: str, budget: int, seed: int) -> dict:
    env = arcade.make(name)
    assert env is not None, name
    fd = env.reset()
    ex = fe.FrontierExplorer(seed=seed)
    levels_at: list[int] = []
    levels = int(fd.levels_completed)
    n_levels = int(getattr(env.environment_info, "num_levels", 0) or 0)
    key = ex.observe(as_grid(fd), fd.available_actions)
    actions = 0
    resets = 0
    t0 = time.monotonic()
    while actions < budget:
        if fd.state == GameState.WIN:
            break
        if fd.state == GameState.GAME_OVER:
            fd = env.reset()
            actions += 1
            resets += 1
            key = ex.observe(as_grid(fd), fd.available_actions)
            continue
        idx, _why = ex.choose(key)
        cand = ex.nodes[key].candidates[idx]
        fd2 = env.step(GameAction.from_name(cand.action), data=cand.data or None)
        actions += 1
        if fd2 is None:
            break
        fd = fd2
        new_levels = int(fd.levels_completed)
        if new_levels > levels:
            levels = new_levels
            levels_at.append(actions)
            ex.reset()                     # new level: fresh graph and status-bar mask
            key = ex.observe(as_grid(fd), fd.available_actions)
            continue
        new_key = ex.observe(as_grid(fd), fd.available_actions)
        ex.record(key, idx, new_key)
        key = new_key
    return {
        "game": name, "levels": levels, "n_levels": n_levels, "levels_at": levels_at,
        "actions": actions, "resets": resets, "won": fd.state == GameState.WIN,
        "nodes": ex.stats["nodes"], "edges": ex.stats["edges"], "tier": ex.active,
        "wall_s": round(time.monotonic() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--games", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    only = [g for g in args.games.split(",") if g] or None
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(REPO / "environment_files"))
    rows = []
    for name in game_names(only):
        row = play(arcade, name, args.budget, args.seed)
        rows.append(row)
        print(f"{row['game']}: levels={row['levels']}/{row['n_levels']} at={row['levels_at']} "
              f"actions={row['actions']} resets={row['resets']} nodes={row['nodes']} tier={row['tier']} "
              f"wall={row['wall_s']}s", flush=True)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "budget": args.budget, "seed": args.seed, "games": rows,
        "games_with_level1": sum(1 for r in rows if r["levels"] >= 1),
        "total_levels": sum(r["levels"] for r in rows),
        "level1_within": {b: sum(1 for r in rows if r["levels_at"] and r["levels_at"][0] <= b) for b in (50, 100, 200, 400, 800)},
    }
    (out_dir / f"explorer_bench_{args.budget}_{args.seed}.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "games"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
