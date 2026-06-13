"""Run the ExplorerAgent against local (offline) ARC-AGI-3 environments and report.

Usage:
    uv run python -m arcagi3.runner --games-dir src/arcagi3/games [--game navg] [--budget 4000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "local-dev")

from arc_agi import Arcade, OperationMode  # noqa: E402

from .agent import ExplorerAgent, HybridAgent, PlayResult  # noqa: E402

AGENTS = {"explorer": ExplorerAgent, "hybrid": HybridAgent}


def discover_games(games_dir: str) -> list[str]:
    out = []
    for meta in sorted(Path(games_dir).rglob("metadata.json")):
        try:
            out.append(json.loads(meta.read_text())["game_id"])
        except Exception:
            pass
    return out


def run_game(game_id: str, games_dir: str, budget: int, seed: int = 0,
             agent_name: str = "hybrid") -> PlayResult:
    logger = logging.getLogger("arcagi3.runner")
    client = Arcade(
        operation_mode=OperationMode.OFFLINE, environments_dir=games_dir, logger=logger
    )
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    if env is None:
        raise RuntimeError(f"could not make env for {game_id}")
    agent = AGENTS[agent_name](max_actions=budget, seed=seed)
    return agent.play(env, game_id=game_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", default="src/arcagi3/games")
    ap.add_argument("--game", default=None, help="single game id (default: all)")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default="hybrid", choices=list(AGENTS))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.WARNING)

    games = [args.game] if args.game else discover_games(args.games_dir)
    if not games:
        print(f"No games found under {args.games_dir}")
        return

    rows = []
    t0 = time.time()
    for gid in games:
        r = run_game(gid, args.games_dir, args.budget, args.seed, args.agent)
        rows.append(r)
        print(
            f"{r.game_id:>10}  levels {r.levels_completed}/{r.win_levels}  "
            f"actions {r.actions:>5}  won={r.won}  states={r.states_seen}  ({r.reason})"
        )

    total_lvls = sum(r.levels_completed for r in rows)
    total_acts = sum(r.actions for r in rows)
    wins = sum(1 for r in rows if r.won)
    print("-" * 64)
    print(
        f"TOTAL  levels {total_lvls}  actions {total_acts}  wins {wins}/{len(rows)}  "
        f"elapsed {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
