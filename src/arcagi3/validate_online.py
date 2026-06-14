"""Validate the agent against the REAL public games via the ARC-AGI-3 online API.

Requires an API key. Put it in a `.env` file at the repo root:  ARC_API_KEY=...
(register at https://three.arcprize.org). The key is never committed (.env is gitignored).

Usage:
    uv run python -m arcagi3.validate_online                 # all available games
    uv run python -m arcagi3.validate_online --game ls20 --budget 4000
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()  # load ARC_API_KEY from .env

from arc_agi import Arcade, OperationMode  # noqa: E402

from .runner import run_reactive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None, help="single game id (default: all available)")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0, help="cap number of games (0=all)")
    args = ap.parse_args()

    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not set. Add it to a .env file or export it.\n"
              "Register at https://three.arcprize.org")
        return

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("arcagi3.validate_online")
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logger)

    try:
        envs = client.get_environments()
        all_games = [e.game_id for e in envs]
    except Exception as e:
        print(f"Could not list games from the API: {e}")
        return
    print(f"{len(all_games)} games available: {all_games}")

    if args.game:
        prefix = args.game
        games = [g for g in all_games if g.startswith(prefix)] or [args.game]
    else:
        games = all_games
    if args.max_games:
        games = games[: args.max_games]

    card = client.open_scorecard(tags=["dev-validate"])
    rows = []
    t0 = time.time()
    for gid in games:
        try:
            env = client.make(game_id=gid, scorecard_id=card)
            if env is None:
                print(f"{gid:>14}  (could not make env)")
                continue
            r = run_reactive(env, gid, args.budget, args.seed)
            rows.append(r)
            print(f"{r.game_id:>14}  levels {r.levels_completed}/{r.win_levels}  "
                  f"actions {r.actions:>5}  won={r.won}  ({r.reason})", flush=True)
        except Exception as e:
            print(f"{gid:>14}  ERROR: {e}", flush=True)

    if rows:
        print("-" * 60)
        print(f"TOTAL  levels {sum(r.levels_completed for r in rows)}  "
              f"actions {sum(r.actions for r in rows)}  "
              f"wins {sum(1 for r in rows if r.won)}/{len(rows)}  "
              f"elapsed {time.time()-t0:.1f}s")
    try:
        client.close_scorecard(card)
    except Exception:
        pass


if __name__ == "__main__":
    main()
