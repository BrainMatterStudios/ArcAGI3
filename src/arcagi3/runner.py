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

# Lazy import so that wm_policy.py is only loaded when needed (avoids import cost on
# hot paths). The 'wm' agent name is the only entry point for WorldModelPolicy.


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
    if agent_name == "reactive":
        from .policy import HybridPolicy as _HP
        return run_reactive(env, game_id, budget, seed, policy_cls=_HP)
    if agent_name == "salience":
        from .salience_explorer import SalienceExplorer as _SE
        return run_reactive(env, game_id, budget, seed, policy_cls=_SE)
    if agent_name == "wm":
        # policy_cls=None -> make_policy() -> respects ARCAGI3_WORLDMODEL env var
        return run_reactive(env, game_id, budget, seed, policy_cls=None)
    agent = AGENTS[agent_name](max_actions=budget, seed=seed)
    return agent.play(env, game_id=game_id)


def run_reactive(env, game_id: str, budget: int, seed: int = 0,
                 policy_cls=None) -> PlayResult:
    """Drive a reactive policy one action at a time, exactly like the Kaggle framework does.

    policy_cls: if None, uses make_policy() (respects ARCAGI3_WORLDMODEL env var);
                if provided, must be a callable(seed=seed) returning a policy instance.
                Defaults to HybridPolicy when ARCAGI3_WORLDMODEL is unset (D0 gate).
    """
    from arcengine import GameAction, GameState

    if policy_cls is not None:
        pol = policy_cls(seed=seed)
    else:
        from .wm_policy import make_policy
        pol = make_policy(seed=seed)
    obs = env.reset()
    actions = 0
    prev_levels = int(obs.levels_completed or 0)
    win_levels = int(obs.win_levels or 0)
    reason = "budget"
    while actions < budget:
        if obs.state == GameState.WIN:
            reason = "win"
            break
        grid = P_to_grid(obs.frame)
        token = pol.decide(
            grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        if token[0] == "reset":
            obs = env.reset()
        elif token[0] == "S":
            obs = env.step(GameAction.from_id(token[1]))
        else:  # click
            obs = env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]})
        actions += 1
        prev_levels = int(obs.levels_completed or 0)
    return PlayResult(game_id, prev_levels, win_levels, actions,
                      obs.state == GameState.WIN, len(pol.gs.wm) if pol.gs else 0, reason)


def P_to_grid(frame):
    from . import perception as P

    return P.to_grid(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", default="src/arcagi3/games")
    ap.add_argument("--game", default=None, help="single game id (default: all)")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default="reactive", choices=list(AGENTS) + ["reactive", "wm", "salience"])
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
