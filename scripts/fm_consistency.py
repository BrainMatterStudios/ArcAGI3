"""C4 measurement-only harness: one-step forward-model prediction consistency.

Replays (state, action, next_state) transitions on local dev games while the reactive
HybridPolicy plays, and for each transition asks the ForwardModel to predict the next
Scene. Reports, per game:
  * confident-rate: fraction of steps the model made a confident claim (valid & known)
  * known-acc:      among confident claims, fraction whose predicted key == real next key

This is READ-ONLY: it imports forward_model (a NEW module not in the live path) and the
live HybridPolicy purely to source real transitions; it changes NO behavior. The model is
built from the policy's learned motion model + the real C3 affordance learner via
forward_model.c3_adapter (the exact C6/C7 wiring).

Usage (MUST set PYTHONPATH=src; `uv run` only adds the pytest pythonpath):
    PYTHONPATH=src uv run python scripts/fm_consistency.py
    PYTHONPATH=src uv run python scripts/fm_consistency.py navg maze --budget 800
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "local-dev")

import numpy as np  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import forward_model as FM  # noqa: E402
from arcagi3 import perception as P  # noqa: E402
from arcagi3.policy import HybridPolicy  # noqa: E402

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")
DEFAULT_GAMES = ["navg", "maze", "collect", "switchdoor", "push"]


def _step_env(env, tok):
    if tok[0] == "reset":
        return env.reset()
    if tok[0] == "S":
        return env.step(GameAction.from_id(tok[1]))
    return env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})


def measure(game_id: str, budget: int) -> dict:
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                    logger=logging.getLogger("fm_consistency"))
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    pol = HybridPolicy(seed=0, enable_affordance=True)
    obs = env.reset()
    steps = confident = correct = 0
    n = 0
    prev_lvl = int(obs.levels_completed or 0)
    while n < budget and obs.state != GameState.WIN:
        grid = P.to_grid(obs.frame)
        bg = P.detect_background(grid)
        tok = pol.decide(grid, obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        # Build the model exactly as C6/C7 would, from the live learned state.
        pred = None
        if tok[0] in ("S", "C") and pol.mm is not None and pol.mm.ok:
            ignore = frozenset(getattr(pol, "distractor_colors", frozenset()))
            fm = FM.ForwardModel(pol.mm, FM.c3_adapter(pol.aff), ignore_colors=ignore)
            sc = fm.scene(grid, bg)
            pred = fm.predict(sc, tok)

        obs = _step_env(env, tok)
        n += 1
        lvl = int(obs.levels_completed or 0)
        if pred is not None and tok[0] != "reset":
            steps += 1
            if pred.valid and pred.known and lvl == prev_lvl:
                confident += 1
                real_key = P.object_state_key(P.to_grid(obs.frame), bg,
                                              ignore_colors=set(getattr(pol, "distractor_colors",
                                                                        frozenset())))
                if pred.scene.key() == real_key:
                    correct += 1
        prev_lvl = lvl
    return {
        "game": game_id,
        "steps": steps,
        "confident_rate": (confident / steps) if steps else 0.0,
        "known_acc": (correct / confident) if confident else 0.0,
        "actions": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=DEFAULT_GAMES)
    ap.add_argument("--budget", type=int, default=800)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    games = args.games or DEFAULT_GAMES
    print(f"{'game':>12}  {'steps':>6}  {'confident%':>10}  {'known-acc%':>10}")
    print("-" * 46)
    for gid in games:
        r = measure(gid, args.budget)
        print(f"{r['game']:>12}  {r['steps']:>6}  "
              f"{r['confident_rate']*100:>9.0f}%  {r['known_acc']*100:>9.0f}%")


if __name__ == "__main__":
    main()
