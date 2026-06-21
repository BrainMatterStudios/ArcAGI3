"""Drive a game, capture the trajectory, and print the inferred mechanic report.

Increment-1 kill-gate driver. Local toys (maze/push/navg) run OFFLINE (known mechanics for
validation); real games (e.g. ls20) run NORMAL.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/infer_mechanics.py <prefix> [budget] [local|real]
"""
from __future__ import annotations
import logging, sys
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.mechanic_inference import infer_mechanics

logging.basicConfig(level=logging.ERROR)
LOCAL = {"maze", "push", "navg", "btnc", "clickbig", "navgc"}


def capture(prefix, budget, mode):
    if mode == "local":
        client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="src/arcagi3/games")
        gid = prefix
        env = client.make(game_id=gid, scorecard_id=f"sc-{gid}")
    else:
        client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("m"))
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
        env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["mechanic"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); traj = []; prev_levels = 0
    bg = P.detect_background(P.to_grid(obs.frame))
    for _ in range(budget):
        g = P.to_grid(obs.frame); st = obs.state
        tok = pol.decide(g, gstate_terminal=(st == GameState.GAME_OVER),
                         gstate_notplayed=(st == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = env.reset(); continue
        elif tok[0] == "S":
            nobs = env.step(GameAction.from_id(tok[1]))
        else:
            nobs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        ng = P.to_grid(nobs.frame); lv = int(nobs.levels_completed or 0)
        traj.append((g, tok, ng, float(lv - prev_levels)))
        prev_levels = lv; obs = nobs
    return traj, bg


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    mode = sys.argv[3] if len(sys.argv) > 3 else ("local" if prefix in LOCAL else "real")
    traj, bg = capture(prefix, budget, mode)
    rep = infer_mechanics(traj, bg)
    print(f"[{prefix}] bg={bg} steps={len(traj)}")
    print(f"  {rep.summary()}")


if __name__ == "__main__":
    main()
