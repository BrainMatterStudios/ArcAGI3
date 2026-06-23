# scripts/hybrid_bakeoff.py
"""Phase-4 three-arm lp85 bake-off: salience vs signature-transfer vs +model guidance.

Measures actions-to-clear per level for each arm — does discovery-model guidance beat the shallow
signature heuristic? Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/hybrid_bakeoff.py [budget]
"""
from __future__ import annotations
import logging, sys, time
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer

logging.basicConfig(level=logging.ERROR)


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def make_lp85():
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("hb"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("lp85"))
    return client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["hybrid"]))


def run_arm(name, engine, budget):
    env = make_lp85()
    obs = _retry(env.reset)
    if hasattr(engine, "reset_all"):
        engine.reset_all()
    per_level = []          # (level, actions spent to clear it)
    cur_level = int(obs.levels_completed or 0)
    actions_this = 0
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = engine.decide(grid=grid, gstate_terminal=(obs.state == GameState.GAME_OVER),
                            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                            levels=cur_level, available=list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            aid = tok[1]
            obs = _retry(lambda: env.step(GameAction.from_id(aid))); actions_this += 1; n += 1
        else:
            x, y = tok[1], tok[2]
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": x, "y": y})); actions_this += 1; n += 1
        nl = int(obs.levels_completed or 0)
        if nl > cur_level:
            per_level.append((cur_level, actions_this)); cur_level = nl; actions_this = 0
    return name, cur_level, per_level, getattr(engine, "_guide_fires", 0)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    arms = [
        ("salience", SalienceExplorer(seed=0)),
        ("transfer(signature)", TransferExplorer(seed=0)),
        ("hybrid(+model)", HybridTransferExplorer(seed=0)),
    ]
    print(f"{'arm':<24}{'levels':<8}{'per-level (lvl,actions)':<34}{'total':<8}{'guide'}", flush=True)
    for name, eng in arms:
        print(f"running {name}...", flush=True)
        nm, levels, per, fires = run_arm(name, eng, budget)
        total = sum(a for _, a in per)
        print(f"{nm:<24}{levels:<8}{str(per):<34}{total:<8}{'fires=' + str(fires)}", flush=True)


if __name__ == "__main__":
    main()
