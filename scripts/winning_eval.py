"""Phase 7 — evaluation for the winning branch (mechanic model-search + reward-goal-learning).

Self-contained level-counting loop (no A_h grader dependency, so m0r0/su15/etc don't crash). Runs
TransferExplorer vs WinningExplorer on the movement-capable HOLDOUT games + the collect control.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/winning_eval.py [budget] [games...]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv; load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.discovery_bakeoff import make_game
from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.winning_explorer import WinningExplorer
from arcengine import GameAction, GameState

logging.basicConfig(level=logging.ERROR)
MOVEMENT_HOLDOUT = ["ls20", "su15", "m0r0", "wa30", "re86"]
DEFAULT = ["collect"] + MOVEMENT_HOLDOUT


def run(eng, game, budget):
    eng.reset_all()
    env = make_game(game)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    per = {}
    n = last = 0
    while n < budget:
        if obs is None or obs.state == GameState.WIN or np.asarray(obs.frame).size == 0:
            break
        grid = P.to_grid(obs.frame)
        tok = eng.decide(grid=grid, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=level, available=list(obs.available_actions or []))
        try:
            if tok[0] == "reset":
                obs = env.reset()
            elif tok[0] == "S":
                obs = env.step(GameAction.from_id(tok[1])); n += 1
            else:
                obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); n += 1
        except Exception:
            break
        if obs is None:
            break
        nl = int(obs.levels_completed or 0)
        if nl > level:
            per[level] = n - last; last = n; level = nl
    return level, per


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT
    print(f"=== Winning eval (full engine) — budget={budget}/game ===\n", flush=True)
    rows = []
    for game in games:
        t_lv, _ = run(TransferExplorer(seed=0), game, budget)
        w = WinningExplorer(seed=0, enable_model_search=True)
        w_lv, w_per = run(w, game, budget)
        lg = getattr(w._ms, "learned_goal", None)
        lgk = (lg.kind, getattr(lg, "color", getattr(lg, "axis", None))) if lg else None
        rows.append((game, t_lv, w_lv, w_per, w._model_plan_starts, w._gave_up, lgk))
        print(f"[{game:8}] transfer={t_lv} winning={w_lv} per={w_per} plans={w._model_plan_starts} "
              f"gaveup={w._gave_up} learned={lgk}", flush=True)

    print("\n" + "-" * 80)
    improved = sum(1 for _g, t, w, *_ in rows if isinstance(t, int) and w > t and _g != "collect")
    regressed = sum(1 for _g, t, w, *_ in rows if isinstance(t, int) and w < t)
    deep = sum(1 for _g, t, w, *_ in rows if isinstance(w, int) and w >= 2 and _g != "collect")
    print(f"HOLDOUT improved={improved} regressed={regressed} | holdout games reaching L2+: {deep}")
    go = improved >= 1 and regressed == 0
    print(f"GO-RULE (holdout strictly improves, no regressions): "
          f"{'PASS — Kaggle candidate' if go else 'FAIL — ship transfer-dense'}")


if __name__ == "__main__":
    main()
