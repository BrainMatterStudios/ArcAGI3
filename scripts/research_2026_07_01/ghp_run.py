"""Run GHP (source-free goal-hypothesis planner) across games and report levels solved. This is the
generalization test: does learn-model + visible-goal-hypothesis + exact-plan crack games WITHOUT source?

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 python scripts/research_2026_07_01/ghp_run.py [game,game,...] [budget]
"""
import logging, sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from ghp import GHP
logging.basicConfig(level=logging.ERROR)

ZERO = ["re86", "wa30", "sb26", "ka59", "bp35", "g50t", "dc22", "sc25"]


def act(env, a, tgt=None):
    if a == 6 and tgt is not None:
        return env.step(GameAction.ACTION6, data={"x": int(round(tgt[1])), "y": int(round(tgt[0]))})
    if a == 5:
        return env.step(GameAction.ACTION5)
    return env.step(GameAction.from_id(a))


def solve(game, budget=3000, verbose=False):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"ghp-{game}")
    obs = env.reset()
    ghp = GHP()
    obs = ghp.learn(env, obs)
    lv = int(obs.levels_completed or 0)
    n = 0
    if verbose:
        print(f"  {game}: avatar_color={ghp.avatar_color} deltas={ghp.deltas} cell={ghp.cell} "
              f"can5={ghp.can5} can6={ghp.can6}")
    tried = 0
    while n < budget:
        grid = P.to_grid(obs.frame)
        if ghp.avatar_pos(grid) is None:
            obs = ghp.learn(env, obs); grid = P.to_grid(obs.frame)
            if ghp.avatar_pos(grid) is None:
                break
        hyps = ghp.goal_hypotheses(grid)
        made_progress = False
        for (cen, col, sz) in hyps:
            plan = ghp.plan_to(grid, cen)
            if plan is None:
                continue
            for a in plan:
                obs = act(env, a); n += 1
                if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
                    made_progress = True; break
                if obs.state == GameState.GAME_OVER:
                    break
            # at/near the goal, try interactions to trigger the reward
            if not made_progress and obs.state not in (GameState.WIN, GameState.GAME_OVER):
                for inter in ([5] if ghp.can5 else []) + ([6] if ghp.can6 else []):
                    obs = act(env, inter, cen); n += 1
                    if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
                        made_progress = True; break
            if int(obs.levels_completed or 0) > lv:
                made_progress = True
            if obs.state == GameState.WIN:
                break
            if obs.state == GameState.GAME_OVER:
                obs = env.reset(); obs = ghp.learn(env, obs); lv = int(obs.levels_completed or 0); break
            if made_progress:
                break
            if n >= budget:
                break
        newlv = int(obs.levels_completed or 0)
        if newlv > lv:
            lv = newlv
        elif not made_progress:
            tried += 1
            if tried > 3:  # exhausted hypotheses without progress
                break
        if obs.state == GameState.WIN:
            break
    return lv, n


def main():
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else ZERO
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    print(f"GHP source-free solver (budget {budget}). Coverage explorer scores 0 on all ZERO games.\n")
    total = 0
    for gm in games:
        try:
            lv, n = solve(gm, budget, verbose=True)
        except Exception as e:
            print(f"  {gm}: ERROR {type(e).__name__}: {e}"); continue
        total += lv
        flag = " <-- CRACKED (was 0)" if (gm in ZERO and lv > 0) else ""
        print(f"{gm}: {lv} levels in {n} actions{flag}\n")
    print(f"TOTAL levels: {total}")


if __name__ == "__main__":
    main()
