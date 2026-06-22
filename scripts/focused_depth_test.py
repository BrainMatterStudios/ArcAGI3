"""Focused 30k A/B: does DepthStallExplorer reach navigation-wall L1 faster than salience?

Decisive games: sk48/ls20/m0r0 (walls, L1 reachable ~25k under salience) + tu93/lf52 (working/slow,
the stall-trigger risk) + vc33 (working, harm check). Runs depth-dense vs salience-dense at a high
budget and reports actions-to-each-level. WIN = a wall game reaches L1 in fewer actions without
tu93/lf52 regressing.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/focused_depth_test.py [budget] [policy]
"""
from __future__ import annotations
import logging, sys, time
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("depth"))
GAMES = ["sk48", "ls20", "m0r0", "tu93", "lf52", "vc33"]


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def make(policy):
    if policy == "depth-dense":
        from arcagi3.depth_stall_explorer import DepthStallExplorer
        return DepthStallExplorer(seed=0, trust_threshold=3, border_mask=2,
                                  coarse_grid_step=4, max_click_targets=256)
    from arcagi3.salience_explorer import SalienceExplorer
    return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2,
                            coarse_grid_step=4, max_click_targets=256)


def run_game(prefix, policy, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["depth"]); env = client.make(game_id=gid, scorecard_id=card)
    pol = make(policy)
    obs = _retry(env.reset); n = 0; best = 0; marks = []; last = 0
    while n < budget:
        st = obs.state
        tok = pol.decide(P.to_grid(obs.frame), st == GameState.GAME_OVER,
                         st == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])])
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > best:
            marks.append(n - last); last = n; best = lv
    return best, marks, n


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    policy = sys.argv[2] if len(sys.argv) > 2 else "depth-dense"
    t0 = time.time()
    print(f"focused depth test: policy={policy} budget={budget}\n", flush=True)
    for g in GAMES:
        try:
            best, marks, n = run_game(g, policy, budget)
        except Exception as e:  # noqa: BLE001
            print(f"  {g}: ERROR {type(e).__name__}: {e}", flush=True); continue
        print(f"  {g}: L{best} acts/lvl={marks} (used {n})", flush=True)
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
