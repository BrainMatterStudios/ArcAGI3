"""Re-traversal overhead diagnostic — does salience waste actions re-navigating to frontiers?

The efficiency lever (the 0.33->0.66 gap) is reaching the same levels in fewer actions. Salience
navigates to a frontier by replaying a stored path (self.plan); those replay steps are NOT new
exploration — they're re-traversal overhead. If that fraction is large, re-traversal reduction
(cache shortcuts / gateway waypoints, coverage-preserving) is a direct efficiency win worth building.
If small, salience is already near-optimal in navigation and efficiency must come from elsewhere.

Measures, per game over a run: actions that were plan-replay (navigation) vs fresh exploration
(a new untried action chosen) vs reset.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/retraversal_diagnostic.py [budget]
"""
from __future__ import annotations
import logging, sys, time
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("rt"))
GAMES = ["sk48", "ls20", "m0r0", "vc33", "tu93", "lf52"]


class MeasuredSalience(SalienceExplorer):
    """Tag each chosen action as navigation (plan replay) vs fresh exploration."""
    def reset_all(self):
        super().reset_all()
        self.nav_actions = 0
        self.explore_actions = 0

    def _choose(self, cur):
        had_plan = bool(self.plan)
        tok = super()._choose(cur)
        # if we returned a step from an existing/!new plan, it's navigation re-traversal
        if had_plan and tok and tok[0] != "reset":
            self.nav_actions += 1
        elif tok and tok[0] != "reset":
            self.explore_actions += 1
        return tok


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def run_game(prefix, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["rt"]); env = client.make(game_id=gid, scorecard_id=card)
    pol = MeasuredSalience(seed=0, trust_threshold=3, border_mask=2,
                           coarse_grid_step=4, max_click_targets=256)
    obs = _retry(env.reset); n = 0; best = 0; resets = 0
    while n < budget:
        st = obs.state
        tok = pol.decide(P.to_grid(obs.frame), st == GameState.GAME_OVER,
                         st == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])])
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = _retry(env.reset); resets += 1
        elif tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))
        n += 1
        best = max(best, int(obs.levels_completed or 0))
    nav, expl = pol.nav_actions, pol.explore_actions
    frac = nav / max(nav + expl, 1)
    return best, nav, expl, resets, frac


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    t0 = time.time()
    print(f"re-traversal diagnostic @ budget={budget}\n", flush=True)
    for g in GAMES:
        try:
            best, nav, expl, resets, frac = run_game(g, budget)
        except Exception as e:  # noqa: BLE001
            print(f"  {g}: ERROR {type(e).__name__}: {e}", flush=True); continue
        print(f"  {g}: L{best}  nav={nav} explore={expl} resets={resets}  "
              f"re-traversal_frac={frac:.2f}", flush=True)
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
