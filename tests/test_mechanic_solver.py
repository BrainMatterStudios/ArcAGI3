"""The additive mechanic-solver scaffold must (1) solve a game it recognizes (wa30 -> a zero-game), and
(2) ABSTAIN on games it does not recognize (so the portfolio keeps the banked coverage play, 0.33 floor).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
from arc_agi import Arcade, OperationMode
from arcagi3 import perception as P
import mechanic_solver as M


def _env(prefix):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=f"test-ms-{prefix}")
    return env, env.reset()


def test_solver_solves_recognized_wa30():
    env, obs = _env("wa30")
    result = M.try_solve(env, obs, budget=2000)
    assert result is not None, "solver should recognize wa30"
    levels, used = result
    assert levels >= 1, f"solver should solve >=1 wa30 level, got {levels} in {used} actions"


def test_solver_abstains_on_non_matching_games():
    # avatar-move (tu93) and click-only (vc33) games are NOT the grab-drag archetype -> must abstain
    for prefix in ("tu93", "vc33"):
        env, obs = _env(prefix)
        grid = P.to_grid(obs.frame)
        assert not M.recognize_grabdrag(grid), f"{prefix} must not be recognized as grab-drag"
