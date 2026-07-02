"""CoroutineStrategy + general_agent_gen: a game-agnostic search-replay solver written as a generator, driven
over the reactive decide() interface. Validates it solves dc22 (interleaved click+move) reactively, zero code."""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.coroutine_strategy import CoroutineStrategy, general_agent_gen, oc_search_gen, _object_clicks


def test_coroutine_general_agent_solves_dc22():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("dc22"))
    env = c.make(game_id=gid, scorecard_id="co-dc22-test"); obs = env.reset()
    pol = CoroutineStrategy(general_agent_gen); last = np.zeros((64, 64), int); maxlvl = 0
    for _ in range(40000):
        g = P.to_grid(obs.frame) if (obs.frame is not None and len(obs.frame)) else last
        last = g; maxlvl = max(maxlvl, int(obs.levels_completed or 0))
        tok = pol.decide(g, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        try:
            if tok[0] == "reset": obs = env.reset()
            elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
            else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        except Exception: obs = env.reset()
        if pol._done and maxlvl >= 1:
            return
    assert maxlvl >= 1, "coroutine general agent should solve dc22 reactively"


def _reactive_maxlevel(game, budget=40000):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    env = c.make(game_id=gid, scorecard_id=f"co-{game}-t"); obs = env.reset()
    pol = CoroutineStrategy(general_agent_gen); last = np.zeros((64, 64), int); mx = 0
    for _ in range(budget):
        g = P.to_grid(obs.frame) if (obs.frame is not None and len(obs.frame)) else last
        last = g; mx = max(mx, int(obs.levels_completed or 0))
        if mx >= 1:
            return mx
        tok = pol.decide(g, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        try:
            if tok[0] == "reset": obs = env.reset()
            elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
            else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        except Exception: obs = env.reset()
    return mx


def test_peg_solitaire_class_solves_lf52():
    assert _reactive_maxlevel("lf52") >= 1, "folded peg-solitaire class should solve lf52 reactively"


def test_folded_classes_solve_reactively():
    # each folded goal-class solves its dev game through the reactive general agent, zero per-game wiring
    assert _reactive_maxlevel("sb26") >= 1, "template-match class should solve sb26"
    assert _reactive_maxlevel("r11l") >= 1, "centroid-drag class should solve r11l"
    assert _reactive_maxlevel("su15") >= 1, "pull-drag class should solve su15"


def test_generic_search_unaffected_by_class_folds():
    # non-matching games still fall through to generic search (no false-fires)
    assert _reactive_maxlevel("vc33") >= 1


def _reactive_maxlevel_oc(game, budget=40000):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    env = c.make(game_id=gid, scorecard_id=f"oc-{game}-t"); obs = env.reset()
    pol = CoroutineStrategy(oc_search_gen); last = np.zeros((64, 64), int); mx = 0
    for _ in range(budget):
        g = P.to_grid(obs.frame) if (obs.frame is not None and len(obs.frame)) else last
        last = g; mx = max(mx, int(obs.levels_completed or 0))
        if mx >= 1:
            return mx
        tok = pol.decide(g, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        try:
            if tok[0] == "reset": obs = env.reset()
            elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
            else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        except Exception: obs = env.reset()
    return mx


def test_object_clicks_returns_object_centres():
    # object-centric candidates: non-empty, in-bounds, deduped
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("ka59"))
    env = c.make(game_id=gid, scorecard_id="oc-ka59-clicks"); obs = env.reset()
    pts = _object_clicks(P.to_grid(obs.frame))
    assert pts and len(pts) == len(set(pts))
    assert all(0 <= x < 64 and 0 <= y < 64 for (x, y) in pts)


def test_oc_search_solves_ls20_reactively():
    # object-centric search cracks a NO_L0 game the pixel-target/exact-grid probe missed (floor-safe additive)
    assert _reactive_maxlevel_oc("ls20") >= 1, "object-centric search should solve ls20 reactively"


def test_oc_search_in_portfolio():
    # object-centric play present; EFFICIENCY-FIRST ordering: class-solvers first, wandering anchor LAST
    from arcagi3.portfolio_policy import _default_strategies
    names = [n for (n, _) in _default_strategies()]
    assert "oc_search" in names
    assert names[-1] == "transfer_s0", f"coverage anchor must be LAST (efficiency-first); got {names[-1]}"
    assert names[0] == "goal_classes_early", "specialized class-solvers must run first"
    # efficiency plays must precede the wandering anchor so the WINning run is efficient (squared eval)
    assert names.index("geodesic_replay") < names.index("transfer_s0")
