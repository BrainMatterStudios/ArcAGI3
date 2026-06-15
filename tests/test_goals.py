"""C5 — goal inference: event-derivation units, no-regression proofs, and game canaries.

Offline, pure-numpy tests (no env for the units): they pin _is_board_swap, each derived
event kind on hand-built 8x8 arrays, the click convention, the Laplace confidence gate and
fair-contradiction refinement, and goal_target_cells instantiation. Plus the core
no-regression contract — the reactive policy emits byte-identical action tokens whether
infer_goals is False or True (M1 is observe-only, non-load-bearing) — a perf micro-bench,
and end-to-end canaries asserting a confident, correctly-typed current_goal() is inferred
on the local dev games. Existing tests/test_agent.py is left untouched.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np

from arcagi3 import goals as G
from arcagi3 import movement as MV

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")
BG = 0
AV = 14   # avatar color
ITEM = 6
GOAL = 4
SWITCH = 8


def _g():
    return np.zeros((8, 8), dtype=np.int8)


def _avatar_mm(color=AV):
    return MV.MotionModel(
        avatar_color=color,
        deltas={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        avatar_colors=frozenset({color}),
    )


# --- _is_board_swap -------------------------------------------------------------------

def test_swap_false_for_within_level_move():
    prev = _g()
    prev[2, 2] = AV
    prev[5, 5] = GOAL
    cur = _g()
    cur[2, 3] = AV  # avatar moved one cell
    cur[5, 5] = GOAL
    assert G._is_board_swap(prev, cur, BG) is False


def test_swap_true_on_shape_change():
    prev = np.zeros((8, 8), dtype=np.int8)
    cur = np.zeros((9, 9), dtype=np.int8)
    assert G._is_board_swap(prev, cur, BG) is True


def test_swap_true_on_whole_board_repaint():
    prev = _g()
    prev[1, 1] = AV
    cur = _g() + 7  # almost every cell changed
    assert G._is_board_swap(prev, cur, BG) is True


def test_swap_true_on_background_color_change():
    prev = _g()
    prev[1, 1] = AV
    cur = np.full((8, 8), 3, dtype=np.int8)  # different dominant color
    cur[1, 1] = AV
    assert G._is_board_swap(prev, cur, BG) is True


# --- _derive_events -------------------------------------------------------------------

def _gi():
    return G.GoalInference()


def _derive(prev, cur, avatar=None, distractors=None):
    return _gi()._derive_events(prev, cur, BG, distractors or set(), avatar)


def test_event_vanish():
    prev = _g(); prev[5, 5] = ITEM
    cur = _g()
    evs = _derive(prev, cur)
    assert any(e.kind == G.EV_VANISH and e.color == ITEM for e in evs)


def test_event_appear():
    prev = _g()
    cur = _g(); cur[5, 5] = ITEM
    evs = _derive(prev, cur)
    assert any(e.kind == G.EV_APPEAR and e.color == ITEM for e in evs)


def test_event_move_obj_is_translation():
    prev = _g(); prev[3, 3] = ITEM
    cur = _g(); cur[3, 4] = ITEM  # rigid translation
    evs = _derive(prev, cur)
    assert any(e.kind == G.EV_MOVE_OBJ and e.color == ITEM for e in evs)


def test_event_count_dn_and_up():
    prev = _g(); prev[1, 1] = ITEM; prev[1, 2] = ITEM
    cur = _g(); cur[1, 1] = ITEM     # lost one cell (no clean translation)
    assert any(e.kind == G.EV_COUNT_DN and e.color == ITEM for e in _derive(prev, cur))
    assert any(e.kind == G.EV_COUNT_UP and e.color == ITEM for e in _derive(cur, prev))


def test_event_recolor_same_count_moved():
    # same count, cells differ, not a clean rigid translation -> RECOLOR (toggle/door)
    prev = _g(); prev[1, 1] = SWITCH; prev[6, 6] = SWITCH
    cur = _g(); cur[1, 1] = SWITCH; cur[6, 1] = SWITCH
    evs = _derive(prev, cur)
    assert any(e.kind == G.EV_RECOLOR and e.color == SWITCH for e in evs)


def test_event_avatar_on_contacted_color():
    mm = _avatar_mm()
    prev = _g(); prev[2, 2] = AV; prev[2, 3] = GOAL  # goal just to the right
    cur = _g(); cur[2, 3] = AV                        # avatar walked ONTO the goal cell
    evs = _derive(prev, cur, avatar=mm)
    assert any(e.kind == G.EV_AVATAR_ON and e.color == GOAL for e in evs)
    # the occlusion drop of the goal color is masked, not reported as a vanish
    assert not any(e.kind in (G.EV_VANISH, G.EV_COUNT_DN) and e.color == GOAL for e in evs)


def test_distractor_color_ignored():
    prev = _g(); prev[0, 0] = 9
    cur = _g(); cur[0, 1] = 9  # a counter animation
    evs = _derive(prev, cur, distractors={9})
    assert all(e.color != 9 for e in evs)


# --- credit / hypothesis formation ----------------------------------------------------

def test_collect_forms_vanish_all():
    gi = _gi()
    grid = _g(); grid[5, 5] = ITEM; grid[6, 6] = ITEM
    gi.on_level_start(grid, BG, 0)
    prev = grid.copy()
    cur = _g(); cur[6, 6] = ITEM  # one item vanished, no avatar contact
    gi.observe_step(prev_grid=prev, cur_grid=cur, prev_action=("S", 1), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    g = gi.current_goal(min_conf=0.0)
    assert g is not None and g.kind == "VANISH_ALL" and g.color == ITEM


def test_reach_forms_for_avatar_contact():
    gi = _gi()
    mm = _avatar_mm()
    grid = _g(); grid[2, 2] = AV; grid[2, 3] = GOAL
    gi.on_level_start(grid, BG, 0)
    cur = _g(); cur[2, 3] = AV
    gi.observe_step(prev_grid=grid, cur_grid=cur, prev_action=("S", 4), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=mm)
    g = gi.current_goal(min_conf=0.0)
    assert g is not None and g.kind == "REACH" and g.color == GOAL


def test_push_to_forms_for_translation():
    gi = _gi()
    grid = _g(); grid[3, 3] = ITEM
    gi.on_level_start(grid, BG, 0)
    cur = _g(); cur[3, 4] = ITEM
    gi.observe_step(prev_grid=grid, cur_grid=cur, prev_action=("S", 4), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    keys = {k[0] for k in gi.model.hyps}
    assert "PUSH_TO" in keys


def test_click_target_convention():
    # token is ("C", x=col, y=row); the clicked color must be read at grid[y, x].
    gi = _gi()
    grid = _g(); grid[6, 2] = 3   # row 6, col 2
    gi.on_level_start(grid, BG, 0)
    cur = _g() + 7                 # swap (level rebuilt) -> no derived events
    gi.observe_step(prev_grid=grid, cur_grid=cur, prev_action=("C", 2, 6), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    g = gi.current_goal(min_conf=0.0)
    assert g is not None and g.kind == "CLICK_TARGET" and g.color == 3


def test_single_step_collect_recovers_target_across_swap():
    # collect-the-last-item: the rewarding transition IS the level swap, so events are empty;
    # the target is recovered from the avatar's intended destination on the OLD board.
    gi = _gi()
    mm = _avatar_mm()
    grid = _g(); grid[2, 2] = AV; grid[2, 3] = ITEM
    gi.on_level_start(grid, BG, 0)
    cur = _g() + 7  # whole-board swap
    gi.observe_step(prev_grid=grid, cur_grid=cur, prev_action=("S", 4), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=mm)
    g = gi.current_goal(min_conf=0.0)
    assert g is not None and g.color == ITEM


# --- confidence gate / refinement -----------------------------------------------------

def test_confidence_laplace_gate():
    h = G.GoalHypothesis(kind="REACH", color=GOAL, support=1, contra=0)
    assert abs(h.confidence - (2.0 / 3.0)) < 1e-9
    h.support = 2
    assert abs(h.confidence - 0.75) < 1e-9
    model = G.GoalModel(hyps={("REACH", GOAL, None): h})
    assert model.best(min_conf=0.6) is h
    assert model.best(min_conf=0.8) is None  # below gate


def test_fair_contradiction_only_when_applicable():
    gi = _gi()
    # level 0: collect ITEM -> VANISH_ALL(ITEM)
    g0 = _g(); g0[5, 5] = ITEM
    gi.on_level_start(g0, BG, 0)
    c0 = _g()
    gi.observe_step(prev_grid=g0, cur_grid=c0, prev_action=("S", 1), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    hyp = gi.model.hyps[("VANISH_ALL", ITEM, None)]
    assert hyp.support == 1 and hyp.contra == 0
    # level 1: ITEM is NOT present at level start -> precondition not applicable -> NO contra
    g1 = _g(); g1[1, 1] = SWITCH
    gi.on_level_start(g1, BG, 1)
    c1 = _g()
    gi.observe_step(prev_grid=g1, cur_grid=c1, prev_action=("S", 1), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    assert hyp.contra == 0, "penalised a hypothesis whose precondition was not applicable"


def test_goal_target_cells_instantiation():
    gi = _gi()
    grid = _g(); grid[5, 5] = ITEM; grid[6, 6] = ITEM
    gi.on_level_start(grid, BG, 0)
    cur = _g(); cur[6, 6] = ITEM
    gi.observe_step(prev_grid=grid, cur_grid=cur, prev_action=("S", 1), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    board = _g(); board[1, 1] = ITEM; board[7, 7] = ITEM
    cells = gi.goal_target_cells(board, bg=BG, min_conf=0.0)
    assert (1, 1) in cells and (7, 7) in cells


def test_reset_all_clears_model():
    gi = _gi()
    grid = _g(); grid[5, 5] = ITEM
    gi.on_level_start(grid, BG, 0)
    gi.observe_step(prev_grid=grid, cur_grid=_g(), prev_action=("S", 1), reward=1.0,
                    bg=BG, distractor_colors=set(), avatar=None)
    assert gi.model.hyps
    gi.reset_all()
    assert not gi.model.hyps and gi.model.levelups_seen == 0


# --- no-regression: reactive token sequences identical for flag off/on ----------------

def _capture_tokens(gid, budget=4000, **kw):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.policy import HybridPolicy

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                    logger=logging.getLogger("test_goals"))
    env = client.make(game_id=gid, scorecard_id=f"sc-{gid}")
    pol = HybridPolicy(seed=0, **kw)
    obs = env.reset()
    toks = []
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        t = pol.decide(grid, obs.state == GameState.GAME_OVER,
                       obs.state == GameState.NOT_PLAYED,
                       int(obs.levels_completed or 0), list(obs.available_actions or []))
        toks.append(t)
        if t[0] == "reset":
            obs = env.reset()
        elif t[0] == "S":
            obs = env.step(GameAction.from_id(t[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": t[1], "y": t[2]})
        n += 1
    return toks, pol


def test_policy_action_trace_unchanged():
    # the core no-regression gate: infer_goals must not change any action token, on the
    # fragile push case AND maze (and a collect game for good measure).
    for gid in ("push", "maze", "collect"):
        base, _ = _capture_tokens(gid, infer_goals=False)
        on, _ = _capture_tokens(gid, infer_goals=True)
        assert base == on, f"infer_goals changed action tokens for {gid}"


# --- integration canaries: flag-on infers a confident, correctly-typed goal -----------

EXPECTED = {
    "btnc": "CLICK_TARGET",     # click the lit button -> CLICK_TARGET
    "clickbig": "PUSH_TO",      # a block visibly translates before reward -> PUSH_TO
    "navg": "REACH",            # navigate avatar onto the goal -> REACH
    "navgc": "REACH",
    "maze": "VANISH_ALL",       # the maze exit is consumed/repainted on reach
}


def test_dev_games_infer_expected_goal_kind():
    # games whose mechanic is cleanly identifiable from observation: the best hypothesis
    # (above the 0.6 confidence gate) must match the ground-truth mechanic.
    for gid, kind in EXPECTED.items():
        _, pol = _capture_tokens(gid, budget=2000, infer_goals=True)
        g = pol.gi.current_goal()
        assert g is not None, f"{gid}: no confident goal inferred"
        assert g.kind == kind, f"{gid}: inferred {g.kind}, expected {kind}"


def test_push_infers_goal_on_block_color():
    # On `push` the block-push that completes the level coincides with the level swap, so the
    # observable signal is the avatar contacting the BLOCK color (color 6). Assert a confident
    # goal keyed on that color is formed (a usable navigate-to-block target for C6); the
    # explicit PUSH_TO label is exercised by clickbig where a block visibly translates.
    _, pol = _capture_tokens("push", budget=3500, infer_goals=True)
    g = pol.gi.current_goal()
    assert g is not None and g.color == 6, f"push inferred {g}"


def test_clickbig_forms_push_to():
    _, pol = _capture_tokens("clickbig", budget=2000, infer_goals=True)
    kinds = {k[0] for k in pol.gi.model.hyps}
    assert "PUSH_TO" in kinds, "clickbig did not form a PUSH_TO hypothesis"


def test_all_dev_games_form_a_confident_goal():
    # the durable M1 deliverable: every dev game yields SOME confident typed hypothesis for
    # C6, drawn from the goal taxonomy.
    for gid in ("btnc", "clickbig", "collect", "maze", "navg", "navgc", "switchdoor"):
        _, pol = _capture_tokens(gid, budget=2000, infer_goals=True)
        g = pol.gi.current_goal()
        assert g is not None, f"{gid}: no confident goal inferred"
        assert g.kind in G.GOAL_KINDS


# --- perf micro-bench: observe_step is cheap on a dense 64x64 frame -------------------

def test_observe_step_perf_under_budget():
    mm = MV.MotionModel(avatar_color=AV, deltas={1: (-1, 0), 2: (1, 0), 3: (0, -1),
                        4: (0, 1)}, avatar_colors=frozenset({AV}))
    b = np.zeros((64, 64), dtype=np.int8)
    rng = np.random.default_rng(0)
    for _ in range(80):
        b[int(rng.integers(0, 64)), int(rng.integers(0, 64))] = int(rng.integers(1, 13))
    b[32, 32] = AV
    a = b.copy()
    a[32, 32] = 0
    a[32, 33] = AV
    gi = _gi()
    gi.on_level_start(b, BG, 0)
    # warm the lru_cache the policy shares
    gi.observe_step(prev_grid=b, cur_grid=a, prev_action=("S", 4), reward=0.0,
                    bg=BG, distractor_colors=set(), avatar=mm)
    t0 = time.perf_counter()
    for _ in range(50):
        gi.observe_step(prev_grid=b, cur_grid=a, prev_action=("S", 4), reward=0.0,
                        bg=BG, distractor_colors=set(), avatar=mm)
    dt = (time.perf_counter() - t0) / 50
    assert dt < 0.003, f"observe_step too slow: {dt*1000:.3f}ms/call"
