"""C3 — affordance model: golden classifier verdicts + no-regression proofs.

Offline, pure-numpy unit tests (no env): they pin the per-color affordance classification
for the 8-label ontology on synthetic grids (BLOCK/PASS/PUSH/COLLECT/TOGGLE/GOAL/HARM),
the Dirichlet confidence / reliability gate, the no-op fast paths, and the click path; plus
the core no-regression contract — the reactive policy emits byte-identical action tokens
whether enable_affordance is False or True (observe-only, non-load-bearing). A flag-on
integration smoke test confirms the model actually learns real verdicts on local games.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np

from arcagi3 import movement as MV
from arcagi3.affordance import AffordanceModel, ColorStat, Effect, Verdict

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")
BG = 0
AV = 14   # avatar color
WALL = 6
ITEM = 3
SWITCH = 8
DOOR = 9


def _g():
    return np.zeros((16, 16), dtype=np.int8)


def _avatar_mm():
    # avatar with >=2 distinct deltas (action-correlated) -> mm.ok and multi-delta
    return MV.MotionModel(
        avatar_color=AV,
        deltas={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        avatar_colors=frozenset({AV}),
    )


# --- classifier: one per ontology label ----------------------------------------------

def test_block_when_avatar_does_not_move():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = WALL  # wall to the right
    a = b.copy()    # action 4 (right) but avatar did not move -> BLOCK
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=False)
    assert out == {WALL: Effect.BLOCK}


def test_pass_when_avatar_walks_onto_unchanged_object():
    # avatar enters a cell holding ITEM, advances, and an equal ITEM cell remains: the color
    # count is unchanged (not collect), it didn't translate (not push), nothing changed
    # remotely (not toggle) -> PASS (passable floor/decoration).
    mm = _avatar_mm()
    b = _g(); b[5, 5] = AV; b[5, 6] = ITEM; b[1, 1] = ITEM
    a = _g(); a[5, 6] = AV; a[5, 5] = ITEM; a[1, 1] = ITEM
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=False)
    assert out == {ITEM: Effect.PASS}


def test_push_when_object_translates_with_avatar():
    mm = _avatar_mm()
    b = _g()
    b[5, 4] = AV
    b[5, 5] = WALL   # block ahead
    a = _g()
    a[5, 5] = AV     # avatar advanced by (0,+1)
    a[5, 6] = WALL   # block pushed by (0,+1)
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=False)
    assert out == {WALL: Effect.PUSH}


def test_collect_when_object_vanishes_under_avatar():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = ITEM
    a = _g()
    a[5, 6] = AV     # avatar arrived; item gone
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=False)
    assert out == {ITEM: Effect.COLLECT}


def test_toggle_when_contact_causes_remote_change():
    # avatar bumps a switch that does NOT move/vanish (it stays a solid lever) and a remote
    # door changes on the same step -> TOGGLE. The switch persisting (avatar didn't enter it)
    # means BLOCK would otherwise apply, but the remote change reclassifies it as TOGGLE.
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = SWITCH   # switch to the right (avatar will not pass through it)
    b[1, 12] = DOOR    # remote door
    a = _g()
    a[5, 6] = AV       # avatar moved onto the switch cell (contact)
    a[5, 5] = SWITCH   # ... but the switch persists (relocated/redrawn, count unchanged)
    a[1, 12] = DOOR
    a[2, 12] = DOOR    # door extended remotely (a real, non-incidental change elsewhere)
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=False)
    assert out == {SWITCH: Effect.TOGGLE}


def test_goal_when_reward_on_contact():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = ITEM
    a = _g()
    a[5, 6] = AV
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=1.0, terminal=False)
    assert out == {ITEM: Effect.GOAL}
    assert m.is_harm(ITEM) is False
    assert ITEM in m.goal_colors()


def test_harm_when_terminal_on_contact():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = ITEM
    a = _g()
    a[5, 6] = AV
    m = AffordanceModel()
    out = m.observe_step(b, a, ("S", 4), mm, BG, reward=0.0, terminal=True)
    assert out == {ITEM: Effect.HARM}
    assert m.is_harm(ITEM) is True
    assert ITEM in m.harm_colors()


# --- no-op fast paths -----------------------------------------------------------------

def test_noop_when_no_motion_model():
    b = _g(); b[5, 5] = AV; b[5, 6] = WALL
    a = b.copy()
    assert AffordanceModel().observe_step(b, a, ("S", 4), None, BG, 0.0, False) == {}


def test_noop_on_zero_delta_action():
    mm = MV.MotionModel(avatar_color=AV, deltas={5: (0, 0), 1: (-1, 0)},
                        avatar_colors=frozenset({AV}))
    b = _g(); b[5, 5] = AV; b[5, 6] = WALL
    a = b.copy()
    assert mm.ok
    assert AffordanceModel().observe_step(b, a, ("S", 5), mm, BG, 0.0, False) == {}


def test_noop_when_walks_onto_open_floor():
    mm = _avatar_mm()
    b = _g(); b[5, 5] = AV
    a = _g(); a[5, 6] = AV  # plain move over empty floor -> no contact, no signal
    assert AffordanceModel().observe_step(b, a, ("S", 4), mm, BG, 0.0, False) == {}


def test_click_collect_path():
    b = _g(); b[3, 3] = ITEM
    a = _g()  # clicked object vanished
    m = AffordanceModel()
    out = m.observe_step(b, a, ("C", 3, 3), None, BG, 0.0, False)
    assert out == {ITEM: Effect.COLLECT}


# --- confidence / voting --------------------------------------------------------------

def test_majority_vote_and_confidence():
    # PASS, PASS, PUSH -> winner PASS with Dirichlet conf (2+0.5)/(3+4) = 2.5/7
    cs = ColorStat(ITEM)
    cs.observe(Effect.PASS, 0)
    cs.observe(Effect.PASS, 1)
    cs.observe(Effect.PUSH, 2)
    e, conf, n = cs.verdict()
    assert e == Effect.PASS
    assert n == 3
    assert abs(conf - (2.5 / 7.0)) < 1e-9


def test_reliable_gate_needs_support_and_confidence():
    # one PASS observation: support 1 -> not reliable even though it's the only vote
    cs = ColorStat(WALL)
    cs.observe(Effect.PASS, 0)
    assert not Verdict(*cs.verdict()).reliable
    cs.observe(Effect.PASS, 1)  # support 2, conf (2.5/6)=0.41 < 0.66 -> still not reliable
    assert not Verdict(*cs.verdict()).reliable


def test_block_reliable_after_repeated_observations():
    # pure-BLOCK confidence is (n+0.5)/(n+4); it crosses the 0.66 reliability gate at n>=7.
    m = AffordanceModel()
    cs = m.stats.setdefault(WALL, ColorStat(WALL))
    for s in range(6):
        cs.observe(Effect.BLOCK, s)
    assert not m.blocks(WALL)  # 6/6 -> conf (6.5/10)=0.65 < 0.66, not yet reliable
    cs.observe(Effect.BLOCK, 6)
    assert m.blocks(WALL)      # 7/7 -> conf (7.5/11)=0.68 >= 0.66, reliable
    e, conf, n = cs.verdict()
    assert e == Effect.BLOCK and n == 7 and conf >= 0.66


def test_harm_goal_reliable_at_single_shot():
    cs = ColorStat(ITEM)
    cs.observe(Effect.HARM, 0)
    v = Verdict(*cs.verdict())
    assert v.effect == Effect.HARM
    assert v.confidence >= 0.9
    assert v.reliable  # HARM/GOAL reliable at support 1


def test_priority_tiebreak_goal_beats_block():
    # equal counts: GOAL must win the tie over BLOCK (terminal signal dominates)
    cs = ColorStat(ITEM)
    cs.observe(Effect.BLOCK, 0)
    cs.observe(Effect.GOAL, 1)
    assert cs.verdict()[0] == Effect.GOAL


def test_navigation_cost_contract():
    m = AffordanceModel()
    # unknown color -> neutral 1.0
    assert m.navigation_cost(99) == 1.0
    # learned harm -> inf
    m.harm.add(ITEM)
    assert m.navigation_cost(ITEM) == float("inf")
    # reliable block -> inf (needs >=7 pure observations to cross the reliability gate)
    cs = m.stats.setdefault(WALL, ColorStat(WALL))
    for s in range(7):
        cs.observe(Effect.BLOCK, s)
    assert m.navigation_cost(WALL) == float("inf")


def test_reset_level_keeps_color_stats_clears_obj():
    m = AffordanceModel()
    m._record(WALL, Effect.BLOCK, 0, obj_key=("o", 1))
    assert WALL in m.stats and ("o", 1) in m.by_obj
    m.reset_level()
    assert WALL in m.stats        # color priors carried across levels
    assert ("o", 1) not in m.by_obj  # per-object stats cleared


# --- DETERMINISM golden: reactive token sequences identical for flag off/on -----------

def _capture_tokens(gid, budget=4000, **kw):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.policy import HybridPolicy

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                    logger=logging.getLogger("test_affordance"))
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
    # the core no-regression gate: enable_affordance must not change any action token, on the
    # fragile push case AND maze (and a collect game for good measure).
    for gid in ("push", "maze", "collect"):
        base, _ = _capture_tokens(gid, enable_affordance=False)
        on, _ = _capture_tokens(gid, enable_affordance=True)
        assert base == on, f"enable_affordance changed action tokens for {gid}"


# --- integration: flag-on learns real verdicts on local games -------------------------

def test_maze_learns_block_for_wall():
    _, pol = _capture_tokens("maze", budget=400, enable_affordance=True)
    # the maze run should have recorded at least one reliable BLOCK (the wall color)
    blocking = [c for c in pol.aff.stats if pol.aff.blocks(c)]
    assert blocking, "maze run learned no reliable BLOCK affordance"


def test_push_learns_push_affordance():
    # The push run must learn that some color's DOMINANT contact effect is PUSH (the movable
    # block). We assert the majority verdict rather than the strict reliability gate: a block
    # that is also pushed into walls accrues some BLOCK votes, which can keep the Dirichlet
    # confidence just under the conservative 0.66 gate while the verdict is still PUSH.
    _, pol = _capture_tokens("push", budget=600, enable_affordance=True)
    pushy = [c for c, s in pol.aff.stats.items() if s.verdict()[0] == Effect.PUSH]
    assert pushy, "push run learned no PUSH-dominant affordance"


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
    m = AffordanceModel()
    m.observe_step(b, a, ("S", 4), mm, BG, 0.0, False)  # warm
    t0 = time.perf_counter()
    iters = 50
    for _ in range(iters):
        m.observe_step(b, a, ("S", 4), mm, BG, 0.0, False)
    per = (time.perf_counter() - t0) / iters
    assert per < 0.005, f"observe_step too slow: {per*1000:.2f}ms/step"
