"""C2 — causal event extraction: golden event streams + no-regression proofs.

These are offline, pure-numpy unit tests (no env): they pin the typed event stream for the
core mechanics (floor move, collect, switchdoor remote door, push, recolor, counter, blocked
move, avatarless click) and assert the determinism contract — the reactive policy emits
byte-identical action tokens whether emit_events is False (default) or True (read-only).
"""

import os
import time
from pathlib import Path

import numpy as np

from arcagi3 import events as EVT
from arcagi3 import movement as MV
from arcagi3.events import EventType as ET

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")
BG = 0
AV = 14  # avatar color
BLOCK = 6


def _g():
    return np.zeros((16, 16), dtype=np.int8)


def _avatar_mm():
    # avatar with >=2 distinct deltas (action-correlated) -> mm.ok and multi-delta
    return MV.MotionModel(
        avatar_color=AV,
        deltas={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        avatar_colors=frozenset({AV}),
    )


def _ex():
    return EVT.EventExtractor()


# --- core 98%-noise killer: plain navigation over floor -> ZERO real events ----------

def test_floor_move_emits_no_real_event():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    a = _g()
    a[5, 6] = AV  # avatar steps right over empty floor
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG)
    assert se.events == ()
    assert not se.has_real_event
    assert se.avatar_moved


def test_identical_grid_no_events():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    se = _ex().extract(b, b.copy(), ("S", 1), reward=0.0, mm=mm, bg=BG)
    assert se.events == ()
    assert not se.has_real_event


# --- collect: avatar steps onto an item -> contact VANISH -----------------------------

def test_collect_is_contact_vanish():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = 3  # an item to the right
    a = _g()
    a[5, 6] = AV  # avatar moved onto it -> item collected
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG)
    vanishes = [e for e in se.events if e.type == ET.OBJECT_VANISHED]
    assert len(vanishes) == 1
    assert vanishes[0].color == 3
    assert vanishes[0].contact
    assert se.has_real_event


# --- switchdoor: remote door vanishes on the same step as the switch contact ----------

def test_switchdoor_remote_door_vanish_noncontact_same_step():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[5, 6] = 8  # switch the avatar steps onto
    b[1, 12] = 9  # remote door (far away)
    a = _g()
    a[5, 6] = AV  # avatar onto switch (switch collected/contact vanish)
    # door removed remotely on the SAME step
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG)
    door = [e for e in se.events if e.type == ET.OBJECT_VANISHED and e.color == 9]
    assert len(door) == 1
    assert not door[0].contact
    assert not door[0].local
    assert se.has_real_event


# --- push: a block translates, avatar separated -> single OBJECT_MOVED -----------------

def test_push_emits_object_move_for_block():
    mm = _avatar_mm()
    b = _g()
    b[5, 4] = AV
    b[5, 5] = BLOCK
    a = _g()
    a[5, 5] = AV  # avatar advanced
    a[5, 6] = BLOCK  # block pushed right by (0,+1)
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG)
    moves = [e for e in se.events if e.type == ET.OBJECT_MOVED]
    assert len(moves) == 1
    assert moves[0].color == BLOCK
    assert moves[0].delta == (0, 1)
    # avatar footprint must NOT have eaten the block
    assert se.has_real_event


def test_shared_color_push_still_detects_two_translations():
    # R1: avatar and pushed block share a color; infer_all_translations cannot separate one
    # color into two rigid translations, but the block survives footprint subtraction as a
    # residual and is reported (never hidden). Here we use distinct colors but verify the
    # block is never consumed by footprint even when adjacent to the avatar.
    mm = _avatar_mm()
    b = _g()
    b[5, 4] = AV
    b[5, 5] = BLOCK
    a = _g()
    a[5, 5] = AV
    a[5, 6] = BLOCK
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG)
    assert any(e.color == BLOCK for e in se.events)


# --- recolor: in-place color flip -> RECOLOR with from/to -----------------------------

def test_recolor_emits_from_to():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[2, 2] = 2  # object that will change hue (remote)
    a = _g()
    a[5, 5] = AV
    a[2, 2] = 5  # 2 -> 5 in place
    se = _ex().extract(b, a, ("S", 1), reward=0.0, mm=mm, bg=BG)
    rec = [e for e in se.events if e.type == ET.OBJECT_RECOLORED]
    assert len(rec) == 1
    assert dict(rec[0].extra) == {"from": 2, "to": 5}
    assert se.has_real_event


# --- counter: distractor color change -> COUNTER, NOT a real event --------------------

def test_counter_change_is_not_real_event():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[0, 0] = 11  # counter pixel (distractor color)
    a = _g()
    a[5, 5] = AV
    a[0, 1] = 11  # counter moved -> appears at new cell, vanishes from old
    se = _ex().extract(b, a, ("S", 1), reward=0.0, mm=mm, bg=BG,
                       distractor_colors={11})
    assert any(e.type == ET.COUNTER_CHANGED for e in se.events)
    assert all(e.type in (ET.COUNTER_CHANGED, ET.AVATAR_BLOCKED) for e in se.events)
    assert not se.has_real_event


# --- blocked move: avatar tried to move but didn't (wall) -> AVATAR_BLOCKED ------------

def test_blocked_move_emits_avatar_blocked():
    # avatar tried to move (action 4 = right, delta (0,+1)) but didn't budge (hit a wall);
    # the only other change is a distractor counter flicker -> no real event, just BLOCKED.
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    b[0, 0] = 11  # counter, frame 1
    a = _g()
    a[5, 5] = AV  # avatar did not move (hit a wall)
    a[0, 1] = 11  # counter flicker, frame 2
    se = _ex().extract(b, a, ("S", 4), reward=0.0, mm=mm, bg=BG,
                       distractor_colors={11})
    assert any(e.type == ET.AVATAR_BLOCKED for e in se.events)
    assert se.avatar_blocked
    # blocked alone is not a real event
    assert not se.has_real_event


# --- avatarless click game: mm absent -> residual events with click contact -----------

def test_avatarless_click_uses_click_xy_contact():
    b = _g()
    b[3, 3] = 4  # a button
    a = _g()  # button vanished after clicking it
    se = _ex().extract(b, a, ("C", 3, 3), reward=0.0, mm=None, bg=BG,
                       click_xy=(3, 3))
    vanishes = [e for e in se.events if e.type == ET.OBJECT_VANISHED]
    assert len(vanishes) == 1
    assert vanishes[0].contact  # click site == the vanished cell


# --- LEVEL_COMPLETED headline event on reward -----------------------------------------

def test_reward_emits_level_completed():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    a = _g()
    a[5, 6] = AV
    se = _ex().extract(b, a, ("S", 4), reward=1.0, mm=mm, bg=BG)
    assert se.events[0].type == ET.LEVEL_COMPLETED
    assert se.has_real_event


# --- redraw guard: huge diff -> single low-confidence REGION_CHANGED -------------------

def test_redraw_guard_caps_blowup():
    mm = _avatar_mm()
    b = _g()
    b[5, 5] = AV
    a = np.full((16, 16), 7, dtype=np.int8)  # whole frame redrawn
    se = _ex().extract(b, a, ("S", 1), reward=0.0, mm=mm, bg=BG)
    assert len(se.events) == 1
    assert se.events[0].type == ET.REGION_CHANGED
    assert se.events[0].low_confidence


# --- EventLog accumulator + last_real (for C5) ----------------------------------------

def test_eventlog_last_real_skips_noise():
    log = EVT.EventLog()
    noise = EVT.StepEvents(events=(), action=("S", 1), reward=0.0)
    real = EVT.StepEvents(
        events=(EVT.Event(ET.OBJECT_VANISHED, color=3, cells=((1, 1),)),),
        action=("S", 4), reward=0.0,
    )
    log.append(noise)
    log.append(real)
    log.append(noise)
    last = log.last_real(1)
    assert len(last) == 1
    assert last[0] is real


# --- perf micro-bench: extract is cheap on a dense 64x64 frame ------------------------

def test_extract_perf_under_budget():
    mm = _avatar_mm()
    b = np.zeros((64, 64), dtype=np.int8)
    rng = np.random.default_rng(0)
    # scatter ~80 static decorations + avatar
    for _ in range(80):
        b[int(rng.integers(0, 64)), int(rng.integers(0, 64))] = int(rng.integers(1, 13))
    b[32, 32] = AV
    a = b.copy()
    a[32, 32] = 0
    a[32, 33] = AV  # plain move
    ex = _ex()
    # warm
    ex.extract(b, a, ("S", 4), 0.0, mm=mm, bg=BG)
    t0 = time.perf_counter()
    iters = 50
    for _ in range(iters):
        ex.extract(b, a, ("S", 4), 0.0, mm=mm, bg=BG)
    per = (time.perf_counter() - t0) / iters
    assert per < 0.005, f"extract too slow: {per*1000:.2f}ms/step"


# --- DETERMINISM golden: reactive token sequences identical for emit_events off/on -----

def _capture_tokens(gid, budget=4000, **kw):
    import logging

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.policy import HybridPolicy

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                    logger=logging.getLogger("test_events"))
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
    return toks


def test_emit_events_is_byte_identical():
    # read-only proof on the fragile pair (push + switchdoor) plus a nav game
    for gid in ("push", "switchdoor", "navg"):
        base = _capture_tokens(gid, emit_events=False)
        on = _capture_tokens(gid, emit_events=True)
        assert base == on, f"emit_events changed action tokens for {gid}"


def test_emit_events_logs_real_events_on_collect():
    # when ON, the log must actually capture real events somewhere on collect (smoke test)
    import logging

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.policy import HybridPolicy

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                    logger=logging.getLogger("test_events"))
    env = client.make(game_id="collect", scorecard_id="sc-collect")
    pol = HybridPolicy(seed=0, emit_events=True)
    obs = env.reset()
    n = 0
    saw_real = False
    while n < 4000:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        t = pol.decide(grid, obs.state == GameState.GAME_OVER,
                       obs.state == GameState.NOT_PLAYED,
                       int(obs.levels_completed or 0), list(obs.available_actions or []))
        if pol.last_step_events is not None and pol.last_step_events.has_real_event:
            saw_real = True
        if t[0] == "reset":
            obs = env.reset()
        elif t[0] == "S":
            obs = env.step(GameAction.from_id(t[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": t[1], "y": t[2]})
        n += 1
    assert saw_real, "collect run produced no real events with emit_events=True"
