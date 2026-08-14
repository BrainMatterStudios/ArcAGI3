"""Perception unit tests: HUD-line learning, canonical state, determinism,
and canonical-state stability under HUD ticks on known ticker games."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from engineered.battery import BatteryConfig, PerceptionProfile, ProbeBattery
from engineered.perception import (
    MaskedState,
    Perception,
    learn_hud_lines,
    lines_to_mask,
    settled_frame,
)


# ---------------------------------------------------------------------------
# learn_hud_lines on synthetic evidence
# ---------------------------------------------------------------------------

def _hud_world_diffs() -> list[np.ndarray]:
    """12 steps of [y, x, old, new]: a roving monotone bar on row 63 (HUD), a
    static blinker (fails rove), an oscillating paddle (fails monotone), and
    a 2-D avatar (never reaches candidate rate on one line)."""
    diffs = []
    for t in range(12):
        pts = [[63, (5 * t) % 64, 3, 0], [63, (5 * t + 1) % 64, 3, 0]]
        if t % 2 == 0:
            v = 7 if (t // 2) % 2 == 0 else 2
            pts += [[40, 8, 9 - v, v], [40, 9, 9 - v, v]]
        if t % 3 == 1:
            p = 10 + 2 * (t % 5)
            pts += [[30, p, 5, 0], [30, p + 2, 0, 5]]
        if t % 3 == 0:
            r, c = 20 + t % 4, 28 + (t // 3) % 4
            pts += [[r, c, 0, 2], [r + 1, c, 0, 2], [r, c + 1, 2, 0]]
        diffs.append(np.array(pts, dtype=int))
    return diffs


def test_learn_hud_lines_finds_only_the_ticker() -> None:
    assert learn_hud_lines(_hud_world_diffs()) == [("row", 63)]


def test_learn_hud_lines_empty_without_evidence() -> None:
    assert learn_hud_lines([]) == []
    # pure board changes, no line-confined housekeeping
    diffs = [np.array([[10 + t, 10 + t, 0, 2]]) for t in range(10)]
    assert learn_hud_lines(diffs) == []


def test_learn_hud_lines_joint_two_row_hud() -> None:
    """m0r0 class: two bars that tick in the SAME step — neither alone
    explains a step, so per-line greedy fails; the set-based prune must
    keep both."""
    diffs = []
    for t in range(12):
        pts = [[0, (3 * t) % 64, 5, 0], [63, (3 * t + 1) % 64, 5, 0]]
        if t % 4 == 0:  # occasional real change: a toggling object (board
            v, w = (0, 2) if (t // 4) % 2 == 0 else (2, 0)  # content revisits)
            pts += [[30, 10, v, w], [30, 11, v, w]]
        diffs.append(np.array(pts, dtype=int))
    assert learn_hud_lines(diffs) == [("row", 0), ("row", 63)]


def test_learn_hud_lines_secondary_channel_needs_action_diversity() -> None:
    """A monotone roving line that never ticks alone (wa30 class) is accepted
    only with >= 3 distinct probed action ids."""
    diffs = []
    for t in range(12):
        p = 10 + 2 * (t % 3)  # oscillating paddle: board content, revisits
        pts = [[30, p, 5, 0], [30, p + 2, 0, 5]]
        if t % 3 == 0:
            pts += [[63, t, 7, 4]]  # bar tick, always during board motion
        diffs.append(np.array(pts, dtype=int))
    ids_diverse = [1, 2, 3, 4] * 3  # ticks land on ids 1, 4, 3, 2
    ids_single = [6] * 12
    assert ("row", 63) in learn_hud_lines(diffs, action_ids=ids_diverse)
    assert learn_hud_lines(diffs, action_ids=ids_single) == []


# ---------------------------------------------------------------------------
# Perception / MaskedState
# ---------------------------------------------------------------------------

def test_masked_state_ignores_hud_only_changes() -> None:
    p = Perception("test", [("row", 63), ("col", 0)])
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[63, 10] = 7
    b[5, 0] = 3
    assert p.frames_equal(a, b)
    assert p.masked_state(a) == p.masked_state(b)
    assert hash(p.masked_state(a)) == hash(p.masked_state(b))
    c = a.copy()
    c[10, 10] = 1
    assert not p.frames_equal(a, c)
    assert p.masked_state(a) != p.masked_state(c)


def test_masked_state_distinguishes_games() -> None:
    a = np.zeros((64, 64), dtype=np.int8)
    assert Perception("g1", []).masked_state(a) != Perception("g2", []).masked_state(a)


def test_lines_to_mask_shapes() -> None:
    m = lines_to_mask([("row", 0), ("col", 63)])
    assert m[0].all() and m[:, 63].all()
    assert int(m.sum()) == 64 + 64 - 1


# ---------------------------------------------------------------------------
# Engine-backed: determinism + canonical stability on ticker games
# ---------------------------------------------------------------------------

def _profile(arcade: Any, gid_of: dict[str, str], stem: str,
             tag: str) -> PerceptionProfile:
    env = arcade.make(game_id=gid_of[stem], scorecard_id=f"t-{tag}-{stem}")
    profile, state = ProbeBattery(BatteryConfig()).run(env)
    assert state.actions_spent <= BatteryConfig().budget
    return profile

@pytest.mark.parametrize("stem", ["tu93", "sb26"])
def test_mask_determinism(arcade: Any, gid_of: dict[str, str], stem: str) -> None:
    """Same config, fresh envs -> byte-identical profiles (the battery has no
    stochastic element; the engine is deterministic per Stage 0 pillar 4)."""
    p1 = _profile(arcade, gid_of, stem, "det1")
    p2 = _profile(arcade, gid_of, stem, "det2")
    assert p1.hud_lines == p2.hud_lines
    assert np.array_equal(p1.self_mask, p2.self_mask)
    assert np.array_equal(p1.reactive_mask, p2.reactive_mask)
    assert np.array_equal(p1.dead_mask, p2.dead_mask)
    assert p1.to_json() == p2.to_json()


# lf52 and vc33 are the two strongest known ticker games: raw no-op rate
# 0.000 vs ~0.99 hand-masked (MEASURED_NOOP) — the HUD row changes on
# essentially every action while the board almost never does.
@pytest.mark.parametrize("stem,max_distinct", [("lf52", 3), ("vc33", 3)])
def test_canonical_state_stable_under_hud_ticks(
    arcade: Any, gid_of: dict[str, str], stem: str, max_distinct: int
) -> None:
    from arcengine import GameAction, GameState

    profile = _profile(arcade, gid_of, stem, "stab")
    perception = profile.perception()
    env = arcade.make(game_id=gid_of[stem], scorecard_id=f"t-stab2-{stem}")
    o = env.reset()
    rng = random.Random(0)
    raw_keys: set[bytes] = set()
    masked_keys: set[MaskedState] = set()
    steps = 0
    while steps < 25:
        if o.state in (GameState.WIN, GameState.GAME_OVER):
            break
        avail = [a for a in (o.available_actions or []) if a]
        aid = rng.choice(avail)
        if aid == 6:
            o = env.step(GameAction.ACTION6,
                         data={"x": rng.randrange(64), "y": rng.randrange(64)})
        else:
            o = env.step(GameAction.from_id(aid))
        frame = settled_frame(o)
        raw_keys.add(np.ascontiguousarray(frame, dtype=np.int8).tobytes())
        masked_keys.add(perception.masked_state(frame))
        steps += 1
    # HUD ticks every action -> raw identity explodes; canonical holds still
    assert len(raw_keys) >= steps - 2, "expected the HUD to tick ~every step"
    assert len(masked_keys) <= max_distinct


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

def test_profile_json_roundtrip(arcade: Any, gid_of: dict[str, str]) -> None:
    p = _profile(arcade, gid_of, "tu93", "json")
    q = PerceptionProfile.from_json(p.to_json())
    assert q.hud_lines == p.hud_lines
    assert np.array_equal(q.self_mask, p.self_mask)
    assert q.to_json() == p.to_json()
