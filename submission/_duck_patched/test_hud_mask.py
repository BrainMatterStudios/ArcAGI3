"""Tests for patches 8/9 (HUD wavefront mask) — synthetic, recorded-frame, and
sandbox coverage.

Recorded-frame tests use the stored episodes at
scratchpad/rl_gate/episodes/*/artifacts/viewer_data_events.jsonl (per-action
64x64 boards) and the 18-game regional HUD map validated 2026-08-01
(scratchpad/ideas/hud_mask.py), inlined below as ground truth.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_hud_mask.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
EPISODES = REPO / "scratchpad/rl_gate/episodes"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import HudMaskTracker, _HUD_TLS  # noqa: E402

# Validated regional HUD map (scratchpad/ideas/hud_mask.py, 2026-08-01): the
# ground truth that any detector-produced mask must stay inside.
HUD_REGIONS = {
    "cd82": ("row", 63), "dc22": ("row", 63), "ka59": ("row", 63), "re86": ("row", 63),
    "s5i5": ("row", 63), "tu93": ("row", 63), "wa30": ("row", 63), "bp35": ("row", 63),
    "tr87": ("row", 63),
    "cn04": ("row", 0), "vc33": ("row", 0), "sp80": ("row", 0), "lf52": ("row", 0),
    "sk48": ("row", 53), "sb26": ("row", 53),
    "r11l": ("col", 0), "lp85": ("col", 0),
    "ls20": ("rows", (61, 62)),
}


def region_mask(stem: str) -> np.ndarray:
    kind, val = HUD_REGIONS[stem]
    region = np.zeros((64, 64), bool)
    if kind == "row":
        region[val, :] = True
    elif kind == "col":
        region[:, val] = True
    else:
        region[val[0]: val[1] + 1, :] = True
    return region


# --- synthetic tracker behavior ------------------------------------------------


def _base_grid() -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int16)
    g[10:14, 10:14] = 5  # some content
    return g


def _with_bar(base: np.ndarray, ticked: int, bar_row: int = 63) -> np.ndarray:
    g = base.copy()
    g[bar_row, :] = 3
    if ticked > 0:
        g[bar_row, :ticked] = 0
    return g


def _run_bar_segment(tracker: HudMaskTracker, base: np.ndarray, level: int,
                     steps: int = 14) -> list[dict]:
    """One reset-free segment: the bar ticks one cell per action."""
    infos = []
    for t in range(1, steps + 1):
        infos.append(
            tracker.observe(_with_bar(base, t), levels_completed=level,
                            state_name="NOT_FINISHED")
        )
    return infos


def _advance_level(tracker: HudMaskTracker, new_base: np.ndarray, new_level: int):
    """Level transition: scene changes AND the bar refills — a segment boundary."""
    return tracker.observe(_with_bar(new_base, 0), levels_completed=new_level,
                           state_name="NOT_FINISHED")


def _confirmed_tracker() -> tuple[HudMaskTracker, np.ndarray]:
    """A tracker whose bar mask is confirmed across MIN_SEGMENTS segments."""
    tracker = HudMaskTracker()
    base = _base_grid()
    tracker.seed(_with_bar(base, 0))
    level = 0
    for _segment in range(HudMaskTracker.MIN_SEGMENTS):
        _run_bar_segment(tracker, base, level)
        level += 1
        base = _base_grid()
        base[20 + level, 20] = 7  # scene differs per level
        _advance_level(tracker, base, level)
    return tracker, base


def test_tracker_masks_ticking_bar_after_cross_segment_confirmation():
    tracker, base = _confirmed_tracker()
    mask = tracker.mask_array()
    assert mask is not None and mask.any(), "bar was never masked"
    assert mask[63, :14].all(), "ticked bar cells must be masked"
    assert not mask[:63, :].any(), "nothing outside the bar row may be masked"

    # Tick-only actions in the next segment: raw diff non-empty, masked empty.
    info = tracker.observe(_with_bar(base, 1), levels_completed=None,
                           state_name="NOT_FINISHED")
    assert info["raw_changed"] is True
    assert info["masked_changed"] is False, "a pure HUD tick must be masked out"


def test_tracker_reports_content_changes_through_the_mask():
    tracker, base = _confirmed_tracker()
    changed = base.copy()
    changed[30, 30] = 9  # real content change
    info = tracker.observe(_with_bar(changed, 1), state_name="NOT_FINISHED")
    assert info["raw_changed"] is True
    assert info["masked_changed"] is True, "content changes must survive the mask"


def test_tracker_not_confirmed_before_min_segments():
    tracker = HudMaskTracker()
    base = _base_grid()
    tracker.seed(_with_bar(base, 0))
    _run_bar_segment(tracker, base, 0)  # a single segment only
    mask = tracker.mask_array()
    assert mask is not None and not mask.any(), (
        "one segment of evidence must not mask anything yet"
    )


def test_reset_refill_does_not_poison_the_statistic():
    """A RESET refills the bar; with the segment break the strip must survive."""
    tracker = HudMaskTracker()
    base = _base_grid()
    tracker.seed(_with_bar(base, 0))
    for _ in range(HudMaskTracker.MIN_SEGMENTS):
        _run_bar_segment(tracker, base, 0)
        tracker.observe(_with_bar(base, 0), is_reset=True, levels_completed=0,
                        state_name="NOT_FINISHED")  # refill via RESET
    _run_bar_segment(tracker, base, 0, steps=11)
    info = tracker.observe(_with_bar(base, 12), state_name="NOT_FINISHED")
    assert info["raw_changed"] is True
    assert info["masked_changed"] is False
    assert tracker.mask_array()[63, :12].all()


def test_unmask_guard_drops_region_permanently():
    """A masked cell changing twice without RESET drops the region for good."""
    tracker, base = _confirmed_tracker()
    assert tracker.mask_array().any()

    # Real content enters the strip: cell (63, 5) flips twice in one segment.
    poked = _with_bar(base, 1)
    poked[63, 5] = 9
    tracker.observe(poked, state_name="NOT_FINISHED")
    poked2 = _with_bar(base, 1)
    poked2[63, 5] = 12
    tracker.observe(poked2, state_name="NOT_FINISHED")

    assert not tracker.mask_array().any(), "guard must drop the whole strip"

    # ... and it must never come back, even after fresh clean confirmations.
    level = 90
    for _ in range(HudMaskTracker.MIN_SEGMENTS + 1):
        _advance_level(tracker, base, level)
        _run_bar_segment(tracker, base, level)
        level += 1
    assert not tracker.mask_array()[63, :5].any(), (
        "a dropped region must stay dropped (edge-strip games depend on this)"
    )
    info = tracker.observe(_with_bar(base, 15), state_name="NOT_FINISHED")
    assert info["masked_changed"] is True


def test_stack_taint_rejects_2d_content_fills():
    """A block painted line-by-line satisfies the per-line statistic (measured
    on vc33 episodes) — the stack taint must keep it out of the mask."""
    tracker = HudMaskTracker()
    base = _base_grid()
    tracker.seed(_with_bar(base, 0))
    level = 0
    for _segment in range(HudMaskTracker.MIN_SEGMENTS + 1):
        # Bar ticks AND a 6-row block fills one column per action.
        g = _with_bar(base, 0)
        for t in range(1, 13):
            g = g.copy()
            g[63, t - 1] = 0            # bar tick
            g[20:26, 10 + t] = 8        # block fill: 6 rows at once
            tracker.observe(g, levels_completed=level, state_name="NOT_FINISHED")
        level += 1
        base = _base_grid()
        base[40, 40 + level] = 7
        _advance_level(tracker, base, level)
    mask = tracker.mask_array()
    assert mask[63, :12].all(), "the true bar must still be masked"
    assert not mask[20:26, :].any(), "the 2D fill must never be masked"
    assert ("H", 20) in tracker._tainted_lines


def test_two_line_bar_is_allowed():
    """ls20's bar is two rows tall — MAX_STACK must not reject it."""
    tracker = HudMaskTracker()
    base = _base_grid()

    def two_row_bar(b: np.ndarray, ticked: int) -> np.ndarray:
        g = b.copy()
        g[61:63, :] = 3
        if ticked:
            g[61:63, :ticked] = 0
        return g

    tracker.seed(two_row_bar(base, 0))
    level = 0
    for _segment in range(HudMaskTracker.MIN_SEGMENTS):
        for t in range(1, 13):
            tracker.observe(two_row_bar(base, t), levels_completed=level,
                            state_name="NOT_FINISHED")
        level += 1
        base = _base_grid()
        base[5, 5 + level] = 7
        tracker.observe(two_row_bar(base, 0), levels_completed=level,
                        state_name="NOT_FINISHED")
    mask = tracker.mask_array()
    assert mask[61, :12].all() and mask[62, :12].all(), "two-row bars are legitimate"


# --- recorded frames -----------------------------------------------------------


def _episode_events(name: str) -> list[dict]:
    path = EPISODES / name / "artifacts" / "viewer_data_events.jsonl"
    if not path.exists():
        pytest.skip(f"episode {name} not present")
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if r.get("type") in ("initial", "action")]


def _replay_episode(events: list[dict]) -> tuple[HudMaskTracker, list[dict], list[dict]]:
    tracker = HudMaskTracker()
    tracker.seed(events[0]["board"])
    infos = []
    for event in events[1:]:
        infos.append(
            tracker.observe(
                event["board"],
                is_reset=(event.get("action_name") == "RESET"),
                levels_completed=event.get("score"),
                state_name=event.get("state"),
            )
        )
    return tracker, infos, events[1:]


# Episodes where the HUD bar demonstrably ticks under real play. The counts were
# measured once (2026-08-02) and asserted as lower bounds so a detector
# regression fails loudly.
TICKING_EPISODES = [
    ("k3_batch1_dc22", "dc22", 40),
    ("k3_batch1_cd82", "cd82", 3),
    ("k3_batch1_vc33", "vc33", 3),
    ("k3_sweep_ka59_a", "ka59", 5),
    ("k3_sweep_bp35_b", "bp35", 5),
]


@pytest.mark.parametrize("name,stem,min_hud_only", TICKING_EPISODES)
def test_recorded_hud_ticks_are_masked(name: str, stem: str, min_hud_only: int):
    """Known no-ops: raw diff non-empty, masked diff empty, on real frames."""
    tracker, infos, events = _replay_episode(_episode_events(name))
    hud_only = sum(
        1
        for info, event in zip(infos, events)
        if info["raw_changed"]
        and not info["masked_changed"]
        and event.get("action_name") != "RESET"
    )
    assert hud_only >= min_hud_only, (
        f"{name}: expected >= {min_hud_only} HUD-only ticks, got {hud_only}"
    )


REGION_EPISODES = [
    ("k3_batch1_cd82", "cd82"),
    ("k3_batch1_dc22", "dc22"),
    ("k3_batch1_sb26", "sb26"),
    ("k3_batch1_vc33", "vc33"),
    ("k3_sweep_dc22_a2", "dc22"),
    ("k3_sweep_ka59_a", "ka59"),
    ("k3_sweep_bp35_b", "bp35"),
    ("k3_sweep_wa30_b2", "wa30"),
    ("k3_sweep_tu93_a2", "tu93"),
    ("k3_sweep_vc33_a2", "vc33"),
    ("k3_sweep_sp80_b2", "sp80"),
    ("ladder_sb26_k27code", "sb26"),
]


@pytest.mark.parametrize("name,stem", REGION_EPISODES)
def test_recorded_mask_never_covers_real_content(name: str, stem: str):
    """Zero masking of real content: every masked cell must lie inside the
    validated HUD region for that game."""
    tracker, _infos, _events = _replay_episode(_episode_events(name))
    mask = tracker.mask_array()
    if mask is None or not mask.any():
        return  # nothing masked at all is trivially safe
    outside = mask & ~region_mask(stem)
    assert not outside.any(), (
        f"{name}: {int(outside.sum())} masked cells outside the validated "
        f"{HUD_REGIONS[stem]} region: {np.argwhere(outside)[:10].tolist()}"
    )


@pytest.mark.parametrize("name,stem", REGION_EPISODES[:6])
def test_recorded_level_transitions_never_fully_masked(name: str, stem: str):
    """A level completion must always read as a (masked) board change."""
    tracker, infos, events = _replay_episode(_episode_events(name))
    for info, event in zip(infos, events):
        if event.get("level_completed"):
            assert info["masked_changed"], (
                f"{name}: level completion at action {event.get('action_num')} "
                "was masked to a no-op"
            )


# --- board_changed integration --------------------------------------------------


class _FakeEnum:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFrame:
    def __init__(self, data) -> None:
        self.data = data


class _FakeState:
    def __init__(self, grid, levels: int = 0, state_name: str = "NOT_FINISHED"):
        self.frame = _FakeFrame([list(row) for row in grid])
        self.levels_completed = levels
        self.raw = type("Raw", (), {"state": _FakeEnum(state_name)})()


class _FakeGame:
    def __init__(self) -> None:
        self.current_state = None


class _FakeSession:
    def __init__(self) -> None:
        self.game = _FakeGame()


class _FakeAction:
    def __init__(self, name: str) -> None:
        self.id = _FakeEnum(name)


def test_board_changed_is_masked_in_execute_action(monkeypatch):
    from inference.framework import solver

    def _stub(self, action, *args, **kwargs):
        return {"executed": True, "board_changed": True}

    monkeypatch.setattr(
        solver._HarnessGameSession, "_execute_action", _stub, raising=True
    )
    status = duck_patches.patch_hud_board_identity()
    assert "OK" in status, status
    patched = solver._HarnessGameSession._execute_action

    session = _FakeSession()
    base = _base_grid()
    level = 0
    session.game.current_state = _FakeState(_with_bar(base, 0), level)

    # Confirm the bar across MIN_SEGMENTS reset-free segments.
    for _segment in range(HudMaskTracker.MIN_SEGMENTS):
        for t in range(1, 13):
            session.game.current_state = _FakeState(_with_bar(base, t), level)
            patched(session, _FakeAction("ACTION1"))
        level += 1
        base = _base_grid()
        base[40, 40 + level] = 7
        session.game.current_state = _FakeState(_with_bar(base, 0), level)
        patched(session, _FakeAction("ACTION1"))

    # A pure HUD tick must now read board_changed=False.
    session.game.current_state = _FakeState(_with_bar(base, 1), level)
    payload = patched(session, _FakeAction("ACTION1"))
    assert payload["board_changed"] is False
    assert payload["board_changed_hud_only"] is True

    # A content change must stay True.
    content = base.copy()
    content[30, 30] = 9
    session.game.current_state = _FakeState(_with_bar(content, 2), level)
    payload = patched(session, _FakeAction("ACTION1"))
    assert payload["board_changed"] is True
    assert "board_changed_hud_only" not in payload

    # Kill switch: with TAAF_HUD_MASK=0 the payload is left alone.
    monkeypatch.setenv("TAAF_HUD_MASK", "0")
    session.game.current_state = _FakeState(_with_bar(content, 3), level)
    payload = patched(session, _FakeAction("ACTION1"))
    assert payload["board_changed"] is True


def test_animation_summary_ignores_masked_cells():
    _HUD_TLS.mask_cells = [[0, 1]]  # row 0, col 1 == the transient cell below
    try:
        mid = [[0, 3], [0, 0]]
        final = [[0, 0], [0, 0]]

        class _Frame:
            def __init__(self, data):
                self.data = data

        class _State:
            frame = _Frame(final)
            animation_frames = [_Frame(mid)]

        out = duck_patches.summarize_animation(_State())
        assert out["animation_changed"] is False
        assert out["animation_changed_cell_count"] == 0
    finally:
        _HUD_TLS.mask_cells = []


# --- sandbox state_hash / diff_frames -------------------------------------------


def _sandbox_state(grid_a, grid_b, hud_mask):
    def frame_payload(grid):
        return {
            "ascii": "",
            "step": 0,
            "level": 1,
            "shape": [len(grid), len(grid[0])],
            "grid": [list(row) for row in grid],
        }

    return {
        "current_frame": frame_payload(grid_a),
        "history": [{"action": "UP", "frame": frame_payload(grid_b)}],
        "valid_actions": [],
        "last_action_result": {},
        "hud_mask": hud_mask,
    }


def test_sandbox_state_hash_and_diff_frames_are_hud_aware():
    from inference.agent import python_tool_sandbox as sandbox_mod

    status = duck_patches.patch_hud_sandbox()
    assert "OK" in status or "SKIP" in status, status
    assert "HUD_MASK_CELLS" in sandbox_mod._SANDBOX_BOOTSTRAP

    grid_a = [[1, 1, 1, 1] for _ in range(4)]
    grid_b = [list(row) for row in grid_a]
    grid_b[0][2] = 5  # differs only at (0, 2)

    code = (
        "same_hash = state_hash(current_frame) == state_hash(history[0].frame)\n"
        "diff = diff_frames(current_frame, history[0].frame)\n"
        "result = {'same_hash': same_hash, 'changed': diff['changed']}\n"
    )

    def run(hud_mask):
        out = sandbox_mod.run_sandboxed_python(
            code=code,
            timeout_seconds=30,
            initial_state=_sandbox_state(grid_a, grid_b, hud_mask),
            action_handler=lambda actions: {},
        )
        assert not out.get("error"), out
        return out["result"]

    masked = run([[0, 2]])
    assert masked["same_hash"] is True, "masked cell must not affect state_hash"
    assert masked["changed"] is False, "masked cell must not show in diff_frames"

    unmasked = run([])
    assert unmasked["same_hash"] is False, "without a mask the diff must be seen"
    assert unmasked["changed"] is True


def test_sandbox_wrapper_injects_live_mask():
    """run_sandboxed_python must inject the thread-local mask into the state."""
    from inference.agent import python_tool_sandbox as sandbox_mod

    duck_patches.patch_hud_sandbox()
    _HUD_TLS.mask_cells = [[0, 2]]
    try:
        grid_a = [[1, 1, 1, 1] for _ in range(4)]
        grid_b = [list(row) for row in grid_a]
        grid_b[0][2] = 5
        state = _sandbox_state(grid_a, grid_b, hud_mask=None)
        del state["hud_mask"]
        out = sandbox_mod.run_sandboxed_python(
            code="result = state_hash(current_frame) == state_hash(history[0].frame)\n",
            timeout_seconds=30,
            initial_state=state,
            action_handler=lambda actions: {},
        )
        assert not out.get("error"), out
        assert out["result"] is True
    finally:
        _HUD_TLS.mask_cells = []
