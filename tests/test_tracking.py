"""Tests for arcagi3.tracking (C1 — Object tracking / persistence).

Mirror style of test_movement.py. All tests use small grids to keep them fast.
The module is DORMANT (no policy.py call site), so these are offline unit tests only.
"""

from __future__ import annotations

import time
import copy

import numpy as np
import pytest

from arcagi3 import tracking as T
from arcagi3.tracking import (
    ObjectTracker,
    TrackedObj,
    TrackResult,
    APPEARED,
    VANISHED,
    MOVED,
    CHANGED,
    RECOLORED,
    SPLIT,
    MERGED,
    _is_rigid_shift,
)
from arcagi3 import perception as P


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid(h: int = 16, w: int = 16, fill: int = 0) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.int8)


def _make_obj(id_: int, color: int, cells: list[tuple[int, int]]) -> TrackedObj:
    """Build a TrackedObj with minimal required fields computed from cells."""
    cells_t = tuple(cells)
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    size = len(cells)
    centroid = (sum(rs) / size, sum(cs) / size)
    return TrackedObj(
        id=id_,
        color=color,
        cells=cells_t,
        bbox=bbox,
        size=size,
        centroid=centroid,
        age=0,
        missed=0,
        first_seen=0,
        last_seen=0,
        velocity=(0.0, 0.0),
        color_history=(color,),
    )


def _tracker(bg: int = 0, ignore: set[int] | None = None) -> ObjectTracker:
    return ObjectTracker(bg=bg, ignore_colors=ignore or set())


def _kinds(result: TrackResult) -> list[str]:
    return [ev.kind for ev in result.events]


# ---------------------------------------------------------------------------
# _is_rigid_shift helper
# ---------------------------------------------------------------------------

def test_is_rigid_shift_true():
    prev = ((0, 0), (0, 1), (1, 0))
    cur  = ((0, 1), (0, 2), (1, 1))   # shifted right by (0,1)
    assert _is_rigid_shift(prev, cur, 0, 1)


def test_is_rigid_shift_false_when_deformed():
    prev = ((0, 0), (0, 1), (1, 0))
    cur  = ((0, 0), (0, 2), (1, 0))   # different shape
    assert not _is_rigid_shift(prev, cur, 0, 0)


# ---------------------------------------------------------------------------
# objects() — stateless segmentation
# ---------------------------------------------------------------------------

def test_objects_basic():
    g = _grid()
    g[2, 2] = 5
    g[10, 10] = 7
    tracker = _tracker(bg=0)
    objs = tracker.objects(g)
    colors = {o.color for o in objs}
    assert colors == {5, 7}
    assert all(o.id == -1 for o in objs)


def test_objects_ignores_color():
    g = _grid()
    g[2, 2] = 5
    g[10, 10] = 7
    tracker = _tracker(bg=0, ignore={7})
    objs = tracker.objects(g)
    assert all(o.color != 7 for o in objs)
    assert any(o.color == 5 for o in objs)


# ---------------------------------------------------------------------------
# moved-keeps-id (rigid, one MOVED, no CHANGED)
# ---------------------------------------------------------------------------

def test_moved_keeps_id():
    g1 = _grid()
    g1[5, 5] = 3
    g2 = _grid()
    g2[5, 6] = 3   # moved right

    tracker = _tracker()
    r1 = tracker.update(g1)
    r2 = tracker.update(g2)

    assert len(r2.objects) == 1
    obj = r2.objects[0]
    # id must be the same as after first frame
    first_id = r1.objects[0].id
    assert obj.id == first_id

    moved_evs = [ev for ev in r2.events if ev.kind == MOVED]
    changed_evs = [ev for ev in r2.events if ev.kind == CHANGED]
    assert len(moved_evs) == 1
    assert len(changed_evs) == 0   # rigid shift -> no CHANGED


# ---------------------------------------------------------------------------
# grew-in-place (CHANGED, no MOVED)
# ---------------------------------------------------------------------------

def test_grew_in_place():
    g1 = _grid()
    g1[5, 5] = 3
    g2 = _grid()
    g2[5, 5] = 3
    g2[5, 6] = 3   # grew right — centroid shifts slightly but delta rounds to 0 col?

    # Use a symmetric grow so centroid stays at same rounded position.
    g1 = _grid()
    g1[5, 5] = 3
    g2 = _grid()
    g2[5, 4] = 3
    g2[5, 5] = 3
    g2[5, 6] = 3   # 3-cell horizontal — centroid still (5,5) -> no MOVED

    tracker = _tracker()
    r1 = tracker.update(g1)
    r2 = tracker.update(g2)

    changed_evs = [ev for ev in r2.events if ev.kind == CHANGED]
    moved_evs   = [ev for ev in r2.events if ev.kind == MOVED]
    assert len(changed_evs) == 1
    assert changed_evs[0].size_delta == 2
    assert len(moved_evs) == 0


# ---------------------------------------------------------------------------
# appeared
# ---------------------------------------------------------------------------

def test_appeared():
    g1 = _grid()
    g2 = _grid()
    g2[3, 3] = 8

    tracker = _tracker()
    r1 = tracker.update(g1)
    r2 = tracker.update(g2)

    appeared = [ev for ev in r2.events if ev.kind == APPEARED]
    assert len(appeared) == 1
    assert appeared[0].color == 8
    assert appeared[0].id >= 0   # real id assigned


# ---------------------------------------------------------------------------
# vanished — only after > occlusion_grace missed frames
# ---------------------------------------------------------------------------

def test_vanished_only_after_grace():
    g1 = _grid()
    g1[5, 5] = 6

    tracker = _tracker()
    r1 = tracker.update(g1)
    obj_id = r1.objects[0].id

    blank = _grid()
    # First missed frame (missed=1, grace=2): still alive, no VANISHED.
    r2 = tracker.update(blank)
    assert VANISHED not in _kinds(r2)
    assert obj_id in tracker._tracks   # still coasting

    # Second missed frame (missed=2, grace=2): still alive.
    r3 = tracker.update(blank)
    assert VANISHED not in _kinds(r3)
    assert obj_id in tracker._tracks

    # Third missed frame (missed=3 > grace=2): now VANISHED.
    r4 = tracker.update(blank)
    van_evs = [ev for ev in r4.events if ev.kind == VANISHED]
    assert any(ev.id == obj_id for ev in van_evs)
    assert obj_id not in tracker._tracks


# ---------------------------------------------------------------------------
# two same-color objects don't swap ids when one moves
# ---------------------------------------------------------------------------

def test_no_id_swap_same_color():
    """Object at (2,2) stays put; object at (10,2) moves to (10,3). Ids must not swap."""
    g1 = _grid()
    g1[2, 2] = 4    # A — close to top
    g1[10, 2] = 4   # B — far from top

    g2 = _grid()
    g2[2, 2] = 4    # A — stays
    g2[10, 3] = 4   # B — moved right

    tracker = _tracker()
    r1 = tracker.update(g1)
    # Sort by centroid row so we can identify A vs B.
    by_row = sorted(r1.objects, key=lambda o: o.centroid[0])
    id_A, id_B = by_row[0].id, by_row[1].id

    r2 = tracker.update(g2)
    by_row2 = sorted(r2.objects, key=lambda o: o.centroid[0])
    # A is still at row 2, B at row 10.
    assert by_row2[0].id == id_A
    assert by_row2[1].id == id_B


# ---------------------------------------------------------------------------
# translation prior disambiguates two same-color movers
# ---------------------------------------------------------------------------

def test_translation_prior_disambiguates():
    """Both color-4 objects move right by 1; prior should match correctly (not cross-swap)."""
    g1 = _grid(w=20)
    g1[2, 2] = 4
    g1[8, 2] = 4

    g2 = _grid(w=20)
    g2[2, 3] = 4   # both moved right
    g2[8, 3] = 4

    tracker = _tracker()
    r1 = tracker.update(g1)
    by_row1 = sorted(r1.objects, key=lambda o: o.centroid[0])
    id_top, id_bot = by_row1[0].id, by_row1[1].id

    r2 = tracker.update(g2)
    by_row2 = sorted(r2.objects, key=lambda o: o.centroid[0])
    assert by_row2[0].id == id_top
    assert by_row2[1].id == id_bot


# ---------------------------------------------------------------------------
# recolor at fixed footprint -> RECOLORED, same id
# ---------------------------------------------------------------------------

def test_recolor_same_id():
    g1 = _grid()
    g1[5, 5] = 2   # switch color

    g2 = _grid()
    g2[5, 5] = 9   # door color — same single cell

    tracker = _tracker()
    r1 = tracker.update(g1)
    orig_id = r1.objects[0].id

    r2 = tracker.update(g2)
    recolored = [ev for ev in r2.events if ev.kind == RECOLORED]
    assert len(recolored) == 1
    assert recolored[0].id == orig_id
    assert recolored[0].extra["old_color"] == 2
    assert recolored[0].extra["new_color"] == 9
    # The object keeps its id.
    assert r2.objects[0].id == orig_id


# ---------------------------------------------------------------------------
# SPLIT: 1 -> 2 with lineage
# ---------------------------------------------------------------------------

def test_split_lineage():
    """A 4-cell block splits into two 2-cell blocks."""
    g1 = _grid()
    g1[5, 4] = 3
    g1[5, 5] = 3
    g1[5, 6] = 3
    g1[5, 7] = 3   # 4-cell horizontal block

    g2 = _grid()
    g2[5, 4] = 3
    g2[5, 5] = 3   # left fragment (2 cells)
    g2[5, 6] = 3
    g2[5, 7] = 3   # right fragment still same position -> falls through to split
    # Actually to trigger split we need a gap: insert background in middle.
    g2 = _grid()
    g2[5, 4] = 3
    g2[5, 5] = 3   # left fragment
    # gap at (5,6)
    g2[5, 7] = 3
    g2[5, 8] = 3   # right fragment (non-adjacent -> separate components)

    tracker = _tracker()
    r1 = tracker.update(g1)
    orig_id = r1.objects[0].id

    r2 = tracker.update(g2)
    split_evs = [ev for ev in r2.events if ev.kind == SPLIT]
    assert len(split_evs) == 1
    assert split_evs[0].id == orig_id
    # Two objects in the result.
    assert len(r2.objects) == 2
    # Largest fragment keeps the original id.
    largest = max(r2.objects, key=lambda o: o.size)
    assert largest.id == orig_id


# ---------------------------------------------------------------------------
# MERGE: 2 -> 1, oldest id kept
# ---------------------------------------------------------------------------

def test_merge_oldest_id():
    """Two adjacent objects merge into one; the one with the lower (older) id is kept.

    Merge is detected via cell-set overlap: both prev objects must each overlap >= 50% of
    their own cells with the merged cur object.  We use two 2-cell objects that are
    adjacent to each other in frame 1; in frame 2 the wall between them disappears and
    they form one 4-cell component whose cells subsume both prev cell-sets.
    """
    g1 = _grid()
    # Object A: cells (5,4),(5,5) — 2 cells
    g1[5, 4] = 7
    g1[5, 5] = 7
    # Object B: cells (5,7),(5,8) — 2 cells  (gap at (5,6))
    g1[5, 7] = 7
    g1[5, 8] = 7

    # Frame 2: fill the gap -> one 5-cell component (5,4)..(5,8)
    # Both prev objects overlap this merged blob by all of their own cells.
    g2 = _grid()
    for c in range(4, 9):
        g2[5, c] = 7

    tracker = _tracker()
    r1 = tracker.update(g1)
    ids = sorted(o.id for o in r1.objects)
    older_id = ids[0]

    r2 = tracker.update(g2)
    merge_evs = [ev for ev in r2.events if ev.kind == MERGED]
    assert len(merge_evs) == 1
    assert merge_evs[0].id == older_id
    assert len(r2.objects) == 1
    assert r2.objects[0].id == older_id


# ---------------------------------------------------------------------------
# distractor color in ignore_colors -> never tracked, no events
# ---------------------------------------------------------------------------

def test_ignore_color_not_tracked():
    g1 = _grid()
    g1[2, 2] = 5   # normal object
    g1[0, 0] = 9   # distractor

    g2 = _grid()
    g2[2, 3] = 5   # moved
    g2[0, 1] = 9   # distractor also moved

    tracker = _tracker(ignore={9})
    r1 = tracker.update(g1)
    r2 = tracker.update(g2)

    all_colors = {o.color for o in r2.objects}
    assert 9 not in all_colors
    assert all(ev.color != 9 for ev in r2.events)


# ---------------------------------------------------------------------------
# reset clears tracks but ids stay monotonic
# ---------------------------------------------------------------------------

def test_reset_monotonic_ids():
    g = _grid()
    g[2, 2] = 5

    tracker = _tracker()
    r1 = tracker.update(g)
    id_before = r1.objects[0].id
    id_counter_before = tracker._next_id

    tracker.reset()
    assert tracker._tracks == {}
    assert tracker._frame == 0
    # _next_id must NOT reset.
    assert tracker._next_id == id_counter_before

    r2 = tracker.update(g)
    id_after = r2.objects[0].id
    # New id must be >= old _next_id (strictly new, not reused).
    assert id_after >= id_counter_before


# ---------------------------------------------------------------------------
# match() is pure: calling twice gives identical results; inputs unmutated
# ---------------------------------------------------------------------------

def test_match_is_pure():
    prev = [_make_obj(0, 3, [(5, 5)])]
    cur  = [_make_obj(-1, 3, [(5, 6)])]

    tracker = _tracker()
    prev_copy = copy.deepcopy(prev)
    cur_copy  = copy.deepcopy(cur)

    r1 = tracker.match(prev, cur)
    r2 = tracker.match(prev, cur)

    # Inputs unmutated.
    assert prev[0].centroid == prev_copy[0].centroid
    assert cur[0].centroid  == cur_copy[0].centroid

    # Identical results.
    assert r1.correspondences == r2.correspondences
    assert [ev.kind for ev in r1.events] == [ev.kind for ev in r2.events]


# ---------------------------------------------------------------------------
# determinism: same input -> same output across repeated calls
# ---------------------------------------------------------------------------

def test_determinism():
    g1 = _grid()
    g1[5, 5] = 4
    g1[5, 10] = 4

    g2 = _grid()
    g2[5, 6] = 4
    g2[5, 11] = 4

    def _run():
        tr = _tracker()
        tr.update(g1)
        r = tr.update(g2)
        return [(o.id, o.centroid) for o in sorted(r.objects, key=lambda o: o.centroid[1])]

    assert _run() == _run()


# ---------------------------------------------------------------------------
# throughput micro-benchmark (~40 objects, 64x64)
# ---------------------------------------------------------------------------

def test_throughput_benchmark():
    """update() on a ~40-object 64x64 frame must complete 200 frames in <1 second."""
    rng = np.random.default_rng(42)
    g = np.zeros((64, 64), dtype=np.int8)
    # Place ~40 single-cell objects of various colors.
    for _ in range(40):
        r, c = rng.integers(0, 64, size=2)
        col = int(rng.integers(1, 15))
        g[r, c] = col

    tracker = ObjectTracker(bg=0)
    N = 200
    start = time.perf_counter()
    for _ in range(N):
        tracker.update(g)
    elapsed = time.perf_counter() - start
    # 200 frames should take well under 1 second on any modern machine.
    assert elapsed < 1.0, f"throughput too slow: {elapsed:.3f}s for {N} frames"
