"""The two Stage-0 target-selection traps, as pure unit tests (no engine).

Trap 1 — hollow-component centroid: a raw centroid of a hollow/non-convex
component lands off-component; the click target must be an actual component
pixel (this failure nulled all sb26 clicks on the first Stage-0 battery run).

Trap 2 — duplicate-color buttons: sb26's live answer buttons are the second,
smaller instances of already-seen colors; a dedup-by-color-only selection
misses every one of them, and a budget cut that drops the duplicate tail
does too (hence the reserve).
"""
from __future__ import annotations

import numpy as np

from engineered.battery import Component, components, select_click_targets


def _ring_frame() -> np.ndarray:
    """A hollow square ring, color 3 on background 0: centroid = dead center,
    which is NOT part of the component."""
    f = np.zeros((64, 64), dtype=np.int16)
    f[20:31, 20:31] = 3
    f[23:28, 23:28] = 0  # hollow the middle
    return f


def test_hollow_component_click_point_is_on_component() -> None:
    frame = _ring_frame()
    comps = components(frame, bg=0)
    assert len(comps) == 1
    c = comps[0]
    # the naive centroid is (25, 25) — inside the hole, off-component
    assert frame[25, 25] == 0
    # the chosen click point must be an actual pixel of the component
    assert frame[c.click_y, c.click_x] == 3


def test_components_are_color_and_connectivity_split() -> None:
    f = np.zeros((64, 64), dtype=np.int16)
    f[5:8, 5:8] = 2      # blob A color 2
    f[5:8, 20:23] = 2    # blob B same color, disconnected
    f[30:33, 5:8] = 7    # blob C different color
    comps = components(f, bg=0)
    assert len(comps) == 3
    assert sum(c.color == 2 for c in comps) == 2


def test_duplicate_color_instances_are_probed_when_budget_allows() -> None:
    # display: large color-4 blob; answer button: small second color-4 blob
    comps = [
        Component(2, 2, 4, 100),   # display, big
        Component(2, 10, 6, 90),   # another display color
        Component(58, 20, 4, 20),  # duplicate-color live button (smaller)
    ]
    picked = select_click_targets(comps, budget=3)
    assert Component(58, 20, 4, 20) in picked, "duplicate instance must be probed"
    # distinct colors come first, duplicates after
    assert picked.index(Component(58, 20, 4, 20)) == 2


def test_duplicate_reserve_survives_budget_cut() -> None:
    # 6 distinct colors + 2 duplicates, but budget 4: a distinct-only cut
    # would drop both duplicates — the reserve must keep them reachable.
    distinct = [Component(1, x, color, 50 - x) for x, color in
                zip(range(6), (1, 2, 3, 4, 5, 6))]
    dupes = [Component(58, 20, 1, 8), Component(58, 30, 2, 8)]
    picked = select_click_targets(distinct + dupes, budget=4, dupe_reserve=2)
    assert len(picked) == 4
    assert sum(c in dupes for c in picked) == 2, "reserved dupe slots missing"


def test_duplicate_ordering_is_spatially_diverse() -> None:
    # two dupes: one adjacent to its color's first instance (more of the same
    # display), one far away (sb26's answer strip) — the far one ranks first.
    comps = [
        Component(2, 2, 4, 100),
        Component(2, 4, 4, 30),    # near dupe
        Component(58, 40, 4, 20),  # far dupe (smaller!)
    ]
    picked = select_click_targets(comps, budget=3)
    assert picked[1] == Component(58, 40, 4, 20)
    assert picked[2] == Component(2, 4, 4, 30)


def test_selection_is_deterministic() -> None:
    comps = [Component(y, x, (y + x) % 5 + 1, 10 + x)
             for y in range(0, 60, 7) for x in range(0, 60, 11)]
    a = select_click_targets(comps, budget=10)
    b = select_click_targets(list(comps), budget=10)
    assert a == b
