"""Engine-driven fixtures for the LIVE dynamic HUD detector (HudMaskTracker).

Encodes the MEASURED behavior (2026-08-08) of the tracker on real offline-engine
frame sequences for the 6 HUD clock bars a fresh audit found missing from the
static map (ar25 col63, ft09 row63, g50t row63, sc25 cols62-63, su15 row63,
tn36 row1) plus 5 regression games from the validated map (cd82, sk48, ls20,
r11l, m0r0).

Frames are NOT generated here: they were produced once by playing ~120 scripted
actions per game (arrows/A5 + effective clicks) in the OFFLINE engine and cached
as .npz (frames + per-step is_reset/levels_completed/state flags). Cache path
(session scratchpad, intentionally NOT committed):

  /private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/\
de216582-726c-415f-9dd5-71c05fb4d2c3/scratchpad/hud_fixtures/

Regenerate with harness.py in that directory:
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python .../hud_fixtures/harness.py

Tests SKIP when the cache is absent so CI without the scratchpad stays green.

Measured verdicts encoded below:
  * All 6 new regions ARE detected (masked fraction of ticking cells = 1.0),
    converging at fed-frame 58-76 (= MIN_SEGMENTS x SEGMENT_ROTATE_STEPS
    virtual-rotation windows: 24+24+10 for per-action ticks; slower for
    every-2nd-action bars).
  * g50t's audited region "row 63 cols 32-63" is really the FULL row 63: over
    120 actions the bar drains right-to-left past col 32 down to col 4. Only
    cols 0-3 never tick and are covered by documented full-line rasterization.
  * ZERO masked cells outside the (full-line) HUD regions on all 11 games.
  * REGRESSION (xfail, strict): ls20's two-row bar (rows 61-62) confirms at 58
    but the bar REFILLS mid-level (fed-frame 86: cols 13-54 repaint at once, no
    RESET / level / state boundary), the refilled cells hit nchg>=2 inside the
    live window, and the unmask guard permanently drops BOTH rows
    (dead_lines). Final mask = 0 cells. A future tracker fix that survives
    in-level bar refills will flip the xfail to XPASS and force its removal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from duck_patches import HudMaskTracker  # noqa: E402

CACHE = Path(
    "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
    "de216582-726c-415f-9dd5-71c05fb4d2c3/scratchpad/hud_fixtures"
)

# Full-line HUD regions per game. The tracker rasterizes confirmed strips to the
# full line by design, so the allowed area is the full row/col through each bar.
# g50t: the audit said cols 32-63, but the measured drain reaches col 4 — the
# bar IS the full row (see module docstring).
ALLOWED_LINES: dict[str, list[tuple[str, int]]] = {
    # 6 newly audited clock bars
    "ar25": [("col", 63)],
    "ft09": [("row", 63)],
    "g50t": [("row", 63)],
    "sc25": [("col", 62), ("col", 63)],
    "su15": [("row", 63)],
    "tn36": [("row", 1)],
    # regression set (validated static map)
    "cd82": [("row", 63)],
    "sk48": [("row", 53)],
    "ls20": [("row", 61), ("row", 62)],
    "r11l": [("col", 0)],
    "m0r0": [("row", 0), ("row", 63)],
}

NEW_GAMES = ["ar25", "ft09", "g50t", "sc25", "su15", "tn36"]
REGRESSION_GAMES = ["cd82", "sk48", "r11l", "m0r0"]  # ls20 handled separately

# Measured convergence (first fed frame with a non-empty mask) — deterministic
# on the cached sequences; bounded loosely so tracker-neutral refactors pass.
FIRST_MASK_BOUND = 80

# Measured HUD-only no-op detections (raw diff non-empty, masked diff empty,
# non-reset) over the full replay — lower bounds, deterministic on the cache.
# ft09/sc25/sk48 measured 0: every scripted action also moved content there.
MIN_HUD_ONLY = {
    "ar25": 13, "g50t": 18, "su15": 46, "tn36": 62,
    "cd82": 27, "r11l": 62, "m0r0": 6,
}


def _allowed_mask(stem: str) -> np.ndarray:
    m = np.zeros((64, 64), bool)
    for kind, idx in ALLOWED_LINES[stem]:
        if kind == "row":
            m[idx, :] = True
        else:
            m[:, idx] = True
    return m


def _load(stem: str):
    path = CACHE / f"{stem}.npz"
    if not path.exists():
        pytest.skip(f"fixture cache {path} not present (see module docstring)")
    d = np.load(path)
    return d["frames"], d["is_reset"], d["levels"], d["states"]


def _replay(stem: str):
    """Fresh tracker over the cached sequence. Returns (tracker, stats)."""
    frames, is_reset, levels, states = _load(stem)
    tracker = HudMaskTracker()
    tracker.seed(frames[0])
    first_mask = None
    hud_only = 0
    change_count = np.zeros(frames[0].shape, np.int64)
    mask_sizes = []
    for i in range(1, len(frames)):
        info = tracker.observe(
            frames[i],
            is_reset=bool(is_reset[i - 1]),
            levels_completed=int(levels[i - 1]),
            state_name=str(states[i - 1]),
        )
        if not is_reset[i - 1]:
            change_count += frames[i - 1] != frames[i]
            if info["raw_changed"] and not info["masked_changed"]:
                hud_only += 1
        mask = tracker.mask_array()
        size = int(mask.sum()) if mask is not None else 0
        mask_sizes.append(size)
        if first_mask is None and size:
            first_mask = i
    return tracker, {
        "first_mask": first_mask,
        "hud_only": hud_only,
        "change_count": change_count,
        "mask_sizes": mask_sizes,
    }


# --- the 6 new regions: tracker converges on every one ---------------------------


@pytest.mark.parametrize("stem", NEW_GAMES)
def test_new_region_is_fully_masked(stem: str):
    """Every region cell that ever ticked must be masked at end of replay."""
    tracker, stats = _replay(stem)
    mask = tracker.mask_array()
    assert mask is not None and mask.any(), f"{stem}: nothing masked"
    ticking = (stats["change_count"] > 0) & _allowed_mask(stem)
    assert ticking.any(), f"{stem}: fixture never ticked the region (bad cache?)"
    unmasked = ticking & ~mask
    assert not unmasked.any(), (
        f"{stem}: {int(unmasked.sum())} ticking HUD cells left unmasked: "
        f"{np.argwhere(unmasked)[:10].tolist()}"
    )


@pytest.mark.parametrize("stem", NEW_GAMES)
def test_new_region_converges_within_rotation_windows(stem: str):
    _tracker, stats = _replay(stem)
    assert stats["first_mask"] is not None and stats["first_mask"] <= FIRST_MASK_BOUND, (
        f"{stem}: first mask at {stats['first_mask']} (bound {FIRST_MASK_BOUND})"
    )


# --- regression: known games still detected --------------------------------------


@pytest.mark.parametrize("stem", REGRESSION_GAMES)
def test_known_region_still_detected(stem: str):
    tracker, stats = _replay(stem)
    mask = tracker.mask_array()
    assert mask is not None and mask.any(), f"{stem}: known HUD no longer detected"
    ticking = (stats["change_count"] > 0) & _allowed_mask(stem)
    unmasked = ticking & ~mask
    assert not unmasked.any(), (
        f"{stem}: {int(unmasked.sum())} ticking HUD cells left unmasked"
    )
    assert stats["first_mask"] <= FIRST_MASK_BOUND


def test_m0r0_masks_both_edge_bars():
    tracker, _stats = _replay("m0r0")
    mask = tracker.mask_array()
    assert mask[0].all() and mask[63].all(), "m0r0 must mask BOTH edge rows fully"


# --- false positives -------------------------------------------------------------


@pytest.mark.parametrize("stem", NEW_GAMES + REGRESSION_GAMES + ["ls20"])
def test_no_cells_masked_outside_hud_lines(stem: str):
    tracker, _stats = _replay(stem)
    mask = tracker.mask_array()
    if mask is None or not mask.any():
        return  # empty mask is trivially FP-free (ls20 ends empty — see below)
    outside = mask & ~_allowed_mask(stem)
    assert not outside.any(), (
        f"{stem}: {int(outside.sum())} cells masked outside the HUD lines: "
        f"{np.argwhere(outside)[:10].tolist()}"
    )


# --- end-to-end value: pure HUD ticks read as no-ops -----------------------------


@pytest.mark.parametrize("stem", sorted(MIN_HUD_ONLY))
def test_hud_only_noops_are_detected(stem: str):
    _tracker, stats = _replay(stem)
    assert stats["hud_only"] >= MIN_HUD_ONLY[stem], (
        f"{stem}: expected >= {MIN_HUD_ONLY[stem]} HUD-only no-op detections, "
        f"got {stats['hud_only']}"
    )


# --- ls20: the one measured miss -------------------------------------------------


def test_ls20_confirms_before_the_refill():
    """rows 61-62 confirm at fed-frame ~58, before the mid-level refill at ~86."""
    tracker, stats = _replay("ls20")
    sizes = stats["mask_sizes"]
    assert stats["first_mask"] is not None and stats["first_mask"] <= 60, (
        "ls20 bar must CONFIRM before the refill"
    )
    assert max(sizes) >= 128, "both bar rows were masked while confirmed"


def test_ls20_mask_survives_in_level_bar_refill():
    """FIXED 2026-08-08: the bar refills mid-level at fed-frame ~86 with no
    RESET/level/state boundary. The refill-aware guard shield
    (`_clear_refilled_lines`: single-frame span-covering restoration to the
    bar's own from_color) now treats it as a per-line segment restart instead
    of letting the unmask guard permanently kill both rows. Was a strict xfail
    documenting the miss; flipped by the fix."""
    tracker, stats = _replay("ls20")
    mask = tracker.mask_array()
    assert mask is not None and mask.any(), "ls20 mask must survive the refill"
    ticking = (stats["change_count"] > 0) & _allowed_mask("ls20")
    assert not (ticking & ~mask).any()
    dead_lines = {(orient, line) for orient, line, *_ in tracker._dead_lines}
    assert dead_lines == set(), f"guard casualties should be empty: {sorted(dead_lines)}"
