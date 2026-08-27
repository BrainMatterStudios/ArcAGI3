"""Unit tests for Wavefront HUD Mask Detector."""

import numpy as np
import pytest
from arcagi3.hud_mask import WavefrontHUDMask


def test_hud_mask_volatility_detection():
    tracker = WavefrontHUDMask(threshold=0.8, min_steps=4)
    base_grid = np.zeros((64, 64), dtype=np.int8)

    for step in range(6):
        grid = base_grid.copy()
        grid[0, :10] = step % 5
        tracker.update(grid)

    mask = tracker.get_mask()
    assert mask[0, 0] == True, "Row 0 cell 0 should be flagged as volatile HUD"
    assert mask[30, 30] == False, "Interior grid cell should not be volatile"


def test_hud_mask_apply_mask():
    tracker = WavefrontHUDMask(threshold=0.8, min_steps=4)
    base_grid = np.full((64, 64), 5, dtype=np.int8)

    for step in range(6):
        grid = base_grid.copy()
        grid[0, :] = step % 3
        tracker.update(grid)

    masked_grid = tracker.apply_mask(base_grid, fill_value=-1)
    assert (masked_grid[0, :] == -1).all(), "Top status bar row should be masked out with fill_value"
