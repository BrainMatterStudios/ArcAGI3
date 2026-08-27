"""Wavefront HUD Mask Detector & 2D Dynamic Status-Bar Filter.

Detects status bars, budget counters, score displays, and edge-hugging progress indicators
that tick across actions in ARC-AGI-3 games. Masking these volatile cells prevents false state graph
explosion and eliminates spurious diffs in full-frame equality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

GRID_SIZE = 64


@dataclass
class WavefrontHUDMask:
    """Wavefront-based HUD mask tracking cell volatility across episodes."""

    threshold: float = 0.85
    min_steps: int = 5
    edge_band_width: int = 4

    def __post_init__(self) -> None:
        self.shape: tuple[int, int] = (GRID_SIZE, GRID_SIZE)
        self.change_counts = np.zeros(self.shape, dtype=np.int32)
        self.total_steps = 0
        self._prev_grid: np.ndarray | None = None

    def update(self, grid: np.ndarray) -> None:
        """Record frame update to track cell volatility."""
        grid_arr = np.asarray(grid, dtype=np.int8)
        if grid_arr.ndim == 3:
            grid_arr = grid_arr[-1]
        
        if self._prev_grid is not None and self._prev_grid.shape == grid_arr.shape:
            changed = (grid_arr != self._prev_grid).astype(np.int32)
            self.change_counts += changed
            self.total_steps += 1
            
        self._prev_grid = grid_arr.copy()

    def get_mask(self) -> np.ndarray:
        """Return boolean mask of volatile HUD cells (True = mask out)."""
        if self.total_steps < self.min_steps:
            return np.zeros(self.shape, dtype=bool)

        freq = self.change_counts / max(1, self.total_steps)
        volatile_mask = freq >= self.threshold

        # Detect edge-hugging status bar bands (horizontal or vertical bars on border)
        edge_mask = np.zeros(self.shape, dtype=bool)
        h, w = self.shape
        b = self.edge_band_width

        # Check top/bottom horizontal bands
        for r in range(min(b, h)):
            if np.mean(freq[r, :]) >= 0.5:
                edge_mask[r, :] = True
        for r in range(max(0, h - b), h):
            if np.mean(freq[r, :]) >= 0.5:
                edge_mask[r, :] = True

        # Check left/right vertical bands
        for c in range(min(b, w)):
            if np.mean(freq[:, c]) >= 0.5:
                edge_mask[:, c] = True
        for c in range(max(0, w - b), w):
            if np.mean(freq[:, c]) >= 0.5:
                edge_mask[:, c] = True

        return volatile_mask | edge_mask

    def apply_mask(self, grid: np.ndarray, fill_value: int = -1) -> np.ndarray:
        """Apply HUD mask to zero out volatile status bar cells."""
        mask = self.get_mask()
        masked_grid = np.asarray(grid, dtype=np.int8).copy()
        if masked_grid.ndim == 3:
            masked_grid = masked_grid[-1]
        masked_grid[mask] = fill_value
        return masked_grid
