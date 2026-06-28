"""RelationalExplorer — abstract relational state key (exploratory).

Subclass of SalienceExplorer. The named dominant failure on ARC-AGI-3 is "True Local
Effect, False World Model": agents track exact pixels but never lift them into structure.
The exact object_state_key (color + precise bbox + size) makes every small translation a
NEW state, exploding the graph and forcing re-exploration. RelationalExplorer keys on a
COARSER, more abstract description — object (color, size-bucket, quantized centroid) — which
collapses translation-equivalent states, shrinks the graph, and lets the explorer revisit
structurally-equivalent situations instead of re-discovering them.

Risk: too coarse merges genuinely-distinct states (breaks the determinism the graph assumes).
rel_quant controls granularity; measured on TUNE/HOLDOUT.

Firewall: enable_relational=False -> byte-identical action trace to SalienceExplorer (v6).
"""

from __future__ import annotations

from . import perception as P
from .salience_explorer import SalienceExplorer
from .transfer_explorer import _size_bucket


class RelationalExplorer(SalienceExplorer):
    def __init__(self, *args, enable_relational: bool = True, rel_quant: int = 4,
                 **kwargs) -> None:
        self.enable_relational = bool(enable_relational)
        self.rel_quant = max(1, int(rel_quant))
        super().__init__(*args, **kwargs)

    def _key(self, grid):
        if not self.enable_relational:
            return super()._key(grid)
        # replicate base masking (volatile + dynamic border), then build an abstract key
        m = self.vt.mask()
        bm = self._border_mask()
        if bm is not None:
            m = m | bm
        if m.any():
            grid = grid.copy()
            grid[m] = self.bg if self.bg is not None else 0
        q = self.rel_quant
        parts = []
        for o in P.connected_components(grid, background=self.bg):
            cr, cc = o.centroid
            parts.append((int(o.color), _size_bucket(o.size), int(cr) // q, int(cc) // q))
        parts.sort()
        return repr(("R", tuple(parts))).encode()
