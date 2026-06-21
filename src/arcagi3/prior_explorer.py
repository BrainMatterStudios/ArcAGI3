"""PriorExplorer — core-knowledge salience priors (no learning).

Subclass of SalienceExplorer. Humans solve unseen games efficiently because of innate
priors: a distinct/odd-one-out object is probably interactive; matching objects probably
pair; the dominant tiled color is probably floor/wall, not a target. We encode these as
soft, per-state click-priority adjustments computed from the object palette — generalizing
by construction (no per-game learning, no reward needed).

Relational palette prior, per state:
  - object color appears on exactly 1 object (odd-one-out)      -> promote to tier 0
  - object color appears on exactly 2 objects (matching pair)   -> promote to tier 0
  - object color is the most common (>= dominant_min objects)   -> demote (likely floor)

Firewall: enable_prior=False -> byte-identical action trace to SalienceExplorer (v6).
"""

from __future__ import annotations

from . import perception as P
from .salience_explorer import MAX_TIER, SalienceExplorer


class PriorExplorer(SalienceExplorer):
    def __init__(self, *args, enable_prior: bool = True, prior_demote: int = 3,
                 dominant_min: int = 4, **kwargs) -> None:
        self.enable_prior = bool(enable_prior)
        self.prior_demote = int(prior_demote)
        self.dominant_min = int(dominant_min)
        super().__init__(*args, **kwargs)

    def _cell_to_color(self, grid):
        m = {}
        counts: dict[int, int] = {}
        for o in P.connected_components(grid, background=self.bg):
            counts[o.color] = counts.get(o.color, 0) + 1
            for (rr, cc) in o.cells:
                m[(rr, cc)] = int(o.color)
        return m, counts

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_prior:
            return cands
        cell2color, counts = self._cell_to_color(grid)
        if not counts:
            return cands
        dominant = max(counts.values())
        out = []
        for (act, tier) in cands:
            if act[0] != "C":
                out.append((act, tier))
                continue
            color = cell2color.get((act[2], act[1]))
            if color is None:
                out.append((act, tier))
                continue
            n = counts.get(color, 99)
            if n <= 2:                                   # odd-one-out / matching pair
                out.append((act, 0))
            elif n == dominant and n >= self.dominant_min:  # dominant tiled color -> floor
                out.append((act, min(MAX_TIER, tier + self.prior_demote)))
            else:
                out.append((act, tier))
        return out
