"""Option 2: perception-augmented transfer (user-directed 2026-06-22).

Pair the validated scene-graph PERCEPTION layer with the banked TransferExplorer (v13=0.33),
additively and coverage-safely: when the scene graph confidently identifies target candidates
(framed objects — e.g. ls20 maroon-in-gray), ADD those cells as tier-0 click candidates. This
augments the candidate SET; it does not reorder existing frontiers (frontier order stays sacred).

Firewall: enable_percept=False, OR no clicks available, OR no confident targets -> byte-identical to
TransferExplorer (== v13). On click games with perceived targets it tries those cells early (bounded
extra cost if perception is wrong); on movement-only games (no action 6) it is byte-identical.
"""

from __future__ import annotations

from . import perception as P
from . import scene_graph as SG
from .transfer_explorer import TransferExplorer


class PerceptionTransferExplorer(TransferExplorer):
    def __init__(self, *args, enable_percept: bool = True, max_targets: int = 8, **kwargs) -> None:
        self.enable_percept = bool(enable_percept)
        self.max_percept_targets = int(max_targets)
        super().__init__(*args, **kwargs)

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_percept or 6 not in available:
            return cands
        try:
            scene = SG.extract(grid, self.bg)
        except Exception:  # never break the agent on a perception hiccup
            return cands
        tgts = scene.get("target_candidates", [])
        if not tgts:
            return cands
        existing = {a for a, _t in cands}
        extra = []
        for t in tgts[: self.max_percept_targets]:
            r, c = int(round(t["centroid"][0])), int(round(t["centroid"][1]))
            tok = ("C", c, r)  # ("C", x=col, y=row)
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and tok not in existing:
                extra.append((tok, 0))   # tier 0: try perceived targets early (additive)
        return cands + extra
