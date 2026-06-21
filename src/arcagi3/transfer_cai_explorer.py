"""TransferCAIExplorer — compose the two strictly-additive efficiency levers.

Both validated as strictly additive (same levels, +efficiency, zero HOLDOUT regression) and they
are ORTHOGONAL:
  - TransferExplorer: on a level-up, learn the rewarding action's color signature and PROMOTE
    matching clicks on later levels (helps multi-level click games reach deep levels faster).
  - CAIPruneExplorer: PRUNE clicks on colors proven no-op within a level (CAI=0), reset per
    level-up (helps no-op-click-heavy games stop wasting the budget).

Neither reranks frontiers (the operation that corrupts coverage), so they compose safely. MRO:
TransferCAI -> Transfer -> CAIPrune -> Salience. CAIPrune drops dead clicks; Transfer then
re-prioritises the survivors; both learn from their own decide hooks. enable_transfer=False AND
enable_cai=False -> byte-identical to v6 (firewall).
"""

from __future__ import annotations

from .cai_prune_explorer import CAIPruneExplorer
from .transfer_explorer import TransferExplorer


class TransferCAIExplorer(TransferExplorer, CAIPruneExplorer):
    pass
