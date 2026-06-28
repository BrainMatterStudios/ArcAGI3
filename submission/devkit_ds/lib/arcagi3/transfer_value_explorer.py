"""TransferSpatialValueExplorer — compose the banked 0.33 lever with learned exploitation.

The banked best (v13 = 0.33) is TransferExplorer: on a level-up it learns the rewarding action's
color signature and PROMOTES matching candidates on later levels (reorders tiers WITH a real reward
signal; full coverage preserved). The SpatialValueExplorer adds the reproducible cluster's actual
machinery: a per-game CNN trained online on BFS distance-to-milestone labels that, once confident,
EXPLOITS toward the next milestone among the UNTRIED candidates.

They are ORTHOGONAL by method:
  - TransferExplorer overrides `_candidates` (tier re-prioritisation by reward signature).
  - SpatialValueExplorer overrides `_choose` (confident-exploit among untried candidates).
  - `decide`/`reset_all` are cooperative super() chains in both (disjoint attrs: _prev_grid /
    reward_* vs _svp_prev_grid / qnet / transitions).

So Transfer re-prioritises the frontier tiers, then the value-CNN — only when its value margin
clears conf_margin and the exploit budget (exploit_cap) is unspent — picks the highest-value
untried candidate; otherwise it falls back to the (transfer-reordered) salience search. Neither
reranks frontiers signal-free or prunes coverage (the operations that burned us), so the combo
keeps the 0.33 base AND adds exploitation where the net is confident.

MRO: TransferSpatialValue -> Transfer -> SpatialValue -> Salience. Firewall: enable_transfer=False
AND enable_value_cnn=False -> byte-identical to v6. Torch fail-safe (GPU-only eval -> v6 + transfer).
"""

from __future__ import annotations

from .spatial_value_explorer import SpatialValueExplorer
from .transfer_explorer import TransferExplorer


class TransferSpatialValueExplorer(TransferExplorer, SpatialValueExplorer):
    pass
