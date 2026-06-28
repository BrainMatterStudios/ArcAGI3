"""TransferRelationalExplorer — combine the two tournament winners.

The @6000 tournament showed the two creative levers are COMPLEMENTARY:
  - RelationalExplorer's abstract state key reaches DEEPER levels on large-state walls
    (cracked sk48/ls20/sc25 that every prior session declared dead), but at high action
    cost (efficiency cratered).
  - TransferExplorer accelerates LATER levels once one is solved (lp85 L1->L5, vc33 L2 7x)
    but only fires on games that reach a 2nd level.

Composition hypothesis: the relational key gives transfer the multi-level reachability it
needs on the walls, and transfer cuts the action cost of the deeper levels relational
unlocks -> wall-cracking WITHOUT the efficiency crater.

Implemented by multiple inheritance so each lever stays a single-purpose unit:
  MRO = TransferRelationalExplorer -> TransferExplorer -> RelationalExplorer -> SalienceExplorer
TransferExplorer contributes decide()/_candidates() (signature learning + re-prioritisation);
RelationalExplorer contributes _key() (abstract state key); both cooperate via super().

Firewall: enable_transfer=False AND enable_relational=False -> byte-identical to v6.
"""

from __future__ import annotations

from .relational_explorer import RelationalExplorer
from .transfer_explorer import TransferExplorer


class TransferRelationalExplorer(TransferExplorer, RelationalExplorer):
    pass
