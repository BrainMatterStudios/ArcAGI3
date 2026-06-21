"""RelationalTransferExplorer — relational-schema trigger transfer (cognitive-science paradigm).

The proven wall: efficiency-to-first-trigger needs TRANSFERABLE knowledge of where the trigger
is. TransferExplorer transfers the trigger's SURFACE COLOR, which only generalizes when color is
stable across levels. Humans transfer the abstract RELATIONAL SCHEMA (Gentner structure-mapping):
"the trigger is the unique object / the rarest / the odd-one-out / the edge one / the one matching
another" — properties that survive surface (color/size) changes across a game's escalating levels.

On each level-up we record the COLOR-AGNOSTIC relational properties the trigger object had; we keep
the schema that recurs across level-ups (stable properties), and on later levels promote candidates
whose object satisfies that schema — even if its color is new. This is the surface->structure
generalization of the one lever that beat blind search, aimed at the games where color-transfer
fails because the trigger recolors between levels.

Firewall: enable_rtransfer=False -> byte-identical to SalienceExplorer (v6).
"""

from __future__ import annotations

from collections import Counter

from . import perception as P
from .salience_explorer import MAX_TIER, SalienceExplorer

GRID = 64


def _size_bucket(n: int) -> int:
    if n <= 4:
        return 0
    if n <= 16:
        return 1
    if n <= 64:
        return 2
    return 3


def _obj_props(o, objs, counts) -> set:
    """Color-AGNOSTIC relational properties of object o within the scene."""
    props = set()
    if counts[o.color] == 1:
        props.add("uniq_color")
    if counts[o.color] <= 2:
        props.add("rare_color")
    if counts[o.color] >= 3:
        props.add("common_color")
    sizes = [x.size for x in objs]
    if o.size == min(sizes):
        props.add("smallest")
    if o.size == max(sizes):
        props.add("largest")
    sb = _size_bucket(o.size)
    if sb == 0:
        props.add("tiny")
    # odd-one-out by shape: unique size-bucket among objects
    if sum(1 for x in objs if _size_bucket(x.size) == sb) == 1:
        props.add("uniq_size")
    r0, c0, r1, c1 = o.bbox
    if r0 <= 2 or c0 <= 2 or r1 >= GRID - 3 or c1 >= GRID - 3:
        props.add("edge")
    if (r0 <= 2 or r1 >= GRID - 3) and (c0 <= 2 or c1 >= GRID - 3):
        props.add("corner")
    return props


class RelationalTransferExplorer(SalienceExplorer):
    def __init__(self, *args, enable_rtransfer: bool = True, min_support: float = 0.5,
                 **kwargs) -> None:
        self.enable_rtransfer = bool(enable_rtransfer)
        self.min_support = float(min_support)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.reward_simple: set = set()
        self.trigger_prop_counts: Counter = Counter()
        self.n_triggers = 0
        self._prev_grid_rt = None

    def _schema(self) -> set:
        """Relational properties present in >= min_support of observed triggers (the stable schema)."""
        if self.n_triggers == 0:
            return set()
        thresh = self.min_support * self.n_triggers
        return {p for p, c in self.trigger_prop_counts.items() if c >= thresh}

    def _scene(self, grid):
        objs = P.connected_components(grid, background=self.bg)
        counts = Counter(o.color for o in objs)
        return objs, counts

    def _obj_at(self, grid, x, y):
        objs, counts = self._scene(grid)
        for o in objs:
            if (y, x) in o.cells:
                return o, objs, counts
        # fallback: bbox containment
        for o in objs:
            r0, c0, r1, c1 = o.bbox
            if r0 <= y <= r1 and c0 <= x <= c1:
                return o, objs, counts
        return None, objs, counts

    def _learn(self, grid, action):
        if action[0] == "S":
            self.reward_simple.add(action)
            return
        o, objs, counts = self._obj_at(grid, action[1], action[2])
        if o is None:
            return
        self.n_triggers += 1
        for p in _obj_props(o, objs, counts):
            self.trigger_prop_counts[p] += 1

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_rtransfer and self.prev_action is not None
                and not gstate_terminal and not gstate_notplayed
                and levels > self.prev_levels and self._prev_grid_rt is not None):
            self._learn(self._prev_grid_rt, self.prev_action)
        action = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._prev_grid_rt = grid
        return action

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        schema = self._schema()
        if not self.enable_rtransfer or (not schema and not self.reward_simple):
            return cands
        objs, counts = self._scene(grid)
        # map cell -> object props
        prop_by_cell = {}
        for o in objs:
            ps = _obj_props(o, objs, counts)
            for (rr, cc) in o.cells:
                prop_by_cell[(rr, cc)] = ps
        out = []
        for (act, tier) in cands:
            if act[0] == "S":
                out.append((act, 0 if act in self.reward_simple else tier))
                continue
            ps = prop_by_cell.get((act[2], act[1]))
            if schema and ps is not None and schema.issubset(ps):
                out.append((act, 0))                                  # matches the learned schema
            elif schema:
                out.append((act, min(MAX_TIER, tier + 2)))            # demote non-matching
            else:
                out.append((act, tier))
        return out
