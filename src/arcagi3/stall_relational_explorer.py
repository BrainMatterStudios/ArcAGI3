"""StallRelationalExplorer — stall-triggered relational re-keying (additive unlock).

RelationalExplorer's abstract state key (color, size-bucket, quantized centroid) CRACKED wall
games salience can't touch (sc25 L0->L2, sk48 0->L1, ls20 0->L1) but cratered efficiency on the
games that don't need abstraction -> net-negative, killed. The untried fix: make it ADDITIVE by
only switching to the relational key once the exact-key explorer has STALLED (no level-up for
stall_trigger actions). Games salience already solves never stall -> stay exact -> no crater;
genuinely stuck wall games stall -> switch to the relational key + re-explore -> unlock.

On switch we clear the (exact-keyed) graph but keep the volatility tracker / background / RNG,
then continue building a fresh graph under the relational key. enable_stallrel=False (or
stall_trigger<=0 with mode forced exact) -> byte-identical to SalienceExplorer (v6).
"""

from __future__ import annotations

from . import perception as P
from .salience_explorer import SalienceExplorer
from .transfer_explorer import _size_bucket


class StallRelationalExplorer(SalienceExplorer):
    def __init__(self, *args, enable_stallrel: bool = True, stall_trigger: int = 1500,
                 rel_quant: int = 4, stall_mode: str = "states", **kwargs) -> None:
        self.enable_stallrel = bool(enable_stallrel)
        self.stall_trigger = int(stall_trigger)
        self.rel_quant = max(1, int(rel_quant))
        # "states": switch when no NEW states discovered for stall_trigger actions (saturation =
        # truly stuck; a slow-but-progressing game like tu93 keeps discovering states so it never
        # triggers). "level": switch on no level-up for stall_trigger actions (misfires on slow games).
        self.stall_mode = stall_mode
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.rel_mode = False
        self._sr_since = 0
        self._sr_levels = 0
        self._sr_last_nodes = 0
        # bisim trigger: track whether the relational key is dynamics-PRESERVING for this game
        self._bisim_edges: dict = {}   # (rel_key, action) -> first rel-successor seen
        self._bisim_ok = 0             # transitions consistent with the recorded rel-successor
        self._bisim_bad = 0            # transitions where the same (rel,action) led elsewhere
        self._sr_prev_grid = None

    def _rel_key(self, grid):
        """The abstract relational key (translation-tolerant), independent of rel_mode."""
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

    def _key(self, grid):
        if not (self.enable_stallrel and self.rel_mode):
            return super()._key(grid)
        return self._rel_key(grid)

    def _switch_to_relational(self):
        # clear the exact-keyed graph; keep vt / bg / rng / counters
        self.rel_mode = True
        self.nodes = {}
        self.root_key = None
        self.active_group = 0
        self.plan = []
        self._expect = None
        self.prev_key = None
        self.prev_action = None
        self.pending = {}

    def _bisim_consistency(self):
        tot = self._bisim_ok + self._bisim_bad
        return (self._bisim_ok / tot) if tot else 0.0

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_stallrel and self.stall_trigger > 0 and not self.rel_mode:
            playing = not gstate_terminal and not gstate_notplayed
            # bisim bookkeeping: does the relational key merge states that behave the SAME?
            if (self.stall_mode == "bisim" and playing and self.prev_action is not None
                    and self.prev_action[0] in ("S", "C") and self._sr_prev_grid is not None):
                rp = self._rel_key(self._sr_prev_grid)
                rc = self._rel_key(grid)
                k = (rp, self.prev_action)
                if k in self._bisim_edges:
                    if self._bisim_edges[k] == rc:
                        self._bisim_ok += 1
                    else:
                        self._bisim_bad += 1
                else:
                    self._bisim_edges[k] = rc
            # stall counter
            if levels > self._sr_levels:
                self._sr_levels = levels
                self._sr_since = 0
            elif self.stall_mode == "states" and len(self.nodes) > self._sr_last_nodes:
                self._sr_last_nodes = len(self.nodes)
                self._sr_since = 0
            if playing:
                self._sr_since += 1
            # trigger: stalled, and (for bisim) the relational abstraction is dynamics-preserving
            ready = self._sr_since >= self.stall_trigger and playing
            if self.stall_mode == "bisim":
                ready = ready and (self._bisim_ok + self._bisim_bad) >= 40 \
                    and self._bisim_consistency() >= 0.9
            if ready:
                self._switch_to_relational()
            self._sr_prev_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
