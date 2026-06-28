"""IrreversibleImpactExplorer — symmetry-breaking action seeking (physics paradigm).

The bottleneck is LEVEL-1 trigger-identification on the wall games (no solved level to transfer
from). Physics inspiration: a level-up is an irreversible symmetry-breaking event — one action
flips a large, structured region and it STAYS flipped. Distractors (HUD counters, avatar motion,
animations) are either cosmetic or reversible. So: learn, per clicked-object COLOR, the mean
IRREVERSIBLE structural impact (object-multiset change that persists, i.e. is not undone by the
next step), and prioritise clicks on the highest-irreversible-impact colors. This differs from the
killed high-impact-click (raw cell-count, no persistence/irreversibility filter) and from
structural-novelty (frontier reorder): it scores ACTIONS by persistent structural effect, to find
the trigger on a fresh level.

Firewall: enable_irr=False -> byte-identical to SalienceExplorer (v6).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from . import perception as P
from .salience_explorer import MAX_TIER, SalienceExplorer


def _multiset(objs):
    return Counter((o.color, (o.size + 3) // 4) for o in objs)


def _ms_dist(a, b):
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


class IrreversibleImpactExplorer(SalienceExplorer):
    def __init__(self, *args, enable_irr: bool = True, impact_min: int = 6, **kwargs) -> None:
        self.enable_irr = bool(enable_irr)
        self.impact_min = int(impact_min)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        # per clicked color: [sum_irreversible_impact, count]
        self.color_impact: dict = defaultdict(lambda: [0.0, 0])
        self._irr_hist = []          # recent (clicked_color, prev_multiset) awaiting persistence check
        self._irr_prev_ms = None
        self._irr_prev_prev_ms = None
        self._irr_pending = None     # (color, ms_before) from last click, to score after we see 2 frames

    def _clicked_color(self, grid, x, y):
        for o in P.connected_components(grid, background=self.bg):
            if (y, x) in o.cells:
                return int(o.color)
        return None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_irr and self.bg is None:
            self.bg = P.detect_background(grid)
        if self.enable_irr and not gstate_terminal and not gstate_notplayed:
            cur_ms = _multiset(P.connected_components(grid, background=self.bg))
            # score the click made TWO steps ago: impact = change from before-click that PERSISTED
            if self._irr_pending is not None and self._irr_prev_ms is not None:
                color, ms_before = self._irr_pending
                changed = _ms_dist(ms_before, self._irr_prev_ms)        # immediate change
                persisted = _ms_dist(ms_before, cur_ms)                 # still changed a step later
                irr = min(changed, persisted)                          # irreversible = changed AND stayed
                rec = self.color_impact[color]
                rec[0] += irr; rec[1] += 1
                self._irr_pending = None
            # record this step's click (if the action we're about to take is a click, set below)
            self._irr_prev_prev_ms = self._irr_prev_ms
            self._irr_prev_ms = cur_ms
            self._pending_ms_before = cur_ms
        action = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        if self.enable_irr and action and action[0] == "C":
            col = self._clicked_color(grid, action[1], action[2])
            if col is not None:
                self._irr_pending = (col, getattr(self, "_pending_ms_before", _multiset(
                    P.connected_components(grid, background=self.bg))))
        return action

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_irr or not self.color_impact:
            return cands
        # colors with high mean irreversible impact
        means = {c: (s / n) for c, (s, n) in self.color_impact.items() if n >= 2}
        hot = {c for c, m in means.items() if m >= self.impact_min}
        if not hot:
            return cands
        cell_color = {}
        for o in P.connected_components(grid, background=self.bg):
            for (rr, cc) in o.cells:
                cell_color[(rr, cc)] = int(o.color)
        out = []
        for (act, tier) in cands:
            if act[0] == "C" and cell_color.get((act[2], act[1])) in hot:
                out.append((act, 0))             # promote high-irreversible-impact colors
            else:
                out.append((act, tier))
        return out
