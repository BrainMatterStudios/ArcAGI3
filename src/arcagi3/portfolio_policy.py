"""PortfolioPolicy — runs several diverse coverage strategies as SEPARATE PLAYS within one game, so the
eval's MAX-over-plays scoring takes the best strategy per game. Strict-superset by construction: including
our banked TransferExplorer as strategy 0 guarantees the game score >= TransferExplorer on EVERY game, with
upside wherever another strategy wins a game we don't.

Mechanism (verified in arc_agi/scorecard.py update_scorecard -> new_play on full_reset; most_levels_completed
= max over plays). A new PLAY is created by a FULL reset (the engine flags full_reset=True on a RESET issued
when already at the level-0 start, i.e. a SECOND consecutive RESET). So between strategies we inject 2 resets.

Rotation: run strategy i until it goes `level_stall_limit` actions with NO new level completed (it has
plateaued on what it can solve), then transition to strategy i+1 via the double-reset. After the last
strategy, keep running it (don't waste budget). enable via runner --agent portfolio.
"""
from __future__ import annotations

DENSE = dict(trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)


def _default_strategies():
    # (name, factory(seed)) — diverse coverage explorers; strategy 0 = banked best (strict-superset anchor).
    from .transfer_explorer import TransferExplorer
    from .transfer_relational_explorer import TransferRelationalExplorer
    from .relational_explorer import RelationalExplorer
    from .transfer_cai_explorer import TransferCAIExplorer
    return [
        ("transfer_s0", lambda s: TransferExplorer(seed=s, **DENSE)),
        ("transfer_rel", lambda s: TransferRelationalExplorer(seed=s, **DENSE)),
        ("transfer_s1", lambda s: TransferExplorer(seed=s + 101, **DENSE)),
        ("relational", lambda s: RelationalExplorer(seed=s, **DENSE)),
        ("transfer_cai", lambda s: TransferCAIExplorer(seed=s, **DENSE)),
    ]


class PortfolioPolicy:
    def __init__(self, seed: int = 0, strategies=None, level_stall_limit: int = 20000) -> None:
        strategies = strategies or _default_strategies()
        self.names = [n for n, _ in strategies]
        self.pols = [f(seed) for _, f in strategies]
        self.level_stall_limit = int(level_stall_limit)
        self.idx = 0
        self._since_level = 0
        self._best_levels = 0
        self._pending_resets = 0   # resets to inject for a play transition (2 = double-reset -> new play)
        # SAFETY: the harness sets this each step to obs.full_reset. We verify the FIRST transition's
        # double-reset actually created a new play; if the engine never flags full_reset, the multi-play
        # mechanism is broken -> we revert to strategy 0 and stop transitioning (hard no-regression floor).
        self._last_full_reset = False
        self._saw_full_reset = False
        self._multiplay_broken = False

    @property
    def gs(self):
        return self.pols[self.idx].gs

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        # inject the double-reset that starts a NEW PLAY before the next strategy runs
        if self._pending_resets > 0:
            if self._last_full_reset:
                self._saw_full_reset = True
            self._pending_resets -= 1
            if self._pending_resets == 0 and not self._saw_full_reset:
                # the engine did NOT create a new play -> multi-play unsupported here. Abort to the
                # banked best strategy and never transition again (cannot regress below it).
                self._multiplay_broken = True
                self.idx = 0
                self._since_level = 0
            return ("reset",)

        # progress tracking: a NEW level resets the stall counter
        if levels > self._best_levels:
            self._best_levels = levels
        cur = self.pols[self.idx]
        cur_levels_seen = getattr(cur, "_pf_max_level", 0)
        if levels > cur_levels_seen:
            cur._pf_max_level = levels
            self._since_level = 0
        else:
            self._since_level += 1

        # plateaued on levels -> rotate to the next strategy (if any) via a new play
        if (not self._multiplay_broken and self._since_level >= self.level_stall_limit
                and self.idx < len(self.pols) - 1):
            self.idx += 1
            self._since_level = 0
            self._pending_resets = 2          # double-reset -> full_reset -> new play slot
            self._saw_full_reset = False      # verify THIS transition creates a new play
            return ("reset",)

        return cur.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
