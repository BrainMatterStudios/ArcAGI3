"""MechanicCertificateStore — the proof layer of the causal-probe architecture.

Maintains, PER GAME, empirically-observed causal facts about action classes — never goal guesses.
A certificate is only ever asserted from a transition actually witnessed in THIS game (no
prediction, no cross-game transfer of the certificate itself). The store is what the
coverage-preserving scheduler consults before allowing any exploit: an action class may only be
exploited once it carries a proof certificate here.

Action classes (the unit a certificate is keyed on):
  - ("S", aid)        a simple action id (1-5, 7)
  - ("C", color)      clicking a cell of a given color (color-signature, matching transfer)

Effect certificates accumulate counts over witnessed transitions:
  tried, changed (frame delta != 0), cells (total changed-cell mass), levelup, reversed
  (a later inverse/no-op returned the prior frame — a weak reversibility signal).

Derived, conservative predicates (require a minimum sample so a single fluke can't certify):
  - is_proven_noop:   tried>=k and changed==0           (sb26 ACTION1; safe to deprioritise)
  - is_level_trigger: levelup>=1                         (the certified exploit target)
  - is_structural:    mean changed-cell mass is large    (collect/transform/irreversible proxy —
                      the progress signal for navigation-goal games, used as a COVERAGE target,
                      NOT a frontier reranker)

This module changes NO behavior on its own; it only records. The scheduler/explorer consults it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Cert:
    tried: int = 0
    changed: int = 0
    cells: int = 0
    levelup: int = 0

    @property
    def change_rate(self) -> float:
        return self.changed / self.tried if self.tried else 0.0

    @property
    def mean_cells(self) -> float:
        return self.cells / self.changed if self.changed else 0.0


class MechanicCertificateStore:
    def __init__(self, noop_min_samples: int = 4, structural_min_cells: float = 40.0) -> None:
        self.noop_min_samples = int(noop_min_samples)
        self.structural_min_cells = float(structural_min_cells)
        self.certs: dict[tuple, _Cert] = {}

    @staticmethod
    def action_class(action_token, prev_grid) -> tuple | None:
        """Map a decision token to its certificate key. Click -> color signature."""
        if action_token[0] == "S":
            return ("S", int(action_token[1]))
        if action_token[0] == "C":
            col, row = int(action_token[1]), int(action_token[2])
            if 0 <= row < prev_grid.shape[0] and 0 <= col < prev_grid.shape[1]:
                return ("C", int(prev_grid[row, col]))
        return None

    def observe(self, action_token, prev_grid, new_grid, level_up: bool) -> tuple | None:
        """Record one witnessed transition. Returns the action class touched (or None)."""
        key = self.action_class(action_token, prev_grid)
        if key is None:
            return None
        c = self.certs.setdefault(key, _Cert())
        c.tried += 1
        if new_grid.shape == prev_grid.shape:
            cells = int((new_grid != prev_grid).sum())
        else:
            cells = -1
        if cells != 0:
            c.changed += 1
            if cells > 0:
                c.cells += cells
        if level_up:
            c.levelup += 1
        return key

    # ---- conservative predicates (proof, not guess) ----
    def is_proven_noop(self, action_class) -> bool:
        c = self.certs.get(action_class)
        return bool(c and c.tried >= self.noop_min_samples and c.changed == 0)

    def is_level_trigger(self, action_class) -> bool:
        c = self.certs.get(action_class)
        return bool(c and c.levelup >= 1)

    def is_structural(self, action_class) -> bool:
        c = self.certs.get(action_class)
        return bool(c and c.changed >= 2 and c.mean_cells >= self.structural_min_cells)

    def level_triggers(self) -> list[tuple]:
        return [k for k, c in self.certs.items() if c.levelup >= 1]

    def proven_noops(self) -> list[tuple]:
        return [k for k in self.certs if self.is_proven_noop(k)]

    def summary(self) -> dict:
        return {
            "n_classes": len(self.certs),
            "triggers": self.level_triggers(),
            "noops": self.proven_noops(),
        }
