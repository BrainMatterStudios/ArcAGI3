"""WinningExplorer — TransferExplorer + optional executable mechanic model-search (winning branch).

The "winning" bet (per the mechanic-model-search mission): discover compact executable transition
rules online, plan through them, and fall back SAFELY to TransferExplorer. This module is the
Phase-0 frozen-baseline wrapper. Its firewall is absolute:

    enable_model_search=False  -> byte-identical action trace to TransferExplorer
    no confident model plan    -> byte-identical action trace to TransferExplorer

The model-search hook (`_confident_plan_action`) is stubbed to return None until the Phase-1
Mechanic Coverage Benchmark (tools/mechanic_coverage.py) clears its kill gate. So today
WinningExplorer is provably == TransferExplorer; the benchmark decides whether the engine behind
the hook is worth building (Phases 2-6). v13 TransferExplorer (public 0.33) is never modified.
"""
from __future__ import annotations

import os

from arcagi3.transfer_explorer import TransferExplorer


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


class WinningExplorer(TransferExplorer):
    """TransferExplorer subclass that may override with a confident model-search plan, else defers
    byte-for-byte to TransferExplorer. Honors env ENABLE_MODEL_SEARCH (default on)."""

    def __init__(self, *args, enable_model_search: bool | None = None, **kwargs) -> None:
        self.enable_model_search = (
            _env_flag("ENABLE_MODEL_SEARCH", True) if enable_model_search is None
            else bool(enable_model_search))
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        # Counters (mission Phase 6): exposed for the evaluation protocol.
        self._model_fires = 0
        self._fallback_actions = 0
        self._model_plan_aborts = 0
        # Model-search state is attached here in Phase 2+; None keeps the firewall closed.
        self._search = None

    def _confident_plan_action(self, grid, available):
        """Return the next action from a high-confidence model-beam plan, or None to defer to
        TransferExplorer. STUB until the coverage gate passes — always None today, so the wrapper
        is byte-identical to TransferExplorer (the regression test enforces this)."""
        return None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_model_search and not gstate_terminal and not gstate_notplayed:
            token = self._confident_plan_action(grid, available)
            if token is not None:
                self._model_fires += 1
                return token
        self._fallback_actions += 1
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
