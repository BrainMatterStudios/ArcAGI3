"""LearnedExplorer — scaffold for a per-game online-LEARNED agent with a T4 FAIL-SAFE FIREWALL.

The leaders' paradigm (StochasticGoose/Tufa: a small per-game model trained ONLINE, reset per level;
Blind-Squirrel distance-to-reward value net) needs a GPU at eval. The Kaggle eval gives a usable Tesla
T4 (cap 7.5) ONLY if the kernel is pushed with --accelerator NvidiaTeslaT4; the DEFAULT P100 (cap 6.0)
is INCOMPATIBLE with torch 2.10 (min cap 7.0) -> CUDA ops FAIL despite cuda.is_available()==True. So the
firewall must verify a tiny CUDA op ACTUALLY succeeds, not just that cuda is "available". See the memory
arcagi3-eval-image-facts.

This base wraps the banked TransferExplorer: it is BYTE-IDENTICAL to it unless ALL of (a) enable_learn,
(b) a learner is attached, and (c) a real CUDA op succeeds hold -- so the banked 0.33 floor is never at
risk (no GPU / no learner / disabled -> pure-numpy TransferExplorer). When the learner IS active it
proposes the action and we fall back to the normal salience choice whenever it abstains (returns None),
so coverage is preserved. Per-game and per-level reset hooks mirror the recipe (model+buffer reset
between levels). This is the foundation for a TTT/RL learner now AND for adopting+hardening the
June-30 open-sourced winner inside our no-regression firewall.
"""
from __future__ import annotations

from .transfer_explorer import TransferExplorer


class Learner:
    """Interface a learned component implements. Default = inert (always abstains) -> the scaffold stays
    byte-identical to TransferExplorer. A real learner (TTT/RL net, or the adopted winner) overrides."""

    def reset_game(self) -> None: ...
    def reset_level(self) -> None: ...
    def observe(self, key, action, next_key, reward) -> None: ...

    def act(self, grid, node, key):
        """Return a chosen action token, or None to defer to the base explorer (abstain)."""
        return None


class LearnedExplorer(TransferExplorer):
    def __init__(self, *args, enable_learn: bool = False, learner: Learner | None = None,
                 require_gpu: bool = True, **kwargs) -> None:
        self.enable_learn = bool(enable_learn)
        self.learner = learner
        self.require_gpu = bool(require_gpu)
        self._gpu_checked = False
        self._gpu_usable = False
        super().__init__(*args, **kwargs)

    # --- T4 fail-safe: a tiny CUDA op must actually SUCCEED (P100 cap-6.0 fails on torch 2.10) -------
    def _gpu_ok(self) -> bool:
        if self._gpu_checked:
            return self._gpu_usable
        self._gpu_checked = True
        if not self.require_gpu:
            self._gpu_usable = True
            return True
        try:
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                self._gpu_usable = False
            else:
                x = torch.ones(8, device="cuda")
                self._gpu_usable = float((x + x).sum().item()) == 16.0   # real op, not just is_available
        except Exception:  # noqa: BLE001  (any torch/CUDA failure -> fall back to numpy)
            self._gpu_usable = False
        return self._gpu_usable

    def _active(self) -> bool:
        return self.enable_learn and self.learner is not None and self._gpu_ok()

    def reset_all(self):
        super().reset_all()
        self._learn_levels = 0
        self._cur_grid_learn = None

    # --- hooks: per-game / per-level reset, online observe, learned action with abstain-fallback ----
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self._active():
            if gstate_notplayed:
                self.learner.reset_game()
            elif levels > self._learn_levels:
                self.learner.reset_level()
            self._learn_levels = levels
            self._cur_grid_learn = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _choose(self, cur):
        if self._active():
            a = self.learner.act(self._cur_grid_learn, self.nodes.get(cur), cur)
            if a is not None:
                return a
        return super()._choose(cur)

    def _record(self, key, action, next_key, reward, cands, terminal):
        super()._record(key, action, next_key, reward, cands, terminal)
        if self._active():
            self.learner.observe(key, action, next_key, reward)
