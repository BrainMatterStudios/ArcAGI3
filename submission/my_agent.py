# =====================================================================
# ARC-AGI-3 submission agent (ARC Prize 2026) — fail-safe adapter.
#
# Wraps arcagi3.policy.HybridPolicy in the official Agent interface. The arcagi3 package is
# written to /kaggle/working/arcagi3 by the notebook (self-contained) and/or attached as the
# arcagi3-agent dataset; we add every plausible path to sys.path. If the package cannot be
# imported for ANY reason, we fall back to a built-in random policy so the agent ALWAYS acts
# (a scored submission beats an ERROR, and tells us import was the issue).
# =====================================================================
import os
import random
import sys
import time
import traceback

_CANDIDATES = [
    "/kaggle/working",                  # notebook writes arcagi3/ here (primary, self-contained)
    "/kaggle/working/ARC-AGI-3-Agents/agents/templates",
    "/kaggle/input/arcagi3-agent",      # dataset fallback
    "/kaggle/input/arcagi3-agent/src",
    "/kaggle/input/arcagi3",
    os.path.dirname(os.path.abspath(__file__)),
]
for _p in _CANDIDATES:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_Policy = None
_IMPORT_ERR = None
# Phase A is opt-in via ARCAGI3_ONLINE=1. The online policy is GPU-ONLY + fail-safe: on a
# P100/no-usable-GPU eval image it disables the model and runs the pure SalienceExplorer (the
# banked 0.33), so enabling it can never ship below the floor. Default OFF until A.4 proves it
# beats 0.33 on real games. Banked v6 = SalienceExplorer (border=2, trust=3) = 0.33.
if os.getenv("ARCAGI3_ONLINE") == "1":
    try:
        from arcagi3.online_explorer import OnlineLearningExplorer as _Policy
    except Exception:
        _Policy = None
if _Policy is None:
    try:
        from arcagi3.salience_explorer import SalienceExplorer as _Policy
    except Exception:
        try:  # fallback to the hybrid if the new module is unavailable
            from arcagi3.policy import HybridPolicy as _Policy
        except Exception as _e:  # noqa: BLE001
            _IMPORT_ERR = "".join(traceback.format_exception(type(_e), _e, _e.__traceback__))
            print(f"[my_agent] arcagi3 import FAILED -> random fallback.\n{_IMPORT_ERR}", flush=True)
_HybridPolicy = _Policy  # back-compat name used below

try:
    from agents.agent import Agent as _BaseAgent
except Exception:  # local dev / stub
    class _BaseAgent:  # minimal stub
        def __init__(self, *a, **k):
            self.game_id = k.get("game_id", "?")
            self.action_counter = 0
            self.frames = []

from arcengine import GameAction as _GA  # noqa: E402
from arcengine import GameState as _GS  # noqa: E402

_TIME_BUDGET_S = 8 * 3600 - 5 * 60


class MyAgent(_BaseAgent):
    """Hybrid explorer (fail-safe). Falls back to random actions if arcagi3 is unavailable."""

    MAX_ACTIONS = float("inf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pol = _HybridPolicy() if _HybridPolicy is not None else None
        self._t0 = time.time()
        seed = int(time.time() * 1e6) % (2 ** 32 - 1)
        random.seed(seed)

    def is_done(self, frames, latest_frame):
        try:
            if latest_frame.state is _GS.WIN:
                return True
        except Exception:
            pass
        return (time.time() - self._t0) >= _TIME_BUDGET_S

    def _avail_ids(self, latest_frame):
        out = []
        for a in (getattr(latest_frame, "available_actions", None) or []):
            try:
                out.append(a.value if hasattr(a, "value") else int(a))
            except Exception:
                pass
        return out

    def _random_action(self, latest_frame):
        st = getattr(latest_frame, "state", None)
        if st is _GS.NOT_PLAYED or st is _GS.GAME_OVER:
            a = _GA.RESET
            a.reasoning = "reset"
            return a
        avail = self._avail_ids(latest_frame) or [1, 2, 3, 4]
        aid = random.choice(avail)
        if aid == 6:
            a = _GA.ACTION6
            a.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            a.reasoning = "random click"
            return a
        a = _GA.from_id(aid)
        a.reasoning = "random"
        return a

    def choose_action(self, frames, latest_frame):
        if self._pol is None:
            return self._random_action(latest_frame)
        try:
            import numpy as np

            arr = np.asarray(latest_frame.frame, dtype=np.int8)
            grid = arr[-1] if arr.ndim == 3 else arr
            st = latest_frame.state
            token = self._pol.decide(
                grid,
                gstate_terminal=(st is _GS.GAME_OVER),
                gstate_notplayed=(st is _GS.NOT_PLAYED),
                levels=int(getattr(latest_frame, "levels_completed", 0) or 0),
                available=self._avail_ids(latest_frame),
            )
            if token[0] == "reset":
                a = _GA.RESET
                a.reasoning = "reset"
                return a
            if token[0] == "S":
                a = _GA.from_id(token[1])
                a.reasoning = "hybrid"
                return a
            a = _GA.ACTION6
            a.set_data({"x": int(token[1]), "y": int(token[2])})
            a.reasoning = "hybrid-click"
            return a
        except Exception as e:  # never die mid-game
            print(f"[my_agent] choose_action error -> random: {e}", flush=True)
            return self._random_action(latest_frame)
