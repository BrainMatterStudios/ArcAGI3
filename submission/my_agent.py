# =====================================================================
# ARC-AGI-3 submission agent (ARC Prize 2026)
# A general, training-free interactive agent: motion model (avatar + per-action
# displacement) + coordinate navigation to candidate goals, with graph-based
# exploration/exploitation fallback. Reactive one-action-per-call interface.
#
# The actual logic lives in the `arcagi3` package, shipped as a Kaggle dataset and
# added to sys.path below. This file is a thin adapter to the official Agent base.
# =====================================================================
import os
import sys
import time

# Make the arcagi3 package importable. The package is uploaded as a Kaggle dataset;
# try the common attach paths plus the working dir.
_CANDIDATES = [
    "/kaggle/input/arcagi3-agent",
    "/kaggle/input/arcagi3-agent/src",
    "/kaggle/input/arcagi3",
    "/kaggle/working/arcagi3-agent",
    "/kaggle/working/ARC-AGI-3-Agents/agents/templates",
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"),
]
for _p in _CANDIDATES:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from arcagi3.policy import HybridPolicy  # noqa: E402

try:  # the official framework base; stubbed for local dev/testing
    from agents.agent import Agent as _BaseAgent  # noqa: E402
except Exception:  # pragma: no cover
    class _BaseAgent:  # minimal stub
        def __init__(self, *a, **k):
            self.game_id = k.get("game_id", "?")
            self.action_counter = 0
            self.frames = []

from arcengine import GameAction as _GA  # noqa: E402
from arcengine import GameState as _GS  # noqa: E402

# Stop ~5 min before the 8h soft budget (Kaggle hard limit is 12h overall).
_TIME_BUDGET_S = 8 * 3600 - 5 * 60


class MyAgent(_BaseAgent):
    """Hybrid explorer adapted to the official reactive Agent interface."""

    MAX_ACTIONS = float("inf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pol = HybridPolicy()
        self._t0 = time.time()

    def is_done(self, frames, latest_frame):
        try:
            if latest_frame.state is _GS.WIN:
                return True
        except Exception:
            pass
        return (time.time() - self._t0) >= _TIME_BUDGET_S

    def choose_action(self, frames, latest_frame):
        try:
            import numpy as np

            arr = np.asarray(latest_frame.frame, dtype=np.int8)
            grid = arr[-1] if arr.ndim == 3 else arr
            st = latest_frame.state
            avail = []
            for a in (getattr(latest_frame, "available_actions", None) or []):
                avail.append(a.value if hasattr(a, "value") else int(a))
            token = self._pol.decide(
                grid,
                gstate_terminal=(st is _GS.GAME_OVER),
                gstate_notplayed=(st is _GS.NOT_PLAYED),
                levels=int(getattr(latest_frame, "levels_completed", 0) or 0),
                available=avail,
            )
            if token[0] == "reset":
                act = _GA.RESET
                act.reasoning = "reset"
                return act
            if token[0] == "S":
                act = _GA.from_id(token[1])
                act.reasoning = "hybrid:explore/navigate"
                return act
            act = _GA.ACTION6
            act.set_data({"x": int(token[1]), "y": int(token[2])})
            act.reasoning = "hybrid:click"
            return act
        except Exception as e:  # never die mid-game
            act = _GA.ACTION1
            try:
                act.reasoning = f"fallback: {e}"
            except Exception:
                pass
            return act
