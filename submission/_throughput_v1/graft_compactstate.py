"""graft_compactstate.py — stop the harness writing O(n^2) bytes per game.

WHY: the 09-10 budget arm died with `OSError: [Errno 28] No space left on device` on 17 of
25 games. Root cause is not storage, it is CHURN. `write_runtime_state` re-serialises the
ENTIRE history after EVERY action with `indent=2`, so bytes written scale with the SQUARE
of the action count: 0.19 GB at the base's 154 actions, 1.12 GB at 371, 7.46 GB at 959, and
~28 GB across 25 concurrent games. Invisible at the live budget, fatal at 3x -- so every
budget-increasing experiment hits it.

THE PATCH: write compact JSON instead of `indent=2`. The file is machine-read only, by
`load_runtime_state` -> `json.loads`, which is indifferent to whitespace. A 64x64 integer
grid at indent=2 puts each int on its own line; compact puts the row on one. **Nothing the
model sees changes** -- not the prompt, not the sandbox view, not `history`, not `ascii`.
It is purely a serialisation-format change to a file only the harness reads.

SECOND WIN, and the local smoke said it is needed: write once per step_env BATCH instead
of once per action. The solver writes after EVERY individual action (solver.py:696), but the
only reader -- the sandbox's `_serialized_runtime_state` -- re-reads once per `action()`
call. So in a batch of k actions, k-1 writes are never read by anyone. This graft suppresses
writes inside a step_env call and flushes exactly one at the end, so the file is fresh
precisely when something reads it. At the measured 2.76 actions/call that is another ~2.8x.

NOT LIVE-LEGAL BY POLICY: this is an instrument patch for diagnostic arms. It does not go
near a submission without its own gate.
"""
from __future__ import annotations

import json
from typing import Any

_STATE: dict[str, Any] = {"installed": False}


def status() -> dict[str, Any]:
    return dict(_STATE)


def install() -> str:
    if _STATE["installed"]:
        return "compactstate: SKIP (already applied)"
    try:
        from inference.agent import runtime_state as rs
    except Exception as exc:  # noqa: BLE001
        return f"compactstate: SKIP (import failed: {exc!r})"
    for name in ("write_runtime_state", "frame_to_payload", "history_entry_to_payload"):
        if getattr(rs, name, None) is None:
            return f"compactstate: SKIP ({name} missing)"

    def write_runtime_state(path, *, current_frame, history) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_frame": rs.frame_to_payload(current_frame),
            "history": [rs.history_entry_to_payload(e) for e in history],
        }
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        # separators kills the space after ':' and ',' too -- on a 4,096-int grid that
        # is most of the file. Same bytes back out of json.loads.
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    rs.write_runtime_state = write_runtime_state
    # the agent imported the symbol directly (`from ... import write_runtime_state`), so
    # rebinding the module attribute alone would leave the old function in use.
    try:
        from inference.agent import tool_agent as ta
        if getattr(ta, "write_runtime_state", None) is not None:
            ta.write_runtime_state = write_runtime_state
    except Exception:  # noqa: BLE001
        pass
    try:
        from inference.framework import solver as sv
        if getattr(sv, "write_runtime_state", None) is not None:
            sv.write_runtime_state = write_runtime_state
    except Exception:  # noqa: BLE001
        pass

    # --- second win: one write per step_env batch, not one per action ---
    batched = "no"
    try:
        from inference.framework import solver as sv
        cls = getattr(sv, "_HarnessGameSession", None)
        if cls is not None and getattr(cls, "step_env", None) and getattr(cls, "write_runtime_state", None):
            stock_write = cls.write_runtime_state
            stock_step = cls.step_env

            def write_state(self):
                # inside a batch: remember that a write is owed, do not perform it
                if getattr(self, "_cs_in_batch", False):
                    self._cs_dirty = True
                    return
                stock_write(self)

            def step_env(self, arguments):
                self._cs_in_batch = True
                self._cs_dirty = False
                try:
                    return stock_step(self, arguments)
                finally:
                    self._cs_in_batch = False
                    # flush exactly once, so the file is current the instant the
                    # sandbox re-reads it after action() returns
                    if getattr(self, "_cs_dirty", False):
                        stock_write(self)

            cls.write_runtime_state = write_state
            cls.step_env = step_env
            batched = "yes"
    except Exception:  # noqa: BLE001
        batched = "no"

    _STATE["installed"] = True
    _STATE["batched"] = batched
    return f"compactstate: compact_json+batch_write({batched}): OK"
