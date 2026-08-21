"""Measurement for bugfix patch 4 (runtime_state_cap).

Builds a synthetic runtime state with n in {100, 1000, 2000} history entries
of full 64x64 frames and times, stock vs patched:
  - write_runtime_state (disk write per action, solver.py:194-199 calls it
    EVERY action), plus resulting file size;
  - _ascii_history_view_payload + json.dumps (the per-action() sandbox state
    pipe, tool_agent.py:1466-1491 -> python_tool_sandbox stdin), plus payload
    size.

Run:  .venv/bin/python submission/_bugfix_pack/bench_runtime_state.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from test_bugfix_pack import _resolve_tree  # noqa: E402

sys.path.insert(0, str(_resolve_tree()))

import graft_bugfix  # noqa: E402
from inference.agent import runtime_state as state_mod  # noqa: E402
from inference.agent import tool_agent as agent_mod  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402

STOCK_WRITE = state_mod.write_runtime_state
STOCK_VIEW = agent_mod._ascii_history_view_payload

REPEATS = {100: 20, 1000: 5, 2000: 3}


def make_history(n: int) -> list[HistoryEntry]:
    entries = []
    for i in range(n):
        grid = tuple(tuple((i + r * 3 + c) % 16 for c in range(64)) for r in range(64))
        entries.append(
            HistoryEntry(action=f"MOUSE(row={i % 64}, col={(i * 7) % 64})",
                         frame=Frame(grid=grid, step=i, level=1 + i // 200))
        )
    return entries


def bench_write(write_fn, history, path: Path, repeats: int) -> tuple[float, int]:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        write_fn(path, current_frame=history[-1].frame, history=history)
        best = min(best, time.perf_counter() - t0)
    return best, path.stat().st_size


def bench_view(view_fn, history, repeats: int) -> tuple[float, int]:
    best = float("inf")
    size = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        payload = view_fn(history)
        rendered = json.dumps({"history": payload}, ensure_ascii=False)
        best = min(best, time.perf_counter() - t0)
        size = len(rendered.encode("utf-8"))
    return best, size


def main() -> None:
    status = graft_bugfix.patch_runtime_state_cap()
    assert status.startswith("runtime_state_cap: OK"), status
    patched_write = state_mod.write_runtime_state
    patched_view = agent_mod._ascii_history_view_payload

    print(f"{'n':>5} {'write stock':>12} {'write patched':>14} {'file stock':>11} "
          f"{'file patched':>13} {'pipe stock':>11} {'pipe patched':>13} "
          f"{'view+dump stock':>16} {'view+dump patched':>18}")
    for n in (100, 1000, 2000):
        history = make_history(n)
        reps = REPEATS[n]
        with tempfile.TemporaryDirectory() as tmp:
            s_t, s_bytes = bench_write(STOCK_WRITE, history, Path(tmp) / "s.json", reps)
            p_t, p_bytes = bench_write(patched_write, history, Path(tmp) / "p.json", reps)
        sv_t, sv_bytes = bench_view(STOCK_VIEW, history, reps)
        pv_t, pv_bytes = bench_view(patched_view, history, reps)
        print(f"{n:>5} {s_t*1000:>10.1f}ms {p_t*1000:>12.1f}ms "
              f"{s_bytes/1e6:>9.2f}MB {p_bytes/1e6:>11.2f}MB "
              f"{sv_bytes/1e6:>9.2f}MB {pv_bytes/1e6:>11.2f}MB "
              f"{sv_t*1000:>14.1f}ms {pv_t*1000:>16.1f}ms")


if __name__ == "__main__":
    main()
