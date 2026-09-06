"""Evidence-integrity aid (EVID, 2026-09-06).

Evidence (docs/research-2026-09-06/J-judge-0906.md §B): the never-passed
walls (cd82 L2, dc22 L2, lf52 L2, ...) do not lose on hypothesis starvation;
they lose on EVIDENCE HANDLING — the post-clear frame diffed as level data,
mis-transcribed references, misread batched logs, a click at (16,25) vs the
model's own (16,32), false memory of when an object appeared. This graft
makes the harness compute the before/after evidence itself and hand it to
the model inside the tool result it already reads.

WHERE IT HOOKS (verified on the june_stock bundle):
  ToolAgent._run_python_tool(state_path, arguments) -> _ToolDispatchResult
  is the ONLY path an `action(...)` batch takes (tool_agent.py: _dispatch_tool
  -> _run_python_tool; analyze() appends `dispatch.content` verbatim as the
  {"role": "tool"} message the model sees, and renders it into the transcript
  through _render_tool_result_display, which shows the `stdout` field). The
  content is the JSON string from _render_tool_payload. The wrapper parses
  it, appends the evidence block to `stdout` (creating the field when the
  snippet printed nothing) and re-dumps with the stock `indent=2` — so the
  model reads it as tool output, the transcript shows it under
  [TOOL RESULT: python], and the persisted history carries it forward.

FRAMES: every executed action appends one HistoryEntry(action, Frame(grid,
step, level)) to the live _HarnessGameSession.history_entries and rewrites
the runtime state (solver.py _execute_action). `step_env.__self__` is that
session (ToolAgent stores it as self._step_env_callback for the duration of
analyze()). The wrapper snapshots len(history_entries) before the stock call
and diffs the entries appended during it; when no session is reachable it
falls back to load_runtime_state(state_path) before/after. Frame.level is
solver._level_number = levels_completed + 1 (capped at number_of_levels), so
a level clear inside a batch is a strict increase of `level` between two
consecutive frames — the frame returned by the clearing action already shows
the NEXT level (cd82 transcript, action 28: MOUSE+SPACE cleared L1 and the
model then read rows 33-44 of the L2 board as L1 evidence).

COORDINATES: (row, col), 0-based. `Frame.ascii` = format_grid_ascii(grid)
= one line per grid row, one char per column, so `.ascii.splitlines()[row][col]`
is the cell; the segmentation `boundary` is `[row, col]`; MOUSE takes
`row`/`col` and the solver maps row -> engine y, col -> engine x
(solver.py:528-534, _model_mouse_action_data). The block states this
convention in its header.

WHAT IT EMITS (all flags read at call time; EVID_ENABLE=0 is byte-identical
stock):
  (a) object-level before/after diff of the batch: connected components by
      color (EVID_CONNECTIVITY 4 [default, = the sandbox's segment_layer] or 8),
      matched across the two frames and reported as MOVED (old bbox -> new
      bbox, delta), APPEARED, VANISHED, RECOLORED (same cells, new color),
      RESHAPED (same color, overlapping cells, different shape; e.g. a HUD
      bar shrinking); components covering >= EVID_BG_FRACTION [0.25] of the
      grid are background and never listed; `[edge]` tags a bbox touching the
      border (HUD bars). Capped at EVID_MAX_ENTRIES [40] entries and
      EVID_MAX_CHARS [1500] chars for the whole block.
  (b) per-action trace for multi-action batches (EVID_TRACE=1, at most
      EVID_TRACE_MAX [12] actions listed): after action k, the position of
      the single component that moved (tracked by color+shape across the
      batch), else the count of changed cells.
  (c) LEVEL-TRANSITION FLAG: when Frame.level rises inside the batch the
      block starts with "LEVEL CLEARED after action k — the frames after it
      belong to the NEXT level; do not diff them against this level" and the
      diff is split at that boundary (old level: before -> frame k-1; new
      level: frame k -> end). GAME_OVER / run-complete notes come from the
      stock last_action_result.

Failure policy: any exception inside the wrapper returns the stock result
untouched (counted in status()["errors"]). No threads, no file writes.
Conventions: module _STATE/_STOCK, install() -> "evidence: OK" / "evidence:
SKIP (...)", status() counters diffs_emitted / level_flags / chars_added.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any

_STATE: dict[str, Any] = {
    "installed": False,
    "diffs_emitted": 0,
    "level_flags": 0,
    "chars_added": 0,
    "traces_emitted": 0,
    "errors": 0,
    "skips": {},
    "per_game": {},
}
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

MARKER = "[EVID]"
DEFAULT_MAX_ENTRIES = 40
DEFAULT_MAX_CHARS = 1500
DEFAULT_TRACE_MAX = 24
DEFAULT_CONNECTIVITY = 4
DEFAULT_BG_FRACTION = 0.25

# inference/utils/grid_utils.py (june_stock) — copied so the pure diff code has no bundle import
ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
ARC_COLOR_NAMES = ("white", "light gray", "gray", "dark gray", "charcoal", "black", "magenta", "pink",
                   "red", "blue", "sky blue", "yellow", "orange", "dark red", "light green", "purple")
_NEIGH = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
}
_KIND_ORDER = {"MOVED": 0, "APPEARED": 1, "VANISHED": 2, "RECOLORED": 3, "RESHAPED": 4}
IN_PLACE_JACCARD = 0.5
RESHAPE_NOISE_DELTA = 2      # RESHAPED with |size delta| <= this and an unchanged bbox is dropped


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("EVID_ENABLE", "1").lower() not in _OFF


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(_env(name, str(default))))
    except ValueError:
        return default


def max_entries() -> int:
    return _env_int("EVID_MAX_ENTRIES", DEFAULT_MAX_ENTRIES, 1)


def max_chars() -> int:
    return _env_int("EVID_MAX_CHARS", DEFAULT_MAX_CHARS, 120)


def trace_enabled() -> bool:
    return _env("EVID_TRACE", "1").lower() not in _OFF


def trace_max() -> int:
    return _env_int("EVID_TRACE_MAX", DEFAULT_TRACE_MAX, 1)


def connectivity() -> int:
    return 8 if _env("EVID_CONNECTIVITY", str(DEFAULT_CONNECTIVITY)) == "8" else 4


def bg_fraction() -> float:
    try:
        return min(1.0, max(0.01, float(_env("EVID_BG_FRACTION", str(DEFAULT_BG_FRACTION)))))
    except ValueError:
        return DEFAULT_BG_FRACTION


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "installed": _STATE["installed"],
            "enabled": enabled(),
            "max_entries": max_entries(),
            "max_chars": max_chars(),
            "trace": trace_enabled(),
            "trace_max": trace_max(),
            "connectivity": connectivity(),
            "bg_fraction": bg_fraction(),
            "diffs_emitted": _STATE["diffs_emitted"],
            "level_flags": _STATE["level_flags"],
            "chars_added": _STATE["chars_added"],
            "traces_emitted": _STATE["traces_emitted"],
            "errors": _STATE["errors"],
            "skips": dict(_STATE["skips"]),
            "per_game": {k: dict(v) for k, v in _STATE["per_game"].items()},
        }


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] = _STATE["skips"].get(reason, 0) + 1


# --------------------------------------------------------------- objects ---
@dataclass(frozen=True)
class Comp:
    color: int
    cells: frozenset
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def shape(self) -> frozenset:
        return frozenset((r - self.r0, c - self.c0) for r, c in self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r1, self.c1)


def grid_shape(grid: Any) -> tuple[int, int]:
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0) if rows else 0
    return rows, cols


def _cell(grid: Any, r: int, c: int) -> int:
    row = grid[r]
    return int(row[c]) if c < len(row) else 0


def components(grid: Any, conn: int = 4) -> list[Comp]:
    """Same-color connected components (4- or 8-connectivity), reading order."""
    h, w = grid_shape(grid)
    if h == 0 or w == 0:
        return []
    neigh = _NEIGH[8 if conn == 8 else 4]
    seen = [[False] * w for _ in range(h)]
    out: list[Comp] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc]:
                continue
            color = _cell(grid, sr, sc)
            seen[sr][sc] = True
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            r0 = r1 = sr
            c0 = c1 = sc
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
                for dr, dc in neigh:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and _cell(grid, nr, nc) == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out.append(Comp(color, frozenset(cells), r0, c0, r1, c1))
    return out


def color_label(color: int) -> str:
    i = max(0, min(15, int(color)))
    return f"{ARC_COLOR_CHARS[i]}/{ARC_COLOR_NAMES[i]}"


def fmt_bbox(b: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = b
    if (r0, c0) == (r1, c1):
        return f"({r0},{c0})"
    return f"({r0},{c0})-({r1},{c1})"


def _edge(b: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = b
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def diff_grids(before: Any, after: Any, *, conn: int = 4, bg_frac: float = DEFAULT_BG_FRACTION) -> dict[str, Any]:
    """Object-level diff of two grids. Returns {entries, changed_cells, note, shape}.

    entries: dicts {kind, color, color2, size, size2, bbox, bbox2, edge}
    kinds: MOVED (same color+shape, new place), APPEARED, VANISHED,
    RECOLORED (identical cells, new color), RESHAPED (same color, overlapping
    cells, different shape). Components >= bg_frac of the grid are treated as
    background and never listed (they change whenever anything moves).
    """
    hb, wb = grid_shape(before)
    ha, wa = grid_shape(after)
    if (hb, wb) != (ha, wa):
        return {"entries": [], "changed_cells": None, "shape": (ha, wa),
                "note": f"grid shape changed {hb}x{wb} -> {ha}x{wa}"}
    h, w = ha, wa
    changed = {(r, c) for r in range(h) for c in range(w) if _cell(before, r, c) != _cell(after, r, c)}
    if not changed:
        return {"entries": [], "changed_cells": 0, "shape": (h, w), "note": "no cell changed"}
    bg_limit = max(2, int(bg_frac * h * w))
    # candidates: components touching a changed cell OR adjacent to one (a bar that only
    # shrank keeps its surviving cells unchanged, yet it is the same object reshaped)
    near = set(changed)
    for r, c in changed:
        for dr, dc in _NEIGH[8]:
            if 0 <= r + dr < h and 0 <= c + dc < w:
                near.add((r + dr, c + dc))
    bc = [b for b in components(before, conn) if b.cells & near]
    ac = [a for a in components(after, conn) if a.cells & near]
    entries: list[dict[str, Any]] = []
    used_b: set[int] = set()
    used_a: set[int] = set()

    # 0. identical (same cells, same color): adjacent bystanders, consumed silently
    for i, b in enumerate(bc):
        for j, a in enumerate(ac):
            if j in used_a:
                continue
            if a.cells == b.cells and a.color == b.color:
                used_b.add(i)
                used_a.add(j)
                break

    def add(kind: str, b: Comp | None, a: Comp | None) -> None:
        ref = a if a is not None else b
        assert ref is not None
        entries.append({
            "kind": kind,
            "color": (b if b is not None else a).color,
            "color2": a.color if (a is not None and b is not None and a.color != b.color) else None,
            "size": (b if b is not None else a).size,
            "size2": a.size if (a is not None and b is not None) else None,
            "bbox": (b if b is not None else a).bbox,
            "bbox2": a.bbox if (a is not None and b is not None) else None,
            "edge": _edge(ref.bbox, h, w),
        })

    # 1. RECOLORED: identical cell set, different color
    for i, b in enumerate(bc):
        if i in used_b:
            continue
        for j, a in enumerate(ac):
            if j in used_a:
                continue
            if a.cells == b.cells and a.color != b.color:
                used_b.add(i)
                used_a.add(j)
                add("RECOLORED", b, a)
                break
    # 2. IN PLACE (judge fix 09-06): same color, cell overlap Jaccard >= IN_PLACE_JACCARD, best
    #    overlap first — an occluded / uncovered STATIONARY tile pairs with itself here, so it can
    #    never be reported as a mover swapped with a distant look-alike. A pure translation that
    #    still overlaps (a 1-cell step of a big tile) is labelled MOVED, anything else RESHAPED.
    cands = []
    for i, b in enumerate(bc):
        if i in used_b:
            continue
        for j, a in enumerate(ac):
            if j in used_a or a.color != b.color:
                continue
            inter = len(a.cells & b.cells)
            if inter == 0:
                continue
            jac = inter / len(a.cells | b.cells)
            if jac >= IN_PLACE_JACCARD:
                cands.append((-jac, i, j))
    for _, i, j in sorted(cands):
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        b, a = bc[i], ac[j]
        if a.size == b.size and a.shape == b.shape:
            add("MOVED", b, a)
        elif b.size >= bg_limit or a.size >= bg_limit:
            continue                      # background-like: consumed, not listed
        else:
            add("RESHAPED", b, a)
    # 3. MOVED: same color + shape, nearest first (only components not matched in place)
    cands = []
    for i, b in enumerate(bc):
        if i in used_b:
            continue
        for j, a in enumerate(ac):
            if j in used_a or a.color != b.color or a.size != b.size or a.shape != b.shape:
                continue
            cands.append((abs(a.r0 - b.r0) + abs(a.c0 - b.c0), i, j))
    for _, i, j in sorted(cands):
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        add("MOVED", bc[i], ac[j])
    # 3b. RESHAPED with a weaker overlap (Jaccard >= 0.3), best overlap first
    cands = []
    for i, b in enumerate(bc):
        if i in used_b:
            continue
        for j, a in enumerate(ac):
            if j in used_a or a.color != b.color:
                continue
            inter = len(a.cells & b.cells)
            if inter == 0:
                continue
            jac = inter / len(a.cells | b.cells)
            if jac >= 0.3:
                cands.append((-jac, i, j))
    for _, i, j in sorted(cands):
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        if bc[i].size >= bg_limit or ac[j].size >= bg_limit:
            continue                      # background-like: consumed, not listed
        add("RESHAPED", bc[i], ac[j])
    # 4. leftovers (only components that actually contain a changed cell)
    for i, b in enumerate(bc):
        if i not in used_b and b.size < bg_limit and b.cells & changed:
            add("VANISHED", b, None)
    for j, a in enumerate(ac):
        if j not in used_a and a.size < bg_limit and a.cells & changed:
            add("APPEARED", None, a)
    entries = [e for e in entries
               if not (e["kind"] == "RESHAPED" and e["bbox2"] == e["bbox"]
                       and abs(int(e["size2"]) - int(e["size"])) <= RESHAPE_NOISE_DELTA)]
    entries.sort(key=lambda e: (_KIND_ORDER[e["kind"]], -e["size"], e["bbox"]))
    return {"entries": entries, "changed_cells": len(changed), "shape": (h, w), "note": ""}


def render_entries(entries: list[dict[str, Any]], cap: int) -> tuple[list[str], int]:
    """Entry lines with the edge-touching RESHAPED bars (HUD timers / progress bars) collapsed
    into one summary line. Returns (lines, entries_not_shown)."""
    bars = [e for e in entries if e["kind"] == "RESHAPED" and e.get("edge")]
    rest = [e for e in entries if not (e["kind"] == "RESHAPED" and e.get("edge"))]
    lines = [render_entry(e) for e in rest[:cap]]
    hidden = max(0, len(rest) - cap)
    if bars:
        colors = sorted({color_label(e["color"]) for e in bars})
        deltas = ", ".join(f"{e['size']}->{e['size2']}" for e in bars[:4]) + (", …" if len(bars) > 4 else "")
        lines.append(f"HUD/edge bars reshaped: {len(bars)} ({', '.join(colors)}; sizes {deltas}) — border strips, not gameplay objects")
    return lines, hidden


def render_entry(e: dict[str, Any]) -> str:
    kind = e["kind"]
    edge = " [edge]" if e.get("edge") else ""
    if kind == "MOVED":
        dr = e["bbox2"][0] - e["bbox"][0]
        dc = e["bbox2"][1] - e["bbox"][1]
        return (f"MOVED {color_label(e['color'])} size {e['size']}: {fmt_bbox(e['bbox'])} -> "
                f"{fmt_bbox(e['bbox2'])} (d row {dr:+d}, col {dc:+d}){edge}")
    if kind == "APPEARED":
        return f"APPEARED {color_label(e['color'])} size {e['size']} at {fmt_bbox(e['bbox'])}{edge}"
    if kind == "VANISHED":
        return f"VANISHED {color_label(e['color'])} size {e['size']} at {fmt_bbox(e['bbox'])}{edge}"
    if kind == "RECOLORED":
        return (f"RECOLORED size {e['size']} at {fmt_bbox(e['bbox'])}: {color_label(e['color'])} -> "
                f"{color_label(e['color2'])}{edge}")
    return (f"RESHAPED {color_label(e['color'])} size {e['size']}->{e['size2']} at {fmt_bbox(e['bbox'])} -> "
            f"{fmt_bbox(e['bbox2'])}{edge}")


def object_count(grid: Any, *, conn: int = 4, bg_frac: float = DEFAULT_BG_FRACTION) -> int:
    h, w = grid_shape(grid)
    limit = max(2, int(bg_frac * h * w))
    return sum(1 for c in components(grid, conn) if c.size < limit)


# ---------------------------------------------------------------- block ---
def _segments(levels: list[int]) -> list[tuple[int, int, int | None]]:
    """Split frame indexes 0..n at level rises. Returns [(start, end, cleared_after_action)]
    where frames[start..end] share a level and cleared_after_action = the 1-based action
    index whose frame (frames[k]) opened the NEXT segment (None for the last segment)."""
    segs: list[tuple[int, int, int | None]] = []
    start = 0
    for k in range(1, len(levels)):
        if levels[k] > levels[k - 1]:
            segs.append((start, k - 1, k))
            start = k
    segs.append((start, len(levels) - 1, None))
    return segs


def _trace_lines(frames: list[Any], actions: list[str], levels: list[int], *, conn: int, bg_frac: float,
                 limit: int) -> list[str]:
    parts: list[str] = []
    tracked: tuple[int, int] | None = None      # (color, size) of the object followed across the batch
    n = len(actions)
    cut_at: int | None = None
    for k in range(1, n + 1):
        if len(parts) >= limit and k < n:
            if cut_at is None:
                cut_at = k
            continue                      # cut, but the FINAL action is always traced below
        if cut_at is not None:
            parts.append(f"(+{n - cut_at} more actions not traced)")
            cut_at = None
        name = actions[k - 1]
        if levels[k] > levels[k - 1]:
            parts.append(f"{k} {name}: LEVEL CLEARED (frame now level {levels[k]})")
            tracked = None
            continue
        d = diff_grids(frames[k - 1], frames[k], conn=conn, bg_frac=bg_frac)
        if d["changed_cells"] is None:
            parts.append(f"{k} {name}: {d['note']}")
            continue
        if d["changed_cells"] == 0:
            parts.append(f"{k} {name}: no change")
            continue
        moved = [e for e in d["entries"] if e["kind"] == "MOVED"]
        mover = None
        if len(moved) == 1:
            mover = moved[0]
        elif moved and tracked is not None:
            for e in moved:
                if (e["color"], e["size"]) == tracked:
                    mover = e
                    break
        if mover is not None:
            tracked = (mover["color"], mover["size"])
            parts.append(f"{k} {name}: mover {color_label(mover['color'])} size {mover['size']} -> "
                         f"{fmt_bbox(mover['bbox2'])}")
        else:
            parts.append(f"{k} {name}: {d['changed_cells']} cells changed")
    return parts


def build_evidence(frames: list[Any], actions: list[str], levels: list[int], *, steps: list[int] | None = None,
                   notes: list[str] | None = None, conn: int | None = None, bg_frac: float | None = None,
                   entry_cap: int | None = None, char_cap: int | None = None, trace: bool | None = None,
                   trace_cap: int | None = None) -> tuple[str, dict[str, Any]]:
    """frames[0] = board before the batch, frames[k] = board after action k (k=1..n);
    actions[k-1] = display name of action k; levels[k] = Frame.level of frames[k].
    Returns (block text, info)."""
    conn = connectivity() if conn is None else conn
    bg_frac = bg_fraction() if bg_frac is None else bg_frac
    entry_cap = max_entries() if entry_cap is None else entry_cap
    char_cap = max_chars() if char_cap is None else char_cap
    trace = trace_enabled() if trace is None else trace
    trace_cap = trace_max() if trace_cap is None else trace_cap
    n = len(actions)
    if len(frames) != n + 1 or len(levels) != n + 1 or n == 0:
        raise ValueError("frames/levels must have one more element than actions")
    if steps and len(steps) == n:
        span = f"actions {steps[0]}-{steps[-1]}" if n > 1 else f"action {steps[0]}"
    else:
        span = f"{n} actions" if n > 1 else "1 action"
    header = (f"{MARKER} harness object diff for {span} ({n} executed in this call); coords are (row,col), "
              f"0-based: row = line index of `.ascii` from the top, col = char index from the left, "
              f"the same row/col MOUSE takes")
    flags: list[str] = []
    body: list[str] = []
    info = {"level_flags": 0, "entries_total": 0, "entries_shown": 0, "truncated": False}
    segs = _segments(levels)
    for seg_idx, (s, e, cleared_k) in enumerate(segs):
        lvl = levels[s]
        if cleared_k is not None:
            flags.append(f"LEVEL CLEARED after action {cleared_k} ({actions[cleared_k - 1]}) — the frames after it "
                         f"belong to the NEXT level (level {levels[cleared_k]}); do not diff them against this level. "
                         f"The completed board of the old level is never returned — the last frame you saw before the "
                         f"clearing action is NOT the completion state; do not read it as one.")
            info["level_flags"] += 1
        if seg_idx > 0:
            body.append(f"level {lvl} start frame (after action {s}): {object_count(frames[s], conn=conn, bg_frac=bg_frac)} "
                        f"non-background objects")
        if e > s:
            label = (f"level {lvl} diff, before action {s + 1} -> after action {e}"
                     if len(segs) > 1 else f"diff, before action 1 -> after action {e}")
            d = diff_grids(frames[s], frames[e], conn=conn, bg_frac=bg_frac)
            ents = d["entries"]
            info["entries_total"] += len(ents)
            if d["changed_cells"] is None:
                body.append(f"{label}: {d['note']}")
                continue
            body.append(f"{label}: {d['changed_cells']} cells changed, {len(ents)} object changes")
            lines, hidden = render_entries(ents, entry_cap)
            info["entries_shown"] += len(ents) - hidden
            body.extend("  " + ln for ln in lines)
            if hidden:
                body.append(f"  (+{hidden} more not shown)")
        elif cleared_k is not None and seg_idx == 0:
            body.append(f"level {lvl}: the clearing action was the first of this call; nothing to diff on this level")
    for note in notes or []:
        flags.append(note)
    trace_line = ""
    if trace and n > 1:
        parts = _trace_lines(frames, actions, levels, conn=conn, bg_frac=bg_frac, limit=trace_cap)
        trace_line = "TRACE per action: " + " | ".join(parts)
        info["trace"] = True
    lines = [header, *flags, *body]
    if trace_line:
        lines.append(trace_line)
    text = "\n".join(lines)
    if len(text) > char_cap:
        # keep header + flags + trace; drop body lines from the end until it fits
        keep_tail = [trace_line] if trace_line else []
        fixed = "\n".join([header, *flags, *keep_tail])
        budget = char_cap - len(fixed) - 32
        kept: list[str] = []
        used = 0
        for ln in body:
            if used + len(ln) + 1 > budget:
                break
            kept.append(ln)
            used += len(ln) + 1
        dropped = len(body) - len(kept)
        text = "\n".join([header, *flags, *kept, f"  (... {dropped} lines cut by EVID_MAX_CHARS)", *keep_tail])
        if len(text) > char_cap:
            text = text[: char_cap - 1] + "…"
        info["truncated"] = True
    info["chars"] = len(text)
    return text, info


def inject(content: str, text: str) -> str:
    """Append the block to the tool result the model sees (the `stdout` field of
    the stock JSON payload; raw append when the content is not that JSON)."""
    payload = None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        stdout = payload.get("stdout")
        stdout = "" if stdout is None else str(stdout)
        payload["stdout"] = (stdout.rstrip("\n") + "\n\n" + text) if stdout.strip() else text
        return json.dumps(payload, indent=2)
    return str(content).rstrip("\n") + "\n\n" + text


# ------------------------------------------------------------- harness view ---
def _session_of(agent: Any) -> Any:
    cb = getattr(agent, "_step_env_callback", None)
    return getattr(cb, "__self__", None)


def _history(agent: Any, state_path: Any, runtime_mod: Any) -> list[Any]:
    sess = _session_of(agent)
    hist = getattr(sess, "history_entries", None) if sess is not None else None
    if isinstance(hist, list):
        return hist
    current, entries = runtime_mod.load_runtime_state(state_path)
    if current is not None and (not entries or entries[-1].frame.grid != current.grid):
        entries = [*entries, runtime_mod.HistoryEntry(action="", frame=current)]
    return list(entries)


def _game_id(agent: Any) -> str:
    try:
        return str(_session_of(agent).game.game_run.game_id or "?")
    except Exception:  # noqa: BLE001
        return "?"


def _notes(agent: Any) -> list[str]:
    out: list[str] = []
    last = getattr(agent, "_last_action_result", None)
    if not isinstance(last, dict):
        return out
    if last.get("run_complete"):
        out.append("RUN COMPLETE: the last action finished the whole game.")
    elif last.get("game_over"):
        out.append("GAME_OVER on the last action — the harness auto-resets this level before your next turn; "
                   "the diff ends at the game-over frame.")
    if last.get("stopped_early") and last.get("stop_reason") not in (None, "", "level_completed", "game_over",
                                                                     "run_complete"):
        out.append(f"batch stopped early: {last.get('stop_reason')} "
                   f"({last.get('executed_count')} of {last.get('requested_count')} executed).")
    return out


def _record(game: str, info: dict[str, Any], chars: int) -> None:
    with _LOCK:
        _STATE["diffs_emitted"] += 1
        _STATE["level_flags"] += int(info.get("level_flags", 0))
        _STATE["chars_added"] += chars
        _STATE["traces_emitted"] += 1 if info.get("trace") else 0
        pg = _STATE["per_game"].setdefault(game, {"diffs": 0, "level_flags": 0, "chars": 0})
        pg["diffs"] += 1
        pg["level_flags"] += int(info.get("level_flags", 0))
        pg["chars"] += chars


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "evidence: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
        from inference.agent import runtime_state as runtime_mod
    except Exception as exc:  # noqa: BLE001
        return f"evidence: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "evidence: SKIP (missing ToolAgent)"
    if getattr(cls, "_run_python_tool", None) is None:
        return "evidence: SKIP (ToolAgent._run_python_tool missing)"
    if getattr(agent_mod, "_ToolDispatchResult", None) is None:
        return "evidence: SKIP (_ToolDispatchResult missing)"

    _STOCK["run_python_tool"] = cls._run_python_tool

    def _run_python_tool(self, state_path, arguments):
        n0 = None
        before = None
        if enabled():
            try:
                hist = _history(self, state_path, runtime_mod)
                n0 = len(hist)
                before = hist[-1].frame if hist else None
            except Exception:  # noqa: BLE001
                n0 = None
        result = _STOCK["run_python_tool"](self, state_path, arguments)
        if not enabled() or n0 is None or result is None or not getattr(result, "step_executed", False):
            return result
        try:
            hist = _history(self, state_path, runtime_mod)
            new = [e for e in hist[n0:] if getattr(e, "frame", None) is not None]
            if not new or before is None:
                _skip("no_new_frames")
                return result
            frames = [before.grid, *[e.frame.grid for e in new]]
            levels = [int(before.level), *[int(e.frame.level) for e in new]]
            actions = [str(e.action or "?") for e in new]
            steps = [int(e.frame.step) for e in new]
            text, info = build_evidence(frames, actions, levels, steps=steps, notes=_notes(self))
            content = inject(str(result.content), text)
            _record(_game_id(self), info, len(content) - len(str(result.content)))
            return agent_mod._ToolDispatchResult(content=content, step_executed=result.step_executed)
        except Exception:  # noqa: BLE001
            with _LOCK:
                _STATE["errors"] += 1
            return result

    _run_python_tool._evid_stock = _STOCK["run_python_tool"]
    cls._run_python_tool = _run_python_tool
    _STATE["installed"] = True
    return "evidence: OK"
