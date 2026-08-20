"""digest_animation: deterministic, compact (~60 token) narration of a multi-frame
ARC-AGI-3 action response, for injection into the duck harness action result.

Motivation (depth study wf_9809fe85): sb26's check verdict exists only in transient
animation frames (a highlight visiting slots in sequence; final board unchanged).
The v12smoke run decoded it manually through ~10 LLM calls and 801s; banksmoke
misdecoded it and got stuck. This digest narrates the same structure in one line.

Structure detected, in order of priority:
  - blinks: consecutive distinct frames toggling the same cell set back and forth
    (ft09 rejection flash, sb26 mismatch flash) -> "flash Rx3 @r0-39c20-28"
  - reveals: a small region painted in place over several frames, ending in a
    stable color (sb26 check pointer painting the expected color at each strip
    slot, in visit order) -> "O@r3c10"
  - movers: connected change-components tracked across frames by proximity,
    rendered as direction-collapsed waypoint chains (sb26 slot ring diving from
    the red tray to the green tray and back) -> "obj r21c14->r35c31->r17c31"
  - fades: steps whose transitions are almost entirely +-1 grayscale ramps are
    dropped (pure fade-in/fade-out carries no order information)
  - scene changes: a step touching > SCENE_CHANGE_CELLS cells is reported as a
    count, not narrated (e.g. the next level's board appearing after a WIN check)

Events are emitted in chronological order, so cross-track ordering (ring dove to
the green tray BETWEEN reveal 2 and reveal 3) survives into the text. That is
exactly the slot-visit-order information sb26's check hides from the final frame.

No LLM, stdlib only, deterministic.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

COLOR_CHARS = "WwgGcBMPRbSYOrNp"

CLUSTER_GAP = 2            # chebyshev gap that still joins two changed cells
TRACK_ATTACH_DIST = 16     # max centroid jump between consecutive comps of one mover
MERGE_BBOX_AREA = 64       # comp merges into the previous waypoint if union bbox <= 8x8
MERGE_CENTER_DIST = 2.0    # ...and its center stays put (a dwell, not a slow slide)
FADE_DOMINANCE = 0.85      # fraction of +-1 grayscale transitions that makes a fade step
SCENE_CHANGE_CELLS = 300   # bigger steps are reported as a count, not narrated
MAX_EVENTS = 14            # hard budget on rendered events

Grid = tuple[tuple[int, ...], ...]


# --------------------------------------------------------------------------- utils

def _norm_grid(grid: Any) -> Grid:
    rows = grid.tolist() if hasattr(grid, "tolist") else grid
    return tuple(tuple(int(c) for c in row) for row in rows or ())


def _color(v: int | None) -> str:
    if v is None:
        return "?"
    v = int(v)
    return COLOR_CHARS[v] if 0 <= v < len(COLOR_CHARS) else "?"


def _diff(a: Grid, b: Grid, mask: frozenset) -> dict:
    out = {}
    for r in range(max(len(a), len(b))):
        ra = a[r] if r < len(a) else ()
        rb = b[r] if r < len(b) else ()
        for c in range(max(len(ra), len(rb))):
            va = ra[c] if c < len(ra) else None
            vb = rb[c] if c < len(rb) else None
            if va != vb and (r, c) not in mask:
                out[(r, c)] = (va, vb)
    return out


def _components(cells: Iterable[tuple[int, int]], gap: int = CLUSTER_GAP) -> list[list[tuple[int, int]]]:
    """Cluster cells with chebyshev distance <= gap into connected components."""
    todo = set(cells)
    offsets = [(dr, dc) for dr in range(-gap, gap + 1) for dc in range(-gap, gap + 1) if dr or dc]
    comps = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        comp, queue = [seed], [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in offsets:
                n = (r + dr, c + dc)
                if n in todo:
                    todo.discard(n)
                    comp.append(n)
                    queue.append(n)
        comps.append(sorted(comp))
    comps.sort(key=lambda comp: comp[0])
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _centroid(cells) -> tuple[float, float]:
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return sum(rows) / len(rows), sum(cols) / len(cols)


def _pos_text(pos: tuple[float, float]) -> str:
    return f"r{round(pos[0])}c{round(pos[1])}"


def _is_fade_step(census: Counter) -> bool:
    total = sum(census.values())
    if not total:
        return False
    fade = sum(
        n for (old, new), n in census.items()
        if old is not None and new is not None and abs(int(old) - int(new)) == 1
        and max(int(old), int(new)) <= 5
    )
    return fade / total >= FADE_DOMINANCE


def _inverse(census: Counter) -> Counter:
    return Counter({(new, old): n for (old, new), n in census.items()})


# --------------------------------------------------------------------------- main

def digest_animation(
    frames: Sequence[Any],
    hud_mask_cells: Iterable[tuple[int, int]] | None = None,
    before: Any | None = None,
) -> str:
    """Render a multi-frame action response as one compact deterministic line.

    ``frames``: the engine's full frame list for one action (grids or ndarrays).
    ``hud_mask_cells``: (row, col) cells to ignore entirely (HUD bars/timers).
    ``before``: the board before the action, if available; sharpens step 1 and
    enables net-change reporting.
    Returns "" for single-frame responses.
    """
    grids = [_norm_grid(f) for f in frames or ()]
    if len(grids) <= 1:
        return ""
    mask = frozenset(tuple(c) for c in hud_mask_cells or ())

    # Collapse identical consecutive frames.
    distinct: list[Grid] = []
    for g in grids:
        if not distinct or distinct[-1] != g:
            distinct.append(g)

    start = _norm_grid(before) if before is not None else distinct[0]
    final = distinct[-1]

    steps = []
    prev = start
    for i, g in enumerate(distinct):
        d = _diff(prev, g, mask)
        prev = g
        if d:
            steps.append({"i": i, "cells": d, "census": Counter(d.values())})

    events: list = []  # (step, text, kind, track_id)

    # --- blink pass: runs of identical cell sets with inverse transitions
    consumed = set()
    j = 0
    while j < len(steps):
        run = [j]
        while (
            run[-1] + 1 < len(steps)
            and steps[run[-1] + 1]["i"] == steps[run[-1]]["i"] + 1
            and set(steps[run[-1] + 1]["cells"]) == set(steps[run[-1]]["cells"])
            and steps[run[-1] + 1]["census"] == _inverse(steps[run[-1]]["census"])
        ):
            run.append(run[-1] + 1)
        if len(run) >= 2:
            first = steps[run[0]]
            on_color = first["census"].most_common(1)[0][0][1]
            cycles = (len(run) + 1) // 2
            r0, c0, r1, c1 = _bbox(first["cells"])
            events.append((first["i"], f"flash {_color(on_color)}x{cycles} @r{r0}-{r1}c{c0}-{c1}", "blink", -1))
            consumed.update(run)
            j = run[-1] + 1
        else:
            j += 1
    steps = [s for k, s in enumerate(steps) if k not in consumed]

    # --- fade + scene-change pass
    kept = []
    for s in steps:
        if _is_fade_step(s["census"]):
            continue
        if len(s["cells"]) > SCENE_CHANGE_CELLS:
            events.append((s["i"], f"then {len(s['cells'])}px scene change", "scene", -1))
            continue
        kept.append(s)
    steps = kept

    # --- track pass: cluster each step, associate comps to movers by proximity.
    # All comps of one step that land on the same track are fused into a single
    # observation (a sliding ring shows up as separate erase/draw edges).
    tracks: list[dict] = []
    for s in steps:
        assignments: dict[int, list] = {}
        for comp in _components(s["cells"]):
            cen = _centroid(comp)
            best, best_d = None, None
            for ti, t in enumerate(tracks):
                d = max(abs(t["last"][0] - cen[0]), abs(t["last"][1] - cen[1]))
                if d <= TRACK_ATTACH_DIST and (best_d is None or d < best_d):
                    best, best_d = ti, d
            if best is None:
                if len(comp) == 1:
                    continue  # isolated single-cell speck (HUD tick)
                tracks.append({"waypoints": [], "last": cen, "last_step": s["i"]})
                best = len(tracks) - 1
            assignments.setdefault(best, []).append(comp)
        for ti in sorted(assignments):
            t = tracks[ti]
            cells = [cell for comp in assignments[ti] for cell in comp]
            b = _bbox(cells)
            cen = _centroid(cells)
            wps = t["waypoints"]
            merged = False
            if wps:
                wp = wps[-1]
                u = (min(wp["bbox"][0], b[0]), min(wp["bbox"][1], b[1]),
                     max(wp["bbox"][2], b[2]), max(wp["bbox"][3], b[3]))
                center_shift = max(abs(cen[0] - wp["pos"][0]), abs(cen[1] - wp["pos"][1]))
                if (
                    (u[2] - u[0] + 1) * (u[3] - u[1] + 1) <= MERGE_BBOX_AREA
                    and center_shift <= MERGE_CENTER_DIST
                    and s["i"] - wp["last_step"] <= 3
                ):
                    wp["bbox"] = u
                    wp["last_step"] = s["i"]
                    wp["n_steps"] += 1
                    wp["pos"] = ((u[0] + u[2]) / 2, (u[1] + u[3]) / 2)
                    merged = True
            if not merged:
                wps.append({
                    "pos": cen, "bbox": b, "first_step": s["i"], "last_step": s["i"],
                    "n_steps": 1, "size": len(cells),
                })
            t["last"] = wps[-1]["pos"]
            t["last_step"] = s["i"]

    # --- reveal detection: a dwell whose bbox ends uniformly painted in a color
    # it did not start with. Read from the actual frame, not the diff census, so
    # a stray ring edge merged into the dwell cannot corrupt the verdict.
    def _region_mode(grid: Grid, bbox: tuple[int, int, int, int]):
        vals = [
            grid[r][c]
            for r in range(bbox[0], bbox[2] + 1)
            for c in range(bbox[1], bbox[3] + 1)
            if r < len(grid) and c < len(grid[r])
        ]
        if not vals:
            return None, 0.0
        v, n = Counter(vals).most_common(1)[0]
        return v, n / len(vals)

    for t in tracks:
        for wp in t["waypoints"]:
            wp["reveal"] = None
            area = (wp["bbox"][2] - wp["bbox"][0] + 1) * (wp["bbox"][3] - wp["bbox"][1] + 1)
            if wp["n_steps"] >= 2 and area <= MERGE_BBOX_AREA:
                v, share = _region_mode(distinct[wp["last_step"]], wp["bbox"])
                if v is not None and share >= 0.6:
                    sv, sshare = _region_mode(start, wp["bbox"])
                    if not (sv == v and sshare >= 0.6):
                        wp["reveal"] = int(v)
        # a dwell often erases (to white) then paints: keep only the later verdict
        reveals = [wp for wp in t["waypoints"] if wp["reveal"] is not None]
        for a, b in zip(reveals, reveals[1:]):
            if max(abs(a["pos"][0] - b["pos"][0]), abs(a["pos"][1] - b["pos"][1])) <= 2:
                a["reveal"] = None

    # --- emit reveal events; emit motion segments for tracks without reveals.
    # events: (step, text, kind, track_id); kind "arrow" is the most expendable.
    for tid, t in enumerate(tracks):
        wps = t["waypoints"]
        has_reveal = any(wp["reveal"] is not None for wp in wps)
        if has_reveal:
            for wp in wps:
                if wp["reveal"] is not None:
                    events.append(
                        (wp["first_step"], f"{_color(wp['reveal'])}@{_pos_text(wp['pos'])}", "reveal", tid)
                    )
            continue
        # direction-collapse the waypoint chain
        keep = [wps[0]]
        last_dir = None
        for wp in wps[1:]:
            dr = wp["pos"][0] - keep[-1]["pos"][0]
            dc = wp["pos"][1] - keep[-1]["pos"][1]
            direction = ((dr > 1) - (dr < -1), (dc > 1) - (dc < -1))
            if direction == last_dir and direction != (0, 0):
                keep[-1] = wp  # extend the segment
            else:
                keep.append(wp)
                last_dir = direction
        if len(keep) == 1:
            wp = keep[0]
            events.append((wp["first_step"], f"chg {wp['size']}px @{_pos_text(wp['pos'])}", "chg", tid))
        else:
            events.append((keep[0]["first_step"], f"obj {_pos_text(keep[0]['pos'])}", "obj", tid))
            for wp in keep[1:]:
                events.append((wp["last_step"], f"->{_pos_text(wp['pos'])}", "arrow", tid))

    events.sort(key=lambda e: e[0])

    # --- graded budget: drop intermediate motion waypoints first (keeping each
    # track's final position), so reveals/blinks -- the order information --
    # survive; only then elide the middle generically.
    if len(events) > MAX_EVENTS:
        last_arrow = {}
        for k, e in enumerate(events):
            if e[2] == "arrow":
                last_arrow[e[3]] = k
        droppable = [k for k, e in enumerate(events) if e[2] == "arrow" and last_arrow[e[3]] != k]
        while len(events) > MAX_EVENTS and droppable:
            mid = len(droppable) // 2
            k = droppable.pop(mid)
            events[k] = None
            events = [e for e in events if e is not None]
            last_arrow = {}
            for k, e in enumerate(events):
                if e[2] == "arrow":
                    last_arrow[e[3]] = k
            droppable = [k for k, e in enumerate(events) if e[2] == "arrow" and last_arrow[e[3]] != k]
    if len(events) > MAX_EVENTS:
        head = events[: MAX_EVENTS - 3]
        tail = events[-2:]
        omitted = len(events) - len(head) - len(tail)
        events = head + [(head[-1][0], f"..{omitted} more..", "elide", -1)] + tail

    # --- header + net change
    header = f"{len(grids)}f anim"
    if start == final:
        header += " (board reverts)"

    parts = [e[1] for e in events]
    text = header + ": " + "; ".join(parts) if parts else header + ": no non-fade changes"

    if before is not None:
        net = _diff(_norm_grid(before), final, mask)
        if not net:
            text += "; no net change"
        elif len(net) <= 8:
            cen = _centroid(list(net))
            colors = Counter(v[1] for v in net.values())
            text += f"; net {len(net)}px {_color(colors.most_common(1)[0][0])}@{_pos_text(cen)}"
        else:
            r0, c0, r1, c1 = _bbox(list(net))
            text += f"; net {len(net)}px r{r0}-{r1}c{c0}-{c1}"
    return text
