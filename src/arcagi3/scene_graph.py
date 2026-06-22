"""Scene-graph extractor — Phase 1 of the Goal Extraction Engine (user-directed 2026-06-22).

NOT goal LEARNING (reward prediction, which failed at AUC 0.457). This is goal PERCEPTION: parse the
frame into objects + relationships + structural cues, the way a human reads the screen. No training,
no CNN, no dataset — pure symbolic visual abstraction.

extract(grid) -> a scene graph:
  objects:         per connected component (color, size, bbox, centroid, shape class)
  agent_candidates: small, rare-color, compact objects (likely the player-controlled token;
                    CONFIRM via interaction-probe before trusting — single frame can't be sure)
  target_candidates: 'framed' objects (a small object enclosed by a ring of one other color, e.g.
                    ls20 maroon-in-gray) or singleton rare-color small statics — likely destinations
  collectibles:    >=3 same-(color,size) small objects scattered (likely collect-all items)
  progress_bar:    a long thin contiguous band of one non-bg color along/near an edge (a meter)
  symmetry:        mirror-match fraction for vertical & horizontal axes (high -> symmetry task)
  containers:      large bg-filled regions ringed by one color (likely fill/arrange targets)

This module only DESCRIBES. goal_hypotheses() turns a scene graph into ranked candidate objectives;
planners (next phase) act on those. Default frontier order stays sacred: the explorer only engages a
planner when a hypothesis is confident, else falls back to salience.
"""

from __future__ import annotations

import numpy as np

from . import perception as P


def _shape_class(o: "P.Obj") -> str:
    h, w, s = o.height, o.width, o.size
    fill = s / max(h * w, 1)
    if s == 1:
        return "point"
    if h == 1 or w == 1:
        return "line"
    if fill >= 0.9 and 0.5 <= h / max(w, 1) <= 2.0:
        return "block"
    if fill < 0.55:
        return "hollow"      # ring/frame/sparse
    return "blob"


def _symmetry_scores(grid: np.ndarray, bg: int) -> dict:
    mask = (grid != bg)
    if mask.sum() == 0:
        return {"vertical": 0.0, "horizontal": 0.0}
    # vertical axis = left-right mirror; horizontal = up-down mirror (compare COLORS where either set)
    def frac(a, b):
        either = (a != bg) | (b != bg)
        if either.sum() == 0:
            return 0.0
        return float(((a == b) & either).sum()) / float(either.sum())
    v = frac(grid, grid[:, ::-1])
    h = frac(grid, grid[::-1, :])
    return {"vertical": round(v, 3), "horizontal": round(h, 3)}


def _is_framed(o: "P.Obj", grid: np.ndarray, bg: int) -> bool:
    """Is this object enclosed by a ring of a single OTHER (non-bg) color? (ls20 maroon-in-gray)."""
    r0, c0, r1, c1 = o.bbox
    if o.size > 16:
        return False
    rr0, cc0 = max(r0 - 1, 0), max(c0 - 1, 0)
    rr1, cc1 = min(r1 + 1, grid.shape[0] - 1), min(c1 + 1, grid.shape[1] - 1)
    border = []
    for c in range(cc0, cc1 + 1):
        border += [grid[rr0, c], grid[rr1, c]]
    for r in range(rr0, rr1 + 1):
        border += [grid[r, cc0], grid[r, cc1]]
    border = [int(x) for x in border]
    ring = [x for x in border if x != o.color]
    if not ring:
        return False
    vals, counts = np.unique(ring, return_counts=True)
    dom = int(vals[int(counts.argmax())])
    return dom != bg and counts.max() / len(ring) >= 0.7


def _progress_bar(objs, grid, bg) -> dict | None:
    """A long thin band of one non-bg color (aspect>=6) near a grid edge -> a meter."""
    H, W = grid.shape
    for o in objs:
        long_thin = (o.width >= 6 and o.height <= 3) or (o.height >= 6 and o.width <= 3)
        if not long_thin:
            continue
        r0, c0, r1, c1 = o.bbox
        near_edge = r0 <= 2 or c0 <= 2 or r1 >= H - 3 or c1 >= W - 3
        if near_edge:
            return {"color": int(o.color), "bbox": o.bbox, "length": max(o.width, o.height)}
    return None


def extract(grid: np.ndarray, bg: int | None = None) -> dict:
    if bg is None:
        bg = P.detect_background(grid)
    objs = P.connected_components(grid, background=bg)
    H, W = grid.shape
    # per-color stats
    by_color: dict[int, list] = {}
    for o in objs:
        by_color.setdefault(int(o.color), []).append(o)

    objects = [{
        "color": int(o.color), "size": int(o.size), "bbox": o.bbox,
        "centroid": (round(o.centroid[0], 1), round(o.centroid[1], 1)),
        "shape": _shape_class(o),
    } for o in objs]

    # agent candidates: small (<=9), compact, of a color that has FEW instances (rare token)
    agent_candidates = [
        {"color": int(o.color), "centroid": (round(o.centroid[0], 1), round(o.centroid[1], 1)),
         "size": int(o.size)}
        for o in objs
        if o.size <= 9 and len(by_color[int(o.color)]) <= 3 and _shape_class(o) in ("point", "block", "blob")
    ]
    # target candidates: framed objects, or rare small statics distinct from agents
    target_candidates = [
        {"color": int(o.color), "centroid": (round(o.centroid[0], 1), round(o.centroid[1], 1)),
         "framed": True}
        for o in objs if _is_framed(o, grid, bg)
    ]
    # collectibles: a color with >=3 small same-size objects
    collectibles = []
    for col, lst in by_color.items():
        small = [o for o in lst if o.size <= 9]
        if len(small) >= 3:
            sizes = {o.size for o in small}
            if len(sizes) <= 2:
                collectibles.append({"color": col, "count": len(small),
                                     "size": int(small[0].size)})
    progress = _progress_bar(objs, grid, bg)
    symmetry = _symmetry_scores(grid, bg)

    return {
        "bg": int(bg), "n_objects": len(objs), "n_colors": len(by_color),
        "objects": objects,
        "agent_candidates": agent_candidates,
        "target_candidates": target_candidates,
        "collectibles": collectibles,
        "progress_bar": progress,
        "symmetry": symmetry,
    }


GOAL_FAMILIES = ["reach_target", "collect_all", "complete_symmetry", "fill_progress", "match_pattern"]


def goal_hypotheses(scene: dict) -> list[dict]:
    """Rank candidate objectives from a scene graph (confidence in [0,1]). No learning."""
    h = []
    if scene["target_candidates"] and scene["agent_candidates"]:
        conf = min(1.0, 0.5 + 0.25 * len(scene["target_candidates"]))
        h.append({"goal": "reach_target", "confidence": round(conf, 2),
                  "targets": scene["target_candidates"]})
    if scene["collectibles"]:
        tot = sum(c["count"] for c in scene["collectibles"])
        h.append({"goal": "collect_all", "confidence": round(min(1.0, 0.4 + 0.1 * tot), 2),
                  "items": scene["collectibles"]})
    sym = max(scene["symmetry"]["vertical"], scene["symmetry"]["horizontal"])
    if sym >= 0.6:
        axis = "vertical" if scene["symmetry"]["vertical"] >= scene["symmetry"]["horizontal"] else "horizontal"
        h.append({"goal": "complete_symmetry", "confidence": round(sym, 2), "axis": axis})
    if scene["progress_bar"]:
        h.append({"goal": "fill_progress", "confidence": 0.5, "bar": scene["progress_bar"]})
    h.sort(key=lambda x: -x["confidence"])
    return h
