"""Frontier explorer — a model-free level explorer (Pack 4, 2026-08-29).

A compact port of the "just-explore" method (dolphin-in-a-coma/arc-agi-3-
just-explore, MIT; 3rd in the ARC-AGI-3 preview with zero LLM calls; median
17 private levels after the graph-reset fix). The idea:

  * Frame processing: 4-connected single-colour segments; status bars are
    segments hugging a border (aspect ratio >= 5, or >= 3 same-shaped twins on
    the same border) and are masked before hashing; click candidates are one
    per segment, grouped into five priority tiers (salient colour + medium
    size first, status-bar segments last); keyboard actions sit in tier 0.
  * Level graph: nodes are masked-frame hashes; each node keeps its untested
    candidates; the explorer takes an untested candidate in the active tier
    at the current node, or walks the shortest path to the nearest node that
    still has one (frontier); when nothing is reachable it opens the next tier.

Pure Python, no numpy; a 64x64 frame segments in a few milliseconds.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

Grid = tuple[tuple[int, ...], ...]

SALIENT = set(range(6, 16))
MIN_WIDTH, MAX_WIDTH = 2, 32
EDGE_DIST = 3
BAR_RATIO = 5
TWINS = 3
N_GROUPS = 5
KEYBOARD = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


# ---------------------------------------------------------------- frames ---
def segments(grid: Grid) -> list[dict[str, Any]]:
    """4-connected same-colour components over the whole grid (background too)."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    label = [[-1] * w for _ in range(h)]
    out: list[dict[str, Any]] = []
    for r0 in range(h):
        for c0 in range(w):
            if label[r0][c0] >= 0:
                continue
            color = grid[r0][c0]
            idx = len(out)
            label[r0][c0] = idx
            stack = [(r0, c0)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and label[nr][nc] < 0 and grid[nr][nc] == color:
                        label[nr][nc] = idx
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            r_min, r_max, c_min, c_max = min(rs), max(rs), min(cs), max(cs)
            out.append({
                "id": idx, "color": int(color), "area": len(cells), "cells": cells,
                "bbox": (r_min, c_min, r_max, c_max),
                "height": r_max - r_min + 1, "width": c_max - c_min + 1,
                "rect": len(cells) == (r_max - r_min + 1) * (c_max - c_min + 1),
            })
    return out


def _edges_of(seg: dict[str, Any], h: int, w: int) -> list[str]:
    r_min, c_min, r_max, c_max = seg["bbox"]
    edges = []
    if c_max < EDGE_DIST:
        edges.append("left")
    if c_min > w - 1 - EDGE_DIST:
        edges.append("right")
    if r_max < EDGE_DIST:
        edges.append("top")
    if r_min > h - 1 - EDGE_DIST:
        edges.append("bottom")
    return edges


def status_bar_mask(grid: Grid, segs: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """just-explore's rule: border-hugging bars (aspect >= 5) or >= 3 twins on a border."""
    if not grid:
        return set()
    h, w = len(grid), len(grid[0])
    mask: set[tuple[int, int]] = set()
    by_edge: dict[str, list[dict[str, Any]]] = {}
    for seg in segs:
        for e in _edges_of(seg, h, w):
            by_edge.setdefault(e, []).append(seg)
    for edge, group in by_edge.items():
        horizontal = edge in ("top", "bottom")
        for seg in group:
            ratio = seg["width"] / seg["height"] if seg["height"] else 0
            is_bar = (ratio >= BAR_RATIO) if horizontal else (ratio <= 1 / BAR_RATIO if ratio else False)
            twins = [t for t in group if t is not seg and t["color"] == seg["color"]
                     and t["rect"] == seg["rect"] and t["width"] == seg["width"] and t["height"] == seg["height"]]
            if is_bar or len(twins) + 1 >= TWINS:
                mask.update(seg["cells"])
    return mask


def frame_hash(grid: Grid, mask: set[tuple[int, int]]) -> str:
    h = hashlib.sha1()
    for r, row in enumerate(grid):
        vals = bytes((16 if (r, c) in mask else int(v)) & 0xFF for c, v in enumerate(row))
        h.update(vals)
        h.update(b"\n")
    return h.hexdigest()[:20]


def group_of(seg: dict[str, Any], masked: bool) -> int:
    salient = seg["color"] in SALIENT
    medium = MIN_WIDTH <= seg["width"] <= MAX_WIDTH and MIN_WIDTH <= seg["height"] <= MAX_WIDTH
    if masked:
        return 4
    if salient and medium:
        return 0
    if medium:
        return 1
    if salient:
        return 2
    return 3


@dataclass
class Candidate:
    kind: str                 # "key" or "click"
    action: str               # engine action name
    data: dict[str, Any]      # {} or {"x":..,"y":..}
    group: int
    label: str


def candidates_for(grid: Grid, available: set[int], mask: set[tuple[int, int]],
                   segs: list[dict[str, Any]], rng: random.Random) -> list[Candidate]:
    out: list[Candidate] = []
    for value, name in KEYBOARD.items():
        if value in available:
            out.append(Candidate("key", name, {}, 0, name))
    if 6 in available:
        for seg in segs:
            cells = seg["cells"]
            masked = all(cell in mask for cell in cells[: min(len(cells), 8)])
            r, c = cells[rng.randrange(len(cells))]
            out.append(Candidate("click", "ACTION6", {"x": int(c), "y": int(r)}, group_of(seg, masked),
                                 f"click({r},{c}) colour {seg['color']} area {seg['area']}"))
    return out


# ----------------------------------------------------------------- graph ---
@dataclass
class Node:
    key: str
    candidates: list[Candidate]
    untested: dict[int, set[int]] = field(default_factory=dict)   # group -> candidate indices
    result: dict[int, int] = field(default_factory=dict)          # idx -> 1 changed / -1 no change
    target: dict[int, str] = field(default_factory=dict)          # idx -> node key

    def __post_init__(self) -> None:
        for i, cand in enumerate(self.candidates):
            self.untested.setdefault(cand.group, set()).add(i)

    def open_in(self, active: int) -> list[int]:
        return [i for g in range(active + 1) for i in sorted(self.untested.get(g, ()))]


class FrontierExplorer:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.rev: dict[str, set[tuple[str, int]]] = {}     # target -> {(source, idx)}
        self.active = 0
        self.dist: dict[str, int] = {}
        self.next_hop: dict[str, int] = {}                 # node -> candidate idx towards frontier
        self.last: tuple[str, int] | None = None
        self.mask: set[tuple[int, int]] | None = None
        self.stats = Counter()

    # -- observation --------------------------------------------------------
    def observe(self, grid: Grid, available: list[int] | set[int]) -> str:
        segs = segments(grid)
        if self.mask is None:
            self.mask = status_bar_mask(grid, segs)
        key = frame_hash(grid, self.mask)
        if key not in self.nodes:
            cands = candidates_for(grid, set(int(a) for a in available), self.mask, segs, self.rng)
            self.nodes[key] = Node(key, cands)
            self.stats["nodes"] += 1
            self._rebuild()
        return key

    # -- learning -----------------------------------------------------------
    def record(self, prev_key: str, idx: int, new_key: str) -> None:
        node = self.nodes.get(prev_key)
        if node is None or idx not in range(len(node.candidates)):
            return
        changed = new_key != prev_key
        node.result[idx] = 1 if changed else -1
        node.untested.get(node.candidates[idx].group, set()).discard(idx)
        if changed:
            node.target[idx] = new_key
            self.rev.setdefault(new_key, set()).add((prev_key, idx))
            self.stats["edges"] += 1
        else:
            self.stats["noops"] += 1
        self._rebuild()

    # -- decision -----------------------------------------------------------
    def choose(self, key: str) -> tuple[int, str]:
        node = self.nodes[key]
        while True:
            open_idx = node.open_in(self.active)
            if open_idx:
                # lowest group first, random inside the group
                best_group = min(node.candidates[i].group for i in open_idx)
                pool = [i for i in open_idx if node.candidates[i].group == best_group]
                i = self.rng.choice(pool)
                self.stats["explore"] += 1
                return i, f"untested tier {best_group}"
            hop = self.next_hop.get(key)
            if hop is not None:
                self.stats["travel"] += 1
                return hop, f"towards frontier (dist {self.dist.get(key)})"
            if self.active < N_GROUPS - 1:
                self.active += 1
                self.stats["tier_advance"] += 1
                self._rebuild()
                continue
            # everything exhausted: random tested-changing edge, else random candidate
            changing = [i for i, r in node.result.items() if r == 1]
            i = self.rng.choice(changing) if changing else self.rng.randrange(len(node.candidates))
            self.stats["random"] += 1
            return i, "exhausted: random"

    # -- distances (BFS from frontier over reverse edges) --------------------
    def _rebuild(self) -> None:
        frontier = [k for k, n in self.nodes.items() if n.open_in(self.active)]
        dist: dict[str, int] = {k: 0 for k in frontier}
        hop: dict[str, int] = {}
        dq = deque(frontier)
        while dq:
            cur = dq.popleft()
            for src, idx in self.rev.get(cur, ()):
                if src not in dist:
                    dist[src] = dist[cur] + 1
                    hop[src] = idx
                    dq.append(src)
        self.dist, self.next_hop = dist, hop
        self.stats["frontier"] = len(frontier)

    def frontier_size(self) -> int:
        return sum(1 for n in self.nodes.values() if n.open_in(self.active))
