"""CoroutineStrategy — drive a GENERATOR-style solver over the reactive decide() interface.

A generator solver is written in synchronous style: `obs = yield token` emits a token and receives the resulting
observation dict {grid, terminal, notplayed, levels, available}. This adapter bridges it to the portfolio's
one-token-per-call decide() contract, so the search-replay paradigm (reset/step to search a dirty run, then
double-reset + replay a clean run) can be written as ordinary control flow instead of a hand-rolled state machine.

Also provides `general_agent_gen`: a game-agnostic solver (single-click + two-phase select->apply affordance
probing, then affordance-pruned BFS, then clean replay) as such a generator. Abstains (benign) when exhausted.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient


class CoroutineStrategy:
    def __init__(self, gen_factory, seed: int = 0):
        self.gs = None
        self._factory = gen_factory
        self._gen = None
        self._done = False

    def _benign(self, available):
        for a in (5, 1, 2, 3, 4):
            if a in available:
                return ("S", a)
        return ("reset",)

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        obs = {"grid": grid, "terminal": bool(gstate_terminal), "notplayed": bool(gstate_notplayed),
               "levels": int(levels or 0), "available": available}
        if self._done:
            return self._benign(available)
        try:
            if self._gen is None:
                self._gen = self._factory(obs)
                tok = next(self._gen)           # prime to first yield (computed from obs0)
            else:
                tok = self._gen.send(obs)
            return tok if tok is not None else self._benign(available)
        except StopIteration:
            self._done = True
            return self._benign(available)


def _delta(a, b):
    return int(np.sum(a[:56, :56] != b[:56, :56]))


# ---------------------------------------------------------------------------
# GOAL-CLASS: peg-solitaire (click-jump-remove; win when one peg remains).
# General + frames-only: auto-detects the peg color (the color with the most
# similar-sized blocks) and the grid spacing (nearest-neighbour distance). A jump =
# click(peg) then click(peg + 2*spacing*dir); legal iff the peg count drops by one.
# ---------------------------------------------------------------------------
def _peg_centers(grid, color):
    bg = P.detect_background(grid)
    return sorted((int(round(o.centroid[1])), int(round(o.centroid[0])))
                  for o in P.connected_components(grid, background=bg) if o.color == color)


def _peg_class(grid):
    from collections import defaultdict
    bg = P.detect_background(grid)
    by = defaultdict(list)
    for o in P.connected_components(grid, background=bg):
        if o.color != bg and o.color != 4 and o.size <= 40:
            by[o.color].append(o)
    best = None
    for c, os in by.items():
        if len(os) < 4:
            continue
        sizes = [o.size for o in os]
        if max(sizes) > 2 * min(sizes) + 2:
            continue
        cen = [(int(round(o.centroid[1])), int(round(o.centroid[0]))) for o in os]
        sp = min((abs(a[0] - b[0]) + abs(a[1] - b[1])) for i, a in enumerate(cen) for b in cen[i + 1:])
        if 3 <= sp <= 16 and (best is None or len(os) > best[0]):
            best = (len(os), c, sp)
    return (best[1], best[2]) if best else None


def peg_solitaire_gen(obs0):
    """try the peg-solitaire class; RETURN (fall through) if it doesn't match or the DFS exhausts."""
    g0 = obs0["grid"]; avail = list(obs0["available"])
    if 6 not in avail:
        return
    cls = _peg_class(g0)
    if cls is None:
        return
    color, sp = cls
    if len(_peg_centers(g0, color)) < 3:
        return
    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    stack = [[]]; seen = set(); tries = 0
    while stack and tries < 400:
        seq = stack.pop(); tries += 1
        obs = yield ("reset",)
        aborted = False
        for (px, py, lx, ly) in seq:
            yield ("C", px, py)
            obs = yield ("C", lx, ly)
            if obs["levels"] >= 1:
                yield ("reset",); yield ("reset",)          # clean replay in a fresh run
                for (ax, ay, bx, by) in seq:
                    yield ("C", ax, ay); yield ("C", bx, by)
                while True:
                    yield ("S", avail[0] if avail else 5)   # solved -> abstain forever
            if obs["terminal"]:
                aborted = True; break
        if aborted:
            continue
        pegs = _peg_centers(obs["grid"], color); pegset = set(pegs)
        key = tuple(pegs)
        if key in seen:
            continue
        seen.add(key)
        for (px, py) in pegs:
            for dx, dy in DIRS:
                mx, my = px + sp * dx, py + sp * dy          # jumped-over cell
                lx, ly = px + 2 * sp * dx, py + 2 * sp * dy  # landing cell
                if not (0 <= lx < 64 and 0 <= ly < 64):
                    continue
                over = any(abs(mx - qx) + abs(my - qy) <= sp // 2 for (qx, qy) in pegset)
                empty = not any(abs(lx - qx) + abs(ly - qy) <= sp // 2 for (qx, qy) in pegset)
                if over and empty:
                    stack.append(seq + [(px, py, lx, ly)])
    return


def general_agent_gen(obs0):
    """game-agnostic search-replay solver as a generator (frames+feedback only, zero per-game code).
    Tries known goal-CLASSES first (peg-solitaire), then falls through to generic affordance search."""
    yield from peg_solitaire_gen(obs0)          # returns (falls through) if not a peg game / DFS exhausts
    g0 = obs0["grid"]; avail = list(obs0["available"])
    macros = [[("S", a)] for a in avail if a in (1, 2, 3, 4, 5)]
    targets = _salient(g0, 16) if 6 in avail else []

    def replay_clean(seq):
        # double-reset -> fresh scored run, then execute the winning sequence
        yield ("reset",); yield ("reset",)
        for m in seq:
            for tok in m:
                yield tok

    # ---- affordance probing (single-click + two-phase select->apply) ----
    base = {}
    if 6 in avail:
        for (cx, cy) in targets:
            yield ("reset",)
            obs = yield ("C", cx, cy)
            if obs["levels"] >= 1:
                yield from replay_clean([[("C", cx, cy)]]); return
            base[(cx, cy)] = 0 if obs["terminal"] else _delta(obs["grid"], g0)
            if not obs["terminal"] and base[(cx, cy)] >= 2:
                macros.append([("C", cx, cy)])
        sels = [t for t in targets if base.get(t, 99) <= 3][:7]
        for s in sels:
            for t in targets[:10]:
                if t == s:
                    continue
                yield ("reset",)
                obs = yield ("C", *s)
                if obs["terminal"] or obs["levels"] >= 1:
                    continue
                b = obs["grid"]
                obs = yield ("C", *t)
                d = 0 if obs["terminal"] else _delta(obs["grid"], b)
                if obs["levels"] >= 1 or abs(d - base.get(t, 0)) >= 2:
                    macros.append([("C", *s), ("C", *t)])

    # ---- affordance-pruned BFS with clean replay on win ----
    seen = set(); q = deque([[]]); nodes = 0
    while q and nodes < 6000:
        seq = q.popleft()
        obs = yield ("reset",)          # capture the root (post-reset) state so the empty seq expands
        won = False; dead = False
        for m in seq:
            for tok in m:
                obs = yield tok
                nodes += 1
                if obs["levels"] >= 1:
                    won = True; break
                if obs["terminal"]:
                    dead = True; break
            if won or dead:
                break
        if won:
            yield from replay_clean(seq); return
        if obs is not None and not dead:
            st = hash(obs["grid"][:56, :56].tobytes())
            if st not in seen:
                seen.add(st)
                for m in macros:
                    q.append(seq + [m])
    while True:                     # exhausted -> abstain (benign, floor-safe)
        yield ("S", avail[0] if avail else 5)
