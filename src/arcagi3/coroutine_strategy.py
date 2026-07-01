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
            if tok is None:
                return self._benign(available)
            # SAFETY: never emit a click when ACTION6 is unavailable (the engine returns a null frame ->
            # would crash the harness, e.g. tn36). Substitute a benign action; the search/plan just no-ops.
            if tok[0] == "C" and 6 not in available:
                return self._benign(available)
            return tok
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


# ---------------------------------------------------------------------------
# GOAL-CLASS: centroid-drag (click-drag handles to move a MOVER onto a GOAL).
# Frames-only role detection: color 6 = mover point; color 15 = goal region (the
# largest color-15 blob NOT containing the mover); color 3 = an un-selected handle.
# One handle is pre-selected; split the goal G into T1=(Gx-5,Gy), T2=(Gx+5,Gy) whose
# average is G: drag handle A to T1, select handle B (color-3), drag it to T2 ->
# centroid == G -> mover on goal. Family-specific colors, but additive + floor-safe.
# ---------------------------------------------------------------------------
_MOVER, _GOALC, _HANDLE = 6, 15, 3


def _mover_center(grid):
    ys, xs = np.where(grid == _MOVER)
    return (int(round(xs.mean())), int(round(ys.mean()))) if len(ys) else None


def _goal_center(grid):
    m = _mover_center(grid)
    comps = [o for o in P.connected_components(grid) if o.color == _GOALC]
    if not comps:
        return None
    def has_m(o):
        if m is None:
            return False
        r0, c0, r1, c1 = o.bbox
        return c0 <= m[0] <= c1 and r0 <= m[1] <= r1
    cand = [o for o in comps if not has_m(o)] or comps
    g = max(cand, key=lambda o: o.size)
    return (int(round(g.centroid[1])), int(round(g.centroid[0])))


def _unsel_handle(grid):
    comps = [o for o in P.connected_components(grid) if o.color == _HANDLE]
    if not comps:
        return None
    h = max(comps, key=lambda o: o.size)
    return (int(round(h.centroid[1])), int(round(h.centroid[0])))


def centroid_drag_gen(obs0):
    """try the centroid-drag class; RETURN (fall through) if roles absent or it doesn't win."""
    g0 = obs0["grid"]; avail = list(obs0["available"])
    if 6 not in avail:
        return
    if _mover_center(g0) is None or _goal_center(g0) is None or _unsel_handle(g0) is None:
        return
    G = _goal_center(g0)
    if G is None:
        return
    gx, gy = G
    yield ("reset",); obs = yield ("reset",)          # double-reset -> fresh clean run
    obs = yield ("C", gx - 5, gy)                      # drag pre-selected handle A to T1
    if obs["levels"] >= 1:
        while True:
            yield ("S", avail[0] if avail else 5)
    hb = _unsel_handle(obs["grid"])
    if hb is not None:
        obs = yield ("C", hb[0], hb[1])               # select handle B
    obs = yield ("C", gx + 5, gy)                      # drag handle B to T2 -> centroid = G
    if obs["levels"] >= 1:
        while True:
            yield ("S", avail[0] if avail else 5)
    return


# ---------------------------------------------------------------------------
# GOAL-CLASS: pull-drag (a click grabs a nearby block and pulls it toward the click;
# undo = ACTION7). Frames-only: TARGET = the color-9 region, BLOCK = a small color-15
# object. Greedy closed loop: click a point on the block->target line within the grab
# radius, re-perceive, undo on stall, until the block center lands in the target box.
# ---------------------------------------------------------------------------
_PULL_TARGET, _PULL_BLOCK, _GRAB = 9, 15, 8


def _pull_target(grid):
    comps = [o for o in P.connected_components(grid) if o.color == _PULL_TARGET and o.size >= 20]
    if not comps:
        return None
    g = max(comps, key=lambda o: o.size)
    r0, c0, r1, c1 = g.bbox
    return (int(round(g.centroid[1])), int(round(g.centroid[0])), (c0, r0, c1, r1))


def _pull_block(grid, tcx, tcy, prev=None):
    comps = [o for o in P.connected_components(grid) if o.color == _PULL_BLOCK and 4 <= o.size <= 20]
    if not comps:
        return None
    if prev is not None:                              # track by CONTINUITY (nearest to last position)
        b = min(comps, key=lambda o: abs(o.centroid[1] - prev[0]) + abs(o.centroid[0] - prev[1]))
    else:                                             # first frame: the block farthest from the target
        b = max(comps, key=lambda o: abs(o.centroid[1] - tcx) + abs(o.centroid[0] - tcy))
    return (int(round(b.centroid[1])), int(round(b.centroid[0])))


def pull_drag_gen(obs0):
    """try the pull-drag class; RETURN (fall through) if target/block absent or it doesn't win."""
    g0 = obs0["grid"]; avail = list(obs0["available"])
    if 6 not in avail or _pull_target(g0) is None:
        return
    yield ("reset",); obs = yield ("reset",)          # clean run
    prev = None
    for _ in range(40):
        g = obs["grid"]
        tgt = _pull_target(g)
        if tgt is None:
            return
        tcx, tcy, (bx0, by0, bx1, by1) = tgt
        blk = _pull_block(g, tcx, tcy, prev)
        if blk is None:
            return
        bcx, bcy = blk; prev = blk
        if bx0 <= bcx <= bx1 and by0 <= bcy <= by1:
            return                                    # already inside (no win) -> fall through
        dx, dy = tcx - bcx, tcy - bcy
        dist = max(1, abs(dx) + abs(dy))
        step = min(_GRAB, dist)
        cx = int(bcx + dx * step / dist); cy = int(bcy + dy * step / dist)
        before = (bcx, bcy)
        obs = yield ("C", max(0, min(63, cx)), max(0, min(63, cy)))
        if obs["levels"] >= 1:
            while True:
                yield ("S", avail[0] if avail else 5)
        nb = _pull_block(obs["grid"], tcx, tcy, before)
        if nb is not None:
            prev = nb
            if abs(nb[0] - before[0]) + abs(nb[1] - before[1]) < 1 and 7 in avail:
                obs = yield ("S", 7)                  # stalled -> free undo
    return


# ---------------------------------------------------------------------------
# GOAL-CLASS: reproduce-a-reference-via-a-palette (pattern-match, e.g. sb26). Affordance-
# grounded, frames-only: PALETTE = selector cells (two-phase probing), SLOTS = applier
# cells, TARGET = the colored group in the palette's colors that is neither. Read the
# target sequence (reading order), then per slot: select the target color, apply; submit.
# ---------------------------------------------------------------------------
def template_goal_gen(obs0):
    g0 = obs0["grid"]; avail = list(obs0["available"])
    if 6 not in avail:
        return
    targets = _salient(g0, 16)
    base = {}
    for (cx, cy) in targets:
        yield ("reset",)
        obs = yield ("C", cx, cy)
        if obs["levels"] >= 1:
            return                                    # single-click win -> let generic path handle it
        base[(cx, cy)] = 0 if obs["terminal"] else _delta(obs["grid"], g0)
    selectors, appliers = set(), set()
    cand = [t for t in targets if base.get(t, 99) < 999]
    for s in cand:
        for t in cand:
            if t == s:
                continue
            yield ("reset",)
            obs = yield ("C", *s)
            if obs["terminal"] or obs["levels"] >= 1:
                continue
            b = obs["grid"]
            obs = yield ("C", *t)
            d = 0 if obs["terminal"] else _delta(obs["grid"], b)
            if obs["levels"] >= 1 or (d >= 2 and abs(d - base.get(t, 0)) >= 2):
                selectors.add(s); appliers.add(t)
    if not selectors or not appliers:
        return
    sel_color = {s: int(g0[s[1], s[0]]) for s in selectors}
    palette = set(sel_color.values())
    slots = sorted(appliers, key=lambda t: (t[1] // 6, t[0]))
    bg = P.detect_background(g0)
    objs = []
    for o in P.connected_components(g0, background=bg):
        if o.color not in palette:
            continue
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        if any(abs(cx - s[0]) + abs(cy - s[1]) <= 3 for s in selectors):
            continue
        if any(abs(cx - t[0]) + abs(cy - t[1]) <= 3 for t in appliers):
            continue
        objs.append((o.color, cy, cx))
    target_seq = [o[0] for o in sorted(objs, key=lambda o: (o[1] // 6, o[2]))]
    if len(slots) < 2 or len(target_seq) < len(slots) or any(c == 0 for c in target_seq[:len(slots)]):
        return
    target_seq = target_seq[:len(slots)]
    yield ("reset",); obs = yield ("reset",)          # clean run
    for i, slot in enumerate(slots):
        sel = next((s for s in selectors if sel_color[s] == target_seq[i]), None)
        if sel is None:
            continue
        yield ("C", *sel)
        obs = yield ("C", slot[0], slot[1])
        if obs["levels"] >= 1:
            while True:
                yield ("S", avail[0] if avail else 5)
    if 5 in avail:
        obs = yield ("S", 5)                          # submit
        if obs["levels"] >= 1:
            while True:
                yield ("S", avail[0] if avail else 5)
    return


def cheap_classes_gen(obs0):
    """FAST goal-class solvers only (frame-detect, no probing/search): peg-solitaire, centroid-drag, pull-drag.
    Each returns in 0 actions if its roles are absent, so this is cheap on non-matching games -> safe to run
    EARLY in the portfolio (before the expensive coverage explorers) so class games are solved fast at eval
    where actions go over a slow gateway. Abstains (benign) if nothing matches."""
    yield from peg_solitaire_gen(obs0)
    yield from centroid_drag_gen(obs0)
    yield from pull_drag_gen(obs0)
    return          # no class matched -> RETURN (generator exhausts) so the portfolio fast-rotates in ~1 action
    # (on a WIN, the matching class holds forever via its own benign loop and never reaches here)


def general_agent_gen(obs0):
    """game-agnostic search-replay solver as a generator (frames+feedback only, zero per-game code).
    Tries known goal-CLASSES (peg / centroid-drag / pull-drag / template-match), then generic search."""
    yield from peg_solitaire_gen(obs0)          # returns (falls through) if not a peg game / DFS exhausts
    yield from centroid_drag_gen(obs0)          # returns (falls through) if not a centroid-drag game
    yield from pull_drag_gen(obs0)              # returns (falls through) if not a pull-drag game
    yield from template_goal_gen(obs0)          # returns (falls through) if not a reproduce-via-palette game
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
