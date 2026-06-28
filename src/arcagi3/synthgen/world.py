"""BET 3 Stage 0 — minimal interactive synthetic engine (4 role-randomized mechanic families).

A deterministic 64x64 gridworld for the held-out-family in-context-inference probe. State = static
walls + entities + a hidden mechanic. `step(action)` returns (reward, done); reward=1 on the family's
win condition (a "level-up"). Frames render as palette-index grids (same as perception.to_grid output);
EVERY role colour is randomized per episode (build_*), so colour identity carries zero signal — the
family must be read from interaction DYNAMICS alone.

Worlds are constructed with explicit entity positions for deterministic mechanic tests; the randomized
build_<family>() generators (tested via invariants) produce playable instances for the probe.
"""
from __future__ import annotations

import numpy as np

from . import canvas as cv

SIZE = 64
BSIZE = 16  # compact grid for the probe (tractable solvers + small model input)
FAMILIES = ["REACH", "COLLECT", "GATE", "PUSH", "AVOID", "TOGGLE", "CHASE", "KEYDOOR"]

# action representation: movement ("M",dr,dc), click ("C",x,y), noop ("N",)
UP = ("M", -1, 0)
DOWN = ("M", 1, 0)
LEFT = ("M", 0, -1)
RIGHT = ("M", 0, 1)
NOOP = ("N",)
MOVES = (UP, DOWN, LEFT, RIGHT)


def click(x: int, y: int):
    return ("C", int(x), int(y))


class World:
    def __init__(self, family, *, bg, wall, avatar=None, avatar_color=0,
                 target=None, target_color=0, walls=None, size=SIZE,
                 items=None, item_color=0, switch=None, switch_color=0,
                 block=None, block_color=0, marker=None, marker_color=0,
                 hazard=None, hazard_color=0, flee_dist=8, switches=None,
                 chase_drift=(0, 1), chase_slow=2, key=None, key_color=0,
                 door=None, door_color=0):
        self.family = family
        self.size = size
        self.bg = int(bg)
        self.wall = int(wall)
        self.avatar = tuple(avatar) if avatar is not None else None
        self.avatar_color = int(avatar_color)
        self.target = tuple(target) if target is not None else None
        self.target_color = int(target_color)
        self.walls = set(walls) if walls else set()
        self.items = set(items) if items else set()
        self.item_color = int(item_color)
        self.switch = tuple(switch) if switch is not None else None
        self.switch_color = int(switch_color)
        self.gate_open = False
        self.block = tuple(block) if block is not None else None
        self.block_color = int(block_color)
        self.marker = tuple(marker) if marker is not None else None
        self.marker_color = int(marker_color)
        self.hazard = tuple(hazard) if hazard is not None else None
        self.hazard_color = int(hazard_color)
        self.flee_dist = int(flee_dist)
        self.switches = set(switches) if switches else set()
        self.chase_drift = tuple(chase_drift)
        self.chase_slow = int(chase_slow)
        self.key = tuple(key) if key is not None else None
        self.key_color = int(key_color)
        self.door = tuple(door) if door is not None else None
        self.door_color = int(door_color)
        self.has_key = False
        self._stepc = 0
        self.solved = False

    # --- geometry ---
    def _blocked(self, cell):
        r, c = cell
        if not (0 <= r < self.size and 0 <= c < self.size):
            return True
        return cell in self.walls

    def _move(self, dr, dc):
        if self.avatar is None:
            return
        nr, nc = self.avatar[0] + dr, self.avatar[1] + dc
        dest = (nr, nc)
        if self.family == "PUSH" and dest == self.block:
            bdest = (nr + dr, nc + dc)
            if self._blocked(bdest) or bdest == self.avatar:
                return  # block can't move -> avatar stuck too
            self.block = bdest
            self.avatar = dest
            return
        if not self._blocked(dest):
            self.avatar = dest

    # --- mechanics ---
    def step(self, action):
        if self.solved:
            return 0.0, True
        self._stepc += 1
        kind = action[0]
        if kind == "M":
            self._move(action[1], action[2])
            if self.family == "COLLECT":
                self.items.discard(self.avatar)  # pick up item on contact
            elif self.family == "KEYDOOR" and self.avatar == self.key:
                self.has_key = True              # pick up the key on contact
        elif kind == "C":
            x, y = action[1], action[2]      # x=col, y=row
            if self.family == "GATE" and self.switch is not None and (y, x) == self.switch:
                self.gate_open = True        # click the switch -> open the gate
            elif self.family == "TOGGLE":
                self.switches.discard((y, x))  # click a switch -> it toggles off
        if self.family == "CHASE" and self.target is not None and self._stepc % self.chase_slow == 0:
            dr, dc = self.chase_drift             # the target drifts (clamped to interior)
            self.target = (min(max(self.target[0] + dr, 1), self.size - 2),
                           min(max(self.target[1] + dc, 1), self.size - 2))
        reward = self._check_win()
        return reward, self.solved

    def _check_win(self):
        if self.family == "REACH":
            if self.avatar is not None and self.avatar == self.target:
                self.solved = True
                return 1.0
        elif self.family == "COLLECT":
            if not self.items:
                self.solved = True
                return 1.0
        elif self.family == "GATE":
            if self.gate_open and self.avatar is not None and self.avatar == self.target:
                self.solved = True
                return 1.0
        elif self.family == "PUSH":
            if self.block is not None and self.block == self.marker:
                self.solved = True
                return 1.0
        elif self.family == "AVOID":
            if self.avatar is not None and _manhattan(self.avatar, self.hazard) >= self.flee_dist:
                self.solved = True
                return 1.0
        elif self.family == "TOGGLE":
            if not self.switches:
                self.solved = True
                return 1.0
        elif self.family == "CHASE":
            if self.avatar is not None and self.avatar == self.target:
                self.solved = True
                return 1.0
        elif self.family == "KEYDOOR":
            if self.has_key and self.avatar is not None and self.avatar == self.door:
                self.solved = True
                return 1.0
        return 0.0

    # --- rendering ---
    def frame(self):
        g = np.full((self.size, self.size), self.bg, dtype=np.int8)
        for (r, c) in self.walls:
            g[r, c] = self.wall
        for (r, c) in self.items:
            g[r, c] = self.item_color
        if self.switch is not None and not self.gate_open:
            g[self.switch] = self.switch_color  # switch disappears when clicked -> gate state VISIBLE
        for (r, c) in self.switches:
            g[r, c] = self.switch_color
        if self.hazard is not None:
            g[self.hazard] = self.hazard_color
        if self.key is not None and not self.has_key:
            g[self.key] = self.key_color        # key disappears when picked up -> state VISIBLE
        if self.door is not None:
            g[self.door] = self.door_color
        if self.marker is not None:
            g[self.marker] = self.marker_color
        if self.block is not None:
            g[self.block] = self.block_color
        if self.target is not None:
            g[self.target] = self.target_color
        if self.avatar is not None:
            g[self.avatar] = self.avatar_color
        return g


# ---------------- on-path affordance label + candidates (env-truth, not a demonstrated policy) ----
def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _away(src, hazard, size):
    """The move that most increases Manhattan distance from `hazard` (staying in bounds)."""
    best, bd = None, -1
    for mv in MOVES:
        nr, nc = src[0] + mv[1], src[1] + mv[2]
        if 0 <= nr < size and 0 <= nc < size:
            d = abs(nr - hazard[0]) + abs(nc - hazard[1])
            if d > bd:
                best, bd = mv, d
    return best or UP


def _toward(src, dst):
    dr, dc = dst[0] - src[0], dst[1] - src[1]
    if abs(dr) >= abs(dc):
        if dr != 0:
            return DOWN if dr > 0 else UP
        return RIGHT if dc > 0 else LEFT
    return RIGHT if dc > 0 else LEFT


def rewarding_action(w: "World"):
    """The action on a shortest solving path from w's current state — the mechanic's rewarding
    affordance. The model must INFER this for a held-out family from interaction history."""
    if w.family == "REACH":
        return _toward(w.avatar, w.target)
    if w.family == "COLLECT":
        nearest = min(w.items, key=lambda it: abs(it[0] - w.avatar[0]) + abs(it[1] - w.avatar[1]))
        return _toward(w.avatar, nearest)
    if w.family == "GATE":
        if not w.gate_open:
            return click(w.switch[1], w.switch[0])  # click(x=col, y=row) on the switch
        return _toward(w.avatar, w.target)
    if w.family == "PUSH":
        return _sokoban_first_move(w)         # general solver; None if the state is a deadlock
    if w.family == "AVOID":
        return _away(w.avatar, w.hazard, w.size)
    if w.family == "TOGGLE":
        sw = min(w.switches, key=lambda s: _manhattan(s, w.avatar))
        return click(sw[1], sw[0])
    if w.family == "CHASE":
        return _toward(w.avatar, w.target)
    if w.family == "KEYDOOR":
        return _toward(w.avatar, w.key if not w.has_key else w.door)
    raise ValueError(w.family)


def _push_transition(avatar, block, mv, walls, size):
    """Pure (avatar, block) -> (avatar, block) under one move, with the World's push rule."""
    dr, dc = mv[1], mv[2]
    nav = (avatar[0] + dr, avatar[1] + dc)

    def blocked(c):
        return not (0 <= c[0] < size and 0 <= c[1] < size) or c in walls
    if nav == block:
        nbl = (block[0] + dr, block[1] + dc)
        if blocked(nbl) or nbl == avatar:
            return avatar, block             # block can't move -> avatar stuck too
        return nav, nbl
    if blocked(nav):
        return avatar, block
    return nav, block


def _sokoban_first_move(w, max_expand=40000):
    """BFS over (avatar, block) states for the shortest push of the block onto the marker; returns the
    first move, or None if the state is an unsolvable deadlock (block cornered)."""
    from collections import deque
    start = (w.avatar, w.block)
    if w.block == w.marker:
        return None
    seen = {start}
    q = deque([(start, None)])
    expanded = 0
    while q and expanded < max_expand:
        (av, bl), first = q.popleft()
        expanded += 1
        for mv in MOVES:
            nav, nbl = _push_transition(av, bl, mv, w.walls, w.size)
            ns = (nav, nbl)
            if ns == (av, bl) or ns in seen:
                continue
            f = mv if first is None else first
            if nbl == w.marker:
                return f
            seen.add(ns)
            q.append((ns, f))
    return None


def candidate_actions(w: "World"):
    """The action set scored by the probe: the 4 moves + a click on every distinct entity cell."""
    cands = list(MOVES)
    for cell in [w.target, w.switch, w.block, w.marker, w.hazard, w.key, w.door,
                 *sorted(w.items), *sorted(w.switches)]:
        if cell is not None:
            cands.append(click(cell[1], cell[0]))
    return cands


def _rand_cell(rng, size, taken, lo=2, hi=None):
    hi = size - 3 if hi is None else hi
    while True:
        cell = (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
        if cell not in taken:
            taken.add(cell)
            return cell


def build(family, rng, size=BSIZE):
    """A randomized, label-solvable instance with every role colour drawn from canvas.roles (so
    colour identity carries zero family signal). Open interior (border walls only)."""
    cols = cv.roles(rng, 8)
    bg, wall = cols[0], cols[1]
    border = set()
    for i in range(size):
        border |= {(0, i), (size - 1, i), (i, 0), (i, size - 1)}
    taken = set(border)
    if family == "REACH":
        av = _rand_cell(rng, size, taken)
        tg = _rand_cell(rng, size, taken)
        return World("REACH", bg=bg, wall=wall, walls=border, size=size,
                     avatar=av, avatar_color=cols[2], target=tg, target_color=cols[3])
    if family == "COLLECT":
        av = _rand_cell(rng, size, taken)
        items = {_rand_cell(rng, size, taken) for _ in range(int(rng.integers(3, 6)))}
        return World("COLLECT", bg=bg, wall=wall, walls=border, size=size,
                     avatar=av, avatar_color=cols[2], items=items, item_color=cols[3])
    if family == "GATE":
        av = _rand_cell(rng, size, taken)
        tg = _rand_cell(rng, size, taken)
        sw = _rand_cell(rng, size, taken)
        return World("GATE", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                     avatar_color=cols[2], target=tg, target_color=cols[3],
                     switch=sw, switch_color=cols[4])
    if family == "PUSH":
        for _try in range(500):
            dr, dc = MOVES[int(rng.integers(4))][1:]     # random push direction
            k = int(rng.integers(2, 6))                  # block->marker distance
            br = int(rng.integers(2, size - 2)); bc = int(rng.integers(2, size - 2))
            mr, mc = br + dr * k, bc + dc * k            # marker beyond the block
            pr, pc = br - dr, bc - dc                    # push-from cell
            cells = [(br, bc), (mr, mc), (pr, pc)]
            if not all(2 <= r <= size - 3 and 2 <= c <= size - 3 for r, c in cells):
                continue
            if len({(br, bc), (mr, mc), (pr, pc)}) < 3:
                continue
            av = _rand_cell(rng, size, {(br, bc), (mr, mc), *border})
            return World("PUSH", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                         avatar_color=cols[2], block=(br, bc), block_color=cols[3],
                         marker=(mr, mc), marker_color=cols[4])
        raise RuntimeError("could not place a PUSH instance")
    if family == "AVOID":
        hz = _rand_cell(rng, size, taken)
        # avatar near the hazard so fleeing is non-trivial but solvable on the open grid
        av = _rand_cell(rng, size, taken)
        return World("AVOID", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                     avatar_color=cols[2], hazard=hz, hazard_color=cols[3], flee_dist=int(rng.integers(7, 11)))
    if family == "TOGGLE":
        av = _rand_cell(rng, size, taken)
        sws = {_rand_cell(rng, size, taken) for _ in range(int(rng.integers(2, 5)))}
        return World("TOGGLE", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                     avatar_color=cols[2], switches=sws, switch_color=cols[3])
    if family == "CHASE":
        av = _rand_cell(rng, size, taken)
        tg = _rand_cell(rng, size, taken)
        drift = MOVES[int(rng.integers(4))][1:]
        return World("CHASE", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                     avatar_color=cols[2], target=tg, target_color=cols[3],
                     chase_drift=drift, chase_slow=int(rng.integers(2, 4)))
    if family == "KEYDOOR":
        av = _rand_cell(rng, size, taken)
        ky = _rand_cell(rng, size, taken)
        dr = _rand_cell(rng, size, taken)
        return World("KEYDOOR", bg=bg, wall=wall, walls=border, size=size, avatar=av,
                     avatar_color=cols[2], key=ky, key_color=cols[3], door=dr, door_color=cols[4])
    raise ValueError(family)
