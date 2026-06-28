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

SIZE = 64

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
                 block=None, block_color=0, marker=None, marker_color=0):
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
        kind = action[0]
        if kind == "M":
            self._move(action[1], action[2])
            if self.family == "COLLECT":
                self.items.discard(self.avatar)  # pick up item on contact
        elif kind == "C":
            x, y = action[1], action[2]      # x=col, y=row
            if self.family == "GATE" and self.switch is not None and (y, x) == self.switch:
                self.gate_open = True        # click the switch -> open the gate
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
        return 0.0

    # --- rendering ---
    def frame(self):
        g = np.full((self.size, self.size), self.bg, dtype=np.int8)
        for (r, c) in self.walls:
            g[r, c] = self.wall
        for (r, c) in self.items:
            g[r, c] = self.item_color
        if self.switch is not None:
            g[self.switch] = self.switch_color
        if self.marker is not None:
            g[self.marker] = self.marker_color
        if self.block is not None:
            g[self.block] = self.block_color
        if self.target is not None:
            g[self.target] = self.target_color
        if self.avatar is not None:
            g[self.avatar] = self.avatar_color
        return g
