"""Factored state + induced transition model + BFS planner. Mirrors Exp-42 (truemodel_planner)
but runs over the INDUCED model (deltas + on_enter_cycle tiles + walls + terminal predicate),
not the env source. State = (agent pos, attr dict, completed-slot set)."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FactoredState:
    pos: tuple
    attrs: tuple  # ((name, value), ...) sorted; dict accepted in ctor below
    completed: frozenset

    def __init__(self, pos, attrs, completed):
        object.__setattr__(self, "pos", tuple(pos))
        object.__setattr__(self, "attrs", tuple(sorted(attrs.items())) if isinstance(attrs, dict) else tuple(sorted(attrs)))
        object.__setattr__(self, "completed", frozenset(completed))

    @property
    def attr_dict(self) -> dict:
        return dict(self.attrs)


@dataclass
class InducedModel:
    deltas: dict          # action_id -> (dr, dc)
    walls: set            # blocked (r, c)
    tiles: dict           # (r, c) -> OnEnterCycle
    width: int
    height: int
    terminal: object      # TerminalPredicate

    def step(self, state: FactoredState, action: int) -> FactoredState:
        dr, dc = self.deltas.get(action, (0, 0))
        nr, nc = state.pos[0] + dr, state.pos[1] + dc
        if not (0 <= nr < self.height and 0 <= nc < self.width) or (nr, nc) in self.walls:
            return state  # blocked -> no move
        attrs = state.attr_dict
        tile = self.tiles.get((nr, nc))
        if tile is not None:
            cur = attrs.get(tile.attribute)
            if cur in tile.order:
                i = tile.order.index(cur)
                attrs[tile.attribute] = tile.order[(i + 1) % len(tile.order)]
        return FactoredState(pos=(nr, nc), attrs=attrs, completed=state.completed)


def _satisfied(state: FactoredState, slots: list[dict]) -> frozenset:
    done = set(state.completed)
    for i, s in enumerate(slots):
        if i in done:
            continue
        if state.pos == s["pos"] and all(state.attr_dict.get(k) == v for k, v in s["attr_req"].items()):
            done.add(i)
    return frozenset(done)


def plan(model: InducedModel, start: FactoredState, slots: list[dict], max_nodes: int = 200_000):
    """BFS over the factored state; goal = all slots satisfied. Returns action list or None."""
    start = FactoredState(start.pos, start.attr_dict, _satisfied(start, slots))
    if len(start.completed) == len(slots):
        return []
    seen = {(start.pos, start.attrs, start.completed)}
    q = deque([(start, [])])
    expanded = 0
    while q and expanded < max_nodes:
        st, path = q.popleft()
        expanded += 1
        for a in model.deltas:
            nxt = model.step(st, a)
            nxt = FactoredState(nxt.pos, nxt.attr_dict, _satisfied(nxt, slots))
            if len(nxt.completed) == len(slots):
                return path + [a]
            k = (nxt.pos, nxt.attrs, nxt.completed)
            if k not in seen:
                seen.add(k)
                q.append((nxt, path + [a]))
    return None


def plan_painted_set(*, deltas, true_walls, paintable_cells, tiles, width, height,
                     start_pos, start_attrs, slots, max_paints=None, max_nodes=200_000):
    """A2 backend: BFS over (pos, attrs, completed, painted). Paintable cells must be painted
    (by entering, subject to max_paints) to be traversed; true_walls always block. Tiles cycle
    attributes exactly as InducedModel.step does. Returns (actions|None, status) with status in
    {"solved","no_solution","intractable"}.

    Painting happens on move-INTO a cell, so the start cell is NOT pre-painted; re-entering the
    start cell later costs one paint from the budget.
    Precedence: a cell in both `true_walls` and `paintable_cells` is treated as a hard wall
    (true_walls checked first)."""
    def attrs_tuple(d):
        return tuple(sorted(d.items()))

    def satisfied(pos, ad, completed):
        done = set(completed)
        for i, s in enumerate(slots):
            if i in done:
                continue
            if pos == s["pos"] and all(ad.get(k) == v for k, v in s["attr_req"].items()):
                done.add(i)
        return frozenset(done)

    start_ad = dict(start_attrs)
    start_completed = satisfied(start_pos, start_ad, frozenset())
    if len(start_completed) == len(slots):
        return [], "solved"
    start = (start_pos, attrs_tuple(start_ad), start_completed, frozenset())
    seen = {start}
    q = deque([(start_pos, start_ad, start_completed, frozenset(), [])])
    expanded = 0
    while q:
        if expanded >= max_nodes:
            return None, "intractable"
        pos, ad, completed, painted, path = q.popleft()
        expanded += 1
        for a, (dr, dc) in deltas.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < height and 0 <= nc < width) or (nr, nc) in true_walls:
                continue
            npainted = painted
            if (nr, nc) in paintable_cells and (nr, nc) not in painted:
                if max_paints is not None and len(painted) >= max_paints:
                    continue  # out of paint -> cannot enter
                npainted = painted | {(nr, nc)}
            nad = dict(ad)
            tile = tiles.get((nr, nc))
            if tile is not None and nad.get(tile.attribute) in tile.order:
                i = tile.order.index(nad[tile.attribute])
                nad[tile.attribute] = tile.order[(i + 1) % len(tile.order)]
            ncompleted = satisfied((nr, nc), nad, completed)
            if len(ncompleted) == len(slots):
                return path + [a], "solved"
            key = ((nr, nc), attrs_tuple(nad), ncompleted, npainted)
            if key not in seen:
                seen.add(key)
                q.append(((nr, nc), nad, ncompleted, npainted, path + [a]))
    return None, "no_solution"
