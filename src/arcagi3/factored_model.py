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
