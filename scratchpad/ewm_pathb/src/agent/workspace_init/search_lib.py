"""search_lib.py — reusable, correct planners over YOUR world_model_engine.

Provided so you only have to model the game (engine + renderer + initial state + which actions
matter); you should NOT hand-write search. Import these from world_model_main_planner.py.

Your engine must be: world_model_engine(state, action) -> (new_state, status)
  status in {"RUNNING","LEVEL_COMPLETED","GAME_OVER"}; action is {"name": "ACTIONk"} (+ "x","y" for ACTION6).

Because the ACTION6 click space is 64*64, you MUST pass an actions_fn(state)->list[action] that returns only
the SENSIBLE candidate actions in a state (e.g. the 4 moves, ACTION5, and the handful of meaningful click
targets you identified). Good candidate generation is what makes search tractable.
"""
from __future__ import annotations
import json
from collections import deque

MOVES = [{"name": f"ACTION{k}"} for k in (1, 2, 3, 4, 5, 7)]


def _default_key(state):
    """Canonical, hashable key for a world-model state (numpy-aware)."""
    def canon(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, dict):
            return {k: canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [canon(x) for x in o]
        return o
    return json.dumps(canon(state), sort_keys=True, default=str)


def bfs_plan(state, engine, actions_fn=None, *, max_nodes=200_000, max_depth=200, key_fn=None):
    """Shortest action sequence from `state` to a LEVEL_COMPLETED status, or None.

    Prunes GAME_OVER branches and revisited states. `actions_fn(state)` -> candidate actions
    (defaults to the 6 simple moves; you should supply clicks). Deterministic engine assumed.
    """
    actions_fn = actions_fn or (lambda s: MOVES)
    key_fn = key_fn or _default_key
    start_key = key_fn(state)
    seen = {start_key}
    q = deque([(state, [])])
    nodes = 0
    while q:
        cur, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions_fn(cur):
            nodes += 1
            if nodes > max_nodes:
                return None
            try:
                nxt, status = engine(cur, action)
            except Exception:
                continue
            if status == "LEVEL_COMPLETED":
                return path + [action]
            if status == "GAME_OVER":
                continue
            k = key_fn(nxt)
            if k in seen:
                continue
            seen.add(k)
            q.append((nxt, path + [action]))
    return None


def greedy_plan(state, engine, actions_fn, heuristic, *, max_nodes=200_000, max_depth=400, key_fn=None):
    """Best-first search toward LEVEL_COMPLETED using heuristic(state)->float (lower = closer).

    Use when BFS is too wide (large state spaces / long solutions). Not optimal, but fast.
    """
    import heapq
    key_fn = key_fn or _default_key
    seen = {key_fn(state)}
    h0 = heuristic(state)
    counter = 0
    pq = [(h0, 0, counter, state, [])]
    nodes = 0
    while pq:
        _, _, _, cur, path = heapq.heappop(pq)
        if len(path) >= max_depth:
            continue
        for action in actions_fn(cur):
            nodes += 1
            if nodes > max_nodes:
                return None
            try:
                nxt, status = engine(cur, action)
            except Exception:
                continue
            if status == "LEVEL_COMPLETED":
                return path + [action]
            if status == "GAME_OVER":
                continue
            k = key_fn(nxt)
            if k in seen:
                continue
            seen.add(k)
            counter += 1
            heapq.heappush(pq, (heuristic(nxt) + len(path) + 1, len(path) + 1, counter, nxt, path + [action]))
    return None
