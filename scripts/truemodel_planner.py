"""Experiment 42 — TRUE-model planner vs the 7961-action salience result on ls20.

Decisive test of the review's claim ("only planning failed; discovery succeeded").

We drive ls20's ACTUAL transition function (the env source is in the repo) and run a
breadth-first search over the real symbolic state:

    (agent_x, agent_y, shape_idx, color_idx, rotation_idx, completed_slots)

Dedup is on that key, so the number of nodes BFS expands == the size of the true
reachable state space. If a correct-model search solves level 1 in N <<< 7961 actions
while exploring a tiny state space, the bottleneck is MODEL DISCOVERY, not planning
horsepower — i.e. the 7961 blind actions reflect having NO correct model, not a hard
planning problem.

Ground-truth mechanic (ls20.py): the avatar has a mutable (shape,color,rotation);
stepping on transformer tiles cycles each; a level slot is satisfied by standing on it
with (shape,color,rotation) matching that slot's spec; level done when all slots match.

Usage: PYTHONPATH=src .venv/bin/python scripts/truemodel_planner.py [max_nodes]
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

from arcengine import ActionInput, GameAction, GameState

LS20_PATH = Path("environment_files/ls20/9607627b/ls20.py")
MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


def load_ls20_class():
    spec = importlib.util.spec_from_file_location("ls20_env", LS20_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Ls20


def fresh_game(Ls20):
    g = Ls20()
    g.perform_action(ActionInput(id=GameAction.RESET))  # full_reset -> sets up level 0
    return g


def state_key(g):
    """The real latent state the win predicate (bejndxqqzf/pbznecvnfr) depends on.

    The step counter is included because ls20 L2+ uses step-reset tiles to refuel the
    step budget; without it, BFS prunes post-reset states as 'already seen' and
    incorrectly concludes those levels are unsolvable.
    """
    steps = getattr(getattr(g, "_step_counter_ui", None), "current_steps", None)
    return (
        g.gudziatsk.x,
        g.gudziatsk.y,
        g.fwckfzsyc,                 # shape index
        g.hiaauhahz,                 # color index
        g.cklxociuu,                 # rotation index
        tuple(g.lvrnuajbl),          # which goal slots are already satisfied
        steps,                       # remaining step budget (None if no counter)
    )


def apply(g, action):
    g.perform_action(ActionInput(id=action))
    return g


def bfs_solve(Ls20, start_level=0, max_nodes=200_000):
    root = fresh_game(Ls20)
    start_idx = root.level_index
    seen = {state_key(root)}
    q = deque([(root, [])])
    expanded = 0
    t0 = time.time()
    while q:
        g, path = q.popleft()
        expanded += 1
        if expanded > max_nodes:
            return None, expanded, len(seen), time.time() - t0, "node budget exhausted"
        for a in MOVES:
            child = copy.deepcopy(g)
            apply(child, a)
            # win = advanced past the level we started on
            if child.level_index > start_idx or child._score > root._score:
                return path + [a.value], expanded, len(seen), time.time() - t0, "solved"
            if child._state == GameState.GAME_OVER:
                continue
            k = state_key(child)
            if k in seen:
                continue
            seen.add(k)
            q.append((child, path + [a.value]))
    return None, expanded, len(seen), time.time() - t0, "frontier exhausted (no solution)"


def main():
    max_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    Ls20 = load_ls20_class()

    g = fresh_game(Ls20)
    print("=== ls20 level 1 — true initial state ===", flush=True)
    print(f"  agent pos      : ({g.gudziatsk.x}, {g.gudziatsk.y})", flush=True)
    print(f"  shape/color/rot idx: {g.fwckfzsyc} / {g.hiaauhahz} / {g.cklxociuu}", flush=True)
    print(f"  goal slots     : {len(g.plrpelhym)}  (need shape={g.ldxlnycps}, colorIdx={g.yjdexjsoa}, rotIdx={g.ehwheiwsk})", flush=True)
    print(f"  StartRot={g.current_level.get_data('StartRotation')} GoalRot={g.current_level.get_data('GoalRotation')} "
          f"StartColor={g.current_level.get_data('StartColor')} GoalColor={g.current_level.get_data('GoalColor')} "
          f"StartShape={g.current_level.get_data('StartShape')}", flush=True)

    print(f"\n=== BFS over the true model (max_nodes={max_nodes:,}) ===", flush=True)
    sol, expanded, unique, dt, why = bfs_solve(Ls20, max_nodes=max_nodes)
    print(f"  result        : {why}", flush=True)
    print(f"  nodes expanded: {expanded:,}", flush=True)
    print(f"  unique states : {unique:,}", flush=True)
    print(f"  search time   : {dt:.1f}s", flush=True)
    if sol is not None:
        print(f"  SOLUTION LEN  : {len(sol)} actions   (salience needed 7961)", flush=True)
        print(f"  action seq    : {sol}", flush=True)
    print("\n=== verdict ===", flush=True)
    if sol is not None:
        print(f"  Correct-model BFS solves ls20 L1 in {len(sol)} actions exploring {unique:,} unique states.", flush=True)
        print("  => The planning problem is TINY. The 7961 blind actions reflect having no", flush=True)
        print("     correct model (discovery), not a hard planning problem.", flush=True)


def bfs_solve_current(game, max_nodes=500_000):
    """BFS from the game's CURRENT level/state (deepcopy snapshots from `game`). Returns the
    action list that clears the current level, or None. Mirrors bfs_solve but does not reset."""
    start_idx = game.level_index
    seen = {state_key(game)}
    q = deque([(game, [])])
    expanded = 0
    while q:
        g, path = q.popleft()
        expanded += 1
        if expanded > max_nodes:
            return None
        for a in MOVES:
            child = copy.deepcopy(g)
            apply(child, a)
            if child.level_index > start_idx or child._score > game._score:
                return path + [a.value]
            if child._state == GameState.GAME_OVER:
                continue
            k = state_key(child)
            if k not in seen:
                seen.add(k)
                q.append((child, path + [a.value]))
    return None


def optimal_actions_per_level(GameClass, up_to_level=0, max_nodes=500_000):
    """Per-level A_h via solve->advance: solve the current level, apply that optimal solution to
    reach the next, repeat. Returns [A_h[0], A_h[1], ...] (None for any level the BFS can't solve)."""
    g = GameClass()
    g.perform_action(ActionInput(id=GameAction.RESET))
    out = []
    for _ in range(up_to_level + 1):
        sol = bfs_solve_current(g, max_nodes=max_nodes)
        out.append(None if sol is None else len(sol))
        if sol is None:
            break
        for aid in sol:
            g.perform_action(ActionInput(id=GameAction.from_id(aid)))
    return out


if __name__ == "__main__":
    main()
