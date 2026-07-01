"""EXPERIMENT #3: general DRAG/CARRY dynamics induction + goal-directed planning (gap: re86 paint, wa30 grab-
drag). No per-game code. The move-based failures need directed planning over induced drag dynamics, not search.

Induction (interventional, frames only):
  1. AVATAR/ACTIVE-PIECE: probe each move action; the object whose centroid translates consistently is the
     controllable. Record per-action (dr,dc).
  2. A5-SELECTION: if ACTION5 is available, test whether it CHANGES which object responds to moves (piece
     cycling, as in re86) -> the game has multiple selectable movables.
  3. TARGETS: salient objects/cells that do NOT move under actions (the destinations).
Planning (goal-directed, greedy): for each movable, select it (cycle A5 if needed), then drag it toward the
COLOR-MATCHED static target using the induced per-action deltas; a moving object that paints/carries covers the
target. This is the human read: "this piece goes on that matching cell."
"""
from __future__ import annotations
import sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P


def _objs(grid):
    bg = P.detect_background(grid)
    out = []
    for o in P.connected_components(grid, background=bg):
        if o.color != bg and o.size >= 3:
            out.append((o.color, o.size, o.centroid, o.bbox))
    return out


def _controllable(env, reset):
    """probe moves; return (per-action delta of the moving object, its color) or None."""
    deltas = {}
    color = None
    for a in (1, 2, 3, 4):
        obs = reset()
        g0 = P.to_grid(obs.frame)
        o0 = {(c): cen for (c, s, cen, bb) in _objs(g0)}
        obs = env.step(GameAction.from_id(a))
        g1 = P.to_grid(obs.frame)
        o1 = {(c): cen for (c, s, cen, bb) in _objs(g1)}
        best = None
        for c in set(o0) & set(o1):
            d = (o1[c][0] - o0[c][0], o1[c][1] - o0[c][1])
            mag = abs(d[0]) + abs(d[1])
            if mag >= 1 and (best is None or mag > best[0]):
                best = (mag, c, d)
        if best:
            deltas[a] = best[2]; color = best[1]
    return (deltas, color) if deltas else (None, None)


def solve(game, verbose=True, budget=200):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"drag-{game}")
    obs = env.reset()
    avail = list(obs.available_actions or [])
    if not any(a in avail for a in (1, 2, 3, 4)):
        if verbose: print(f"{game:>6}: no move actions -> n/a"); return dict(game=game, solved=False)

    def reset():
        nonlocal obs; obs = env.reset(); return obs

    deltas, actor_color = _controllable(env, reset)
    if deltas is None:
        if verbose: print(f"{game:>6}: no controllable object found"); return dict(game=game, solved=False)

    # static targets: salient objects that did NOT move; match movable by color
    obs = env.reset(); g0 = P.to_grid(obs.frame)
    has_a5 = 5 in avail
    # movable colors = actor color (+ any others revealed by A5 cycling)
    def active_color(grid):
        # object nearest a cursor (color 0) if present, else the actor color
        ys, xs = np.where(grid == 0)
        objs = _objs(grid)
        if len(ys) and objs:
            cy, cx = ys.mean(), xs.mean()
            return min(objs, key=lambda o: abs(o[2][0]-cy)+abs(o[2][1]-cx))[0]
        return actor_color

    # infer targets by color: cells of a movable-color that are "static" reference (frame-adjacent / distinct).
    # generic: a target for color c = a small cluster of color c that the movable should reach.
    from collections import defaultdict
    colcells = defaultdict(list)
    bg = P.detect_background(g0)
    fr = (g0 == 4)
    adj = np.zeros_like(fr)
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            adj |= np.roll(np.roll(fr, dr, 0), dc, 1)
    targets = {}
    for c in set(int(v) for v in g0.flatten()):
        if c in (bg, 0, 4): continue
        m = (g0 == c)
        tm = m & adj
        if tm.sum() >= 2:
            ys, xs = np.where(tm); targets[c] = (ys.mean(), xs.mean())

    def step(a):
        nonlocal obs; obs = env.step(GameAction.from_id(a))
    def won(): return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1

    # order movables: cycle through A5 selections, drag active toward its color target
    order = list(targets.keys()) or [actor_color]
    obs = env.reset(); n = 0
    for color in order:
        ty, tx = targets.get(color, (32, 32))
        if has_a5:
            for _ in range(6):
                if active_color(P.to_grid(obs.frame)) == color: break
                step(5); n += 1
                if won(): break
        for _ in range(40):
            if won(): break
            g = P.to_grid(obs.frame)
            # position of the active movable = cursor (0) if present else color centroid
            ys, xs = np.where(g == 0)
            if len(ys): cy, cx = ys.mean(), xs.mean()
            else:
                mm = (g == color)
                if not mm.any(): break
                yy, xx = np.where(mm); cy, cx = yy.mean(), xx.mean()
            dy, dx = ty - cy, tx - cx
            if abs(dy) + abs(dx) < 2: break
            # pick the move action whose delta best reduces (dy,dx)
            best_a = min(deltas, key=lambda a: abs((cy+deltas[a][0])-ty) + abs((cx+deltas[a][1])-tx))
            step(best_a); n += 1
        if n > budget: break
    solved = won()
    if verbose:
        print(f"{game:>6}: actor=c{actor_color} deltas={ {a:deltas[a] for a in deltas} } "
              f"targets={list(targets.keys())} A5={int(has_a5)} -> "
              f"{'SOLVED in '+str(n)+' actions' if solved else 'unsolved'}")
    return dict(game=game, solved=solved, actions=n)


if __name__ == "__main__":
    games = sys.argv[1:] or ["re86", "wa30", "tr87", "tu93", "m0r0"]
    print("General DRAG/CARRY dynamics induction + goal-directed drag (zero per-game code):\n")
    res = []
    for g in games:
        try:
            res.append(solve(g))
        except Exception as e:  # noqa: BLE001
            print(f"{g:>6}: ERROR {type(e).__name__}: {e}"); res.append(dict(game=g, solved=False))
    print(f"\nSUMMARY: solved {sum(r['solved'] for r in res)}/{len(res)}")
