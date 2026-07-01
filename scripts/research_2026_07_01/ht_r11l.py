"""Frames-ONLY solver for ARC-AGI-3 game r11l, level 0.

MECHANIC (reverse-engineered from environment_files/r11l/495a7899/r11l.py):
  r11l is CLICK-ONLY (ACTION6). Each "group" has:
    - N draggable HANDLE sprites (roefwulewcui-*), colored, click-selectable.
    - a MOVER sprite (roefwu-*) that is auto-positioned at the CENTROID of its handles.
    - a GOAL region (flkdtg-*).
  A click either (a) SELECTS a handle if it lands on one, or (b) DRAGS the currently
  selected handle so its CENTER moves to the clicked cell (blocked only by walls
  wakneh-*). Moving a handle shifts the centroid, i.e. moves the mover.
  WIN a level == every group's mover overlaps its goal. Level 0 = one group,
  two handles; get their centroid (the mover) onto the 7x7 goal.

FRAMES-ONLY DETECTION (no engine introspection):
  color 6  -> the single mover-center pixel.
  color 15 -> goal diamond + mover block + handle centers; goal = largest color-15
              component that does NOT contain the mover pixel (goal doesn't move).
  color 3  -> the (currently un-selected) handle diamond.
  Initially exactly one handle is pre-selected, so the first DRAG needs no location;
  the second handle is located as the color-3 blob, clicked to select, then dragged.

ATTACK (L0): split the goal center G into two symmetric drop points
  T1=(Gx-5,Gy), T2=(Gx+5,Gy) whose average is G. Drag handle A to T1, select
  handle B (color-3 blob), drag it to T2 -> centroid == G -> mover on goal -> WIN.
Deterministic + resettable + MAX-over-runs scoring => search-then-replay is valid.

RUN:
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 .venv/bin/python \
      scripts/research_2026_07_01/ht_r11l.py
"""
from __future__ import annotations

import numpy as np

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

MOVER_COLOR = 6
MARK_COLOR = 15   # goal outline + handle/mover centers
UNSEL_HANDLE_COLOR = 3


def _grid(obs):
    return P.to_grid(obs.frame)


def _pixels(grid, color):
    ys, xs = np.where(grid == color)
    return list(zip(xs.tolist(), ys.tolist()))  # (x, y)


def mover_center(grid):
    px = _pixels(grid, MOVER_COLOR)
    if not px:
        return None
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    return (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys))))


def goal_center(grid):
    """Largest color-15 connected component that does not contain the mover pixel."""
    m = mover_center(grid)
    comps = [o for o in P.connected_components(grid) if o.color == MARK_COLOR]
    if not comps:
        return None

    def has_mover(o):
        if m is None:
            return False
        r0, c0, r1, c1 = o.bbox
        return c0 <= m[0] <= c1 and r0 <= m[1] <= r1

    cand = [o for o in comps if not has_mover(o)] or comps
    g = max(cand, key=lambda o: o.size)
    cr, cc = g.centroid  # (row=y, col=x)
    return (int(round(cc)), int(round(cr)))


def unselected_handle_center(grid):
    """Centroid of the color-3 (un-selected) handle blob, as (x, y)."""
    comps = [o for o in P.connected_components(grid) if o.color == UNSEL_HANDLE_COLOR]
    if not comps:
        return None
    h = max(comps, key=lambda o: o.size)
    cr, cc = h.centroid
    return (int(round(cc)), int(round(cr)))


def click(env, x, y):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})


def solve_level0(env, obs, log):
    grid = _grid(obs)
    G = goal_center(grid)
    if G is None:
        log.append("no goal found")
        return obs
    gx, gy = G
    T1 = (gx - 5, gy)
    T2 = (gx + 5, gy)
    log.append(f"goal_center={G} -> T1={T1} T2={T2}")

    # 1) drag the pre-selected handle onto T1
    obs = click(env, *T1)
    log.append(f"click T1 {T1} -> state={obs.state} levels={obs.levels_completed}")

    # 2) locate the still-un-selected handle (color 3) and click it to SELECT
    grid = _grid(obs)
    hb = unselected_handle_center(grid)
    if hb is None:
        log.append("no un-selected handle found for step 2")
        return obs
    obs = click(env, *hb)
    log.append(f"select handle B at {hb} -> state={obs.state} levels={obs.levels_completed}")

    # 3) drag handle B onto T2 -> centroid == G -> mover on goal
    obs = click(env, *T2)
    log.append(f"click T2 {T2} -> state={obs.state} levels={obs.levels_completed}")
    return obs


def main():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("r11l"))
    env = c.make(game_id=gid, scorecard_id="x")
    obs = env.reset()
    log = []
    obs = solve_level0(env, obs, log)
    print("\n".join(log))
    won = obs.levels_completed >= 1
    print(f"\nRESULT: levels_completed={obs.levels_completed} state={obs.state} "
          f"L0_SOLVED={'YES' if won else 'NO'}")
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
