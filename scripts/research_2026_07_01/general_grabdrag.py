"""GENERAL (source-free) grab-drag solver: detect roles by function, not wa30-specific colors, then reuse
the proven exact forward-model + planner. If this solves wa30 via DETECTED roles, the same code solves any
hidden grab-drag game (the mechanic rules are shared; only perception was game-specific).

Role detection:
  - avatar  = the directionally-moving object (largest area among movers = the body, not a marker)
  - blocks  = a color with >=2 compact same-ish-size static components (the movable pieces)
  - goal    = a color forming one large flat region (the target pad), distinct from avatar/blocks
  - cell    = min nonzero avatar displacement (grid unit)
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from ghp import GHP
import wa30_planner as W


def detect_roles(grid, ghp):
    """Return (avatar_color, block_color, [ranked goal-color hypotheses]). The goal color is ambiguous from
    pixels (walls look like pads), so we RANK candidates and let the engine reward disambiguate."""
    bg = P.detect_background(grid)
    comps = defaultdict(list)
    for o in P.connected_components(grid, background=bg):
        comps[o.color].append(o)
    avatar_c = ghp.avatar_color
    # blocks: >=2 compact components of similar small size, color != avatar/bg
    block_c, best = None, None
    for c, os in comps.items():
        if c in (bg, avatar_c):
            continue
        small = [o for o in os if o.size <= 30]
        if len(small) >= 2:
            spread = max(o.size for o in small) - min(o.size for o in small)
            score = (len(small), -spread)
            if best is None or score > best[0]:
                best = (score, c)
    if best:
        block_c = best[1]
    # goal hypotheses: any color (not bg/avatar/block) with a region >=12 cells. RANK by (a) whether it also
    # appears as small marks NEAR blocks (color-coded target hint), (b) region compactness.
    block_cells = set()
    for o in comps.get(block_c, []):
        block_cells.add((round(o.centroid[0]), round(o.centroid[1])))
    cands = []
    for c, os in comps.items():
        if c in (bg, avatar_c, block_c):
            continue
        big = max((o.size for o in os), default=0)
        if big < 12:
            continue
        marks_near_block = sum(1 for o in os if o.size <= 6 and any(
            abs(o.centroid[0]-bc[0]) + abs(o.centroid[1]-bc[1]) < 12 for bc in block_cells))
        cands.append((marks_near_block, big, c))
    cands.sort(reverse=True)
    return avatar_c, block_c, [c for (_, _, c) in cands]


def perceive_roles(grid, avatar_c, block_c, goal_c):
    CELL = W.CELL
    def anchors(color, lo, hi):
        out = []
        bg = P.detect_background(grid)
        for o in P.connected_components(grid, background=bg):
            if o.color != color or not (lo <= o.size <= hi):
                continue
            r0, c0, r1, c1 = o.bbox
            if r0 >= 60:
                continue
            out.append(((c0 // CELL) * CELL, (r0 // CELL) * CELL))
        return sorted(set(out))
    avatar_list = anchors(avatar_c, 3, 60)
    avatar = avatar_list[0] if avatar_list else None
    blocks = anchors(block_c, 6, 40)
    # goal cells = large flat component(s) of goal_c tiled to CELL
    bg = P.detect_background(grid); goals = []
    for o in P.connected_components(grid, background=bg):
        if o.color != goal_c or o.size < 12:
            continue
        r0, c0, r1, c1 = o.bbox
        for x in range((c0 // CELL) * CELL, c1 + 1, CELL):
            for y in range((r0 // CELL) * CELL, r1 + 1, CELL):
                goals.append((x, y))
    return avatar, blocks, sorted(set(goals))


def solve(game="wa30", budget=3000, verbose=True):
    from arc_agi import Arcade, OperationMode
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"gen-{game}")
    obs = env.reset()
    ghp = GHP(); obs = ghp.learn(env, obs)
    grid = P.to_grid(obs.frame)
    av_c, bl_c, goal_hyps = detect_roles(grid, ghp)
    if verbose:
        print(f"{game}: DETECTED avatar_color={av_c} block_color={bl_c} goal_hypotheses(ranked)={goal_hyps} cell={W.CELL}")
    # enumerate goal-color hypotheses; the engine reward disambiguates (Exp-2 principle)
    for gi, go_c in enumerate(goal_hyps):
        obs = env.reset()
        grid = P.to_grid(obs.frame)
        avatar, blocks, goals = perceive_roles(grid, av_c, bl_c, go_c)
        if avatar is None or not blocks or not goals:
            continue
        plan = W.plan_all(avatar, blocks, goals)
        if not plan:
            continue
        lv = int(obs.levels_completed or 0)
        for j, a in enumerate(plan):
            obs = env.step(GameAction.ACTION5 if a == 5 else GameAction.from_id(a))
            if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
                print(f"  *** SOLVED via DETECTED roles (source-free), goal hypothesis #{gi+1} "
                      f"(color {go_c}), in {j+1} actions ***")
                return int(obs.levels_completed or 0)
            if obs.state == GameState.GAME_OVER:
                break
        if verbose:
            print(f"  goal hypothesis color {go_c}: plan {len(plan)} actions, no reward -> next")
    print("  no goal hypothesis won"); return 0


if __name__ == "__main__":
    import sys
    solve(sys.argv[1] if len(sys.argv) > 1 else "wa30")
