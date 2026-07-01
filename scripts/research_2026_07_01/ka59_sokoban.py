"""ka59 discrete-sokoban solver (third archetype). Empirically: 1 ACTION = 1 cell (3px), select block by
click, win = each player block nested centered in a target frame (block anchor == frame anchor + 1px). Build
an exact forward model (move selected block 1 cell if free; blocks + walls are obstacles), BFS over
(selected, block-positions) to nest all blocks, execute (click-select + moves). Validated against the engine.

Perception here uses engine ground truth to validate the MODEL+PLANNER first (as we did for wa30); source-free
role perception is the follow-up once the mechanic is proven.
"""
from __future__ import annotations
from collections import deque
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

CELL = 3


def _rects(sprites):
    return [(s.x, s.y, s.x + s.width, s.y + s.height) for s in sprites]


def _overlap(x, y, rects, w=CELL, h=CELL):
    for (x0, y0, x1, y1) in rects:
        if x < x1 and x + w > x0 and y < y1 and y + h > y0:
            return True
    return False


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ka59"))
    env = client.make(game_id=gid, scorecard_id="ka59-sok")
    obs = env.reset()
    g = env._game; lvl = g.current_level

    blocks = [(s.x, s.y) for s in lvl.get_sprites_by_tag("0022vrxelxosfy")]
    frames = [(s.x, s.y) for s in lvl.get_sprites_by_tag("0010xzmuziohuf")]
    targets = [(fx + 1, fy + 1) for (fx, fy) in frames]   # nested-centered anchor for a 3x3 block in 5x5 frame
    block_tags = {"0022vrxelxosfy"}
    frame_tags = {"0010xzmuziohuf", "0027jbgxilrocf"}
    # wall PIXELS = solid (non-transparent) pixels of collidable sprites that are not blocks/frames
    wall_px = set()
    import numpy as _np
    for s in lvl.get_sprites():
        if not getattr(s, "is_collidable", False):
            continue
        if any(t in block_tags or t in frame_tags for t in s.tags):
            continue
        px = _np.asarray(s.pixels)
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                if px[i, j] != -1:
                    wall_px.add((s.x + j, s.y + i))   # x=col, y=row
    if verbose:
        print(f"blocks={blocks} targets={targets} wall_pixels={len(wall_px)}")

    DIRS = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}

    def blocked(x, y, others):
        # block footprint = 3x3 pixels at (x,y); blocked if any footprint pixel hits a wall or another block
        for dxp in range(CELL):
            for dyp in range(CELL):
                if (x + dxp, y + dyp) in wall_px:
                    return True
        for (ox, oy) in others:
            if abs(x - ox) < CELL and abs(y - oy) < CELL:
                return True
        return False

    def is_win(bpos):
        # every target has a block on it
        return all(any(b == t for b in bpos) for t in targets)

    # BFS over (selected_idx, block positions). Actions: select(i) [click], move selected [1 cell].
    start = (0, tuple(blocks))
    if is_win(start[1]):
        return 0
    seen = {start}
    q = deque([(start, [])])
    plan = None
    while q and plan is None:
        (sel, bpos), path = q.popleft()
        # select a different block
        for i in range(len(bpos)):
            if i != sel:
                ns = (i, bpos)
                if ns not in seen:
                    seen.add(ns); q.append((ns, path + [("select", i)]))
        # move selected block
        for a, (dx, dy) in DIRS.items():
            x, y = bpos[sel]
            nx, ny = x + dx, y + dy
            others = [b for j, b in enumerate(bpos) if j != sel]
            if blocked(nx, ny, others):
                continue
            nb = tuple(nx if j == sel and False else (nx, ny) if j == sel else b for j, b in enumerate(bpos))
            nb = tuple((nx, ny) if j == sel else b for j, b in enumerate(bpos))
            ns = (sel, nb)
            if ns not in seen:
                if is_win(nb):
                    plan = path + [("move", a)]; break
                seen.add(ns); q.append((ns, path + [("move", a)]))
    if plan is None:
        print("  no plan found (may need push-chains)"); return 0
    if verbose:
        print(f"  plan: {len(plan)} steps, {sum(1 for s in plan if s[0]=='move')} moves")

    # execute — clicks are in DISPLAY coords; invert the camera (display_to_grid) to hit a block's game cell
    cam = g.camera
    _disp_cache = {}
    def game_to_display(gx, gy):
        if (gx, gy) in _disp_cache:
            return _disp_cache[(gx, gy)]
        for dx in range(64):
            for dy in range(64):
                gp = cam.display_to_grid(dx, dy)
                if gp and gp[0] == gx and gp[1] == gy:
                    _disp_cache[(gx, gy)] = (dx, dy); return (dx, dy)
        return None
    def click_block(idx):
        bx, by = [(s.x, s.y) for s in lvl.get_sprites_by_tag("0022vrxelxosfy")][idx]
        disp = game_to_display(bx + 1, by + 1)   # block center in game -> display
        if disp is None:
            return obs
        return env.step(GameAction.ACTION6, data={"x": disp[0], "y": disp[1]})

    lv = int(obs.levels_completed or 0)
    msel, mbpos = 0, tuple(blocks)   # model state, simulated alongside
    def eng_blocks():
        return [(s.x, s.y) for s in lvl.get_sprites_by_tag("0022vrxelxosfy")]
    def eng_sel():
        s = getattr(g, "prkgpeyexo", None); return (s.x, s.y) if s else None
    for step_i, (kind, arg) in enumerate(plan):
        if kind == "select":
            msel = arg
            obs = click_block(arg)
        else:
            dx, dy = DIRS[arg]
            x, y = mbpos[msel]; nx, ny = x + dx, y + dy
            others = [b for j, b in enumerate(mbpos) if j != msel]
            if not blocked(nx, ny, others):
                mbpos = tuple((nx, ny) if j == msel else b for j, b in enumerate(mbpos))
            obs = env.step(GameAction.from_id(arg))
        eb = eng_blocks()
        model_sel_pos = mbpos[msel]
        if verbose and (kind == "select" or sorted(eb) != sorted(mbpos)):
            print(f"  step {step_i} {kind} {arg}: model={mbpos} sel_idx={msel} model_sel_pos={model_sel_pos} "
                  f"| engine={eb} engine_sel={eng_sel()}")
        if sorted(eb) != sorted(mbpos):
            print(f"  >>> DIVERGENCE at step {step_i}: engine blocks {eb} != model {list(mbpos)}"); break
        if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
            print(f"  *** ka59 L0 SOLVED (sokoban model + BFS) ***"); return int(obs.levels_completed or 0)
        if obs.state == GameState.GAME_OVER:
            print("  GAME_OVER during execution"); break
    print(f"  did not solve (levels {obs.levels_completed})"); return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
