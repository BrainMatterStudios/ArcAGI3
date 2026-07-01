"""re86 paint-to-stencil solver (candidate archetype). L0: 2 single-color pieces (11, 9) + a target stencil
with color-11 and color-9 cells. Move each piece so its (painted) pixels cover its matching-color target cells.
ACTION5 cycles the active piece; ACTION1-4 move it (and paint a trail). Heuristic: for each color, select that
piece and drag it over its target cells; the win check composites pieces vs the stencil.

Engine-truth perception to validate the mechanic first.
"""
from __future__ import annotations
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("re86"))
    env = client.make(game_id=gid, scorecard_id="re86-solve")
    obs = env.reset()
    g = env._game; lvl = g.current_level

    tgt = lvl.get_sprites_by_tag("0054xnsuqceejm")[0]; tp = np.asarray(tgt.pixels)
    need = {}
    for i in range(tp.shape[0]):
        for j in range(tp.shape[1]):
            v = int(tp[i, j])
            if v != -1 and v != 4:
                need.setdefault(v, []).append((tgt.x + j, tgt.y + i))
    tgt_centroid = {c: (np.mean([p[0] for p in ps]), np.mean([p[1] for p in ps])) for c, ps in need.items()}
    if verbose:
        print("target colors + centroids:", {c: (round(x), round(y)) for c, (x, y) in tgt_centroid.items()})

    def pieces():
        out = {}
        for pc in lvl.get_sprites_by_tag("0031cppcuvqlbi"):
            px = np.asarray(pc.pixels)
            cols = [int(v) for v in px.flatten() if v not in (-1, 0)]
            if cols:
                out[max(set(cols), key=cols.count)] = pc
        return out

    def active_color():
        # active piece has a color-0 cursor cell
        for pc in lvl.get_sprites_by_tag("0031cppcuvqlbi"):
            px = np.asarray(pc.pixels)
            if (px == 0).any():
                cols = [int(v) for v in px.flatten() if v not in (-1, 0)]
                return max(set(cols), key=cols.count) if cols else None
        return None

    def piece_centroid(color):
        pc = pieces().get(color)
        if pc is None:
            return None
        px = np.asarray(pc.pixels); ys, xs = np.where(px == color)
        return (pc.x + xs.mean(), pc.y + ys.mean())

    def step(a):
        nonlocal obs
        obs = env.step(GameAction.from_id(a))

    lv = int(obs.levels_completed or 0)
    n = 0
    for color in (11, 9):
        # select the piece of this color via ACTION5 cycling
        for _ in range(4):
            if active_color() == color:
                break
            step(5); n += 1
        # drag it toward the target-color centroid
        tx, ty = tgt_centroid[color]
        for _ in range(40):
            pcen = piece_centroid(color)
            if pcen is None:
                break
            dx, dy = tx - pcen[0], ty - pcen[1]
            if abs(dx) + abs(dy) < 2:
                break
            a = (4 if dx > 0 else 3) if abs(dx) >= abs(dy) else (2 if dy > 0 else 1)
            step(a); n += 1
            if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
                print(f"  *** re86 L0 SOLVED (paint heuristic, {n} actions) ***"); return int(obs.levels_completed or 0)
            if obs.state == GameState.GAME_OVER:
                print("  GAME_OVER"); return 0
    if verbose:
        print(f"  did not solve in {n} actions (levels {obs.levels_completed})")
    return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
