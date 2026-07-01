"""GENERAL source-free PATTERN-MATCH solver (second archetype). Structure shared by pattern-match games:
a TARGET sequence (answer key), a set of empty SLOTS, and a PALETTE/tray of colored tiles. Solve = for each
slot in reading order, click the tile whose color matches the target, click the slot, then submit.

Perception is by band/role (top=answer key, middle=slots, bottom=tiles), no hardcoded colors. Verified on
sb26 L0; the same recipe transfers to a hidden pattern-match game with this layout family.
"""
from __future__ import annotations
import numpy as np
from arcengine import GameAction, GameState
from arcagi3 import perception as P


def _square(o):
    w = o.bbox[3] - o.bbox[1] + 1
    h = o.bbox[2] - o.bbox[0] + 1
    return h > 0 and 0.5 <= w / h <= 2.0 and o.size >= 8


def perceive(grid):
    bg = P.detect_background(grid)
    objs = P.connected_components(grid, background=bg)
    H = grid.shape[0]
    # 1) palette tiles = square filled blobs in the bottom band
    tiles = sorted((o.centroid[1], o.color, o.centroid)
                   for o in objs if (o.bbox[0] + o.bbox[2]) / 2 > H * 0.78 and o.color != bg and _square(o))
    tile_colors = {c for _, c, _ in tiles}
    # 2) answer key = top-band squares whose color is one of the tile colors (filters structural frame color)
    ak = [(o.centroid[1], o.color) for o in objs
          if (o.bbox[0] + o.bbox[2]) / 2 < H * 0.22 and o.color in tile_colors]
    ak = [c for _, c in sorted(ak)]
    # 3) slots = small middle-band marks; keep the dominant row (cluster by rounded row), ordered by col
    mid = [(o.centroid[0], o.centroid[1], o.centroid) for o in objs
           if H * 0.30 < (o.bbox[0] + o.bbox[2]) / 2 < H * 0.62 and o.size <= 12]
    slots = []
    if mid:
        from collections import Counter
        rows = Counter(round(r / 3) for r, _, _ in mid)
        best_row = rows.most_common(1)[0][0]
        slots = sorted((c, cen) for r, c, cen in mid if round(r / 3) == best_row)
    return ak, tiles, slots


def solve(game="sb26", verbose=True):
    from arc_agi import Arcade, OperationMode
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"pm-{game}")
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    ak, tiles, slots = perceive(grid)
    if verbose:
        print(f"{game}: answer_key={ak}  #tiles={len(tiles)} tile_colors={[c for _,c,_ in tiles]}  #slots={len(slots)}")
    if not ak or not tiles or len(slots) < len(ak):
        print("  pattern-match perception incomplete -> abstain"); return 0

    def click(cen):
        return env.step(GameAction.ACTION6, data={"x": int(round(cen[1])), "y": int(round(cen[0]))})

    lv = int(obs.levels_completed or 0)
    used_tiles = set()
    for i, target in enumerate(ak):
        # find an unused tile whose color matches the target
        ti = next((k for k, (_, col, _) in enumerate(tiles) if col == target and k not in used_tiles), None)
        if ti is None:
            print(f"  no tile for target color {target} -> abstain"); return 0
        used_tiles.add(ti)
        obs = click(tiles[ti][2])                 # select the tile
        obs = click(slots[i][1])                  # place into slot i
        if obs.state == GameState.GAME_OVER:
            break
    obs = env.step(GameAction.ACTION5)            # submit
    if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
        print(f"  *** SOLVED sb26-class via source-free pattern-match ***")
        return int(obs.levels_completed or 0)
    print(f"  did not solve (levels {obs.levels_completed}, state {obs.state})")
    return int(obs.levels_completed or 0)


if __name__ == "__main__":
    import sys
    solve(sys.argv[1] if len(sys.argv) > 1 else "sb26")
