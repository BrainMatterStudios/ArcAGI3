"""Phase Q v5 GATE-DETECTION diagnostic: drive the known ls20 solution and, at the goal-adjacent
frame (and at wall-blocked frames along the way), dump everything a "goal-like object" detector could
key on — so the detector is chosen against real pixels, not abstraction.

Per step it prints: engine avatar (x,y,rot) + cklxociuu, whether the move was BLOCKED (avatar didn't
translate), a global color inventory (to separate floor/wall/goal colors), the connected components in
the goal region (the framed structure), and — for each of the 4 move directions — the colors in the
target band one pitch (5 cells) away from the avatar footprint. No assertions; pure observation.
"""
from __future__ import annotations

import logging
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from arcagi3 import perception as P  # noqa: E402

logging.basicConfig(level=logging.ERROR)
ED = "environment_files"
SOLUTION = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]
PITCH = 5
# action id -> (drow, dcol) per the ground-truth memory (grid deltas, pitch 5)
DIR = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

# goal region window (ground truth: goal at grid ~(12,34), framed color-5 block)
GR0, GR1, GC0, GC1 = 6, 22, 28, 44


def fmt_window(grid, bg, r0, r1, c0, c1):
    lines = ["      " + "".join(f"{c % 10}" for c in range(c0, c1))]
    for r in range(r0, r1):
        cells = "".join("." if grid[r, c] == bg else f"{grid[r, c]:x}" for c in range(c0, c1))
        lines.append(f"r{r:>3} {cells}")
    return "\n".join(lines)


def avatar_cells(grid, ax, ay):
    """Engine avatar anchor (x,y) -> grid footprint cells. grid_row=y+2, col=x; sprite is 5x5."""
    r0, c0 = ay + 2, ax
    cells = []
    for r in range(r0, r0 + 5):
        for c in range(c0, c0 + 5):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                cells.append((r, c))
    return cells


def band_colors(grid, cells, drow, dcol, bg):
    """Colors in the band one pitch away from the avatar footprint in direction (drow,dcol)."""
    cnt = Counter()
    for (r, c) in cells:
        tr, tc = r + drow * PITCH, c + dcol * PITCH
        if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]:
            v = int(grid[tr, tc])
            if v != bg:
                cnt[v] += 1
    return dict(cnt)


def main():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ED,
                    logger=logging.getLogger("diag"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id="gatediag")
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    bg = P.detect_background(grid)
    g = env._game
    s = g.gudziatsk
    goal_color = g.current_level.get_data("GoalColor")
    print(f"bg={bg}  GoalColor={goal_color}  GoalRotation={g.current_level.get_data('GoalRotation')}")
    print(f"engine goal sprite tag rjlbuycveu present; avatar start x={s.x} y={s.y}")
    inv = Counter(int(v) for v in grid.flatten())
    print(f"global color inventory (color:count): {dict(sorted(inv.items()))}")

    for i, a in enumerate(SOLUTION):
        ax0, ay0 = s.x, s.y
        ck_before = g.cklxociuu
        obs = env.step(GameAction.from_id(a))
        grid = P.to_grid(obs.frame)
        moved = (s.x, s.y) != (ax0, ay0)
        blocked = "" if moved else "  <<< BLOCKED (avatar did not translate)"
        flip = "  cklxociuu FLIP" if g.cklxociuu != ck_before else ""
        print(f"\n--- step {i} act={a} dir={DIR.get(a)}  avatar({ax0},{ay0})->({s.x},{s.y}) "
              f"rot{s.rotation} ck{ck_before}->{g.cklxociuu}{flip} lvl={obs.levels_completed} "
              f"state={obs.state.name}{blocked}")
        # per-direction target band from the avatar's CURRENT footprint
        cells = avatar_cells(grid, s.x, s.y)
        for aid, (dr, dc) in DIR.items():
            bc = band_colors(grid, cells, dr, dc, bg)
            tag = " <== this move's dir" if aid == a else ""
            print(f"    dir act={aid} {(dr,dc)}: band_colors={bc}{tag}")

    # final goal-region structure
    print("\n=== goal-region window (final frame) ===")
    print(fmt_window(grid, bg, GR0, GR1, GC0, GC1))
    print("  connected components in goal region (color,size,bbox):")
    for o in sorted(P.connected_components(grid, background=bg), key=lambda o: (o.color, o.size)):
        r0, c0, r1, c1 = o.bbox
        if r1 >= GR0 and r0 <= GR1 and c1 >= GC0 and c0 <= GC1:
            print(f"    color={o.color:>2} size={o.size:>3} bbox={o.bbox}")


if __name__ == "__main__":
    main()
