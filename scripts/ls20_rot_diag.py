"""Phase Q ls20 PERCEPTION DIAGNOSTIC (rotation/rot-tile): drive the known solution on the offline
engine and, per step, print engine ground truth (avatar sprite x/y/rotation, cklxociuu) ALONGSIDE the
rendered grid pixels and connected_components output near the rot tile. Goal: SEE exactly what the
frame looks like at the moment the avatar steps on the rot tile, so a visit detector can be designed
against real data rather than abstraction. No assertions — pure observation harness.
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402

logging.basicConfig(level=logging.ERROR)
ED = "environment_files"
SOLUTION = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]

# Window around the rot tile (grid row 32, col 19) — show context.
R0, R1, C0, C1 = 28, 38, 14, 26


def fmt_window(grid, bg):
    lines = ["      " + "".join(f"{c % 10}" for c in range(C0, C1))]
    for r in range(R0, R1):
        cells = "".join("." if grid[r, c] == bg else f"{grid[r, c]:x}" for c in range(C0, C1))
        lines.append(f"r{r:>3} {cells}")
    return "\n".join(lines)


def objs_near(grid, bg):
    out = []
    for o in P.connected_components(grid, background=bg):
        r0, c0, r1, c1 = o.bbox
        if r1 >= R0 and r0 <= R1 and c1 >= C0 and c0 <= C1:
            out.append(o)
    return out


def main():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ED,
                    logger=logging.getLogger("diag"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id="diag")
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    bg = P.detect_background(grid)
    g = env._game
    start_idx = g.dhksvilbb.index(g.current_level.get_data("StartRotation"))
    print(f"bg={bg}  StartRotation idx={start_idx}  GoalRotation={g.current_level.get_data('GoalRotation')}  "
          f"dhksvilbb={g.dhksvilbb}")
    s = g.gudziatsk
    print(f"initial avatar sprite: x={s.x} y={s.y} rot={s.rotation} cklxociuu={g.cklxociuu}")
    print("=== initial window (hex color, '.'=bg) ===")
    print(fmt_window(grid, bg))

    for i, a in enumerate(SOLUTION):
        ck_before = g.cklxociuu
        sx, sy, srot = s.x, s.y, s.rotation
        obs = env.step(GameAction.from_id(a))
        grid = P.to_grid(obs.frame)
        ck_after = g.cklxociuu
        flipped = "  <<< cklxociuu FLIPPED" if ck_after != ck_before else ""
        print(f"\n--- step {i} action={a}  avatar({sx},{sy})rot{srot} -> ({s.x},{s.y})rot{s.rotation}  "
              f"cklxociuu {ck_before}->{ck_after}{flipped}  level={obs.levels_completed} state={obs.state.name}")
        print(fmt_window(grid, bg))
        objs = objs_near(grid, bg)
        print(f"  objs near window ({len(objs)}):")
        for o in sorted(objs, key=lambda o: (o.color, o.size)):
            print(f"    color={o.color:>2} size={o.size:>3} bbox={o.bbox} centroid=({o.centroid[0]:.1f},{o.centroid[1]:.1f})")


if __name__ == "__main__":
    main()
