"""Cleaner ka59 mechanic probe: click one candidate block to select it, then issue single moves and track
which OBJECT actually moves and by how many cells (momentum = slide > 1 cell). Establishes the control model
before any planner.
Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/research_2026_07_01/ka59_probe2.py [clickx clicky]
"""
import logging, sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)


def obj_centroids(g, bg):
    d = {}
    for o in P.connected_components(g, background=bg):
        if o.size >= 6:
            d.setdefault(o.color, []).append((round(o.centroid[0], 1), round(o.centroid[1], 1), o.size))
    return d


client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ka59"))
env = client.make(game_id=gid, scorecard_id="ka59-probe2")
obs = env.reset(); g = P.to_grid(obs.frame); bg = P.detect_background(g)

# click a color-4 block (candidate player block) in the left room
cx, cy = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (13, 34)
print(f"click-select @({cx},{cy})")
obs = env.step(GameAction.ACTION6, data={"x": cx, "y": cy}); g = P.to_grid(obs.frame)
print("objects after select (color -> [(cy,cx,size)]):")
for c, lst in sorted(obj_centroids(g, bg).items()):
    print(f"  {c}: {lst}")

names = {1: "up", 2: "down", 3: "left", 4: "right"}
for a in (4, 4, 4, 2, 2, 1, 3):  # push right repeatedly (detect momentum), then other dirs
    before = obj_centroids(g, bg)
    obs = env.step(GameAction.from_id(a)); g = P.to_grid(obs.frame)
    after = obj_centroids(g, bg)
    moved = []
    for c in after:
        b = before.get(c, []); a2 = after.get(c, [])
        # match by nearest size, report max centroid shift for this color
        for (cy2, cx2, s2) in a2:
            best = None
            for (cy1, cx1, s1) in b:
                dd = abs(cy2-cy1)+abs(cx2-cx1)
                if best is None or dd < best[0]:
                    best = (dd, cy1, cx1)
            if best and best[0] > 0.4:
                moved.append((c, s2, f"({best[1]},{best[2]})->({cy2},{cx2}) d={best[0]:.1f}"))
    print(f"ACTION{a}({names[a]}): levels={obs.levels_completed} state={obs.state} moved={moved}")
    if obs.state in (GameState.WIN, GameState.GAME_OVER):
        print(f"  terminal {obs.state}"); break
