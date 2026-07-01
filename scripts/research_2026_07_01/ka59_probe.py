"""Probe ka59's actual mechanic in-engine to verify the decode BEFORE building a planner:
- dump objects (blocks vs target frames) and their sizes (frame accepts block of frame.w-2 x frame.h-2)
- does ACTION6 click select a player block (recolor to 0)?
- does ACTION1-4 move the selected block, and does pushing another block impart MOMENTUM (slide >1 cell)?

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/research_2026_07_01/ka59_probe.py
"""
import logging
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)
chars = "0123456789ABCDEF"


def dump(g, label):
    print(f"--- {label} ---")
    for row in g[::2, ::2]:
        print("".join(chars[v] if 0 <= v < 16 else "?" for v in row))


def objs_of(g, bg):
    os = P.connected_components(g, background=bg)
    return sorted(os, key=lambda o: -o.size)


client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ka59"))
env = client.make(game_id=gid, scorecard_id="ka59-probe")
obs = env.reset(); g = P.to_grid(obs.frame); bg = P.detect_background(g)
print(f"{gid}: avail={obs.available_actions} win_levels={obs.win_levels} bg={bg}")
dump(g, "reset")
print("\nobjects (color,size,bbox=(r0,c0,r1,c1),wxh):")
for o in objs_of(g, bg):
    if o.size <= 2:  # skip 1-cell noise
        continue
    print(f"  color={o.color:2d} size={o.size:3d} bbox={o.bbox} {o.width}x{o.height} centroid=({o.centroid[0]:.1f},{o.centroid[1]:.1f})")

# candidate player blocks vs frames: frames are hollow (border only); blocks are filled.
# Try clicking the centroid of each non-background compact object and see if a recolor-to-0 happens.
print("\n== probe ACTION6 click = select? ==")
for o in objs_of(g, bg)[:8]:
    if o.size <= 2:
        continue
    cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
    before0 = int((g == 0).sum())
    obs = env.step(GameAction.ACTION6, data={"x": cx, "y": cy}); g2 = P.to_grid(obs.frame)
    after0 = int((g2 == 0).sum())
    d = int((g2 != g).sum())
    print(f"  click color={o.color} @({cx},{cy}): changed_cells={d} color0 {before0}->{after0} "
          f"levels={obs.levels_completed} state={obs.state}")
    g = g2

# now probe movement of the selected block + momentum
print("\n== probe ACTION1-4 move (selected block) ==")
for a in (1, 2, 3, 4, 1, 2, 3, 4):
    before = g.copy()
    obs = env.step(GameAction.from_id(a)); g = P.to_grid(obs.frame)
    changed = int((g != before).sum())
    # locate the color-0 selected marker movement
    b0 = np.where(before == 0); a0 = np.where(g == 0)
    mv = None
    if len(b0[0]) and len(a0[0]):
        mv = (round(a0[0].mean()-b0[0].mean(), 1), round(a0[1].mean()-b0[1].mean(), 1))
    print(f"  ACTION{a}: changed={changed} sel(color0) move={mv} levels={obs.levels_completed} state={obs.state}")
    if obs.state in (GameState.WIN, GameState.GAME_OVER):
        print(f"   (terminal {obs.state})"); break
