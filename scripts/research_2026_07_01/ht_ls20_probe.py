"""ht_ls20_probe.py — Investigate ls20 rotation state & frame observability.

Crux questions:
 1. What is the win condition? (already read: bejndxqqzf = shape&color&rotation match at goal)
 2. Is rotation state EVER reflected in the rendered frame? Prove by comparing
    the htkmubhry (state indicator) sprite pixels across the 4 rotation values,
    per shape.
 3. Is the goal-rotation indicator (srgbthxut) frame-visible too?
"""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('ls20'))
print("game_id:", gid)
env = c.make(game_id=gid, scorecard_id='x')
obs = env.reset()
g = env._game
print("avail actions:", obs.available_actions, "state:", obs.state, "levels_completed:", obs.levels_completed)
print("current_level_index:", g._current_level_index)
print("dhksvilbb (rotations):", g.dhksvilbb)
print("StartRotation:", g.current_level.get_data("StartRotation"), "GoalRotation:", g.current_level.get_data("GoalRotation"))
print("StartShape:", g.current_level.get_data("StartShape"), "StartColor:", g.current_level.get_data("StartColor"))
print("cklxociuu (rot idx):", g.cklxociuu, "fwckfzsyc (shape):", g.fwckfzsyc, "hiaauhahz (color):", g.hiaauhahz)
print("goal target rot idx ehwheiwsk:", g.ehwheiwsk, " target shape ldxlnycps:", g.ldxlnycps, " target color yjdexjsoa:", g.yjdexjsoa)

htk = g.htkmubhry
print("\nhtkmubhry pos:", htk.x, htk.y, "scale:", getattr(htk, 'scale', None), "visible:", htk.is_visible, "size:", htk.width, htk.height)

# --- ROTATION VISIBILITY TEST on the state-indicator sprite htkmubhry ---
# For each of the 6 shapes, render the sprite at each of the 4 rotations and
# compare pixel arrays. If any two rotations differ, rotation is frame-visible
# for that shape.
print("\n=== ROTATION VISIBILITY: htkmubhry.render() per shape x rotation ===")
import copy
for shape_idx in range(len(g.ijessuuig)):
    htk.pixels = g.ijessuuig[shape_idx].pixels.copy()
    renders = []
    for rot in g.dhksvilbb:
        htk.set_rotation(rot)
        renders.append(np.array(htk.render()))
    # pairwise distinct count
    distinct = []
    for r in renders:
        if not any(np.array_equal(r, d) for d in distinct):
            distinct.append(r)
    print(f"shape {shape_idx}: {len(distinct)} distinct renders across 4 rotations "
          f"(shapes {'ROTATION-VISIBLE' if len(distinct) > 1 else 'rotationally SYMMETRIC (invisible)'})")

# Which shape is actually used in L0 (StartShape) and at goals?
print("\nStartShape idx:", g.current_level.get_data("StartShape"))
print("Goal shape idxs (ldxlnycps):", g.ldxlnycps)
