"""ht_ls20_frame.py — Prove rotation is visible in the ACTUAL 64x64 frame.

The engine builds each frame with camera.render(current_level.get_sprites()),
then applies interface overlays. We force each rotation value into the engine,
rebuild the frame exactly as the engine does, and diff.
"""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('ls20'))
env = c.make(game_id=gid, scorecard_id='x')
obs = env.reset()
g = env._game

def build_frame():
    return np.array(g.camera.render(g.current_level.get_sprites()))

full = {}
for rot_idx in range(4):
    g.cklxociuu = rot_idx
    g.htkmubhry.set_rotation(g.dhksvilbb[rot_idx])
    full[rot_idx] = build_frame()

print("frame shape:", full[0].shape)
base = full[3]  # StartRotation idx3 (270)
for rot_idx in range(4):
    diff = np.argwhere(full[rot_idx] != base)
    if len(diff):
        print(f"rot idx {rot_idx} (angle {g.dhksvilbb[rot_idx]}): {len(diff)} px differ from start(idx3); "
              f"bbox rows {diff[:,0].min()}-{diff[:,0].max()} cols {diff[:,1].min()}-{diff[:,1].max()}")
    else:
        print(f"rot idx {rot_idx} (angle {g.dhksvilbb[rot_idx]}): IDENTICAL to start(idx3)")

# Show the indicator patch (htkmubhry at engine (3,55) scale2 -> frame rows ~55-60, cols ~3-8)
print("\n=== htkmubhry indicator patch [rows 54-62, cols 2-12] per rotation ===")
for rot_idx in range(4):
    patch = full[rot_idx][54:63, 2:13]
    print(f"\n-- rotation idx {rot_idx} (angle {g.dhksvilbb[rot_idx]}) --")
    for row in patch:
        print("".join(f"{int(v):3d}" if v >= 0 else "  ." for v in row))

# Also confirm the GOAL indicator srgbthxut renders the TARGET rotation distinctly.
print("\n=== goal indicator srgbthxut ===")
gi = g.srgbthxut[0]
print("goal indicator pos:", gi.x, gi.y, "target rot idx:", g.ehwheiwsk[0], "angle:", g.dhksvilbb[g.ehwheiwsk[0]])
