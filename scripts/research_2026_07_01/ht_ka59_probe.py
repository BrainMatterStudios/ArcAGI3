import numpy as np, copy
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))
def fresh():
    env=c.make(game_id=gid, scorecard_id='x'); env.reset(); return env
env=fresh(); g=env._game; lvl=g.current_level
# boundary pixels
b=lvl.get_sprites_by_tag("0029ifoxxfvvvs")[0]
px=b.pixels
print("0029 shape",px.shape,"unique",np.unique(px))
print("border rows nonneg count interior:", (px[1:-1,1:-1]>=0).sum(), "total>=0", (px>=0).sum())
# level dims
print("level width/height attr:", getattr(lvl,'width',None), getattr(lvl,'height',None))
print("camera", g.camera)
# test deepcopy
try:
    g2=copy.deepcopy(g); print("deepcopy OK")
except Exception as e:
    print("deepcopy FAIL", e)
# helper: block positions of tag 0022
def blocks(gg):
    return sorted((s.x,s.y) for s in gg.current_level.get_sprites_by_tag("0022vrxelxosfy"))
def sel(gg):
    return (gg.prkgpeyexo.x,gg.prkgpeyexo.y)
def settle(env, maxp=30):
    g=env._game
    for _ in range(maxp):
        if not g.lphmmaeepj and not g.hrknegnjkg: break
        env.step(GameAction.ACTION1)
    return
print("\n--- single action probes from reset ---")
for a in [1,2,3,4]:
    env=fresh(); g=env._game
    print("before blocks",blocks(g),"sel",sel(g),"steps",g.urgssjskot.current_steps)
    obs=env.step(GameAction.from_id(a)); settle(env)
    print(" ACTION%d -> blocks"%a,blocks(env._game),"sel",sel(env._game),"steps",env._game.urgssjskot.current_steps,"state",obs.state,"lvl",obs.levels_completed)
# click select block B
env=fresh(); g=env._game
# need display coords: grid_to_display? use camera.display_to_grid inverse via letterbox ~9
# block B at grid (18,21). Try clicking; find display coord by brute force
from arcengine import GameAction as GA
found=None
for dx in range(0,64):
  for dy in range(0,64):
    r=g.camera.display_to_grid(dx,dy)
    if r and r[0]==19 and r[1]==22:
        found=(dx,dy);break
  if found:break
print("display for grid(19,22):",found)
