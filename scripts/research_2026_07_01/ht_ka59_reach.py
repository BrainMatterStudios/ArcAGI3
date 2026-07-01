import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))
env=c.make(game_id=gid, scorecard_id='x'); env.reset(); g=env._game; lvl=g.current_level
b=lvl.get_sprites_by_tag("0029ifoxxfvvvs")[0]
px=b.pixels
# print compact map of >=0 region for 0029 (rows y, cols x), origin at b.x,b.y=(-3,-3)
print("0029 origin",b.x,b.y,"shape",px.shape)
rows=[]
for i in range(px.shape[0]):
    rows.append("".join("#" if px[i,j]>=0 else "." for j in range(px.shape[1])))
# print a downsampled view
for i in range(0,px.shape[0]):
    print("%3d "%(b.y+i)+rows[i])
