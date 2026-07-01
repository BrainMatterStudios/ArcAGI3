import logging; logging.basicConfig(level=logging.ERROR)
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ka59"))
env = client.make(game_id=gid, scorecard_id="ka59-reach"); obs = env.reset()
g = env._game; lvl = g.current_level
CELL=3
blocks=[(s.x,s.y) for s in lvl.get_sprites_by_tag("0022vrxelxosfy")]
frames=[(s.x,s.y) for s in lvl.get_sprites_by_tag("0010xzmuziohuf")]
targets=[(fx+1,fy+1) for (fx,fy) in frames]
wall=set()
for s in lvl.get_sprites():
    if getattr(s,"is_collidable",False) and not any(t in ("0022vrxelxosfy","0010xzmuziohuf") for t in s.tags):
        px=np.asarray(s.pixels)
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                if px[i,j]!=-1: wall.add((s.x+j,s.y+i))
def free(x,y):
    return all((x+a,y+b) not in wall for a in range(CELL) for b in range(CELL))
print("blocks",blocks,"targets",targets)
for t in targets: print(f"  target {t} footprint free? {free(*t)}")
# single-block reachability from each start (ignore other block)
for bi,st in enumerate(blocks):
    seen={st}; q=deque([st])
    while q:
        x,y=q.popleft()
        for dx,dy in [(0,-3),(0,3),(-3,0),(3,0)]:
            nx,ny=x+dx,y+dy
            if (nx,ny) not in seen and free(nx,ny):
                seen.add((nx,ny)); q.append((nx,ny))
    reach=[t for t in targets if t in seen]
    print(f"  block{bi} @{st}: reaches {len(seen)} cells; targets reachable: {reach}")
