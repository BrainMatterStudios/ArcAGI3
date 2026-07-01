import logging; logging.basicConfig(level=logging.ERROR)
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ka59"))
env = client.make(game_id=gid, scorecard_id="ka59-map"); obs = env.reset()
g = env._game; lvl = g.current_level
print("0003 destructible:", len(lvl.get_sprites_by_tag("0003umnkyodpjp")),
      " 0001 other:", len(lvl.get_sprites_by_tag("0001uqqokjrptk")),
      " Enemy:", len(lvl.get_sprites_by_tag("Enemy")))
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
def free(x,y): return all((x+a,y+b) not in wall for a in range(CELL) for b in range(CELL))
st=blocks[0]; seen={st}; q=deque([st])
while q:
    x,y=q.popleft()
    for dx,dy in [(0,-3),(0,3),(-3,0),(3,0)]:
        nx,ny=x+dx,y+dy
        if (nx,ny) not in seen and free(nx,ny): seen.add((nx,ny)); q.append((nx,ny))
# render 3-cell map over x,y in 0..48
print("map (.=free #=wall B=block T=target r=reach-from-B0):")
for y in range(0,48,3):
    row=""
    for x in range(0,48,3):
        c='#' if not free(x,y) else '.'
        if (x,y) in seen and c=='.': c='r'
        if (x,y) in targets: c='T'
        if (x,y) in blocks: c='B'
        row+=c
    print(f"y{y:2d} "+row)
