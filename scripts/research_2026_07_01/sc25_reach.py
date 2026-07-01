import logging; logging.basicConfig(level=logging.ERROR)
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("sc25"))
env = client.make(game_id=gid, scorecard_id="sc25-reach"); obs = env.reset()
g = env._game; lvl = g.current_level
av = lvl.get_sprites_by_tag("pluyoo") or [s for s in lvl.get_sprites() if s.name=="pluyoo"]
go = [s for s in lvl.get_sprites() if s.name=="exydhv"][0]
avs = av[0]
# wall pixels from all collidable non-avatar/goal sprites
wall=set()
for s in lvl.get_sprites():
    if getattr(s,"is_collidable",False) and s.name not in ("pluyoo","exydhv"):
        px=np.asarray(s.pixels)
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                if px[i,j]!=-1: wall.add((s.x+j,s.y+i))
print(f"avatar@({avs.x},{avs.y}) {avs.width}x{avs.height}  goal@({go.x},{go.y})  wall_px={len(wall)}")
def free(x,y,w,h):
    return all(0<=x<64 and 0<=y<64 and (x+a,y+b) not in wall for a in range(w) for b in range(h))
for step,(w,h,lbl) in [(2,(4,4,"scale1")),(4,(8,8,"scale2"))]:
    st=(avs.x,avs.y)
    if not free(st[0],st[1],w,h):
        print(f"  {lbl}: avatar start not free at {w}x{h}"); continue
    seen={st}; q=deque([st]); goal_hit=False
    while q:
        x,y=q.popleft()
        # goal reached if avatar box overlaps goal
        if x < go.x+go.width and x+w > go.x and y < go.y+go.height and y+h > go.y:
            goal_hit=True; break
        for dx,dy in [(0,-step),(0,step),(-step,0),(step,0)]:
            nx,ny=x+dx,y+dy
            if (nx,ny) not in seen and free(nx,ny,w,h): seen.add((nx,ny)); q.append((nx,ny))
    print(f"  {lbl} (step {step}, box {w}x{h}): reaches {len(seen)} cells, goal_reachable={goal_hit}")
