"""Probe A6 click->grid mapping, camera offset, grid cell size."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('bp35'))
env = c.make(game_id=gid, scorecard_id='x'); env.reset()
lm = env._game.oztjzzyqoek; tm = lm.hdnrlfmyrj; cam = lm.camera
print("grid_size", getattr(tm,'grid_size',None))
print("camera rczgvgfsfb (scroll):", cam.rczgvgfsfb)
print("cam w,h", cam.width, cam.height)
# hyntnfvpgl: pixel(worldx, worldy) -> grid cell?
for (wx,wy) in [(0,0),(6,6),(18,42),(3*6,23*6)]:
    print("hyntnfvpgl", (wx,wy), "->", tm.hyntnfvpgl(wx,wy))
# test: to break grid cell (gx,gy), need click dx,dy with dx//6==gx, (dy+scrolly)//6==gy
scrolly = cam.rczgvgfsfb[1]
print("scrolly", scrolly)
# player at (3,23); cell in gravity dir (dy=-1) is (3,22) wall. breakable near: (7,19) etc.
# compute display click for a grid cell:
def cell_to_click(gx,gy,scrolly):
    dx = gx*6
    dy = gy*6 - scrolly
    return dx,dy
for cell in [(3,22),(7,19),(5,15)]:
    dx,dy=cell_to_click(*cell,scrolly)
    back = tm.hyntnfvpgl(dx, dy+scrolly)
    print("cell",cell,"-> click",(dx,dy),"-> grid",back, "onscreen" , 0<=dy<64)
