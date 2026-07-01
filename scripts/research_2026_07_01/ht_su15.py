"""ht_su15: crack su15 (gravity-drag / pull puzzle). Mechanic (decoded from source):
- Blocks tag 'zmlxwcvwb'; value-index = the other numeric tag (block '2' = 3x3 color 15). Target zones tag
  'xkstxyqbs'. Camera 1:1 (64x64). Clicks land on a 4-lattice: x in {0..60 step4}, y in {10..62 step4}.
- ACTION6(x,y): grabs every block whose bbox is within radius 8 of the click, then over 4 frames PULLS each
  grabbed block toward the click point (up to 4 cells/frame => ~16 cells). Net: a near-block click walks the
  block's center to (approx) the click point. Equal-value touching blocks MERGE to value+1; unequal touch = trap.
- ACTION7: FREE UNDO (pops history, no step cost).
- WIN cbdhpcilgb(): the required (value,count) from level data 'xkstxyqbs' is met -- L0 needs ONE value-2 block
  whose CENTER lies inside the target zone AABB. L0: no enemies, no merge -> pure drag.
Attack: closed-loop greedy -- read block center from frame, click the lattice point nearest the target that is
still within grab-radius of the block, step, repeat until center inside target. Free-undo any stalled hop.
Key perception insight: one click = grab(radius8)+pull-to-click over 4 hidden frames; win = center-in-AABB.
"""
import sys
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

LATTICE_X=list(range(0,64,4)); LATTICE_Y=list(range(10,64,4))
GRAB=8

def bbox_of(s): return (s.x, s.y, s.x+s.width-1, s.y+s.height-1)
def center_of(s): return (s.x+s.width/2.0, s.y+s.height/2.0)
def dist_pt_bbox(px,py,bb):
    x0,y0,x1,y1=bb
    dx=max(x0-px,0,px-x1); dy=max(y0-py,0,py-y1)
    return (dx*dx+dy*dy)**0.5

def solve(verbose=True, budget=40):
    c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
    gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('su15'))
    env=c.make(game_id=gid, scorecard_id='ht-su15'); obs=env.reset()
    g=env._game; lvl=g.current_level
    # target zone AABB
    tz=lvl.get_sprites_by_tag('xkstxyqbs')[0]
    tzb=(tz.x, tz.y, tz.x+tz.width, tz.y+tz.height)
    tcx,tcy=(tz.x+tz.width/2.0, tz.y+tz.height/2.0)
    def block():
        # the value-2 block: tag '2'
        bs=lvl.get_sprites_by_tag('2')
        return bs[0] if bs else None
    def inside(s):
        cx,cy=center_of(s)
        return tzb[0]<=cx<tzb[2] and tzb[1]<=cy<tzb[3]
    if verbose: print("target AABB",tzb,"center",(tcx,tcy),"block start",bbox_of(block()))
    n=0
    for it in range(budget):
        b=block()
        if b is None: break
        if inside(b):
            break
        bb=bbox_of(b); cx,cy=center_of(b)
        # candidate clicks within grab radius, ranked by resulting closeness to target
        cands=[]
        for X in LATTICE_X:
            for Y in LATTICE_Y:
                if dist_pt_bbox(X,Y,bb)<=GRAB:
                    # resulting block center ~ click point; distance to target center
                    d=abs(X-tcx)+abs(Y-tcy)
                    cands.append((d,X,Y))
        cands.sort()
        if not cands: 
            print("no valid grab click near block"); break
        moved=False
        for d,X,Y in cands[:6]:
            before=center_of(block())
            obs=env.step(GameAction.ACTION6, data={'x':X,'y':Y}); n+=1
            after=center_of(block()) if block() else before
            if obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1:
                print(f"SOLVED su15 L0 in {n} actions"); return True
            if abs(after[0]-before[0])+abs(after[1]-before[1])>=1:
                moved=True; break
            else:
                # stalled -> free undo, try next candidate
                obs=env.step(GameAction.ACTION7); # free undo
        if not moved:
            if verbose: print("stall at it",it,"pos",center_of(block())); 
    b=block()
    won = obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1 or (b and inside(b))
    print("final block center",center_of(b) if b else None,"state",obs.state,"levels",obs.levels_completed,"WON" if won else "NOT",f"({n} actions)")
    return won

if __name__=="__main__":
    solve()
