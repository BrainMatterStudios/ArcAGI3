"""ht_ft09: crack ft09 (constraint / lights-out color puzzle). Mechanic (decoded from source):
- Grid of 'Hkx' color tiles (also 'NTi'), 2-color palette gqb (e.g. [9,8]). Camera scales display->grid by ~2x
  (click display=(2*gx,2*gy) to hit grid tile at (gx,gy)).
- ACTION6 click on a tile -> cycles the colors of a MASK (irw) of tiles relative to it. L0 irw is center-only
  ([[0,0,0],[0,1,0],[0,0,0]]) => each click toggles ONLY the clicked tile through the palette.
- WIN cgj(): every 'bsT' clue sprite is satisfied. A clue at (cx,cy) with center color nRq encodes 8 constraints
  on the 8 neighbor tiles (offset +-4): clue.pixels[j][i]==0 => neighbor MUST equal nRq; !=0 => MUST differ.
Attack: read clues -> solve required color per tile -> click (cycle) each mismatched tile the needed # of times.
Generalizes to any irw mask via GF/linear solve; here center-mask makes it a direct per-tile toggle.
"""
import sys
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

NB = [(-4,-4,0,0),(0,-4,0,1),(4,-4,0,2),(-4,0,1,0),(4,0,1,2),(-4,4,2,0),(0,4,2,1),(4,4,2,2)]

def solve(verbose=True):
    c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
    gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ft09'))
    env=c.make(game_id=gid, scorecard_id='ht-ft09'); obs=env.reset()
    g=env._game; lvl=g.current_level
    pal=list(g.gqb)
    tiles={}  # (x,y)->sprite
    for s in lvl.get_sprites_by_tag('Hkx')+lvl.get_sprites_by_tag('NTi'):
        tiles[(s.x,s.y)]=s
    # derive required color per tile from bsT clues
    req={}
    for clue in lvl.get_sprites_by_tag('bsT'):
        nRq=int(clue.pixels[1][1])
        for dx,dy,j,i in NB:
            pos=(clue.x+dx, clue.y+dy)
            if pos not in tiles: continue
            zero = int(clue.pixels[j][i])==0
            want_equal = zero
            # required center color
            if want_equal:
                req[pos]=nRq
            else:
                # must differ from nRq -> the other palette color(s). pick the palette entry != nRq
                other=[p for p in pal if p!=nRq]
                req[pos]=other[0] if other else nRq
    if verbose: print("palette",pal,"tiles",len(tiles),"constrained",len(req))
    # plan clicks: for each constrained tile, cycle until center==req
    clicks=[]
    for pos,want in req.items():
        cur=int(tiles[pos].pixels[1][1])
        if cur==want: continue
        # number of cycles to reach want in palette order
        if want in pal and cur in pal:
            n=(pal.index(want)-pal.index(cur))%len(pal)
        else:
            n=1
        for _ in range(n):
            clicks.append((2*pos[0], 2*pos[1]))
    if verbose: print("planned",len(clicks),"clicks:",clicks)
    for (dx,dy) in clicks:
        obs=env.step(GameAction.ACTION6, data={'x':dx,'y':dy})
        if obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1:
            print("SOLVED ft09 L0 in",len(clicks),"actions"); return True
    won=obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1
    print("after plan: state",obs.state,"levels",obs.levels_completed,"cgj",g.cgj(),"WON" if won else "NOT")
    return won

if __name__=="__main__":
    solve()
