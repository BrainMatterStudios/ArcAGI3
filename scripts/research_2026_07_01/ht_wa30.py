"""ht_wa30: crack wa30 (grab-drag). Mechanic (decoded from source):
- avatar tag 'wbmdvjhthc', cell=4. ACTION1..4 set rotation(facing) via pjedoipwee THEN move 1 cell.
  Rotation updates even if the move is BLOCKED -> you can 'turn to face' by bumping.
- ACTION5: if holding -> release; else grab the block in the FACED-adjacent cell (vwiozbtqgi = rotation+adjacency).
- WIN ymzfopzgbq: every block 'geezpjgiyd' anchor in goal region 'fsjjayjoeg' (wyzquhjerd) AND none held.
Attack: A* over the true forward model (multi-block sokoban-with-carry), executed on the engine.
Frames-only perception insight at bottom.
"""
import sys, heapq
from itertools import permutations
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

CELL=4

def faced(ax,ay,rot):
    if rot==0:  return (ax, ay-CELL)
    if rot==180:return (ax, ay+CELL)
    if rot==90: return (ax+CELL, ay)
    return (ax-CELL, ay)
def rot_of(dx,dy):
    if dy<0: return 0
    if dx>0: return 90
    if dy>0: return 180
    return 270
D={1:(0,-CELL),2:(0,CELL),3:(-CELL,0),4:(CELL,0)}

class Model:
    def __init__(self, walls):
        self.walls=walls  # set of blocked anchor cells (borders + collidables that aren't blocks)
    def step(self, s, a, other_blocks):
        ax,ay,bx,by,rot,held=s
        blocked = self.walls | set(other_blocks)
        if a==5:
            if held: return (ax,ay,bx,by,rot,False)
            if faced(ax,ay,rot)==(bx,by): return (ax,ay,bx,by,rot,True)
            return s
        dx,dy=D[a]
        if not held:
            nr=rot_of(dx,dy); tgt=(ax+dx,ay+dy)
            if tgt not in blocked and tgt!=(bx,by):
                return (tgt[0],tgt[1],bx,by,nr,held)
            return (ax,ay,bx,by,nr,held)  # bumped: face updated, no move
        # held: move avatar+block rigidly
        nav=(ax+dx,ay+dy); nbl=(bx+dx,by+dy)
        if (nav not in blocked or nav==(bx,by)) and (nbl not in blocked or nbl==(ax,ay)):
            return (nav[0],nav[1],nbl[0],nbl[1],rot,held)
        return s

def astar(model, start, goal_xy, other_blocks):
    gx,gy=goal_xy
    def h(s): return (abs(s[2]-gx)+abs(s[3]-gy))//CELL
    def win(s): return (s[2],s[3])==goal_xy and not s[5]
    if win(start): return []
    pq=[(h(start),0,0,start,[])]; best={start:0}; c=0
    while pq:
        f,g,_,s,path=heapq.heappop(pq)
        if g>best.get(s,1<<30): continue
        for a in (1,2,3,4,5):
            ns=model.step(s,a,other_blocks); ng=g+1
            if ns==s and a!=5: 
                # still allow bump-to-turn (changes rot) -> ns differs; if identical skip
                pass
            if ng>=best.get(ns,1<<30): continue
            if win(ns): return path+[a]
            best[ns]=ng; c+=1
            heapq.heappush(pq,(ng+h(ns),ng,c,ns,path+[a]))
            if c>400000: return None
    return None

def plan(avatar, blocks, goals, walls):
    n=len(blocks)
    best=None
    # try assignments block->goal (permutations of goals) and placement orders
    for perm in permutations(range(n)):
        # perm[k] = goal index for block k? we'll iterate order of blocks and assign greedily
        pass
    for order in permutations(range(n)):
        for gperm in permutations(range(n)):
            ax,ay,rot=avatar[0],avatar[1],0
            bpos=list(blocks); placed=[False]*n; full=[]; ok=True
            for k in order:
                gy=goals[gperm[k]]
                others=[bpos[j] for j in range(n) if j!=k]
                m=Model(walls)
                p=astar(m,(ax,ay,bpos[k][0],bpos[k][1],rot,False),gy,others)
                if p is None: ok=False; break
                s=(ax,ay,bpos[k][0],bpos[k][1],rot,False)
                for a in p: s=m.step(s,a,others)
                ax,ay,rot=s[0],s[1],s[4]; bpos[k]=(s[2],s[3]); full+=p
            if ok and (best is None or len(full)<len(best)):
                best=full
        # first order that yields any plan is fine; keep searching for shorter
    return best

def solve(verbose=True):
    c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
    gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('wa30'))
    env=c.make(game_id=gid, scorecard_id='ht-wa30'); obs=env.reset()
    g=env._game; lvl=g.current_level
    av=lvl.get_sprites_by_tag('wbmdvjhthc')[0]
    avatar=(av.x,av.y)
    blocks=[(s.x,s.y) for s in lvl.get_sprites_by_tag('geezpjgiyd')]
    goals=sorted(set((x,y) for (x,y) in g.wyzquhjerd if x%CELL==0 and y%CELL==0))
    # walls = collidable anchors that are NOT blocks + borders (from engine pkbufziase minus block cells)
    walls=set(g.pkbufziase)-set(blocks)
    if verbose: print("avatar",avatar,"blocks",blocks,"goals",goals,"nwalls",len(walls))
    pl=plan(avatar,blocks,goals,walls)
    if pl is None:
        print("NO PLAN"); return False
    if verbose: print("PLAN len",len(pl),pl)
    # execute
    for a in pl:
        obs=env.step(GameAction.from_id(a))
        if obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1:
            print("SOLVED wa30 L0 in",len(pl),"actions (win at step)"); return True
    won = obs.state==GameState.WIN or int(obs.levels_completed or 0)>=1
    print("after plan: state",obs.state,"levels",obs.levels_completed,"WON" if won else "NOT")
    return won

if __name__=="__main__":
    solve()
