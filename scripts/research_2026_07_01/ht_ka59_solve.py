import copy, collections
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))

def make():
    env=c.make(game_id=gid, scorecard_id='x'); env.reset(); return env

# build grid->display lookup for clicks
env0=make(); g0=env0._game
grid2disp={}
for dx in range(64):
    for dy in range(64):
        r=g0.camera.display_to_grid(dx,dy)
        if r: grid2disp[(r[0],r[1])]=(dx,dy)

GOAL={(3,24),(36,18)}
def blocks(g): return tuple(sorted((s.x,s.y) for s in g.current_level.get_sprites_by_tag("0022vrxelxosfy")))
def selpos(g): return (g.prkgpeyexo.x,g.prkgpeyexo.y)
def statekey(g): return (blocks(g), selpos(g))
def settle(env,maxp=60):
    g=env._game
    for _ in range(maxp):
        if not g.lphmmaeepj and not g.hrknegnjkg: break
        env.step(GameAction.ACTION1)
def is_win(env):
    return env._game.dbmlcqbquh()

# actions: moves 1-4, and click-select each of the two block centers
# click needs current block positions; we compute dynamically
def do_action(env, act):
    g=env._game
    if act in (1,2,3,4):
        env.step(GameAction.from_id(act)); settle(env)
    else: # ('click', (bx,by))
        bx,by=act[1]
        d=grid2disp.get((bx+1,by+1))
        if d is None: return False
        env.step(GameAction.ACTION6, data={'x':d[0],'y':d[1]}); settle(env)
    return True

# BFS via deepcopy of env
start=make(); settle(start)
root_key=statekey(start._game)
q=collections.deque()
q.append((start, []))
seen={root_key}
best=None
nodes=0
while q:
    env,path=q.popleft()
    nodes+=1
    if is_win(env):
        best=path; break
    if len(path)>=14: continue
    g=env._game
    bl=blocks(g)
    acts=[1,2,3,4]+[('click',b) for b in bl]
    for a in acts:
        env2=copy.deepcopy(env)
        try:
            ok=do_action(env2,a)
        except Exception as e:
            continue
        if not ok: continue
        k=statekey(env2._game)
        if k in seen: continue
        # dead if out of steps
        if env2._game.urgssjskot.current_steps<=0: continue
        seen.add(k)
        q.append((env2, path+[a]))
    if nodes%200==0:
        print("nodes",nodes,"frontier",len(q),"seen",len(seen))
    if nodes>20000: print("giving up"); break

print("SEARCH DONE nodes",nodes,"seen",len(seen))
if best is None:
    print("NO SOLUTION within depth")
else:
    print("SOLUTION length",len(best))
    print(best)
    # verify from fresh
    v=make()
    for a in best: do_action(v,a)
    print("VERIFY win?",v._game.dbmlcqbquh(),"levels_completed",v.step(GameAction.ACTION1).levels_completed if False else 'n/a')
    print("final blocks",blocks(v._game))
