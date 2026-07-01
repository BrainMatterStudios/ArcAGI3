from arc_agi import Arcade, OperationMode
from arcengine import GameAction
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))
def make():
    env=c.make(game_id=gid, scorecard_id='x'); env.reset(); return env
def blocks(g): return sorted((s.x,s.y) for s in g.current_level.get_sprites_by_tag("0022vrxelxosfy"))
def sel(g): return (g.prkgpeyexo.x,g.prkgpeyexo.y)
def settle(env,maxp=60):
    g=env._game; n=0
    while g.lphmmaeepj or g.hrknegnjkg:
        env.step(GameAction.ACTION1); n+=1
        if n>maxp: break
    return n
env=make(); g=env._game
print("init blocks",blocks(g),"sel",sel(g))
# select block A stays selected (it's prkgpeyexo=first). move right repeatedly
for i in range(6):
    before=blocks(g); s=sel(g); st=g.urgssjskot.current_steps
    env.step(GameAction.ACTION4); n=settle(env)
    print("R#%d sel_before=%s blocks_before=%s -> blocks=%s sel=%s settlepumps=%d steps=%d lphm=%d hrk=%d"%(
        i,s,before,blocks(g),sel(g),n,g.urgssjskot.current_steps,len(g.lphmmaeepj),len(g.hrknegnjkg)))
