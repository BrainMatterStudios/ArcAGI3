from arc_agi import Arcade, OperationMode
from arcengine import GameAction
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))
env=c.make(game_id=gid, scorecard_id='x'); obs=env.reset(); g=env._game
grid2disp={}
for dx in range(64):
    for dy in range(64):
        r=g.camera.display_to_grid(dx,dy)
        if r: grid2disp[(r[0],r[1])]=(dx,dy)
def blocks(): return sorted((s.x,s.y) for s in g.current_level.get_sprites_by_tag("0022vrxelxosfy"))
def sel(): return (g.prkgpeyexo.x,g.prkgpeyexo.y)
def click_block(bx,by):
    d=grid2disp[(bx+1,by+1)]
    return env.step(GameAction.ACTION6,data={'x':d[0],'y':d[1]})
def mv(a): return env.step(GameAction.from_id(a))
def show(tag): print("%-14s blocks=%s sel=%s state=%s lvl=%s steps=%d"%(tag,blocks(),sel(),obs.state,obs.levels_completed,g.urgssjskot.current_steps))
show("init")
# A is selected at (9,21). Move right x2 to (15,21)
obs=mv(4); show("A->12")
obs=mv(4); show("A->15")
# launch B rightward -> should land (33,21)
obs=mv(4); show("launchB")
# select B (now at 33,21), move to (36,18)
obs=click_block(33,21); show("selB")
obs=mv(4); show("B->36,21")   # right 33->36
obs=mv(1); show("B->36,18")   # up 21->18
# select A (at 15,21), move to (3,24)
obs=click_block(15,21); show("selA")
for i in range(4):
    obs=mv(3); show("A left %d"%i)
obs=mv(2); show("A down->24")
print("\nFINAL win?",g.dbmlcqbquh(),"state",obs.state,"levels_completed",obs.levels_completed)

print("\n=== CLEAN REPLAY FROM FRESH RESET ===")
env2=c.make(game_id=gid, scorecard_id='x'); ob=env2.reset(); g2=env2._game
g2d={}
for dx in range(64):
    for dy in range(64):
        r=g2.camera.display_to_grid(dx,dy)
        if r: g2d[(r[0],r[1])]=(dx,dy)
dB=g2d[(34,22)]; dA=g2d[(16,22)]
print("click B display:",dB,"click A display:",dA)
seq=[('m',4),('m',4),('m',4),('c',dB),('m',4),('m',1),('c',dA),('m',3),('m',3),('m',3),('m',3),('m',2)]
for kind,v in seq:
    if kind=='m': ob=env2.step(GameAction.from_id(v))
    else: ob=env2.step(GameAction.ACTION6,data={'x':v[0],'y':v[1]})
print("RESULT levels_completed=",ob.levels_completed,"state=",ob.state,"actions=",len(seq))
assert ob.levels_completed>=1, "FAILED"
print("CONFIRMED: ka59 level 0 solved in",len(seq),"actions")
