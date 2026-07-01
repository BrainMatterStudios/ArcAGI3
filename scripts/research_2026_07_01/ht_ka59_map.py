from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith('ka59'))
env=c.make(game_id=gid, scorecard_id='x'); obs=env.reset()
g=env._game
lvl=g.current_level
print("state",obs.state,"avail",obs.available_actions,"levels_completed",obs.levels_completed)
tags=["0022vrxelxosfy","0001uqqokjrptk","0003umnkyodpjp","0010xzmuziohuf","0027jbgxilrocf","0015qniapgwsvb","0029ifoxxfvvvs","0007zqjfknlfvm","Enemy","Collider"]
for t in tags:
    ss=lvl.get_sprites_by_tag(t)
    print("== tag",t,"count",len(ss))
    for s in ss:
        print("   pos(x,y)=(%d,%d) w=%d h=%d rot=%s tags=%s"%(s.x,s.y,s.width,s.height,getattr(s,'rotation',None),[x for x in s.tags if x!=t]))
print("player prkgpeyexo pos",g.prkgpeyexo.x,g.prkgpeyexo.y,"w",g.prkgpeyexo.width,"h",g.prkgpeyexo.height)
print("stepcounter",lvl.get_data("StepCounter"))
