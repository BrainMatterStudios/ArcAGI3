import os
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('g50t'))
env = c.make(game_id=gid, scorecard_id='x')
obs = env.reset()
print("gid", gid, "avail", obs.available_actions, "state", obs.state, "lvl", obs.levels_completed)

ctrl = env._game.vgwycxsxjz
lvl = env._game.current_level

TAGS = {
 "vtwcsmdoqp":"enemy(qeixtoeawu)",
 "akfoiqesdk":"enemyzone(tddbvyfbvq)",
 "rsrdfsruqh":"WALL(bognpjtvzt)",
 "qftsebtxuc":"PLAYER(azxhtyyauk)",
 "gpkhwmwioo":"PHASE(ekfhaifjds)",
 "ovhuyqtghw":"marker(inaylmmhhy)",
 "Ghost":"Ghost",
 "kjrcloicja":"switchB(mcqullpcwz)",
 "medyellngi":"switchA(mxmeqbrmab)",
 "hxztohfdlx":"GATE(dryrnuvljg)",
 "hgglgttaui":"linker(hyztgkifvb)",
 "mpreboxmgc":"linkerpt(ugfrhsffov)",
 "gilbljmfbc":"GOAL(lrtamslcit)",
 "ppfvilwwnk":"TIMER(ofihnvwckg)",
}
print("\n=== ALL SPRITES BY TAG ===")
for tag,name in TAGS.items():
    sp = lvl.get_sprites_by_tag(tag)
    if sp:
        print(f"\n{name} [{tag}] count={len(sp)}")
        for s in sp:
            print(f"   x={s.x} y={s.y} w={s.width} h={s.height} rot={getattr(s,'rotation',None)} vis={s.is_visible} tags={s.tags}")

print("\n=== controller derived ===")
print("player(dzxunlkwxt) x,y", ctrl.dzxunlkwxt.x, ctrl.dzxunlkwxt.y)
print("goal(whftgckbcu) x,y", ctrl.whftgckbcu.x, ctrl.whftgckbcu.y, "-> WIN player at", ctrl.whftgckbcu.x+1, ctrl.whftgckbcu.y+1)
print("num phases (drofvwhbxb)", len(ctrl.drofvwhbxb), "phase x:", [p.x for p in ctrl.drofvwhbxb])
print("gates/obstacles (uwxkstolmf)", len(ctrl.uwxkstolmf))
for g in ctrl.uwxkstolmf:
    print("   gate x,y", g.x, g.y, "w,h", g.width, g.height, "rot", g.rotation, "dpdubazedr(one-way?)", g.dpdubazedr)
print("buttons (hamayflsib)", len(ctrl.hamayflsib), [type(b).__name__ for b in ctrl.hamayflsib])
for b in ctrl.hamayflsib:
    print("   button", type(b).__name__, "x,y", b.x, b.y, "outputs?", getattr(b,'nexhtmlmxh',None) or getattr(b,'ytztewxdin',None))
print("enemies (kgvnkyaimw)", len(ctrl.kgvnkyaimw))
print("timer sprite x", env._game.twyixucrqi.x, "w", env._game.twyixucrqi.width, "-> lose when -x>w i.e x<", -env._game.twyixucrqi.width)
print("jarvstobjt(cell)=6")

grid = P.to_grid(obs.frame)
import numpy as np
g = np.array(grid)
print("\ngrid shape", g.shape)
