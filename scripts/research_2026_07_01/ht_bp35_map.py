"""ht_bp35_map: dump the true tilemap for bp35 level 1 via engine introspection."""
import sys
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('bp35'))
env = c.make(game_id=gid, scorecard_id='x')
obs = env.reset()
g = env._game
lm = g.oztjzzyqoek           # uakietkqfso level-logic object
tm = lm.hdnrlfmyrj           # klmsuijofik tilemap
player = lm.twdpowducb

print("level index", lm.qswcochjodb)
print("gravity vivnprldht(True=up-index? dy=-1 if True):", lm.vivnprldht)
print("player pos", player.qumspquyus, "name", player.flrpnczugo)
print("tilemap attrs:", [a for a in dir(tm) if not a.startswith('_')][:60])

# find grid extent by scanning
minx=miny=10**9; maxx=maxy=-10**9
cells={}
for x in range(-2, 80):
    for y in range(-2, 200):
        ts = tm.jhzcxkveiw(x, y)
        if ts:
            names = tuple(sorted(set(t.flrpnczugo if hasattr(t,'flrpnczugo') else t.name for t in ts)))
            cells[(x,y)] = [t.name for t in ts]
            minx=min(minx,x); maxx=max(maxx,x); miny=min(miny,y); maxy=max(maxy,y)
print("extent x", minx, maxx, "y", miny, maxy, "num filled", len(cells))

# collect name -> positions
from collections import defaultdict
byname=defaultdict(list)
for (x,y),names in cells.items():
    for n in names:
        byname[n].append((x,y))
print("\nDISTINCT tile names and counts:")
for n,ps in sorted(byname.items(), key=lambda kv:-len(kv[1])):
    print(f"  {n:20s} count={len(ps):4d}  sample={ps[:4]}")

# gem location
for gemname in ['fjlzdjxhant']:
    print(f"\nGEM {gemname} at:", byname.get(gemname))
