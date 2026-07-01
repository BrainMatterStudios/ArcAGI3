"""ht_bp35_ascii: render bp35 level map as ASCII with legend."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('bp35'))
env = c.make(game_id=gid, scorecard_id='x'); env.reset()
lm = env._game.oztjzzyqoek; tm = lm.hdnrlfmyrj; player = lm.twdpowducb

sym = {
 'xcjjwqfzjfe':'#',   # wall
 'jcyhkseuorf':'=',   # structure hi-y
 'qclfkhjnaac':'B',   # breakable (A6 break -> fall)
 'aknlbboysnc':'~',   # moving obstacle (crush)
 'fjlzdjxhant':'G',   # GEM
 'oonshderxef':'.',   # passable toggle-off
 'yuuqpmlxorv':'o',   # toggle block
 'lrpkmzabbfa':'^',   # gravity flip
 'etlsaqqtjvn':'*',
 'ubhhgljbnpu':'X','hzusueifitk':'X',  # spikes
}
px,py = player.qumspquyus
maxx=11; ys=range(-1,37)
print("   " + "".join(str(x%10) for x in range(maxx)))
for y in ys:
    row=[]
    for x in range(maxx):
        if (x,y)==(px,py): row.append('P'); continue
        ts=tm.jhzcxkveiw(x,y)
        if not ts: row.append(' '); continue
        names=[t.name for t in ts]
        s='?'
        for n in names:
            if n in sym: s=sym[n]; break
        row.append(s)
    mark=''
    if y==py: mark=' <-player'
    if y==7: mark=' <-gem row'
    print(f"{y:3d} "+"".join(row)+mark)
print("\nLegend: # wall  = struct  B breakable  ~ mover  G gem  . passable  o toggle  ^ gravflip  * spread  X spike")
print(f"player {(px,py)}  gem (3,7)  gravity dy={-1 if lm.vivnprldht else 1}")
