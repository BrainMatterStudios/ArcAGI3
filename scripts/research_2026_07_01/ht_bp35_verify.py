"""ht_bp35_verify: independent fresh-reset verification of the winning sequence + trace."""
import json
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
GID = next(e.game_id for e in c.get_environments() if e.game_id.startswith('bp35'))
seq = json.load(open('scripts/research_2026_07_01/ht_bp35_solution.json'))['seq']

env = c.make(game_id=GID, scorecard_id='x'); obs = env.reset()
lm = env._game.oztjzzyqoek
print("START player", lm.twdpowducb.qumspquyus, "gem (3,7) avail", obs.available_actions)
amap = {'L': 3, 'R': 4}
for i, a in enumerate(seq):
    if a[0] in amap:
        obs = env.step(GameAction.from_id(amap[a[0]])); desc = {'L':'LEFT(A3)','R':'RIGHT(A4)'}[a[0]]
    else:
        obs = env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]}); desc = f'BREAK(A6 click {a[1]},{a[2]})'
    p = lm.twdpowducb.qumspquyus
    print(f"{i+1:2d} {desc:24s} player={p} moves={env._game.hbqwwgceeqp} lvl={obs.levels_completed} state={obs.state}")
print("\nFINAL levels_completed =", obs.levels_completed, "=> WIN" if obs.levels_completed>=1 else "=> FAIL")
