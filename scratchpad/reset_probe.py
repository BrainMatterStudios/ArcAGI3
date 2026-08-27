"""Empirically confirm reset semantics (full_reset vs level_reset) for prefix-replay design.
No deepcopy. Uses only env.reset()/env.step()."""
import sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

def mk(game):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    e = next(x for x in c.get_environments() if x.game_id.startswith(game))
    return c.make(game_id=e.game_id, scorecard_id="probe")

def lv(o):
    return int(getattr(o, "levels_completed", 0) or 0)

game = sys.argv[1] if len(sys.argv) > 1 else "tu93"
env = mk(game)
o = env.reset()
print(f"[{game}] fresh reset: state={o.state} levels={lv(o)} avail={list(o.available_actions)}")

# take a handful of moves, watch levels_completed
moves = [a for a in (1,2,3,4,5) if a in o.available_actions] or [1,2,3,4]
for i in range(40):
    o = env.step(GameAction.from_id(moves[i % len(moves)]))
    if o.state == GameState.GAME_OVER:
        print(f"  step {i}: GAME_OVER at levels={lv(o)}")
        break
print(f"[{game}] after 40 moves: state={o.state} levels={lv(o)}")

# now test single reset (mid-level, action_count>0) => expect level_reset (score preserved)
before = lv(o)
o2 = env.reset()
print(f"[{game}] single reset (mid-level): state={o2.state} levels={lv(o2)} full_reset={getattr(o2,'full_reset',None)} (score before={before})")

# second reset (now action_count==0) => expect full_reset to level 0
o3 = env.reset()
print(f"[{game}] double reset: state={o3.state} levels={lv(o3)} full_reset={getattr(o3,'full_reset',None)}")
