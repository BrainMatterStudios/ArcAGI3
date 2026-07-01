"""Offline initial-frame inspector: dump coarse grid + object structure so we can identify avatar/goal
entities and pick the cleanest Lever-B prototype target.
Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scratchpad/inspect_offline.py <game> [nsteps action]
"""
import logging, sys
from collections import Counter
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)

game = sys.argv[1] if len(sys.argv) > 1 else "bp35"
client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
env = client.make(game_id=gid, scorecard_id=f"insp-{game}")
chars = "0123456789ABCDEF"

def show(obs, label):
    g = P.to_grid(obs.frame)
    bg = P.detect_background(g)
    objs = P.connected_components(g, background=bg)
    print(f"--- {label}: state={obs.state} levels={obs.levels_completed}/{obs.win_levels} "
          f"avail={obs.available_actions} bg={bg} #objs={len(objs)} ---")
    for row in g[::2, ::2]:
        print("".join(chars[v] if 0 <= v < 16 else "?" for v in row))
    cc = Counter(int(v) for v in g.flatten())
    print("cells by color:", dict(sorted(cc.items())))
    # small objects (candidate avatar/goal) = rare colors / tiny components
    small = [(o.color, o.size, (round(o.cy,1), round(o.cx,1))) for o in objs if o.size <= 12]
    print("small objs (color,size,(cy,cx)):", small[:25])

obs = env.reset()
show(obs, "reset")
