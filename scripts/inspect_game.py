"""Inspect a real game's mechanics at local speed (NORMAL mode).

Renders the frame as a compact color grid and reports object structure + what each action
does (does the avatar move? what changes?). Helps understand why the agent fails.

Usage: PYTHONPATH=src python scripts/inspect_game.py <game_prefix> [actions e.g. 1,1,4,5,6]
"""
import logging
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
prefix = sys.argv[1] if len(sys.argv) > 1 else "ls20"
seq = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else []

client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("t"))
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
card = client.open_scorecard(tags=["inspect"])
env = client.make(game_id=gid, scorecard_id=card)


def show(obs, label):
    g = P.to_grid(obs.frame)
    bg = P.detect_background(g)
    objs = P.connected_components(g, background=bg)
    # downsample to a coarse text view (every other row/col)
    sub = g[::2, ::2]
    chars = "0123456789ABCDEF"
    print(f"--- {label}: state={obs.state} levels={obs.levels_completed}/{obs.win_levels} "
          f"avail={obs.available_actions} bg={bg} #objs={len(objs)} ---")
    for row in sub:
        print("".join(chars[v] if 0 <= v < 16 else "?" for v in row))
    # object summary
    from collections import Counter
    cc = Counter(o.color for o in objs)
    print("objects by color (color:count):", dict(cc))


obs = env.reset()
show(obs, "reset")
for a in seq:
    if a == 6:
        obs = env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
    else:
        obs = env.step(GameAction.from_id(a))
    show(obs, f"after ACTION{a}")
    if obs.state in (GameState.WIN, GameState.GAME_OVER):
        print("TERMINAL"); break
