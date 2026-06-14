"""Auto-survey all real games' mechanics at local speed (NORMAL mode).

For each game: available actions, win_levels, background, #objects, color histogram, and
whether a controllable avatar is detectable (an object whose motion correlates with the
action). Builds a mechanic taxonomy to target improvements + pick tractable games.

Usage: PYTHONPATH=src python scripts/survey_games.py
"""
import logging
from collections import Counter

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import movement as MV
from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("t"))
card = client.open_scorecard(tags=["survey"])
games = sorted(e.game_id for e in client.get_environments())

print(f"{'game':<16}{'avail':<16}{'win':<4}{'#obj':<6}{'bg':<4}{'avatar':<8}{'kind'}")
for gid in games:
    try:
        env = client.make(game_id=gid, scorecard_id=card)
        obs = env.reset()
        g = P.to_grid(obs.frame)
        bg = P.detect_background(g)
        objs = P.connected_components(g, background=bg)
        avail = list(obs.available_actions or [])
        simple = [a for a in (1, 2, 3, 4, 5) if a in avail]
        # probe each simple action once from reset; detect a consistent single-object mover
        votes = {}
        cur = g
        for aid in simple:
            o2 = env.step(GameAction.from_id(aid))
            if o2 is None or o2.state in (GameState.WIN, GameState.GAME_OVER):
                break
            g2 = P.to_grid(o2.frame)
            for color, d in MV.infer_all_translations(cur, g2, bg).items():
                votes.setdefault(color, {})[aid] = d
            cur = g2
        # avatar = color with most distinct deltas across actions
        avatar = None
        if votes:
            avatar = max(votes, key=lambda c: (len(set(votes[c].values())), len(votes[c])))
            ndist = len(set(votes[avatar].values()))
            avatar = f"c{avatar}/{ndist}d"
        if avail == [6]:
            kind = "click-only"
        elif 6 in avail and simple:
            kind = "mixed"
        elif simple and avatar:
            kind = "avatar-move"
        elif simple:
            kind = "simple-noavatar"
        else:
            kind = "?"
        print(f"{gid:<16}{str(avail):<16}{obs.win_levels:<4}{len(objs):<6}{bg:<4}{str(avatar):<8}{kind}", flush=True)
    except Exception as e:
        print(f"{gid:<16}ERROR {e}", flush=True)
