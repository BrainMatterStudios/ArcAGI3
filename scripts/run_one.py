"""Run the reactive policy on one real game with live progress (level-ups + heartbeats).

Usage: PYTHONPATH=src python scripts/run_one.py <game_prefix> <budget>
"""
import logging
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.policy import HybridPolicy

logging.basicConfig(level=logging.ERROR)
prefix = sys.argv[1] if len(sys.argv) > 1 else "vc33"
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

client = Arcade(operation_mode=OperationMode.ONLINE, logger=logging.getLogger("t"))
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
card = client.open_scorecard(tags=["run-one"])
env = client.make(game_id=gid, scorecard_id=card)
pol = HybridPolicy()
obs = env.reset()
win_levels = int(obs.win_levels or 0)
best = 0
t0 = time.time()
n = 0
print(f"{gid}: win_levels={win_levels} avail={obs.available_actions}", flush=True)
while n < budget:
    if obs.state == GameState.WIN:
        print(f"WIN at action {n}", flush=True)
        break
    g = P.to_grid(obs.frame)
    tok = pol.decide(g, obs.state == GameState.GAME_OVER, obs.state == GameState.NOT_PLAYED,
                     int(obs.levels_completed or 0), list(obs.available_actions or []))
    if tok[0] == "reset":
        obs = env.reset()
    elif tok[0] == "S":
        obs = env.step(GameAction.from_id(tok[1]))
    else:
        obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
    n += 1
    lv = int(obs.levels_completed or 0)
    if lv > best:
        best = lv
        print(f"  LEVEL UP -> {lv}/{win_levels} at action {n} (t={time.time()-t0:.0f}s, "
              f"phase={pol.phase}, states={len(pol.gs.wm) if pol.gs else 0})", flush=True)
    if n % 400 == 0:
        print(f"  ...action {n}, best {best}/{win_levels}, phase={pol.phase}, "
              f"avatar={pol.mm.avatar_color if pol.mm else None}, "
              f"states={len(pol.gs.wm) if pol.gs else 0}, t={time.time()-t0:.0f}s", flush=True)
print(f"DONE {gid}: best {best}/{win_levels} actions {n} time {time.time()-t0:.0f}s", flush=True)
