"""Extract wa30 L0 GROUND-TRUTH logical state from the engine internals (offline) to use as TDD oracles:
avatar anchor, block anchors, goal-cell set, wall/obstacle set. Also expose the game instance so we can
validate a forward model against real engine steps.
"""
import logging
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)

client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("wa30"))
env = client.make(game_id=gid, scorecard_id="wa30-truth")
obs = env.reset()

# find the underlying game instance
print("env type:", type(env).__name__, "attrs:", [a for a in dir(env) if not a.startswith('__')][:40])
game = None
for attr in ("_game", "game", "_env", "environment", "_environment"):
    g = getattr(env, attr, None)
    if g is not None and hasattr(g, "current_level"):
        game = g; print("found game via", attr); break
if game is None:
    # deep search
    for a in dir(env):
        try: v = getattr(env, a)
        except Exception: continue
        if hasattr(v, "current_level"):
            game = v; print("found game via", a); break
print("game:", type(game).__name__ if game else None)
if game:
    lvl = game.current_level
    print("avatar (wbmdvjhthc):", [(s.x, s.y, getattr(s,'rotation',None)) for s in lvl.get_sprites_by_tag("wbmdvjhthc")])
    print("blocks (geezpjgiyd):", [(s.x, s.y, s.width, s.height) for s in lvl.get_sprites_by_tag("geezpjgiyd")])
    print("goal cells (wyzquhjerd) count:", len(getattr(game, "wyzquhjerd", set())))
    print("goal cells sample:", sorted(getattr(game, "wyzquhjerd", set()))[:20])
    print("collidable/walls (pkbufziase) count:", len(getattr(game, "pkbufziase", set())))
    print("obstacles (qthdiggudy):", sorted(getattr(game, "qthdiggudy", set()))[:20])
    print("step counter:", game.current_level.get_data("StepCounter"))
