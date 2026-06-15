"""A/B real-game sweep: run a policy over live ARC-AGI-3 games and report levels.

Usage:
  PYTHONPATH=src python scripts/ab_salience.py <policy> <budget> [game_prefixes...]
    policy: salience | reactive
  e.g. PYTHONPATH=src python scripts/ab_salience.py salience 30000 tu93 vc33

With no prefixes it sweeps ALL public games. Prints per-game levels + a TOTAL line.
NORMAL mode plays locally after fetching env metadata from the live API (key in .env).
"""
import logging
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402

logging.basicConfig(level=logging.ERROR)

POLICY = sys.argv[1] if len(sys.argv) > 1 else "salience"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
PREFIXES = sys.argv[3:]


import os  # noqa: E402

TRUST = int(os.getenv("TRUST", "1"))


def make_policy():
    if POLICY == "reactive":
        from arcagi3.policy import HybridPolicy
        return HybridPolicy()
    from arcagi3.salience_explorer import SalienceExplorer
    return SalienceExplorer(trust_threshold=TRUST)


client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("t"))
envs = client.get_environments()
if PREFIXES:
    envs = [e for e in envs if any(e.game_id.startswith(p) for p in PREFIXES)]
card = client.open_scorecard(tags=[f"ab-{POLICY}"])

print(f"policy={POLICY} trust={TRUST} budget={BUDGET} games={len(envs)}", flush=True)
rows = []
t_all = time.time()
for e in envs:
    gid = e.game_id
    env = client.make(game_id=gid, scorecard_id=card)
    pol = make_policy()
    obs = env.reset()
    win_levels = int(obs.win_levels or 0)
    best = 0
    n = 0
    t0 = time.time()
    while n < BUDGET:
        if obs.state == GameState.WIN:
            break
        g = P.to_grid(obs.frame)
        tok = pol.decide(g, obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        n += 1
        best = max(best, int(obs.levels_completed or 0))
    dt = time.time() - t0
    rows.append((gid, best, win_levels, n, dt))
    print(f"  {gid:>12}  {best}/{win_levels}  actions={n}  {dt:.0f}s  "
          f"({n/max(dt,1e-9):.0f} act/s)", flush=True)

total = sum(b for _, b, _, _, _ in rows)
won = sum(1 for _, b, w, _, _ in rows if w and b >= w)
print(f"TOTAL levels={total}  wins={won}/{len(rows)}  elapsed={time.time()-t_all:.0f}s",
      flush=True)
