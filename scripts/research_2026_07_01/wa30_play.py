"""Play wa30 end-to-end with the grab-drag planner: perceive -> plan -> execute, re-plan each level.
Reports levels solved + actions/level vs the human baseline (RHAE). wa30 is a zero-game the coverage
explorer scores 0 on, so any level solved is a genuine crack.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 python scripts/research_2026_07_01/wa30_play.py
"""
import logging
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
import wa30_planner as W
logging.basicConfig(level=logging.ERROR)

client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
env_info = next(e for e in client.get_environments() if e.game_id.startswith("wa30"))
baseline = list(env_info.baseline_actions or [])
env = client.make(game_id=env_info.game_id, scorecard_id="wa30-play")
obs = env.reset()

def act(a):
    return env.step(GameAction.ACTION5 if a == 5 else GameAction.from_id(a))

total = 0
lvl = 0
per_level = []
while lvl < (obs.win_levels or 9) and total < 6000:
    grid = P.to_grid(obs.frame)
    avatar, blocks, pads = W.perceive(grid)
    if avatar is None or not blocks or not pads:
        print(f"L{lvl}: perceive failed avatar={avatar} #blocks={len(blocks)} #pads={len(pads)}"); break
    plan = W.plan_all(avatar, blocks, pads)
    if plan is None:
        print(f"L{lvl}: no plan (blocks={blocks} pads={pads})"); break
    used = 0
    for a in plan:
        obs = act(a); total += 1; used += 1
        nl = int(obs.levels_completed or 0)
        if nl > lvl or obs.state == GameState.WIN:
            base = baseline[lvl] if lvl < len(baseline) else None
            rhae = min(1.15, (base / used) ** 2) if base else None
            per_level.append((lvl, used, base, rhae))
            print(f"L{lvl} SOLVED in {used} actions (human baseline={base}, RHAE={rhae})")
            lvl = nl
            break
        if obs.state == GameState.GAME_OVER:
            print(f"L{lvl}: GAME_OVER after {used} actions (plan was {len(plan)}; level step-budget too "
                  f"tight for the sequential planner) — stopping"); break
    else:
        print(f"L{lvl}: plan exhausted ({used} actions) without level-up — stopping"); break
    if obs.state in (GameState.WIN, GameState.GAME_OVER):
        break

print(f"\nbaseline_actions (human) per level: {baseline}")
print(f"RESULT wa30: solved {lvl}/{obs.win_levels} levels in {total} total actions "
      f"(coverage explorer = 0/{obs.win_levels}, a zero-game)")
if per_level:
    import numpy as np
    rh = [r for (_,_,_,r) in per_level if r is not None]
    print(f"per-level RHAE mean = {np.mean(rh):.3f} (vs coverage-explorer 0.000 — wa30 was a zero-game)")
