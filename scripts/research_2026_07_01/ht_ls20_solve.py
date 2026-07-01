"""ht_ls20_solve.py — Solve ls20 level 0 by live-engine BFS.

The engine is deterministic and resettable. We BFS over action sequences,
replaying each unique state's prefix once from a fresh reset, deduping by the
internal state signature (avatar pos, shape, color, rotation, steps, lives).
Goal: levels_completed >= 1. Then verify the found sequence from a fresh reset.
"""
import sys
from collections import deque
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('ls20'))

ACTIONS = [1, 2, 3, 4]

_ENV = c.make(game_id=gid, scorecard_id='x')

def new_env():
    _ENV.reset()
    return _ENV

def sig(env):
    g = env._game
    return (g.gudziatsk.x, g.gudziatsk.y, g.fwckfzsyc, g.hiaauhahz, g.cklxociuu,
            g._step_counter_ui.current_steps, g.aqygnziho)

def replay(seq):
    """Replay a sequence from fresh reset; return (obs, env)."""
    env = new_env()
    obs = None
    for a in seq:
        obs = env.step(GameAction.from_id(a))
    return obs, env

# BFS
start_env = new_env()
start_sig = sig(start_env)
print("start sig (x,y,shape,color,rot,steps,lives):", start_sig)
print("target: shape", start_env._game.ldxlnycps, "color", start_env._game.yjdexjsoa,
      "rot", start_env._game.ehwheiwsk, "goal cells:",
      [(s.x, s.y) for s in start_env._game.plrpelhym])

visited = {start_sig}
frontier = deque([([], start_sig)])
solution = None
expansions = 0
MAX_DEPTH = 30

while frontier:
    seq, s = frontier.popleft()
    if len(seq) >= MAX_DEPTH:
        continue
    # replay prefix once
    env = new_env()
    for a in seq:
        env.step(GameAction.from_id(a))
    for a in ACTIONS:
        # branch: need a fresh env replaying seq then a
        env_b = new_env()
        for pa in seq:
            env_b.step(GameAction.from_id(pa))
        obs = env_b.step(GameAction.from_id(a))
        expansions += 1
        if obs.levels_completed >= 1:
            solution = seq + [a]
            print(f"\nSOLVED at depth {len(solution)} after {expansions} expansions")
            print("sequence:", solution)
            frontier.clear()
            break
        if obs.state == GameState.GAME_OVER:
            continue
        ns = sig(env_b)
        if ns not in visited:
            visited.add(ns)
            frontier.append((seq + [a], ns))
    if solution:
        break
    if expansions % 2000 == 0 and expansions:
        print(f"  ...{expansions} expansions, frontier {len(frontier)}, visited {len(visited)}, depth~{len(seq)+1}")

if not solution:
    print("NO SOLUTION found within depth", MAX_DEPTH, "expansions", expansions, "visited", len(visited))
    sys.exit(1)

# VERIFY from fresh reset
print("\n=== VERIFY from fresh reset ===")
obs, env = replay(solution)
print("levels_completed:", obs.levels_completed, "state:", obs.state, "len:", len(solution))
assert obs.levels_completed >= 1, "verification FAILED"
print("VERIFIED WIN. Shortest winning sequence (actions 1=up,2=down,3=left,4=right):")
print(solution)
