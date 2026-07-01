"""VERIFY dc22 decoded win-condition in-engine: avatar (color 14) reaching the goal cell (color 11)
triggers a level-up. Script a greedy goal-directed navigation and observe whether (a) the avatar reaches
the c11 goal, and (b) a level-up fires — confirming the source read. If the path is blocked (maze panel
needed), report that empirically.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scratchpad/dc22_verify.py [budget]
"""
import logging, sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)

AV, GOAL = 14, 11


def centroid(grid, color):
    ys, xs = np.where(grid == color)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean())), len(ys)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("dc22"))
    env = client.make(game_id=gid, scorecard_id="verify-dc22")
    obs = env.reset()
    g = P.to_grid(obs.frame)
    av = centroid(g, AV); go = centroid(g, GOAL)
    print(f"dc22 {gid}: state={obs.state} avail={obs.available_actions} win_levels={obs.win_levels}")
    print(f"avatar(c14)={av}  goal(c11)={go}  grid colors={sorted(set(g.flatten().tolist()))}")
    if not av or not go:
        print("!! avatar or goal not found by color at start — dumping coarse frame");
        chars="0123456789ABCDEF"
        for row in g[::2,::2]:
            print("".join(chars[v] if 0<=v<16 else "?" for v in row))
        return

    # 1) learn per-action displacement of the avatar centroid
    print("\n-- probing action effects on avatar centroid --")
    disp = {}
    for a in (1, 2, 3, 4):
        before = centroid(g, AV)
        obs = env.step(GameAction.from_id(a)); g2 = P.to_grid(obs.frame)
        after = centroid(g2, AV)
        dy = dx = None
        if before and after:
            dy = after[0][0]-before[0][0]; dx = after[0][1]-before[0][1]
        disp[a] = (dy, dx)
        print(f"  ACTION{a}: davatar=(dy={dy}, dx={dx})  levels={obs.levels_completed}")
        g = g2
        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            print(f"  (terminal {obs.state} during probe)");

    # 2) greedy navigate: pick the action that most reduces Manhattan dist avatar->goal
    print("\n-- greedy goal-directed navigation --")
    n = 4; start_lv = int(obs.levels_completed or 0); best_d = 1e9; stuck = 0
    while n < budget:
        g = P.to_grid(obs.frame)
        av = centroid(g, AV); go = centroid(g, GOAL)
        if not av or not go:
            print(f"  [{n}] avatar/goal vanished (state={obs.state}, levels={obs.levels_completed})"); break
        ay, ax = av[0]; gy, gx = go[0]
        d = abs(ay-gy)+abs(ax-gx); best_d = min(best_d, d)
        lv = int(obs.levels_completed or 0)
        if lv > start_lv:
            print(f"  [{n}] *** LEVEL-UP {start_lv}->{lv} — WIN-CONDITION CONFIRMED (avatar reached goal) ***")
            start_lv = lv; best_d = 1e9
        # choose greedy action by predicted centroid move
        best_a, best_gain = None, -1e9
        for a, (dy, dx) in disp.items():
            if dy is None: continue
            nd = abs((ay+dy)-gy) + abs((ax+dx)-gx)
            gain = d - nd
            if gain > best_gain:
                best_gain, best_a = gain, a
        if best_a is None:
            best_a = (n % 4) + 1
        prev = av[0]
        obs = env.step(GameAction.from_id(best_a)); n += 1
        g2 = P.to_grid(obs.frame); av2 = centroid(g2, AV)
        moved = av2 and (abs(av2[0][0]-prev[0]) + abs(av2[0][1]-prev[1]) > 0.1)
        if not moved:
            stuck += 1
            # blocked: try a random other direction
            disp_try = [a for a in (1,2,3,4) if a != best_a]
            a2 = disp_try[stuck % 3]
            obs = env.step(GameAction.from_id(a2)); n += 1
        else:
            stuck = 0
        if n % 50 == 0:
            print(f"  [{n}] dist={d:.1f} best={best_d:.1f} stuck={stuck} levels={obs.levels_completed} state={obs.state}")
        if obs.state == GameState.WIN:
            print(f"  [{n}] WIN state"); break
        if obs.state == GameState.GAME_OVER:
            print(f"  [{n}] GAME_OVER (step budget exhausted?) levels={obs.levels_completed}")
            obs = env.reset(); start_lv = 0
    print(f"\nRESULT: final levels_completed={obs.levels_completed}  min avatar->goal dist reached={best_d:.1f}")
    print("Read: level-up => decoded reach-cell win CONFIRMED. If min dist stays >0 and no level-up => path "
          "blocked (maze-panel mechanic needed) — confirms the hidden-mechanic barrier.")


if __name__ == "__main__":
    main()
