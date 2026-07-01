"""LEVER-B SEED: a visible-goal-conditioned solver. Auto-detect the avatar (object that moves under
ACTION1-4), bind the salient visible goal, greedily navigate toward it (re-observing each step so gravity/
slide are handled implicitly), and on stall try interaction (ACTION5 / click goal / undo) to trip the
gating mechanic. Measures level-ups on games our coverage explorer scores 0 on. ADDITIVE PoC — proves (or
refutes) that binding the visible goal + goal-directed action can crack a zero-game.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scratchpad/goal_solver.py <game> <goal_color> [budget]
"""
import logging, sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)


def centroid(grid, color):
    ys, xs = np.where(grid == color)
    return (float(ys.mean()), float(xs.mean())) if len(ys) else None


def main():
    game = sys.argv[1]; goal_color = int(sys.argv[2]); budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"gsolve-{game}")
    obs = env.reset(); g = P.to_grid(obs.frame)
    avail = [a for a in list(obs.available_actions or []) if a in (1, 2, 3, 4)]
    print(f"{gid}: avail={obs.available_actions} win_levels={obs.win_levels} goal_color={goal_color} "
          f"goal@{centroid(g, goal_color)}")

    # --- detect avatar: the color whose centroid shifts most across the move actions ---
    base_cent = {c: centroid(g, c) for c in range(16) if centroid(g, c) is not None}
    deltas = {}          # action -> (dy,dx) of avatar
    move = {}            # per-color total centroid shift
    for a in avail:
        obs = env.step(GameAction.from_id(a)); g2 = P.to_grid(obs.frame)
        for c, bc in base_cent.items():
            nc = centroid(g2, c)
            if nc:
                move[c] = move.get(c, 0.0) + abs(nc[0]-bc[0]) + abs(nc[1]-bc[1])
        g = g2
    # avatar = the moving color that is NOT the background and NOT the goal, smallest area preferred
    bg = P.detect_background(g)
    cand = sorted([c for c in move if move[c] > 0.5 and c not in (bg, goal_color)],
                  key=lambda c: (-move[c],))
    avatar = cand[0] if cand else None
    print(f"avatar color detected = {avatar} (movement scores: "
          f"{ {c: round(move[c],1) for c in sorted(move, key=lambda x:-move[x])[:5]} })")
    if avatar is None:
        print("!! no avatar detected — abort"); return

    # learn per-action avatar delta cleanly from current position
    for a in avail:
        b = centroid(g, avatar)
        obs = env.step(GameAction.from_id(a)); g = P.to_grid(obs.frame); n = centroid(g, avatar)
        deltas[a] = (n[0]-b[0], n[1]-b[1]) if (b and n) else (0, 0)
    print("learned avatar deltas:", {a: (round(d[0],1), round(d[1],1)) for a, d in deltas.items()})

    # --- greedy goal-directed navigation with stall-interaction ---
    n_act = len(avail) * (len(base_cent) if False else 0)  # (probing already spent some actions)
    n = 0; start_lv = int(obs.levels_completed or 0); best_d = 1e9; stall = 0; wins = 0
    can_grab = 5 in (obs.available_actions or []); can_click = 6 in (obs.available_actions or [])
    while n < budget:
        g = P.to_grid(obs.frame)
        av = centroid(g, avatar); go = centroid(g, goal_color)
        if go is None:  # goal consumed/hidden -> treat as progress cue
            go = None
        lv = int(obs.levels_completed or 0)
        if lv > start_lv:
            wins += 1; print(f"  [{n}] *** LEVEL-UP {start_lv}->{lv} on {game} — visible-goal solve WORKS ***")
            start_lv = lv; best_d = 1e9; stall = 0
            deltas = {}  # relearn next level
        if av is not None and go is not None:
            d = abs(av[0]-go[0]) + abs(av[1]-go[1]); best_d = min(best_d, d)
            best_a, best_gain = None, -1e9
            for a, (dy, dx) in deltas.items():
                nd = abs((av[0]+dy)-go[0]) + abs((av[1]+dx)-go[1])
                if d - nd > best_gain: best_gain, best_a = d - nd, a
            if best_a is None: best_a = avail[n % len(avail)]
        else:
            best_a = avail[n % len(avail)]
        prev_av = centroid(g, avatar)
        obs = env.step(GameAction.from_id(best_a)); n += 1
        moved = centroid(P.to_grid(obs.frame), avatar)
        progressed = prev_av and moved and (abs(moved[0]-prev_av[0]) + abs(moved[1]-prev_av[1]) > 0.1)
        stall = 0 if progressed else stall + 1
        # stall -> try interaction to trip a gating mechanic
        if stall >= 6:
            g = P.to_grid(obs.frame); go = centroid(g, goal_color)
            if can_grab:
                obs = env.step(GameAction.ACTION5); n += 1
            elif can_click and go is not None:
                obs = env.step(GameAction.ACTION6, data={"x": int(round(go[1])), "y": int(round(go[0]))}); n += 1
            else:
                obs = env.step(GameAction.from_id(avail[(n+stall) % len(avail)])); n += 1
            stall = 0 if stall < 12 else 0
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); start_lv = int(obs.levels_completed or 0)
        if n % 400 == 0:
            dd = best_d if best_d < 1e9 else -1
            print(f"  [{n}] best_dist={dd:.1f} levels={obs.levels_completed} state={obs.state} wins={wins}")
    print(f"\nRESULT {game}: wins={wins} final_levels={obs.levels_completed} min_dist={best_d:.1f}")


if __name__ == "__main__":
    main()
