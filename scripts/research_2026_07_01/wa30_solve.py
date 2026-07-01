"""First mechanic-planner attempt: wa30 grab-drag. Decode: avatar (c14) moves 1 cell/action; ACTION5 grabs
an adjacent block (geezpjgiyd, normally c4; held->c0); moving drags it; ACTION5 releases; win = ALL blocks
on the goal pad (c9 border / c2 fill), none held. This scripts: for each block -> navigate avatar adjacent ->
grab -> drag onto a free pad cell -> release. Reports level-ups. First honest test of goal-conditioned
mechanic execution on a movement zero-game.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/research_2026_07_01/wa30_solve.py [budget]
"""
import logging, sys
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
logging.basicConfig(level=logging.ERROR)
AV = 14


def grid(obs): return P.to_grid(obs.frame)
def cen(g, c):
    ys, xs = np.where(g == c)
    return (float(ys.mean()), float(xs.mean())) if len(ys) else None


def objs(g, color, bg):
    return [o for o in P.connected_components(g, background=bg) if o.color == color]


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("wa30"))
    env = client.make(game_id=gid, scorecard_id="wa30-solve")
    obs = env.reset(); g = grid(obs); bg = P.detect_background(g)
    print(f"{gid}: avail={obs.available_actions} win_levels={obs.win_levels} bg={bg}")

    # learn avatar per-action delta (avatar = c14)
    deltas = {}
    for a in (1, 2, 3, 4):
        b = cen(g, AV); obs = env.step(GameAction.from_id(a)); g = grid(obs); n = cen(g, AV)
        deltas[a] = (round(n[0]-b[0], 1), round(n[1]-b[1], 1)) if (b and n) else (0, 0)
    print("avatar(c14) deltas:", deltas)
    # pad cells (goal): c9 border, c2 fill region centroid
    pad = cen(g, 2) or cen(g, 9)
    blocks = objs(g, 4, bg)
    print(f"pad(goal c2/c9) centroid={pad}  #blocks(c4)={len(blocks)} "
          f"block_centroids={[(round(o.centroid[0],1),round(o.centroid[1],1)) for o in blocks]}")

    def step(a, data=None):
        nonlocal obs, g
        if a == 6:
            obs = env.step(GameAction.ACTION6, data=data)
        else:
            obs = env.step(GameAction.from_id(a))
        g = grid(obs)
        return obs

    def nav_toward(ty, tx, max_steps=40):
        """greedy move avatar toward (ty,tx); return final dist."""
        for _ in range(max_steps):
            a0 = cen(g, AV)
            if a0 is None: return 1e9
            d = abs(a0[0]-ty) + abs(a0[1]-tx)
            if d <= 3: return d
            best_a, best = None, d
            for a, (dy, dx) in deltas.items():
                nd = abs((a0[0]+dy)-ty) + abs((a0[1]+dx)-tx)
                if nd < best: best, best_a = nd, a
            if best_a is None: return d
            step(best_a)
            if obs.state == GameState.GAME_OVER:
                return -1
        return abs(cen(g, AV)[0]-ty) + abs(cen(g, AV)[1]-tx) if cen(g, AV) else 1e9

    # --- probe grab: navigate to first block, ACTION5, check a block starts moving with avatar ---
    start_lv = int(obs.levels_completed or 0)
    if 5 not in (obs.available_actions or []):
        print("no ACTION5 (grab) available — abort"); return
    b0 = blocks[0]
    print(f"\n-- probe grab on block @({b0.centroid[0]:.1f},{b0.centroid[1]:.1f}) --")
    d = nav_toward(b0.centroid[0], b0.centroid[1])
    print(f"  navigated adjacent, dist={d}, avatar@{cen(g,AV)}")
    held0 = int((g == 0).sum())
    step(5); held1 = int((g == 0).sum())
    print(f"  ACTION5: color0(held-marker) {held0}->{held1} state={obs.state}")
    # move and see if a c4 block centroid follows the avatar
    bset_before = [(round(o.centroid[0],1),round(o.centroid[1],1)) for o in objs(g,4,bg)]
    step(4); bset_after = [(round(o.centroid[0],1),round(o.centroid[1],1)) for o in objs(g,4,bg)]
    print(f"  after grab+move-right: blocks {bset_before} -> {bset_after}")
    grabbed = bset_before != bset_after
    print(f"  GRAB {'CONFIRMED (a block moved with avatar)' if grabbed else 'NOT detected'}")

    # --- attempt full solve: for each block, nav to it, grab, drag onto pad, release ---
    print("\n-- attempt solve --")
    n = 0
    while n < budget:
        lv = int(obs.levels_completed or 0)
        if lv > start_lv:
            print(f"  *** LEVEL-UP {start_lv}->{lv} — wa30 grab-drag SOLVE WORKS ***"); start_lv = lv
        bl = objs(g, 4, bg)
        if not bl or pad is None:
            break
        # pick the block farthest from pad still not on it
        target = max(bl, key=lambda o: abs(o.centroid[0]-pad[0])+abs(o.centroid[1]-pad[1]))
        nav_toward(target.centroid[0], target.centroid[1]); n += 1
        step(5)  # grab
        nav_toward(pad[0], pad[1]); n += 1
        step(5)  # release
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); g = grid(obs); start_lv = int(obs.levels_completed or 0)
        n += 5
        if n % 200 < 8:
            print(f"  [{n}] levels={obs.levels_completed} state={obs.state} "
                  f"blocks_on_pad?~ {[(round(o.centroid[0]),round(o.centroid[1])) for o in objs(g,4,bg)]}")
    print(f"\nRESULT wa30: final_levels={obs.levels_completed}")


if __name__ == "__main__":
    main()
