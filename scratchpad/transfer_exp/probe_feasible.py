"""Feasibility probe: can a segmentation-guided random explorer clear levels offline?
If level 2 is unreachable within budget, the transfer experiment cannot be run at all."""
import os, sys, logging, random, time, json
os.environ["ONLY_RESET_LEVELS"] = "true"
logging.disable(logging.CRITICAL)
import numpy as np
from scipy import ndimage
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

BUDGET = int(os.environ.get("BUDGET", 1500))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
GAMES = os.environ.get("GAMES", "").split(",") if os.environ.get("GAMES") else None

_S4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)

def frame_np(o):
    a = np.asarray(o.frame)
    return a[-1] if a.ndim == 3 else a

def click_targets(grid, rng, k=40):
    """One candidate click per 4-connected same-colour component, random interior pixel."""
    out = []
    for col in np.unique(grid):
        lab, n = ndimage.label(grid == col, structure=_S4)
        if n == 0: continue
        for idx in range(1, n + 1):
            ys, xs = np.nonzero(lab == idx)
            j = rng.randrange(len(ys))
            out.append((int(ys[j]), int(xs[j])))
            if len(out) >= k: return out
    return out

def run(game_id, arcade, seed, budget=BUDGET):
    rng = random.Random(seed)
    env = arcade.make(game_id=game_id, scorecard_id=f"probe-{seed}-{game_id[:6]}")
    o = env.reset()
    acts = 0; reached = {}; prev = 0
    while acts < budget and o.state != GameState.WIN:
        if o.state == GameState.GAME_OVER:
            o = env.step(GameAction.RESET); acts += 1; continue
        avail = [a for a in (o.available_actions or []) if a != 0]
        if not avail: break
        aid = rng.choice(avail)
        if aid == 6:
            tg = click_targets(frame_np(o), rng)
            if not tg: break
            y, x = tg[rng.randrange(len(tg))]
            o = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        else:
            o = env.step(GameAction.from_id(aid))
        acts += 1
        if o.levels_completed > prev:
            reached[o.levels_completed] = acts; prev = o.levels_completed
    return reached, o.win_levels

arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
if GAMES:
    envs = [e for e in envs if e.game_id.split("-")[0][:4] in GAMES]

print(f"budget={BUDGET} seeds={SEEDS}")
print(f"{'game':6} {'winlv':>5} {'L1@':>6} {'L2@':>6} {'L3@':>6}")
t0 = time.time(); r2 = 0; rows = []
for e in envs:
    stem = e.game_id.split("-")[0][:4]
    best = {}
    for s in SEEDS:
        reached, winlv = run(e.game_id, arcade, s)
        for k, v in reached.items():
            best[k] = min(best.get(k, 10**9), v)
    if 2 in best: r2 += 1
    rows.append({"game": stem, "win_levels": winlv, "reached": best})
    print(f"{stem:6} {winlv:>5} {str(best.get(1,'-')):>6} {str(best.get(2,'-')):>6} {str(best.get(3,'-')):>6}", flush=True)
print(f"\nreached L2 in >=1 seed: {r2}/{len(envs)}   elapsed {time.time()-t0:.0f}s")
json.dump(rows, open("scratchpad/transfer_exp/feasibility.json","w"), indent=1)
