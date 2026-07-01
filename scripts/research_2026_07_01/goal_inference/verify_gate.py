"""STAGE 2a — VERIFY gate: trust an induced predicate ONLY if planning to SATISFY it from the L0 START re-derives
the KNOWN L0 win (levels 0->1). This kills spurious correlates (ar25 reach) and certifies plannability. Reach +
Collect planners built; Cover/Template abstain (planners in 2b). Uses the false-positive-proof harness."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np
from collections import deque
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H, stage1_v4 as V
from arcagi3 import perception as P
from arcengine import GameAction, GameState

def _avatar(g, ac): 
    ys,xs=np.where(np.isin(g,list(ac))) if ac else (np.array([]),np.array([]))
    return (int(round(ys.mean())),int(round(xs.mean()))) if len(ys) else None

def _bfs_to(grid, ac, deltas, targets, bg):
    """BFS the avatar centroid to within 2 of a target cell, using learned deltas + current-frame walls
    (non-bg, non-agent, non-target = obstacle). Returns action list or None."""
    start=_avatar(grid,ac)
    if start is None or not targets: return None
    tset=set(targets)
    def block(r,c):
        if not(0<=r<64 and 0<=c<64): return True
        v=int(grid[r,c]); return v!=bg and v not in ac and (r,c) not in tset
    seen={start}; q=deque([(start,[])])
    while q:
        (r,c),path=q.popleft()
        if any(abs(r-tr)+abs(c-tc)<=2 for tr,tc in tset): return path
        for a,(dr,dc) in deltas.items():
            nr,nc=r+dr,c+dc
            if (nr,nc) not in seen and not block(nr,nc):
                seen.add((nr,nc)); q.append(((nr,nc),path+[a]))
        if len(seen)>6000: break
    return None

def plan_to_satisfy(pred, grid, model, bg, env_replay):
    """Return an action list to satisfy pred from `grid`. For collect, re-perceives after each sub-target via a
    LIVE replay env (walking removes collectibles). Reach/Collect only; else None (abstain)."""
    ac=model.agent_colors; deltas=model.move.deltas
    if not deltas: return None
    if pred.cls=="reach":
        tgt=[tuple(p) for p in np.argwhere(grid==pred.c)]
        return _bfs_to(grid, ac, deltas, tgt, bg)
    if pred.cls=="collect":
        # sequentially reach each pred.c cell; re-perceive via the live env between sub-goals
        plan=[]; obs=env_replay()  # env reset to L0 start
        for _ in range(40):
            g=P.to_grid(obs.frame) if len(obs.frame) else grid
            tgt=[tuple(p) for p in np.argwhere(g==pred.c)]
            if not tgt: break
            sub=_bfs_to(g, ac, deltas, tgt, bg)
            if not sub: break
            for a in sub:
                obs=obs=env_step(obs, a); plan.append(a)
                if obs is None: return plan
        return plan or None
    return None

_ENV=None
def env_step(obs, a): 
    global _ENV
    return _ENV.step(GameAction.from_id(a))

def verify(game):
    cap=H.capture(game)
    if cap is None: return f"{game}: L0 not solved"
    surv,submit,wd,bg=V.induce(cap)
    if not surv: return f"{game}: no predicate -> abstain"
    pred=surv[0][0]; cls=surv[0][2]
    if cap["model"] is None: return f"{game}: {cls} but no dynamics model -> abstain"
    global _ENV
    _ENV=H._mk(game); obs=_ENV.reset(); g0=P.to_grid(obs.frame) if len(obs.frame) else cap["start_grid"]
    def replay(): 
        global _ENV; o=_ENV.reset(); return o
    plan=plan_to_satisfy(pred, g0, cap["model"], bg, replay)
    if not plan: return f"{game}: {cls} -> UNPLANNABLE (abstain) [correct if goal is non-spatial]"
    # execute from a FRESH L0 start; re-derive iff levels 0->1 DURING the plan
    o=_ENV.reset(); base=int(o.levels_completed or 0); won=False
    for a in plan:
        o=_ENV.step(GameAction.from_id(a))
        if int(o.levels_completed or 0)>base: won=True; break
        if o.state==GameState.GAME_OVER: break
    return f"{game}: {cls:8s} plan={len(plan):3d} -> {'RE-DERIVES L0 -> TRUST' if won else 'does NOT re-derive -> ABSTAIN'}"

for g in (sys.argv[1:] or ["ar25","m0r0","vc33","su15","cd82","tu93"]):
    try: print(" ", verify(g))
    except Exception as e: import traceback; print(f"  {g}: ERR {e}")
