"""MECHANIC LEARNING step 2 — GREEDY EFFECT-DRIVEN planner: each step, PROBE candidate actions from the current
accumulated state, pick the one that most increases the induced predicate progress (#covered), append, repeat.
The mechanic is learned online by probing. Measured k->k+1 with an independent recheck."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H, stage1_v4 as V, goal_directed as GD
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient
from arcagi3 import perception as _PC
from arcengine import GameAction, GameState

def _replay(env, toks):
    o=env.reset()
    for tk in toks: o=GD._apply(env,tk)
    return o

def greedy(game, max_steps=60, budget_s=180):
    t0=time.time()
    cap=H.capture(game); surv,_,_,_=V.induce(cap)
    if not surv or surv[0][2]!="cover": return f"{game}: not cover"
    yc=surv[0][0].yc; base=cap["base"]
    env=H._mk(game); o=_replay(env, cap["sol"]); g0=P.to_grid(o.frame)
    Y0=[tuple(p) for p in np.argwhere(g0==yc)]
    def covered(g): return sum(1 for (r,c) in Y0 if int(g[r,c])!=yc)
    avail=list(o.available_actions or [])
    accumulated=list(cap["sol"]); cur_cov=covered(g0); won=False
    for step in range(max_steps):
        if time.time()-t0>budget_s: break
        o=_replay(env, accumulated); g=P.to_grid(o.frame)
        if int(o.levels_completed or 0)>base: won=True; break
        # candidate actions re-perceived from the CURRENT state
        cands=[("S",a) for a in avail if a in (1,2,3,4,5)]
        if 6 in avail:
            rem=[(int(c),int(r)) for (r,c) in np.argwhere(g==yc)]
            bg2=_PC.detect_background(g)
            cents=[(int(round(o.centroid[1])),int(round(o.centroid[0]))) for o in _PC.connected_components(g,background=bg2)]
            seen_c=set(); cc=[]
            for (cx,cy) in _salient(g,24)+cents+rem[:12]:
                if (cx,cy) not in seen_c: seen_c.add((cx,cy)); cc.append((cx,cy))
            cands += [("C",cx,cy) for (cx,cy) in cc]
        best=None
        for tok in cands:
            o=_replay(env, accumulated); o=GD._apply(env, tok)
            if o.state==GameState.GAME_OVER: continue
            g2=P.to_grid(o.frame) if len(o.frame) else g
            dc=covered(g2)-cur_cov; adv=int(o.levels_completed or 0)>base
            score=(1000 if adv else 0)+dc
            if best is None or score>best[0]: best=(score,tok,covered(g2),adv)
        if best is None or (best[0]<=0 and not best[3]): break   # stuck
        accumulated.append(best[1]); cur_cov=best[2]
        if best[3]: won=True; break
    # independent recheck: fresh replay of accumulated, confirm k->k+1 in-window
    rc="n/a"
    if won:
        e2=H._mk(game); o2=e2.reset(); adv=None
        for i,tk in enumerate(accumulated):
            o2=GD._apply(e2,tk)
            if int(o2.levels_completed or 0)>base and i>=len(cap["sol"]): adv=i; break
        rc=f"IN-WINDOW k->k+1 at idx {adv} (post-prefix), final levels={int(o2.levels_completed or 0)}"
    return (f"{game}: greedy cover Y0={len(Y0)} covered0={covered(g0)} -> "
            f"{'SOLVED L1' if won else 'not solved (cov '+str(cur_cov)+'/'+str(len(Y0))+')'} "
            f"plan_len={len(accumulated)-len(cap['sol'])} | recheck: {rc}")

for g in (sys.argv[1:] or ["lp85","dc22"]):
    try: print(" ", greedy(g))
    except Exception as e: import traceback; print(f"  {g}: ERR {e}")
