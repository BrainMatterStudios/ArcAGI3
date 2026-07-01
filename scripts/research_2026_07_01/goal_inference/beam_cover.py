"""MECHANIC LEARNING step 3 — BEAM search over probed effects (escapes greedy local optima). Keep top-K action
sequences by #covered; expand each with re-perceived candidates; until a level advances. Measured k->k+1 fresh."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H, stage1_v4 as V, goal_directed as GD
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient
from arcengine import GameAction, GameState

def _replay(env,toks):
    o=env.reset()
    for tk in toks: o=GD._apply(env,tk)
    return o

def beam(game, K=5, depth=12, budget_s=240):
    t0=time.time(); cap=H.capture(game); surv,_,_,_=V.induce(cap)
    if not surv or surv[0][2]!="cover": return f"{game}: not cover"
    yc=surv[0][0].yc; base=cap["base"]; sol=cap["sol"]
    env=H._mk(game); o=_replay(env,sol); g0=P.to_grid(o.frame); avail=list(o.available_actions or [])
    Y0=[tuple(p) for p in np.argwhere(g0==yc)]
    def covered(g): return sum(1 for (r,c) in Y0 if int(g[r,c])!=yc)
    def cand(g):
        cs=[("S",a) for a in avail if a in (1,2,3,4,5)]
        if 6 in avail:
            rem=[(int(c),int(r)) for (r,c) in np.argwhere(g==yc)]
            bg=P.detect_background(g)
            cents=[(int(round(o.centroid[1])),int(round(o.centroid[0]))) for o in P.connected_components(g,background=bg)]
            seen=set(); out=[]
            for t in _salient(g,20)+cents+rem[:10]:
                if t not in seen: seen.add(t); out.append(("C",t[0],t[1]))
            cs+=out
        return cs
    beam=[(covered(g0), [])]                       # (cov, extra_actions)
    seen_cov={}
    for d in range(depth):
        if time.time()-t0>budget_s: break
        nxt=[]
        for cov,extra in beam:
            o=_replay(env, sol+extra); g=P.to_grid(o.frame)
            for tok in cand(g):
                o=_replay(env, sol+extra); o=GD._apply(env,tok)
                if o.state==GameState.GAME_OVER: continue
                if int(o.levels_completed or 0)>base:
                    return _finish(game, sol, extra+[tok], base, len(Y0))
                g2=P.to_grid(o.frame) if len(o.frame) else g; c2=covered(g2)
                nxt.append((c2, extra+[tok]))
        if not nxt: break
        nxt.sort(key=lambda r:-r[0])
        # dedup by coverage signature to keep the beam diverse
        beam=[]; sigs=set()
        for c2,ex in nxt:
            if c2 not in sigs or len([1 for s in sigs if s==c2])<2:
                beam.append((c2,ex)); sigs.add(c2)
            if len(beam)>=K: break
        best=beam[0][0]
    return f"{game}: beam cover -> not solved (best {beam[0][0]}/{len(Y0)}) depth-used<= {depth}"

def _finish(game, sol, extra, base, ny):
    e2=H._mk(game); o2=e2.reset(); adv=None
    for i,tk in enumerate(sol+extra):
        o2=GD._apply(e2,tk)
        if int(o2.levels_completed or 0)>base and i>=len(sol): adv=i; break
    return f"{game}: beam cover -> SOLVED L1 (k->k+1) plan_extra={len(extra)} | recheck IN-WINDOW idx={adv} final={int(o2.levels_completed or 0)}"

for g in (sys.argv[1:] or ["lp85"]):
    try: print(" ", beam(g))
    except Exception as e: import traceback; print(f"  {g}: ERR {e}")
