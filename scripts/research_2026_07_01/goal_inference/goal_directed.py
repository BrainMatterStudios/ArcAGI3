"""STAGE 2b — GOAL-DIRECTED SEARCH: use the induced predicate as a PROGRESS HEURISTIC to guide a best-first
search over moves+clicks toward satisfying it. Works for ANY mechanic (search discovers it) + ANY predicate with
a progress measure. Test: does it re-derive cd82's TEMPLATE win from L0 (the VERIFY gate for a structured goal)?"""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np, heapq
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H, stage1_v4 as V
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient
from arcagi3.coroutine_strategy import _exact_state
from arcengine import GameAction, GameState

def progress(pred, g):
    """higher = closer to satisfying pred."""
    if pred.cls=="template":
        dr,dc=pred.off
        return sum(1 for (r,c) in pred.cells if int(g[r,c])==int(g[r+dr,c+dc]))
    if pred.cls=="cover":
        return sum(1 for (r,c) in pred.Y if int(g[r,c])!=pred.yc)
    if pred.cls=="collect":
        return -int(np.count_nonzero(g==pred.c))
    if pred.cls=="reach":
        ys,xs=np.where(g==pred.c)
        return 0
    return 0

def _apply(env, tok):
    if tok[0]=="reset": return env.reset()
    if tok[0]=="S": return env.step(GameAction.from_id(tok[1]))
    return env.step(GameAction.ACTION6, data={"x":int(tok[1]),"y":int(tok[2])})

def directed_search(game, pred, prefix, base_level, max_nodes=4000, budget_s=90):
    import time; t0=time.time()
    env=H._mk(game)
    def reach():
        o=env.reset()
        for tk in prefix: o=_apply(env,tk)
        return o
    o=reach(); g0=P.to_grid(o.frame); avail=list(o.available_actions or [])
    acts=[("S",a) for a in avail if a in (1,2,3,4,5)]+[("C",cx,cy) for (cx,cy) in _salient(g0,10)]
    # best-first on -progress (maximize), tie by depth
    seen={_exact_state(g0)}; h0=progress(pred,g0)
    frontier=[(-h0,0,[])]; tie=0; nodes=0; best_h=h0
    while frontier and nodes<max_nodes and time.time()-t0<budget_s:
        negh,depth,seq=heapq.heappop(frontier)
        for a in acts:
            o=reach(); dead=won=False
            for tk in seq:
                o=_apply(env,tk)
                if int(o.levels_completed or 0)>base_level: won=True;break
                if o.state==GameState.GAME_OVER: dead=True;break
            if won: return seq, True, nodes
            if dead: continue
            o=_apply(env,a); nodes+=1
            if int(o.levels_completed or 0)>base_level: return seq+[a], True, nodes
            if o.state==GameState.GAME_OVER: continue
            g=P.to_grid(o.frame) if len(o.frame) else g0
            e=_exact_state(g)
            if e not in seen:
                seen.add(e); tie+=1; h=progress(pred,g); best_h=max(best_h,h)
                heapq.heappush(frontier,(-h,depth+1,seq+[a]))
    return None, False, nodes

# VERIFY on L0: directed search from L0 start should re-derive the win
for game in (sys.argv[1:] or ["cd82"]):
    cap=H.capture(game); surv,submit,wd,bg=V.induce(cap)
    if not surv: print(f"{game}: no predicate"); continue
    pred=surv[0][0]
    seq,won,nodes=directed_search(game, pred, [], 0)
    print(f"{game}: predicate={pred.cls}({pred.repr}) | directed L0 search -> "
          f"{'RE-DERIVES win in '+str(len(seq))+' actions (VERIFY PASS)' if won else 'did not re-derive'} (nodes={nodes})")
