"""MECHANIC LEARNING step 1 — EFFECT PROBER (frames-only): from L1, probe candidate actions and measure each
one's effect on the induced predicate's progress (#covered) + the frame delta. Learns which action(s) advance
the goal = the covering mechanic. Single actions first; if none cover, probe two-action compounds."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H, stage1_v4 as V, goal_directed as GD
from arcagi3 import perception as P
from arcagi3.general_search_strategy import _salient
from arcengine import GameAction, GameState

def reach(game, sol):
    env=H._mk(game); o=env.reset()
    for tk in sol: o=GD._apply(env,tk)
    return env,o

def probe(game):
    cap=H.capture(game); surv,_,_,_=V.induce(cap)
    if not surv or surv[0][2]!="cover": return f"{game}: not cover"
    yc=surv[0][0].yc; base=cap["base"]
    env,o=reach(game, cap["sol"]); g0=P.to_grid(o.frame); bg=P.detect_background(g0)
    Y0=[tuple(p) for p in np.argwhere(g0==yc)]
    def covered(g): return sum(1 for (r,c) in Y0 if int(g[r,c])!=yc)
    c0=covered(g0); avail=list(o.available_actions or [])
    print(f"\n{game}: exit={yc} |Y0|={len(Y0)} covered0={c0} avail={avail}")
    # candidate actions: moves/A5 + clicks at exits, at salient objects, and a coarse grid
    clicks=[(int(c),int(r)) for (r,c) in Y0[:6]]                       # click ON exits
    clicks+= [(cx,cy) for (cx,cy) in _salient(g0,12)]                  # click salient objects
    results=[]
    for a in [x for x in avail if x in (1,2,3,4,5)]:
        env,o=reach(game,cap["sol"]); o=env.step(GameAction.from_id(a))
        g=P.to_grid(o.frame) if len(o.frame) else g0
        results.append((f"move{a}", covered(g)-c0, int(np.sum(g!=g0)), int(o.levels_completed or 0)>base))
    if 6 in avail:
        for (x,y) in clicks:
            env,o=reach(game,cap["sol"]); o=env.step(GameAction.ACTION6,data={"x":x,"y":y})
            g=P.to_grid(o.frame) if len(o.frame) else g0
            results.append((f"click({x},{y})", covered(g)-c0, int(np.sum(g!=g0)), int(o.levels_completed or 0)>base))
    # report actions with the biggest coverage gain
    results.sort(key=lambda r:-r[1])
    print("  top single-action effects (Δcovered, Δframe, won):")
    for name,dcov,dfr,won in results[:6]: print(f"    {name:16s} Δcov={dcov:+d} Δframe={dfr:4d} won={won}")
    best=results[0]
    if best[1] <= 0:
        print("  -> NO single action increases coverage; mechanic is COMPOUND (2-step) or indirect")
        # probe two-action compounds: (select-click on a salient) then (click on an exit)
        sal=[(cx,cy) for (cx,cy) in _salient(g0,10)]
        found=None
        for (sx,sy) in sal[:8]:
            for (r,c) in Y0[:8]:
                env,o=reach(game,cap["sol"]); o=env.step(GameAction.ACTION6,data={"x":sx,"y":sy})
                if o.state==GameState.GAME_OVER: continue
                o=env.step(GameAction.ACTION6,data={"x":int(c),"y":int(r)})
                g=P.to_grid(o.frame) if len(o.frame) else g0
                if covered(g)-c0 > 0:
                    found=((sx,sy),(int(c),int(r)),covered(g)-c0); break
            if found: break
        print(f"  compound select->place probe: {'FOUND '+str(found) if found else 'none found (mechanic is not select-then-place-on-exit)'}")
    else:
        print(f"  -> single-action mechanic: {best[0]} covers (+{best[1]})")

for g in (sys.argv[1:] or ["lp85","s5i5","dc22"]):
    try: probe(g)
    except Exception as e: import traceback; print(f"  {g}: ERR {e}")
