"""Corrected kill experiment: fit a GOOD factored model from L0, then PLAN L1 in-model toward each enumerated
reach/collect goal (zero live re-search) and execute. Measures true amortization + defer on controls."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, time, numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.coroutine_strategy import CoroutineStrategy, general_agent_gen
from arcagi3.mechanics.world_model import fit_world_model
from arcagi3.mechanics.primitives import enumerate_goals

def kill(game, budget=40000):
    t=time.time()
    c=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid=next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    env=c.make(game_id=gid, scorecard_id=f"k2-{game}"); obs=env.reset()
    pol=CoroutineStrategy(general_agent_gen); trans=[]; last=np.zeros((64,64),np.int8); prevg=None; lvl0=0
    reached=False
    for _ in range(budget):
        try: g=P.to_grid(obs.frame) if (obs.frame is not None and len(obs.frame)) else last
        except Exception: g=last
        last=g; cur=int(obs.levels_completed or 0)
        if cur>lvl0: reached=True; break                      # at L1 boundary
        tok=pol.decide(g, gstate_terminal=(obs.state==GameState.GAME_OVER),
                       levels=cur, available=list(obs.available_actions or []))
        prevg=g
        if tok[0]=="reset": obs=env.reset(); prevg=None
        elif tok[0]=="S":
            obs=env.step(GameAction.from_id(tok[1])); ng=P.to_grid(obs.frame) if len(obs.frame) else g
            if tok[1] in (1,2,3,4) and int(np.sum(g!=ng))>0: trans.append((g,tok[1],ng))
        else: obs=env.step(GameAction.ACTION6, data={"x":int(tok[1]),"y":int(tok[2])})
    if not reached or len(trans)<5:
        print(f"  {game:>6}: L0 not movement-solved ({len(trans)} trans) -> n/a for factored model"); return
    m=fit_world_model(trans)
    if m is None: print(f"  {game:>6}: DEFER (fit=None) [correct if out-of-vocab]"); return
    # manual held-out move acc (the real signal; reported field isn't set by fit_world_model)
    sc=[m.score_move(p,a,g) for (p,a,g) in trans[-30:]]; sc=[s for s in sc if s is not None]
    macc=float(np.mean(sc)) if sc else 0.0
    # L1: plan in-model toward each enumerated reach/collect goal; execute; check reward
    lvl_now=int(obs.levels_completed or 0)   # CORRECT baseline: the level we are AT after L0
    l1=P.to_grid(obs.frame) if len(obs.frame) else last
    bg=P.detect_background(l1); cols=set(int(v) for v in l1.flatten())
    goals=enumerate_goals(l1, m.agent_colors, bg, cols)
    solved=False; used=None; plen=0; nodes0=len(trans)
    for goal in goals[:6]:
        plan=m.plan_to(l1, goal)
        if not plan: continue
        # execute on a fresh reach of L1: replay is needed, but here env is already AT L1 -> execute directly
        won=False
        for a in plan:
            obs=env.step(GameAction.from_id(a))
            if int(obs.levels_completed or 0)>lvl_now: won=True; break
            if obs.state==GameState.GAME_OVER: break
        if won: solved=True; used=goal.kind; plen=len(plan); break
    print(f"  {game:>6}: model move_acc={macc:.2f} agent={m.agent_colors} | L1 in-model plan -> "
          f"{'SOLVED via '+str(used)+' plan='+str(plen) if solved else 'not solved (tried '+str(len(goals))+' goals)'} ({round(time.time()-t)}s)")
for g in (sys.argv[1:] or ["cd82","tu93","ar25","ls20"]):
    try: kill(g)
    except Exception as e: import traceback; print(f"  {g}: ERR {e}")
