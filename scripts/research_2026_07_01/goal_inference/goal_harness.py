"""False-positive-PROOF measurement harness for goal-inference experiments.
capture(): solve L0 with search-replay, record the exact L0 solution (sol), the WIN TRANSITION (pre/post frame +
  winning action), a sample of ORDINARY frames, the fitted delta model, and the base level index after L0.
evaluate(): re-reach L1 by replaying sol, run a candidate PLANNER, execute, and report SOLVED **only** if
  levels_completed advances PAST base DURING the executed actions. Includes a NEGATIVE control (reach-goal, must
  be NOT solved) and a POSITIVE control (search L1, must be SOLVED) so the harness itself is trustworthy."""
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np, heapq
from collections import deque, Counter
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.coroutine_strategy import CoroutineStrategy, general_agent_gen, _exact_state, _novelty_cell, _salient
from arcagi3.mechanics.world_model import fit_world_model

def _mk(game):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    return c.make(game_id=gid, scorecard_id=f"gh-{game}")

def _apply(env, tok):
    if tok[0] == "reset": return env.reset()
    if tok[0] == "S": return env.step(GameAction.from_id(tok[1]))
    return env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})

def capture(game, budget=40000):
    env = _mk(game); obs = env.reset(); pol = CoroutineStrategy(general_agent_gen)
    start_grid = P.to_grid(obs.frame) if len(obs.frame) else np.zeros((64,64),np.int8)
    last = np.zeros((64,64), np.int8); steps = []; ordinary = []
    for _ in range(budget):
        g = P.to_grid(obs.frame) if len(obs.frame) else last; last = g
        if int(obs.levels_completed or 0) >= 1: break
        tok = pol.decide(g, gstate_terminal=(obs.state==GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        prev = g; obs = _apply(env, tok)
        nxt = P.to_grid(obs.frame) if len(obs.frame) else prev
        steps.append((prev, tok, nxt, int(obs.levels_completed or 0), obs.state==GameState.GAME_OVER))
        if tok[0] != "reset" and len(ordinary) < 80 and int(np.sum(prev!=nxt))>0: ordinary.append(prev)
    wins = [i for i,s in enumerate(steps) if s[3] >= 1]
    if not wins: return None
    wi = wins[0]; win_prev, win_tok, win_grid, _, _ = steps[wi]
    dj = 0
    for j in range(0, wi):
        if steps[j][1][0] == "reset": dj = j
    sol = [steps[k][1] for k in range(dj+1, wi+1) if steps[k][1][0] != "reset"]
    trans = [(p, t[1], n) for (p,t,n,_,_) in steps if t[0]=="S" and t[1] in (1,2,3,4) and int(np.sum(p!=n))>0]
    model = fit_world_model(trans) if len(trans) >= 5 else None
    return dict(game=game, sol=sol, win_prev=win_prev, win_grid=win_grid, win_tok=win_tok, start_grid=start_grid,
                ordinary=ordinary, model=model, base=1,
                agent_colors=(model.agent_colors if model else set()))

def reach_l1(env, sol):
    """re-reach L1 (deterministic) by a fresh reset + replay of the L0 solution. Returns (obs, base)."""
    obs = env.reset()
    for tok in sol: obs = _apply(env, tok)
    return obs, int(obs.levels_completed or 0)

def evaluate(cap, planner, label=""):
    """planner(grid, cap) -> list of action ids (1..5) OR list of tokens. SOLVED iff levels advances past base
    DURING execution (NOT before)."""
    env = _mk(cap["game"]); obs, base = reach_l1(env, cap["sol"])
    if base < 1: return dict(label=label, ok=False, reason=f"re-reach failed (base={base})")
    grid = P.to_grid(obs.frame) if len(obs.frame) else cap["win_grid"]
    plan = planner(grid, cap)
    if not plan: return dict(label=label, ok=False, plan_len=0, reason="no plan")
    solved = False
    for a in plan:
        tok = a if isinstance(a, tuple) else ("S", a)
        obs = _apply(env, tok)
        if int(obs.levels_completed or 0) > base: solved = True; break
        if obs.state == GameState.GAME_OVER: break
    return dict(label=label, ok=solved, plan_len=len(plan), base=base,
                final=int(obs.levels_completed or 0))

# ---- controls ----
def reach_planner(grid, cap):
    """NEGATIVE control: plan the avatar to the salient/contact color (the thing that FAILED)."""
    m = cap["model"];
    if m is None: return None
    from arcagi3.mechanics.primitives import enumerate_goals
    bg = P.detect_background(grid); cols = set(int(v) for v in grid.flatten())
    for goal in enumerate_goals(grid, m.agent_colors, bg, cols)[:6]:
        p = m.plan_to(grid, goal)
        if p: return p
    return None

def search_l1_planner_factory(cap):
    """POSITIVE control: SEARCH L1 (novelty over moves+clicks) re-reaching per candidate; returns the winning
    action list. Proves L1 is solvable AND the harness can report a TRUE success."""
    def planner(grid0, cap):
        acts = [("S",a) for a in (1,2,3,4)] + [("C",cx,cy) for (cx,cy) in _salient(grid0, 12)]
        env = _mk(cap["game"]); 
        seen=set(); frontier=[(0,0,[])]; cellv=Counter(); tie=0; nodes=0; base=cap["base"]
        while frontier and nodes < 3000:
            _,_,seq = heapq.heappop(frontier)
            obs,_ = reach_l1(env, cap["sol"]); ok=True
            for a in seq:
                obs = _apply(env, a)
                if int(obs.levels_completed or 0)>base: return seq
                if obs.state==GameState.GAME_OVER: ok=False; break
            if not ok: continue
            for a in acts:
                obs,_ = reach_l1(env, cap["sol"]); dead=False
                for b in seq:
                    obs=_apply(env,b)
                    if obs.state==GameState.GAME_OVER: dead=True;break
                if dead: continue
                obs=_apply(env,a); nodes+=1
                if int(obs.levels_completed or 0)>base: return seq+[a]
                if obs.state==GameState.GAME_OVER: continue
                g2=P.to_grid(obs.frame) if len(obs.frame) else grid0
                e=_exact_state(g2)
                if e not in seen: seen.add(e); tie+=1; heapq.heappush(frontier,(cellv[_novelty_cell(g2)],tie,seq+[a]))
        return None
    return planner

if __name__ == "__main__":
    import sys
    for game in (sys.argv[1:] or ["cd82","tu93","ar25"]):
        cap = capture(game)
        if cap is None: print(f"  {game}: L0 not solved"); continue
        neg = evaluate(cap, reach_planner, "reach(neg-ctrl)")
        pos = evaluate(cap, search_l1_planner_factory(cap), "search(pos-ctrl)")
        print(f"  {game}: base={cap['base']} model={'ok' if cap['model'] else 'None'} | "
              f"NEG reach -> solved={neg['ok']} (want False) | POS search -> solved={pos['ok']} (want True)")

def sanity_positive(cap):
    """GUARANTEED true transition: replay sol[:-1], then the final L0 action MUST cause 0->1. If evaluate-style
    detection reports solved here, the harness can report a true win (not always-false)."""
    env = _mk(cap["game"]); obs = env.reset()
    for tok in cap["sol"][:-1]: obs = _apply(env, tok)
    base = int(obs.levels_completed or 0)
    obs = _apply(env, cap["sol"][-1])
    return dict(base_before_final=base, final=int(obs.levels_completed or 0),
                detects_win=(int(obs.levels_completed or 0) > base))
