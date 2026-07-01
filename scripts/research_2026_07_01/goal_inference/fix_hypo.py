"""Fix hypothesis: learned-CONTACT goal + color-based wall/hazard avoidance, tested through the RE-REACH flow
(replay L0 solution to reach L1) that previously failed. If L1 solves here, the crux is fixed."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np
from collections import deque
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.coroutine_strategy import CoroutineStrategy, general_agent_gen
from arcagi3.mechanics.world_model import fit_world_model
from arcagi3.mechanics.primitives import ReachColor, enumerate_goals

def centroid(grid, colors):
    ys, xs = np.where(np.isin(grid, list(colors)))
    return (int(round(ys.mean())), int(round(xs.mean()))) if len(ys) else None

def plan_reach(grid, ac, deltas, target_color, bg, blocked_colors):
    ys, xs = np.where(np.isin(grid, list(ac)))
    if not len(ys): return None
    start = (int(round(ys.mean())), int(round(xs.mean())))
    tgt = set(map(tuple, np.argwhere(grid == target_color)))
    if not tgt: return None
    def block(r, c):
        if not (0 <= r < 64 and 0 <= c < 64): return True
        v = int(grid[r, c])
        return v in blocked_colors and (r, c) not in tgt
    seen = {start}; q = deque([(start, [])])
    while q:
        (r, c), path = q.popleft()
        if any(abs(r-tr)+abs(c-tc) <= 3 for tr, tc in tgt): return path
        for a, (dr, dc) in deltas.items():
            nr, nc = r+dr, c+dc
            if (nr, nc) not in seen and not block(nr, nc):
                seen.add((nr, nc)); q.append(((nr, nc), path+[a]))
        if len(seen) > 8000: break
    return None

def run(game, budget=40000):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    env = c.make(game_id=gid, scorecard_id=f"fh-{game}"); obs = env.reset()
    pol = CoroutineStrategy(general_agent_gen); last = np.zeros((64,64),np.int8); steps = []
    for _ in range(budget):
        g = P.to_grid(obs.frame) if len(obs.frame) else last; last = g
        if int(obs.levels_completed or 0) >= 1: break
        tok = pol.decide(g, gstate_terminal=(obs.state==GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        prev = g
        if tok[0]=="reset": obs=env.reset()
        elif tok[0]=="S": obs=env.step(GameAction.from_id(tok[1]))
        else: obs=env.step(GameAction.ACTION6, data={"x":int(tok[1]),"y":int(tok[2])})
        nxt = P.to_grid(obs.frame) if len(obs.frame) else prev
        steps.append((prev, tok, nxt, int(obs.levels_completed or 0), obs.state==GameState.GAME_OVER))
    win = [i for i,s in enumerate(steps) if s[3] >= 1]
    if not win: print(f"  {game:>6}: L0 not solved"); return
    wi = win[0]; win_prev, _, win_grid, _, _ = steps[wi]
    # extract clean L0 solution = tokens from the last double-reset before the win
    dj = 0
    for j in range(0, wi):
        if steps[j][1][0]=="reset": dj = j
    sol = [steps[k][1] for k in range(dj+1, wi+1) if steps[k][1][0] != "reset"]
    trans = [(p, t[1], n) for (p,t,n,_,_) in steps if t[0]=="S" and t[1] in (1,2,3,4) and int(np.sum(p!=n))>0]
    terms = [(p, t) for (p,t,n,_,tm) in steps if tm]
    m = fit_world_model(trans)
    if m is None: print(f"  {game:>6}: fit=None"); return
    ac = m.agent_colors; bg = P.detect_background(win_grid)
    # LEARNED CONTACT GOAL: color at the avatar's END position (the cell it reached to win)
    contact = None
    wtok = steps[wi][1]
    if wtok[0]=="S" and wtok[1] in m.move.deltas:
        ap = centroid(win_prev, ac)
        if ap:
            dr, dc = m.move.deltas[wtok[1]]; cr, cc = ap[0]+dr, ap[1]+dc
            if 0 <= cr < 64 and 0 <= cc < 64: contact = int(win_prev[cr, cc])
    if contact is None or contact in ac or contact == bg:   # fallback: avatar END cell color
        ae = centroid(win_grid, ac)
        if ae: contact = int(win_prev[ae[0], ae[1]])
    goal_color = contact if (contact is not None and contact not in ac and contact != bg) else None
    # color-based walls + hazards (transfer across levels)
    wall_colors = {int(P.to_grid(steps[0][0])[r,c]) for (r,c) in list(m.move.walls)[:200]} if m.move.walls else set()
    g0 = steps[0][0]; wall_colors = {int(g0[r,c]) for (r,c) in m.move.walls} - {bg} - set(ac)
    hazard_colors = set()
    for (p, t) in terms:
        if t[0]=="S" and t[1] in m.move.deltas:
            ap = centroid(p, ac)
            if ap:
                dr, dc = m.move.deltas[t[1]]; hr, hc = ap[0]+dr, ap[1]+dc
                if 0 <= hr < 64 and 0 <= hc < 64:
                    hz = int(p[hr, hc])
                    if hz != bg and hz not in ac: hazard_colors.add(hz)
    blocked = (wall_colors | hazard_colors)
    # RE-REACH L1 via replaying the L0 solution (the flow that failed)
    obs = env.reset()
    for t in sol:
        obs = env.step(GameAction.from_id(t[1])) if t[0]=="S" else env.step(GameAction.ACTION6, data={"x":t[1],"y":t[2]})
    lvl = int(obs.levels_completed or 0)
    l1 = P.to_grid(obs.frame) if len(obs.frame) else win_grid
    diff = int(np.sum(l1 != win_grid))
    print(f"  {game:>6}: re-reach lvl={lvl} sol_len={len(sol)} | frame diff vs win-moment: {diff} px "
          f"(0 = sol reproduces exactly)")
    # DIAGNOSTIC: kill2's approach (enumerate_goals + model.plan_to) at the RE-REACHED L1
    k2 = "no-plan"
    cols = set(int(v) for v in l1.flatten())
    for goal in enumerate_goals(l1, ac, bg, cols)[:6]:
        p = m.plan_to(l1, goal)
        if p:
            for a in p:
                obs = env.step(GameAction.from_id(a))
                if int(obs.levels_completed or 0) > lvl: k2 = f"SOLVED via {goal.kind}({getattr(goal,'color','?')}) plan={len(p)}"; break
                if obs.state==GameState.GAME_OVER: break
            if k2.startswith("SOLVED"): break
            k2 = "planned-not-won"
    print(f"  {game:>6}: contact-goal={goal_color} walls={sorted(wall_colors)} re-reach lvl={lvl} | kill2-approach@re-reach: {k2}")
for g in (sys.argv[1:] or ["cd82","tu93","ar25"]):
    try: run(g)
    except Exception as e: import traceback; print(f"  {g}: ERR {e}"); traceback.print_exc()
