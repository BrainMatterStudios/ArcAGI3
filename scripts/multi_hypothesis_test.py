"""Option 1: multi-hypothesis verifying planner (user-directed 2026-06-22).

The planner-layer failure was committing to ONE hypothesis when appearance underdetermines the goal.
Fix: generate several candidate objectives from the scene graph, run each one's bounded mini-planner
in-game, and KEEP whichever actually produces a level-up. Bounded inference cost, only on level 1.

Tests planners: reach_target, interact_at_target (nav then interact — the ls20 suspicion),
collect_all (visit collectibles), click_targets, complete_symmetry. Reports which (if any) reaches
L1 and in how many actions vs salience's baseline.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/multi_hypothesis_test.py [game] [budget_each]
"""
from __future__ import annotations
import logging, sys, time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3 import scene_graph as SG

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("mh"))


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def grid_safe(obs):
    try:
        return P.to_grid(obs.frame)
    except Exception:
        return None


def centroids(g, bg):
    out = {}
    for col in np.unique(g):
        if int(col) == bg:
            continue
        ys, xs = np.where(g == col)
        out[int(col)] = (ys.mean(), xs.mean(), len(ys))
    return out


def identify_agent(env, bg, simple):
    cand = {}
    for aid in simple:
        obs = _retry(env.reset); g0 = grid_safe(obs)
        if g0 is None:
            continue
        c0 = centroids(g0, bg)
        obs = _retry(lambda: env.step(GameAction.from_id(aid))); g1 = grid_safe(obs)
        if g1 is None:
            continue
        c1 = centroids(g1, bg)
        for col in c1:
            if col in c0 and c0[col][2] <= 25:
                d = (c1[col][0] - c0[col][0], c1[col][1] - c0[col][1])
                if abs(d[0]) > 0.3 or abs(d[1]) > 0.3:
                    cand.setdefault(col, {})[aid] = (d[0], d[1])
    if not cand:
        return None, {}
    agent = max(cand, key=lambda col: (len(cand[col]),
                                       sum(abs(a) + abs(b) for a, b in cand[col].values())))
    return agent, cand[agent]


def step(env, tok):
    if tok[0] == "S":
        return _retry(lambda: env.step(GameAction.from_id(tok[1])))
    return _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))


def nav_action(agent_pos, target, model):
    ar, ac = agent_pos
    tr, tc = target
    cur = (tr - ar) ** 2 + (tc - ac) ** 2
    best_a, best_d = None, None
    for aid, (dr, dc) in model.items():
        nd = (tr - (ar + dr)) ** 2 + (tc - (ac + dc)) ** 2
        if best_d is None or nd < best_d:
            best_d, best_a = nd, aid
    return best_a, (best_d is not None and best_d < cur - 0.1)


def run_planner(env, name, agent, model, bg, simple, budget):
    """Execute one hypothesis planner for up to `budget` actions; return actions-to-L1 or None."""
    obs = _retry(env.reset)
    interact_acts = [a for a in [5, 2] if a in simple or a == 5]   # interact candidates
    n = 0
    while n < budget:
        g = grid_safe(obs)
        if g is None or obs.state == GameState.GAME_OVER:
            obs = _retry(env.reset); n += 1; continue
        if int(obs.levels_completed or 0) >= 1:
            return n
        scene = SG.extract(g, bg)
        cs = centroids(g, bg)
        apos = (cs[agent][0], cs[agent][1]) if agent in cs else None
        targets = [t["centroid"] for t in scene["target_candidates"]]
        collions = []
        for col in scene["collectibles"]:
            for o in scene["objects"]:
                if o["color"] == col["color"] and o["size"] <= 9:
                    collions.append(o["centroid"])

        if name == "click_targets":
            tg = targets or [o["centroid"] for o in scene["objects"] if o["size"] <= 9]
            if not tg:
                obs = step(env, ("S", simple[0])); n += 1; continue
            r, cc = tg[n % len(tg)]
            obs = step(env, ("C", int(cc), int(r))); n += 1; continue

        if name == "complete_symmetry":
            # greedy: cycle simple actions (already shown weak, included for completeness)
            obs = step(env, ("S", simple[n % len(simple)])); n += 1; continue

        # navigation-based planners need an agent
        if apos is None or not model:
            return None
        if name == "collect_all":
            goals = collions or targets
        else:
            goals = targets
        if not goals:
            obs = step(env, ("S", simple[0])); n += 1; continue
        tr, tc = min(goals, key=lambda t: (t[0] - apos[0]) ** 2 + (t[1] - apos[1]) ** 2)
        on_target = (tr - apos[0]) ** 2 + (tc - apos[1]) ** 2 <= 6
        if name == "interact_at_target" and on_target:
            # at the target -> try interacting (5 then 2)
            for ia in (5, 2):
                obs = step(env, ("S", ia)); n += 1
                if grid_safe(obs) is not None and int(obs.levels_completed or 0) >= 1:
                    return n
                if obs.state == GameState.GAME_OVER:
                    break
            continue
        ba, progress = nav_action(apos, (tr, tc), model)
        if not progress:
            others = [a for a in simple if a != ba]
            ba = others[n % len(others)] if others else ba
        obs = step(env, ("S", ba)); n += 1
    return None


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    t0 = time.time()
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["mh"]))
    obs = _retry(env.reset); bg = P.detect_background(grid_safe(obs))
    avail = [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])]
    simple = [a for a in [1, 2, 3, 4] if a in avail]
    agent, model = identify_agent(env, bg, simple)
    print(f"{game}: agent={agent} model={ {k:(round(v[0],1),round(v[1],1)) for k,v in model.items()} } avail={avail}", flush=True)
    planners = ["reach_target", "interact_at_target", "collect_all", "click_targets", "complete_symmetry"]
    for name in planners:
        try:
            res = run_planner(env, name, agent, model, bg, simple, budget)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True); continue
        print(f"  {name}: {'L1 in '+str(res)+' actions' if res else 'no L1 in '+str(budget)}", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
