"""Decisive test of the Goal Extraction Engine: does goal-directed navigation reach a navigation
game's L1 in ~human-efficient actions (<<salience's 7961 on ls20)?

Pipeline (no learning): probe to identify the agent (the small object with a consistent per-action
displacement) + its action->delta model; read target candidates from the scene graph; greedily move
the agent toward the nearest target; on stall, cycle to the next target / re-probe. Pure symbolic
perception + greedy planning.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/reach_target_test.py [game] [budget]
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
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("rt"))


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def centroids(g, bg):
    out = {}
    for col in np.unique(g):
        if int(col) == bg:
            continue
        ys, xs = np.where(g == col)
        out[int(col)] = (ys.mean(), xs.mean(), len(ys))
    return out


def step(env, aid):
    return _retry(lambda: env.step(GameAction.from_id(aid)))


def identify_agent(env, bg, simple):
    """Probe each simple action; agent = small-ish color with the most consistent nonzero displacement.
    Returns (agent_color, {aid: (drow, dcol)}). Resets between probes for clean deltas."""
    deltas = {}
    cand = {}
    for aid in simple:
        obs = _retry(env.reset)
        g0 = P.to_grid(obs.frame); c0 = centroids(g0, bg)
        obs = step(env, aid); g1 = P.to_grid(obs.frame); c1 = centroids(g1, bg)
        for col in c1:
            if col in c0 and c0[col][2] <= 25:   # small-ish (token, not terrain)
                d = (c1[col][0] - c0[col][0], c1[col][1] - c0[col][1])
                if abs(d[0]) > 0.3 or abs(d[1]) > 0.3:
                    cand.setdefault(col, {})[aid] = (round(d[0], 1), round(d[1], 1))
    # agent = the color that moves on the MOST actions with the LARGEST coherent steps
    if not cand:
        return None, {}
    agent = max(cand, key=lambda col: (len(cand[col]),
                                       sum(abs(a) + abs(b) for a, b in cand[col].values())))
    return agent, cand[agent]


def run(game, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["rt"]))
    obs = _retry(env.reset)
    bg = P.detect_background(P.to_grid(obs.frame))
    avail = [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])]
    simple = [a for a in [1, 2, 3, 4] if a in avail]

    agent, model = identify_agent(env, bg, simple)
    print(f"{game}: agent_color={agent} action_model={model}", flush=True)
    if agent is None or len(model) < 2:
        print(f"{game}: agent not cleanly identified -> reach_target N/A", flush=True)
        return

    obs = _retry(env.reset)
    n = budget_used = 0
    tried_targets = []
    while n < budget:
        g = P.to_grid(obs.frame)
        if obs.state == GameState.WIN:
            break
        if obs.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            obs = _retry(env.reset); continue
        if int(obs.levels_completed or 0) >= 1:
            print(f"{game}: REACHED L1 in {n} actions  (salience needs ~7961)", flush=True)
            return
        cs = centroids(g, bg)
        if agent not in cs:
            obs = step(env, simple[0]); n += 1; continue
        ar, ac, _ = cs[agent]
        scene = SG.extract(g, bg)
        targets = [t["centroid"] for t in scene["target_candidates"]] or \
                  [(o["centroid"]) for o in scene["objects"] if o["size"] <= 9 and o["color"] != agent]
        if not targets:
            obs = step(env, simple[0]); n += 1; continue
        # nearest target the agent hasn't already sat on
        tr, tc = min(targets, key=lambda t: (t[0] - ar) ** 2 + (t[1] - ac) ** 2)
        # greedy: pick action whose delta most reduces distance to (tr,tc)
        best_a, best_d = None, None
        cur_d = (tr - ar) ** 2 + (tc - ac) ** 2
        for aid, (dr, dc) in model.items():
            nd = (tr - (ar + dr)) ** 2 + (tc - (ac + dc)) ** 2
            if best_d is None or nd < best_d:
                best_d, best_a = nd, aid
        if best_d is not None and best_d >= cur_d - 0.1:
            # no action reduces distance (blocked) -> take any other simple action to dislodge
            others = [a for a in simple if a != best_a]
            best_a = others[n % len(others)] if others else best_a
        obs = step(env, best_a); n += 1
    print(f"{game}: did NOT reach L1 in {budget} (reach_target greedy)", flush=True)


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    t0 = time.time()
    run(game, budget)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
