"""EXP 1 — EFFICIENCY AUDIT. For each level our coverage explorer solves, compute the ACTUAL offline
per-level score using the REAL human baseline_actions (present in all 25 local envs), and compare against
the score an efficiency-optimal (geodesic-replay) agent would get. Answers: is efficiency a REAL score lever
(current << achievable) or inert (current ~ achievable)? Uses BOTH the SDK-linear score min(100,base/act*100)
and the OFFICIAL squared RHAE min(1.15,(base/act)^2).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scratchpad/efficiency_audit.py [budget] [game,game,...]
"""
from __future__ import annotations
import logging, sys
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
logging.basicConfig(level=logging.ERROR)

DEFAULT = ["vc33","cd82","tu93","lp85","lf52","ar25","sp80","su15","m0r0","tr87","ls20"]


def geodesic(nodes, start, goal):
    if start == goal: return 0
    if start not in nodes: return None
    seen = {start}; q = deque([(start, 0)])
    while q:
        k, d = q.popleft(); node = nodes.get(k)
        if not node: continue
        for a, (nk, _r) in node.edges.items():
            if nk == goal: return d + 1
            if nk not in seen: seen.add(nk); q.append((nk, d + 1))
    return None


def lin(base, act):   # SDK offline per-level score, 0..100
    return min(100.0, base / act * 100.0) if act > 0 else 0.0

def rhae(base, act):  # official squared RHAE, capped 1.15
    return min(1.15, (base / act) ** 2) if act > 0 else 0.0


def run(game, budget, baselines):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("eff"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"eff-{game}")
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); n = 0; prev_levels = 0; resets = 0
    marks = []
    while n < budget:
        if obs.state == GameState.WIN: break
        tok = pol.decide(P.to_grid(obs.frame), obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        src_key, src_action = pol.prev_key, pol.prev_action
        if tok == ("reset",):
            obs = env.reset(); resets += 1
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > prev_levels and src_key is not None and src_action is not None:
            marks.append((n, src_key, src_action)); prev_levels = lv
    rows = []
    start = pol.root_key; prev_n = 0
    for i, (an, src, act) in enumerate(marks):
        g = geodesic(pol.nodes, start, src)
        actual = an - prev_n
        base = baselines[i] if i < len(baselines) else None
        rows.append((i, actual, g, base))
        nxt = pol.nodes.get(src)
        start = nxt.edges.get(act, (start, 0))[0] if nxt else start
        prev_n = an
    return rows, resets


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    games = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    base_map = {}
    for e in client.get_environments():
        base_map[e.game_id.split("-")[0]] = list(getattr(e, "baseline_actions", []) or [])

    print(f"EXP 1 — EFFICIENCY AUDIT (SalienceExplorer, budget {budget}). "
          f"score=min(100,base/act*100); rhae=min(1.15,(base/act)^2)\n")
    hdr = f"{'game':>6} {'lvl':>3} {'human':>6} {'actual':>7} {'geo':>5} {'headrm':>7} " \
          f"{'sc_act':>7} {'sc_geo':>7} {'rhae_act':>9} {'rhae_geo':>9}"
    print(hdr); print("-"*len(hdr))
    agg = {"sc_act":[], "sc_geo":[], "rhae_act":[], "rhae_geo":[]}
    tot_resets = 0
    for gname in games:
        baselines = base_map.get(gname, [])
        rows, resets = run(gname, budget, baselines)
        tot_resets += resets
        if not rows:
            print(f"{gname:>6}  (0 levels @ budget; resets={resets})"); continue
        for (lv, actual, g, base) in rows:
            if base is None:
                print(f"{gname:>6} {lv:>3}  (no baseline for this level idx)"); continue
            sca = lin(base, actual); scg = lin(base, g) if g else 0.0
            ra = rhae(base, actual); rg = rhae(base, g) if g else 0.0
            hr = (actual / g) if g else None
            agg["sc_act"].append(sca); agg["sc_geo"].append(scg)
            agg["rhae_act"].append(ra); agg["rhae_geo"].append(rg)
            hrs = f"{hr:6.1f}x" if hr else "  n/a"
            gs = str(g) if g is not None else "unrch"
            print(f"{gname:>6} {lv:>3} {base:>6} {actual:>7} {gs:>5} {hrs:>7} "
                  f"{sca:>7.2f} {scg:>7.2f} {ra:>9.4f} {rg:>9.4f}")
    print("-"*len(hdr))
    n = len(agg["sc_act"])
    if n:
        print(f"\nN levels scored = {n}   total resets across games = {tot_resets}")
        print(f"MEAN per-level SDK score   actual = {np.mean(agg['sc_act']):.2f} / 100   "
              f"geodesic-ceiling = {np.mean(agg['sc_geo']):.2f} / 100   "
              f"=> {np.mean(agg['sc_geo'])/max(1e-9,np.mean(agg['sc_act'])):.1f}x recoverable")
        print(f"MEAN per-level RHAE(sq)    actual = {np.mean(agg['rhae_act']):.4f}       "
              f"geodesic-ceiling = {np.mean(agg['rhae_geo']):.4f}       "
              f"=> {np.mean(agg['rhae_geo'])/max(1e-9,np.mean(agg['rhae_act'])):.1f}x recoverable")
        print("\nRead: if geodesic-ceiling >> actual, EFFICIENCY IS A REAL SCORE LEVER (current score is left "
              "on the table by over-exploration). If ~equal, efficiency is inert.")


if __name__ == "__main__":
    main()
