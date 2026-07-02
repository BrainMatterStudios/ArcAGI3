"""Offline efficiency A/B: score a policy under the REAL squared RHAE (min(1.15,(base/act)^2)), max-over-plays,
using the engine's OWN scorecard tracking (trustworthy) + env baseline_actions. Reports linear (old gate) AND
squared (real eval) side by side so efficiency gains below baseline are VISIBLE.
Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python ab_efficiency.py <policy> <budget> [games...]
  policy: portfolio | salience | geodesic
"""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, os
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

POLICY = sys.argv[1] if len(sys.argv) > 1 else "portfolio"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
GAMES = sys.argv[3:] or ["tu93","vc33","cd82","lf52","m0r0","su15","ar25","lp85"]

def make_policy():
    if POLICY == "salience":
        from arcagi3.salience_explorer import SalienceExplorer
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    if POLICY == "geodesic":
        from arcagi3.geodesic_replay_explorer import GeodesicReplayExplorer
        return GeodesicReplayExplorer(seed=0, trust_threshold=3, border_mask=2)
    from arcagi3.portfolio_policy import PortfolioPolicy
    return PortfolioPolicy(seed=0)

def lin(base, act): return min(100.0, base/act*100.0) if act and act>0 else 0.0
def sq(base, act):  return min(1.15, (base/act)**2) if act and act>0 else 0.0

def run_game(client, e, budget):
    gid = e.game_id
    baselines = list(getattr(e, "baseline_actions", []) or [])
    card = client.open_scorecard(tags=[f"abeff-{POLICY}-{gid}"])
    env = client.make(game_id=gid, scorecard_id=card)
    import numpy as np
    pol = make_policy()
    obs = env.reset(); n = 0; last = np.zeros((64,64), np.int8)
    while n < budget:
        if obs.state == GameState.WIN: break
        if hasattr(pol, "_last_full_reset"):
            pol._last_full_reset = bool(getattr(obs, "full_reset", False))
        g = P.to_grid(obs.frame) if len(obs.frame) else last
        last = g
        tok = pol.decide(g, obs.state == GameState.GAME_OVER, obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        if tok[0] == "reset": obs = env.reset()
        elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
        else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
    sc = client.get_scorecard(card)
    return sc, baselines, n

def official_run_score(la, completed_flags, baselines):
    """EXACT reference-toolkit formula: iterate ALL game levels; unreached => completed=False (score 0) but
    weight (level_index=i+1) STILL counts. score = min(weighted_avg, max_weights/total_weights*100)."""
    total_score = total_weights = max_weights = 0.0
    for i, base in enumerate(baselines):
        w = i + 1
        if i < len(la) and i < len(completed_flags) and completed_flags[i] and la[i] > 0:
            s = min(115.0, (base / la[i]) ** 2 * 100.0)
        else:
            s = 0.0
        total_score += s * w; total_weights += w
        if s > 0: max_weights += w
    if total_weights == 0: return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100.0)

def score_runs(sc, baselines):
    """official per-run score (depth-weighted, all-levels denominator); game score = MAX over runs."""
    envs = getattr(sc, "environments", None) or []
    best = 0.0; best_run = None; runs = []
    for envsc in envs:
        for run in (getattr(envsc, "runs", None) or []):
            la = getattr(run, "level_actions", None) or []
            ls = getattr(run, "level_scores", None) or []
            completed = [ (i < len(ls) and ls[i] > 0) for i in range(len(la)) ]
            rs = official_run_score(la, completed, baselines)
            runs.append((la, rs));
            if rs > best: best = rs; best_run = (la, rs)
    n_levels = len(baselines)
    return best, best_run, runs, n_levels

def main():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    all_envs = {e.game_id.split("-")[0]: e for e in client.get_environments()}
    print(f"policy={POLICY} budget={BUDGET}  OFFICIAL depth-weighted RHAE (ref toolkit, max-over-runs)\n")
    print(f"{'game':>6} {'score':>7} {'#lv':>4}  best-run per-level actions vs baseline")
    print("-"*70)
    ts=[]
    for g in GAMES:
        e = all_envs.get(g)
        if not e: print(f"{g:>6}  (not found)"); continue
        sc, baselines, n = run_game(client, e, BUDGET)
        bs, br, runs, nlv = score_runs(sc, baselines)
        ts.append(bs)
        acts = br[0] if br else []
        print(f"{g:>6} {bs:>7.2f} {nlv:>4}  base={baselines}\n{'':>13}act={acts}")
    print("-"*70)
    if ts: print(f"MEAN score = {sum(ts)/len(ts):.2f} / 100  (official depth-weighted RHAE)")

if __name__ == "__main__":
    main()
