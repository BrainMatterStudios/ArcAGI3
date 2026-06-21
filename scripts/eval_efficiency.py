"""Kaggle-predictive evaluation harness (efficiency-aware + held-out).

WHY: dev *levels* don't predict the Kaggle score (33/35/36 dev levels all = 0.33), because the
real metric is SQUARED action-efficiency per level: S_level = min(1.15, (human/agent_actions)^2),
averaged over HIDDEN games. We can't see human baselines, but we CAN track actions-to-each-level
(the controllable quantity) and a relative efficiency proxy, on a TUNE/HOLDOUT split so we catch
overfitting (the failure mode that collapsed the StochasticGoose leader 12.58%->0.25%).

Outputs per game: levels reached + actions-to-each-level (RHAE-style first-exposure cost).
Aggregate: mean levels and a normalized efficiency score on TUNE vs HOLDOUT separately. A change
is only "good" if it improves HOLDOUT efficiency without dropping HOLDOUT levels.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/eval_efficiency.py [budget] [policy]
"""
from __future__ import annotations
import logging, sys, json, time
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("eval"))

# Fixed split so the metric is comparable across runs (chosen arbitrarily-but-stable, not tuned).
# HOLDOUT approximates "unseen games" — the only number that should drive ship/no-ship decisions.
TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]


def make_policy(name: str):
    if name == "salience":
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name in ("tour-yield", "tour-dfs"):
        from arcagi3.tour_explorer import TourExplorer
        mode = "yield" if name == "tour-yield" else "dfs"
        return TourExplorer(seed=0, trust_threshold=3, border_mask=2, frontier_mode=mode)
    if name == "transfer":
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name == "transfer-dense":  # the ACTUAL submission config (dense click lattice)
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2,
                                coarse_grid_step=4, max_click_targets=256)
    if name == "transfer-promote":  # ablation: promote matches but DON'T demote the rest
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2, transfer_demote=0)
    if name == "transfer-shape":  # ablation: signature = color + size-bucket
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2, sig_mode="colorshape")
    if name == "salience-dense":  # v6 with the same dense lattice (fair baseline)
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2,
                                coarse_grid_step=4, max_click_targets=256)
    if name == "prior":
        from arcagi3.prior_explorer import PriorExplorer
        return PriorExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name == "relational":
        from arcagi3.relational_explorer import RelationalExplorer
        return RelationalExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name in ("relational8", "relational2"):
        from arcagi3.relational_explorer import RelationalExplorer
        q = 8 if name == "relational8" else 2
        return RelationalExplorer(seed=0, trust_threshold=3, border_mask=2, rel_quant=q)
    if name == "combo2":  # transfer + relational rel_quant=2 (finer)
        from arcagi3.transfer_relational_explorer import TransferRelationalExplorer
        return TransferRelationalExplorer(seed=0, trust_threshold=3, border_mask=2, rel_quant=2)
    if name == "slide":
        from arcagi3.slide_nav_explorer import SlideNavExplorer
        return SlideNavExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name == "goalslide":
        from arcagi3.goal_slide_explorer import GoalDirectedSlideExplorer
        return GoalDirectedSlideExplorer(seed=0, trust_threshold=3, border_mask=2, stall_trigger=0)
    if name in ("salient", "salient3"):
        from arcagi3.salient_state_explorer import SalientStateExplorer
        return SalientStateExplorer(seed=0, trust_threshold=3, border_mask=2,
                                    rare_max=2 if name == "salient" else 3)
    if name in ("combo", "transfer-rel"):
        from arcagi3.transfer_relational_explorer import TransferRelationalExplorer
        return TransferRelationalExplorer(seed=0, trust_threshold=3, border_mask=2)
    raise ValueError(name)


def _retry(fn, tries=5, delay=1.0):
    """Retry an API call through transient three.arcprize.org timeouts (flaky network)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"    [retry {i+1}/{tries}] API error: {type(e).__name__}", flush=True)
            time.sleep(delay * (i + 1))


def run_game(prefix: str, policy_name: str, budget: int):
    try:
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    except StopIteration:
        return None
    card = client.open_scorecard(tags=["eval"]); env = client.make(game_id=gid, scorecard_id=card)
    pol = make_policy(policy_name)
    obs = _retry(env.reset); n = 0; best = 0; marks = []; last = 0
    while n < budget:
        st = obs.state
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(st == GameState.GAME_OVER),
                         gstate_notplayed=(st == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > best:
            marks.append(n - last); last = n; best = lv
    return {"levels": best, "actions_to_level": marks, "total_actions": n}


def efficiency_score(marks):
    """Relative efficiency proxy in [0,1+]: rewards FEWER actions per completed level.
    No human baseline available, so use a fixed reference of 200 actions/level (~cheap-level scale);
    per-level min(1.15, (ref/actions)^2), summed. Comparable across policies; NOT the literal Kaggle
    number, but moves the same direction (fewer actions -> higher)."""
    ref = 200.0
    return round(sum(min(1.15, (ref / max(a, 1)) ** 2) for a in marks), 3)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    policy = sys.argv[2] if len(sys.argv) > 2 else "salience"
    t0 = time.time()
    results = {}
    for split, games in (("TUNE", TUNE), ("HOLDOUT", HOLDOUT)):
        for g in games:
            r = run_game(g, policy, budget)
            if r is None:
                print(f"  {g}: NOT IN DEV SET", flush=True); continue
            r["eff"] = efficiency_score(r["actions_to_level"])
            results[g] = {**r, "split": split}
            print(f"  [{split}] {g}: L{r['levels']} eff={r['eff']} acts/lvl={r['actions_to_level']}", flush=True)
    for split in ("TUNE", "HOLDOUT"):
        rs = [v for v in results.values() if v["split"] == split]
        if rs:
            print(f"== {split}: mean_levels={np.mean([v['levels'] for v in rs]):.2f} "
                  f"sum_eff={sum(v['eff'] for v in rs):.2f} (n={len(rs)})", flush=True)
    print(f"\nelapsed {time.time()-t0:.0f}s  budget={budget}  policy={policy}", flush=True)
    out = f"/tmp/eval_{policy}_{budget}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
