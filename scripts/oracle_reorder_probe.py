"""BET 2 — ORACLE-REORDER CEILING PROBE for the click-affordance lever (next-paradigm build plan,
2026-06-28). KILL-TEST, NOT A BUILD. No training, no model: a pure replay over the banked policy's
offline trajectories.

The merged #1 ("decode-tufa") proposal is a within-level change-affordance head that learns which
click cell will change the board, so the explorer stops wasting actions on no-op (dead) clicks before
the level-up. Its OWN decisive weakness: it pays efficiency only on levels already COMPLETED, and the
scored games wall at L0/L1 -> the classic "dev-efficiency is Kaggle-inert" reduction that killed
CAIPrune (0.28). So we first measure ONLY the zero-training ORACLE CEILING: give the explorer a
PERFECT change-affordance oracle (it skips every click that would not change the masked state) and
recompute the per-level efficiency. A real Conv2d head can only do WORSE than this ceiling, so if the
ceiling is thin the whole arm dies here, in an afternoon, with no model code.

Definitions (faithful to the existing harness):
- no-op click = a click whose edge is a self-loop in the explorer graph: masked state key unchanged
  (exactly the CAIPrune / Phase-E ~51%-no-op definition; uses the explorer's own `_key`).
- The change-affordance oracle removes EVERY no-op click within a level (reorders them to "after" the
  level-up = never needs them). The level-up action is causal (reward>0 => state changed) so it is
  never a no-op and always survives. Removing a no-op click leaves the state -> trajectory unchanged,
  so the saving is exact and independent per level => this is a true UPPER BOUND for the lever.
- per-level RHAE = min(1.15, (200/actions)^2) (the eval_efficiency.py proxy; ref=200 = "human" scale).
- per-game weighted-RHAE (Kaggle-style) = sum_L( L * S_L ) / sum_L( L ) over COMPLETED levels,
  weight L = the 1-indexed level number (deep levels weighted heavily, per docs.arcprize methodology).

Pre-registered decision (from the build plan):
- decisive number = mean per-game WEIGHTED-RHAE gain on the HOLDOUT split (the L0/L1 scored-set
  proxies). If < ~0.05 absolute -> DEAD, same wall as "dev-efficiency Kaggle-inert" (most likely).
- only if > ~0.15 -> build the within-level Conv2d head and test held-out AUC >> 0.5.

Usage:
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/oracle_reorder_probe.py [budget] [policy] [games]
  budget default 12000; policy default transfer-dense (the banked submission config);
  games default = the eval TUNE+HOLDOUT split.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402

logging.basicConfig(level=logging.ERROR)

REF = 200.0
CAP = 1.15

# eval_efficiency.py split. HOLDOUT = the unseen-game proxy; these wall at L0/L1 = the scored set.
TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]


def s_level(actions: float) -> float:
    """Per-level squared-efficiency score, ref=200 actions (the eval_efficiency proxy)."""
    return min(CAP, (REF / max(actions, 1.0)) ** 2)


def oracle_gain(levels: list[tuple[int, int]]) -> dict:
    """levels = [(actual_actions, noop_clicks), ...] in completion order (1-indexed level = position+1).
    Returns the actual vs perfect-change-affordance-oracle efficiency, unweighted (summed proxy,
    comparable to eval_efficiency sum_eff) and weighted (Kaggle per-game score, weight = level ordinal).
    """
    if not levels:
        return {"n_levels": 0, "unweighted_actual": 0.0, "unweighted_oracle": 0.0,
                "unweighted_gain": 0.0, "weighted_actual": 0.0, "weighted_oracle": 0.0,
                "weighted_gain": 0.0}
    ua = uo = wa = wo = wsum = 0.0
    for i, (actual, noop) in enumerate(levels):
        ordinal = i + 1
        oracle = max(actual - noop, 1)  # oracle skips every no-op click in this level
        sa, so = s_level(actual), s_level(oracle)
        ua += sa
        uo += so
        wa += ordinal * sa
        wo += ordinal * so
        wsum += ordinal
    return {
        "n_levels": len(levels),
        "unweighted_actual": round(ua, 3), "unweighted_oracle": round(uo, 3),
        "unweighted_gain": round(uo - ua, 3),
        "weighted_actual": round(wa / wsum, 3), "weighted_oracle": round(wo / wsum, 3),
        "weighted_gain": round((wo - wa) / wsum, 3),
    }


def make_policy(name: str):
    if name == "transfer-dense":
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2,
                                coarse_grid_step=4, max_click_targets=256)
    if name == "salience-dense":
        from arcagi3.salience_explorer import SalienceExplorer
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2,
                                coarse_grid_step=4, max_click_targets=256)
    if name == "transfer":
        from arcagi3.transfer_explorer import TransferExplorer
        return TransferExplorer(seed=0, trust_threshold=3, border_mask=2)
    raise ValueError(f"unknown policy {name}")


def run_game(client, prefix: str, policy_name: str, budget: int):
    """Drive the policy OFFLINE; per completed level return (actual_actions, noop_clicks).
    A click is a no-op when its graph edge is a self-loop: the explorer's masked key is unchanged
    from the state the click was taken in to the resulting state (read off pol.prev_key across steps).
    """
    try:
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    except StopIteration:
        return None
    env = client.make(game_id=gid, scorecard_id=f"oracle-{prefix}")
    pol = make_policy(policy_name)
    obs = env.reset()
    n = 0
    best = 0
    cur_actions = 0          # actions spent in the current (in-progress) level
    cur_noop_clicks = 0      # no-op clicks spent in the current level
    levels: list[tuple[int, int]] = []
    # to score the PREVIOUS action's edge we need the key it was taken from and the resulting key.
    prev_key_before = None   # masked key the last action was issued from
    prev_was_click = False
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        # pol.prev_key is now the masked key of THIS state (the state `tok` is issued from), and is
        # the resulting key of the PREVIOUS action -> classify the previous action's edge now.
        cur_key = pol.prev_key
        if prev_was_click and prev_key_before is not None and cur_key == prev_key_before:
            cur_noop_clicks += 1   # self-loop edge => no-op click
        if tok == ("reset",):
            obs = env.reset()
            prev_was_click = False
            prev_key_before = None
            continue
        prev_key_before = cur_key
        prev_was_click = tok[0] == "C"
        if tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        cur_actions += 1
        lv = int(obs.levels_completed or 0)
        if lv > best:
            # the action just taken closed the level; it is causal (not counted as no-op).
            levels.append((cur_actions, cur_noop_clicks))
            best = lv
            cur_actions = 0
            cur_noop_clicks = 0
    return {"levels": levels, "total_actions": n, "states": len(pol.nodes)}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    policy = sys.argv[2] if len(sys.argv) > 2 else "transfer-dense"
    if len(sys.argv) > 3:
        splits = [("CUSTOM", sys.argv[3].split(","))]
    else:
        splits = [("TUNE", TUNE), ("HOLDOUT", HOLDOUT)]
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("oracle"))
    print(f"BET 2 oracle-reorder ceiling probe | policy={policy} budget={budget}")
    print("per-level: actual=actions, noop=no-op clicks before level-up, oracle=actual-noop "
          "(perfect change-affordance ceiling)\n")
    for split, games in splits:
        per_game_weighted = []
        per_game_unweighted = []
        print(f"==== {split} ====")
        for g in games:
            r = run_game(client, g, policy, budget)
            if r is None:
                print(f"  {g:>6}: (not found)")
                continue
            res = oracle_gain(r["levels"])
            if res["n_levels"] == 0:
                print(f"  {g:>6}: 0 levels @ budget (no completed level -> oracle inert)")
                continue
            per_game_weighted.append(res["weighted_gain"])
            per_game_unweighted.append(res["unweighted_gain"])
            detail = "  ".join(
                f"L{i+1}[a={a},noop={no},orc={max(a-no,1)}]" for i, (a, no) in enumerate(r["levels"]))
            print(f"  {g:>6}: lvls={res['n_levels']}  wRHAE {res['weighted_actual']:.3f}->"
                  f"{res['weighted_oracle']:.3f} (+{res['weighted_gain']:.3f})  | {detail}")
        if per_game_weighted:
            mw = float(np.mean(per_game_weighted))
            mu = float(np.mean(per_game_unweighted))
            verdict = ("KILL (<0.05: dev-efficiency Kaggle-inert)" if mw < 0.05
                       else "GREY (0.05-0.15: weak, do not build head)" if mw < 0.15
                       else "PROCEED (>0.15: build the within-level Conv2d head)")
            print(f"  -- {split} mean per-game WEIGHTED-RHAE gain = {mw:.3f}  "
                  f"(unweighted {mu:.3f}, n={len(per_game_weighted)}) -> {verdict}")
        print()


if __name__ == "__main__":
    main()
