"""Counterfactual: if we replayed each duck level-win as (winning attempt only) in a fresh play,
what per-game score would the replay play get vs the actual pass-0 score?
Winning attempt per level = actions after the last RESET within that level's history slice.
Replay play cost per level = 1 (RESET prefix, engine counts it) + winning-attempt actions.
"""
import json, sys, glob

def score(levels_scores):  # depth-weighted, per arc_agi
    tw = sum(i+1 for i in range(len(levels_scores)))
    ts = sum(s*(i+1) for i, s in enumerate(levels_scores))
    mw = sum((i+1) for i, s in enumerate(levels_scores) if s > 0)
    if tw == 0: return 0.0
    return min(ts/tw, mw/tw*100)

for path in sorted(glob.glob(sys.argv[1] + "/*/benchmark.json")):
    bm = json.load(open(path))
    print("=== ", path)
    tot_orig, tot_replay, n = 0.0, 0.0, 0
    for run in bm["game_runs"]:
        base = run.get("base_actions_per_level")
        apl = run["actions_per_level"]
        lc = run["levels_completed"]
        hist = run["history"]
        if not base or lc == 0: continue
        # slice history by level
        idx = 0; orig_scores = []; replay_scores = []
        for L, a in enumerate(apl):
            seg = hist[idx:idx+a]; idx += a
            if L < lc and a > 0:
                orig_scores.append(min(115.0, (base[L]/a)**2*100))
                # winning attempt: after last RESET in segment
                last_reset = -1
                for i, rec in enumerate(seg):
                    if rec["action"]["id"] == "RESET": last_reset = i
                win_len = len(seg) - (last_reset + 1)
                cost = win_len + (1 if last_reset >= 0 else 0)  # replay RESET prefix only if needed
                cost = max(cost, 1)
                replay_scores.append(min(115.0, (base[L]/cost)**2*100))
            else:
                orig_scores.append(0.0); replay_scores.append(0.0)
        so, sr = score(orig_scores), score(replay_scores)
        tot_orig += so; tot_replay += max(so, sr); n += 1
        if sr > so * 1.05:
            print(f"  {run['game_id']}: lvls {lc}/{run['number_of_levels']} orig {so:.2f} -> replay {sr:.2f}")
    if n: print(f"  TOTAL over {n} scoring games: orig {tot_orig:.1f} -> max(orig,replay) {tot_replay:.1f}  ({tot_replay/max(tot_orig,1e-9):.2f}x)")
