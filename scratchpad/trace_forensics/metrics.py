#!/usr/bin/env python
"""Behavioral scorecard over parsed episodes -> metrics.json"""
import json, os, sys
import numpy as np
from collections import Counter, defaultdict

ROOT = "/Users/ahmed/Documents/ArcAGI3/scratchpad/trace_forensics"
PARSED = os.path.join(ROOT, "parsed")
KILL = "/Users/ahmed/Documents/ArcAGI3/scratchpad/killexp_data"

def episode_metrics(ep):
    d = json.load(open(os.path.join(PARSED, ep + ".json")))
    segs, acts, man = d["segments"], d["actions"], d["manifest"]
    all_recs = [rc for s in segs for rc in s["records"]]
    n_req = len(all_recs)
    if not n_req: return None

    # probes before first action(...) call
    probes_before_first_action = 0
    for rc in all_recs:
        if rc["n_action_calls"]: break
        probes_before_first_action += 1

    # inspect-only vs acting requests
    acting = [rc for rc in all_recs if rc["n_action_calls"]]
    inspect = [rc for rc in all_recs if not rc["n_action_calls"]]

    feat_frac = {}
    for f in ("segmentation", "ascii", "diff_frames", "bfs_path", "flood_fill",
              "button_rank", "history", "transitions"):
        feat_frac[f] = round(sum(1 for rc in all_recs if f in rc["feats"]) / n_req, 3)

    # actions per turn (analysis_step)
    by_step = defaultdict(list)
    for a in acts: by_step[a["analysis_step"]].append(a)
    apt = [len(v) for v in by_step.values()]
    batch_sizes = [a["batch_size"] for a in acts if a.get("batch_index") == 1 and a.get("batch_size")]

    # no-effect actions + perseveration
    noeff = [a for a in acts if a["board_changed"] is False]
    disp = [(a["kind"], a["row"], a["col"], a["name"]) for a in acts]
    dup_immediate = sum(1 for i in range(1, len(disp)) if disp[i] == disp[i-1])
    # longest run of identical consecutive actions
    longest = run = 1 if disp else 0
    for i in range(1, len(disp)):
        run = run + 1 if disp[i] == disp[i-1] else 1
        longest = max(longest, run)
    # repeat of an action that previously had no effect on same level
    noeff_keys = set()
    noeff_repeat = 0
    for a in acts:
        k = (a["level"], a["kind"], a["row"], a["col"], a["name"])
        if a["board_changed"] is False:
            if k in noeff_keys: noeff_repeat += 1
            noeff_keys.add(k)

    reasoning_chars = sum(rc["reasoning_chars"] for rc in all_recs)
    hypo = sum(rc["hypo_hits"] for rc in all_recs)
    revise = sum(rc["revise_hits"] for rc in all_recs)

    lv_completes = [a["action_num"] for a in acts if a["level_completed"]]
    n_levels = man["levels_completed"] or 0

    m = {
        "episode": ep, "game": man["game"],
        "levels": f"{man['levels_completed']}/{man['number_of_levels']}",
        "score": round(man.get("final_score") or 0, 1),
        "n_model_requests": n_req, "n_actions": len(acts),
        "probes_before_first_action": probes_before_first_action,
        "inspect_to_act_ratio": round(len(inspect) / max(len(acting), 1), 2),
        "actions_per_acting_turn_mean": round(np.mean(apt), 2) if apt else 0,
        "actions_per_acting_turn_max": max(apt) if apt else 0,
        "frac_actions_in_batch_ge3": round(sum(b for b in batch_sizes if b >= 3) / max(len(acts), 1), 3) if acts else 0,
        "feat_frac": feat_frac,
        "noeff_actions": len(noeff),
        "noeff_frac": round(len(noeff) / max(len(acts), 1), 3),
        "noeff_repeat_same_level": noeff_repeat,
        "immediate_dup_actions": dup_immediate,
        "longest_identical_run": longest,
        "hypo_per_1k_reasoning_chars": round(1000 * hypo / max(reasoning_chars, 1), 2),
        "revise_hits": revise,
        "reasoning_chars_per_action": round(reasoning_chars / max(len(acts), 1)),
        "actions_per_level_won": round(len(acts) / n_levels, 1) if n_levels else None,
        "first_win_action_num": lv_completes[0] if lv_completes else None,
    }
    return m

def killexp_click_audit(ep, game):
    """Level-1 clicks vs exhaustive ground truth (which is L1-reset based)."""
    z = np.load(os.path.join(KILL, game + ".npz"))
    rows = z["rows"]
    eff = {}
    for y, x, nch, far, spread, lv, over, win in rows:
        eff[(int(y), int(x))] = bool(nch > 0 or lv > 0 or over or win)
    d = json.load(open(os.path.join(PARSED, ep + ".json")))
    l1 = [a for a in d["actions"] if a["level"] == 1 and a["kind"] == "CLICK"]
    audited = [(a["action_num"], a["row"], a["col"], eff.get((a["row"], a["col"]))) for a in l1]
    n_eff = sum(1 for *_, e in audited if e)
    return {"episode": ep, "l1_clicks": len(l1), "l1_clicks_effective": n_eff,
            "l1_click_hit_rate": round(n_eff / max(len(l1), 1), 3),
            "first_10": audited[:10]}

if __name__ == "__main__":
    eps = [f[:-5] for f in sorted(os.listdir(PARSED)) if f.endswith(".json")]
    out = {"episodes": [], "killexp_audit": []}
    for ep in eps:
        m = episode_metrics(ep)
        if m: out["episodes"].append(m)
    for ep in ("k3_ft09", "control_ft09_27b", "teacher_ft09_instruct",
               "k3_sweep_ft09_b2", "ladder_ft09_k25"):
        if os.path.exists(os.path.join(PARSED, ep + ".json")):
            out["killexp_audit"].append(killexp_click_audit(ep, "ft09"))
    with open(os.path.join(ROOT, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for m in out["episodes"]:
        print(json.dumps({k: v for k, v in m.items() if k != "feat_frac"}))
    print("--- killexp ---")
    for a in out["killexp_audit"]:
        print(json.dumps(a))
