#!/usr/bin/env python
"""Count named K3 behaviors per episode + ft09 three-way early-turn table."""
import json, os, re
import numpy as np
from collections import defaultdict

ROOT = "/Users/ahmed/Documents/ArcAGI3/scratchpad/trace_forensics"
PARSED = os.path.join(ROOT, "parsed")

def load(ep):
    return json.load(open(os.path.join(PARSED, ep + ".json")))

def behaviors(ep):
    d = load(ep)
    segs, acts = d["segments"], d["actions"]
    recs = [rc for s in segs for rc in s["records"]]
    txt = lambda rc: (rc["reasoning"] or "") + " " + (rc["content"] or "")
    allcode = lambda rc: "\n".join(rc["codes"])

    out = {"episode": ep}
    # B1 opening: first request = segmentation inventory, no action
    out["opens_with_seg_inventory"] = bool(recs and "segmentation" in recs[0]["feats"] and not recs[0]["n_action_calls"])
    # B2 probe->verify->burst: single-action turn followed (within next 2 acting turns) by batch>=4
    by_step = defaultdict(list)
    for a in acts: by_step[a["analysis_step"]].append(a)
    steps = sorted(k for k in by_step if k is not None)
    sizes = [len(by_step[s]) for s in steps]
    b2 = sum(1 for i in range(len(sizes)-1) if sizes[i] <= 2 and sizes[i+1] >= 4)
    out["probe_then_burst_events"] = b2
    # B3 result-dict verification in acting code
    acting = [rc for rc in recs if rc["n_action_calls"]]
    out["acting_reqs"] = len(acting)
    out["acting_reqs_checking_result"] = sum(
        1 for rc in acting if re.search(r"last_action_result|level_completed|res\.get|r\.get|result\.get", allcode(rc)))
    # B4 explicit hypothesis naming / revision
    out["named_hypothesis_reqs"] = sum(1 for rc in recs if re.search(r"[Hh]ypothes|\bH[123]\b", txt(rc)))
    out["explicit_revision_reqs"] = sum(1 for rc in recs if re.search(
        r"hypothesis (?:was |is )?(?:wrong|failed)|failed too|revis(?:e|ing|ion)|reconsider|must be wrong", txt(rc), re.I))
    # B5 offline solver/search code (no action, loops/search, substantial)
    out["solver_code_reqs"] = sum(1 for rc in recs if not rc["n_action_calls"] and rc["code_chars"] > 350
                                  and re.search(r"\bfor \w+ in\b", allcode(rc)))
    # search-style solver among ALL reqs (consistency/offset/orientation search)
    out["consistency_search_reqs"] = sum(1 for rc in recs if re.search(
        r"offset|orientation|consisten|conflict|rotat", allcode(rc)) and re.search(r"\bfor \w+ in\b", allcode(rc)))
    # B6 revert/undo probes
    out["revert_mentions"] = sum(1 for rc in recs if re.search(r"\brevert|undo\b", txt(rc) + allcode(rc), re.I))
    # B7 cross-level rule carry
    out["cross_level_rule_reqs"] = sum(1 for rc in recs if re.search(
        r"cross-level|same mechanic|mechanic (?:is )?consistent|rule (?:confirmed|from L\d|generaliz)|as in (?:level|L)\d|consistent (?:with|—) (?:earlier|previous|L\d)", txt(rc), re.I))
    # B8 plan states predicted outcome before acting ("Predict"/"expect")
    out["prediction_reqs"] = sum(1 for rc in recs if re.search(r"\bpredict|expect(?:ed)?:? \b", txt(rc), re.I))
    # first effective action
    fe = next((a["action_num"] for a in acts if a["board_changed"]), None)
    out["first_board_changing_action"] = fe
    return out

def early_table(ep, n_segs=15):
    d = load(ep)
    by_step = defaultdict(list)
    for a in d["actions"]: by_step[a["analysis_step"]].append(a)
    rows = []
    seg_recs = d["segments"][:n_segs]
    step = 0
    for i, s in enumerate(seg_recs, 1):
        feats = sorted({f for rc in s["records"] for f in rc["feats"]})
        n_insp = sum(1 for rc in s["records"] if not rc["n_action_calls"])
        n_act_calls = sum(rc["n_action_calls"] for rc in s["records"])
        rows.append({"seg": i, "n_reqs": len(s["records"]), "inspect_reqs": n_insp,
                     "action_calls": n_act_calls, "feats": feats})
    return rows

if __name__ == "__main__":
    k3_eps = ["k3_ft09", "k3_sweep_ft09_b2", "k3_batch1_sb26", "k3_batch1_vc33", "k3_batch1_su15",
              "k3_batch1_cd82", "k3_sweep_ar25_a2", "k3_sweep_tu93_a2", "k3_sweep_vc33_a2",
              "k3_sweep_sc25_b2", "k3_sweep_re86_b2"]
    others = ["control_ft09_27b", "teacher_ft09_instruct", "ladder_ft09_k25", "ladder_ft09_k27code"]
    res = {"k3": [], "others": []}
    for ep in k3_eps: res["k3"].append(behaviors(ep))
    for ep in others: res["others"].append(behaviors(ep))
    json.dump(res, open(os.path.join(ROOT, "behavior_counts.json"), "w"), indent=1)
    for grp in ("k3", "others"):
        print(f"===== {grp} =====")
        for b in res[grp]:
            print(json.dumps(b))
