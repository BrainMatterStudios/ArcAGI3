#!/usr/bin/env python
"""Parse a rl_gate episode (trace.jsonl + viewer_data_events.jsonl) into a compact
structured summary for behavioral forensics. No boards/images retained."""
import json, re, sys, os
from collections import Counter

EPS_ROOT = "/Users/ahmed/Documents/ArcAGI3/scratchpad/rl_gate/episodes"
OUT_ROOT = "/Users/ahmed/Documents/ArcAGI3/scratchpad/trace_forensics/parsed"

CODE_FEATS = {
    "segmentation": r"\bsegmentation\b",
    "ascii": r"\.ascii\b",
    "diff_frames": r"\bdiff_frames\b",
    "bfs_path": r"\bbfs_path\b",
    "flood_fill": r"\bflood_fill\b",
    "button_rank": r"\bbutton_rank\b",
    "history": r"\bhistory\b",
    "transitions": r"\btransitions\b|\blast_transition\b",
    "action_call": r"\baction\(",
}
HYPO_RE = re.compile(r"\bhypothes|the rule\b|rule is\b|rule:\s|goal is\b|mechanic\b|I (?:believe|think|suspect)\b|likely\b|seems? (?:to|like)\b|pattern\b", re.I)
REVISE_RE = re.compile(r"\b(wrong|incorrect|didn'?t work|not the case|actually|instead|revise|re-?think|contradict|however|but wait|no effect|nothing changed|unchanged)\b", re.I)

def parse_action_display(disp):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp or "")
    if m: return ("CLICK", int(m.group(1)), int(m.group(2)))
    return (disp, None, None)

def extract_code(msg):
    out = []
    for tc in (msg.get("tool_calls") or []):
        try:
            args = json.loads(tc["function"]["arguments"])
            out.append(args.get("code", ""))
        except Exception:
            out.append(tc["function"].get("arguments", ""))
    return out

def parse(ep):
    d = os.path.join(EPS_ROOT, ep)
    man = json.load(open(os.path.join(d, "manifest.json")))
    recs = [json.loads(l) for l in open(os.path.join(d, "trace.jsonl"))]
    evp = os.path.join(d, "artifacts", "viewer_data_events.jsonl")
    events = [json.loads(l) for l in open(evp)] if os.path.exists(evp) else []

    # ---- model requests -> segments (a segment starts when last req msg is 'user')
    segments = []  # each: dict
    cur = None
    for r in recs:
        ms = r["request"]["messages"]
        if not r.get("response") or "choices" not in r["response"]:
            continue  # failed/aborted request
        msg = r["response"]["choices"][0]["message"]
        codes = extract_code(msg)
        reasoning = msg.get("reasoning") or ""
        content = msg.get("content") or ""
        is_new = ms[-1]["role"] == "user"
        rec_sum = {
            "seq": r["seq"], "latency_s": r.get("latency_s"),
            "reasoning_chars": len(reasoning), "content_chars": len(content),
            "code_chars": sum(len(c) for c in codes),
            "n_action_calls": sum(len(re.findall(r"\baction\(", c)) for c in codes),
            "feats": sorted({k for k, pat in CODE_FEATS.items()
                             for c in codes if re.search(pat, c)}),
            "hypo_hits": len(HYPO_RE.findall(reasoning + " " + content)),
            "revise_hits": len(REVISE_RE.findall(reasoning + " " + content)),
            "reasoning": reasoning, "content": content, "codes": codes,
        }
        if is_new or cur is None:
            cur = {"records": []}
            segments.append(cur)
        cur["records"].append(rec_sum)

    # ---- action events
    acts = []
    prev_ascii = None
    for e in events:
        if e.get("type") == "initial":
            prev_ascii = e.get("board_ascii")
        if e.get("type") != "action":
            continue
        kind, rr, cc = parse_action_display(e.get("action_display"))
        acts.append({
            "action_num": e["action_num"], "analysis_step": e["analysis_step"],
            "name": e.get("action_name"), "kind": kind, "row": rr, "col": cc,
            "level": e["level"], "board_changed": e.get("board_changed"),
            "level_completed": e.get("level_completed"),
            "game_over": e.get("game_over"), "score": e.get("score"),
            "batch_index": e.get("batch_index"), "batch_size": e.get("batch_size"),
        })

    out = {"episode": ep, "manifest": {k: man.get(k) for k in
            ("game", "levels_completed", "number_of_levels", "final_score",
             "n_turns", "n_actions", "n_model_requests", "state", "wall_s",
             "tokens_total", "tokens_generated")},
           "segments": segments, "actions": acts}
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, ep + ".json"), "w") as f:
        json.dump(out, f)
    # console summary
    n_act_recs = sum(1 for s in segments for rc in s["records"] if rc["n_action_calls"])
    print(f"{ep}: {len(recs)} reqs, {len(segments)} segments, {len(acts)} actions, "
          f"{man.get('levels_completed')}/{man.get('number_of_levels')} levels, state={man.get('state')}")

if __name__ == "__main__":
    for ep in sys.argv[1:]:
        parse(ep)
