#!/usr/bin/env python
"""Render a readable per-request digest of a parsed episode for qualitative reading."""
import json, os, re, sys

ROOT = "/Users/ahmed/Documents/ArcAGI3/scratchpad/trace_forensics"

def clip(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n] + ("..." if len(s) > n else "")

def main(ep, max_segs=999, rlen=400, clen=300, codelen=260):
    d = json.load(open(os.path.join(ROOT, "parsed", ep + ".json")))
    acts_by_step = {}
    for a in d["actions"]:
        acts_by_step.setdefault(a["analysis_step"], []).append(a)
    lines = [f"=== {ep} === {d['manifest']}"]
    seg_i = 0
    acting_i = 0
    for s in d["segments"]:
        seg_i += 1
        if seg_i > max_segs: break
        lines.append(f"\n--- SEGMENT {seg_i} ---")
        for rc in s["records"]:
            lines.append(f"[req {rc['seq']}] feats={rc['feats']} n_action_calls={rc['n_action_calls']}")
            if rc["reasoning"]:
                lines.append(f"  THINK: {clip(rc['reasoning'], rlen)}")
            if rc["content"]:
                lines.append(f"  SAY: {clip(rc['content'], clen)}")
            for c in rc["codes"]:
                lines.append(f"  CODE: {clip(c, codelen)}")
            if rc["n_action_calls"]:
                acting_i += 1
        # attach actions for the matching analysis step(s) heuristically by order
    # actions listed separately with analysis_step
    lines.append("\n=== ACTIONS (by analysis_step) ===")
    for st in sorted(acts_by_step, key=lambda v: (v is None, v)):
        row = acts_by_step[st]
        summ = ", ".join(f"{a['kind']}({a['row']},{a['col']})" if a['kind']=='CLICK' else str(a['name'])
                         for a in row)
        fl = "".join("W" if a["level_completed"] else ("x" if a["board_changed"] is False else ".") for a in row)
        lines.append(f"step {st} L{row[0]['level']}: n={len(row)} [{fl}] {clip(summ, 300)}")
    out = os.path.join(ROOT, "digests", ep + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines))
    print(out, len(lines), "lines")

if __name__ == "__main__":
    for ep in sys.argv[1:]:
        main(ep)
