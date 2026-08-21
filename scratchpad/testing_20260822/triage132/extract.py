#!/usr/bin/env python3
"""Re-derive first-level-completion wall-times on COMPETITION-GEOMETRY (7920s) runs.

Sources:
  A. 20260815-135107-shipped (Qwen3.8 shipped, 28 sessions, 7920s boxes)
     - rows: offkaggle/results/20260815-135107-shipped/patch_closure_result.json
       fields: rows[].clone_id, source_game, levels_completed, actions_per_level, actions_total
     - timing: /Users/ahmed/Documents/ArcAGI3/taaf_harness_artifacts/transcripts/k0NN_p0.txt
       last appended segment; per-turn headers '--- analysis_step=S | action=A | HH:MM:SS |'
       (A = 1-based index of the NEXT action; executed-so-far = A-1), plus
       'run_elapsed_seconds': X prints for t0 calibration.
  B. ft09_ablation ft09-base / ft09-struct (Qwen3.6, 28+28 sessions, 7920s, single game ft09)
     - same schema; transcripts inside each dir's taaf_harness_artifacts/transcripts.
  C. smoke benchmark.json corpus (duck38-v12, 4h-class boxes) — per-action
     history[i].wallclock_seconds; first completion = history[apl[0]-1].wallclock_seconds.

First completion bracket (transcript sessions): completion of level 1 happens at
cumulative action apl[0]; t_lo = last header with A <= apl[0]; t_hi = first header
with A >= apl[0]+1. Elapsed = header_time - t0, with t0 = median(header_time -
run_elapsed_seconds) over calibration pairs in the segment (fallback: wave start /
first header).
"""
import json, os, re, glob
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__))
HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d) \|")
ELAPSED = re.compile(r"'run_elapsed_seconds': ([0-9.]+)")
LC_TRUE = re.compile(r"'level_completed': True")

def parse_transcript(path):
    """Return last-segment events: headers [(t_sec, action_next, line)], calib [(t_hdr, elapsed)],
    completion marker lines [(line, t_hdr_before)]."""
    headers = []      # (line_no, step, action, t)
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    events = []
    for i, line in enumerate(lines):
        m = HDR.match(line)
        if m:
            t = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
            events.append(("hdr", i, int(m.group(1)), int(m.group(2)), t))
            continue
        for mm in ELAPSED.finditer(line):
            events.append(("el", i, float(mm.group(1)), None, None))
        if LC_TRUE.search(line):
            events.append(("lc", i, None, None, None))
    # segment split on header restarts
    segs = [[]]
    prev_step, prev_t = None, None
    for ev in events:
        if ev[0] == "hdr":
            step, t = ev[2], ev[4]
            if prev_t is not None:
                dt = t - prev_t
                if dt < 0 and -dt > 43200:
                    t += 86400  # midnight wrap; keep monotone
                    dt = t - prev_t
                restart = (step < prev_step) or (dt < -3600)
                if restart:
                    segs.append([])
                    t = ev[4]  # reset wrap base
            prev_step, prev_t = step, t
            segs[-1].append(("hdr", ev[1], step, ev[3], t))
        else:
            if segs[-1] or len(segs) == 1:
                segs[-1].append(ev)
    seg = segs[-1]
    # unwrap midnight inside final segment
    hdrs = []
    base = None
    lastt = None
    for ev in seg:
        if ev[0] == "hdr":
            t = ev[4]
            if lastt is not None and t < lastt - 43200:
                t += 86400
            lastt = t
            hdrs.append((ev[1], ev[3], t))  # (line, action_next, t)
    calib = []
    lcs = []
    cur_hdr_t = None
    hi = 0
    for ev in seg:
        if ev[0] == "hdr":
            cur_hdr_t = hdrs[hi][2]; hi += 1
        elif ev[0] == "el" and cur_hdr_t is not None:
            calib.append((cur_hdr_t, ev[2]))
        elif ev[0] == "lc" and cur_hdr_t is not None:
            lcs.append((ev[1], cur_hdr_t))
    return hdrs, calib, lcs

def first_completion_minutes(path, apl0, t0_fixed=None):
    """Return (lo_min, hi_min, t0_used, n_headers, quality) or None if no headers."""
    hdrs, calib, lcs = parse_transcript(path)
    if not hdrs:
        return None
    if t0_fixed is not None:
        t0 = t0_fixed
        q = "wave_start"
    elif calib:
        t0 = median(h - e for h, e in calib)
        q = f"calib_n{len(calib)}"
    else:
        t0 = hdrs[0][2]
        q = "first_header"
    lo = max((t for (_l, a, t) in hdrs if a <= apl0), default=None)
    hi = min((t for (_l, a, t) in hdrs if a >= apl0 + 1), default=None)
    return dict(lo=None if lo is None else (lo - t0) / 60.0,
                hi=None if hi is None else (hi - t0) / 60.0,
                t0=t0, n_headers=len(hdrs), quality=q,
                last_action=max(a for (_l, a, _t) in hdrs),
                seg_span_min=(hdrs[-1][2] - hdrs[0][2]) / 60.0,
                n_lc_markers=len(lcs))

def hhmmss(s):
    return s // 3600 % 24, s // 60 % 60, s % 60

sessions = []

# ---- A + B: patch_closure waves with transcripts ----
WAVES = [
    dict(name="w38_shipped", model="qwen3.8", arm="shipped",
         result="/Users/ahmed/Documents/ArcAGI3/offkaggle/results/20260815-135107-shipped/patch_closure_result.json",
         tdir="/Users/ahmed/Documents/ArcAGI3/taaf_harness_artifacts/transcripts",
         t0_fixed=13 * 3600 + 51 * 60 + 7),  # dir timestamp 20260815-135107
    dict(name="w36_ft09_base", model="qwen3.6", arm="ft09_base",
         result="/Users/ahmed/Documents/ArcAGI3/scratchpad/ft09_ablation/ft09-base/patch_closure_result.json",
         tdir="/Users/ahmed/Documents/ArcAGI3/scratchpad/ft09_ablation/ft09-base/taaf_harness_artifacts/transcripts",
         t0_fixed=None),
    dict(name="w36_ft09_struct", model="qwen3.6", arm="ft09_struct",
         result="/Users/ahmed/Documents/ArcAGI3/scratchpad/ft09_ablation/ft09-struct/patch_closure_result.json",
         tdir="/Users/ahmed/Documents/ArcAGI3/scratchpad/ft09_ablation/ft09-struct/taaf_harness_artifacts/transcripts",
         t0_fixed=None),
]

for w in WAVES:
    d = json.load(open(w["result"]))
    for r in d["rows"]:
        apl = r.get("actions_per_level") or []
        levels = r.get("levels_completed", 0)
        tp = os.path.join(w["tdir"], f"{r['clone_id']}_p0.txt")
        rec = dict(corpus=w["name"], model=w["model"], arm=w["arm"],
                   clone=r["clone_id"], game=r["source_game"], levels=levels,
                   levels_total=r.get("levels_total"),
                   actions_total=r.get("actions_total"),
                   apl0=(apl[0] if apl else None),
                   box_min=round((r.get("wallclock_s") or 0) / 60, 1),
                   state=r.get("state"), first_lo=None, first_hi=None, notes="")
        if levels > 0 and os.path.exists(tp) and apl:
            fc = first_completion_minutes(tp, apl[0], w["t0_fixed"])
            if fc:
                rec["first_lo"], rec["first_hi"] = fc["lo"], fc["hi"]
                rec["t0_quality"] = fc["quality"]
                rec["n_headers"] = fc["n_headers"]
                rec["last_action_hdr"] = fc["last_action"]
                rec["seg_span_min"] = round(fc["seg_span_min"], 1)
                rec["n_lc_markers"] = fc["n_lc_markers"]
        elif levels > 0:
            rec["notes"] = "no transcript"
        sessions.append(rec)

# ---- C: smoke benchmark.json corpus (4h-class boxes) ----
SMOKE_ROOT = "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad"
for bf in sorted(glob.glob(os.path.join(SMOKE_ROOT, "*", "benchmark.json"))):
    d = json.load(open(bf))
    label = d.get("label", "")
    for gr in d.get("game_runs", []):
        h = gr.get("history") or []
        if not h or h[-1].get("wallclock_seconds") is None:
            continue
        wall = h[-1]["wallclock_seconds"] / 60.0
        apl = gr.get("actions_per_level") or []
        levels = gr.get("levels_completed", 0)
        first = None
        if levels > 0 and apl and 0 < apl[0] <= len(h):
            first = h[apl[0] - 1]["wallclock_seconds"] / 60.0
        sessions.append(dict(corpus="smoke:" + os.path.basename(os.path.dirname(bf)),
                             model="qwen3.8-v12", arm=label,
                             clone=None, game=gr["game_id"][:4], levels=levels,
                             levels_total=gr.get("number_of_levels"),
                             actions_total=len(h), apl0=(apl[0] if apl else None),
                             box_min=round(wall, 1), state=gr.get("state"),
                             first_lo=first, first_hi=first, notes="benchmark_history"))

json.dump(sessions, open(os.path.join(OUT, "sessions.json"), "w"), indent=1)
print(f"sessions: {len(sessions)}")
for s in sessions:
    if s["corpus"].startswith("smoke"):
        continue
    fl = "-" if s["first_lo"] is None else f"{s['first_lo']:.1f}"
    fh = "-" if s["first_hi"] is None else f"{s['first_hi']:.1f}"
    print(f"{s['corpus']:16s} {s['clone']} {s['game']} lv={s['levels']} apl0={s['apl0']} "
          f"first=[{fl},{fh}]m box={s['box_min']}m {s.get('t0_quality','')} "
          f"hdrN={s.get('n_headers','')} lastA={s.get('last_action_hdr','')} span={s.get('seg_span_min','')} {s['notes']}")
