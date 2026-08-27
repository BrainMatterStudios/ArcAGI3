#!/usr/bin/env python3
"""Scan analyzer transcripts for wrong-model spiral signatures:
- carried world-model staleness (same 'World model' text across turns)
- level progress per turn (from USER PROMPT 'level N' state lines)
- no-progress streaks in actions"""
import re, sys, hashlib, json, glob

def scan(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    turns = re.split(r"\n--- analysis_step=", text)[1:]
    rows = []
    for t in turns:
        header = t.split("\n",1)[0]
        m = re.match(r"(\d+) \| action=(\d+)", header)
        step, act = (int(m.group(1)), int(m.group(2))) if m else (None,None)
        lvl = None
        ml = re.search(r"Current state: step \d+, level (\d+)", t)
        if ml: lvl = int(ml.group(1))
        wm = re.search(r"- World model: (.*)", t)
        wmtxt = wm.group(1).strip() if wm else ""
        # executed actions this turn from python action results
        exe = len(re.findall(r'"executed": true', t))
        bc = len(re.findall(r'"board_changed": true', t))
        rows.append(dict(step=step, act=act, lvl=lvl, wm=hashlib.md5(wmtxt.encode()).hexdigest()[:8] if wmtxt else "-", wmlen=len(wmtxt), exe=exe, bc=bc, wmtxt=wmtxt))
    return rows

for path in sys.argv[1:]:
    rows = scan(path)
    name = path.split("/")[-3]+"/"+path.split("/")[-1]
    # find longest run of turns with same wm hash and no level change
    best = (0,None,None); cur=1
    for i in range(1,len(rows)):
        if rows[i]["wm"]==rows[i-1]["wm"] and rows[i]["wm"]!="-" and rows[i]["lvl"]==rows[i-1]["lvl"]:
            cur+=1
            if cur>best[0]: best=(cur, rows[i-cur+1]["act"], rows[i]["act"])
        else: cur=1
    lvls = [r["lvl"] for r in rows if r["lvl"]]
    print(f"{name}: turns={len(rows)} max_lvl={max(lvls) if lvls else 0} "
          f"longest_stale_wm_run={best[0]} turns (actions {best[1]}..{best[2]})")
    # print distinct world models count and their first/last turn
    seen={}
    for i,r in enumerate(rows):
        if r["wm"]!="-": seen.setdefault(r["wm"], [i,i,r["wmtxt"]])[1]=i
    print(f"  distinct_world_models={len(seen)}")
    for h,(a,b,txt) in list(seen.items())[:8]:
        print(f"   wm {h} turns {a}-{b}: {txt[:140]}")
