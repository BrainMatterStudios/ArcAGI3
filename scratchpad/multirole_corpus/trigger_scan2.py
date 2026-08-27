#!/usr/bin/env python3
"""Stage-1 trigger replay v2 — uses harness-generated USER PROMPT header lines
(reliable) per turn: executed count + level. Trigger: actions_since_level_start >= A."""
import re, sys

def scan(path, A=15):
    text = open(path, encoding='utf-8', errors='replace').read()
    parts = re.split(r"\n--- analysis_step=", text)[1:]
    lvl_prev=None; since=0; fires=[]; rows=[]
    for i,t in enumerate(parts):
        m = re.match(r"(\d+) \| action=(\d+)", t)
        act=int(m.group(2)) if m else None
        lvl = re.search(r"Current state: step \d+, level (\d+)", t)
        lvl = int(lvl.group(1)) if lvl else lvl_prev
        ex = re.search(r"The code executed (\d+) actions? in the previous sequence", t)
        n = int(ex.group(1)) if ex else 0
        if lvl!=lvl_prev: lvl_prev=lvl; since=0
        since += n
        rows.append((i,act,lvl,n,since))
        if since>=A:
            fires.append((i,act,lvl,since)); since=0
    return fires, rows

for path in sys.argv[1:]:
    f,rows = scan(path)
    name="/".join(path.split("/")[-3::2])
    total=sum(r[3] for r in rows)
    print(f"{name}: turns={len(rows)} actions={total} fires={len(f)} -> {[(a,'L%s'%l,s) for _,a,l,s in f]}")
