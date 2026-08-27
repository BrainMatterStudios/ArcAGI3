"""Phase-0 harvest: from existing 27B harness runs, keep the BEST run per game (most levels completed)
and extract its turns as RFT SFT data (clean single-wrap, same format as the validated extractor).
Sources: /tmp/dr (reproduction, 1 pass) + /tmp/p4 (prim4, 3 passes)."""
import json,glob,re,os
from collections import defaultdict
def sections(t):
    parts=re.split(r'(\[[A-Z][A-Z ]*(?::[^\]]*)?\])', t); d={}
    for i in range(1,len(parts),2): d[parts[i].strip('[]').strip()]=(parts[i+1] if i+1<len(parts) else '').strip()
    return d
def run_levels(F):
    lv=0
    for l in open(F):
        if not l.strip(): continue
        e=json.loads(l)
        if e.get('type')=='action' and e.get('level_completed'): lv+=1
        lv=max(lv,int(e.get('level',0)))
    return lv
def turns(F):
    ex=[]
    for l in open(F):
        if not l.strip(): continue
        e=json.loads(l)
        if e.get('type')!='analysis' or not e.get('transcript'): continue
        t=e['transcript']
        if 'request_error' in t: continue
        s=sections(t); sysp=s.get('SYSTEM PROMPT',''); usr=s.get('USER PROMPT','')
        think=s.get('THINKING',''); asst=s.get('ASSISTANT','')
        tool=next((v for k,v in s.items() if k.startswith('TOOL CALL')),'')
        if not sysp or not usr or '<function=' not in tool: continue
        comp=(f"<think>\n{think[:600]}\n</think>\n\n" if think else "")+(asst.strip()+"\n\n" if asst else "")+tool.strip()
        ex.append({"messages":[{"role":"system","content":sysp[:4000]},{"role":"user","content":usr[:3000]},{"role":"assistant","content":comp.strip()}]})
    return ex
# gather all runs per game across both sources
best=defaultdict(lambda:(-1,None))
for F in glob.glob("/tmp/dr/artifacts/*_events.jsonl")+glob.glob("/tmp/p4/artifacts/*_events.jsonl"):
    g=os.path.basename(F).split('-')[0]; lv=run_levels(F)
    if lv>best[g][0]: best[g]=(lv,F)
rft=[]; kept=0
for g,(lv,F) in sorted(best.items()):
    if lv>=1:  # keep games where the best run completed >=1 level
        ts=turns(F); rft.extend(ts); kept+=1
        print(f"  {g}: best={lv} levels, {len(ts)} turns")
os.makedirs("scratchpad/rft_data",exist_ok=True)
open("scratchpad/rft_data/train.jsonl","w").write('\n'.join(json.dumps(r) for r in rft)+'\n')
toks=sum(len(m['content']) for r in rft for m in r['messages'])//4
print(f"\nRFT dataset: {len(rft)} success turns from {kept} games | ~{toks:,} tokens")
