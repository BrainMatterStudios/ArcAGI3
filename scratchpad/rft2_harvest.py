"""RFT round-2 harvest: from the TUNED model's generation runs, keep the BEST run per game
(most levels) and extract its turns as SFT data. Exclude the 5 held-out games. Compare yield
vs round-1 to decide tuned-only vs combined."""
import json,glob,re,os
from collections import defaultdict
HOLD={"ft09","re86","sb26","sc25","tu93"}
def sections(t):
    parts=re.split(r'(\[[A-Z][A-Z ]*(?::[^\]]*)?\])', t); d={}
    for i in range(1,len(parts),2): d[parts[i].strip('[]').strip()]=(parts[i+1] if i+1<len(parts) else '').strip()
    return d
def run_levels(F):
    lv=0
    for l in open(F):
        if not l.strip(): continue
        e=json.loads(l); lv=max(lv,int(e.get('level',0)))
    return lv
def turns(F):
    ex=[]
    for l in open(F):
        if not l.strip(): continue
        e=json.loads(l)
        if e.get('type')!='analysis' or not e.get('transcript') or 'request_error' in e['transcript']: continue
        s=sections(e['transcript']); sysp=s.get('SYSTEM PROMPT',''); usr=s.get('USER PROMPT','')
        think=s.get('THINKING',''); asst=s.get('ASSISTANT',''); tool=next((v for k,v in s.items() if k.startswith('TOOL CALL')),'')
        if not sysp or not usr or '<function=' not in tool: continue
        comp=(f"<think>\n{think[:600]}\n</think>\n\n" if think else "")+(asst.strip()+"\n\n" if asst else "")+tool.strip()
        ex.append({"messages":[{"role":"system","content":sysp[:4000]},{"role":"user","content":usr[:3000]},{"role":"assistant","content":comp.strip()}]})
    return ex
best=defaultdict(lambda:(-1,None))
for F in glob.glob("/tmp/rft2gen/artifacts/*_events.jsonl"):
    g=os.path.basename(F).split('-')[0]; lv=run_levels(F)
    if lv>best[g][0]: best[g]=(lv,F)
r2=[]; kept=0
for g,(lv,F) in sorted(best.items()):
    if g in HOLD: continue
    if lv>=1:
        ts=turns(F); r2.extend(ts); kept+=1
        print(f"  {g}: best={lv} lvl, {len(ts)} turns")
os.makedirs("scratchpad/rft2_data",exist_ok=True)
open("scratchpad/rft2_data/train_r2only.jsonl","w").write('\n'.join(json.dumps(r) for r in r2)+'\n')
toks=sum(len(m['content']) for r in r2 for m in r['messages'])//4
print(f"\nROUND-2 (tuned-only): {len(r2)} turns, {kept} games, ~{toks:,} tok")
# round-1 count for comparison
r1n=sum(1 for _ in open("submission/_rftdata/train.jsonl")) if os.path.exists("submission/_rftdata/train.jsonl") else 0
print(f"ROUND-1 was: {r1n} turns")
