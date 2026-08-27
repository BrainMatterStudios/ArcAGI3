"""Track A1: CLEAN re-harvest fixing the two bugs Fable found:
(1) CREDIT-ASSIGNMENT: keep only turns AT/BEFORE the final level-completion (drop post-completion flailing).
(2) TRUNCATION MISMATCH: train on FULL prompts (no aggressive char cap); drop examples too long instead of
    truncating their END (which would corrupt the tool_call). Source = base trajectories in /tmp/rft2gen."""
import json,glob,re,os
from collections import defaultdict
HOLD={"ft09","re86","sb26","sc25","tu93"}
MAXCHARS=15000  # ~3750 tok proxy; training kernel also filters by real tokens

def sections(t):
    parts=re.split(r'(\[[A-Z][A-Z ]*(?::[^\]]*)?\])', t); d={}
    for i in range(1,len(parts),2): d[parts[i].strip('[]').strip()]=(parts[i+1] if i+1<len(parts) else '').strip()
    return d

def best_run_and_cutoff(F):
    """Return (levels, last_completion_line_idx). Level-completion is on action events."""
    lv=0; last_complete=-1
    for i,l in enumerate(open(F)):
        if not l.strip(): continue
        try: e=json.loads(l)
        except: continue
        if e.get('type')=='action' and e.get('level_completed'):
            lv+=1; last_complete=i
        lv=max(lv,int(e.get('level',0)))
    return lv, last_complete

def turns(F, cutoff):
    ex=[]; kept=0; dropped_long=0
    for i,l in enumerate(open(F)):
        if not l.strip(): continue
        if cutoff>=0 and i>cutoff: break   # CREDIT-ASSIGNMENT: stop after last completion
        try: e=json.loads(l)
        except: continue
        if e.get('type')!='analysis' or not e.get('transcript') or 'request_error' in e['transcript']: continue
        s=sections(e['transcript']); sysp=s.get('SYSTEM PROMPT',''); usr=s.get('USER PROMPT','')
        think=s.get('THINKING',''); asst=s.get('ASSISTANT',''); tool=next((v for k,v in s.items() if k.startswith('TOOL CALL')),'')
        if not sysp or not usr or '<function=' not in tool: continue
        comp=(f"<think>\n{think[:1500]}\n</think>\n\n" if think else "")+(asst.strip()+"\n\n" if asst else "")+tool.strip()
        # FULL prompts (no sysp[:4000]/usr[:3000] cap) — match inference
        msgs=[{"role":"system","content":sysp},{"role":"user","content":usr},{"role":"assistant","content":comp.strip()}]
        if sum(len(m['content']) for m in msgs)>MAXCHARS: dropped_long+=1; continue  # drop, don't END-truncate
        ex.append({"messages":msgs}); kept+=1
    return ex, dropped_long

best=defaultdict(lambda:(-1,None,-1))
for F in glob.glob("/tmp/rft2gen/artifacts/*_events.jsonl"):
    g=os.path.basename(F).split('-')[0]
    lv,cut=best_run_and_cutoff(F)
    if lv>best[g][0]: best[g]=(lv,F,cut)
out=[]; drop_tot=0; games=0
for g,(lv,F,cut) in sorted(best.items()):
    if g in HOLD or lv<1: continue
    ts,dl=turns(F,cut); out.extend(ts); drop_tot+=dl; games+=1
    print(f"  {g}: L{lv}, cutoff@{cut}, {len(ts)} clean turns (+{dl} dropped-too-long)")
os.makedirs("submission/_rftclean",exist_ok=True)
open("submission/_rftclean/train.jsonl","w").write('\n'.join(json.dumps(r) for r in out)+'\n')
toks=sum(len(m['content']) for r in out for m in r['messages'])//4
print(f"\nCLEAN dataset: {len(out)} turns, {games} games, ~{toks:,} tok | dropped-too-long: {drop_tot}")
open("submission/_rftclean/dataset-metadata.json","w").write(json.dumps(
  {"title":"arc3-rft-clean","id":"ahmedmobasher86/arc3-rft-clean","licenses":[{"name":"CC0-1.0"}]},indent=2))
