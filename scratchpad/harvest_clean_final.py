"""DEFINITIVE clean harvest (Track A1, factory-reviewed). Fixes ALL flagged bugs:
- FULL thinking (no cap) — truncated reasoning reproduces the token-diet hidden-crater.
- Gate on a REAL level-field INCREASE (max level over run > start=1) = genuine completion (level is 1-indexed).
- Keep turns up to the last level increase (drop post-progress flailing).
- EXCLUDE held-out games (no leakage).
- FULL system prompt (was 2/3-truncated before).
- Drop examples > CHARCAP (fits max_length; do NOT end-truncate -> would corrupt the tool_call).
Source: base trajectories in /tmp/rft2gen (round-2 generation = base, LoRA never served)."""
import json,glob,re,os
HOLD={"ft09","re86","sb26","sc25","tu93"}
CHARCAP=38000   # ~9500 tok proxy; train max_length=10240
def sections(t):
    parts=re.split(r'(\[[A-Z][A-Z ]*(?::[^\]]*)?\])', t); d={}
    for i in range(1,len(parts),2): d[parts[i].strip('[]').strip()]=(parts[i+1] if i+1<len(parts) else '').strip()
    return d
def scan(F):
    lv=0; last_inc=-1
    for i,l in enumerate(open(F)):
        if not l.strip(): continue
        try: cur=int(json.loads(l).get('level',0))
        except: continue
        if cur>lv: lv=cur; last_inc=i
    return lv,last_inc
def turns(F,cut):
    ex=[]
    for i,l in enumerate(open(F)):
        if not l.strip() or i>cut: 
            if i>cut: break
            continue
        try: e=json.loads(l)
        except: continue
        if e.get('type')!='analysis' or not e.get('transcript') or 'request_error' in e['transcript']: continue
        s=sections(e['transcript']); sysp=s.get('SYSTEM PROMPT',''); usr=s.get('USER PROMPT','')
        think=s.get('THINKING',''); asst=s.get('ASSISTANT',''); tool=next((v for k,v in s.items() if k.startswith('TOOL CALL')),'')
        if not sysp or not usr or '<function=' not in tool: continue
        comp=(f"<think>\n{think}\n</think>\n\n" if think else "")+(asst.strip()+"\n\n" if asst else "")+tool.strip()  # FULL thinking
        msgs=[{"role":"system","content":sysp},{"role":"user","content":usr},{"role":"assistant","content":comp.strip()}]
        if sum(len(m['content']) for m in msgs)>CHARCAP: continue
        ex.append({"messages":msgs,"_game":os.path.basename(F).split('-')[0]})
    return ex
out=[]; games=set(); comp_p=0; fail_p=0
for F in glob.glob("/tmp/rft2gen/artifacts/*_events.jsonl"):
    g=os.path.basename(F).split('-')[0]
    if g in HOLD: continue          # NO held-out leakage
    lv,cut=scan(F)
    if lv>=2:                        # real completion (level advanced past 1-indexed start)
        ts=turns(F,cut)
        for t in ts: t.pop("_game",None)
        out.extend(ts); games.add(g); comp_p+=1
    else: fail_p+=1
# ASSERT no held-out leakage
assert not (games & HOLD), f"LEAK: {games & HOLD}"
os.makedirs("submission/_rftclean",exist_ok=True)
open("submission/_rftclean/train.jsonl","w").write('\n'.join(json.dumps(r) for r in out)+'\n')
toks=sum(len(m['content']) for r in out for m in r['messages'])//4
print(f"completing passes={comp_p} failures-dropped={fail_p}")
print(f"games={sorted(games)}  (held-out excluded: {sorted(HOLD)})")
print(f"CLEAN-FINAL: {len(out)} turns, ~{toks:,} tok, FULL thinking + FULL system prompt")
