"""v2 extractor: CLEAN single-wrap tool_call (the [TOOL CALL] section is already a complete
<tool_call>...</tool_call>), keep more system prompt, hold out 5 whole games."""
import json,glob,re,os,random
ART="/tmp/dr/artifacts"; OUT="scratchpad/arc3_sft"
HOLDOUT={"ft09","re86","sb26","sc25","tu93"}
def sections(t):
    parts=re.split(r'(\[[A-Z][A-Z ]*(?::[^\]]*)?\])', t); d={}
    for i in range(1,len(parts),2): d[parts[i].strip('[]').strip()]=(parts[i+1] if i+1<len(parts) else '').strip()
    return d
def build(F):
    ex=[]
    for l in open(F):
        if not l.strip(): continue
        e=json.loads(l)
        if e.get('type')!='analysis' or not e.get('transcript'): continue
        t=e['transcript']
        if 'request_error' in t or 'Insufficient credits' in t: continue
        s=sections(t)
        sysp=s.get('SYSTEM PROMPT',''); usr=s.get('USER PROMPT','')
        think=s.get('THINKING',''); asst=s.get('ASSISTANT','')
        tool=next((v for k,v in s.items() if k.startswith('TOOL CALL')),'')
        if not sysp or not usr or not tool or '<function=' not in tool: continue   # REQUIRE a real tool call
        comp=""
        if think: comp+=f"<think>\n{think[:600]}\n</think>\n\n"
        if asst: comp+=asst.strip()+"\n\n"
        comp+=tool.strip()                       # CLEAN: tool already is <tool_call>...</tool_call>
        msgs=[{"role":"system","content":sysp[:4000]},{"role":"user","content":usr[:3000]},{"role":"assistant","content":comp.strip()}]
        ex.append({"game":os.path.basename(F).split('-')[0],"messages":msgs})
    return ex
train=[];held=[]
for F in sorted(glob.glob(f"{ART}/*_events.jsonl")):
    g=os.path.basename(F).split('-')[0]
    (held if g in HOLDOUT else train).extend(build(F))
random.seed(0); random.shuffle(train)
nval=max(20,len(train)//12); valid=train[:nval]; tr=train[nval:]
os.makedirs(OUT,exist_ok=True)
for name,rows in [("train",tr),("valid",valid),("test",held)]:
    open(f"{OUT}/{name}.jsonl","w").write('\n'.join(json.dumps({"messages":r["messages"]}) for r in rows)+'\n')
print(f"train={len(tr)} valid={len(valid)} test={len(held)}")
# sanity: verify NO double-wrap + tool call present
c=tr[0]["messages"][2]["content"]
print("double-wrap present:", c.count("<tool_call>")>1, "| ends </tool_call>:", c.rstrip().endswith("</tool_call>"), "| has action(:", "action(" in c)
