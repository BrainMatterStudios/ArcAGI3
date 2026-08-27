import json, sys, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
def p(*a): print(*a, flush=True)
def comp_loss(model, tok, rows, tag):
    tl=0.0; nt=0; n=0
    for r in rows:
        m=r["messages"]
        full=tok.apply_chat_template(m, tokenize=True, add_generation_prompt=False)
        prompt=tok.apply_chat_template(m[:2], tokenize=True, add_generation_prompt=True)
        c0=len(prompt)
        if len(full)<=c0 or len(full)>2560: continue
        ids=mx.array(full)[None]
        logits=model(ids[:, :-1]); targets=ids[0,1:]
        lp=nn.losses.cross_entropy(logits[0], targets, reduction="none")
        mask=mx.arange(targets.shape[0])>=(c0-1)
        tl+=float((lp*mask).sum()); nt+=int(mask.sum()); n+=1
        if n%20==0: p(f"  [{tag}] {n} ex, running loss {tl/max(1,nt):.4f}")
    return tl/max(1,nt), nt, n
rows=[json.loads(l) for l in open("scratchpad/arc3_sft/test.jsonl")][:80]
p(f"loaded {len(rows)} held-out examples; loading BASE...")
base,tokb=load("mlx-community/Qwen2.5-7B-Instruct-4bit")
bl,ntb,nb=comp_loss(base,tokb,rows,"base"); p(f"BASE held-out loss: {bl:.4f} (ppl {2.718281828**bl:.3f}, {nb} ex)")
del base; mx.clear_cache()
p("loading FINE-TUNED...")
ft,tokf=load("mlx-community/Qwen2.5-7B-Instruct-4bit", adapter_path="scratchpad/arc3_adapter")
fl,_,_=comp_loss(ft,tokf,rows,"ft"); p(f"FINETUNED held-out loss: {fl:.4f} (ppl {2.718281828**fl:.3f})")
p(f"DELTA {bl-fl:+.4f} ({100*(bl-fl)/bl:+.1f}%) -> {'GO' if fl<bl-0.02 else 'MARGINAL/NO-GO'}")
