"""Direct apples-to-apples: mean cross-entropy on the ASSISTANT completion tokens over the held-out
(5 unseen games) test set, for BASE vs BASE+adapter."""
import json, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
def comp_loss(model, tok, rows):
    tot_loss=0.0; tot_tok=0
    for r in rows:
        m=r["messages"]
        full=tok.apply_chat_template(m, tokenize=True, add_generation_prompt=False)
        prompt=tok.apply_chat_template(m[:2], tokenize=True, add_generation_prompt=True)
        c0=len(prompt)                      # completion starts here
        if len(full)<=c0 or len(full)>2560: continue
        ids=mx.array(full)[None]
        logits=model(ids[:, :-1])           # predict next token
        targets=ids[0,1:]
        # loss only on completion tokens (positions c0-1 .. end)
        lp=nn.losses.cross_entropy(logits[0], targets, reduction="none")
        mask=mx.arange(targets.shape[0])>=(c0-1)
        tot_loss += float((lp*mask).sum()); tot_tok += int(mask.sum())
    return tot_loss/max(1,tot_tok), tot_tok
rows=[json.loads(l) for l in open("scratchpad/arc3_sft/test.jsonl")][:120]
base,tokb=load("mlx-community/Qwen2.5-7B-Instruct-4bit")
bl,nt=comp_loss(base,tokb,rows); print(f"BASE      held-out completion loss: {bl:.4f}  (ppl {2.718281828**bl:.3f}, {nt} tok, {len(rows)} ex)")
ft,tokf=load("mlx-community/Qwen2.5-7B-Instruct-4bit", adapter_path="scratchpad/arc3_adapter")
fl,_=comp_loss(ft,tokf,rows); print(f"FINETUNED held-out completion loss: {fl:.4f}  (ppl {2.718281828**fl:.3f})")
print(f"DELTA: {(bl-fl):.4f}  ({100*(bl-fl)/bl:+.1f}%)  -> {'GO: fine-tune generalizes to unseen games' if fl<bl-0.02 else 'NO-GO / marginal'}")
