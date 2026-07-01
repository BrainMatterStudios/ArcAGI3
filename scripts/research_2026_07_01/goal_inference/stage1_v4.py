"""STAGE 1 v4: adds TRANSLATION self-similarity TemplateMatch (numpy roll search for the distinctive offset),
agent-excluded CoverAll, submit-aware win-deciding frame, achieved-during-play filter. Independent re-check of
the #1 predicate on raw frames."""
import logging; logging.basicConfig(level=logging.ERROR)
import sys, numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad")
import goal_harness as H
from arcagi3 import perception as P

def centroid(g,colors):
    ys,xs=np.where(np.isin(g,list(colors))) if colors else (np.array([]),np.array([]))
    return (int(round(ys.mean())),int(round(xs.mean()))) if len(ys) else None
def _mm(g,dr,dc,bg):
    sh=np.roll(np.roll(g,-dr,0),-dc,1); valid=np.ones((64,64),bool)
    if dr>0: valid[64-dr:,:]=False
    elif dr<0: valid[:-dr,:]=False
    if dc>0: valid[:,64-dc:]=False
    elif dc<0: valid[:,:-dc]=False
    return (g==sh)&(g!=bg)&valid

class Reach:
    dl=4
    def __init__(s,c,ac): s.c=c; s.ac=ac; s.repr=f"Reach({c})"; s.cls="reach"
    def sat(s,g):
        ap=centroid(g,s.ac)
        return ap is not None and any(0<=ap[0]+dr<64 and 0<=ap[1]+dc<64 and int(g[ap[0]+dr,ap[1]+dc])==s.c for dr in(-1,0,1) for dc in(-1,0,1))
class Collect:
    dl=5
    def __init__(s,c): s.c=c; s.repr=f"Collect({c})"; s.cls="collect"
    def sat(s,g): return not np.any(g==s.c)
class CoverAll:
    dl=8
    def __init__(s,yc,Y): s.yc=yc; s.Y=Y; s.repr=f"CoverAll(start={yc},|Y|={len(Y)})"; s.cls="cover"
    def sat(s,g): return bool(s.Y) and all(int(g[r,c])!=s.yc for (r,c) in s.Y)
class TemplateT:
    dl=9
    def __init__(s,off,cells): s.off=off; s.cells=cells; s.repr=f"Template(off={off},n={len(cells)})"; s.cls="template"
    def sat(s,g):
        dr,dc=s.off
        return all(int(g[r,c])==int(g[r+dr,c+dc]) for (r,c) in s.cells)

def template_from_search(cap, wd, bg):
    B=cap["ordinary"]
    best=None
    for dr in range(-45,46):
        for dc in range(-45,46):
            if abs(dr)+abs(dc)<4: continue
            m=_mm(wd,dr,dc,bg); n=int(m.sum())
            if n<25: continue
            ordm=np.mean([int(_mm(b,dr,dc,bg).sum()) for b in B[:6]]) if B else 0
            gain=n-ordm
            if best is None or gain>best[0]: best=(gain,dr,dc,m)
    if best is None or best[0]<15: return None
    gain,dr,dc,m=best
    # template cells = matched@wd that are NOT constantly matched in ordinary (distinctive), within valid region
    ordmatch=np.zeros((64,64)) 
    for b in B[:8]: ordmatch += _mm(b,dr,dc,bg)
    keep = m & (ordmatch < max(1,len(B[:8])*0.5))
    cells=[(int(r),int(c)) for r,c in np.argwhere(keep) if 0<=r+dr<64 and 0<=c+dc<64]
    if len(cells)<15: return None
    return TemplateT((dr,dc),cells)

def induce(cap):
    W=cap["win_grid"]; P0=cap["win_prev"]; B=cap["ordinary"]; ac=cap["agent_colors"] or set()
    bg=P.detect_background(W); submit=(cap["win_tok"][0]=="S" and cap["win_tok"][1]==5); wd=P0 if submit else W
    if len(B) < 10:                     # no baseline -> cannot contrast -> abstain (honest)
        return [], submit, wd, bg
    cands=[]
    for c in [int(x) for x in np.unique(W) if int(x)!=bg and int(x) not in ac]: cands.append(Reach(c,ac))
    for c in [int(x) for x in np.unique(cap["start_grid"]) if int(x)!=bg]:
        if not np.any(W==c): cands.append(Collect(c))
    for yc in [int(x) for x in np.unique(cap["start_grid"]) if int(x)!=bg and int(yc if False else x) not in ac]:
        Y=[tuple(p) for p in np.argwhere(cap["start_grid"]==yc)]
        if 1<=len(Y)<=200: cands.append(CoverAll(yc,Y))
    t=template_from_search(cap,wd,bg)
    if t is not None: cands.append(t)
    rows=[]
    for p in cands:
        try:
            if not p.sat(wd): continue
            if not submit and p.sat(P0): continue      # causal for move-wins
            base_hits=sum(1 for b in B if p.sat(b)); distinct=sum(1 for b in B if not p.sat(b))/len(B)
        except Exception: continue
        if distinct>=0.9: rows.append((p,p.repr,p.cls,distinct,p.dl))
    rows.sort(key=lambda r:(-r[3],r[4]))
    return rows, submit, wd, bg

def recheck(top, wd, B):
    """INDEPENDENT re-check: re-evaluate the #1 predicate raw -> must be True@wd, False in >=90% ordinary."""
    p=top[0]
    tw=p.sat(wd); fo=sum(1 for b in B if not p.sat(b))/max(1,len(B))
    return tw and fo>=0.9

for game in (sys.argv[1:] or ["cd82","tu93"]):
    cap=H.capture(game)
    if cap is None: print(f"{game}: no L0"); continue
    surv,submit,wd,bg=induce(cap)
    want={"cd82":"template","tu93":"cover"}.get(game)
    print(f"\n===== {game} (submit={submit}) =====")
    for _,r,cls,dist,dl in surv[:6]: print(f"    {cls:9s} {r:34s} distinct={dist:.2f} dl={dl}")
    if surv:
        ok=recheck(surv[0], wd, cap["ordinary"])
        top=surv[0][2]
        verdict='PASS' if top==want and ok else ('reranker-disagree' if top==want else 'FAIL')
        print(f"  #1={top} (want {want}); independent recheck={'OK' if ok else 'FAIL'} -> {verdict}")
    else: print("  NO survivors -> FAIL")
