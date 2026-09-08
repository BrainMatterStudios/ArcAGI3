import json, statistics as st, sys
sys.path.insert(0,".")
from analysis import main, official, CAP
rep=main(); games=rep["probe_0908"]["games"]   # baselines/level counts are static per game
def m(x): return sum(x)/len(x)
ns=[g["n"] for g in games]
def share(k,n): W=n*(n+1)//2; return 100*sum(range(1,min(k,n)+1))/W
# cadence ceiling: perfect comprehension, actions == baseline, budget = calls*apc actions
print("CADENCE CEILING (perfect comprehension at baseline actions; score at eff 100 => share)")
for calls,apc in [(52,3.0),(52,5.0),(52,10.0),(80,3.0),(104,3.0),(52,1.0)]:
    budget=calls*apc; ks=[]; sc=[]
    for g in games:
        b=g["b"]; cum=0; k=0
        for x in b:
            if cum+x<=budget: cum+=x; k+=1
            else: break
        ks.append(k); sc.append(share(k,g["n"]))
    print(f"  calls {calls} x {apc} act/call = {budget:.0f} actions: levels/game {m(ks):.2f} | zero-level games {sum(1 for k in ks if k==0)} | full solves {sum(1 for k,g in zip(ks,games) if k==g['n'])} | local score {m(sc):.2f} | LB-eq(x0.40) {0.40*m(sc):.2f}")
print("baseline totals per game: mean %.0f median %.0f min %d max %d" % (m([sum(g['b']) for g in games]), st.median([sum(g['b']) for g in games]), min(sum(g['b']) for g in games), max(sum(g['b']) for g in games)))
print("cumulative baseline actions for first k levels (mean over 25):", [round(m([sum(g['b'][:k]) for g in games]),0) for k in range(1,6)])
print("calls needed at 3 act/call for first k levels (mean):", [round(m([sum(g['b'][:k]) for g in games])/3,1) for k in range(1,6)])
# mixture profiles: p = distribution of levels/game over the hidden set (public-25 level-count mix as proxy), eff factor f
S=[m([share(k,n) for n in ns]) for k in range(0,7)]
print("mean weight share for k levels (public-25 n mix):", [round(x,2) for x in S])
profiles={
 "OURS local (75-row mix)": None,
 "A: all L1, eff 1.0": [0,1,0,0,0,0,0],
 "B: 50% L1 / 50% L2, eff 1.0": [0,.5,.5,0,0,0,0],
 "C: 25% zero, 25% L1, 30% L2, 20% L3, eff 1.0": [.25,.25,.30,.20,0,0,0],
 "D: all L2, eff 1.0": [0,0,1,0,0,0,0],
 "E: 20% zero,20% L1,30% L2,20% L3,10% L4, eff 1.0": [.2,.2,.3,.2,.1,0,0],
 "F: all L3, eff 1.0": [0,0,0,1,0,0,0],
 "G: 15% zero,15% L1,25% L2,25% L3,20% L4, eff 1.0": [.15,.15,.25,.25,.20,0,0],
}
from collections import Counter
allrows=[g for v in rep.values() for g in v["games"]]
c=Counter(min(g["L"],6) for g in allrows); ours=[c[k]/len(allrows) for k in range(7)]
profiles["OURS local (75-row mix)"]=ours
print("\nPROFILES (score = eff_factor * sum_k p_k * S_k; eff factors: 1.0 at <=baseline, 0.64 at 1.25x, 0.444 at 1.5x, 0.25 at 2x)")
for name,p in profiles.items():
    raw=sum(pk*S[k] for k,pk in enumerate(p)); lv=sum(k*pk for k,pk in enumerate(p))
    print(f"  {name:55s} levels/game {lv:.2f} | raw {raw:5.2f} | @1.25x {0.64*raw:5.2f} | @1.5x {0.444*raw:5.2f} | @2x {0.25*raw:5.2f}")
print("ours level distribution p0..p6:", [round(x,2) for x in ours])
