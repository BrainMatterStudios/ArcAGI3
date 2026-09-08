import json, statistics as st, sys
sys.path.insert(0, ".")
from analysis import main, official, per_level_eff, CAP, BUDGET
rep = main()
LIVE = {"base_family_draws": [3.25, 2.58, 4.31, 2.45]}   # keith byte-copy 09-05, redraw, yield900 #1, yield900 #2
live_mean = st.mean(LIVE["base_family_draws"])
out = {}
def m(xs): return sum(xs)/len(xs)

def decompose(games):
    n=len(games); by_level={1:0.0,2:0.0,3:0.0}; capped=0
    depth_loss=[]; eff_loss=[]; score=[]
    for g in games:
        W=g["W"]; unc=0.0; share=0.0
        for l in range(g["L"]):
            e=per_level_eff(g["b"][l],g["apl"][l]); c=100*(l+1)*e/100/W
            key=1 if l==0 else (2 if l==1 else 3)
            by_level[key]+=c; unc+=c; share+=100*(l+1)/W
        if unc>share+1e-9: capped+=1
        s=g["rec"]; score.append(s); depth_loss.append(100-share); eff_loss.append(share-s)
    tot=sum(score)
    return dict(mean=m(score), by_level_share={k:v/tot for k,v in by_level.items()}, by_level_pts={k:v/n for k,v in by_level.items()},
                depth_loss=m(depth_loss), eff_loss=m(eff_loss), capped_games=capped)

def cf(games, mode, wave_eff):
    """counterfactual mean official score"""
    res=[]
    for g in games:
        b=g["b"]; apl=list(g["apl"]); L=g["L"]; n=g["n"]
        effs=[per_level_eff(b[l],apl[l]) for l in range(L)]
        ge = m(effs) if effs else wave_eff          # this game's mean efficiency (score units, 0..115)
        def with_levels(Lnew, eff_new=None, eff_old=None):
            a=[0]*n
            for l in range(min(Lnew,n)):
                if l<L and eff_old is None: a[l]=apl[l]
                else:
                    e=(eff_new if eff_new is not None else ge) if l>=L else eff_old
                    a[l]= b[l]/((e/100)**0.5) if e>0 else 10**9
            return official(b,a,min(Lnew,n))
        if mode=="base": res.append(g["rec"])
        elif mode=="a_plus1": res.append(with_levels(L+1))
        elif mode=="b_perfect": res.append(with_levels(L,eff_old=100.0) if L else 0.0)
        elif mode=="c_zero_to_L1": res.append(with_levels(1,eff_new=wave_eff) if L==0 else g["rec"])
        elif mode=="d_double": res.append(with_levels(2*L))
        elif mode=="d2_double_zero1": res.append(with_levels(max(1,2*L)))
        elif mode=="e_all_levels": res.append(with_levels(n))
        elif mode=="e2_all_levels_perfect": res.append(100.0)
        elif mode=="f_2x_baseline": res.append(with_levels(L,eff_old=25.0) if L else 0.0)
        elif mode=="f2_all_levels_2x": res.append(25.0)
        elif mode=="g_plus1_perfect": res.append(with_levels(L+1,eff_new=100.0))
    return m(res)

pooled=[]
for name,v in rep.items():
    games=v["games"]; pooled+=games
    effs=[per_level_eff(g["b"][l],g["apl"][l]) for g in games for l in range(g["L"])]
    wave_eff=m(effs)
    ratios=[g["apl"][l]/g["b"][l] for g in games for l in range(g["L"])]
    d=decompose(games)
    cfs={k:cf(games,k,wave_eff) for k in ["base","a_plus1","b_perfect","c_zero_to_L1","d_double","d2_double_zero1","e_all_levels","f_2x_baseline","g_plus1_perfect"]}
    # time split
    ts=[g["ts"] for g in games if g["ts"]]
    frac_wall=[(g["wall"]-g["ts"]["t_last_complete"])/g["wall"] for g in games if g["ts"]]
    wall_calls=[]; tot_calls=0; wall_c=0
    for g in games:
        if not g["ts"]: continue
        cbl=g["ts"]["calls_by_level"]; wl=g["L"]+1
        wc=sum(c for lv,c in cbl.items() if lv is not None and lv>=wl); tc=g["ts"]["total_calls"]
        wall_c+=wc; tot_calls+=tc
    calls_needed=[sum(g["b"])/3.0 for g in games]
    calls_needed_L1=[g["b"][0]/3.0 for g in games]
    out[name]=dict(levels=v["levels"], mean=v["mean_rec"], levels_per_game=v["levels"]/25, zero=sum(1 for g in games if g["L"]==0),
                   wave_eff_pts=wave_eff, actions_over_baseline_median=st.median(ratios), actions_over_baseline_mean=m(ratios),
                   n_completed_levels=len(effs), eff_at_cap=sum(1 for e in effs if e>=CAP-1e-9), eff_below_25=sum(1 for e in effs if e<25),
                   decomp=d, cf=cfs, time=dict(frac_budget_on_uncompleted_level_mean=m(frac_wall), median=st.median(frac_wall),
                   calls_at_wall_share=wall_c/tot_calls, calls_total=tot_calls,
                   calls_needed_full_solve_at_3_per_call=dict(mean=m(calls_needed),median=st.median(calls_needed),max=max(calls_needed),min=min(calls_needed)),
                   calls_needed_L1=dict(mean=m(calls_needed_L1),median=st.median(calls_needed_L1),max=max(calls_needed_L1))),
                   games=[dict(gid=g["gid"],n=g["n"],L=g["L"],score=round(g["rec"],2),share=round(100*sum(range(1,g["L"]+1))/g["W"],2),
                               eff=[round(per_level_eff(g["b"][l],g["apl"][l]),1) for l in range(g["L"])],
                               a_over_b=[round(g["apl"][l]/g["b"][l],2) for l in range(g["L"])], b=g["b"], apl=g["apl"],
                               wall_frac=round((g["wall"]-g["ts"]["t_last_complete"])/g["wall"],3) if g["ts"] else None) for g in games])
# pooled
effs=[per_level_eff(g["b"][l],g["apl"][l]) for g in pooled for l in range(g["L"])]
wave_eff=m(effs)
out["POOLED75"]=dict(mean=m([g["rec"] for g in pooled]), levels_per_game=sum(g["L"] for g in pooled)/75, wave_eff_pts=wave_eff,
                     actions_over_baseline_median=st.median([g["apl"][l]/g["b"][l] for g in pooled for l in range(g["L"])]),
                     decomp=decompose(pooled), cf={k:cf(pooled,k,wave_eff) for k in ["base","a_plus1","b_perfect","c_zero_to_L1","d_double","d2_double_zero1","e_all_levels","f_2x_baseline","g_plus1_perfect"]})
# calibration
loc_kv5=m([rep["probe_0908"]["mean_rec"],rep["yield900_0907"]["mean_rec"]])
out["CALIB"]=dict(live_draws=LIVE["base_family_draws"], live_mean=live_mean, local_kv5_mean=loc_kv5, local_three_mean=m([v["mean_rec"] for v in rep.values()]),
                  keith_commit_local=6.76, keith_commit_live=3.25,
                  ratio_kv5=live_mean/loc_kv5, ratio_three=live_mean/m([v["mean_rec"] for v in rep.values()]), ratio_keith=3.25/6.76,
                  hidden85_implied=(110*live_mean-25*loc_kv5)/85)
# profile grid: uniform k levels/game at actions/baseline ratio q on the public-25 level-count distribution
ns=[g["n"] for g in rep["probe_0908"]["games"]]
grid={}
for k in [1,2,3,4,5]:
    for q in [0.8,1.0,1.25,1.5,2.0,3.0]:
        e=min(CAP,100/q**2)
        vals=[]
        for n in ns:
            W=n*(n+1)//2; kk=min(k,n); unc=sum((l+1)*e for l in range(kk))/W; share=100*sum(range(1,kk+1))/W
            vals.append(min(unc,share))
        grid[f"k{k}_q{q}"]=m(vals)
out["GRID"]=grid
out["LEVEL_COUNTS"]=dict(ns=ns, mean_n=m(ns), mean_L1_share=m([100/(n*(n+1)//2) for n in ns]), mean_L12_share=m([300/(n*(n+1)//2) for n in ns]),
                         mean_L123_share=m([600/(n*(n+1)//2) for n in ns]), mean_L1234_share=m([1000/(n*(n+1)//2) for n in ns]))
json.dump(out,open("results_arith.json","w"),indent=1,default=str)
# print compact
for k,v in out.items():
    if k in ("GRID","LEVEL_COUNTS","CALIB"): print(k, json.dumps(v, indent=None)); continue
    print("==",k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items() if a not in ("games","decomp","cf","time")})
    print("  decomp", json.dumps({a:(round(b,3) if isinstance(b,float) else b) for a,b in v["decomp"].items()}))
    print("  cf", json.dumps({a:round(b,2) for a,b in v["cf"].items()}))
    if "time" in v: print("  time", json.dumps(v["time"]))
