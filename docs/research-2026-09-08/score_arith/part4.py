import json, re, statistics as st, sys
sys.path.insert(0,".")
from analysis import main, WAVES
rep=main()
HDR=re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d)")
def m(x): return sum(x)/len(x)
def wave_time(name):
    wave=WAVES[name]; rows=[]
    for g in rep[name]["games"]:
        stem=g["gid"]; tp=[p for p in (wave/"transcripts").glob(f"{stem}*.txt")][0]
        ev=[p for p in (wave/"artifacts").glob(f"{stem}*_events.jsonl")][0]
        step_level={}
        for line in open(ev):
            if '"analysis"' not in line: continue
            try: r=json.loads(line)
            except ValueError: continue
            if r.get("type")=="analysis": step_level[r["analysis_step"]]=r["level"]
        times=[]; last=None; day=0
        for line in open(tp, errors="replace"):
            h=HDR.match(line)
            if not h: continue
            s=int(h.group(1)); t=int(h.group(3))*3600+int(h.group(4))*60+int(h.group(5))
            if last is not None and t<last-3600: day+=86400
            last=t; times.append((s,t+day))
        t0=times[0][1]; wall=g["wall"]; L=g["L"]
        # first header whose step's level is the wall level (L+1); steps repeat headers after yields -> take min time
        wall_t=[t for s,t in times if step_level.get(s,1)>=L+1]
        t_wall=(min(wall_t)-t0) if wall_t else wall
        frac=max(0.0,min(1.0,(wall-t_wall)/wall))
        rows.append(dict(gid=stem,L=L,n=g["n"],t_wall_start_s=round(t_wall),frac_on_uncompleted=round(frac,3),headers=len(times),steps_seen=len(step_level)))
    return rows
out={}
for name in WAVES:
    rows=wave_time(name); out[name]=rows
    fr=[r["frac_on_uncompleted"] for r in rows]; nz=[r["frac_on_uncompleted"] for r in rows if r["L"]>0]
    print(f"== {name}: frac of 7920 s spent on the level never completed: mean {m(fr):.3f} median {st.median(fr):.3f} | games with >=1 level only: mean {m(nz):.3f} | games at 1.0 (zero-level) {sum(1 for r in rows if r['L']==0)} | time-to-wall mean {m([r['t_wall_start_s'] for r in rows]):.0f} s")
json.dump(out,open("time_split.json","w"),indent=1)
allr=[r for v in out.values() for r in v]
print("POOLED75 frac mean %.3f median %.3f; per-level-completed mean time %.0f s (sum t_wall / sum L)" % (m([r["frac_on_uncompleted"] for r in allr]), st.median([r["frac_on_uncompleted"] for r in allr]), sum(r["t_wall_start_s"] for r in allr)/sum(r["L"] for r in allr)))
print("probe wave per game:", [(r["gid"],r["L"],r["t_wall_start_s"],r["frac_on_uncompleted"]) for r in out["probe_0908"]])
