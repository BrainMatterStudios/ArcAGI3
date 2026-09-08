"""Score arithmetic over the three best 25-game Modal waves. Read-only on the repo."""
import json, re, statistics as st, glob, os, sys
from pathlib import Path
REPO = Path("/Users/ahmed/Documents/ArcAGI3")
WAVES = {
 "probe_0908":    REPO/"offkaggle/results/20260908T0752-keith_probe/20260908-095256-regime-keith_probe",
 "kv10_0903":     REPO/"offkaggle/results/20260903T0553-kv10-keith/20260903-075332-regime-keith",
 "yield900_0907": REPO/"offkaggle/results/20260906T2251-keith_yield900/20260907-005108-regime-keith_yield900",
}
CAP=115.0; BUDGET=7920.0

def official(baselines, apl, levels):
    """scorecard.py:170-171 per level, :196-206 weights/cap; taaf game.py:381-415 mirror."""
    n=len(baselines); W=n*(n+1)//2; tot=0.0; maxw=0
    for l in range(n):
        w=l+1; a=apl[l] if l<len(apl) else 0
        if l<levels and a>0:
            s=min(CAP,(baselines[l]/a)**2*100)
        else: s=0.0
        if s>0: maxw+=w
        tot+=s*w
    return min(tot/W, maxw/W*100)

def per_level_eff(b,a): return min(CAP,(b/a)**2*100) if a>0 else 0.0

def load(wave):
    r=json.load(open(wave/"results.json"))
    sc=json.load(open(wave/"score.json"))
    bm=json.load(open(wave/"benchmark.json"))
    bmap={x["game_id"]:x["base_actions_per_level"] for x in bm["game_runs"]}
    games=[]
    for g in r["games"]:
        gid=g["game_id"]; b=g.get("baselines") or bmap[gid]; apl=g["actions_per_level"]; L=g["levels_completed"]; n=g["number_of_levels"]
        assert b==bmap[gid], (gid,b,bmap[gid])
        rec=official(b,apl,L)
        games.append(dict(gid=gid[:4],b=b,apl=apl,L=L,n=n,W=n*(n+1)//2,score_file=g["score"],score_json=sc["games"][gid]["score"],rec=rec,
                          actions=g["actions"],wall=g["wallclock_s"],state=g["state"]))
    return r,games

def wave_mean(games,key="rec"): return sum(g[key] for g in games)/len(games)

# --- time split from transcripts + events ---
RES_RE=re.compile(r"res \{.*?'level': (\d+),.*?'level_completed': (True|False),.*?'run_elapsed_seconds': ([\d.]+)")
HDR_RE=re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d)")
def time_split(wave, g_by_stem):
    out={}
    for tp in sorted((wave/"transcripts").glob("*.txt")):
        stem=tp.stem; txt=tp.read_text(errors="replace")
        t_last_complete=0.0; ncomp=0
        for m in RES_RE.finditer(txt):
            lvl,comp,t=int(m.group(1)),m.group(2)=="True",float(m.group(3))
            if comp: t_last_complete=t; ncomp+=1
        # calls per step and step->level from events
        ev=wave/"artifacts"/f"{stem}_events.jsonl"
        step_level={}
        if ev.exists():
            for line in open(ev):
                if '"analysis"' not in line: continue
                try: rec=json.loads(line)
                except ValueError: continue
                if rec.get("type")=="analysis": step_level[rec["analysis_step"]]=rec["level"]
        cur=None; calls_by_level={}; total_calls=0
        for line in txt.splitlines():
            h=HDR_RE.match(line)
            if h: cur=int(h.group(1)); continue
            if line.startswith("[MODEL RESPONSE META]"):
                total_calls+=1; lv=step_level.get(cur); calls_by_level[lv]=calls_by_level.get(lv,0)+1
        out[stem]=dict(t_last_complete=t_last_complete,ncomp_seen=ncomp,calls_by_level=calls_by_level,total_calls=total_calls)
    return out

def main():
    report={}
    all_games={}
    for name,wave in WAVES.items():
        r,games=load(wave); all_games[name]=games
        # validation
        maxdiff=max(abs(g["rec"]-g["score_json"]) for g in games)
        ts=time_split(wave,None)
        for g in games:
            stem=[s for s in ts if s.startswith(g["gid"])]
            g["ts"]=ts[stem[0]] if stem else None
        report[name]=dict(games=games,maxdiff=maxdiff,levels=sum(g["L"] for g in games),
                          mean_rec=wave_mean(games),mean_file=wave_mean(games,"score_file"),
                          cadence=[l.strip() for l in open(wave/"summary.txt") if l.strip().startswith(("SCORE","CADENCE","CLIENT"))])
    json.dump({k:{kk:vv for kk,vv in v.items() if kk!="games"} for k,v in report.items()},open(Path(__file__).parent/"validation.json","w"),indent=1)
    return report
if __name__=="__main__":
    rep=main()
    for name,v in rep.items():
        print(f"== {name}: levels {v['levels']} | recomputed mean {v['mean_rec']:.3f} | results.json mean {v['mean_file']:.3f} | max |diff| vs score.json {v['maxdiff']:.2e}")
        for c in v["cadence"]: print("   ",c[:160])
