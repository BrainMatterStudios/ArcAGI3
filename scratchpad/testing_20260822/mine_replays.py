#!/usr/bin/env python3
"""Human-replay corpus miner — 2026-08-21.

Single pass over the 340 open human replays (scratchpad/human_replays/) producing:
  1. budgets_v2.json  — per-game efficiency priors (per-level marginal actions,
     human RHAE vs engine baseline_actions, probe budgets, reset stats)
  2. motifs.json      — per-game macro-action motifs (trigrams, runs, click-cell
     repeats, tempo signature, post-levelup pause)
  3. dispatch.json    — archetype labels from action mix + frame-0 available_actions
  4. fixture_candidates.json — winning traces for games not in the 6 fixtures
  5. ours_vs_human.json — 3.8 wave-1 shipped arm vs human medians, per game
Read-only over the corpus; all outputs to this scratchpad dir.
"""
import json, glob, os, statistics as st
from collections import Counter, defaultdict
from datetime import datetime

REPO = '/Users/ahmed/Documents/ArcAGI3'
HR = f'{REPO}/scratchpad/human_replays/extracted/public_games-dataset'
OUT = os.path.dirname(os.path.abspath(__file__))
FIXTURES_HAVE = {'dc22','ka59','m0r0','sk48','tu93','wa30'}

def median(xs): return st.median(xs) if xs else None
def p90(xs):
    if not xs: return None
    xs=sorted(xs); return xs[min(len(xs)-1,int(round(0.9*(len(xs)-1))))]

# --- engine baselines ---
baselines={}
for md in glob.glob(f'{REPO}/environment_files/*/*/metadata.json'):
    d=json.load(open(md))
    g=d['game_id'].split('-')[0]
    baselines[g]={'game_id':d['game_id'],'baseline_actions':d['baseline_actions'],
                  'tags':d.get('tags',[]),'win_levels':len(d['baseline_actions'])}

def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()

per_game=defaultdict(lambda: {'replays':[]})

files=sorted(glob.glob(f'{HR}/*/*.recording.jsonl'))
for fp in files:
    game=fp.split('/')[-3] if False else os.path.basename(os.path.dirname(fp))
    uuid=os.path.basename(fp).split('.')[0]
    lines=open(fp).readlines()
    seq=[]   # (action_id, x, y, t, levels_completed)
    avail0=None
    for i,raw in enumerate(lines[:-1]):
        rec=json.loads(raw)
        t=None
        try: t=parse_ts(rec['timestamp'])
        except Exception: pass
        d=rec.get('data',rec)
        ai=d.get('action_input')
        if avail0 is None and d.get('available_actions'):
            avail0=d['available_actions']
        if ai is None: continue
        aid=ai.get('id')
        if isinstance(aid,str):  # live-API copies use 'ACTION6'/'RESET' strings
            aid=0 if aid.upper()=='RESET' else int(aid[-1]) if aid[-1].isdigit() else None
        x=y=None
        data=ai.get('data') or {}
        if aid==6: x,y=data.get('x'),data.get('y')
        seq.append((aid,x,y,t,d.get('levels_completed',0)))
    # summary line
    last=json.loads(lines[-1])
    ld=last.get('data',last)
    card=None
    if 'cards' in ld:
        card=list(ld['cards'].values())[0]
    abl=card['actions_by_level'][0] if card and card.get('actions_by_level') else []
    won = (card and card.get('states') and card['states'][0]=='WIN')
    total_actions = card['total_actions'] if card else len(seq)
    levels = ld.get('levels_completed', card['levels_completed'][0] if card else None)
    # marginal per-level costs from cumulative actions_by_level
    marg={}
    prev=0
    for lvl,cum in abl:
        marg[lvl]=cum-prev; prev=cum
    # probe budget: actions before first level-up
    first_lu = abl[0][1] if abl else None
    # resets from action lines
    resets=sum(1 for a in seq if a[0]==0)
    per_game[game]['replays'].append(dict(uuid=uuid,won=bool(won),levels=levels,
        total_actions=total_actions,marg=marg,first_levelup=first_lu,resets=resets,
        seq=seq,avail0=avail0))

# ---------- aggregate ----------
budgets={}; motifs={}; dispatch={}; fixture_cands={}
for game,gd in sorted(per_game.items()):
    reps=gd['replays']
    winners=[r for r in reps if r['won']]
    base=baselines.get(game,{})
    bl=base.get('baseline_actions',[])
    # per-level marginal medians over ALL replays that completed that level
    lvl_stats={}
    for li in range(1,len(bl)+1):
        xs=[r['marg'][li] for r in reps if li in r['marg']]
        if xs:
            b=bl[li-1]
            med=median(xs); mn=min(xs)
            lvl_stats[li]={'n':len(xs),'median':med,'min':mn,'baseline':b,
                'rhae_med':round(min(115,(b/med)**2*100),1) if med else None,
                'rhae_best':round(min(115,(b/mn)**2*100),1) if mn else None}
    # human true score (median-human play): level-weighted, per scoring formula
    wsum=sum(i+1 for i in range(len(bl)))
    def play_score(marg):
        s=0
        for li in range(1,len(bl)+1):
            if li in marg and marg[li]>0:
                s+=min(115,(bl[li-1]/marg[li])**2*100)*(li)  # weight level_index+1 where index starts 0 → li
        return s/wsum if wsum else None
    human_scores=[play_score(r['marg']) for r in reps if r['marg']]
    budgets[game]={'n_replays':len(reps),'n_winners':len(winners),
        'win_levels':base.get('win_levels'),
        'probe_before_L1_winners':{'median':median([r['first_levelup'] for r in winners if r['first_levelup']]),
                                   'p90':p90([r['first_levelup'] for r in winners if r['first_levelup']])},
        'total_actions_winners_median':median([r['total_actions'] for r in winners]),
        'resets_median':median([r['resets'] for r in reps]),
        'per_level':lvl_stats,
        'human_true_score_median':round(median(human_scores),1) if human_scores else None,
        'human_true_score_best':round(max(human_scores),1) if human_scores else None}
    # ---- motifs (winners only; all if no winners) ----
    pool = winners or reps
    tri=Counter(); runs=[]; cellrep=[]; dts=[]; post_lu_dts=[]; norm_dts=[]
    commit_ratio=[]
    for r in pool:
        seq=r['seq']
        toks=[]
        for aid,x,y,t,lc in seq:
            if aid==6 and x is not None:
                toks.append(f'C{ x//16 },{ y//16 }')  # 4x4 coarse cells on 64 grid
            else:
                toks.append({0:'RESET',1:'UP',2:'DOWN',3:'LEFT',4:'RIGHT',5:'A5',6:'CLICK'}.get(aid,str(aid)))
        for i in range(len(toks)-2):
            tri[(toks[i],toks[i+1],toks[i+2])]+=1
        # longest same-id run
        best=cur=1
        for i in range(1,len(seq)):
            cur = cur+1 if seq[i][0]==seq[i-1][0] else 1
            best=max(best,cur)
        runs.append(best)
        # click same-cell consecutive repeat fraction
        clicks=[(x//8,y//8) for aid,x,y,t,lc in seq if aid==6 and x is not None]
        if len(clicks)>1:
            cellrep.append(sum(1 for i in range(1,len(clicks)) if clicks[i]==clicks[i-1])/ (len(clicks)-1))
        # tempo
        ts=[t for _,_,_,t,_ in seq if t]
        d=[ts[i]-ts[i-1] for i in range(1,len(ts)) if 0<ts[i]-ts[i-1]<60]
        if d: dts.extend(d)
        # post-levelup pause: dt of first 5 actions after each level-up vs global median
        lus=[i for i in range(1,len(seq)) if seq[i][4]>seq[i-1][4]]
        for lu in lus:
            for j in range(lu+1,min(lu+6,len(seq))):
                if seq[j][3] and seq[j-1][3]:
                    dd=seq[j][3]-seq[j-1][3]
                    if 0<dd<60: post_lu_dts.append(dd)
        # probe-then-commit: mean dt first 20 actions vs last 20 (winners)
        if r['won'] and len(d)>50:
            a=st.mean(d[:20]); b2=st.mean(d[-20:])
            if b2>0: commit_ratio.append(a/b2)
    motifs[game]={'top_trigrams':[{'seq':list(k),'n':v} for k,v in tri.most_common(6)],
        'longest_run_median':median(runs),
        'click_samecell_repeat_frac':round(st.mean(cellrep),3) if cellrep else None,
        'dt_median_s':round(median(dts),2) if dts else None,
        'post_levelup_dt_median_s':round(median(post_lu_dts),2) if post_lu_dts else None,
        'early_vs_late_dt_ratio':round(st.mean(commit_ratio),2) if commit_ratio else None}
    # ---- dispatch ----
    allseq=[a for r in reps for a in r['seq']]
    n=len(allseq) or 1
    mix={k:round(sum(1 for a in allseq if a[0]==v)/n,3) for k,v in
         [('reset',0),('up',1),('down',2),('left',3),('right',4),('a5',5),('click',6)]}
    move=mix['up']+mix['down']+mix['left']+mix['right']
    arch='AVATAR' if move>0.5 else ('CLICK' if mix['click']>0.5 else 'MIXED')
    av0=Counter(tuple(r['avail0']) for r in reps if r['avail0'])
    dispatch[game]={'archetype_from_mix':arch,'action_mix':mix,
        'frame0_available_actions':[{'set':list(k),'n':v} for k,v in av0.most_common(3)],
        'tags':base.get('tags',[])}
    # ---- fixture candidates ----
    if winners:
        best=min(winners,key=lambda r:r['total_actions'])
        fixture_cands[game]={'have_fixture':game in FIXTURES_HAVE,
            'n_winners':len(winners),
            'best_winner':{'uuid':best['uuid'],'total_actions':best['total_actions'],
                           'levels':best['levels'],'resets':best['resets']},
            'file':f'{HR}/{game}/{best["uuid"]}.recording.jsonl'}

json.dump(budgets,open(f'{OUT}/budgets_v2.json','w'),indent=1)
json.dump(motifs,open(f'{OUT}/motifs.json','w'),indent=1)
json.dump(dispatch,open(f'{OUT}/dispatch.json','w'),indent=1)
json.dump(fixture_cands,open(f'{OUT}/fixture_candidates.json','w'),indent=1)

# ---- ours vs human ----
ours={}
w1=json.load(open(f'{REPO}/scratchpad/qwen38/wave1_shipped_result.json'))
for row in w1['rows']:
    g=row['source_game']
    bl=baselines.get(g,{}).get('baseline_actions',[])
    apl=row.get('actions_per_level',[])
    ours[g]={'levels_completed':row['levels_completed'],'levels_total':row['levels_total'],
             'actions_per_level':apl}
cmp={}
for g,b in budgets.items():
    o=ours.get(g)
    if not o: continue
    rows=[]
    for li,ls in b['per_level'].items():
        oa=o['actions_per_level'][int(li)-1] if int(li)-1<len(o['actions_per_level']) else None
        ratio=round(oa/ls['median'],2) if oa and ls['median'] else None
        rows.append({'level':li,'human_med':ls['median'],'ours_38':oa,
                     'baseline':ls['baseline'],'ours_over_human':ratio})
    cmp[g]={'human_levels_won_median':median([r['levels'] for r in per_game[g]['replays'] if r['won']]),
            'ours_levels':o['levels_completed'],'levels_total':o['levels_total'],'per_level':rows,
            'human_true_score_median':b['human_true_score_median']}
json.dump(cmp,open(f'{OUT}/ours_vs_human.json','w'),indent=1)
print('DONE', len(budgets),'games')
