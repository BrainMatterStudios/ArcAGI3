#!/usr/bin/env python3
"""Falsifier 1: EXPECT-QUEUE SAVINGS.

Batch reconstruction: benchmark.json history entries carry generated_tokens>0
on the first action of an LLM turn and ==0 on continuation actions, so a
"batch" here = all env actions executed within one LLM turn (turn-level
grouping; a turn's python code may issue several action() calls -- history
alone cannot split those, so this is the measurable proxy for the queue).
Per-step frames come from intermediate_states.pkl (st[gi][i] = state BEFORE
hist[i]; verified len(states) == len(hist)+1).

Verified stratum: token-gap groups >100 actions are the xplsmoke zero-LLM
frontier-explorer sweeps (engine-speed dts, all 6 in xplsmoke), NOT LLM
action() batches. Primary metric excludes them; a secondary metric includes
them.

Halt rule simulated (pre-registered simple rule): walk batch steps in order;
HALT the remainder at the first step whose post-frame is bit-identical to its
pre-frame (strict no-op) OR equal to the frame BEFORE the batch started (no
cumulative progress), UNLESS that step completed a level. The halting step is
still spent; only the remainder counts as saved.

Harm metrics: among skipped (post-halt) steps, how many actually changed the
frame, and how many completed a level.

PASS threshold: skipped >= 5% of batched actions (primary stratum).
NOTE: unpickling our own recorded corpus (trusted, task-required).
"""
import json, pickle
from collections import defaultdict, Counter
import numpy as np

DIR = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad'
RUNS = ['xpl7','xpl5','xpl4','xpl2','xd','y180','y180b','digest1','v12smoke',
        'banksmoke','packv22','xplsmoke','depthdiag']
OUT = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/9a35ebe3-1f42-4aac-97ed-d1de24bf2346/scratchpad/falsifiers/falsifier1_results.json'

per_game = []
skipped_games = []
tot_actions_all = 0

for r in RUNS:
    b = json.load(open(f'{DIR}/{r}/benchmark.json'))
    st = pickle.load(open(f'{DIR}/{r}/intermediate_states.pkl','rb'))
    for gi, g in enumerate(b['game_runs']):
        hist = g['history'] or []
        s = st[gi]
        tot_actions_all += len(hist)
        if len(s) != len(hist) + 1:
            skipped_games.append({'run': r, 'game': g['game_id'],
                                  'hist': len(hist), 'states': len(s)})
            continue
        frames = [np.asarray(x.frame.data) for x in s]
        lvls = [x.levels_completed for x in s]
        toks = [h['generated_tokens'] for h in hist]
        batches = []
        i = 0
        while i < len(hist):
            j = i
            while j + 1 < len(hist) and toks[j+1] == 0:
                j += 1
            batches.append((i, j))
            i = j + 1
        gstat = {'run': r, 'game': g['game_id'], 'n_actions': len(hist),
                 'n_batches': len(batches)}
        for stratum in ('llm', 'explorer'):
            gstat[stratum] = {'n_multi_batches': 0, 'batched_actions': 0,
                              'skipped_actions': 0, 'halted_batches': 0,
                              'blind_batches': 0, 'batch_sizes': [],
                              'skipped_effective': 0, 'skipped_levelups': 0}
        for (i0, j0) in batches:
            size = j0 - i0 + 1
            if size < 2:
                continue
            stratum = 'explorer' if size > 100 else 'llm'
            gs = gstat[stratum]
            gs['n_multi_batches'] += 1
            gs['batched_actions'] += size
            gs['batch_sizes'].append(size)
            pre_batch = frames[i0]
            halt_at = None
            for k in range(i0, j0 + 1):
                if lvls[k+1] > lvls[k]:
                    continue
                noop = np.array_equal(frames[k+1], frames[k])
                back_to_start = (k > i0) and np.array_equal(frames[k+1], pre_batch)
                if noop or back_to_start:
                    halt_at = k - i0
                    break
            if halt_at is not None:
                gs['halted_batches'] += 1
                if halt_at == 0:
                    gs['blind_batches'] += 1
                for k in range(i0 + halt_at + 1, j0 + 1):
                    gs['skipped_actions'] += 1
                    if lvls[k+1] > lvls[k]:
                        gs['skipped_levelups'] += 1
                    elif not np.array_equal(frames[k+1], frames[k]):
                        gs['skipped_effective'] += 1
        per_game.append(gstat)

def agg_stratum(name):
    keys = ['n_multi_batches','batched_actions','skipped_actions',
            'halted_batches','blind_batches','skipped_effective','skipped_levelups']
    tot = {k: sum(g[name][k] for g in per_game) for k in keys}
    tot['skipped_pct_of_batched'] = round(
        100.0*tot['skipped_actions']/tot['batched_actions'], 2) if tot['batched_actions'] else 0.0
    sizes = Counter()
    for g in per_game: sizes.update(g[name]['batch_sizes'])
    tot['batch_size_dist'] = dict(sorted(sizes.items()))
    return tot

llm = agg_stratum('llm')
xpl = agg_stratum('explorer')
both_batched = llm['batched_actions'] + xpl['batched_actions']
both_skipped = llm['skipped_actions'] + xpl['skipped_actions']

agg = defaultdict(lambda: {'batched': 0, 'skipped': 0})
for g in per_game:
    gid = g['game'].split('-')[0]
    agg[gid]['batched'] += g['llm']['batched_actions']
    agg[gid]['skipped'] += g['llm']['skipped_actions']
worst = sorted(((gid, v['skipped'], v['batched'],
                 100.0*v['skipped']/v['batched'] if v['batched'] else 0.0)
                for gid, v in agg.items()), key=lambda x: -x[1])

res = {
    'threshold_pct': 5.0,
    'total_env_actions_in_corpus': tot_actions_all,
    'primary_llm_batches': llm,
    'secondary_explorer_sweeps': xpl,
    'combined_skipped_pct': round(100.0*both_skipped/both_batched, 2) if both_batched else 0.0,
    'verdict': 'PASS' if llm['skipped_pct_of_batched'] >= 5.0 else 'FAIL',
    'worst_games_llm': [{'game': w[0], 'skipped': w[1], 'batched': w[2],
                         'skip_pct': round(w[3],1)} for w in worst[:6]],
    'per_game': per_game,
    'alignment_skipped_games': skipped_games,
}
json.dump(res, open(OUT, 'w'), indent=1)
p = llm
print(f"PRIMARY (LLM-turn batches, size 2-100): batched={p['batched_actions']} "
      f"skipped={p['skipped_actions']} ({p['skipped_pct_of_batched']}%)")
print(f"  multi-batches={p['n_multi_batches']} halted={p['halted_batches']} "
      f"blind(step1)={p['blind_batches']}")
print(f"  harm: skipped-but-effective={p['skipped_effective']} "
      f"skipped-levelups={p['skipped_levelups']}")
print(f"SECONDARY (xplsmoke explorer sweeps >100): batched={xpl['batched_actions']} "
      f"skipped={xpl['skipped_actions']} ({xpl['skipped_pct_of_batched']}%), "
      f"halted={xpl['halted_batches']}")
print(f"COMBINED skipped pct: {res['combined_skipped_pct']}%")
print("VERDICT (primary):", res['verdict'])
for w in worst[:6]:
    print(f"  {w[0]}: skipped {w[1]}/{w[2]} ({w[3]:.1f}%)")
if skipped_games:
    print("alignment-skipped games:", skipped_games)
print("LLM batch size distribution:", llm['batch_size_dist'])
