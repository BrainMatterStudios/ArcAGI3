#!/usr/bin/env python3
"""Falsifier 2: NO-OP GUARD CATCH RATE.

Ground truth strict no-op action = post-frame bit-identical to pre-frame AND
no levels_completed change (same definition that produced the recorded
no-op count; full-frame identity, no HUD mask -- no HUD mask data exists in
this corpus, noted as a deviation).

Guard predicate (candidate): refuse action A (id + coords) if
  (a) the identical action key was already executed this level, AND
  (b) its LAST execution changed zero pixels (strict no-op), AND
  (c) no DIFFERENT action key changed the frame since that execution.
State resets on level completion (and per game).

Replay: every env action in recorded order per game; the guard is evaluated
against the recorded stream (open-loop replay -- refusals do not alter the
subsequent stream; deviation noted). Catches = refused among strict no-ops;
false catches = refused among effective actions.

PASS threshold: >=60% of strict no-ops caught AND false catches <2% of
effective actions.
NOTE: unpickling our own recorded corpus (trusted, task-required).
"""
import json, pickle
import numpy as np

DIR = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad'
RUNS = ['xpl7','xpl5','xpl4','xpl2','xd','y180','y180b','digest1','v12smoke',
        'banksmoke','packv22','xplsmoke','depthdiag']
OUT = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/9a35ebe3-1f42-4aac-97ed-d1de24bf2346/scratchpad/falsifiers/falsifier2_results.json'

def akey(a):
    d = a.get('data') or {}
    return (a['id'], d.get('x'), d.get('y'))

tot_noop = 0; tot_eff = 0; caught = 0; false_catch = 0
per_game = []
false_examples = []
noop_runs_v12 = 0  # replicate the 11-run duck-v12 subset count for provenance
V12 = set(RUNS) - {'packv22','xplsmoke'}

for r in RUNS:
    b = json.load(open(f'{DIR}/{r}/benchmark.json'))
    st = pickle.load(open(f'{DIR}/{r}/intermediate_states.pkl','rb'))
    for gi, g in enumerate(b['game_runs']):
        hist = g['history'] or []
        s = st[gi]
        if len(s) != len(hist) + 1:
            continue
        frames = [np.asarray(x.frame.data) for x in s]
        lvls = [x.levels_completed for x in s]
        # per-level guard state
        last_exec = {}         # key -> (changed, seq)
        gl = {'run': r, 'game': g['game_id'], 'noops': 0, 'caught': 0,
              'effective': 0, 'false': 0}
        seq = 0
        global_last_change = {'seq': -1, 'key': None}
        for i, h in enumerate(hist):
            if i > 0 and lvls[i] > lvls[i-1]:
                last_exec = {}
                global_last_change = {'seq': -1, 'key': None}
            k = akey(h['action'])
            levelup = lvls[i+1] > lvls[i]
            changed = levelup or (not np.array_equal(frames[i+1], frames[i]))
            # guard decision BEFORE executing
            refuse = False
            if k in last_exec:
                prev_changed, prev_seq = last_exec[k]
                if not prev_changed:
                    interceding = (global_last_change['seq'] > prev_seq and
                                   global_last_change['key'] != k)
                    if not interceding:
                        refuse = True
            # tally
            if changed:
                gl['effective'] += 1
                if refuse:
                    gl['false'] += 1
                    if len(false_examples) < 15:
                        false_examples.append({'run': r, 'game': g['game_id'],
                                               'i': i, 'action': h['action']})
            else:
                gl['noops'] += 1
                if r in V12:
                    noop_runs_v12 += 1
                if refuse:
                    gl['caught'] += 1
            # update state with the actual recorded outcome
            last_exec[k] = (changed, seq)
            if changed:
                global_last_change = {'seq': seq, 'key': k}
            seq += 1
        tot_noop += gl['noops']; caught += gl['caught']
        tot_eff += gl['effective']; false_catch += gl['false']
        per_game.append(gl)

catch_rate = 100.0*caught/tot_noop if tot_noop else 0.0
false_rate = 100.0*false_catch/tot_eff if tot_eff else 0.0
verdict = 'PASS' if (catch_rate >= 60.0 and false_rate < 2.0) else 'FAIL'
res = {
    'threshold': '>=60% catch, <2% false-catch',
    'strict_noops_all13': tot_noop,
    'strict_noops_11run_v12_subset': noop_runs_v12,
    'caught': caught, 'catch_rate_pct': round(catch_rate, 2),
    'effective_actions': tot_eff, 'false_catches': false_catch,
    'false_catch_rate_pct': round(false_rate, 3),
    'verdict': verdict,
    'hud_mask': 'NOT AVAILABLE in corpus; full-frame identity used (matches the recorded no-op definition)',
    'false_examples': false_examples,
    'per_game': per_game,
}
json.dump(res, open(OUT, 'w'), indent=1)
print(f"strict no-ops (13 runs)={tot_noop}  [11-run v12 subset: {noop_runs_v12}]")
print(f"caught={caught} ({catch_rate:.2f}%)  false={false_catch}/{tot_eff} ({false_rate:.3f}%)")
print("VERDICT:", verdict)
top = sorted(per_game, key=lambda g: -g['caught'])[:8]
for g in top:
    print(f"  {g['run']}/{g['game']}: noops={g['noops']} caught={g['caught']} false={g['false']}")
