#!/usr/bin/env python3
"""Falsifier 3: MULTI-TOOL-CALL TURN ERASURE FREQUENCY (measurement, no pass/fail).

Counts, from transcripts of all 13 corpus runs:
- assistant responses ([MODEL RESPONSE META] blocks) and their tool_call_count
- responses with >=2 tool calls; among them, whether a NON-FINAL tool call
  dispatched an env action (action( in a non-final [TOOL CALL: python] body)
  -- the preserve_history=False trigger at tool_agent.py (env action from a
  non-final call in the loop; see lines ~1990-1997 in
  submission/_adopt/taaf-src/src/ARC3-Inference/inference/agent/tool_agent.py)
- mid-turn 'request_error:' / 'error:' ANALYZER STATUS signatures (the other
  two preserve_history=False -> full turn-history revert paths)
Rates reported per 1000 requests (requests = responses + failed requests).
"""
import json, re, glob

DIR = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad'
RUNS = ['xpl7','xpl5','xpl4','xpl2','xd','y180','y180b','digest1','v12smoke',
        'banksmoke','packv22','xplsmoke','depthdiag']
OUT = '/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/9a35ebe3-1f42-4aac-97ed-d1de24bf2346/scratchpad/falsifiers/falsifier3_results.json'

META = re.compile(r'\[MODEL RESPONSE META\]')
TCC = re.compile(r'^tool_call_count: (\d+)', re.M)
per_run = []
tot = {'responses': 0, 'multi': 0, 'multi_nonfinal_action': 0,
       'request_errors': 0, 'generic_errors': 0}
for r in RUNS:
    row = {'run': r, 'responses': 0, 'multi': 0, 'multi_nonfinal_action': 0,
           'request_errors': 0, 'generic_errors': 0}
    for path in sorted(glob.glob(f'{DIR}/{r}/transcripts/*.txt')):
        text = open(path).read()
        # split into response units at META markers
        mpos = [m.start() for m in META.finditer(text)]
        row['responses'] += len(mpos)
        for j, mp in enumerate(mpos):
            mend = mpos[j+1] if j+1 < len(mpos) else len(text)
            body = text[mp:mend]
            m = TCC.search(body)
            n = int(m.group(1)) if m else 0
            if n >= 2:
                row['multi'] += 1
                calls = re.split(r'\[TOOL CALL: ', body)[1:]
                nonfinal = calls[:-1]
                if any('action(' in c.split('[TOOL RESULT')[0] for c in nonfinal):
                    row['multi_nonfinal_action'] += 1
        row['request_errors'] += len(re.findall(r'request_error:', text))
        row['generic_errors'] += len(re.findall(r'\[ANALYZER STATUS\]\nerror: ', text))
    for k in tot: tot[k] += row[k]
    per_run.append(row)

requests = tot['responses'] + tot['request_errors']  # failed requests produce no META
res = {
    'runs': per_run,
    'totals': tot,
    'total_requests_est': requests,
    'per_1000_requests': {
        'multi_tool_call_responses': round(1000.0*tot['multi']/requests, 3),
        'multi_with_nonfinal_env_action_(erasure_trigger)':
            round(1000.0*tot['multi_nonfinal_action']/requests, 3),
        'request_error_reverts': round(1000.0*tot['request_errors']/requests, 3),
        'generic_error_reverts': round(1000.0*tot['generic_errors']/requests, 3),
    },
}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res['totals'], indent=1))
print('requests(est):', requests)
print(json.dumps(res['per_1000_requests'], indent=1))
for row in per_run:
    print(row)
