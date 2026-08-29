#!/usr/bin/env python3
"""Audit of scoring claims. Read-only over repo data; prints numbers."""
import json, glob, statistics as st, collections, sys
from pathlib import Path

REPO = Path('/Users/ahmed/Documents/ArcAGI3')
sys.path.insert(0, str(REPO / 'reference/arc-agi-toolkit'))
from arc_agi.scorecard import Card, EnvironmentScorecard, Scorecard  # real scorer
from arc_agi.models import EnvironmentInfo
from arcengine import GameState

# ---------------- baselines ----------------
BASE = {}
for g in sorted((REPO / 'environment_files').iterdir()):
    for m in sorted(glob.glob(str(g / '*/metadata.json'))) + sorted(glob.glob(str(g / '*/*/metadata.json'))):
        d = json.loads(Path(m).read_text())
        if d.get('game_id') and d.get('baseline_actions'):
            BASE[g.name] = list(d['baseline_actions'])
N = {g: len(b) for g, b in BASE.items()}
print('public games:', len(BASE), 'N distribution:', sorted(collections.Counter(N.values()).items()))

def real_score(game, apl, levels, state_win=False):
    """Score a single play through the REAL arc_agi scorer, building a Card the way
    Scorecard.update_scorecard would (cumulative action count at each level-up)."""
    card = Card(game_id=game)
    card.inc_play_count('g0')
    cum = 0
    for i in range(levels):
        cum += apl[i]
        card.actions[0] = cum
        card.set_levels_completed('g0', i + 1)
    # trailing actions on the unfinished level
    cum += sum(apl[levels:])
    card.actions[0] = cum
    card.states[0] = GameState.WIN if state_win else GameState.NOT_FINISHED
    ei = EnvironmentInfo(game_id=game, baseline_actions=BASE[game], title=game, tags=[])
    es = EnvironmentScorecard._calculate_score(card, game, 0, ei, None)
    return es.score

def env_score(levels_completed, apl, n, baselines):
    """rescore_waves.py:env_score, copied verbatim for comparison."""
    total_w = 0; total_score = 0.0; max_w = 0
    for i in range(n):
        w = i + 1; total_w += w
        a = apl[i] if i < len(apl) else 0
        completed = i < (levels_completed or 0)
        s = min(115.0, (baselines[i] / a) ** 2 * 100.0) if (completed and a > 0) else 0.0
        if s > 0: max_w += w
        total_score += s * w
    return round(min(total_score / total_w, max_w / total_w * 100.0), 4)

def cap(levels, n):
    return 100.0 * levels * (levels + 1) / (n * (n + 1))

# ---------------- datasets ----------------
def load_wave(path):
    d = json.load(open(path)); rows = []
    for r in d['rows']:
        rows.append(dict(src=Path(path).name, game=r['source_game'], clone=r['clone_id'],
                         lv=int(r['levels_completed'] or 0), n=r['levels_total'],
                         apl=list(r['actions_per_level']), score=r['score']))
    return rows

qwen = load_wave(REPO / 'scratchpad/qwen38/wave1_shipped_result.json')
banked = []
for stem in ['pc_base', 'w2_base', 'pc_cand', 'w2_cand', 'pkg', 'struct', 'struct2', 'struct3']:
    banked += load_wave(REPO / f'scratchpad/banked_waves_20260809/{stem}.json')
smokes = []
for f in sorted(glob.glob(str(REPO / 'scratchpad/multirole_corpus/*/benchmark.json'))):
    d = json.load(open(f))
    for g in d['game_runs']:
        smokes.append(dict(src=d['label'], game=g['game_id'][:4], clone='', lv=g['levels_completed'],
                           n=g['number_of_levels'], apl=list(g['actions_per_level']), score=g['final_score']))

# ---------------- Q5: real-scorer cross-check ----------------
def crosscheck(rows, label):
    mism = 0
    for r in rows:
        rs = real_score(r['game'], r['apl'], r['lv'])
        es = env_score(r['lv'], r['apl'], r['n'], BASE[r['game']])
        if abs(rs - es) > 5e-4 or (r['score'] is not None and abs(rs - float(r['score'])) > 5e-4):
            mism += 1
            if mism <= 5: print('   MISMATCH', label, r['game'], r['lv'], r['apl'], 'real', rs, 'env', es, 'stored', r['score'])
    print(f'REAL arc_agi scorer vs rescore env_score vs stored score: {label} rows={len(rows)} mismatches={mism}')
crosscheck(banked, 'banked(224)')
crosscheck(qwen, 'qwen38 wave1')
crosscheck(smokes, 'duck38-v12 smokes')

# ---------------- Q2: efficiency on completed levels ----------------
def eff_table(rows, label):
    ratios = []; s_i = []; per_level = collections.defaultdict(list)
    for r in rows:
        b = BASE[r['game']]
        for i in range(r['lv']):
            a = r['apl'][i]
            ratios.append(a / b[i]); s = min(115.0, 100.0 * (b[i] / a) ** 2); s_i.append(s)
            per_level[i + 1].append(a / b[i])
    ratios.sort()
    q = st.quantiles(ratios, n=4) if len(ratios) >= 4 else [None] * 3
    print(f'\n[{label}] completed levels={len(ratios)} actions/baseline: median={st.median(ratios):.2f} '
          f'mean={st.mean(ratios):.2f} Q1={q[0]:.2f} Q3={q[2]:.2f} min={ratios[0]:.2f} max={ratios[-1]:.2f}')
    print(f'   frac ratio>1.0 (slower than human) = {sum(x > 1 for x in ratios)}/{len(ratios)} = {sum(x > 1 for x in ratios)/len(ratios):.2f}; '
          f'frac s_i==115 (capped) = {sum(s >= 115 for s in s_i)}/{len(s_i)}; frac s_i<100 = {sum(s < 100 for s in s_i)}/{len(s_i)}; '
          f'frac s_i<50 = {sum(s < 50 for s in s_i)}/{len(s_i)}; mean s_i={st.mean(s_i):.1f}')
    for L in sorted(per_level):
        v = per_level[L]; print(f'   L{L}: n={len(v)} median ratio={st.median(v):.2f} frac>1={sum(x>1 for x in v)/len(v):.2f}')
    return ratios

eff_table(qwen, 'Qwen3.8 shipped wave1 (28 plays, 08-15)')
eff_table(smokes, 'duck38-v12 family smokes (39 runs, 08-17..21, cherry-picked games)')
eff_table(banked, 'banked 8 waves (224 plays, 08-09, pre-Qwen3.8 duck-base 27B)')

# ---------------- per-game score vs cap ----------------
def score_vs_cap(rows, label, per_game_max=False):
    print(f'\n[{label}] per-play score vs completion cap' + (' (per-game MAX over clones)' if per_game_max else ''))
    sel = rows
    if per_game_max:
        best = {}
        for r in rows:
            if r['game'] not in best or r['score'] > best[r['game']]['score']: best[r['game']] = r
        sel = list(best.values())
    tot_s = tot_c = 0.0; lost_eff = 0.0
    print(f"   {'game':5s} {'lv/N':6s} {'score':>7s} {'cap':>7s} {'eff-loss':>8s}")
    for r in sorted(sel, key=lambda r: (r['game'], r['clone'])):
        c = cap(r['lv'], r['n']); s = r['score']; tot_s += s; tot_c += c; lost_eff += c - s
        print(f"   {r['game']:5s} {r['lv']}/{r['n']:<4d} {s:7.3f} {c:7.3f} {c-s:8.3f}")
    n = len(sel)
    print(f'   MEAN score={tot_s/n:.4f}  MEAN cap={tot_c/n:.4f}  lost-to-efficiency={lost_eff/n:.4f} '
          f'({lost_eff/max(tot_c,1e-9)*100:.1f}% of cap)  lost-to-depth=100-cap={100-tot_c/n:.2f}  cap/score={tot_c/tot_s:.3f}')
    return tot_s / n, tot_c / n

qs_play, qc_play = score_vs_cap(qwen, 'Qwen3.8 wave1')
qs_best, qc_best = score_vs_cap(qwen, 'Qwen3.8 wave1', per_game_max=True)

# ---------------- Q3: level distribution + counterfactuals ----------------
def level_dist(rows, label):
    c = collections.Counter(min(r['lv'], 3) for r in rows)
    n = len(rows)
    print(f'\n[{label}] plays={n} levels: 0={c[0]} ({c[0]/n:.2f}) 1={c[1]} ({c[1]/n:.2f}) 2={c[2]} ({c[2]/n:.2f}) 3+={c[3]} ({c[3]/n:.2f})')
level_dist(qwen, 'Qwen3.8 wave1 per-play')
level_dist(banked, 'banked 224 per-play')
level_dist(smokes, 'duck38-v12 smokes (biased game selection)')

def counterfactual(rows, label):
    base = st.mean(r['score'] for r in rows)
    # (a) every play that reached >=1 level reaches one more, new level at cap (s=115) -> score = cap(lv+1)
    #     keep realized efficiency on existing levels: new score = min(sum w s / W + w_new*100/W ... ) simpler: cap-based
    a_cap = st.mean(cap(r['lv'] + 1, r['n']) if r['lv'] >= 1 else r['score'] for r in rows)
    a_keep = []
    for r in rows:
        if r['lv'] >= 1:
            # keep realized per-level scores, add next level at 100 (baseline pace)
            b = BASE[r['game']]; W = r['n'] * (r['n'] + 1) / 2; tot = 0.0; mw = 0
            for i in range(r['lv']):
                s = min(115.0, 100.0 * (b[i] / r['apl'][i]) ** 2); tot += s * (i + 1); mw += i + 1
            tot += 100.0 * (r['lv'] + 1); mw += r['lv'] + 1
            a_keep.append(min(tot / W, mw / W * 100))
        else:
            a_keep.append(r['score'])
    # (b) every play reaches >=1 level: zero-level plays get L1 at cap
    b_cap = st.mean(cap(max(r['lv'], 1), r['n']) if r['lv'] == 0 else r['score'] for r in rows)
    b_pace = st.mean((100.0 * 2 / (r['n'] * (r['n'] + 1))) * min(1.0, 0.86 ** -2 if False else 1.0) if r['lv'] == 0 else r['score'] for r in rows)
    print(f'\n[{label}] counterfactuals (per-play mean): observed={base:.3f}')
    print(f'   every L1+ play gets +1 level at cap (115)         -> {a_cap:.3f}  (x{a_cap/base:.2f})')
    print(f'   every L1+ play gets +1 level at exactly human pace -> {st.mean(a_keep):.3f}')
    print(f'   every 0-level play clears L1 at cap                -> {b_cap:.3f}  (x{b_cap/base:.2f})')
    print(f'   ALL plays reach L1 (0->1 at cap) AND L1->L2 at cap  -> '
          f'{st.mean(cap(max(r["lv"],1)+1 if r["lv"]<=1 else r["lv"], r["n"]) if True else 0 for r in rows):.3f}')
counterfactual(qwen, 'Qwen3.8 wave1')

# ---------------- Q4: local vs LB ----------------
lb = [1.29, 1.74, 1.55, 1.33, 1.51, 1.66, 1.50, 1.24, 1.00, 0.99, 1.45, 1.65]
same_bytes = [1.29, 1.74, 1.45]  # duck-38-v2 exact-byte draws per handoffs
print(f'\nLB series n={len(lb)} mean={st.mean(lb):.3f} sd={st.stdev(lb):.3f}; exact-byte duck-38-v2 draws {same_bytes} mean={st.mean(same_bytes):.3f}')
print(f'local Qwen3.8 wave1: per-play mean={qs_play:.3f}, per-game-max mean={qs_best:.3f}; ratio local/LB(same bytes) = {qs_play/st.mean(same_bytes):.2f}x (per-play) / {qs_best/st.mean(same_bytes):.2f}x (best-of-clones)')

# ---------------- Q6: handoff arithmetic ----------------
Ns = [N[g] for g in sorted(N)]
for k in (1, 2, 3):
    print(f'k={k} cap mean over 25 public games = {st.mean(cap(k, n) for n in Ns):.2f}  (per-N: ' + ', '.join(f'N{n}:{cap(k,n):.2f}' for n in sorted(set(Ns))) + ')')
print(f'all-win one game / 110 = {100/110:.2f}; /55 = {100/55:.2f}')
print(f'"perfect efficiency on levels already cleared": local cap/score = {qc_play/qs_play:.3f} (per-play) -> LB 1.409 x = {1.409*qc_play/qs_play:.2f}; best-of-clones {qc_best/qs_best:.3f} -> {1.409*qc_best/qs_best:.2f}')
print(f'"L1 on all games": k=1 cap mean = {st.mean(cap(1,n) for n in Ns):.2f}; 0.8x that = {0.8*st.mean(cap(1,n) for n in Ns):.2f}; 2 x 1.409 = {2*1.409:.2f}')
# What does 2.82 need? fraction of games at L1 already
frac_l1 = sum(r['lv'] >= 1 for r in qwen) / len(qwen)
print(f'fraction of Qwen3.8 wave1 plays with >=1 level: {frac_l1:.2f}')

# ---------------- Q1 demo: multi-play + reset semantics through the REAL scorer ----------------
print('\n=== Q1 demonstrations on the real scorer ===')
ei = EnvironmentInfo(game_id='ft09', baseline_actions=BASE['ft09'], title='ft09', tags=[])
# play 1: 100 actions, 1 level; play 2 (after full reset -> new_play): 50 actions, win all 6 levels quickly
card = Card(game_id='ft09'); card.inc_play_count('p1')
for _ in range(100): card.inc_action_count('p1')
card.set_levels_completed('p1', 1)
card.inc_play_count('p2'); cum = 0
for i, a in enumerate([43, 12, 23, 28, 65, 37]):
    for _ in range(a): card.inc_action_count('p2')
    card.set_levels_completed('p2', i + 1)
card.set_state('p2', GameState.WIN)
sc = Scorecard(cards={'ft09': card})
es = EnvironmentScorecard.from_scorecard(sc, [ei])
env = es.environments[0]
print('two plays -> per-play scores', [round(r.score, 3) for r in env.runs], 'game score (max over plays) =', round(env.score, 3), 'actions (sum over plays) =', env.actions)
# RESET mid-level: counts as +1 action in the current level bucket
card = Card(game_id='ft09'); card.inc_play_count('p1')
for _ in range(20): card.inc_action_count('p1')
card.inc_reset_count('p1')  # level reset (not full)
for _ in range(20): card.inc_action_count('p1')
card.set_levels_completed('p1', 1)
sc = Scorecard(cards={'ft09': card}); env = EnvironmentScorecard.from_scorecard(sc, [ei]).environments[0]
print('L1 cleared with 20 actions + RESET + 20 actions -> level_actions =', env.runs[0].level_actions[:2], 'resets=', env.runs[0].resets, 'score=', round(env.score, 3), '(baseline 43: 41 actions -> (43/41)^2*100=', round((43/41)**2*100, 1), ')')
# levels_completed DEcrease appends a tuple too (quirk)
card = Card(game_id='ft09'); card.inc_play_count('p1')
for _ in range(10): card.inc_action_count('p1')
card.set_levels_completed('p1', 1)
for _ in range(5): card.inc_action_count('p1')
card.set_levels_completed('p1', 0)   # hypothetical drop back to 0 (full reset without new play)
print('levels_completed 1 -> 0 inside one play: actions_by_level =', card.actions_by_level[0])
sc = Scorecard(cards={'ft09': card}); env = EnvironmentScorecard.from_scorecard(sc, [ei]).environments[0]
print('   scorer then reads level_scores =', [round(x, 1) for x in env.runs[0].level_scores], 'levels_completed(field)=', env.runs[0].levels_completed, 'score=', round(env.score, 3))
