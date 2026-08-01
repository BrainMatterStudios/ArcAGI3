# Iterative test programme — 2026-08-01

Seven independent research lenses (neuroscience, physics, biology, mathematics,
chemistry, competitive strategy, human behaviour), each blinded to this repo and to
the field's prior answers. This plan turns their output into an ordered experiment
queue with kill criteria and an explicit compute budget.

**The discipline, which inverts how this campaign has worked:** nothing reaches a GPU
sweep without an offline result, and nothing reaches a submission slot without a GPU
result. Thirty-four scored submissions have so far produced one usable distribution
and zero measured improvements, because arms went straight to slots.

---

## 0. The finding that reorders everything

**Frame identity has been broken all along.** 18 of 25 games paint a step-budget bar
into the frame, and the counter ticks *before* the legality check. Measured over 1,200
random actions per game:

```
corpus no-op rate   raw 0.208  ->  HUD-masked 0.455   (+24.7 points)

lf52  0.000 -> 0.996      tu93  0.000 -> 0.601
vc33  0.000 -> 0.995      r11l  0.000 -> 0.458
s5i5  0.000 -> 0.949      bp35  0.000 -> 0.369
```

The HUD is a *row*: as the bar decrements, different pixels within it change at
different times, so the **row** differs on every action while **no single pixel**
does. A per-pixel "spectator" test finds nothing; a per-region test finds this.

HUD locations, from 340 human sessions and confirmed here:
row 63 — cd82 dc22 ka59 re86 s5i5 tu93 wa30 bp35 tr87 · row 0 — cn04 vc33 sp80 lf52 ·
row 53 — sk48 sb26 · col 0 — r11l lp85 · rows 61-62 — ls20

**Consequences.** `board_changed` is wrong on 13/25 games and reports "changed" on
100% of actions in 6. Every graph-search, novelty, empowerment and cycle-elimination
idea in this document depends on frame identity and was therefore untestable until
now. It also explains why the 2026-08-01 graph explorer failed: hashes never merged,
so the state graph degenerated to a tree.

**Masking is E1 and everything else is downstream of it.**

---

## 1. What the sweep established

**Verified by measurement or source (act on these):**

| finding | source |
|---|---|
| HUD breaks frame identity; corpus no-op 20.8% -> 45.5% | measured here |
| All 15 decoded win predicates have the form `N_unsatisfied == 0` — none is a symmetry, entropy or compressibility extremum | game source |
| Avatar identifiable by action-displacement mutual information on 13/25 | measured here |
| Click-effectiveness predictable from colour alone on 4/10 testable games (0.94-0.98), weak on 6 | measured here |
| Engine exposes `_get_hidden_state`, `_get_valid_clickable_actions`, `set_level` | verified here |
| Animation is large and discarded — up to 61 frame layers per action | measured here |
| Scored actions accumulate across resets; `a(l)` is per-level independent | measured 2026-08-01 |
| Per-level score saturates at `a* ~ 0.9325 x baseline` — effort below earns exactly zero | scoring source |
| The objective is convex in actions, so the smooth Lagrangian optimum is a MINIMUM; optima are at corners | scoring source |
| Humans: 41 actions in the first 60s, whole action inventory tried in 9, 57.4% immediate repeat on first use | 340 sessions |
| Humans use reset as rollback-and-replay (35.8% followed by >=50% identical replay); agent uses it only for death recovery, 0/43 after a win | 340 sessions + 61 episodes |
| Humans use ACTION7 undo 418 times; agent has used it zero times, ever | 340 sessions + 61 episodes |
| Objectness for click targeting is ALREADY exploited: human lift 2.31x, agent 2.30x | 340 sessions + 61 episodes |
| Depth is the binding constraint: median depth 2 caps the score at 8.3% however efficient | scoring + episodes |

**Refuted (do not build):**
- Prägnanz — goals are *not* symmetry/compressibility extrema (from source).
- Spectator single-pixel odometer — zero pixels change on >=95% of actions (the real effect is regional).
- Objectness priors for click targeting — the well is dry, measured.
- Gravity, counting, containment, occlusion priors — 3, 3, 5 and 2 games of 25.
- Priors as a *policy* — one lens bolted objectness + dead-action memory + HUD masking onto a random policy across 25 games and got no benefit. These are plumbing for a planner, not a policy.

**Unresolved, gating:**
- Determinism-violation rates are contaminated by animation. Needs clean re-measurement (E2).
- `_get_hidden_state()` is narrower than claimed — did not vary on the games checked.

---

## 2. Tier 0 — offline, no GPU, no model

~13k steps/s single core through the full Arcade wrapper. 25 games with ground-truth
oracles and 15 known win predicates, so most of these are *supervised* tests.

**E1. HUD mask everywhere.** DONE — measured above. Ship a per-game mask into every
frame-identity path: `board_changed`, hashing, novelty, dead-action memory.
*Kill:* n/a, already established. *Cost:* hours.

**E2. Re-measure with masking on.** Redo three things that were run on broken frame
identity: (a) determinism violations, with animation properly handled — settle which
games are frame-Markov; (b) state-collapse ratio for graph search; (c) the cross-level
transfer experiment, whose knowledge carrier may have been fine while its substrate
was broken. *Kill:* if masked collapse stays ~1.0x, graph search is dead for a
different reason and everything downstream of it drops. *Cost:* half a day.

**E3. Defect-count goal detector.** Every decoded goal is `N_unsatisfied == 0`.
Identify consumed species by monotone-decrease-plus-reset-restoration, then rank
states by the resulting defect count. *Test:* against the 15 known predicates — a
properly supervised evaluation. *Pre-registered bar:* true goal states in the top 2nd
percentile of reachable states on >=20/25, AUC > 0.95. *Kill:* if a generic
entropy/compressibility control does comparably, the apparatus is over-engineered.
*Cost:* 1 day. **Highest-value item in the programme.**

**E4. Confirm-by-repetition opening book.** ~30 fixed actions, no model: try each
action, repeat 2-3x to confirm consistency and read off the displacement, intersect
changed-cell sets to find the controlled object, seed the dead set. Humans complete
this in 9 actions / 17 seconds; our agent has no equivalent. Level 1 carries weight
1/36, so the opener costs ~3 points of environment score against a depth ceiling that
runs 8.3% -> 58.3% from two levels to six. *Kill:* if it fails to recover the action
model on >=15/25. *Cost:* 1-2 days.

**E5. Cycle elimination and geodesic replay.** If two masked frames hash identically,
the actions between are a provable null cycle — deletable with no replay. Then BFS the
recorded graph for the shortest path to a winning node. Cannot regress: a candidate is
adopted only after being executed and observed to win. *Kill:* if median reduction
< 15%. *Cost:* 1 day. Note this is worth far less than it looks under accumulate
semantics — its real use is shortening the *policy* carried to later levels.

**E6. Selection-vs-avatar-vs-none classifier.** Only 7/25 games have a fixed avatar;
in 10 the controlled thing is a *selection* signalled by recolouring, and 8 have none.
Hunting for an avatar is actively misleading and produced two frozen ft09 runs.
*Kill:* if it cannot classify >=20/25 correctly against the source-derived labels.
*Cost:* 1 day.

**E7. ACTION7 as the probe primitive.** Undo in all 6 games offering it, and free of
in-game step budget in 5. Humans 418 uses, us zero. *Verify first:* whether undo still
counts as a scored action — that decides everything. *Cost:* hours to check, 1 day to use.

---

## 3. Tier 1 — GPU commit sweeps

`make_benchmark_kaggle_official_110(competition_sim=True)` builds the submission-shaped
110-game benchmark from the 25 public games with a local competition arcade — shared
scorecard, hidden baselines, clone IDs — so a commit kernel produces a real,
submission-shaped score at zero submission cost, retrievable via `kaggle kernels output`.

```
per-game box    wall (110 games, concurrency 28 = 4 waves)   sweeps / 30h week
   7920s real            8.8h + load                              ~3
    900s                 1.0h + load                             ~25
    600s                 0.7h + load                             ~35
```

A shortened box changes the regime, so absolute scores are not comparable to real
submissions — but **arms are comparable to each other**, which is what an A/B needs.
That is 25-35 paired comparisons per week against 7 submission slots.

**G1. Variance A/B.** temp 0.6/top_k 20 versus 0.9/50 — the max-over-draws bet, which
currently rests on an untested assumption that wider sampling fattens the upper tail
rather than lowering the mean. Paired sweeps give the distribution directly.
*Allocation:* 8 sweeps (~6h).

**G2. Best Tier-0 survivor, integrated.** Whichever of E3/E4/E6 clears its kill
criterion, wired into the duck and measured end-to-end. *Allocation:* 8 sweeps.

**G3. Context-levers confirmation.** The arm submitted today, measured properly rather
than from one noisy draw. *Allocation:* 4 sweeps.

Leaves ~10h/week headroom for reruns and failures.

---

## 4. Tier 2 — submission slots

One per day. Every day used — an unused day is a discarded lottery ticket, and
E[best of N] runs 1.21 at N=8 and 1.41 at N=90 on the measured base distribution.
Slots go to arms that cleared Tier 1; otherwise farm draws from the best-known config.
Freeze and farm exclusively for the final ~30 days.

---

## 5. Order of work

1. **E1 + E2** — masking, then re-measure everything built on broken frame identity.
   Nothing else is trustworthy until this is done.
2. **E3** — the defect-count goal detector, supervised against known predicates.
3. **E4, E6** in parallel — opening book and controlled-object classifier.
4. **G1** — variance sweeps, which run independently of all the above.
5. **E5, E7** — geodesic replay and undo, both cheap and both contingent on E2.
6. Whatever clears Tier 1 goes to a slot.

## 6. Standing risks

- **The 25 public games are the sample; 55 unseen private games are the target.** The
  Foundation states public scores are "emphatically not a valid measure of progress",
  and one competitor went 4.29 local to 0.22 hidden. Tier 0 and Tier 1 rule things
  *out* cheaply; only a slot can rule something *in*.
- **Priors are not policies.** Directly measured this sweep: objectness + dead-action
  memory + HUD masking on a random policy produced no benefit. Every Tier-0 result
  must be judged on scored outcome, not on whether the mechanism works.
- **Roughly half the sweep's headline claims did not survive checking.** Verify each
  load-bearing claim before building on it. Two of today's refutations were of my own
  conclusions, and one refutation of mine was itself wrong.
