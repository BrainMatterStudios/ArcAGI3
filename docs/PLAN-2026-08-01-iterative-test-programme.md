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
CORRECTED BY MEASUREMENT (rig run #3, 2026-08-01). Shortening the per-game box does
not shrink the regime, it eliminates it:

  110 games @ 600s   ->  mean 8.8 ACTIONS/game (median 6), a level completed in only
                         5 of 110 games. 28 games share one GPU, so a turn costs ~70s
                         wall. Six actions is barely past the opening.
  the real 7920s box ->  ~113 actions/game, consistent with the audit's measured
                         median of 78 in real recorded episodes.

Buy per-game realism by cutting GAME COUNT, not the box — one concurrency wave:

  28 games @ 7920s (1 wave)  ->  ~2.2h + load  ->  ~11 sweeps / 30h week   REALISTIC
  28 games @ 3600s (1 wave)  ->  ~1.0h + load  ->  ~23 sweeps, ~51 actions/game
  110 games @ 600s (4 waves) ->  ~0.7h         ->  MEASURES ALMOST NOTHING
```

Power comes from 28 realistic games, not 110 truncated ones. The earlier claim of
25-35 sweeps/week was wrong: it assumed a shortened box preserves the regime, and the
measurement says it does not. The honest figure is **~11 realistic sweeps per week**,
still a large gain over 7 submission slots, and the only instrument that can resolve
effects the public leaderboard cannot.

**G0. Measure rho(public, private). RUN THIS FIRST — it gates G1 and all of Tier 2.**

The max-over-draws programme assumed the public half is a noisy proxy for the private
half *of the same run*, so selecting the best public draw selects a good private score.
**The public 55 and private 55 are different games.** If the noise is predominantly
per-game rather than per-run, then rho(public, private) ~ 0, selecting on the public
leaderboard returns a *random* private draw, and best-of-N yields the private MEAN, not
the private max. That would remove the justification for both the variance arm and the
farming programme.

*Design:* run the 110-game competition-sim N times; split each run into two disjoint
55-game halves; correlate half-A mean against half-B mean across runs. This decomposes
run-level noise (serving throughput, scheduling — correlates the halves) from game-level
noise (does not). It measures exactly the quantity the strategy depends on.

*Read:* rho >= ~0.5 — selection works, farming and variance both stand. rho ~ 0 —
selection buys nothing; stop spending effort on draw strategy and move it all to Tier 0
capability work. Intermediate — compute the attenuated expected gain and decide on the
number.

*Allocation:* 8 sweeps (~6h). Cheapest possible answer to the most consequential open
question in the campaign.

**G1. Variance A/B — HELD until G0 reports.** temp 0.6/top_k 20 versus 0.9/50. The arm
is built, pushed and verified, but its entire rationale is downstream of G0.
*Allocation:* 8 sweeps (~6h), conditional.

**G2. Best Tier-0 survivor, integrated.** Whichever of E3/E4/E6 clears its kill
criterion, wired into the duck and measured end-to-end. *Allocation:* 8 sweeps.

**G3. Context-levers confirmation.** The arm submitted today, measured properly rather
than from one noisy draw. *Allocation:* 4 sweeps.

Leaves ~10h/week headroom for reruns and failures.

---

## 4. Tier 2 — submission slots

One per day; never leave a day unused, since a draw costs nothing beyond the day and
the banked best is never lost. Slots go to arms that cleared Tier 1; otherwise draw
from the best-known config.

**The strength of the farming rationale is now conditional on G0.** E[best of N] runs
1.21 at N=8 and 1.41 at N=90 on the measured base distribution — but that is the
expected max of the *public* score. What is scored is the private half, on different
games. If rho ~ 0 the realised private score is the mean regardless of how many draws
we take, and farming is merely free rather than valuable. Do not restate the 1.41
figure as a private-score expectation until G0 reports.

Additional mechanics confirmed by the competitive lens and worth holding:
- **Scoring counts all 110 environments whether or not the agent touches them.**
  Abandoning a hopeless game costs nothing in the denominator, so early abandonment is
  strictly free — this strengthens allocation work and closes the "skip games" idea.
- **Max-over-plays is structurally unreachable**: competition mode forces level-resets
  and permits one `make` per environment.
- **Games are near-binary.** Hosts report Opus 4.6 at 0.0% *or* 97.1% on the same game
  depending only on harness, and 0.0% under both on bp35. Persistence on a non-yielding
  game has near-zero option value; abandon early and hard.
- **Dead clicks count as scored actions** (host-confirmed, contradicting the technical
  report), and there is an animation observation tax. Both are irrelevant at our current
  depth and become significant at levels 4-7, where weights are 14-25% each.
- Scorecards auto-publish after ~30-45 min of inactivity; keep it continuously active.
- **Baselines verified current.** Local `metadata.json` is dated 2026-06-27/24, after
  the 2026-04-14 scoring change (median-human baseline, 1.15x cap). tu93 reads
  `[19, 16, 34, 42, 123, ...]`, matching the values reconstructed independently from 340
  human sessions. Efficiency arithmetic in this plan is sound.

---

## 5. Order of work

1. **E1 + E2** — masking, then re-measure everything built on broken frame identity.
   Nothing else is trustworthy until this is done. Offline, no GPU.
2. **G0** — rho(public, private). Runs on GPU in parallel with the Tier-0 work and
   decides whether the entire draw/variance programme has a rationale.
3. **E3** — the defect-count goal detector, supervised against known predicates.
4. **E4, E6** in parallel — opening book and controlled-object classifier.
5. **E5, E7** — geodesic replay and undo, both cheap and both contingent on E2.
6. **G1** — variance sweeps, only if G0 justifies them.
7. Whatever clears Tier 1 goes to a slot.

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

---

## REORDERED 2026-08-02 — depth is the entire objective

Two independent judges, working from disjoint samples (10 rig-completed levels; 66
episode-completed levels), converged: **efficiency headroom is exhausted and 100% of
remaining score is depth.** Corroborated on the hidden set: 0.36 levels/game x 3.52%
mean cap = 1.27 = our exact leaderboard score.

DECISIONS:
- **G0 CANCELLED.** The decision it gates is invariant to its answer: E[best of 90
  public draws] = 1.41 < rank-16 at 1.50, so farming cannot reach the target even at
  rho=1. Frees ~16 GPU-h.
- **Efficiency track KILLED**: E5 geodesic replay, cycle elimination, token diets,
  boardfix-as-arm (also directionally negative per-turn). Ceiling ~+8% relative
  against a binding cap.
- **Exploration REPRICED as nearly free**: 1.28x action slack before any loss; a level
  completed at 2x baseline still yields 25% of its weight vs 0% for not completing.
  E4 (opening book) and E7 (ACTION7 probing) move UP.
- **Extreme-value framing**: one game taken 0->6 = +0.673 > the whole 0.59 gap to the
  leader. Objective is max P(some game goes deep).

BUILD QUEUE (this file's E-numbers superseded where they conflict):
- D1 defect pack: game_over world-model wipe (36/44 episodes), RESET un-strip
  (solver.py:116-117 strips unconditionally; humans rollback-replay 35.8%), ACTION7
  guidance (mapping fix alone measured inert).
- D2 HUD-masked stall detector (zero-level vs completer separation: distinct-boards
  0.672 vs 0.940, dead-reissue 15.6% vs 1.3%) + deep-first allocator. Allocator must
  be deadline-aware or freed slots refill with depth-0 games.
- D3 rig: action-denominated per-game budget (max_actions_per_game exists at
  solver.py:745; wall-clock budgets caused the boardfix confound).

Validation ladder unchanged: offline vs 44 recorded episodes -> one rig sweep -> slot.
