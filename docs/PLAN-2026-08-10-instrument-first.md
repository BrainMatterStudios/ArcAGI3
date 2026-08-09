# PLAN 2026-08-10 — instrument first, then free wins, then certify

Supersedes the four-track plan in `HANDOFF-2026-08-09-structural-campaign.md` §5.
Written after the 08-09 evening review cascade, which changed the position materially.
Every claim below carries its evidence; anything uncertified is labelled as such.

## 0. What changed on the night of 08-09

Four independent reviews plus a six-lens research workflow ran against the campaign's
own artifacts. Net effect: **the campaign's one replicated positive is now suspect, and
the most exciting new lever turns out to have been shipped in August and scored 0.76.**
Nothing here is a new setback — both facts were already true. We could not see them
because the measuring instrument was reading a statistic that is not the score.

### 0.1 The instrument was wrong (the finding that reframes everything)
All five screen classifiers decide on an **unweighted, ft09-excluded level count**:
`struct_screen_config.py:158,239,309`; `patch_closure_config.py:96-101,216-230,267-282`;
`package_screen_config.py:108-113,182-184,227-233`; `classify*.py` are thin CLIs.
None references `baseline_actions`. Meanwhile `pc_driver.pc_env_score`
(`pc_driver.py:438-451`) already computes the TRUE objective per row and stores it at
`:652` — under a header that reads
`# --- offline scoring (informative only; the classifier reads levels)`.

Re-scored on the true objective (implementation validated against the stored per-row
score: **140/140 rows, zero mismatches > 5e-4**):

| wave | raw levels (the verdict stat) | TRUE score, 25 games | ft09 term |
|---|---|---|---|
| base_w1 | 11 | 1.4751 | 14.286 |
| base_w2 | 12 | 1.2039 | 14.286 |
| struct_w1 | 17 | 1.5198 | 0.000 |
| struct_w2 | 12 | 0.7866 | 0.814 |
| struct_w3 | 12 | 0.7667 | 0.000 |

base 1.3395 (n=2) vs struct 1.0244 (n=3) = **−23.5%**, versus the reported
"+2 levels/wave replicated positive". Excluding ft09 (the screen's own convention)
struct is **+32%** — so the sign hinges entirely on ft09, the largest single score term,
which `patch_closure_config.py:51` removes from the decision rule on a stated rationale
("8 levels") that is **factually wrong**: metadata and every banked row say 6.

**Honest limit:** n=2 vs n=3, and struct's best wave (1.520) beats both base waves, so
the overall means are NOT statistically separated. The consistent, replicated part is
ft09: base 2/2, struct 0/3, with struct burning 68–124 actions on a level base clears
in 7. Treat "struct is a regression" as a strong hypothesis, not a verdict.

Data rescued to `scratchpad/banked_waves_20260809/` (was the only copy, in another
session's `/private/tmp`).

### 0.2 Corrected facts (supersede earlier statements in this campaign)
- **Slot power.** We hold n=9 controls, so the equal-n two-sample formula was the wrong
  frame. Correct: `se = sd·sqrt(1/9 + 1/k)`. One new slot detects **0.58–0.74** at 80%
  power (0.455 at 50% power); two slots ~0.43–0.54. Retire `0.761/delta^2`. With n=1 vs
  9 the smallest attainable non-parametric p is **0.10** — one slot can never certify
  distribution-free at any effect size.
- **A slot returns two observables, not one.** `totalBytes` of the gateway-written
  parquet is in every submission row; null across the nine identical base draws is
  **3676.8 ± 21.8**, corr with score 0.003. Useful only for "touches more games /
  more distinct outcomes" hypotheses — it is NOT a second score, and its near-zero
  correlation may mean it carries little about outcome quality. Pre-register it as a
  secondary endpoint when an arm predicts a coverage change. Scored-run *logs* remain
  unretrievable (confirmed: `kaggle kernels output` returns the COMMIT run).
- **The context-window lever was already shipped.** Submission 55163445
  (`arc-agi-3-duck-levers`, scriptVersionId 339499554) carried
  `_LOCAL_ANALYZER_CONTEXT_WINDOW = 49152` AND the `//3 -> //4` estimator fix and scored
  **0.76** (`SUBMISSION-LEDGER.md:43,102-103`). The seam is proven
  (`harness_levers.py:26-28,154-158`); the economics are not.
- **Wall-clock, not actions, is the binding constraint.** 140/140 recorded game-runs
  ended `gave_up` at the full box; zero hit an action budget. Prefill 1,884 tok/s vs
  generation 164.5 tok/s at 11.45:1 prompt:generated ≈ 1:1 in time. Doubling retained
  prompt ≈ +50% per-turn latency ≈ one third fewer turns, and depth is what scores.
  The trimmer is pinned at the ceiling on only **11.8%** of requests.
- **The chars/3 estimator overcounts by 1.44×** (p50 1.39, p90 1.66) measured over 3,556
  real requests (`harness_levers.py:10-16`) — worse than the "~1/3" I first estimated.
- **patch10 (`TAAF_WIN_REPLAY`) is SAFE**, but for a different reason than first argued:
  corruption requires the replay guard to pass, the guard requires `levels_completed==0`,
  and that requires `full_reset()` — which creates a fresh play slot. Mutually exclusive.
  All three gateway branches are benign. Defect found: its soft-time guard
  (`duck_patches.py:1447-1449`) is **inert at eval** because `soft_end_time = None` under
  `run_as_submission`.
- **The geodesic postpass has never fired in any scored submission** (`api.py:424-425`
  refuses a competition-mode remake). Free to remove.

## 1. Track 0 — repair the instrument (ZERO GPU, ZERO SLOTS, do first)

This is the highest-value work available, because it re-adjudicates every verdict the
campaign has ever recorded at a cost of about one day and no budget.

1. Promote `pc_env_score` from "informative only" to **the decision statistic** in
   `patch_closure_config`, `package_screen_config`, `struct_screen_config`.
2. Delete `LEVELS_EXCLUDED_GAMES = ("ft09",)` or replace it with a justified rule; the
   current justification is factually wrong. Report ft09 explicitly either way.
3. Re-adjudicate the banked closure, package and struct verdicts on the true objective.
   Publish a corrected verdict table alongside the original one.
4. Add to every report: per-game true score, a completed-level depth histogram, and the
   share of clones where the completion-share cap binds (measured 29–58%, so efficiency
   fully determines the score on roughly half the games — a depth-only analysis is not
   sufficient).
5. Add a regression test asserting classifier output == `pc_env_score` aggregation, so
   the two can never drift apart again.

**Kill criterion:** none — this is repair, not a bet. **Expected outcome:** at least one
recorded verdict flips. Struct is the leading candidate; closure (+6 then +1) is next.

## 2. Track 1 — diagnose the ft09 regression (ZERO GPU)

**Hypothesis (falsifiable):** the structural plan channel inflates actions on games whose
early levels yield to short direct solutions, and the quadratic efficiency term makes
that inflation expensive. ft09 is the visible case: base clears L1 in 7 actions, struct
spends 68–124 and mostly fails.

**Test, entirely from banked rows:** for every game and wave, correlate the struct arm's
plan usage and plan length against per-level action inflation vs the base arm. If the
inflation is general and merely largest on ft09, struct is a broad harm and the
ft09-excluded +32% is an artifact of which games were excluded. If ft09 is idiosyncratic,
struct may still be net positive and needs the corrected certification.

**Why this matters more than another wave:** it decides whether the 4–6 wave, ~12 GPU-hour
struct certification is worth booking at all — and it costs nothing.

## 3. Track 2 — free wins (ZERO GPU, small builds, ship as a stack)

These are configuration and plumbing corrections rather than behavioural bets, which is
the class most likely to survive public→hidden transfer (v13 transfer-dense +41% dev and
CAI-prune +47% dev both scored Kaggle-inert; config fixes have no such failure mode).

| # | change | why | cost |
|---|---|---|---|
| F1 | `_estimate_tokens` `//3 -> //4` | recovers ~30% of the window we ALREADY pay for, at zero extra prefill; errs high (~1.08×) so it cannot overrun the served window | exists in `harness_levers.py` |
| F2 | trimmer drops from the MIDDLE, not the front | front-drop invalidates the whole KV prefix past the ~3.1k system prompt on every trim, against a measured 49.3% hit rate; pure latency recovery | not built, small |
| F3 | `TAAF_GEODESIC_POSTPASS=0` | never fires at eval; removes ~2 API round-trips per game | one env pin |
| F4 | fix patch10 soft-time guard | inert at eval today; bounded exposure but the protection is illusory | few lines |
| F5 | delete the "minimize actions" line in `prompts.py:17` | the completion-share cap binds long before efficiency; actions on an uncompleted level cost nothing | one line |

F1+F2 are the substantive pair: **more usable context at no wall-clock cost**, which is
strictly better than the window raise that was already tried and scored 0.76.

**Do NOT raise the window to 65536.** `LOCAL_ANALYZER_MAX_OUTPUT=0` means no `max_tokens`
is sent, so vLLM grants `max_model_len − prompt_len`; at 65536 that starves generation to
~5.8k tokens against thinking traces. 49152 leaves ~21k.

## 4. Track 3 — what earns GPU (next fresh quota, Saturday)

Quota resets Saturday; ~3h remain this week, so **~1 wave, not 11** (the "11 waves"
figure is a fresh-quota number quoted from `build_rig.py:23`).

- **Wave A (first, once Track 0 lands):** the Track 2 free-win stack vs base, scored on
  the corrected objective, at **eval geometry (28 clones @ 7920s)** — not rig geometry
  (10 @ 3600s), which under-tests the 28-way contention that is the whole risk. This is
  the never-run eval-geometry A/B from `PLAN-REVIEW-2026-08-04.md:29`.
- **Wave B (conditional):** struct certification, 4–6 waves — **only if** Track 1 says
  the ft09 loss is idiosyncratic. If Track 1 shows general action inflation, skip it and
  instead test struct-OFF as an arm.
- Every wave reports the true objective, ft09 included, plus the depth histogram.

## 5. Track 4 — slot doctrine (revised)

- Slots are **transfer tests for offline-certified stacks**, never instruments. Nothing
  below ~+0.6 is readable in one slot.
- Pre-register the reading rule and any secondary endpoint (`totalBytes`, null
  3676.8 ± 21.8) **before** firing, in the submission message.
- Duplicate the best-certified config into both final selections (private LB, same run,
  no rerun).
- ~84 slots remain to the ~Oct 20 freeze after tonight.

## 6. What would have to be true to reach 1.86

Blunt: the base mean is 0.970 and the field leader is 1.86 on a best-of-61 draw (true
mean likely ~1.5). Closing that needs roughly **+0.5 to +0.9 of real, transferable
gain** — far beyond anything measured. No single lever in this plan does it.

The honest reading of tonight is that the campaign has been **optimising against a
mis-specified objective for weeks**, so the recorded history of "no arm has a measured
improvement" is not strong evidence that no arm helps — it is evidence that we could not
tell. Track 0 is therefore not bookkeeping; it is the first opportunity to learn whether
anything we built ever worked.

The plausible path, in order of expected value:
1. A corrected instrument reveals an existing arm was mis-adjudicated (cheap, possible).
2. The free-win stack (F1+F2) buys real turns-per-game at no wall-clock cost, and turns
   convert to depth, and depth is quadratically weighted (cheap, plausible, untested).
3. Removing a harm (over-planning / action inflation) beats adding a mechanism —
   suggested by the ft09 pattern (cheap, testable in Track 1).
4. Everything else is a capability problem we cannot buy: K3 scored 47.6 on ft09 where
   our 27B scored 0.0, and both alternative brains gated NO_GO with controls.

If Tracks 0–2 all come back null, the honest conclusion is that 1.30 is near our ceiling
with this brain, and the remaining play is draw-farming into the two final selections —
which caps around ~1.41 (E[max] over remaining slots) and does not win.

## 7. Immediate next actions (in order)

1. Track 0 items 1–3 (classifier fix + re-adjudication) — half a day, no budget.
2. Track 1 ft09 diagnosis from banked rows — half a day, no budget.
3. Build the Track 2 stack behind env gates with tests, dry-run verified.
4. Saturday: Wave A at eval geometry.
5. Revert or commit the stale 35B edit in `scratchpad/taaf_scored_ref/setup_commands.json`
   — it points the harness at the NO_GO model and is uncommitted working-tree state.
