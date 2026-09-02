# EXPERIMENT 2026-09-02 — Same-boot A/B instrument control (A/A + reversed B/A)

Pre-registered 2026-09-02 ~10:30 UTC, BEFORE either kernel was pushed. Nothing below
may be edited after data arrives; post-hoc readings go in a separate "exploratory" section.

## Why this exists (and how it serves the >7 goal)

The harness-lever ladder (TP9 -> TP10 -> TP11 -> TP12) is graded entirely by the
same-boot two-phase kernel (`submission/_v22_ab/build_v22_ab.py`). Every such kernel to
date ran STOCK in phase 1 and the GRAFT in phase 2, and no stock-vs-stock control has ever
been run. Phase 2 runs on a server whose vLLM prefix cache, CUDA graphs and page cache are
warm from 25 games of identical initial prompts. If phase 2 is systematically advantaged,
the TP9 "+50% levels" read (arc3-v22-ab-tp9: 1.04 -> 1.56 lv/game) is confounded, and the
live pair 1.71 stock -> 1.72 TP9 (subs 55927189 / 55950252) is exactly what a null lever
would produce.

Zero-slot local reads are the ONLY instrument for harness work between daily slots. If it
is broken, every lever read is void and no slot should carry a lever until it is fixed.
This experiment adds no points by itself; it decides whether the ladder is readable.

## Design

Two kernels, pushed together (Kaggle allows two concurrent GPU sessions), each a byte-level
variant of `arc3-v22-ab-tp9` differing ONLY in the phase list and a read-only `/metrics`
snapshot per phase (verified by notebook diff: cells 0 and 8 only).

| kernel | phase 1 | phase 2 | graft installed | flags |
|---|---|---|---|---|
| arc3-v22-ab-tp9 (already run 09-01) | stock | tp9 | graft_pipeline | 0 / TP9=1 |
| **arc3-v22-aa** | stock_a | stock_b | graft_pipeline | 0 / 0 |
| **arc3-v22-ba-tp9** | tp9 | stock | graft_pipeline | TP9=1 / 0 |

Together these form a crossover. Facts established before the design (all from primary
data this session):
- Every game in every phase runs to the 7,920 s per-game cap (25/25 `gave_up` in all four
  prior phases), so a phase is a fixed ~2.2 h and each kernel ~5 h.
- TP9's flags are read per call (`graft_pipeline.py:146`), so reversing the order is valid.
- No cross-phase state leaks through the harness: the runtime-state file is overwritten at
  game start and unlinked at game end (`solver.py:271, :336`), the analyzer is built fresh
  per game (`solver.py:1224`), and the transcript file is write-only in the agent.
- The remaining cross-phase carrier is the vLLM server itself (prefix cache, graphs). The
  metrics snapshot measures it; it is deliberately NOT reset, because the question is
  whether the instrument AS USED is readable.

## Metrics and estimators

Primary: levels/game, paired per game (25 games). Secondary: mean score, zero-level count,
total actions, per-phase vLLM prefix-hit rate and mean e2e latency.

Let D = phase2 - phase1 (paired mean). Known: D_AB = +0.52 lv (paired sd 0.94, SE 0.19;
score +2.15; zero 8 -> 3). Cross-boot stock-vs-stock (tp10 kernel stock - tp9 kernel
stock): +0.32 lv, sd 1.26, SE 0.25.

- TP9 effect T = (D_AB - D_BA) / 2   (SE ~0.14).
- Order effect, pooled: O_hat = (D_AB + D_BA + D_AA) / 3   (least squares under
  D_AB = T+O, D_BA = O-T, D_AA = O; SE ~0.12). D_AA alone has SE ~0.19-0.25 and cannot
  bound O below 0.3 on its own — it is reported, but O_hat is the estimate.
- Position-specific TP9 effects, reported explicitly: T1 = D_AA - D_BA (TP9 in position 1),
  T2 = D_AB - D_AA (position 2). Their difference has SE ~0.23, so an interaction cannot be
  detected at n=25; it is reported, not used for a decision.
- Every quantity is reported with its paired SE and a 95% interval.

## Phase-validity rules (a void phase voids its kernel's reading)

- A TP9 phase whose graft counters (resumes_injected + perturbations_injected +
  retries_used) are all zero is VOID: the arm did not fire. A stock phase with any nonzero
  graft counter is VOID: the arm leaked.
- A phase whose vLLM server pid changed, whose vLLM counter deltas are negative, or whose
  metrics snapshot carries an `error`, is VOID: the server restarted or was unreachable
  mid-phase (boot cell 8's recovery path can also flip the serving args on restart).
- A phase with fewer than 25 game records is VOID.

## Reading rules (levels/game, paired; decided before data)

- **R1 instrument.** |D_AA| >= 0.30 or O_hat's 95% interval excludes 0 -> an order effect
  is DETECTED; all prior single-order same-boot reads are void; future A/Bs must be
  counterbalanced (two kernels, opposite orders) and no lever flies on a single-order read.
  Otherwise -> "no large order effect detected" (NOT "instrument certified": with SE ~0.12
  a true O of ~0.25 remains possible); single-order reads may be used only for effects
  >= 0.5 lv, and anything smaller needs the counterbalanced pair.
- **R2 TP9 confirmed.** D_BA <= -0.25 (TP9 first still beats stock second) AND T >= +0.30
  -> TP9 is real and order-independent locally. Then: the live 1.72 is a variance draw;
  proceed to the TP9+TP10 composition A/B, counterbalanced.
- **R3 TP9 not slot-worthy.** D_BA >= +0.10 (stock second beats TP9 first) -> T <= 0.21
  with an interval including 0; the original +50% is not reproduced. TP9 is DEMOTED (not
  "refuted" — the data cannot exclude a small true effect): it does not fly, it leaves the
  composition plan, and the lever list is re-derived with the instrument question settled
  first. The flat live pair needs no further explanation under this branch.
- **R4 inconclusive.** -0.25 < D_BA < +0.10 -> T reported with SE; no TP9-containing arm
  flies until a counterbalanced replicate.
- **R5 mechanism (A/A kernel ONLY — in B/A the arm changes the request mix, so its vLLM
  deltas are descriptive, not a position read).** If stock_b shows prefix-hit rate > +0.15
  absolute over stock_a or mean e2e latency lower by > 15%, the order effect has a named
  cause (warm cache) and the next A/A adds `POST /reset_prefix_cache` between phases as the
  design fix under test.

## Tonight's slot (09-03 00:01 UTC) — decision tree, also pre-registered

Both kernels should complete ~5 h after push. Decision at the read:
- R2 -> fly the TP9 redraw (byte-identical, svid 346574220 attested) to grow live n.
- R3, R1-fail, or R4 -> fly the stock v31-copy redraw (svid 346312727) — grows n on the
  base and costs nothing in expectation; TP9 does NOT fly.
- Kernels not complete by 22:00 UTC -> stock redraw by default.
Ahmed's go is required before any submission is launched.

## Cost and push order

~10 GPU-hours (two ~5 h sessions), zero submission slots. Weekly GPU quota cannot be read
from the CLI; B/A is pushed FIRST because it is the kernel that decides TP9 — if the second
push is refused on quota, the decisive half still runs. Each kernel is pushed from its own
directory (`submission/_v22_ab/push_<design>/`) so the shared kernel-metadata.json can never
route a push to the wrong notebook.

## Judge review (fresh-context adversarial, before push)

Verdict SHIP-WITH-FIXES; all fixes applied above: per-design push dirs; graft-counter and
server-pid attestation per phase; R1 pooled estimator instead of a single-kernel pass;
R3 relabelled demoted-not-refuted; R5 restricted to A/A; phase-validity rules added.
Judge-verified clean: no graft state leaks into a stock phase (flag read per call on all
three wrapped methods incl. the retry path); `bm.run` is re-entrant (solver deep-copied per
run, fresh stop event, per-call pass indices, seed not sent so sampling is random in both
phases); phase 2 overwrites phase 1's artifacts/transcripts but the model never reads them.
Judge's alternative for the NEXT round if this one is inconclusive: two B/A kernels instead
of A/A + B/A tighten T's SE from 0.14 to 0.12 and give the boot-to-boot sd of D directly.

## Results (filled AFTER completion; nothing above changed)

### B/A kernel `arc3-v22-ba-tp9` — COMPLETE 14:58 UTC (svid in Kaggle; results in
`submission/_v22_ab/results/arc3-v22-ba-tp9/ab_results.json`)

| phase | position | score | lv/game | zero | actions | graft counters | pid | prefix-hit | requests | mean e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| tp9 | 1 | 3.91 | 1.08 | 7 | 2,727 | resumes 708 / perturb 0 / retries 0 | 631->631 | 0.137 | 1,705 | 113 s |
| stock | 2 | 5.66 | 1.20 | 6 | 2,798 | 0 / 0 / 0 | 631->631 | 0.151 | 1,786 | 108 s |

Both phases VALID (arm fired in tp9, silent in stock; pid unchanged; no negative deltas;
25/25 games each, all to the 7,920 s cap).

**D_BA = +0.120 lv/game** (paired sd 0.88, SE 0.18, 95% CI [-0.23, +0.47]; 7 games up,
3 down; score +1.74). Stock in position 2 beat TP9 in position 1.

- **T = (D_AB - D_BA)/2 = +0.20** (SE ~0.13, CI ~[-0.06, +0.46]).
- O' = (D_AB + D_BA)/2 = +0.32 (order-effect estimate from the two B-containing kernels;
  pooled O_hat awaits the A/A).

**Pre-registered verdict: R3 fires (D_BA >= +0.10) — TP9 is DEMOTED, not slot-worthy.**
The original +0.52 is not reproduced with the order reversed; the point estimate of the
true lever is +0.20 with an interval including 0, and the order effect estimate (+0.32) is
larger than the lever estimate. The flat live pair (1.71 -> 1.72) needs no further
explanation. Per the decision tree: TP9 does NOT fly tonight; the stock v31-copy redraw
(svid 346312727) is the default arm, pending Ahmed's go.

Boundary note (exploratory, labelled): D_BA sits 0.02 above the R3 threshold, well inside
one SE of R4; the decision is the same under R4 (no TP9 flight), so the boundary does not
change any action. Mechanism note (exploratory): in this kernel phase 2 had only a slightly
higher prefix-hit rate (+0.014) and 4.5% lower mean latency, and 5% more completed
requests — a warm cache is not obviously the carrier; TP9's resume injections (708) changed
the request mix, so R5 stays A/A-only as pre-registered. Also notable: the livelock breaker
and the retry path never fired in 25 games — only the resume behaviour was ever live.

### A/A kernel `arc3-v22-aa` — COMPLETE 15:07 UTC (`submission/_v22_ab/results/arc3-v22-aa/ab_results.json`)

| phase | position | score | lv/game | zero | actions | graft | pid | prefix-hit | requests | mean e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| stock_a | 1 | 4.45 | 1.08 | 6 | 2,736 | 0/0/0 | 631->631 | 0.141 | 1,766 | 107.9 s |
| stock_b | 2 | 4.58 | 1.20 | 6 | 2,570 | 0/0/0 | 631->631 | 0.147 | 1,756 | 107.6 s |

Both phases VALID. **D_AA = +0.12 lv/game** (paired sd 1.17, SE 0.23, CI [-0.34, +0.58];
10 games up, 7 down — identical bytes, identical server: re86 went 3 -> 0, sb26 1 -> 4).

### Crossover verdict (all three kernels)

| quantity | estimate | SE | 95% CI |
|---|---|---|---|
| T (TP9 effect) | +0.20 | 0.13 | [-0.05, +0.45] |
| O_hat (order effect, pooled) | +0.25 | 0.12 | [+0.03, +0.48] |
| T1 (TP9 in position 1) | 0.00 | ~0.29 | — |
| T2 (TP9 in position 2) | +0.40 | ~0.30 | — |

- **R1: ORDER EFFECT DETECTED** (O_hat's interval excludes 0, marginally). Every prior
  single-order same-boot read is void as a lever measurement; all future A/Bs must be
  counterbalanced (opposite-order pair) and nothing flies on a single-order read.
- **R3 (from B/A): TP9 DEMOTED.** T = +0.20 with an interval including 0; the position-1
  effect is exactly zero.
- **R5: NO warm-cache signature.** Prefix-hit rate +0.006 and mean latency -0.3% between
  stock phases — the prefix cache is NOT the carrier of the order effect. Whatever it is
  (or if it is a 1-in-20 coincidence), resetting the cache would not fix it; only
  counterbalancing does.

**The finding that matters most (exploratory, but it is arithmetic on the data above):**
the per-game paired sd of identical stock vs stock is ~1.2 levels — larger than the mean
itself. A 25-vs-25 same-boot A/B therefore has a minimum detectable effect of roughly
+0.5 lv/game (~+45%). Combined with the live per-draw sd ~0.3 on a mean ~1.7, and ~60 slots
left, **incremental harness levers of the size the ladder has been chasing (+10-30%)
are unmeasurable both locally and live**. Only step changes (>= +0.5 lv/game locally,
>= +1.0 on the LB) can be detected with the instruments and slots that exist. This is a
constraint on WHAT to build, not just on how to measure it.

**Tonight (09-03 00:01Z):** per the tree, the stock v31-copy redraw (runner
`scripts/submit_v31copy_20260903.py`, mock green). Awaiting Ahmed's go; nothing launched.
