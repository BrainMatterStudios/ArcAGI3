# Click kill-experiments: no-op predictor + affordance-language oracle

Date: 2026-07-22. All measurement, no agent built. Offline engine only (`environment_files/`),
project venv, zero GPU, zero paid API.

Scripts: `scratchpad/killexp_harvest.py` (exhaustive sweep), `scratchpad/killexp_analyze.py`
(both experiments). Raw ground truth: `scratchpad/killexp_data/*.npz` + `*.json`.

---

## VERDICTS

### EXPERIMENT 1 — no-op click predictor: **KILL** (as a learned model)

**The number that decides it: a model-free probe of one click per connected component
reconstructs the entire 4096-cell affordance map at a median 99.95% cell accuracy and
median 100% live-affordance recall, in a median of 34 clicks.**

The learned predictor is real but *dominated*. It never gets to be the cheapest thing
that works:

- **Within-game it works** but needs far more labels than the thing it replaces. On the
  honest component-grouped split it prunes a median **89.4%** of the click space at zero
  affordance loss — using ~2000 labelled clicks. The 34-click model-free probe gets a
  strictly better map for 1/60th of the budget. There is no regime where you would run
  the forest instead.
- **Cross-game it fails**, as our history predicted. Trained on 13 games, tested on the
  held-out 14th, it prunes a median of only **37.1%** at zero loss, and **0.0%** on
  cn04/tn36 and 1.6% on s5i5. It cannot ship pre-trained.

**Note the stated kill rule is uninformative and should not be used.** "Precision ≥95% at
50% recall" is passed trivially: 12/14 games have a no-op base rate above 95%, so the
constant classifier "everything is a no-op" already clears the bar. Worse, 99% precision
over ~3900 pruned cells permits discarding ~39 live cells — and several games only *have*
~25–40 live cells, so a "99% precise" pruner can legitimately delete every affordance in
the game. Every number below is therefore also reported as **prune fraction vs.
live-affordance loss**, which is the decision-relevant framing.

### EXPERIMENT 2 — is the affordance language expressive enough: **KILL**

**The number that decides it: the true affordance set is exactly captured by a predicate
in the language for 2 of 14 games (14%); allowing approximation at F1 ≥ 0.90 gets 6 of 14
(43%). The bar was ~75% (6 of 8). It fails by a wide margin.**

Failure mode is not a missing predicate — it is that **live and dead objects are frequently
pixel-identical**. In ft09 the affordance set is 8 connected components of colour 9, size
36 each; the 12 *dead* components are also colour 9, size 36. No predicate over colour,
size, rarity, centroid-ness or avatar-distance can separate them, at any conjunction depth.
The distinguisher is configurational (which slot in a layout, relation to a target pattern),
which this language cannot express. Per the kill rule I did not run the information-gain
probing simulation — the language dies first.

---

## Setup and what was actually measured

The offline engine turned out to be fast enough (~0.1 ms per reset+click) that **sampling
was unnecessary**: I swept **all 4096 cells exhaustively on all 25 games** — complete ground
truth, not an estimate. Total runtime **65 seconds**. No run was capped.

Protocol per cell: `env.reset()` → single `ACTION6(x,y)` → compare the settled frame to the
level-0 base frame. Reset-before-every-click makes the 4096 probes independent.

**Determinism verified, not assumed:** 200 random cells re-probed on ft09/bp35/sb26/su15/
tn36/r11l gave **200/200 identical outcomes** on all 6 games. The "one observation is ground
truth forever" premise holds.

**HUD masking is a prerequisite, not a detail.** Cells changing on ≥90% of all 4096 clicks
are treated as step counters and masked. Games like tn36 and s5i5 have exactly **one** such
cell — and without masking it, **100% of clicks look effective** (4096/4096). A single
ticking counter pixel is enough to destroy all no-op signal. Any deployed pruner must mask
first.

### Which games are actually click games (verified, not assumed)

19 of 25 games offer ACTION6 — this matches the brief. But the useful set is smaller:

- **6 games do not offer ACTION6 at all**: g50t, ls20, re86, tr87, tu93, wa30.
  *`ls20` was in the brief's candidate list and is not a click game.*
- **5 games offer ACTION6 but have ZERO live clicks anywhere in the 4096-cell space at the
  level-0 start state**: ar25, m0r0, sc25, sk48, sp80.
  *`ar25` and `sk48` were in the brief's candidate list; both are 0/4096.*
- **14 games have live clicks** and are used throughout below: bp35, cd82, cn04, dc22, ft09,
  ka59, lf52, lp85, r11l, s5i5, sb26, su15, tn36, vc33.

That second bullet is a result in its own right: on 5 games, clicking at the start state is a
4096-wide branch with a guaranteed payoff of exactly zero.

---

## (b) Base rates — the early-exit check

Clicks are overwhelmingly dead, so the premise survives: **median no-op rate 96.41%** across
the 14 live-click games. Effect classes (`local` = all changed cells within Chebyshev 3 of
the click; `global` = otherwise):

| game | avail actions | no-op % | no-op | local | global | level-up |
|---|---|---|---|---|---|---|
| ar25 | 1,2,3,4,5,6,7 | 100.00 | 4096 | 0 | 0 | 0 |
| bp35 | 3,4,6,7 | 93.85 | 3844 | 36 | 216 | 0 |
| cd82 | 1,2,3,4,5,6 | 99.39 | 4071 | 0 | 25 | 0 |
| cn04 | 1,2,3,4,5,6 | 92.31 | 3781 | 0 | 315 | 0 |
| dc22 | 1,2,3,4,6 | 97.71 | 4002 | 0 | 94 | 0 |
| ft09 | 6 | 92.97 | 3808 | 0 | 288 | 0 |
| g50t | 1,2,3,4,5 | 100.00 | 4096 | 0 | 0 | 0 |
| ka59 | 1,2,3,4,6 | 99.78 | 4087 | 0 | 9 | 0 |
| lf52 | 1,2,3,4,6,7 | 98.24 | 4024 | 0 | 72 | 0 |
| lp85 | 6 | 97.66 | 4000 | 0 | 96 | 0 |
| ls20 | 1,2,3,4 | 100.00 | 4096 | 0 | 0 | 0 |
| m0r0 | 1,2,3,4,5,6 | 100.00 | 4096 | 0 | 0 | 0 |
| r11l | 6 | 23.85 | 977 | 0 | 3119 | 0 |
| re86 | 1,2,3,4,5 | 100.00 | 4096 | 0 | 0 | 0 |
| s5i5 | 6 | 95.90 | 3928 | 0 | 168 | 0 |
| sb26 | 5,6,7 | 96.48 | 3952 | 16 | 128 | 0 |
| sc25 | 1,2,3,4,6 | 100.00 | 4096 | 0 | 0 | 0 |
| sk48 | 1,2,3,4,6,7 | 100.00 | 4096 | 0 | 0 | 0 |
| sp80 | 1,2,3,4,5,6 | 100.00 | 4096 | 0 | 0 | 0 |
| su15 | 6,7 | 17.19 | 704 | 12 | 3380 | 0 |
| tn36 | 6 | 96.34 | 3946 | 150 | 0 | 0 |
| tr87 | 1,2,3,4 | 100.00 | 4096 | 0 | 0 | 0 |
| tu93 | 1,2,3,4 | 100.00 | 4096 | 0 | 0 | 0 |
| vc33 | 6 | 99.22 | 4064 | 0 | 32 | 0 |
| wa30 | 1,2,3,4,5 | 100.00 | 4096 | 0 | 0 | 0 |

Two outliers, r11l (23.9%) and su15 (17.2%), are click-canvas games where most of the board
responds. **Zero single clicks caused a level-up in any of the 25 games** — no level is one
click deep.

---

## (c–d) Experiment 1: the predictor

Model: `RandomForestClassifier(n_estimators=200, min_samples_leaf=2)`. No neural nets, no GPU.
Features: clicked colour; k=3 and k=5 colour neighbourhoods; connected-component size, width,
height; colour-rarity rank; is-centroid; distinct colours in the k=5 patch; offset from
component centroid; Chebyshev distance to the presumed avatar (found by tracking cells that
move under ACTION1-4). k=3 and k=5 scored within noise of each other; k=5 reported.

### The split matters enormously — read this before the tables

A random 60/40 **cell** split (as briefed) gives a **perfect** result: P@50%recall = 1.000 and
R@99%precision = 1.000 on 14/14 games, with or without positional features. **This is not a
real result.** Cells of one object are near-duplicates, so a random split trains on some cells
of an object and tests on its siblings. That measures interpolation *inside an already-probed
object* — which requires no model at all, just a flood fill.

The honest test is a **component-grouped split**: train on 60% of connected components, test
on entirely unprobed ones. This measures the only thing a model could add — predicting whether
an object you have *never clicked* is live. All results below use it, plus positional features
(`x`, `y`, border distance) dropped so the forest cannot memorise regions.

### Within-game, component-grouped — prune fraction at a given live-affordance loss

| game | live cells | test cells | prune @ 0% lost | @≤1% | @≤5% | @≤10% |
|---|---|---|---|---|---|---|
| bp35 | 53 | 460 | 0.885 | 0.885 | 0.889 | 0.896 |
| cd82 | 16 | 345 | 0.128 | 0.128 | 0.128 | 0.614 |
| cn04 | 153 | 3902 | 0.949 | 0.950 | 0.952 | 0.955 |
| dc22 | 47 | 3802 | 0.975 | 0.975 | 0.976 | 0.976 |
| ft09 | 144 | 2668 | 0.942 | 0.942 | 0.948 | 0.951 |
| ka59 | 9 | — | degenerate: no live component landed in the train split | | | |
| lf52 | 54 | 3548 | 0.960 | 0.960 | 0.965 | 0.969 |
| lp85 | 40 | 1015 | 0.867 | 0.867 | 0.891 | 0.897 |
| r11l | 61 | 689 | 0.904 | 0.904 | 0.909 | 0.913 |
| s5i5 | 104 | 3916 | 0.842 | 0.842 | 0.867 | 0.868 |
| sb26 | 96 | 3632 | 0.239 | 0.239 | 0.900 | 0.904 |
| su15 | 81 | 712 | 0.000 | 0.000 | 0.007 | 0.813 |
| tn36 | 15 | 537 | 0.972 | 0.972 | 0.972 | 0.974 |
| vc33 | 32 | — | degenerate: no live component landed in the train split | | | |
| **MEDIAN** | | | **0.894** | 0.894 | 0.904 | 0.909 |

Same runs in the briefed precision/recall framing (predicting the no-op class):

| game | base rate | P@50%rec | P@80%rec | R@99%prec | R@99.9%prec |
|---|---|---|---|---|---|
| bp35 | 0.8848 | 1.0000 | 1.0000 | 1.000 | 1.000 |
| cd82 | 0.9536 | 0.9953 | 0.9854 | 0.790 | 0.134 |
| cn04 | 0.9608 | 1.0000 | 1.0000 | 0.990 | 0.988 |
| dc22 | 0.9876 | 1.0000 | 1.0000 | 0.997 | 0.987 |
| ft09 | 0.9460 | 1.0000 | 1.0000 | 1.000 | 0.998 |
| ka59 | 0.9975 | degenerate | | 0.000 | 0.000 |
| lf52 | 0.9848 | 1.0000 | 1.0000 | 0.995 | 0.980 |
| lp85 | 0.9606 | 1.0000 | 1.0000 | 0.934 | 0.903 |
| r11l | 0.9115 | 1.0000 | 1.0000 | 0.992 | 0.992 |
| s5i5 | 0.9734 | 1.0000 | 1.0000 | 0.914 | 0.889 |
| sb26 | 0.9736 | 0.9997 | 0.9997 | 0.950 | 0.923 |
| su15 | 0.8862 | 0.9903 | 0.9903 | 0.811 | 0.000 |
| tn36 | 0.9721 | 1.0000 | 1.0000 | 1.000 | 1.000 |
| vc33 | 0.9322 | degenerate | | 0.000 | 0.000 |

P@50%recall ≥ 0.99 on all 12 non-degenerate games — passing the briefed kill rule, but as
noted, so does the constant classifier. The prune-vs-loss table is the one to trust.

### (e) Cross-game transfer — train on 13, test on the held-out 14th

| game | live | prune @ 0% lost | @≤1% | @≤5% | @≤10% | P@50%rec | R@99%prec |
|---|---|---|---|---|---|---|---|
| bp35 | 252 | 0.369 | 0.485 | 0.624 | 0.814 | 0.9990 | 0.929 |
| cd82 | 25 | 0.012 | 0.012 | 0.013 | 0.013 | 0.9939 | 1.000 |
| cn04 | 315 | **0.000** | 0.001 | 0.004 | 0.008 | 0.9584 | 0.000 |
| dc22 | 94 | 0.916 | 0.916 | 0.925 | 0.939 | 1.0000 | 0.967 |
| ft09 | 288 | 0.638 | 0.639 | 0.655 | 0.669 | 1.0000 | 0.713 |
| ka59 | 9 | 0.990 | 0.990 | 0.990 | 0.990 | 1.0000 | 1.000 |
| lf52 | 72 | 0.661 | 0.661 | 0.667 | 0.696 | 1.0000 | 0.988 |
| lp85 | 96 | 0.780 | 0.780 | 0.791 | 0.806 | 1.0000 | 0.868 |
| r11l | 3119 | 0.148 | 0.156 | 0.211 | 0.249 | 1.0000 | 0.622 |
| s5i5 | 168 | 0.016 | 0.016 | 0.018 | 0.020 | 0.9590 | 0.017 |
| sb26 | 144 | 0.373 | 0.377 | 0.468 | 0.763 | 0.9970 | 0.837 |
| su15 | 3392 | 0.105 | 0.164 | 0.199 | 0.241 | 1.0000 | 0.751 |
| tn36 | 150 | **0.000** | 0.000 | 0.002 | 0.005 | 0.9634 | 0.000 |
| vc33 | 32 | 0.974 | 0.974 | 0.975 | 0.977 | 1.0000 | 1.000 |
| **MEDIAN** | | **0.371** | 0.431 | 0.546 | 0.683 | | |

For cn04, s5i5 and tn36 the transferred model's P@50%recall (0.958/0.959/0.963) is *at* their
base rate — it has learned nothing transferable at all. **Verdict: must be learned per-game
online; do not ship pre-trained.** This is the sixth independent confirmation of the
cross-game-transfer failure already in our notes.

### The baseline that kills it: one probe per connected component

No model. Segment the masked frame into 4-connected same-colour components, click one random
cell of each, and label the whole component by that outcome.

| game | components (= clicks) | live cells | cell accuracy | live recall |
|---|---|---|---|---|
| bp35 | 191 | 252 | 0.9812 | 0.694 |
| cd82 | 15 | 25 | 1.0000 | 1.000 |
| cn04 | 9 | 315 | 1.0000 | 1.000 |
| dc22 | 40 | 94 | 1.0000 | 1.000 |
| ft09 | 65 | 288 | 1.0000 | 1.000 |
| ka59 | 14 | 9 | 1.0000 | 1.000 |
| lf52 | 59 | 72 | 0.9902 | 0.444 |
| lp85 | 40 | 96 | 0.9961 | 0.833 |
| r11l | 38 | 3119 | 0.9207 | 1.000 |
| s5i5 | 30 | 168 | 0.9990 | 1.000 |
| sb26 | 26 | 144 | 0.9805 | 0.444 |
| su15 | 28 | 3392 | 1.0000 | 1.000 |
| tn36 | 88 | 150 | 0.9707 | 0.200 |
| vc33 | 11 | 32 | 1.0000 | 1.000 |
| **MEDIAN** | **34** | | **0.9995** | **1.000** |

**34 clicks replaces 4096.** A ~120x branching-factor reduction, exact on 6/14 games, with no
training, no model, no transfer assumption and no risk of silently pruning an affordance you
never observed.

Its only failure mode is **partially-live components** — a large region where only a sub-part
responds (bp35, lf52, sb26, tn36 have 1–2 each). An adaptive variant (probe 1 cell; for
components larger than 8 cells probe 5; if the 5 disagree, probe that component exhaustively)
lifts the median to **100% cell accuracy and 100% live recall in a median 99 clicks**, perfect
on 10/14 games, but blows up to 3446 clicks on r11l and 744 on tn36 where live components are
genuinely heterogeneous. Both variants remain far cheaper than 4096 and far cheaper than the
~2000 labels the forest needs.

---

## Experiment 2: the hypothesis language

Language as briefed, as boolean masks over the grid: `colour == c`; `colour-rarity rank <= r`;
`comp_size == s` / `<= s`; `is-centroid-of-its-component`; Chebyshev distance to avatar `<= 1`
and `<= 3`; same row / same column as avatar; non-background; row-contains-object;
column-contains-object — plus all depth-2 conjunctions, and (going *beyond* the brief to be
fair to the idea) all depth-2 disjunctions and greedy 3-term DNF.

Ground truth = the exhaustive sweep, so this is an exact oracle test, not an estimate.

| game | true live cells | best AND (F1) | best OR (F1) | 3-term DNF (F1) | verdict | best rule found |
|---|---|---|---|---|---|---|
| dc22 | 94 | **1.000** | 0.979 | 1.000 | **EXACT** | `comp_size==47` |
| vc33 | 32 | **1.000** | 0.889 | 1.000 | **EXACT** | `color==9` |
| su15 | 3392 | 0.985 | 0.998 | 0.999 | approx | `color==5 OR rarity_rank<=2 OR comp_size==5` |
| cn04 | 315 | 0.952 | 0.952 | 0.952 | approx | `rarity_rank<=3` |
| r11l | 3119 | 0.938 | 0.948 | 0.948 | approx | `color==5 OR rarity_rank<=4` |
| lp85 | 96 | 0.909 | 0.909 | 0.909 | approx | `comp_size==40` |
| s5i5 | 168 | 0.868 | 0.897 | 0.868 | FAIL | `rarity_rank<=4` |
| bp35 | 252 | 0.737 | 0.737 | 0.737 | FAIL | `color==14` |
| ka59 | 9 | 0.667 | 0.692 | 0.667 | FAIL | `rarity_rank<=2` |
| cd82 | 25 | 0.625 | 0.667 | 0.625 | FAIL | `comp_size<=16` |
| sb26 | 144 | 0.615 | 0.456 | 0.471 | FAIL | `rarity_rank<=4 AND comp_size==16` |
| ft09 | 288 | 0.571 | 0.565 | 0.571 | FAIL | `color==9` |
| tn36 | 150 | 0.477 | 0.563 | 0.563 | FAIL | `color==0 OR comp_size==3` |
| lf52 | 72 | 0.364 | 0.421 | 0.421 | FAIL | `color==14 OR comp_size==1` |

**Exact: 2/14 (14%). Approximate at F1 ≥ 0.90: 6/14 (43%). Bar was ~75%. KILL.**

Adding disjunction and a third DNF term bought almost nothing (median F1 gain +0.005) — the
language is not short a connective, it is short a *concept*.

### What the language is actually missing

Per-colour breakdown of live vs dead components on the failing games:

```
ft09  color 9 : 20 components,  8 live.  live sizes=[36]*8    dead sizes=[36]*12
bp35  color 3 : 178 components, 28 live. live sizes=[1]*28    dead sizes=[1]*150
lf52  color 1 : 48 components,  8 live.  live sizes=[1]*8     dead sizes=[1]*40
      color 14: 5 components,   2 live.  live sizes=[12,12]   dead sizes=[12,12,12]
sb26  color 9 : 2 components,   1 live.  live size=[16]       dead size=[20]
tn36  color 5 : 41 components,  4 live.  live sizes=[3]*4     dead sizes=[14,15,15,16,...]
```

In ft09, bp35 and lf52 the live and dead objects are **identical in colour and in size**.
There is no appearance-based feature — at any conjunction depth, with any connective — that
separates them. What distinguishes a live ft09 blob from a dead one is its *role in the
configuration* (which slot it occupies relative to a target pattern), which requires
relational/positional predicates the language does not contain and which would need the
target pattern itself to be parsed first.

A version-space prober cannot converge on a hypothesis that is not in its space. Adding the
missing predicates is not a small extension: it means inferring the puzzle's layout semantics,
which is the original problem.

---

## Threats to validity — read before acting

1. **Level-0 start state only.** Everything above is a single state per game. I ran a partial
   depth check (sweep the full 4096 again after 1 and after 3 live clicks) on 6 games. It is
   interpretable for ft09, sb26 and lp85 (whose deeper states stay HUD-clean), and the
   component-probe accuracy is **depth-stable** there: ft09 1.000/1.000/1.000 at depths 0/1/3,
   lp85 0.996/0.996/0.996, sb26 0.981/0.970/0.962. For tn36, s5i5 and cn04 the deeper sweep
   was uninterpretable because an animation starts and every click registers as a change
   without HUD masking. So: encouraging on 3 games, **unverified in general**. Re-verify at
   depth before shipping anything.
2. **Component-homogeneity is the load-bearing assumption**, and it is not universal —
   partially-live components exist in 7 of 14 games and are exactly where the cheap probe
   loses recall (tn36 0.200, lf52 0.444, sb26 0.444).
3. `scikit-learn 1.9.0` was installed into `.venv` for this experiment (it was absent).
4. The avatar-distance feature comes from a heuristic mover-detection pass and is `-1` on the
   click-only games (no move actions to detect motion with).

---

## What surprised me

- **The whole thing is exhaustively solvable offline in 65 seconds.** I planned to sample a
  few hundred clicks per game; the offline engine does reset+click in ~0.1 ms, so I swept all
  4096 cells on all 25 games and got *complete* ground truth instead of an estimate. This
  ground truth is banked in `scratchpad/killexp_data/` and cost nothing. I did not expect the
  measurement problem to simply evaporate.
- **The briefed experimental design would have produced a false positive.** A random 60/40
  cell split yields a flawless 1.000/1.000 on every game, which reads as a slam-dunk BUILD.
  It is an artifact of sibling cells within one object. The component-grouped split is what
  turns it into an honest, and much less exciting, result. Given our history of green results
  that died at eval, this one nearly joined them.
- **The model is real but arrives too late to matter.** I expected either "the predictor works"
  or "it doesn't". The actual answer is "it works and is irrelevant" — a 34-click flood-fill
  probe beats it. Segmentation, not learning, is where the branching-factor win lives.
- **5 of the 19 ACTION6 games have zero live clicks at the start state** (ar25, m0r0, sc25,
  sk48, sp80). Any breadth search on those games burns a 4096-wide branch for a guaranteed
  zero return. Detecting this costs 4096 clicks ≈ 2 seconds offline, or ~34 with the component
  probe.
- **Three of the brief's eight candidate click games were wrong** (ls20 has no ACTION6; ar25
  and sk48 are 0/4096). Worth re-verifying game lists against the engine.
- **One ticking HUD pixel destroys 100% of the no-op signal.** tn36 and s5i5 each have exactly
  one cell that changes on every click; unmasked, all 4096 clicks look effective. Volatility
  masking is not a refinement, it is a precondition.
- **No single click completes a level in any of the 25 games** — zero level-ups across 102,400
  probes.
- **Identical objects with different affordances** (ft09's 8 live vs 12 dead identical colour-9
  blobs) is the cleanest evidence I have seen for why appearance-based affordance priors keep
  failing here. This is a specific, mechanistic explanation for our repeated cross-game
  transfer failures, not just another negative result.
