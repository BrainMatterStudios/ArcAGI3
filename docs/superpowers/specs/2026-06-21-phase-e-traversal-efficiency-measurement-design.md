# Phase E — Traversal-Efficiency Measurement (Increment 1) — design

Date: 2026-06-21
Branch: phase0-prune-oracle (continues; rename/PR later)
Status: design approved in principle (brainstorming); pending written-spec review.

## 1. Why this exists

The Phase 0a prune probe robustly killed the "learn a per-state signal to skip exploration"
family (on-path vs off-path states are observationally indistinguishable at decision time;
best AUC 0.45–0.54 vs a 0.65 bar). But the same trajectory tooling exposed a different, signal-
free opportunity. On tu93 (9 levels, 24,193 actions), the action budget breaks down as:

- **frontier-walks ≈ 62%** — walking known states to reach the next untried-action frontier
  (2,042 walks, mean 7.4 / median 3 / max 26 steps);
- **redundant probes ≈ 24%** — `fresh_local_test`s whose result was an already-known state
  (8,404 local tests produced only 2,649 new states);
- discovery ≈ 11%; reset-bounces ≈ 0.2% (already well-handled).

The squared-efficiency metric `S = min(1.15, (human/agent_actions)^2)` punishes every wasted
action, so cutting re-traversal is the highest-leverage lever that needs **no learned signal**
(which we have proven, repeatedly, we cannot obtain offline). The standing memory record
declared the efficiency lever "closed" by *diagnosis* ("9 act/state is healthy coverage") but
never *tested* a re-traversal cut; that "healthy" number is exactly the 86% re-traversal /
SOTA-action-efficiency gap, re-interpreted.

This increment does **not** change the agent. It measures, across all 16 tune+holdout games,
**where the re-traversal concentrates and whether it is safely cuttable**, so the eventual cut
(a follow-on spec) targets the biggest safe pool with evidence — the same measure-then-build
gate that made Phase 0a cheap.

## 2. Scope

### In scope
- `InstrumentedExplorer` — a read-only subclass of the banked `SalienceExplorer` that produces
  **byte-identical decisions** (same seed → same action trace) but tags which `_choose` branch
  fired and accumulates re-traversal statistics.
- `scripts/retraversal_audit.py` — runs it across the 16 tune+holdout games and reports per-game
  and aggregate anatomy.
- A results table + a memory note recording the Increment-2 target (or a kill).

### Out of scope (decided 2026-06-21)
- The actual traversal cut (becomes a follow-on spec once the data names the target).
- ANY change to `SalienceExplorer`'s behavior or the banked v6 submission.
- Holdout is *measured* here for anatomy only; the ship gate happens in Increment 2.

## 3. Feasibility facts (verified 2026-06-21)
- Live throughput ~510 actions/sec; 16 games × ~40k ≈ 10–15 min total.
- `SalienceExplorer._choose` (src/arcagi3/salience_explorer.py:235) has exactly these branches:
  reward-exploit → plan-replay (with `cur != self._expect` invalidation) → hierarchical tier
  (`fresh_local_test` / `new_plan_to_frontier`, escalating `active_group`) → reset-bounce →
  random. The subclass copies this control flow verbatim and tags each return.
- tune/holdout split (from `scripts/eval_efficiency.py`):
  TUNE = vc33, cd82, sc25, lp85, lf52, tu93, ar25, sp80;
  HOLDOUT = su15, sk48, re86, wa30, m0r0, ls20, tn36, tr87.

## 4. Design

### §1 InstrumentedExplorer (behavior-identical, tagged)
Subclass `SalienceExplorer`; override `_choose` with a **verbatim copy** of the parent's logic,
adding only: a `Counter` of branch tags (`exploit`, `plan_replay_walk`, `fresh_local_test`,
`new_plan_to_frontier`, `plan_invalidation`, `reset_bounce`, `random_*`), a record of each new
plan's length, and a `seen` set to mark whether a `fresh_local_test`'s result is a known state
(classified post-hoc from the trace, since the result is only observed on the next `decide`).

**Firewall for the tool itself:** a test asserts the InstrumentedExplorer's action trace is
byte-identical to `SalienceExplorer`'s on the local dev games and on one real game at a fixed
seed. If the copy ever drifts, the test fails — the measurement must reflect the banked agent,
not a variant.

### §2 retraversal_audit.py
For each of the 16 games: run `InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2)`
at a fixed budget (default 40000), capture the trace (reusing the Phase-0a capture pattern), and
compute per game:
- decision-branch percentages (discovery / frontier-walk / redundant-probe / plan-invalidation /
  reset-bounce);
- frontier-walk stats: count, total steps, mean/median/max length, and the share of total
  actions spent walking;
- redundant-probe rate = (local tests whose result was already-seen) / (all local tests);
- plan-invalidation count and the wasted partial-walk steps they discarded.
Aggregate across games (action-weighted), and print a ranked table of cuttable pools.

### §3 Decision rule for Increment 2 (the output)
**Pre-registered proceed/kill threshold (chosen a priori):** Increment 2 proceeds on a pool only
if that pool accounts for **≥30% of total actions on ≥10 of the 16 games** AND has a structurally-
safe (coverage-preserving or model-free-detectable) cut. Otherwise → **clean KILL** of the
efficiency lever. This keeps the build/kill call from being a post-hoc read of the table.

Target the pool that is **largest × most uniform across games × structurally safe to cut**:
- **Frontier-walks** are safe to cut by *ordering* changes (tour ordering, plan-invalidation
  reduction) that don't drop coverage — candidate if walks are far from a minimal tour.
- **Redundant probes** need cheap *deterministic-structure* prediction (no-op/reversal detection)
  to cut safely — candidate only if a large, uniform, model-free-detectable share exists.
- **Plan-invalidations** (re-walks from discarded plans) are safe to cut if frequent.

If no pool is both large *and* safely cuttable (walks near a minimal tour; probes need a learned
model; invalidations rare), that is a **clean KILL** of the efficiency lever — recorded, no agent
code touched. (User-confirmed branch, 2026-06-21.)

### §4 Firewall & gate principles for the eventual cut (Increment 2, stated now)
- The cut ships behind a **default-off ctor param**; the banked v6 config
  (`SalienceExplorer(trust_threshold=3, border_mask=2)`) stays byte-identical (factory_checks).
- **Honest risk:** a traversal-*ordering* change is **not** a structural strict-superset (unlike
  the additive `coarse_grid_step` fix) — it changes the path taken, so it could complete fewer
  levels on a hidden game even when dev improves. Safety therefore rests on (a) HOLDOUT levels
  must not drop AND HOLDOUT efficiency improves (`scripts/eval_efficiency.py`), and (b) a real
  Kaggle submission to confirm before promoting to default. dev ≠ Kaggle risk is higher here than
  for the dense-lattice fix. The flag stays so we can always revert to 0.33.

## 5. Deliverables
1. `src/arcagi3/instrumented_explorer.py` — `InstrumentedExplorer` (behavior-identical, tagged).
2. `tests/test_instrumented_explorer.py` — byte-identical-trace test + tag-accounting tests.
3. `scripts/retraversal_audit.py` — the 16-game audit driver.
4. A per-game + aggregate results table; a memory note recording the Increment-2 target or kill.

Estimated effort: ~half a day (measurement only; no agent behavior change).

## 6. Success criteria (process, not outcome)
A trustworthy, behavior-identical-verified anatomy of re-traversal across all 16 games that names
**one** Increment-2 target — or a defensible kill. A kill is a success (it avoids building the
wrong cut). The only failure mode is an instrumented agent that diverges from the banked one
(caught by the byte-identical test) or a measurement that misattributes the waste.

## 7. Risks & caveats
- **Instrumentation drift.** The verbatim `_choose` copy could diverge from the parent on future
  edits → the byte-identical-trace test is mandatory and must run in CI/the suite.
- **Redundant-probe irreducibility.** A probe is only "redundant" in hindsight; cutting it safely
  needs model-free structural prediction. The audit measures the *detectable* share, not the
  ideal — don't over-claim cuttability.
- **dev ≠ Kaggle.** This increment informs a target; the Increment-2 cut's real safety needs the
  holdout gate + a live submission. The 0.33 floor is never regressed (this increment changes no
  agent behavior; the cut is flag-gated).
- **Optimistic walk-reduction.** "Better tour ordering" may only shave the long-walk tail
  modestly; the audit's walk-length distribution must justify the expected saving before building.

## Results (2026-06-21) → KILL (pre-registered)

Run: `scripts/retraversal_audit.py 40000` over the 16 tune+holdout games (~1032 s).
Raw: `/tmp/retraversal_audit.json`, `/tmp/retraversal_audit.log`.

### Anatomy is sharply game-dependent (not one uniform pattern)

| pool | action-weighted mean | games ≥30% |
|---|---|---|
| `pct_walk` (frontier-walks) | 0.155 | **3 / 16** |
| `pct_redundant_probe` (no-op/known-state probes) | 0.510 | **13 / 16** |

- **Frontier-walks** concentrate in avatar games: tu93 0.62, ls20 0.46, sp80 0.41 (mean walk-len
  tu93 7.4 / ls20 13.9). Only 3/16 ≥30%.
- **Redundant probes** dominate click games: lp85 0.98, su15 0.88, sc25 0.77, vc33 0.77, m0r0 0.51,
  wa30 0.48 — 13/16 ≥30%, and this is a *conservative under-estimate* (nodes also created by
  `_observe`, so true redundancy is higher).

### Verdict: KILL the general efficiency lever (pre-registered ≥30%-on-≥10/16 + structurally-safe cut)
- **Redundant-probe pool** passes breadth (13/16) but has **no structurally-safe cut.** It is
  dominated by tier-9 lattice **no-op clicks**; distinguishing a no-op from a productive click
  *before* testing needs a transition model (killed: learned models regress) or a content heuristic
  — which is the **already-killed lean/high-impact-click lever** (tied random in breadth; lean
  targeting drops the off-object goal clicks the dense `coarse_grid_step` lattice was *added* to
  catch — the tn36 no-free-lunch). Cutting it is not coverage-preserving.
- **Walk pool** has a coverage-preserving (ordering) cut but **fails breadth** (3/16 ≥30%) — it
  would help only a few avatar games.

So the big pool is irreducible and the safely-cuttable pool is too narrow → **clean KILL** (the
user-pre-committed branch). The measure-first gate did its job: it stopped us building the intuitive
walk-reduction cut, which the data shows helps only ~3 games, and re-confirmed — from the efficiency
angle — the documented no-free-lunch wall (the real cost is redundant click-probes that can't be
safely pruned without re-treading killed levers).

### Honest residual (low-EV, not pursued without direction)
The one safe-but-narrow option is a **walk-reduction / better frontier-tour ordering on the avatar
wall games** (tu93 62%, ls20 46%, sp80 41%; action-weighted walks = 15.5% of all actions). Halving
walks ≈ 8% fewer total actions ≈ ~1.16× on affected levels — modest, narrow, and a non-additive
ordering change (dev≠Kaggle risk, needs the holdout gate + a live submission). Below the
pre-registered bar; recorded as a residual, not a recommendation.

**Outcome:** a successful Increment 1 — the audit named no safely-cuttable general lever and avoided
building the wrong cut. Banked 0.33 untouched (no agent behavior changed; suite 190 green). Tooling
kept: `InstrumentedExplorer` + `scripts/retraversal_audit.py`. Path to >0.33 remains the June-30
SOTA disclosure + hardening.
