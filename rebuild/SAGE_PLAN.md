# T2 PLAN — Cracking the Stuck Games: SAGE (Salience-graph Anchored Goal-directed Engine)

> Produced by the software-factory T2 design phase (orchestrator→tier→team→judge),
> 2026-06-15. **Gate verdict: PASS** (3 Opus judges, SAGE ranked #1 by all; scores 7/8/8.5;
> `combine([REVISE,PASS,PASS]) = PASS`; no security_block). **HALTS here for human approval —
> no code, no submission.** Floor to defend: SalienceExplorer = 33 levels @30k / 174 tests /
> 8 local games (v6 banked-and-staged). Ship bar: ≥34 strict-superset; campaign win ≥36.

---

## 1. Recommendation: SAGE, hardened per the judges' required-changes

SAGE adds the one capability the explorer provably lacks: after a bounce-to-root reset, the
agent re-walks from `root_key` via tier-random selection with **no logic to navigate back to a
known-rewarding configuration** (verified: `reset_all()` runs only in `__init__`, so the graph
persists across in-game resets, but nothing exploits it). SAGE fills exactly that gap and nothing
more.

**Two genuinely-new deltas (everything else is reuse of the existing graph):**
1. **Post-reset reward-replay** — bounded BFS from `root_key` to the nearest node carrying a
   reward-bearing edge, replayed via the *existing* `self.plan`/`self._expect` divergence
   machinery. Plans **only over empirically observed edges** (no parametric model, no avatar
   detector, no motion model, no occupancy map).
2. **Visits-tiebreak in `_path_to_frontier`** — among equal-depth frontier nodes prefer the
   least-visited, so horizon-limited games extend depth instead of re-walking shallow states.

**Grafted from runners-up:** a bounded+memoized BFS (node cap ~2000, `graph_version` key, evict
>512) that also fixes a **pre-existing latent throughput bug** (current `_path_to_frontier` is an
unbounded BFS = 87ms on an 80k-node graph, 14.5× the per-action budget) — flag-gated alongside SAGE
or proven byte-identical; plus the strict-dominance activation gate and two-instance generalization
requirement. **Dropped:** RPF's hot-path tie-break (arms the fragile tu93 win) and MAFO's
macro-chaining (the sokoban-regression class).

## 2. Why it will NOT repeat the prior regressions
- **C6/C7 planner (tu93 3→0):** that planner used a *parametric* predictor that hallucinated
  transitions. SAGE plans only over observed edges — on a no-avatar game there is no chainable
  edge, plan length <2, gate never engages, `decide()` returns the **byte-identical token**. The
  failure class is structurally impossible.
- **Occupancy+A\* (push regress):** SAGE adds no occupancy map / A\*; never gates graph exploration.
- **Tier-aware exhaustion (push regress):** unchanged; SAGE sits in the fallback slot and falls
  through to the unchanged `_choose` on any divergence.
- **Goal-inference (neutral):** SAGE is pure *achievement* — converts an already-banked reward into
  a repeatable navigate-back exploit; no typed goal hypothesis.
- **wm+planner "ties-where-it-engages, regresses-elsewhere" trap:** strict-dominance gate fires
  reward-replay *only* when the reward edge is NOT already current-tier-reachable from `cur`; never
  touches `_choose` steps 1–2 that carry the floor.

**Banked-submission firewall:** default-OFF behind constructor arg AND env flag; committed
notebook byte-identical to v6; OFF-state full gate (33/174/8-of-8, identical per-game breakdown)
asserted before any flag-on measurement.

## 3. Phased build plan (each additive, off-by-default, independently gated)
Gate ladder (fixed order, cheapest first): **G2 tests → G3 local → G1b spot → G1 full.**
- G1 full: `BORDER=2 PYTHONPATH=src uv run python scripts/ab_salience.py salience 30000`
- G1b spot: `... salience 30000 tu93 vc33 ar25 re86 wa30 tr87`
- G2: `PYTHONPATH=src uv run python -m pytest tests/ -q`
- G3 local: `PYTHONPATH=src uv run python -m arcagi3.runner --agent salience --budget 8000 --games-dir src/arcagi3/games` (+ push canary @16000 via test_submission.py)

- **Phase 0 — Baseline lock (no code):** capture golden per-game 33 / 174 / 8-of-8.
- **Phase 1 — Observe-only + redundancy proof (OFF, emits no action):** build the BFS-to-reward,
  visits-tiebreak, and bounded+memoized BFS cache; instrument a counter of the *exact* situations
  SAGE would act on. **Kill-gate:** if that counter is ~0 on tr87 AND ar25, SAGE adds nothing →
  abandon, ship v6. Add GAME_OVER/reset + slide-endpoint unit tests. Gate: byte-identical 33.
- **Phase 2 — Activate reward-replay (high-risk gate):** wire into the fallback slot behind flag
  with strict-dominance; stale-reward eviction on divergence; fail-safe test (malformed grids /
  corrupt graph never raise). Gate: full ladder, push 3/3 first, tu93 ≥9 hard veto, strict-superset.
- **Phase 3 — Activate visits-tiebreak:** re86 watch-only (must-not-regress, NOT a target). Gate:
  full ladder + budget-neutrality on local games.
- **Phase 4 — Generalization decision (no submission):** accept only as a strict superset with net
  ≥+1 AND a lift on two independent instances of one mechanic class (tr87+wa30 OR ar25+sb26).

## 4. Go/No-Go
| Checkpoint | GO | NO-GO (revert) |
|---|---|---|
| Phase 1 redundancy counter | >0 on tr87 AND ar25 | ~0 → abandon, ship v6 |
| Phase 1 OFF-state gate | 33/174/8-8 byte-identical | any drift → fix or abandon |
| BFS-cache diff test | identical paths on 25 transcripts | any divergence → flag-gate harder/drop |
| Phase 2/3 strict-superset | G1 ≥33, no game→0, tu93≥9 | any prev-nonzero game→0, tu93<9, or ≤32 |
| Phase 4 campaign | ≥34 + two-instance lift | parity-with-any-regression → revert |

Pre-committed kill: not a strict superset with net ≥+1 after the effort box → revert like the prior
six levers (a tie at 33 that regresses any previously-nonzero game is an automatic revert).

## 5. Generalization safeguards
Zero fitted surface (no learned param/table/CNN); no game IDs/colors/coords/mechanic detectors;
two-instance requirement (single-game lift treated as overfit, not shipped); dormant where there is
no reward evidence (byte-identical, cannot hallucinate on unseen games).

## 6. Honest EV
Moderate-to-low. Six consecutive post-breakthrough levers were neutral-or-regressing; the real wins
were representation fixes (now exhausted). SAGE's novelty is narrow but real and code-confirmed.
**Realistic primary targets:** tr87 + ar25 (reward already banked → replay can fire), validated by
wa30/sb26. **Out of scope:** re86 (empty reward memory at 0 levels → frontier nudge only), click
puzzles (need visual reasoning). Expected: real-but-small chance of +1 to +3; substantial chance of
another documented neutral revert caught cheaply by the Phase 1 counter. Downside bounded by the
no-regression gate — worst case we re-confirm 33 is the ceiling and ship v6, cheaply.

---
**This is a plan only. Awaiting human go-ahead before Phase 0.**
