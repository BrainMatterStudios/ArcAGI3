# Phase E Increment 2 — Walk-Reduction Frontier Ordering (A/B) — design

Date: 2026-06-21
Branch: phase0-prune-oracle (continues)
Status: design approved in principle (brainstorming); pending written-spec review.

## 1. Why this exists

Phase E Increment 1 (the re-traversal audit) returned a pre-registered KILL of the *general*
efficiency lever: the dominant waste is redundant no-op click-probes (13/16 games ≥30%) which have
no structurally-safe cut, and the safely-cuttable frontier-walk pool is too narrow (only 3/16 games
≥30%: tu93 0.62, ls20 0.46, sp80 0.41). The user elected to build that **narrow, low-EV residual
anyway** — a coverage-preserving frontier-ordering change that cuts walk distance on the avatar wall
games — as the one remaining structurally-safe lever, judged purely by a hard A/B gate.

This is explicitly a fast validate-or-kill experiment, not a confident build. Expected upside is
modest (~8% fewer actions on ~3 games ≈ ~1.16× on affected levels) and the change is non-additive
(it alters the path taken), so it carries real dev≠Kaggle risk.

## 2. Scope

### In scope
- `TourExplorer(SalienceExplorer)` — a subclass with two frontier-ordering modes (`yield`, `dfs`)
  plus a passthrough `v6` mode, selected by a `frontier_mode` ctor param.
- Additive `"tour-yield"` / `"tour-dfs"` policies in `scripts/eval_efficiency.py`'s `make_policy`.
- A/B vs the `salience` baseline on TUNE/HOLDOUT at budgets 6000 AND 30000; a verdict.

### Out of scope
- ANY modification to `SalienceExplorer` (the banked 0.33 agent). The firewall is structural: the
  banked class is literally untouched, so it cannot regress.
- Promoting a winner to the submission default (separate step; requires a real Kaggle submission).
- The redundant-probe pool (killed in Increment 1 — no safe cut).

## 3. Feasibility facts (verified this session)
- `SalienceExplorer._choose` calls `self._path_to_frontier(cur, g)` only when `cur` has no untried
  action ≤ g; it returns an action-path to a frontier (a node with an untried action ≤ p) or `None`.
  Overriding `_path_to_frontier` in a subclass changes *which* frontier is walked to, nothing else.
- `_observe(key, cands, terminal)` is the single creation point for graph nodes → the natural hook
  for discovery-order bookkeeping.
- Frontier selection uses no RNG (BFS order); the only RNG is within-tier action choice, untouched.
- `scripts/eval_efficiency.py` reports per-game levels + actions-to-each-level + a sum-efficiency
  proxy, on the fixed TUNE/HOLDOUT split. Live throughput ~510 act/s → both-budget A/B is minutes.

## 4. Design

### §1 TourExplorer (subclass; SalienceExplorer untouched)
`src/arcagi3/tour_explorer.py`:
```
class TourExplorer(SalienceExplorer):
    __init__(self, *args, frontier_mode="v6", **kwargs):
        super().__init__(*args, **kwargs)
        self.frontier_mode = frontier_mode   # "v6" | "yield" | "dfs"
        self._disc_order = {}                 # key -> first-seen counter (side-channel)
        self._disc_counter = 0
```
- Override `_observe(key, cands, terminal=False)`: call `super()._observe(...)`; if `key` is new to
  `self._disc_order`, record `self._disc_order[key] = self._disc_counter; self._disc_counter += 1`.
  Pure bookkeeping — does not affect any decision.
- Override `_path_to_frontier(start, p)`:
  - `frontier_mode == "v6"` → `return super()._path_to_frontier(start, p)` (byte-identical default).
  - else: if `start` itself has an untried action ≤ p → `return []` (matches parent). BFS from
    `start` over `node.edges` computing the shortest action-path to every reachable node; collect
    every reachable node that `has_untried_le(p)` as a candidate frontier with `(path, depth)`. If
    none → `None`. Select:
    - **`yield`** — among frontiers at the MINIMAL depth, choose the one with the most
      `untried_le(p)` actions; deterministic tie-break: lowest `_disc_order`, then key bytes.
    - **`dfs`** — among ALL frontiers, choose the one with the LARGEST `_disc_order` (most recently
      discovered); tie-break: minimal depth, then key bytes.
  - return the chosen frontier's action-path.

All selections are deterministic (no new RNG), so seeded runs reproduce. Coverage is preserved: every
mode still walks to *some* frontier whenever one exists and the tier loop still escalates only when no
frontier ≤ g remains anywhere — so the set of explored states (hence reachable levels) is identical at
eval scale; only the action-order/path differs.

### §2 A/B gate + pre-registered kill criterion
Extend `scripts/eval_efficiency.py` `make_policy` additively (the `"salience"` branch is byte-identical
to today):
```
"tour-yield" -> TourExplorer(seed=0, trust_threshold=3, border_mask=2, frontier_mode="yield")
"tour-dfs"   -> TourExplorer(seed=0, trust_threshold=3, border_mask=2, frontier_mode="dfs")
```
Run `salience`, `tour-yield`, `tour-dfs` at budget **6000 AND 30000** on TUNE/HOLDOUT.

**Pre-registered ship rule (a priori):** a mode wins iff, at BOTH budgets, it
1. **improves HOLDOUT sum-efficiency** vs `salience`, AND
2. **does not drop HOLDOUT mean-levels** vs `salience`, AND
3. **does not drop TUNE levels** vs `salience` (no-regression on the seen split).

If neither mode satisfies all three at both budgets → **KILL both**; banked 0.33 stays (and the
subclass is kept only as dead, default-unused tooling or reverted, user's call). A winner is NOT
auto-promoted: promotion to the submission default requires a real Kaggle submission confirming lift
(dev≠Kaggle; ordering change is not a structural strict-superset).

### §3 Deliverables
1. `src/arcagi3/tour_explorer.py` — `TourExplorer` (two modes + v6 passthrough).
2. `tests/test_tour_explorer.py` — (a) `frontier_mode="v6"` byte-identical action trace vs
   `SalienceExplorer` on a local game (firewall); (b) `yield`/`dfs` complete the local dev games (no
   coverage loss on the toys); (c) unit tests on a hand-built graph: `yield` picks the max-untried
   nearest frontier, `dfs` picks the most-recently-discovered frontier.
3. Additive `make_policy` modes in `scripts/eval_efficiency.py`.
4. A/B results table (salience vs tour-yield vs tour-dfs, both budgets) + the pre-registered verdict.

Estimated effort: ~half a day.

## 5. Success criteria (process)
A trustworthy A/B that either names a winning mode (per the pre-registered rule, both budgets) or a
clean kill — with the banked 0.33 provably untouched (SalienceExplorer unmodified; v6-passthrough
byte-identical test green).

## 6. Risks & caveats
- **dev ≠ Kaggle / non-additive.** The ordering change is not a structural strict-superset; a
  HOLDOUT win may not transfer. Hence the both-budget gate AND a required live submission before
  promotion. The flag/subclass split means the banked agent is never at risk.
- **Low EV.** Greedy-nearest is already a decent tour heuristic; the realistic upside is small and a
  kill is the likely outcome. This is an explicit, user-chosen long shot.
- **Budget-dependent level drops.** At 6000 a reorder could complete a level later and miss it; the
  rule treats any HOLDOUT level drop at either budget as a fail (conservative — correct, since we
  can't prove eval-scale completion).
- **Subclass plumbing drift.** The `v6`-passthrough must stay byte-identical to the parent; the
  firewall test guards it and must run in the suite.
