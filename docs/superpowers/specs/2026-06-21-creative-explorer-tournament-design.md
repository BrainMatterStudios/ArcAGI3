# Creative Explorer Tournament — Design

**Date:** 2026-06-21
**Branch:** phase0-prune-oracle
**Goal:** Beat the banked v6 = 0.33 Kaggle score with a *genuinely new* lever, not a re-run of
the prior-session kill graveyard.

## Scoring correction (the premise)

Prior sessions optimized against a wrong model (`game = min(levels, efficiency)`). The official
methodology (docs.arcprize.org/methodology) is:

- **per level:** `S_L = min(1.15, (human_actions / agent_actions)^2)`; uncompleted level = 0.
- **per game:** weighted average of `S_L`, weight = the 1-indexed level number. Later levels
  dominate. Uncompleted levels sit in the denominator as zeros.
- **total:** average of game scores over the **hidden** games.

Implication: the highest-value target is **reaching DEEP levels efficiently on unseen games**.
The one dense, *correct* signal nobody exploited is the **reward edge from a level you already
solved**, reused on the same game's later levels (escalating mechanics share structure). Every
killed lever failed for one of three reasons — learned from too-sparse reward, steered by
non-goal signals (frame-change/counters), or reordered with no signal. Within-game cross-level
transfer dodges all three.

## Shared backbone

- Each candidate is a **subclass of `SalienceExplorer`**, feature-flagged off by default.
- **Firewall:** feature off ⇒ byte-identical action trace to v6 (test, like `TourExplorer`).
  The banked 0.33 floor is never modified. Nothing ships without explicit user go-ahead.
- Wired into `scripts/eval_efficiency.py::make_policy` as `transfer`, `prior`, `relational`.
- **Measurement:** TUNE/HOLDOUT split, actions-to-each-level, budget 6000 then 30000.
- **Pre-registered win bar (per candidate, vs `salience` baseline):** improves HOLDOUT `sum_eff`
  **AND** no HOLDOUT level drop **AND** no TUNE level drop. HOLDOUT = stand-in for hidden games
  (overfitting collapsed the leader 12.58%→0.25%). A pass = *submission candidate*, user-gated.
  Honest caveat: dev ≠ Kaggle.

## Candidate 1 — `TransferExplorer` (recommended)

Within-game reward-signature transfer. On every reward edge (`reward>0`), extract a signature of
the rewarding action from the *previous* grid:
- simple action → `("S", aid)`
- click → the `Obj` containing/nearest `(x,y)`; signature `(color, size_bucket, shape_bucket)`
  (shape = aspect-ratio × filledness buckets).

Accumulate signatures across the game's solved levels. **Promote** matching candidates to a higher
salience tier in later levels, so each new level tries the historically-rewarding action-class
first. Coverage preserved (priority-only change); reorders **with a real reward signal** (unlike
the killed signal-free frontier reorderings). Firewall: no signature learned ⇒ identical to v6.

## Candidate 2 — `PriorExplorer`

Hand-coded core-knowledge salience priors, no learning. Per-state click-priority boosts: odd-one-out
distinctness (unique color/shape among many), matching-pair detection (≥2 objects sharing
color+shape → pairing mechanics), rarity. Simple actions untouched. Firewall: prior weight 0 ⇒ v6.

## Candidate 3 — `RelationalExplorer` (exploratory)

Relational state key: augment `object_state_key` with per-color counts + adjacency/containment
pairs while abstracting exact bbox coords, collapsing translation-equivalent states. Keep the exact
key for edge-recording; use the abstract key only for frontier-equivalence. Highest uncertainty.

## Process

Build all three (isolated files) + firewall tests + harness wiring, then run baseline + 3
candidates as parallel background sweeps at 6000. Survivors confirmed at 30000. Compare to the
pre-registered bar; iterate or kill. Document results; submit only on user go-ahead.
