# Phase 0a — Prune-Capability Oracle Probe (design)

Date: 2026-06-21
Branch: factory/experiment
Status: design approved in principle (brainstorming); pending written-spec review.

## 1. Why this exists

The new strategic bet is an **offline, cross-game meta-learned exploration/affordance prior**:
train a transferable "which states/actions are progress-relevant" prior offline (where dense
signal exists — across many synthetic + the 3 public games), then run it **inference-only** at
eval to focus exploration on unseen games. The intent is to sidestep the confirmed
sparse-per-game-reward wall and lift the squared action-efficiency score above the banked 0.33.

The memory record (`arcagi3-research-findings`) raises one decisive objection to this bet that
must be answered **before** any generator or training code is written:

> The wall is **not** goal-identification (knowing which cell/action is interactive). It is
> **coverage + goal-achievement**. m0r0 *correctly* learned the rewarding colors and still scored
> 1/6 ("the gap is NOT goal-IDENTIFICATION but goal-ACHIEVEMENT"). Wall levels (tu93 L6 = 20905
> actions) are *genuine* large-state coverage at a healthy 8.9 actions/new-state, where
> "reordering search without a signal doesn't cut expected coverage."

A learned affordance/salience prior is mechanically a better **prioritizer**. Under the squared
metric `S_level = min(1.15, (human/agent_actions)^2)`, a prioritizer only moves the score if it
**prunes** — visits *fewer* states before the level-up — not if it merely **reorders** which
states are visited first. Reorder-only has been measured inert across ~8 prior levers.

**Therefore Phase 0a proves, cheaply and falsifiably, whether prune is (a) possible and (b)
learnable — before building anything trainable.** It is pure analysis on recorded trajectories of
the banked v6 agent. No model, no procedural generator, no new agent, no change to the 0.33 floor.

## 2. Scope

### In scope
- A single analysis script (`scripts/prune_oracle.py`) that runs v6, captures trajectories, and
  computes the Tier-1 and Tier-2 metrics below.
- A results table + a memory write recording the verdict and the per-game prune map.

### Out of scope (explicitly deferred behind this gate — decided 2026-06-21)
- The procedural game generator.
- Any learned model / prior.
- The "can a small model train on T4 with no local CUDA GPU" eval-feasibility de-risk.
- Any modification to `SalienceExplorer` / the banked v6 submission.

Rationale: zero EV in de-risking *how* to train a prior until we have shown a prior *could* help.
If §5 returns "kill", we have spent ~half a day instead of weeks.

## 3. Feasibility facts (verified 2026-06-21, this repo)
- `ARC_API_KEY` present and non-empty in `.env`; `OperationMode.NORMAL` works.
- **Live throughput ~510 actions/sec** (measured: 300 actions on vc33 in 0.6 s) — NORMAL mode
  serves locally-cached game defs. A 30k-action wall run is ~60 s; 5–6 wall levels is minutes.
  (Supersedes the stale "~2.6 actions/sec" note.)
- `SalienceExplorer.nodes: dict[bytes,_Node]` is the full state graph. Each `_Node.edges` is
  `action -> (next_key, reward)`. **Reward-bearing edges are exactly the level-up transitions**
  (`reward = levels - prev_levels`). Per-step state-acted-from is readable as `pol.prev_key`
  after each `decide()` call, with `pol.prev_action`. No explorer changes are required to capture
  a full trace.
- Interpreter: `.venv/bin/python` (3.12). Run pattern:
  `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/prune_oracle.py ...`.

## 4. Experiment design

### §1 Data capture
For each target game, run `SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)` (the exact
banked v6 config) under a budget high enough to complete the target level. Log per step:
`(step, levels_before, state_key = pol.prev_key, action, reward)`. At end, serialize the final
`pol.nodes` graph (keys + edges + per-node visit counts + candidate tiers).

Segment the trace into levels by the level-up steps (where `levels` increments). For a target
level L: `L_window = [first step of L, the level-up step of L]`. A state "belongs to L" if its
**first-discovery step** falls in `L_window`.

**Target levels** (must have a ground-truth level-up edge): tu93 L6, vc33 L3, m0r0 L2, plus
opportunistically ls20, lp85, cd82 where a completed level-up edge appears.

**Uncompleted walls** (decided 2026-06-21): before excluding a wall v6 fails to complete at the
default budget, run v6 at high budget (≥150k actions) to try to capture a level-up edge. If it
still does not complete, report it as `no ground truth — excluded (known blind spot)`. Never
silently drop.

### §2 Tier-1 — does a prune *ceiling* exist?
Over the recorded edge set, compute the shortest path from the L-start state to the L-up state
(BFS over `_Node.edges`, restricted to states in `L_window` — you can only path through
discovered edges). Report, per level:
- `discovered_states` = |states first-seen in L_window|
- `path_states` = states on the shortest start→levelup path
- `prune_ceiling_states = 1 - path_states / discovered_states`
- `actual_actions` = actions spent from L-start to level-up (the record's "marks")
- `path_actions` = action count of the shortest path
- `prune_ceiling_actions = 1 - path_actions / actual_actions`

Interpretation: ceiling ≈ 0 means the goal genuinely requires visiting ~everything → prune
impossible there. A high ceiling means most explored states were off the productive path — a prune
opportunity *exists*. **Caveat (stated, not hidden):** this is an optimistic oracle — it already
knows the goal location, so a high ceiling is necessary but not sufficient.

### §3 Tier-2 — is the prune *learnable*?
For each state in `L_window`, label `on_path` (on the shortest start→levelup path) vs `off_path`.
Extract features computable **only from what is observable at the state's discovery time** (no
hindsight): e.g.
- object count; object-size histogram (min/median/max/#small);
- number of new colors vs the parent state;
- the salience **tier** of the action that discovered this state (0..9);
- frontier degree (number of untried candidates);
- board fill fraction; number of distinct colors.

Fit a cheap classifier (logistic regression and a depth-≤3 tree) to predict `on_path`, evaluate by
**ROC-AUC** with grouped/held-out folds across games (so AUC measures *transfer*, not per-game
memorization). Report AUC against two controls (per the "always use a baseline control" rule):
- **Base-rate control:** trivial classifier predicting the majority class.
- **Label-shuffle control:** same features, labels permuted — confirms AUC > 0.5 is real signal,
  not an artifact of class imbalance or small samples.

AUC ≈ 0.5 (≈ shuffle control) → on-path and off-path states are observationally identical at
decision time → **no prior, learned or not, can prune** → kill that mechanic. AUC clearly above
the controls → a transferable prior is plausible.

### §4 Decision rule — the output is a per-game map, not a binary

Pre-registered thresholds (chosen a priori, not tuned on the results):
- **"High ceiling"** = `prune_ceiling_actions ≥ 0.5` (≥ half the actions were off the productive
  path).
- **"AUC clearly > control"** = held-out (cross-game) ROC-AUC `≥ 0.65` **and** `≥ shuffle_AUC + 0.10`.

| Tier-1 ceiling | Tier-2 AUC | Verdict for that game/mechanic |
|---|---|---|
| `< 0.5` | — | Prune impossible — true coverage wall; a prior cannot help |
| `≥ 0.5` | not clearly > control | Ceiling exists but invisible at decision time → **kill** the learned-prior bet for this mechanic |
| `≥ 0.5` | clearly > control | **Build** — a prior could prune; the generator should target this mechanic family |

### §5 Expected result & honest branches
Prediction from the record: **click / known-mechanic games (vc33) → high ceiling + high AUC**;
**avatar coverage walls (tu93) → high ceiling + low AUC** (the "no guiding signal" wall). If so,
the bet **narrows** to a transferable high-structural-impact-click prior — a real but bounded lift
that extends the one proven win (dense-lattice / impact heuristic), *not* a tu93 crack. That is a
finding to record, not a failure.

Branches:
- **All games AUC ≈ control →** kill the cross-game-prior bet; record that the 0.33 ceiling holds
  against this lever too; stop.
- **Click games pass, avatar walls fail →** proceed to Phase 0b but scope the generator + prior to
  the click/structural-impact mechanic family only; set expectations to a bounded efficiency lift.
- **Broad pass (avatar walls too) →** proceed to the full generator + Phase 1 prior as originally
  envisioned.

## 5. Deliverables
1. `scripts/prune_oracle.py` — capture + Tier-1 + Tier-2 + controls; prints a per-level table and
   saves JSON to `/tmp/prune_oracle_<game>.json`.
2. A results table (per game: ceiling_states, ceiling_actions, AUC, AUC_shuffle, n_states).
3. A memory write to `arcagi3-research-findings` recording the verdict and the per-game prune map
   so it is never re-derived.

Estimated effort: ~half a day. No generator, no model, no GPU work until §4 says "build".

## 6. Success criteria for Phase 0a (process, not outcome)
Phase 0a "succeeds" by producing a **trustworthy, control-validated verdict** — kill OR build OR
narrow — for each probed game, with the optimistic-oracle caveat made explicit. A "kill" outcome
is a successful Phase 0a (it saved weeks). The only failure mode is an un-controlled or
ground-truth-less number that misleads the build/kill decision.

## 7. Risks & caveats
- **Optimistic oracle.** Tier-1 assumes goal location is known; real exploration cannot walk
  straight to an unseen goal. High Tier-1 ceiling is necessary, not sufficient. Tier-2 is the
  learnability check that partially closes this gap; full proof still needs Phase 1 (train a prior,
  measure holdout actions).
- **Small positive class.** On-path states are few per level → AUC variance is high. Mitigate by
  pooling across games and reporting the shuffle control + sample sizes.
- **State-key recurrence across levels.** Segment strictly by first-discovery step within
  `L_window`; do not attribute a state to L if it was first seen earlier.
- **Shortest-path over recorded edges only.** Correct oracle (cannot path through undiscovered
  edges), but if the explorer recorded a suspiciously short reset-edge path, validate the path is
  reset-free / within-level before trusting `path_actions`.
- **Dev ≠ Kaggle.** This probe informs a build/kill decision; it does not itself change the
  leaderboard. The ship gate remains `scripts/eval_efficiency.py` HOLDOUT efficiency, and the
  0.33 floor is never regressed (Phase 0a touches no agent code).
