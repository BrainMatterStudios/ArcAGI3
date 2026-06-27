# Phase P — Invisible-State Prevalence Map — design

Date: 2026-06-21
Branch: phase0-prune-oracle (continues)
Status: design approved in principle (brainstorming); pending written-spec review.

## 1. Why this exists

Three measure-first kills this session (prune-prior, re-traversal audit, walk-reduction A/B) plus a
25-agent new-directions workflow all converged on the same pointer: the efficiency levers top out
near 0.33, and **the wall games that actually gate the hidden-averaged score are mostly perception /
invisible-state walls** that none of the efficiency ideas touch. This phase attacks that wall.

The canonical, fully reverse-engineered case is **ls20 L1** ([[arcagi3-ls20-mechanic-ground-truth]]):
the agent has a hidden **rotation** attribute that gates the goal (goal cell is a wall until
`rotation == GoalRotation`). Rotation is **invisible in the frame** — the avatar renders byte-for-byte
identically across all 4 rotations. The agent changes rotation by stepping on a rotation tile, with
**zero visual signal**. So a frame-based state key cannot distinguish "correctly oriented" from "not"
→ the explorer can never know when to enter the goal. ls20 L1 = 0/8000 even after perception fixes.

**Structural insight:** invisible state shows up as **determinism violations** in the frame-based
transition function — the same masked-state-key + same action → different next-key. The current
explorer's suspicion filter *suppresses* this as noise; a fix would *exploit* it, augmenting the state
with minimal history (interaction counts) to restore a complete, Markov representation. That is a
genuinely different attack — about **correctness/completeness**, not efficiency — and it is offline,
no-LLM, self-supervised by determinism.

Before building any fix, this phase **measures how widespread invisible-state actually is** across the
dev games (distinguished from animation noise), anchored on ls20 ground truth. Pure measurement; no
agent change; 0.33 untouched.

## 2. Scope

### In scope
- **Passive prevalence map**: determinism-violation rate + history-resolvable rate per dev game.
- **ls20 ground-truth anchor** as a GO/NO-GO on whether the passive signal is even observable.
- **Minimal active-probe fallback** on ls20: if passive violation is too sparse, a scripted probe that
  drives the agent to the goal-gate at each rotation and checks whether the violation then appears and
  resolves.

### Out of scope (deferred to a follow-on spec, gated on this map's verdict)
- The actual history-augmentation fix (state-splitting in the explorer).
- A general active-probing mechanism for unseen games.
- ANY change to `SalienceExplorer` / the banked agent.

## 3. Feasibility facts (verified this session / from memory)
- `SalienceExplorer._key(grid)` computes the masked object-state key (border_mask + VolatilityTracker +
  distractor-mask already applied). The map measures residual non-determinism *after* existing masking,
  isolating "invisible state + residual animation".
- Live throughput ~510 act/s; the dev games are reachable via `OperationMode.NORMAL`. ls20 is in the
  HOLDOUT split.
- ls20 ground truth (for the anchor + active probe): rot tile at grid (32,19); goal at grid (12,34);
  rotation = `(rot-tile step count) mod 4`, start idx 3 (270°), goal idx 0 (0°); action→grid deltas
  pitch 5 (act1 up (-5,0), act2 down (+5,0), act3 left (0,-5), act4 right (0,+5)); BFS solution
  `[3,3,3,1,1,1,1,4,4,4,1,1,1]` (13 actions, hits rot tile once); StepCounter 42/life, 3 lives. A local
  engine file `environment_files/ls20/9607627b/ls20.py` exists for fully-offline deterministic runs
  (verify presence in the plan; otherwise use the live API, which is also deterministic).
- Reuses the trajectory-capture pattern (prune_oracle / InstrumentedExplorer). No agent behavior change.

## 4. Design

### §4.1 Passive detection — determinism-violation multimap
Capture trajectories per game (banked v6 config). Per step log: `masked_key`, `action`, `next_key`,
and enough raw state to derive history features (avatar/cursor position, the set of salient objects
with stable identities, per-object overlap/click counts so far). Build the multimap
`(masked_key, action) → {next_key}`. A pair with ≥2 distinct next-keys is a **violation**. Report the
raw violation rate (violating pairs / total distinct pairs, and violating *transitions* / total).

### §4.2 History-resolvability discriminator
Candidate history-feature family (small, general, ls20-motivated): for each **salient object** O (the
rare-color / small connected components perception already tiers high) and each N ∈ {2,3,4}, the
feature `f = (interaction_count(O) mod N)`, where interaction = avatar-overlaps-O (move games) or
click-on-O (click games). A feature **resolves** a violating pair iff, grouping that pair's transition
instances by `f`, every group has a single `next_key` (i.e. `(masked_key, f, action) → next_key` is a
function). Search for the **minimal** resolver (fewest objects, smallest N). Per game report: fraction
of violations resolved by *some* history feature, and the resolving feature(s).
**Over-merge control:** if a violation is resolved by *un-masking* a currently-masked cell (not by any
history feature), tag it `over_merge` (a representation bug, not invisible state) and report separately.

### §4.3 ls20 anchor — GO/NO-GO on the whole passive approach
ls20's hidden rotation gates **only** the goal-entry transition, so the passive violation is observable
only where the agent reaches the goal-gate at ≥2 different rotations. The anchor reports: (a) did any
violation appear on ls20 at all (observability), and (b) was it resolved by the rot-tile object's
`count mod 4` (correctness). **If the anchor does not fire, passive detection is insufficient for
gated-goal invisible-state** — a decisive finding, not a tuning failure.

### §4.4 Minimal active-probe fallback (ls20-specific, scripted)
If §4.3 shows the passive violation is sparse/absent, run a scripted ls20 probe using the ground-truth
coordinates: for r in {0,1,2,3}, drive the agent (known action deltas) to the rotation tile, step on it
`r` times (setting rotation), navigate to the goal-adjacent cell, attempt goal-entry, and record
`(from_masked_key, action, outcome, level_completed)`. The probe **confirms the mechanism is
observable-by-active-probing** iff: the `from_masked_key` is identical across r (rotation invisible),
the outcome differs across r (entry succeeds at the matching rotation, blocked otherwise), and the
difference is explained exactly by `rot-tile-count mod 4`. This is an ls20-specific diagnostic (uses
ground truth), not a general method — its job is to de-risk the eventual active-probing fix before we
commit to it.

### §4.5 Output + pre-registered interpretation
Per game: raw violation rate, history-resolved rate, resolving feature, over_merge rate, walled/solved
status. ls20 anchor: fired? resolved by rot-mod-4? Active probe (if run): observable? Walled-vs-solved
comparison. **Verdict** (pre-registered bar for "real, common blocker": the anchor fires AND **≥2
walled games beyond ls20** show history-resolved violations *concentrated at gating transitions* whose
resolved-rate is clearly above the solved-game controls — not diffuse noise):
- **(i)** anchor fires AND the ≥2-walled-games bar is met → invisible-state is a real, common blocker →
  build the history-augmentation fix (next spec).
- **(ii)** anchor fires but only ls20 lights up → narrow lever; reconsider EV.
- **(iii)** anchor does NOT fire passively BUT the active probe confirms observability-by-probing →
  passive detection is dead; the next design is **active probing**, not passive history-augmentation.
- **(iv)** neither passive nor active confirms on ls20 → the invisible-state attack as framed is wrong;
  back to the drawing board (the perception wall may need a different representation entirely).

## 5. Deliverables
1. `src/arcagi3/invisible_state.py` — pure analysis: violation multimap + history-resolvability +
   over-merge tagging (numpy only; no agent change).
2. `scripts/invisible_state_map.py` — capture driver over the dev games + ls20 anchor + the scripted
   ls20 active probe.
3. A per-game results table + the pre-registered verdict.
4. A memory note recording the prevalence verdict + the active-probe outcome.

Estimated effort: ~1 day (richer per-step logging + the feature search + the scripted probe).

## 6. Success criteria (process)
A trustworthy prevalence map that (a) self-validates on the ls20 anchor (fires + resolves, or honestly
reports it can't passively), (b) separates invisible-state from animation noise and over-merge, and
(c) returns one of verdicts (i)-(iv) to direct the next session. A "passive detection is insufficient"
result is a success (it redirects to active probing before a wasted build).

## 7. Risks & caveats
- **Sparse observability for gated goals.** The central risk; the anchor + active probe are designed to
  measure it directly rather than assume it away.
- **Weak feature family → false negatives.** Calibrated by the ls20 anchor (the family must resolve the
  case we fully understand before we trust its prevalence numbers elsewhere).
- **Animation/over-merge confounds.** Separated by the history-resolvability test (noise isn't
  history-resolvable) and the explicit over_merge tag (resolved by un-masking, not history).
- **Revisit requirement.** Violations only appear if trajectories revisit a state under different
  history; run adequate budget (ls20's reset-heavy 3-lives/42-steps helps).
- **dev ≠ Kaggle.** This is a diagnostic to direct R&D, not a leaderboard change. The 0.33 floor is
  untouched (no agent behavior modified).

## Results (2026-06-21) → VERDICT (iii): passive insufficient, active probe CONFIRMS

Re-grounding note: the banked agent is `TransferExplorer` v13 (still 0.33), which inherits
`SalienceExplorer._key` verbatim — so the map (driven by SalienceExplorer) reflects the banked key.
The ls20 mechanic was re-confirmed from engine source (`environment_files/ls20/9607627b/ls20.py`):
**rotation-gated** (`bejndxqqzf` requires `cklxociuu == ehwheiwsk[goal]`); the repo's
`experiment-overview.html` paint/moving-goal writeup is superseded/wrong.

### Active probe (`scripts/invisible_state_probe.py`) — DECISIVE
Offline engine, state-injection: drive the known solution to goal-adjacent, inject `cklxociuu = r`
for r in {0,1,2,3}, take the goal-entry action. Result:
- full solution wins (validates the rotation model);
- goal-adjacent **frame identical** across rotations AND **masked state-key identical**;
- outcomes **WIN only at idx 0** (== GoalRotation), BLOCKED at 1/2/3.

→ Direct ground-truth proof: ls20's wall is an **invisible-state representation-completeness** problem
— identical observation, outcome gated by a hidden rotation counter — **resolvable by augmenting the
state with that counter.**

### Passive map (`scripts/invisible_state_map.py`, 16 games @30k) — sparse, as predicted

| signal | result |
|---|---|
| INVISIBLE_STATE fired | only **lf52** (3) — a *solved* game; possibly genuine or a feature artifact |
| **ls20 anchor** | **0** (1 violation, classified over_merge) — passive detection did NOT fire |
| walled games (sc25/re86/wa30/tn36) | **0** INVISIBLE_STATE each |
| dominant category everywhere | **OVER_MERGE** (tu93 56, sp80 66, m0r0 40, sk48 30, …) |

The passive signal is too sparse for gated goals: the agent rarely reaches the rotation gate at ≥2
rotations during normal exploration, so no determinism violation is observed there (ls20 had 1 total
violation, and it was over_merge). The feature family is not the problem — it *did* fire on lf52 — so
the absence on ls20/walled games is genuine sparsity, the predicted failure mode.

### Verdict: (iii) — passive detection insufficient; active probing / history-augmentation is the path
The map alone would have (wrongly) suggested invisible-state is near-absent; the active probe shows it
is real, outcome-gating, and counter-resolvable on the canonical wall. So a *passive* "wait for
determinism violations" detector is dead for gated goals — the fix must **actively** build a
history-augmented state (track interaction counters at candidate gating objects as part of the node
key, and deliberately revisit gates under different histories). This is the next-spec direction.

**Secondary finding (separate concern):** OVER_MERGE dominates violations across nearly all games — the
current masking (`border_mask` + `VolatilityTracker`) is over-merging visibly-different states fairly
widely. Worth a focused look (it can cost coverage), independent of the invisible-state work.

**Outcome:** a successful Phase P — the map + probe together delivered a clear, ground-truth-anchored
verdict that redirects from passive detection to active history-augmentation, and proved ls20's wall is
breakable by representation (not exploration/planning). No agent behavior changed; 0.33 untouched.
`invisible_state.py` + the two scripts are kept as the foundation for the next-spec history-augmented
agent. Synergy: Discovery (Exp 49, proven ~14× but blocked on ls20 *by this invisible gate*) becomes
the natural consumer of a rotation-counter-augmented state.
