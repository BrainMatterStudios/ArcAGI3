# Phase Q — History-Augmented State (crack ls20) — design

Date: 2026-06-21
Branch: winning/mechanic-model-search (continues)
Status: design approved in principle (brainstorming); pending written-spec review.

## 1. Why this exists

Phase P proved (engine-source + active probe) that ls20's wall is an **invisible-state
representation-completeness** problem: the agent's hidden **rotation** gates the goal but renders
identically in the frame, so the banked masked key cannot distinguish "correctly oriented" from not —
identical frame+key, win only at the matching rotation. The verdict: a *passive* detector is too
sparse; the fix is to **eagerly augment the state key** so the hidden variable becomes part of the
node identity.

This phase builds that agent. Success = **crack ls20 L1 efficiently** (today only brute-blind reaches
it at ~7900 actions ≈ 0 efficiency) **without regressing any dev game.** It is reward-free (dodges the
value-head kill, Exp 54), and it directly unblocks the one proven method (Discovery, Exp 49, blocked on
ls20 *by this invisible gate*). ls20 is a HOLDOUT game, so an efficient crack contributes to the score.

## 2. Mechanism — eager, self-gating, occlusion-aware

Node key = `base_masked_object_key  +  sorted( (object_signature, visit_count mod N) )` over the
**static non-background objects the avatar has stepped on.**

- **Self-gating.** On a plain maze the avatar only ever stands on background → no object visits → no
  counters → the key is **byte-identical to the banked agent**. On ls20 the rotation tile is the one
  walkable glyph → it accrues a `mod 4` counter → the 4 rotation states become 4 distinct,
  separately-explorable nodes → the explorer reaches `(goal-adjacent, rotation == GoalRotation)` and
  its goal-entry edge deterministically wins.
- **Counter form = `mod N` (N=4 default), NOT raw.** ls20's hidden state is *cyclic*
  (`rotation = (start + visits) mod 4`). Raw/capped counts over-split (visit 1 and visit 5 are the same
  rotation but would get different keys) → the graph never converges. `mod N` exactly encodes a
  period-N cyclic hidden variable. N is the one hyperparameter; default 4 (rotation). A general
  variant folds a small set {2,3,4} (captures any period ≤4) at higher key cost — a follow-up if the
  fixed N fails to generalize; v1 targets ls20 so N=4 is correct and minimal.
- **Occlusion-aware visit detection (the linchpin).** When the avatar stands on the rot tile, the tile
  glyph is **hidden under the avatar** — so "avatar overlaps a current-frame object" (what the Phase P
  map used, and why it read 0 on ls20) cannot see the visit. The agent must keep **object-location
  memory**: remember each static non-bg object's location from frames where it is visible; a *visit*
  to object O = the avatar's tracked position enters O's remembered cell(s) (typically coinciding with
  O disappearing under the avatar). Avatar position is tracked via `movement.infer_translation` (it
  already detected ls20's avatar: 2 colors). Object signature = `(color, bbox)` for static glyphs;
  moving objects get inconsistent signatures → not counted (graceful — they don't pollute the key).

## 3. Architecture
`src/arcagi3/history_augmented_explorer.py`: `HistoryAugmentedExplorer(SalienceExplorer)` overriding
`_key` (append the counter tuple) and the per-step hook that updates avatar position + object-location
memory + visit counts. Built off `SalienceExplorer` (key-identical to banked `TransferExplorer` v13);
the augmentation is orthogonal to Transfer's reward-signature reordering and can be composed later.
A `augment=False` default makes it **byte-identical to the banked agent** (firewall test).

## 4. Firewall + pre-registered gate
- `augment=False` → byte-identical to `SalienceExplorer` (firewall test in the suite).
- **Pre-registered success (strict, user-chosen):**
  1. **ls20 L1 cracked efficiently** — completed in ≪7900 actions (target: low hundreds; the maze is
     small, optimal 13), ideally L2+ too (they have other GoalRotations).
  2. **Zero dev regression** — no TUNE/HOLDOUT game drops levels vs banked on the `eval_efficiency`
     A/B (the augmentation must not explode other games into intractability).
- If it cracks ls20 but regresses elsewhere → tighten the object filter (§5) and re-gate; ship behind
  the flag only if a clean configuration is found. The banked 0.33 is never at risk (subclass, default
  off, parent untouched).

## 5. Risks & mitigations
- **Occlusion-aware visit detection is hard and load-bearing.** If it fails to register rot-tile
  visits, ls20 won't crack. Mitigation: validate the visit-counter against the ls20 ground truth first
  (does the rot-tile counter increment exactly when the engine's `cklxociuu` changes?) before judging
  the crack. The Phase P active probe + engine source give an exact oracle.
- **State explosion on walkable non-gate objects.** A path tile the avatar repeatedly crosses gets a
  growing `mod N` counter → more states → possible regression. (Collectibles are safer: they vanish on
  contact, counter maxes at 1.) The no-regression A/B measures it; if it bites, tighten the object
  filter to *repeatedly-visitable static glyphs that occlude-and-reappear* (the rot-tile signature),
  excluding consumed/decorative objects.
- **Avatar / object detection reliability.** No avatar learned → no counters → no change (graceful, no
  regression, but no crack on that game).
- **Period assumption (N).** N=4 is ls20-correct but assumes the period; generalization to other
  periods is an explicit follow-up, not a v1 requirement.

## 6. Deliverables
1. `src/arcagi3/history_augmented_explorer.py` — the subclass (off byte-identical; on = augmented key).
2. `tests/test_history_augmented_explorer.py` — (a) `augment=False` byte-identical firewall on a local
   game; (b) unit test: stepping a known static glyph increments its `mod N` counter and changes the
   key; (c) unit: no walkable objects → key == banked key.
3. An **ls20 validation run** (offline engine): does the agent crack ls20 L1 (and L2+), and does the
   rot-tile counter track the engine's `cklxociuu`?
4. A **dev no-regression A/B** (`eval_efficiency`, augment on vs off) + the pre-registered verdict.
5. A memory note recording the outcome (first wall-crack, or the precise failure mode).

Estimated effort: ~1–1.5 days (the occlusion-aware visit detection is the bulk).

## 7. Success criteria (process)
A firewalled agent that either (a) cracks ls20 L1 efficiently with zero dev regression — the first
crack of an invisible-state wall — or (b) returns a precise, ground-truth-anchored failure mode (visit
detection failed / state explosion / period mismatch) that directs the next iteration. The banked 0.33
is provably untouched throughout.

## Interim result (2026-06-21) — first implementation FAILS at the oracle; fix direction identified

Tasks 1–3 built + tested green (firewall byte-identical; occlusion + multi-color-avatar unit tests
pass; a code review caught and fixed a multi-color-avatar identity bug via `infer_all_translations`).
Task 4 ls20 crack run (`scripts/ls20_crack.py 8000`):
- banked (augment off): ls20 **L1** @8000.
- history-augmented: ls20 **L0** (regressed), **rot-oracle 4148/7546 (55%) mismatches**, one counted
  object `(color 0, bbox 31,21–32,22)` with **88 visits**.

**Diagnosis:** the visit-counter does NOT track the engine rotation. Likely compound cause: (a) the
avatar is a 5×5 block moving on a 5-cell pitch while the rot tile is ~1 cell, so block-overlap fires
across a band rather than on the engine's precise anchor-step; and/or (b) the counted color-0 object
is a maze element near (but not) the rot tile at (32,19), so the agent is counting the wrong thing.
The spurious 88-count explodes the augmented graph → ls20 regresses L1→L0.

**This is the spec's anticipated failure mode (visit detection), surfaced precisely by the cklxociuu
oracle.** Fix = engine-grounded calibration of visit detection: instrument which cells the avatar
actually occupies vs the known rot-tile cell (32,19), align the visit event to the engine anchor-step,
and drive `rot_mismatch → ~0` against the oracle before re-judging the crack. The mechanism (mod-N key
augmentation) is sound and unit-proven; the perception of "stepped on the rot tile" is what needs
iteration. Banked 0.33 untouched throughout (subclass, firewalled). Decision pending: iterate the
detector now vs checkpoint.

## Iteration update (2026-06-21) — two root causes; one fixed, one is the real blocker; redesign proposed

Drove the known solution on the offline engine with engine introspection (true avatar pos + cklxociuu
+ rot-tile sprite). Findings:
- **Rot trigger = engine (19,30) = grid (32,19); rotation flips exactly when the avatar steps onto it**
  (the arrow glyph at grid rows 31–33/cols 20–22 IS occluded then — n_small 12→9). So the occlusion
  model is correct and the agent was counting (part of) the real rot tile.
- **Root cause 1 (FIXED): counter was cumulative across the whole run, but the engine resets rotation
  on every life-loss/level-restart** (`cklxociuu = index(StartRotation)`; 42 steps/life × 3 lives).
  Fix: `_reset_history()` on terminal/not-played/level-up. This removed the L1→L0 regression
  (augmented now matches banked L1).
- **Root cause 2 (NOT solved — the real blocker): avatar segmentation.** ls20's avatar body color (9)
  is **also used by static maze objects**, so `infer_all_translations` (which needs a color's WHOLE
  mask to translate) only ever learns the head (color 12). The avatar footprint is head-only → its
  overlap with the offset rot glyph is unreliable → oracle stays ~66% mismatch, no crack. A
  translation-mask segmentation (take all cells that arrived by the avatar's delta) was tried and
  **regressed** (noisier), so it was reverted.

**Status:** mechanism (mod-N key augmentation) proven by unit tests; firewall intact; ls20 no longer
regresses but does NOT crack; banked 0.33 untouched. **Proposed redesign (next session):** drop
avatar-overlap entirely and use **disappearance/occlusion-event detection** — count a visit when a
tracked static glyph cluster *disappears* (it can only vanish because the avatar stepped onto/over it),
edge-triggered on present→absent, grouping the multi-cell arrow as one cluster. This sidesteps the
avatar-segmentation problem (no need for a precise avatar footprint at all) and is the most promising
path to drive `rot_mismatch → 0` and crack ls20. Checkpointed here after honest non-converging
iteration rather than thrashing further inline.

## Redesign v2 result (2026-06-21) — disappearance detector also blocked by avatar segmentation; CHECKPOINT

Implemented the disappearance/occlusion-event detector (count a confirmed-static glyph cluster that
disappears; 6 unit tests green incl. firewall + a moving-object-excluded test). It did NOT crack ls20
either: the counts show it counting the AVATAR (25-cell color-12-head + color-9-body clusters) when the
avatar pauses ≥2 frames against walls, and 8-adjacency clustering merges the avatar with adjacent
glyphs. Root reason: the avatar's *parts* are themselves small objects, so "don't track the avatar
footprint" did not avoid the avatar — its blocks pollute as clusters. **The avatar-segmentation
problem is therefore intrinsic to BOTH detectors.**

**Honest conclusion:** the history-augmented *mechanism* is proven and firewalled (banked 0.33 never
at risk), but **robustly perceiving "the avatar stepped on the rot tile" from ls20 pixels is a real,
substantial perception problem** — color-sharing (body color 9 == maze color), occlusion, 5×5 blocks
on a 5-cell pitch, blocked-move pauses, and glyph-merging all conspire. Two detector designs and 4+
inline iterations did not converge. Checkpointing rather than thrashing.

**Recommended path for a dedicated next effort:** use the repo's **dormant `tracking.py`
`ObjectTracker`** (built for object persistence + typed events across frames, currently unused in the
live path) to get stable object identity through occlusion, and explicitly segment the avatar (the
single rigidly-translating multi-cell block) so it can be excluded — then either detector becomes
reliable, validated to `rot_mismatch → 0` against the `cklxociuu` oracle. That is a focused perception
sub-project, not a tail-of-session tweak. The mechanism, the firewall, the oracle, and the unit
harness are all in place to support it.
