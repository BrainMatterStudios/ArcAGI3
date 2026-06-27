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
