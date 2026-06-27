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

## Redesign v3 result (2026-06-27) — PERCEPTION SOLVED (mobility memory + occlusion check); but the augmentation itself is KILLED by a broad dev regression

A fresh session built the engine-introspection diagnostic the v2 checkpoint asked for
(`scripts/ls20_rot_diag.py`: drive the known solution, print engine ground truth — avatar sprite x/y,
`cklxociuu` — beside the rendered pixels and `connected_components` near the rot tile). It revealed the
exact mechanic with zero ambiguity: the rot glyph (colors 0/1 at grid rows 31–33) is occluded by the
avatar **exactly on the single step where `cklxociuu` flips** (step 5 of the 13-step solution, engine
(19,35)→(19,30)). bg auto-detects to **4** (the border) while the maze floor is **3**, so the floor is
one 892-cell mega-object and small glyphs/avatar-parts are the only ≤16-cell components.

**Two root causes of the v1/v2 failures, fixed:**
1. *A paused avatar is indistinguishable from a static glyph by stability alone* (the avatar sits ≥2
   frames on every blocked move). The only discriminator is HISTORY: it moved before it paused. Fix:
   per-component identity across frames (per-color global nearest-centroid greedy) with a sticky
   `ever_moved` flag; mobile components are excluded from glyph clusters forever. This needs no
   color-based avatar segmentation (body color 9 is shared with the maze) because matching is
   per-COMPONENT — the moving avatar-body component is mobile while static maze-9 components are not.
2. *Pollution from non-avatar occlusions.* Even with mobility memory, two false sources remained: the
   avatar's spawn block (counted once when it first leaves spawn → into floor) and **HUD digits at rows
   59–62 changing value** (old digit cluster "disappears" → counted). Fix: an **occlusion check** — a
   present→absent edge counts only if a vanished cell is now covered by a *mobile* (ever_moved)
   component. "Vanished into background" (spawn→floor) and "replaced by another static thing" (HUD digit
   change) both fail the check; only "a moving avatar stepped onto it" passes.

**Perception result (decisive, validated against the `cklxociuu` oracle):** global occlusion tally over
a 1500-step exploration went from `{HUD×32, spawn×6, rot-glyph×6}` to **rot-glyph ONLY**. The rot-glyph
counter alone tracks `(cklxociuu − StartRotation) mod 4` on **~91%** of frames (133/1500 mismatch, vs
the v1 55–66%); the residual is the per-life rotation reset (the engine resets `cklxociuu` to
StartRotation on each life, but an individual life-loss raises no terminal signal the agent can see).
9 unit tests green incl. the byte-identical `augment=False` firewall and a new
`test_paused_then_moving_object_not_counted` that captures the real failure mode. **The stated blocker
— "robustly perceive the avatar stepping on the rot tile" — is solved.**

**But the end-to-end augmentation is KILLED by the no-regression gate (the deeper, real blocker):**
- *ls20 crack is fragile.* actions-to-L1 over 10 seeds (`scripts/ls20_speed.py`): banked solves **5/10**
  (mean 5381), histaug solves **2/10** (mean 3090). Augmentation *accelerates* the seeds it cracks
  (seed 0: 4269 vs 7961; seed 4: 1912 vs 3044) but *regresses* seeds 1/3/6 (banked solves, histaug
  does not) and rescues none. At **double budget (16000)** histaug STILL fails 1/3/6 → the regression
  is **not** recoverable state-inflation; it is fundamental graph fragmentation.
- *Broad dev regression* (`eval_efficiency`, budget 6000, augment on vs off): TUNE mean_levels
  **1.75 → 1.00** (tu93 collapses L5→L0, ar25 L2→L1), HOLDOUT sum_eff **1.19 → 0.06**. Only ls20 itself
  improves (L0→L1). Folding the rotation counter into the **GLOBAL node key** turns every position into
  4 rotation-distinct nodes and, on any game where the avatar occludes static glyphs, fragments
  exploration catastrophically.

**Verdict (pre-registered DoD):** (1) oracle hugely improved (~9% within-life) but not ≈0 (per-life
desync); (2) ls20 cracked on some seeds, faster, but not robustly; (3) **FAILS no-regression — decisively**;
(4) firewall green / banked 0.33 untouched (TransferExplorer v13 never references this subclass;
`augment=False` byte-identical). Per validate-or-kill, **the history-augmented-state approach is killed
as a promotable agent.** The lesson is sharper than "perception is hard": *even with perfect perception,
a GLOBAL key augmentation is the wrong shape for a frontier graph explorer.* The only viable redesign is
a **SELECTIVE** augmentation — fold the cyclic counter into the key only for the outcome-gating node(s)
(the goal-adjacent cell), not globally — plus a way to observe life-loss to reset the per-life counter.
That is a new design, not a tweak; it should not be attempted without an explicit go-ahead, since the
firewall already protects the banked score and global augmentation is a proven net loss.

## Selective augmentation design (v4, 2026-06-27) — exhaustion-triggered split [APPROVED, in progress]

Go-ahead given to build the selective redesign. The v3 kill localized the fault precisely: global key
augmentation splits *every* position by the hidden phase, fragmenting exploration on every game. v4
makes the split **selective** so the gate is distinguished while the rest of the graph is byte-identical
to banked.

**Mechanism diagnosed (why banked fails the seeds it fails).** SalienceExplorer marks an action `tried`
once it records an edge for it (`node.edges[action]`). At the goal-adjacent **gate** node (key K, with
rotation invisible), the agent tries "enter goal", is blocked at the wrong rotation, and records the
edge — so goal-entry is now `tried` at K and is **never retried**, even after the agent later steps on
the rot tile and the true rotation becomes 0. The node looks identical, so the win is unreachable.

**Core rule.** Append the hidden-phase tag to a node's key **iff both**: (1) the node is **exhausted**
(no untried candidate actions remain — the agent is stuck there), and (2) a **manipulable cyclic hidden
state exists** — some occlusion counter has cycled (value ≥ 2), proving a revisitable glyph the agent
can use to change the hidden phase. Tag = `sum(counts) % counter_mod` (a single 0..3 value), so an
exhausted base splits into at most `counter_mod` variants, never the whole graph.

**Why it cracks ls20.** The gate exhausts after the first blocked goal-entry. On the next visit its base
is in the exhausted set → key becomes `gate|P|phase`. After the agent steps on the rot tile (phase
changes), the gate at phase 0 is a **fresh node with goal-entry untried** → it tries it → win. This
repairs the "exhausted gate, never retried" failure with the minimum possible state inflation.

**Why it should not regress the dev suite.**
- Games with no occludable cyclic glyph: `counts` never cycles → condition (2) false → **no
  augmentation ever** → behaves like banked even at `augment=True`.
- Games with such a glyph: only *exhausted dead-ends* split (≤`counter_mod` each), never the productive
  open graph. Re-checked by the `eval_efficiency` no-regression gate (budget 6000, on vs off).
- `augment=False` stays **byte-identical** (firewall untouched).

**Implementation (one file, `history_augmented_explorer.py`).**
- `_key(grid)`: `base = super()._key(grid)`; return `base + b"|P|" + repr(phase).encode()` iff
  `self.augment and base in self._exhausted_bases and self._cyclic_active()`, else `base`.
- After each `decide`, look up the current node; if it has no untried actions at any tier
  (`not node.has_untried_le(MAX_TIER)`), add its **base** key to `self._exhausted_bases` (lazy; the
  re-key takes effect on the next visit — single level, since the base is what's stored).
- `self._cyclic_active()` = `any(c >= 2 for c in self._counts.values())`.
- `_reset_history` also clears `self._exhausted_bases`.

**Tunable to validate:** the `≥2` cyclic threshold (vs `≥1`) — `≥2` excludes one-shot collectibles
(count caps at 1) common on other games. Confirm against the no-regression eval; loosen only if ls20
needs it and the dev suite tolerates it.

**Pre-registered DoD (unchanged from §8 of the handoff):** (1) ls20 actions-to-L1 improves vs banked
across seeds (`scripts/ls20_speed.py`); (2) **no TUNE/HOLDOUT regression** on `eval_efficiency` at budget
6000 (and spot-check 30000) — this is the gate that killed v3; (3) firewall green / `augment=False`
byte-identical / banked 0.33 untouched; (4) unit tests for the exhaustion-split behavior + firewall.

## v4 RESULT (2026-06-27) — exhaustion key-split FAILED; pivoted to PHASE-GATED RETRY; ls20 crackable but a fundamental crack-vs-regression tradeoff

The exhaustion-triggered KEY split above failed for the same root reason as v3: *any* change to a node
key mid-exploration breaks the explorer's navigation (its learned paths reference the old keys), and
"exhausted" turned out to capture most of the graph, not just the gate. Measured: 1/10 ls20 seeds (worse
than banked's 5/10). So the whole **key-augmentation family is dead** — the explorer's identity IS its
key.

**Pivot — phase-gated action RETRY (keys never change).** The blocked goal-entry at the gate is a
self-loop edge (`next_key == key`, no reward). Mechanism: record, per `(key, action)`, the SET of hidden
phases it was seen blocked at; when the phase changes, re-open (delete the self-loop edge so it's untried
again) every blocked move not yet confirmed blocked at the new phase, at a configurable priority
`retry_tier`. A move blocked at all `counter_mod` phases is a confirmed static wall and never re-opened
(cap). Real navigation edges (`next != key`) and the node key are never touched → navigation is
byte-identical to banked. `augment=False` is the firewall. Implemented in
`src/arcagi3/history_augmented_explorer.py` (15 unit tests, incl. firewall).

**The knob is `retry_tier` (how eagerly the retry fires). Measured spectrum** (ls20 = `ls20_speed.py`
10 seeds; dev = `eval_efficiency` 6000, salience baseline TUNE sum_eff 6.26 / HOLDOUT 1.19):

| retry_tier | ls20 | TUNE sum_eff | HOLDOUT sum_eff | verdict |
|---|---|---|---|---|
| 0 (eager, high prio) | **6/10, seed-0 ~28% faster, seed-2 rescue** | 4.47 (tu93 L5→L0) | 2.05 | cracks, **regresses** |
| 1 | 5/10 (no crack) | 5.00 (lp85/lf52 down) | 2.05 | worst of both |
| MAX_TIER (last resort) | 5/10 (no crack) | 6.26 (clean) | 1.19 | safe, **no crack** |
| (stuck-gated variant) | 5/10 (no crack) | 6.21 (clean) | 1.19 | safe, no crack |

**Conclusion — fundamental tradeoff, not a tuning miss.** Only the eager (tier-0) retry cracks ls20
(rescues a seed banked never solves, solves another faster) — proving ls20's invisible-state wall **is**
crackable from pixels (a project first). But that eagerness diverts productive games (tu93 collapses),
and net across 16 games it is slightly negative. Every deprioritization that protects the dev suite also
fires too rarely to crack ls20. Root cause: **the explorer cannot distinguish the outcome-gating GATE
(worth retrying) from an ordinary WALL (waste)** — both are blocked self-loops, so any rule eager enough
to retry the gate also retries walls everywhere.

**Shipped state:** mechanism committed with `retry_tier = MAX_TIER` **default = safe / non-regressing**
(`augment=True` ≈ banked); firewall green; banked 0.33 untouched. The perception detector (mobility +
occlusion, v3 RESULT above) remains the solid win.

**Next sub-project (approved): SEMANTIC gate-identification.** Re-open a blocked move only when it is a
move *into a salient, goal-like object* (the gate), not into a wall — then the retry can be eager (tier
0) for the gate alone, cracking ls20 without diverting on walls. This needs perception to label "the cell
this move targets is a salient target object," and gets its own spec → plan → TDD cycle. This phase-gated
retry mechanism is the substrate it builds on.

## v5 SEMANTIC gate-ID design + INVALIDATING FINDING (2026-06-27) — CHECKPOINT before implementation

Design (approved): gate the *recording* of a blocked move, not just its priority. Add a blocked self-loop
to `_blocked_phases` ONLY at a "gate frame" — the avatar (the mobile components from `_flag_mobility`) is
adjacent to a salient goal-like object — and re-open recorded (gate) moves at `retry_tier=0` (eager).
Walls (no adjacent goal object) are never recorded → never re-opened → byte-identical to banked; only the
gate gets eager retry → cracks ls20 without the dev regression. Self-gating firewall unchanged.

**Pre-implementation verification KILLED the salience heuristic (good — caught before coding).** Drove
the known solution to the goal-adjacent cell and dumped objects near the avatar (engine (34,15) → grid
rows 15-19 cols 34-38). ls20's goal is a **framed structure of COMMON colors**, not a rare-color object:
`color 3 size 27 (8,32)-(16,40)` = frame/border (same color as the maze floor!); `color 5 size 38
(9,33)-(15,39)` = goal fill; `color 9 size 5 (11,35)-(13,37)` = inner marker (GoalColor 9, shared with
avatar body + maze). The ONLY rare-color object near the avatar is **the avatar's own head (color 12,
size 10)**. So the approved predicate ("adjacent to a salient *rare-color* object", `salient_click_targets`
priority ≤1) latches onto the AVATAR and misses the goal — it cannot work for ls20.

**What actually distinguishes ls20's goal:** a FRAMED block — a color-5 fill bordered by color-3 — of
medium size, distinct from the maze floor and the avatar. Candidate reworked "goal-like" detectors for
the next session (each must be verified to flag the goal but NOT walls, then run through the speed +
no-regression gates):
1. **Framed-structure detection:** an object/region enclosed by a border of a different color (the
   color-5 fill inside a color-3 frame). Most specific to "a target you enter".
2. **Medium non-floor / non-avatar object adjacency:** avatar adjacent to a medium (size ~8-64) object
   whose color is neither the dominant floor/wall color nor an avatar color. Looser; more false-positive
   risk — needs measuring how often the avatar is blocked beside such objects elsewhere.
3. **Target-cell-color approach:** distinguish the gate by what the blocked move runs INTO (goal-fill
   color vs wall/floor), using the avatar position from the mobility tracker + the move direction. Needs
   ls20's wall-vs-floor-vs-goal colors mapped first (open question: are walls a distinct color from the
   color-3 floor? the avatar moved freely through color-3 in the rot-tile window).

**Status:** no v5 code written (verification was read-only). The phase-gated retry mechanism (v4,
committed, `retry_tier=MAX_TIER` safe default) is the substrate. Resume by picking a reworked goal-like
detector above, verifying it on the gate frame with `scripts/ls20_rot_diag.py`-style introspection, then
TDD + speed/no-regression validation. Firewall + banked 0.33 untouched throughout.

## v5 RESULT (2026-06-27) — BUILT: semantic gate-ID cracks ls20 AND passes the 6000 gate (a first); but budget-fragile, and tighter discrimination overfits

Built the reworked detector (candidate 2, "medium non-floor/non-avatar interior object adjacency") as
`gate_only=True` on the phase-gated-retry substrate: a blocked self-loop is recorded into
`_blocked_phases` **iff** its move points at a goal-like object this frame (`_compute_gate_actions`:
the avatar — the mobile components — is adjacent, in the move direction, to a static component that is
medium-sized `GATE_MIN_SIZE..GATE_MAX_SIZE`, interior `>GATE_EDGE` from the grid edge, and a colour that
is not bg / the floor colour (largest component) / an avatar colour). With walls never recorded, the
retry can be **eager (`retry_tier=0`) for the gate alone**. `augment=False` byte-identical firewall and
the `gate_only=False` default are both unchanged; 20 unit tests green. Detector verified on real ls20
frames first (`scripts/ls20_gate_diag.py`): fires UP at the goal-entry and nowhere else on the solution
path; flags ONLY the colour-5 size-38 framed goal.

**Pre-registered DoD — all three PASS (the project first):**
1. *ls20 speed* (`ls20_speed.py` 10 seeds, budget 8000): histaug **6/10** vs banked **5/10** — rescues
   seed 2 (banked never solves it) and solves seed 0 ~28% faster (5778 vs 7961); regresses no seed
   banked solves. The crack v4's eager tier-0 also got, but now gated to the goal.
2. *No regression @6000* (`eval_efficiency`, same-session A/B): **TUNE sum_eff 6.26 == 6.26** (tu93 stays
   **L5**; the game v4 tier-0 collapsed to L0), **HOLDOUT 1.19 → 1.20** (ls20 **L0 → L1** cracked), 14/16
   games byte-identical. *This is the gate v4 could not pass — no v4 tier both cracked ls20 AND held the
   dev suite at 6000; v5 does.*
3. *Firewall*: `augment=False` byte-identical; `gate_only=False` / `retry_tier=MAX_TIER` default
   unchanged; banked 0.33 (TransferExplorer v13) never references the subclass.

**The honest bound (found by probing beyond spec): the win is BUDGET-FRAGILE, and closing the gap OVERFITS.**
- *Two calibrations were needed for the 6000 pass.* `GATE_MIN_SIZE` started at 6 → flagged tu93's size-8/9
  maze cells; tu93 has a cyclic glyph (phase flips ~hundreds of times), so the eager retry diverted it
  L5→L2. Raising the floor to **20** (the goal is a multi-tile *structure*, not a glyph) zeroed tu93's
  gate fires (raising the floor only REMOVES flags, so it cannot add a regression). This recovered tu93.
- *lf52 still diverges.* `eval_efficiency` 6000 shows lf52 L1 for both but **+6 actions** (169 vs 163, both
  eff-capped so the metric is unchanged) — the gate fires on lf52's medium colour-9/11/3 objects (sizes
  20–63, overlapping ls20's 38). At **budget 12000 offline** this compounds: salience reaches **L2**, gate
  stalls at **L1** (a real, budget-delayed regression the 6000 gate hides).
- *Tighter discrimination cannot fix lf52 without overfitting.* Census of gate-flagged objects: a "framed"
  test (`bbox` ring ≥60% one colour) does NOT separate them — lf52's colour-11/3 size-20 objects are framed
  too; even "framed by the FLOOR colour" fires on lf52 (its floor is unstable, largest component flips
  0↔5 frame-to-frame). The ONLY separator left is stacking **framed-by-floor AND size≥24** so that exactly
  ls20's single goal passes and the dev set's objects don't — i.e. tuning three appearance conditions to
  the visible TUNE/HOLDOUT split, the exact overfitting the split exists to reject.

**Verdict (validate-or-kill).** v5 is the **first config that both cracks ls20 and passes the pre-registered
6000 no-regression gate** — a genuine advance over v4, and proof the invisible-state wall is crackable from
pixels with a *targeted* (not global) retry. But it is **promotable only behind the opt-in flag, with a
documented higher-budget cost on dynamic games (lf52)**; it is NOT a universal clean win. The deeper lesson
sharpens v4's: the gate's defining property is **mechanical** (its blocked-ness depends on the hidden
phase), not **visual** — so *appearance-based* gate-ID narrows the gate-vs-wall ambiguity but cannot
robustly escape it without per-game overfitting. The only ambiguity-free signal is the mechanical one
(does un-blocking correlate with a phase change?), which can only be learned by retrying — the v4 cost.
**Shipped state:** mechanism committed, `gate_only=False` default (banked-identical), firewall green,
banked 0.33 untouched. Promotion (flip the submission config to `gate_only=True, retry_tier=0` to bank
ls20) is **user-gated** (a Kaggle submit). Code: `src/arcagi3/history_augmented_explorer.py`
(`_compute_gate_actions`, the `gate_only` branch in `_record`); harness: `scripts/ls20_gate_diag.py`.
