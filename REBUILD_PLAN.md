# ARC-AGI-3 Agent — Rebuild Plan (structured world-model + planning)

> Goal: move from the current training-free **explorer** (1→9 real levels this session,
> at its architecture ceiling) to a **structured world-model + planning agent** that
> infers and *achieves* goals on unseen games. Executed via orchestrated agent teams with
> a judge gate and model tiering by complexity.

## Why a rebuild (root-cause from 6 measured experiments)

The current agent's two wins were **bug fixes** (distractor-masking, click-salience); six
subsequent levers (tier-aware exploration, volatile masking, go-explore progress-prio,
multi-color avatar, occupancy+A* nav, goal-inference) were all neutral-or-regressing. The
clincher: goal-inference *correctly identified* rewarding colors on m0r0 yet levels didn't
improve — **the bottleneck is goal ACHIEVEMENT (planning under per-level mechanics), not
identification**, and nav/exploration tweaks keep regressing the fragile sokoban (push).
Conclusion: incremental tweaks are exhausted; we need genuine scene understanding +
forward modeling + planning. That is multi-component, so we orchestrate it.

## Target architecture (components)

Each component is independently designed, implemented, judged, and measured. Complexity
tier drives the model used (see Orchestration).

1. **C1 — Object tracking / persistence** (tier: medium)
   Track objects across frames with stable identities (color+shape+motion), so we can say
   "object X moved/appeared/vanished/changed" rather than diffing raw pixels. Foundation
   for everything below. Interface: `objects(frame) -> [TrackedObj{id,color,cells,bbox}]`,
   `match(prev_objs, cur_objs) -> correspondences + events`.

2. **C2 — Causal event extraction** (tier: high)
   Separate **agent-caused incidental change** (avatar moving over floor/maze) from
   **game events** (door opens, item collected, switch toggled, counter ticks). Key idea:
   attribute changes to the avatar's footprint/trajectory; residual changes = events.
   Output: a typed event stream per step. Fixes the "98%-progress noise" failure.

3. **C3 — Affordance model** (tier: high)
   Per object color/type, learn the avatar's interaction effect: blocks / passable /
   pushes / collects / toggles / harms. Learned online from (contact → event) pairs.
   Interface: `affordance(color) -> {BLOCK,PUSH,COLLECT,TOGGLE,GOAL,HARM,NONE}` with
   confidence. Enables planning and push-safe navigation.

4. **C4 — Forward world model** (tier: high)
   A learned/structured transition model: given state + action, predict next state using
   avatar motion (C-existing) + affordances (C3). Cheap, object-level (not pixels). Enables
   lookahead planning without burning real interactions. Start rule-based from affordances;
   optionally a small CNN later (GPU available at eval).

5. **C5 — Goal inference** (tier: high)
   On each level-up, capture the *event that immediately preceded reward* (from C2, robust
   to the level-change masking that defeated the naive version) and form a goal hypothesis
   (e.g. "make all COLLECT objects vanish", "PUSH block onto GOAL marker", "avatar reaches
   GOAL cell"). Maintain/refine hypotheses across levels.

6. **C6 — Planner** (tier: high)
   Given the world model (C4) + goal hypothesis (C5) + occupancy/affordances, plan an
   action sequence achieving the goal (A*/BFS over the forward model, or subgoal
   decomposition: reach key → reach door; push block to target via sokoban-safe pushes).
   Reuses `spatial.py` (occupancy+A*, already built+tested) push-safely (exclude/most movable
   objects from walls; plan pushes explicitly).

7. **C7 — Integration policy** (tier: medium)
   A new `WorldModelPolicy` that orchestrates: explore to learn affordances/goal → plan →
   execute → fall back to the proven graph explorer when uncertain. MUST NOT regress the
   current agent: kept behind a selector; promoted only if it beats the current agent on
   the measurement protocol.

## Measurement protocol (the judge's ground truth)

Every component/integration is accepted only if BOTH hold:
- **No regression:** local suite 8/8 (27/27 levels) + push 3/3 + `pytest` green.
- **Real-game gain (or neutral for foundations):** A/B at fixed budget on a held-out set
  {tu93, ls20, m0r0, su15, vc33, re86, wa30} via `run_one`/NORMAL mode; integration must
  beat the current agent's level total at matched budget, no game regressed >0.
Reverts are mandatory on regression (this session's discipline).

## Orchestration (agent teams + workflows + judge gate + model tiering)

Two workflows (Workflow tool). Model tiering: **opus** = design, judging, high-complexity
implementation (C2–C6); **sonnet** = medium implementation (C1, C7) + verification;
**haiku** = mechanical edits/test scaffolding/log triage.

- **W1 Design** (run first, doesn't need the submission score): per component, fan out
  2–3 independent design proposals (opus), an adversarial judge (opus) scores
  {correctness, generality, no-regression-risk, feasibility} and picks/synthesizes the
  blueprint. Output: `rebuild/design/<Cx>.md` blueprints + a build order.
- **W2 Implement** (run after results review + design): pipeline per component in
  dependency order — implement (model by tier, `isolation: worktree`) → adversarial
  judge/review (opus) → measurement agent runs the protocol → gate: keep only if judge
  passes AND measurement shows no-regression(+gain). Failed components are reverted and
  re-queued with judge feedback. Final integration agent wires W2-passed components into
  `WorldModelPolicy` behind the selector and runs the full protocol.

## Sequencing

1. (now) Write this plan + author W1/W2 scripts.
2. (now) Run **W1 Design** → judged blueprints.
3. (on submission score) Review the real leaderboard result with the user.
4. Run **W2 Implement** in build order: C1 → C2 → C3 → (C4, C5) → C6 → C7, each gated.
5. Re-measure; if `WorldModelPolicy` beats current → promote + resubmit.
