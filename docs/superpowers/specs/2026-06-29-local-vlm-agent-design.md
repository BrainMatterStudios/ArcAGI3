# Local-VLM Goal-Prior Agent — design (2026-06-29)

## Why
The campaign's W1 wall ("offline goal-ID unlearnable") was only ever tested with TINY CNNs. The
goal-legibility probe (2026-06-29) shows goal-legibility VARIES: cd82's goal+controls are clearly legible
to vision-reasoning (rotate/position a piece to a target shown in the HUD), tu93's is not. On legible games
our TransferExplorer wastes ~15x the actions a goal-aware reasoner needs (366 vs ~20 on cd82 L0) = the RHAE
efficiency headroom the squared private metric rewards. A LOCAL VLM (legal offline, runs on T4) as a
goal-prior was NEVER tested. This is the legal-offline analogue of the benchmark winner (online GPT-5.5 goal
prior + hypothesis testing). See memory [[arcagi3-goal-legibility-finding]].

## Hard constraints
- Eval: Kaggle T4x2 (15GB each), NO internet, 12h, offline. VLM weights must SHIP EMBEDDED (Kaggle dataset).
- A 7B VLM (~14GB fp16) fits one T4. Inference is SLOW (~seconds/generation) -> the VLM CANNOT be called
  every action (thousands of actions x seconds = blows 12h). It must be a PERIODIC goal-prior.
- Banked TransferExplorer (0.33) is the FIREWALL floor: VLM unavailable/uncertain -> pure TransferExplorer.

## Architecture
`VLMGuidedExplorer(TransferExplorer)` — coverage backbone unchanged (W3-safe), VLM adds an ADDITIVE bias:
1. **Backbone:** TransferExplorer (DENSE) drives every action — guarantees coverage + the 0.33 floor.
2. **VLM goal-prior (periodic):** called at level start AND when stuck (no progress in N actions), NOT per
   action. Input: the rendered current frame (+ a short interaction summary). Output (structured): (a) a
   goal hypothesis in words, (b) a target REGION (bbox) or object to steer toward, (c) a confidence.
3. **Additive steering (W3-safe):** the VLM's target region PROMOTES matching candidate actions to a high
   tier (try-first) WITHOUT removing any existing candidate -> coverage preserved, only ORDER within budget
   shifts toward the goal. Abstain (low confidence) -> no change -> pure TransferExplorer. This is the
   coverage-safe injection point (SalienceExplorer `_pick_from_batch` / a new top tier), the ONLY one that
   didn't regress in prior work when gated.
4. **Interactive refinement:** after deploying a VLM target, if progress (level-up or new states) -> keep;
   if no progress in M actions -> re-query the VLM with the updated frame (hypothesis refinement loop).
5. **T4 fail-safe:** reuse `learned_explorer._gpu_ok` pattern; no usable GPU / VLM load fail -> byte-identical
   to TransferExplorer.

## Model choice
Target eval model: a T4-runnable VLM, ~3-7B (Qwen2.5-VL-7B-Instruct or -3B; SmolVLM-2B; Moondream2 as a
tiny fallback). Dev on Mac MPS uses the smallest that demonstrates goal-reasoning (start 3B, scale to 7B).
Pick by the feasibility gate below.

## Build phases (resumable)
- **P0 FEASIBILITY GATE (do first):** install transformers; run a small VLM on Mac/MPS; feed it cd82's
  rendered frame + the control-probe montage; ask for (goal hypothesis, target region). PASS if it names the
  goal ("align/rotate the piece to the target") and a plausible region; FAIL if it can't reason about the
  frame at all -> the paradigm is dead for T4-sized models, stop. (Scripts: scratchpad/render_levelup.py,
  control_probe.py already produce the images.)
- **P1 OFFLINE LOOP:** build VLMGuidedExplorer with the VLM mocked (inject a fixed target region for cd82),
  verify the additive-steering reaches cd82 L0 in << 366 actions vs TransferExplorer. Proves the steering
  mechanism captures the efficiency before wiring the real (slow) VLM.
- **P2 REAL VLM IN LOOP (dev):** wire the real VLM (MPS), measure actions-to-levelup on the legible dev
  games (cd82, and survey which of the 25 are legible) vs TransferExplorer. Gate: strict-superset levels OR
  big efficiency win on legible games, no regression on illegible ones (firewall).
- **P3 T4 DEPLOY:** embed VLM weights as a Kaggle dataset; submission notebook loads on T4 (pin
  --accelerator NvidiaTeslaT4); confirm 12h-budget-safe (cap VLM calls/game); A/B on dev-as-proxy; submit
  only on a measured win (user-gated).

## Risks (kill-criteria)
- P0 fail: T4-sized VLM can't reason about frames -> dead (matches the survey's pessimism, but properly
  tested this time). - VLM too slow even periodic -> blows 12h. - Legible-game fraction on the HIDDEN set
  too low -> Kaggle-inert (efficiency only pays on completed legible levels). - Steering regresses illegible
  games -> must stay strictly additive + confidence-gated (firewall).

## Firewall
New file `vlm_guided_explorer.py` + opt-in flag; TransferExplorer + submission untouched until a measured
win. enable_vlm=False / no GPU / VLM load fail -> byte-identical to banked 0.33.
