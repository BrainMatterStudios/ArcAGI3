# NEXT-SESSION HANDOFF (2026-06-28)

Fresh-session instructions to resume the ARC-AGI-3 campaign. Branch `winning/mechanic-model-search`,
all committed (NOT pushed). Interpreter `.venv/bin/python`; always prefix `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src`.
Never run a bare full `pytest` (one test hangs). Banked submission = **TransferExplorer = 0.33** (untouched).

## 0. FIRST THING: the build plan is READY
The `next-paradigm-research` workflow (run `wf_eab5ed08-c8d`) COMPLETED. Full plan:
**`docs/superpowers/specs/2026-06-28-next-session-build-plan.md`**. 18 candidates → 2 survivors; honest
verdict: **no survivor escapes the SPIRIT of W1-W4**; both research arms are odds-on to join the graveyard,
so run them as CHEAP KILL-TESTS only. Sequence:
- **Day-1 AM — BET 2 (cheapest, decisive, NO training):** oracle-reorder ceiling probe for the click-affordance
  lever. Pure replay over existing dev traces: per COMPLETED level on L0/L1-walling games, count pre-levelup
  no-op clicks a perfect change-affordance oracle would reorder after the level-up; convert to weighted-RHAE
  (quadratic). KILL if gain < ~0.05 (the upper bound of the whole arm; likely thin = the "dev-efficiency
  Kaggle-inert" reduction that killed CAIPrune 0.28). Only if > ~0.15 build the within-level Conv2d head. Files:
  `agent.py` L49-60 (salient_click_targets), `cai_prune_explorer.py` (51% no-op audit), `spatial_value_explorer.py`
  L45-60 (the click-VALUE head that DIVERTS — stay an action-space prior, never a value signal).
- **Day-1 PM → Day-2 — BET 3 (settles W1 weak-vs-strong):** held-out-family synthetic feasibility probe.
  USE THE EXISTING `src/arcagi3/synthgen/{canvas,genres}.py` generator: 3-4 mechanic families (gate/switch/
  collect/reach), role-randomized so colour/shape carry zero surface signal; train an in-context GRU /
  small decision-transformer over (frame,action,frame-change) history, LEAVE-ONE-MECHANIC-FAMILY-OUT; measure
  held-out-family level-up rate vs random (synthetic→synthetic, zero sim-to-real gap). KILL if ≈ random
  (confirms W1-STRONG: in-context inference can't generalize across families even synthetically → meta-RL
  hypothesis class closed). Only on a clear margin proceed to leave-one-REAL-game-out.
- **On 2026-06-30 drop — BET 1 (rank-1 ceiling, gated on the external event):** ADOPT + HARDEN the open-source
  #1 winner into the hardened harness. Classify their core into a Learner slot (policy → `Learner.act`;
  exploration heuristic → abstain-default add-only; online model → T4 fail-safe path). Reproduce their number
  on shared dev games FIRST, then graft with banked-0.33 as the fallback branch. Day-1 kill-test: does it clear
  L0/L1 on games TransferExplorer cannot? If yes → harden against their preview overfit-collapse (12.58→0.25);
  if it walls identically → adopt-inert, ship banked 0.33. **Hold dual-T4 capacity in reserve for fast integration.**

## 1. State of play (read the memories)
- `arcagi3-rhae-headroom.md` — the efficiency line + the 7-discriminator battery + the help-vs-derail wall.
- `arcagi3-eval-image-facts.md` — GPU (T4 via `--accelerator NvidiaTeslaT4`; P100 default is cap-6.0 broken)
  + the LEARNED HARNESS + the 3 learned learners (all characterized-killed).
- `arcagi3-history-augmented-killed.md`, `arcagi3-ls20-mechanic-ground-truth.md` — the ls20 invisible-state line.
- `arcagi3-research-findings.md`, `arcagi3-scoring-and-transfer.md` — the full graveyard + scoring.

## 2. The WALLS (do not rebuild anything that reduces to these)
- **W1** offline goal-ID unlearnable (CNN LOGO-AUC 0.457). Corollary: any learned model, WHEN CONFIDENT,
  ACTS — with available signals it is confidently-wrong enough to divert/regress.
- **W2** within-game reward starved (2-5 edges/game); frame-change/effect MISLEADS.
- **W3** frontier-reorder (promote/demote/prune) corrupts coverage at finite budget.
- **W4** help-vs-derail discriminator UNLEARNABLE in the deployment regime (help & derail cases have
  identical features) → cannot safely deploy the two real capabilities (ls20 crack, ChainMacro 21x efficiency).

## 3. The DURABLE ASSETS (build ON these)
- **Hardened learned harness** — `src/arcagi3/learned_explorer.py`: `LearnedExplorer(TransferExplorer)` with a
  **T4 fail-safe** (`_gpu_ok` runs a real CUDA op) + an abstain-default `Learner` interface (`see/observe/
  act/noop_set`, `learn_mode` propose|prune). The FIRST learned arch that cannot catastrophically collapse
  coverage (tu93 stays L5). `enable_learn=False` → byte-identical to banked. **Any new model slots into
  `Learner.act()`** (or the June-30 winner's model). 9 tests in `tests/test_learned_explorer.py`.
- `learned_dynamics.py` (DynamicsLearner: effect model, propose+prune — killed) and `value_exploit_learner.py`
  (distance-to-reward value exploit — killed) are firewalled characterized NEGATIVES; reuse `_build_net`.
- `RHAE headroom oracle` (`scripts/rhae_headroom.py`): 35x within-game efficiency headroom (uncapturable so far).
- ls20 interaction-HISTORY latent recovery (`history_augmented_explorer.py`) — the one signal that escaped W1.
- `chain_macro_explorer.py` (gated/typed_tripwire), `scripts/learnable_disc_probe.py`,
  `scripts/chain_recurrence.py`, `mechanic_inference.py`, `events.py`.

## 4. How to run the standard gates
- offline level A/B vs banked: drive `TransferExplorer(**DENSE)` vs the new policy on `environment_files`
  games (DENSE = `seed=0, trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256`);
  see the scratchpad A/B scripts pattern. Watch lp85 (deep levels), tu93 (no-collapse), cd82/vc33/lf52/ar25
  (regression-prone).
- no-regression eval (live API, slow): `scripts/eval_efficiency.py 6000 <policy>` — TUNE/HOLDOUT split.
- firewall: `enable_*=False` MUST be byte-identical to banked (the 0.33 floor).

## 5. Honest verdict carried in
The reproducible offline non-LLM paradigm caps near 0.33 (non-learned AND learned-with-available-signals both
exhausted this session). Winning needs a SIGNAL/METHOD not in available observations — the workflow hunts for
it; if it finds none that escapes W1-W4, the realistic path is **adopt+harden the June-30 open-source winner**
into the hardened harness (pre-build per the `june30-adopt` lens). Do NOT re-run graveyard levers expecting
Kaggle movement. Promotion (any submission) is USER-GATED. Use the superpowers flow (brainstorm if redesigning;
TDD for code; validate-or-kill against the no-regression + headroom gates).
