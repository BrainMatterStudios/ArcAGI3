# External-repo pattern extraction — 2026-07-27

Companion to the 103-agent web research (memory: `arcagi3-research-2026-07-27-external-landscape`).
Four repos cloned to `scratchpad/external/` and mined line-by-line by two Explore agents.
This file records the portable mechanics and the local trace evidence that reshaped B1.

## 1. Trace evidence (our own episodes, mock runs excluded)

Latency-filtered sweep of `scratchpad/rl_gate/episodes/` (60 dirs; avg latency < 1s = mock):
- "X is not valid right now" retry loops: **0 in all real episodes** (the alarming 3,010-count
  was `smoke_sq01_230519`, a ReplayMockLLM run, latency 0.001s).
- "You have not acted yet" non-acting turns: **0 in every k3_sweep_* / control_ft09_27b run**
  (current config family). The 333/95/69 counts were old instruct-teacher and ladder
  experiments (`teacher_cd82_instruct`, `ladder_sb26_k27code`, `ladder_ft09_k27code`).
- Malformed tool-call JSON: **0 everywhere**; markup-recovery already handles the common case.
- Engine already rejects illegal actions BEFORE execution at zero scored-action cost
  (solver.py `step_env`: `action.id.value not in available_actions` → error payload), and
  MOUSE row/col are clamped to 0-63 pre-engine (`_normalize_actions`).

**Conclusion: Reki's JSON self-repair / legal-action masking / bounds checks solve problems
our stack does not have.** B1 is re-scoped to the one genuinely missing capability: a
grounded effect memory + click prior (below).

## 2. StochasticGoose (`ARC3-solution/custom_agents/action.py`)

- Only learning signal: binary frame change via exact `np.array_equal` on one-hot frames
  (L393-401). Free to compute; no CNN needed to *measure* it (CNN only predicts it).
- Dedup: `md5(frame.tobytes() + action_idx)` set + 200K deque (L246-253) — unique
  (state, action) pairs only.
- Sampling: 5 action logits + 4096 coord logits, sigmoid (not softmax), coord mass divided
  by 4096 so all clicks together weigh like one discrete action (L204-205), availability
  mask −inf (L177-198).
- Wipes buffer + re-initializes CNN and Adam on every level transition (L341-354); the
  code's own TODO doubts this. Cross-level persistence is an open improvement axis.
- **Port**: per-step `board_changed` bookkeeping → (action → tried/changed) table +
  64×64 click-productivity grid. No gradients, no GPU.

## 3. Sensi (`sensi/agents/sensi_llm.py`, 1126 lines, SQLite persistence)

- Three-tier fact ladder: `guesses` → `figured_out` (LLM-curated promotion) → `fact`
  (programmatic gate — but gated on an **LLM-judge "sense score" ≥ 8** that sees only the
  agent's own text, never the frame → the hallucination cascade; the paper itself names
  "programmatic pixel comparison" as the missing fix, sensi-paper.tex:761,793).
- `losing_actions_seqs` table: action sequences ending in GAME_OVER, persisted per game.
- Determinism (pass@10 = pass@1): the Actor is a thin map from curated text lists to one
  action and never sees the frame — all stochasticity quarantined in perception.
- **Port**: the ladder maps onto our two-tier ledger; the lesson is that promotion must be
  anchored on measured signals (frame diff, score delta), never LLM self-agreement.

## 4. baseline1 (`arc-3-agents-baseline1`, use `papers/paper02/agents/ewma_sv_v1.6/`)

- Four fixed interfaces the model must fill: `world_model_engine(state, action)`,
  `initial_state_reconstruction`, `state_renderer` (64×64 int16), `planner(state)`.
- **Verification = exact replay**: re-simulate every recorded step; rendered frame must
  equal observed frame exactly AND predicted status must equal observed status
  (`verify_world_model.py`); planner separately verified to complete solved levels
  (`verify_main_planner.py`). 180s timeout doubles as MDL pressure ("model too slow →
  refactor"). Render-override patch hook = warning-level "modeling debt" escape valve.
- Anti-cheat is prompt-load-bearing: "strictly forbidden to load real game observations
  into the world model engine" (else it becomes a lookup table).
- MDL prompts: "Assume, by default, that the real game mechanics are simpler than your
  current explanation… compress the ontology and dynamics."
- Verification is agent-run (prompted), not controller-enforced. Whole outer loop is
  Codex-CLI + on-disk workspace — must be re-implemented for our harness; the verifier
  scripts themselves are plain Python.
- Ablation (arXiv 2607.15439): verification was the strongest component; textual variant
  beat the flexible executable one.

## 5. AERA (`aera-arc3-paper/agent.py`, 405 lines)

- 4-field step protocol: `HYPOTHESIS / UNCERTAIN / NEXT_ACTION / REASON`.
- **Paper-vs-code discrepancy**: the advertised entropy gate (UNCERTAIN length < ~50 chars
  ends EXPLORE) is only *logged*; the shipped gate is a keyword match on REASON
  ("HIGH"/"confident"/"certain") plus a flat 15-step explore budget. If porting "the
  gate," implement the threshold for real.
- Linear EXPLORE → VERIFY(≤3 targeted actions) → PLAN pipeline; docstring claims loop-back
  on revision but code proceeds regardless.

## 6. What this feeds

- **B1 effects pack** (`submission/_duck_effects/`): harness-measured effect memory
  (action → tried/changed per level; 64×64 click grid) surfaced to the model each turn —
  Goose's signal + Sensi's grounding fix, at Duck-lesson-compatible weight.
- **D4 phase-gate doctrine block**: AERA's explore-before-plan discipline as prompt text
  (the real gate, not the keyword one).
- **B4 (later, bigger)**: baseline1-style replay verification of a persistent world model,
  verification-first per the ablation.
