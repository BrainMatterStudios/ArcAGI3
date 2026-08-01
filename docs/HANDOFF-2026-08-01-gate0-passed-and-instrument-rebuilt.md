# HANDOFF 2026-08-01 — Gate 0 PASSED, adapter serving fixed, measurement instrument rebuilt

**START HERE.** Supersedes `HANDOFF-2026-07-31-serving-failures-and-b1-debut.md` on
every point where they differ (notably: the merge is NOT lossy — that earlier
diagnosis was wrong and is corrected in §2). Read §0 and §7 first.

---

## 0. Immediate state — what is in flight right now

| item | state |
|---|---|
| **Submission 55160933** | duck-sft v4, submitted 2026-08-01 11:20 UTC, **awaiting score** |
| Branch | `winning/duck-patched`, **10 commits ahead of origin, NOT pushed** |
| Kaggle GPU quota | **EXHAUSTED** (30h weekly cap hit 2026-07-31 ~21:45 UTC). Ahmed expects reset 2026-08-01. |
| Background jobs | none running |
| Adapter training on corpus_v3 | **deliberately on hold** until 55160933 scores (Ahmed's call) |

**The one number that matters next:** what 55160933 scores. It is the first
scored run in the campaign that actually serves a fine-tuned model. Base
distribution on identical bytes is n=8, **mean 0.929, sd 0.195, range 0.69–1.27**.
A result well outside that range validates the distillation track; a result
inside it is *ambiguous*, not damning — one draw cannot resolve a small effect.

---

## 1. Standing rule (new, from Ahmed 2026-07-31)

**Every submission from now on must be a designed experiment aimed at climbing the
leaderboard.** No routine resubmissions, no base-distribution yardstick draws
unless the draw itself is the experiment. Before proposing any submission, state
the falsifiable hypothesis, why it could score higher, and how the outcome will be
read given one draw is noisy.

---

## 2. Gate 0 PASSED — and it corrects an earlier wrong diagnosis

`serve-verify-k3` v9, `checkpoint-8`, full 32768 ctx, 5536 target tokens, 4/4 rows:

```
base (adapter disabled)      NLL 0.8497
adapter attached (unmerged)  NLL 0.7329   gain +0.1168
merged                       NLL 0.7350   gain +0.1148   RETAINED 98.2%
merged relative gain = 13.51%   (A1 §1 bar: >= 2%)   -> MERGE GATE: PASS
in-kernel merge+save rc=0 in 390s; 14 safetensors shards
vLLM: engine init 145s, KV cache 26.21 GiB, "Application startup complete"
```

**Run-8's adapter learned real signal and the merge preserves it.**

**CORRECTION — do not repeat this mistake.** An earlier session claim that bf16
merge rounding destroyed ~2/3 of the fine-tune, and that native LoRA serving was
required, was **WRONG**. The weight-level `rel_err` (0.6355 / 0.6776 / 0.7484,
byte-identical across runs) is real but is *not* a proxy for functional retention:
rounding error is unstructured noise, the adapter's effect is a coherent low-rank
direction that survives it. A synthetic reproduction also showed fp32 *accumulation*
changes nothing — storage dtype binds, not accumulation. **Measure the quantity of
interest, not a proxy for it.**

Timing datum for the 9h budget: **merge + serve costs ~9 minutes**.

### Gate 0 remaining gap
`_probe_messages()` fix is **built but never run** (quota). The probe replays corpus
rows whose assistant turns carry `tool_calls` of type `"function"`; this vLLM's chat
parser rejects that (`pydantic: Input should be 'custom'`) → HTTP 400 *before any
generation*. So **well-formed generation from the merged model is UNVERIFIED**.
Re-run `serve-verify-k3` when quota returns. The duck harness builds its own
payloads rather than replaying corpus rows, so it does not traverse that path.

Also still unrun: the `sft_adapter` (step-15) arm. A1 §1 wants verdicts for both.
Build it with `GATE_CKPT=sft_adapter python build_serve_verify.py && kaggle kernels push`.

---

## 3. Why duck-sft shipped instead of B1 (independent research)

A 23-agent research sweep ran with **no access to CLAUDE.md, memory, or docs/** —
deliberately unbiased. Full result in the session transcript; load-bearing findings:

- **Capability is the binding constraint, not scaffolding.** Controlled: same
  scaffold, GPT-5.4 → 8 games solved / 41.29%; GPT-5.5 → 15 games / 58.12%. Public
  board: Opus 4.8 1.5%, GPT-5.6 7.8%, Opus 5 30.2%. **~20× model spread vs the
  ±0.1–0.3 that scaffold arms contest.**
  *Caveat, important:* that spread is between **base models**. It is NOT evidence
  that SFT on 435 samples moves a 27B. Do not over-claim it in support of the
  adapter track (an earlier framing did).
- **Scoring is quadratic in efficiency**, verified from `scorecard.py`:
  `min((baseline/actual)² × 100, 115)`, level weight = level number, uncompleted
  levels still count in the denominator. Mean game score at 1.0× human clearing 1
  level = 3.52; at 2.0× = 0.88. The field at ~1.26 ≈ "clears level 1 at ~1.7× human".
  Both depth and efficiency are steep; efficiency is bought by capability, not scheduling.
- **MOUSE lattice gap is real and non-recoverable from observation.** 9/25 games have
  camera scale > 1. Dead-coordinate fractions: tu93 **62.9%**, ka59 50.6%, lp85 40.6%,
  dc22 31.2%, m0r0 26.1%. Out-of-grid clicks are silently discarded **but still cost
  an action**, and the prompt never mentions scale or letterboxing. This is the
  strongest unexploited prompt-only lever found. Realized (as opposed to
  coordinate-space) dead-click rate is **unmeasured**.
- **B1 effect memory is undercut.** `board_changed` has **no control-flow consumers**
  (9 call sites, one advisory sentence), and the prompt already exposes
  `previous_frame`/`transitions`/`segmentation`.
- **The knowledge-wipe fix protects an empty box.** Over 6,673 assistant responses:
  `cross_level_notes` **0.00%**, `Goal model` 0.10%, `Action model` 1.95%. The
  bottleneck is the model not *writing* durable knowledge, not the harness deleting it.
- **Scored-rerun artifacts are NOT retrievable** (verified against duck-sft, which was
  scored 0.95 yet returns only its commit-run log: `TRUE_SUBMISSION=False`). So
  in-kernel telemetry cannot be read back. A submission yields exactly one number.

---

## 4. A/A noise floor MEASURED — two registered gates cannot fire

`scratchpad/rl_gate/aa_noise_floor.py` (no GPU; reads existing episodes). Four
same-config replicate pairs → **per-game RMS 0.707 levels, 7.019 score points**
(reproduces the research's independent ~7.0 estimate exactly). One pair swung
27.78 → 41.67 on an *identical* config.

On a 13-game panel at the registered 2 paired rollouts/game (SD of total = 1.80):

| gate | threshold | z | one-sided FP |
|---|---|---|---|
| A1 Gate 2 (adapter) | +3 | 1.66 | 4.8% — marginal, stands |
| A1 Gate 3 (doctrine) | +2 | 1.11 | **13.4%** |
| B-track Gate B | +2 | 1.11 | **13.4%** |

**Minimum detectable effect at 2σ is +3.6 total levels, not +2.** For +2 to reach
2σ needs **7 paired rollouts/game (~91 per arm)**. Recorded as **A1-PROTOCOL
amendment 7.1** with three options — raise to +4, go to 7 rollouts, or declare
exploratory. **Write the choice down BEFORE either gate runs.**

Two caveats recorded there, not buried: n=4 gives a 95% CI on RMS_levels of
**0.40–2.63**; and the A/A mean is **+0.50 levels / +4.17 score, not zero**, which
is equally consistent with an **order effect** (every `_a2`/`_b2` ran second).

---

## 5. The instrument — what IS and is NOT measurable today

The research's "no model-in-the-loop A/B exists anywhere" is **too strong**. Precisely:

| arm type | measurable now? | how |
|---|---|---|
| Prompt/harness (doctrine, effects, lattice, ledger) | **YES** | `ab_driver.py --upstream <any OpenAI-compatible endpoint>` |
| Adapter/checkpoint | **NO** | needs the real 27B on GPU, one vLLM session per arm |

All pieces are on disk: `run_rollout.py` accepts any `--upstream`; `ab_driver.py`
runs paired arms as **separate subprocesses** (so the `concurrency=16` objection
does not apply to it); all 13 holdout games present in
`scratchpad/holdout_arcint/environment_files`. The Claude shim proved the path.
**Honest caveat:** measuring on a proxy model and shipping on Qwen assumes transfer.

`ab_driver.py` now **randomises arm order per (rollout, game)** under `--seed` and
records `arm_order_log` — it previously ran base first every time, which would have
turned any order-dependent bias into a fake effect.

---

## 6. Data / assets changed today

- **`corpus_v3`** (`submission/_sft_k3/corpus_v3/`, `prep_dataset_v3.py`): 435 samples
  (379/56), **episode-isolated** (0 episode overlap, 4 of 5 val games entirely unseen),
  **100% kimi-k3** (v2 had silently mixed in 22 `kimi-k2.7-code` samples), leakage
  **measured not asserted**. Targets byte-identical to v1 on all 435 shared samples.
  **NOT uploaded to Kaggle yet**; training kernel still points at `corpus-v2`.
  Audit of Gemini's v2: its "0% validation episode overlap" claim was **false**
  (52.2% of val rows came from episodes in train) and its flail-pruning removed
  **1 turn across 457 samples**.
- **Claude-teacher corpus**: training AUTHORIZED by Ahmed 2026-07-31 (Anthropic never
  answered 3 requests; his call, his responsibility, recorded in
  `scratchpad/claude_corpus/DO_NOT_TRAIN.md`). 25-game sweep launched then **stopped
  at his request** after 21 calls on `ar25` (~$6.52 equiv). Resumable: start
  `claude_shim.py --port 8114`, then `sweep_driver.py`. Partial `ar25` has no
  manifest so it recaptures cleanly.

---

## 7. Next actions, in order

1. **Read 55160933's score.** It gates the adapter track.
2. **When GPU quota resets:** re-run `serve-verify-k3` (probe fix already built) to
   close the generation gap; then the `sft_adapter` arm.
3. **Decide and record A1 amendment 7.1's option** before any gate runs.
4. **Check the order-effect hypothesis** — cheap, and it invalidates paired designs
   if real.
5. **Adapter on corpus_v3** — only after (1). Needs corpus upload + a one-line
   kernel retarget. Neither costs GPU; can be staged any time.
6. **MOUSE-lattice disclosure** — the best prompt-only candidate; measure realized
   dead-click rate first, since the effect is currently a guess.

---

## 8. Laws added today (all paid for)

1. **An env toggle is not a shipped arm.** Every scored arm must print positive
   in-kernel proof its code applied. (B1 v1 shipped `EFFECT_MEMORY=1` with no code
   that reads it.)
2. **A builder is not a build.** Check `.ipynb` mtime against its builder before
   trusting a kernel. (Gate 0 was ERROR for days on a notebook predating its own fix.)
3. **Never let an unreadable measurement produce a verdict.** Three instances in one
   day: `val_leakage_percentage: 0.0` hardcoded; an NLL probe returning 0.0 from zero
   rows and "diagnosing" a dead checkpoint; a submit guard reading an empty status as
   failure and **forfeiting 11 hours of a slot**. Assert the measurement happened.
4. **Measure the quantity of interest, not a proxy.** Weight `rel_err` said the merge
   destroyed the fine-tune; NLL said 98.2% survived. NLL was right.
5. **Randomise arm order** in any paired design, and log the realised order.

---

## 9. Known-good invocations

```bash
# A/A noise floor (no GPU) — re-run whenever new replicate pairs land
.venv/bin/python scratchpad/rl_gate/aa_noise_floor.py --panel 13 --rollouts 2

# Paired harness A/B against any endpoint (order now randomised)
.venv/bin/python scratchpad/rl_gate/ab_driver.py \
  --games ff01,sy01,sq01,cs01,cx01,dm01,lo01,sc01,ic01,fs03,sk01,bd01,tp02 \
  --arms "base" "effects=EFFECT_MEMORY:1,APPLY_EFFECTS_PATCH:1" \
  --upstream http://127.0.0.1:8114/v1 --rollouts 2 --tag aug_w1 --seed 0

# Gate 0, per artifact (needs GPU quota)
GATE_CKPT=checkpoint-8 .venv/bin/python submission/_serve_verify_k3/build_serve_verify.py
kaggle kernels push -p submission/_serve_verify_k3

# Claude teacher sweep (resumable)
.venv/bin/python scratchpad/claude_teacher/claude_shim.py --port 8114 --model opus &
.venv/bin/python scratchpad/claude_teacher/sweep_driver.py &
```

---

## 10. Uncommitted / unpushed

- Branch is **10 commits ahead, never pushed** — Ahmed's approval gate on shared state.
- Still-modified and deliberately untouched: `submission/_duck_shadow/*` (closed EWM
  track, predates this session).
