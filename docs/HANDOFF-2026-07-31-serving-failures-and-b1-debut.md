# HANDOFF 2026-07-31 — two silent serving failures found and fixed; B1 debuts (exploratory)

**Read this before acting on HANDOFF-2026-07-27 or the campaign memories.** It
corrects the record on the SFT debut and on the effect-memory arm. Nothing in the
campaign has yet produced a measured improvement over base duck.

## 0. Where we actually stand

| | |
|---|---|
| Public LB | **rank 100 / 1993, score 1.27** |
| What 1.27 is | base-duck farm draw #7 (Jul 29) — variance, not an improvement |
| Base distribution (identical bytes, n=8) | 0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69 → **mean 0.929, sd 0.195** |
| Field | #1 = 1.86, top-10 = 1.56, top-20 = 1.47, top-50 = 1.35, top-100 = 1.27 |
| A-track | **Gate 0 never passed.** Gates 1-3 not started. |
| B-track | B1 shipped ungated as an exploratory draw (amendment 6.1) |
| Measured deltas over base | **none, on any arm** |

Our rank sits at the top of the base distribution. Treat it as luck, not progress.

## 1. CRITICAL — the Duck-SFT 0.95 never served the adapter

`arc-agi-3-duck-sft` scored **0.95** on 2026-07-31, described as the run-8
fine-tuned adapter. It served the **base FP8 snapshot**. Two independent causes,
either one sufficient:

**(a) The model swap was a no-op.** `duck-sft.ipynb` cell 9 did:
```python
cmd_mod = command.replace("driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot", "/tmp/merged_sft")
```
That concatenated string does not occur in `setup_commands.json`. Owner and slug are
separate variables and the path is built by
`MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)`. Verified against
both local copies (`scratchpad/taaf_scored_ref/`, `submission/_adopt/taaf-src/`).
The `.replace()` returned the command unchanged.

**(b) The merge almost certainly crashed anyway.** The merge subprocess runs
`dequantize_fp8_inplace` → `strip_quantization_runtime` → `merge_and_unload`, the same
chain that fails in `serve-verify-k3` with
`RuntimeError: Promotion for Float8 Types is not supported`. Its return code was
printed but never checked, and `/tmp/merged_sft` absence silently selected the base.

0.95 sits dead-centre in the base mean of 0.929. **This is the 2026-07-11
"fine-tune RETRACTED" failure repeating** — and it happened despite
`_serve_verify_k3/serving_assert.py` having been built specifically to catch it. No
shipped notebook ever imported it.

**Fixed** in `build_duck_sft.py`: the swap now anchors on the real `MODEL_PATH =`
assignment (build fails if that anchor moves), the merge subprocess return code and
output completeness are checked, and the setup script hard-asserts that the served
path *is* `/tmp/merged_sft` — a missing merge now kills the run instead of quietly
scoring base. Verified by applying the swap to the real `setup_commands.json` and
`ast.parse`-ing the result.

## 2. CRITICAL — Gate 0 has never run with the corrected code

`arc-agi-3-serve-verify-k3` is in `ERROR`. Root cause: **the pushed notebook is
stale.** Its log prints `strip:` but never `dequant:`, and it ran `checkpoint-16`;
the current builder calls `dequantize_fp8_inplace` and defaults to `checkpoint-8`.
The `.ipynb` on disk (Jul 26 11:58) predated its own builder (Jul 26 21:49), and the
pushed version was older still — it predates the FP8 dequant fix (commit 6cf8f13).
So it stripped the FP8 scales unapplied, left weights in `float8_e4m3fn`, and
`merge_and_unload()` died promoting Float8 against Float.

**Fixed**: notebook rebuilt from the current builder (dequant present, `checkpoint-8`
default). **Not yet pushed or run** — Gate 0 is a 3-4h GPU job and was left for a
slot that does not compete with tonight's submission.

**Law added — a builder is not a build.** Every kernel must be rebuilt and re-pushed
before it is trusted; check `.ipynb` mtime against its builder's.

## 3. B1 effect memory — was a phantom arm, now real

`_duck_effects` v1 (armed by `auto_submit.py` for the 2026-08-01 slot) set
`EFFECT_MEMORY=1` and nothing else. Nothing shipped reads that variable:
`effect_memory.py` was never imported into the notebook and is absent from the
`taaf-src-hybrid` bundle; its only other consumer is `scratchpad/rl_gate/run_rollout.py`,
a local-harness path gated on `APPLY_EFFECTS_PATCH`. It would have run as plain base
duck under an effect-memory label.

**Fixed**: `build_duck_effects.py` rewritten on the `duck-patched` mechanism —
`effect_memory.py` is inlined into the customization-hook cell (after the bundle is on
`sys.path`, before `bm.run()`), with anchor assertions and cell-key preservation. The
cell hard-fails if the wrappers do not land. Proven twice: executed against
`scratchpad/taaf_scored_ref`, and the Kaggle commit run for v2 logged
```
[effects] verify _summarize_step_sequence wrapped: OK
[effects] verify _build_user_prompt wrapped: OK
[effects] verify toggle reads enabled: OK
[effects] verify block renders: OK
[effects] ARM ACTIVE — EFFECT_MEMORY=1
```

**Standing rule: an env toggle is not a shipped arm.** Every scored arm must print
positive in-kernel proof that its code applied.

## 4. Tonight's submission (armed)

`scratchpad/autosubmit_effects.sh`, nohup'd, log at `scratchpad/autosubmit_effects.log`.
Waits for 00:05 UTC, re-confirms kernel `COMPLETE`, submits **duck-effects v2**, then
confirms by unique marker `B1-effects-exploratory` rather than trusting the exit code.
Replaces `auto_submit.py`, which used a blind 90 s sleep (the pattern that ERRORed the
Jul 30 SFT submission) and would have died unhandled on a push failure.

**This is a deliberate deviation from B-TRACK-PROTOCOL §1/§3**, authorized by Ahmed
after the conflict was raised, recorded as amendment 6.1. The result is
**exploratory**: one draw cannot be separated from a distribution with sd 0.195.
Gate B is not satisfied or waived by it.

## 5. Next actions, in order

1. **Push and run serve-verify-k3** (rebuilt) for both `checkpoint-8` and
   `sft_adapter`. Until Gate 0 is green no adapter result means anything.
2. If Gate 0 is green, **Gate 1 → Gate 2** per `docs/A1-PROTOCOL-2026-08.md`. The
   paired `ab_driver` on the pinned 13-game holdout is the only instrument that can
   issue a GO — the public LB cannot (≈45 draws/arm for +0.10).
3. **Gate B for B1** on the A-selected config, regardless of tonight's number.
4. Review the unsubmitted **grid-burner** patch in `_duck_patched` (Gemini, Jul 28,
   +216 lines, 27 tests pass, never scored) — it is a candidate arm, not a result.

## 6. What was reviewed and left alone

`_sft_v2/`, `_sft_k3/prep_dataset_v2.py`, `corpus_v2/` (Jul 29, SFT v2 corpus prep) are
committed as-is — not audited in this pass, not on any gate path yet.
