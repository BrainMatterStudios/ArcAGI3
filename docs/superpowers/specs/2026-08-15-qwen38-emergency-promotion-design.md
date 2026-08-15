# Qwen 3.8 Emergency Promotion Design

**Date:** 2026-08-15  
**Status:** Approved in chat for design; implementation awaits written-spec review  
**Owner:** Ahmed Mobasher  
**Competition:** ARC Prize 2026 — ARC-AGI-3

## Objective

Promote the existing `duck-38` brain-swap arm to a fully attested Kaggle
submission and use the 2026-08-16 UTC submission slot instead of `duck-p3` if,
and only if, every integrity, serving, identity, and race gate passes.

The promotion is urgent because Qwen 3.8 is the first campaign lever with an
architecture-class measured effect. At the frozen 28-clone geometry it produced
21 levels versus Qwen 3.6's 17. The stored row means were 2.5291 versus 1.4872;
the campaign's prescribed per-source-game-max aggregation gives 2.7215 versus
1.6638. This is a directional screen, not a certified live mean: it is one wave
per model and the gains are concentrated.

## Scope

This change may:

- finish the existing local Qwen 3.8 snapshot download;
- add integrity verification and tests for the snapshot and notebook builder;
- create the private Kaggle model dataset;
- create and commit-run the private `arc-agi-3-duck-38` Kaggle kernel;
- attest the remote notebook bytes, dataset binding, kernel version, and script
  version ID;
- create a one-shot Qwen 3.8 submission runner with a safe `duck-p3` fallback;
- stop the armed `duck-p3` process only after Qwen 3.8 becomes fully eligible;
- consume the 2026-08-16 UTC competition slot with `duck-38` when all gates pass;
- monitor the submission and append its identity and score to the ledger.

This change will not alter the duck harness, prompt, sampling settings,
concurrency, context window, per-game budget, or patch stack. It will not push
the Git branch, deploy production software, publish a public dataset, or merge
anything.

## Frozen Arm

The scored arm is `submission/_duck_base/duck-base.ipynb` plus one behavioral
variable: the served model changes from `vrfai/Qwen3.6-27B-FP8` to
`Qwen/Qwen3.8-27B-FP8`.

The three coordinated substitutions are:

1. Kaggle dataset reference:
   `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` to
   `ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot`.
2. Setup-command `MODEL_OWNER` and `MODEL_SLUG` to the new private dataset.
3. Setup-command `SERVED_MODEL_NAME`, which also feeds
   `LOCAL_ANALYZER_MODEL_ID` and `INFERENCE_ANALYZER_MODEL`.

The probe-both wheel and environment mount changes are commit-only reliability
machinery already exercised by `duck-mem` and `duck-p3`; they do not execute in
the scored gateway path.

## Local Artifact Integrity

The current local snapshot is incomplete: `outside.safetensors` is still a
partial download. That file contains the language-model head, final norm, and
vision encoder, so the model is ineligible until it is complete.

Before upload, verification must prove all of the following:

- `snapshot_download` returns successfully with no `.incomplete` files;
- every filename referenced by `model.safetensors.index.json` exists;
- the snapshot contains `layers-0.safetensors` through
  `layers-63.safetensors`, `mtp.safetensors`, and `outside.safetensors`;
- every entry in `crc32.txt` matches a freshly computed CRC32;
- the snapshot contains the tokenizer, chat template, generation config,
  preprocessor config, and model config;
- total non-cache bytes exceed the existing 25,000,000,000-byte floor;
- cache metadata and incomplete files are excluded from the Kaggle dataset.

Failure of any check is fatal. A byte-count-only guard is insufficient.

## Kaggle Dataset and Kernel

After local integrity passes:

1. Create the private dataset
   `ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot`, preferring tar mode.
2. Query Kaggle until the dataset is visible and verify its reported size is
   consistent with the local snapshot.
3. Build `duck-38.ipynb` deterministically and record its canonical code-cell
   SHA-256.
4. Push `ahmedmobasher86/arc-agi-3-duck-38` with an RTX Pro 6000 accelerator.
5. Require the commit run to reach `COMPLETE`; `ERROR` is fatal for this slot.
6. Pull the exact remote version and require its canonical hash to match local.
7. Verify the remote kernel binds the Qwen 3.8 dataset and contains all three
   model-swap anchors, with no unrelated patch machinery.
8. Recover and pin the scriptVersionId from Kaggle's output metadata.

A commit-run `COMPLETE` proves packaging and mount viability, but not scored-run
serving. Serving evidence is the already completed Modal smoke plus the scored
setup's fail-loud model substitutions and real-model smoke test.

## Slot Arbitration

The Qwen 3.8 runner and `duck-p3` must never race independently.

- Until Qwen 3.8 is fully attested, `duck-p3` remains the fallback.
- The decision cutoff is 2026-08-15 23:30 UTC.
- If Qwen 3.8 is eligible at the cutoff, stop any exact
  `submit_p3_20260816.py` process, verify it has not claimed its marker, and arm
  only the Qwen 3.8 runner.
- If Qwen 3.8 is not eligible, leave `duck-p3` unchanged and do not attempt a
  rushed Qwen 3.8 submission.
- The selected runner must re-attest remote version, scriptVersionId, local
  hash, remote hash, required markers, kernel `COMPLETE`, and an unused UTC-day
  slot immediately before handoff to `submit_gated.py`.
- Use one shared arbitration marker so two different runners cannot both claim
  the day.

The target is 2026-08-16 00:01 UTC. `submit_gated.py` retains its ten-minute
settle guard, submission watch, never-played fingerprint detection, and ledger
append.

If a Qwen 3.8 submission returns `ERROR`, a `duck-p3` fallback is allowed only
when Kaggle's API positively confirms the daily slot remains available and the
fallback can still pass all gates inside the registered window. Transport
uncertainty is never treated as permission to double-submit.

## Evidence and Reporting Corrections

The Qwen 3.8 result must be reported with both statistics, labeled correctly:

- row mean over 28 clones: 2.5291 versus 1.4872;
- official rig aggregation, per source game max then mean over 25 games:
  2.7215 versus 1.6638.

The existing artifact's inherited `pre_registered_reading` describes a
different shipped-versus-patched experiment. It must not be cited as a clean
Qwen 3.8 preregistration. The contemporaneous field-sweep document and the
banked 0.2712 same-arm spread support a large-effect directional reading only.
No statistical-significance or live-mean claim is authorized from this one
wave.

## Tests and Verification

Before external upload:

- add snapshot-manifest tests covering missing files, incomplete files, CRC
  mismatch, and success;
- add builder tests proving deterministic output, exact dataset substitution,
  all setup-command replacements, and absence of unrelated behavioral changes;
- add a regression test that distinguishes the 28-row mean from the prescribed
  25-source-game-max mean;
- run the Qwen 3.8 runner in mock/dry-run mode.

Before submission:

- verify the remote dataset and kernel through Kaggle's read-only APIs;
- verify Modal serving identity remains `Qwen/Qwen3.8-27B-FP8`;
- run targeted Qwen 3.8 tests;
- run the full repository suite with `--import-mode=importlib`, because the
  default pytest import mode currently collides on the two `test_planner.py`
  modules;
- require zero failures. The existing single xfail and seven environment
  deselections remain acceptable.

## Stop Conditions

Do not create or submit the competition arm if any of these occurs:

- incomplete or checksum-invalid snapshot;
- private dataset missing, wrong-sized, or incorrectly bound;
- kernel commit `ERROR`, unavailable, or not `COMPLETE` by the cutoff;
- local/remote notebook hash mismatch;
- scriptVersionId ambiguity;
- served-model, dataset, or swap-marker mismatch;
- unrelated patch/config drift;
- failed targeted or full tests;
- UTC-day slot already used;
- inability to prove that only one runner is armed.

On a stop condition, preserve `duck-p3` as the slot fallback and report the
exact failed gate. Never weaken a gate to meet the clock.

## Post-Submission Decision Rule

The first live `duck-38` score is a transfer probe:

- `> 1.30`: individually exceeds the historical base band and promotes Qwen
  3.8 to the new campaign floor;
- `0.69–1.30`: inconclusive at one draw; keep Qwen 3.8 under evaluation because
  the offline effect is large, then obtain another attested draw;
- `< 0.69`: strong negative-transfer warning; inspect serving identity and
  behavior before any second draw;
- `0.00` with the never-played fingerprint or `ERROR`: infrastructure failure,
  not a behavioral verdict.

After a positive transfer, every future harness lever must be re-tested on top
of Qwen 3.8. The next two research priorities are the hidden-reasoning memory
capture audit and expectation-checked mechanical action queues.
