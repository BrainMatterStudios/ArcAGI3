# arc3-flashnext-flight — FLIGHT: STOCK duck × Flash-Next NVFP4 (submittable)

The competition-submittable kernel for the model-swap arm (2026-08-30). STOCK
duck harness (anim-20260807 bundle, the bytes that flew 1.55) served by
sonpham's **Qwen3.8-Flash-Next NVFP4** on ONE RTX Pro 6000. No grafts, no 27B
mount — the model axis is the experiment.

Base: `submission/_flashnext_smoke/build_flashnext_smoke.py` (v4), whose
kernel PASSED the 25-game stock read on Kaggle — levels/game **1.16**,
zero-level **4/25**, boot **950 s** on rung `gcp_exact`
(`submission/_flashnext_smoke/results_v4/`). The flight builder reuses that
build verbatim via runpy (gate ASSEMBLE, BOOT ladder, duck-env exports incl.
the BUNDLE_DIR re-pin + sonpham path pruning, hard attestation) and applies
only the flight deltas below.

## What runs

* **Commit path** (`kernels push` → commit run): v12-style small smoke — 3
  games `vc33-5430563c / sb26-7fbdac44 / tn36-ef4dde99`, per-game cap
  2,400 s, `soft_end = NOTEBOOK_START_EPOCH + 7,200 s` (3,600 s of smoke +
  boot allowance). Phase telemetry + `flashnext_flight_smoke.json` verdict
  (`SMOKE-OK` / `SMOKE-FAIL`); grep the log for `FLASHNEXT FLIGHT SMOKE:`.
* **Scored rerun** (`KAGGLE_IS_COMPETITION_RERUN`): the v12 path untouched —
  full competition games from the gateway, `n_passes 1`, gateway scorecard is
  the record, dummy-parquet branch preserved for non-submission runs. Phase
  telemetry does NOT run in this branch (begin/end + play seam gated on
  `run_as_submission`).
* **9 h time guard** (scored path only, after boot, inline — same arithmetic
  as `graft_throughput.time_guard_per_game_s`, no graft installs):
  `elapsed = time.time() - NOTEBOOK_START_EPOCH`, then
  `bm.solver.max_runtime_s_per_game = max(6600.0, (32400 - elapsed - 300) / 4)`
  — 4 waves at concurrency 28 over ~110 games, printed as
  `flight time guard: ...`. With the measured 740–1,550 s assemble+boot this
  lands at ~7,600–7,900 s/game; the 6,600 s floor protects against a slow
  boot ever starving the waves.
* **Server watchdog** (their vLLM-dev has none): background thread polls
  `GET {base}/v1/models` every 60 s; on 3 CONSECUTIVE failures it relaunches
  the server with the SAME argv/env via `start_server` (their
  `serving_env()`; `ARCH_OVERRIDE` still holds the booted rung). Events in
  `/kaggle/working/flashnext-watchdog.log`; the teardown shim sets the stop
  event first so a clean shutdown is never "rescued".
* **Attestation stays HARD**: `architectures ==
  ["Qwen4ExpForConditionalGeneration"]`, `model_type == "qwen4_exp"`, 206
  shards > 186 GB, live greedy decode through the analyzer endpoint — a
  scored run that silently served nothing dies before spending the slot.

## Identity / attestation procedure (svid capture post-commit)

1. Build + validate locally; record the builder's **code-cell sha256** (also
   asserted by the validator's freshness check).
2. Push (on Ahmed's go only). After the commit run completes, capture the
   **scriptVersionId (svid)**: `kaggle kernels output` /
   `/api/v1/kernels/output?version_label=vN` storage URLs embed it, and
   `/api/v1/kernels/pull?...&version_label=vN` returns the committed code —
   hash it and confirm it matches the local build (the v8-harvest re-attest
   pattern, `scripts/submit_v8harvest_20260829.py`).
3. Pull the commit log and verify: `attest: config sha256 ...`,
   `attest: decode fingerprint ...`, `attest: OK`, boot rung (`gcp_exact`
   expected; `reduced` = DEGRADED-CONFIG, do not submit without discussion),
   and `FLASHNEXT FLIGHT SMOKE: SMOKE-OK`.
4. Submit that exact version; record svid + submission id in the campaign
   log. At scoring time the same attestation cells run again — identity is
   enforced in-run, not just at commit.

## Pre-registered reading rule (fixed before flight)

**This arm draws from a DIFFERENT distribution than the 27B series** — the
base-27B band (n=11, mean 0.965, band **0.69–1.30**) does not directly bound
it. Record, per the first draw:

| Live read | Interpretation |
|---|---|
| in 0.69–1.30 | still informative for a new arm: no live evidence the swap moves the mean; a second draw would be needed to separate it from the 27B distribution (CV ~0.17–0.20 ⇒ one draw is underpowered) |
| **≥ 1.45** | model-swap POSITIVE live (clears the 27B band + the duck-38 draws at 1.29/1.45) |
| **> 1.74** | re-banks the best live result (prior best 1.55) |
| **< 0.5** | serving/wiring failure SUSPICION, not a model verdict — pull the kernel log first (boot rung, watchdog restarts, attest lines, gateway wait) before reading it as capability |

Known drags to weigh in the read: the local smoke showed an **efficiency
drag** (local per-level efficiency ~0 on slow L1s — Flash-Next spends more
actions per level than the human baselines on some games), but **levels/depth
drive the completion cap**, and the smoke's levels signal (1.16 vs 27B's
0.84–1.04, zero-level 4/25 vs 8–9/25) is what this flight tests live.

## Files

* `build_flashnext_flight.py` — builder (runs the smoke build via runpy,
  applies the flight deltas; writes the .ipynb + kernel-metadata.json)
* `arc3-flashnext-flight.ipynb` — the kernel (18 cells)
* `kernel-metadata.json` — id `ahmedmobasher86/arc3-flashnext-flight`,
  NvidiaRtxPro6000, internet off, competition source attached, same 6
  datasets as the smoke, **no model_sources**
* `validate_flashnext_flight.py` — structure + per-cell syntax
  (await-wrapped) + metadata + freshness + pyflakes; asserts the scored
  branch strings, the time-guard line, no graft installs outside comments,
  the 3 smoke games and the 2,400 s cap

## Run

```sh
.venv/bin/python submission/_flashnext_flight/build_flashnext_flight.py
.venv/bin/python submission/_flashnext_flight/validate_flashnext_flight.py
# push (ONLY on Ahmed's go — this kernel is submittable):
cd submission/_flashnext_flight && python3 -m kaggle kernels push -p .
```
