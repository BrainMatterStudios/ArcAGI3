# Off-Kaggle certification rig

Runs the patch-closure experiment geometry (28 competition-sim clones, 7920 s
per game, concurrency 28 — the frozen `GEOMETRY` of
`submission/_ab_patch_closure/patch_closure_config.py`) **off Kaggle**: an
exact-fidelity vLLM server on Modal (H100, free credits) + a local wave runner
on this Mac that drives the untouched rig machinery (`pc_driver.pc_main`, the
scored `benchmark_initial.pkl` solver, `behav_probe`, real
`CompetitionArcadeServer` over `environment_files/`) against that endpoint.

Serving fidelity anchor: `git show HEAD:scratchpad/taaf_scored_ref/setup_commands.json`
(the SCORED eval serving config). `offkaggle/test_offkaggle.py` parses that git
blob and fails if any constant here drifts from it. HF repo id
`vrfai/Qwen3.6-27B-FP8` was verified live 2026-08-10 (public, FP8 W8A8
compressed-tensors; the Kaggle dataset `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
is its mirror).

## Files

| file | what |
|---|---|
| `modal_vllm_serve.py` | Modal app: H100, pinned `vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6`, HF snapshot cached in a `modal.Volume`, the scored serve flags verbatim, bearer-token auth proxy, scale-to-zero + 6 h hard cap, `warm` and `smoke` entrypoints |
| `run_wave.py` | local runner: ONE arm (`shipped` or `base`) at the frozen geometry against the remote endpoint; writes `patch_closure_result.json` under `offkaggle/results/<ts>-<arm>/` |
| `test_offkaggle.py` | host-only tests (no network/GPU/modal): constants vs the HEAD blob, arm-env contracts, results layout vs `classify_shipped.py` |

## One-time setup

```sh
# 1. modal client + auth (any python >=3.9; NOT needed inside .venv)
pip install modal
modal setup                      # browser auth against your modal.com account

# 2. the bearer token secret (the ONLY auth on the endpoint)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # copy this
modal secret create arc3-vllm-token TOKEN=<paste-the-random-value>
export ARC3_VLLM_TOKEN=<the-same-value>      # run_wave/smoke read this
```

The secret name `arc3-vllm-token` and the env key `TOKEN` are load-bearing
(constants in `modal_vllm_serve.py`).

## Run order

```sh
# 0. local tests (always; seconds, no network)
.venv/bin/python offkaggle/test_offkaggle.py

# 1. warm the model cache — CPU-only container, pays the ~30 GB HF download
#    ONCE into the volume so no H100 minute is ever spent downloading
modal run offkaggle/modal_vllm_serve.py::warm

# 2. deploy the server (prints the web URL, shaped
#    https://<workspace>--arc3-vllm-serve.modal.run — the harness base URL is
#    that + /v1). Deploying costs nothing until a request arrives.
modal deploy offkaggle/modal_vllm_serve.py

# 3. smoke: /v1/models unauthenticated (mirrors pc_driver's probe), a 401
#    check with a WRONG token (proves auth is on), then the scored kernel's
#    exact smoke chat completion. First hit cold-starts the H100 (~10-15 min).
modal run offkaggle/modal_vllm_serve.py::smoke \
    --url https://<workspace>--arc3-vllm-serve.modal.run --token "$ARC3_VLLM_TOKEN"

# 4. one arm (~2.2 h wall each; run from the repo root, .venv python).
#    SHIPPED FIRST is not required here — each arm is its own process, and the
#    shipped process never loads duck_patches by construction.
.venv/bin/python offkaggle/run_wave.py --arm shipped \
    --base-url https://<workspace>--arc3-vllm-serve.modal.run/v1
.venv/bin/python offkaggle/run_wave.py --arm base \
    --base-url https://<workspace>--arc3-vllm-serve.modal.run/v1

# 5. the pre-registered verdict, unchanged classifier
.venv/bin/python submission/_ab_patch_closure/classify_shipped.py \
    offkaggle/results/<ts>-shipped/patch_closure_result.json \
    offkaggle/results/<ts>-base/patch_closure_result.json

# 6. teardown (idle scale-down happens anyway; this makes it immediate)
modal app stop arc3-vllm
```

Between-arm gap costs at most one idle window (10 min); running arm 2 within
that window keeps the container warm and skips the cold start.

## Cost honesty ($30/month credits, H100 ≈ $4/h list)

| step | GPU time | ≈ cost |
|---|---|---|
| `warm` (CPU container) | none | ~$0.05 |
| `smoke` incl. cold start | ~15 min | ~$1 |
| one arm at full geometry | ~2.3 h | ~$9 |
| **full 2-arm session** | **~5 h** | **~$20** |
| volume storage (30 GB, monthly) | — | low single $ |

So: **one full certification session per month**, plus smoke tests. There is
no headroom for a botched wave — run the local tests and the smoke first,
every time. The `warm` step exists so a re-launch after idle scale-down costs
~5 min of load, not a re-download. Delete the volume
(`modal volume delete arc3-hf-cache`) only if you're done for the month.

Cost guards built in: `scaledown_window=600` (scale to zero after 10 min
idle), an in-container watchdog that hard-exits after 6 h
(`ARC3_MAX_LIFETIME_S` at deploy time to change), and `max_containers=1` so a
request burst can never spin up a second H100.

## Auth model

- The duck harness natively sends `Authorization: Bearer $LOCAL_ANALYZER_API_KEY`
  on every analyzer call (`tool_agent._headers` → `build_headers`);
  `run_wave.py` sets that env var from `--token`/`$ARC3_VLLM_TOKEN`. No
  harness change needed — verified end-to-end against a local mock (824/824
  calls carried the bearer).
- `pc_driver._pc_serving_probe` hits `/models` with **no** auth header (byte-
  frozen file), so the Modal proxy exempts `GET /v1/models` (and `GET
  /health`) from auth. Everything else 401s without the token.

## Fidelity caveats (put these in the pre-registration)

1. **GPU**: H100 80 GB (Hopper, native FP8) vs the eval's RTX Pro 6000 96 GB
   (Blackwell, native FP8). Same FP8 hardware path — that's why not
   A100/L40S — but 16 GB less KV cache at 65536 ctx × 28 concurrent means
   more scheduler preemption, and SM90 vs SM120 kernel selection means
   numerics are not bit-identical. Serve SEMANTICS (flags, versions,
   sampling env) are identical and test-pinned.
2. **Throughput → deliberation count**: the 7920 s per-game box is wall-clock.
   A slower/loaded H100 yields fewer analyzer turns per box than the Kaggle
   GPU. Both arms suffer identically (same endpoint, same session), so the
   A/B delta is fair — but the absolute scores are NOT comparable to the
   banked Kaggle waves, which weakens the `banked_base_band` side-reading in
   `classify_shipped`. Read the same-session delta only.
3. **Network hop**: Modal web endpoints cap a single HTTP response at 150 s,
   then continue via 303 redirects; the harness's `requests.post` follows
   redirects (default), so long generations work, but per-call latency is
   higher than the Kaggle-local loopback and the 900 s analyzer timeout
   applies per redirect leg, not end-to-end.
4. **Concurrency 28 from one Mac against one H100**: kept at the geometry's
   28 — vLLM queues excess load harmlessly (requests wait in its scheduler;
   nothing errors). `--concurrency` exists to derate for diagnostics, but a
   derated wave records its true geometry and `classify_shipped` will refuse
   the pair (verified: a 60 s mock pair classifies INVALID with exactly the
   geometry reason and nothing else).
5. **flashinfer pin**: PyPI name is `flashinfer-python`; pinned `==0.6.6` to
   match the wheelhouse stamp. First deploy is the first live test of the
   image resolve (vllm 0.19.0 + torch 2.10.0 on debian_slim py3.12) — if uv
   fails to resolve, fix the image, never the pins.

## Known-good local evidence (2026-08-10)

- `test_offkaggle.py`: 7/7 pass.
- Both arms ran end-to-end against a mock brain at the full 28-clone/25-game
  shape (60 s plumbing box): shipped arm with the anti-apply guard + absence
  proof, base arm with `apply_all()` + stall-900 identity on every session;
  all analyzer calls carried the bearer; artifacts classified by the
  untouched `classify_shipped.py`.
- What is NOT yet tested (no Modal auth on this machine): the Modal deploy
  itself — image build, cold start, web URL, proxy under real concurrency.
  The smoke entrypoint is the gate for all of that.
