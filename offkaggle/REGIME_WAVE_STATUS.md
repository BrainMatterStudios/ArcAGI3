# Regime wave runner — status (2026-09-02)

Off-Kaggle runner that drives the UNMODIFIED June-stock duck harness against the
Modal Flash-Next endpoint at the live geometry and extracts per-call telemetry,
so the `keith` (public V14) and `flight` (our Flash-Next flight) analyzer
configurations can be compared by mechanism, one knob at a time.

Files (all NEW, nothing tracked was edited):

| file | what |
|---|---|
| `offkaggle/run_regime_wave.py` | the runner (`--arm keith|flight`, `--base-url`, `--games`, `--out`, `--dry-run`) |
| `offkaggle/test_run_regime_wave.py` | 13 host-only tests (no GPU/Modal/internet), all passing |
| `offkaggle/REGIME_WAVE_STATUS.md` | this file |

Status: **built, tested, dry-run proven. NOT yet run against the endpoint**
(per instruction: wait for the smoke to pass). No request was sent to
`a-m-mobasher--arc3-flashnext-serve.modal.run` by this work.

## 1. How to launch

One process per arm (the harness reads `LOCAL_ANALYZER_CONTEXT_WINDOW` /
`LOCAL_ANALYZER_MAX_OUTPUT` at import time; the runner asserts the imported
constants match the arm). Run the two arms back to back so the second one
skips the cold start (15-min idle scale-down).

```sh
cd /Users/ahmed/Documents/ArcAGI3

# 0. host-only tests (~12 s) and the no-network dry run (~30 s, 25 games)
.venv/bin/python offkaggle/test_run_regime_wave.py
.venv/bin/python offkaggle/run_regime_wave.py --dry-run --arm keith \
    --out /private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/regime_dry

# 1. the public-notebook regime (CONTEXT_WINDOW 32768, MAX_OUTPUT 0)
.venv/bin/python offkaggle/run_regime_wave.py --arm keith \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --games all --out offkaggle/results

# 2. our flight's analyzer env (CONTEXT_WINDOW 24576, MAX_OUTPUT 4096), everything else identical
.venv/bin/python offkaggle/run_regime_wave.py --arm flight \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --games all --out offkaggle/results

# 3. stop the container (otherwise 15 min idle billing)
/Users/ahmed/Library/Python/3.14/bin/modal app stop arc3-flashnext
```

Token: read from `~/.config/arc3/vllm_token` (or `$ARC3_VLLM_TOKEN` / `--token`)
into memory only; it is never printed or written (`arm_env.json` and
`results.json` carry `<redacted>`; the dry-run test greps every artifact and
the process output for the dry-run token).

Useful flags: `--games tu93,ft09` (stems or full ids), `--skip-preflight`,
`--per-game-s` / `--concurrency` (diagnostics only — the artifact records
`matches_public25: false`), `--wave-cap-s` (default 9,000 s = 2.5 h),
`--progress-every` (default 120 s status line).

Ctrl-C once = graceful: `bm.request_stop()` cancels the solver task, games are
marked `cancelled`, benchmark.json / score.json / telemetry / summary are still
written; exit rc 130. Twice = hard exit. Verified on the dry run: exit 1.4 s
after SIGINT with 25/25 games `cancelled` and every artifact present.

Outputs under `<out>/<ts>-regime-<arm>/`: `results.json` (per-game levels,
score, actions, wallclock, state; /metrics before/after/delta; stock shas;
arm env), `telemetry.json` (per-game + pooled, definitions inside),
`summary.txt` (one screen, also printed), `arm_env.json`,
`metrics_before.prom` / `metrics_after.prom`, `requests_shim.jsonl` (one line
per HTTP call), plus the harness's own `benchmark.json`, `score.json`,
`transcripts/`, `prompts/`, `artifacts/`.

Reading the pair: put the two `summary.txt` side by side. The `ARM KNOB` line
shows the env AND what the harness itself reported in `[ANALYZER STATUS]`
(keith: `max_output_tokens: server default`, `context_budget_tokens: 31744`;
flight: `4096` / `19968`) — the mechanism attestation. The comparison rows are
`CADENCE` (calls/turns/steps per game, actions per call), `REASONING`
(chars/call mean/median/p90, no-tool-call share, **length-finish share** — the
truncation the flight's `max_tokens=4096` is suspected to cause), `TURNS`
(step-executed vs yielded share), `CLIENT` (e2e per call from the shim) and
`VLLM` (e2e/queue/inference means, gen tok/s, MTP acceptance, preemptions from
the /metrics delta).

## 2. What the runner reproduces, with evidence

| item | value | evidence |
|---|---|---|
| agent bytes | `scratchpad/bundles/june_stock/src/ARC3-Inference`, tree sha `74ab6910…` (42 files) | asserted at start; == keith's dataset copy (`scratchpad/search/field-delta/keith_smoke_v1/src/ARC3-Inference`, same sha; `diff -rq` clean) |
| framework | `scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework`, sha `f68b6850…` | == keith's copy |
| solver pkls | `scratchpad/taaf_scored_ref/{benchmark_initial,deploy_target}.pkl`, sha `7f619ac0…` / `f0dc4b59…` | byte-identical to keith's dataset copy |
| notebook cell 3 process env | `MPLBACKEND=Agg`, `TAAF_RUN_AS_SUBMISSION=0`, `TAAF_MINIMAL_DIAGNOSTICS=1`, `ONLY_RESET_LEVELS=true` | test parses cell 3 |
| cell 13 settings | 7920 s/game, analyzer_timeout 900, concurrency 28, max_actions None, request logs off | test parses cell 13; `geometry.matches_public25` in the artifact |
| cell 15 games | the 25 public ids in that order, OFFLINE arcade over `environment_files/` (the commit run's path), `bm.run(minimal_diagnostics=True)` + `bm._save_json()` + frozen scorer | test parses cell 15 |
| keith analyzer env | KEITH_REGIME.md §5 == keith's persisted `taaf_setup_env.json` | both parsed in tests |
| flight analyzer env | flight notebook cell 9 `duck_env` (the `>= 32768` branch: 24576 / 4096) | ast-parsed in a test; arms differ on exactly the two window keys |

## 3. Telemetry: reproduces the judge's numbers

The extractor works on the harness's own transcripts (no agent change): a
call = one `[MODEL RESPONSE META]` section; a turn = one
`--- analysis_step=N | action=M | … ---` header; reasoning chars = the
stripped `[THINKING]` section after each META. On keith's V14 commit run
(`scratchpad/search/judge/kout/keithtyser/transcripts__*`):

| judge (J-judge.md F4) | extractor | note |
|---|---|---|
| 55 requests/game | 1,371 calls = **54.84**/game | == `vllm:e2e_request_latency_seconds_count` 1,371 in his `vllm-metrics-final.prom` |
| reasoning mean 3,406 / median 2,206 | **3,406.5 / 2,206** | pooled over 1,371 calls |
| 53 analysis steps/game | **53.24** turns/game | his "analysis steps" are `analyze()` turns; distinct `analysis_step` numbers are only 27.3/game because **648 of 1,331 turns (49 %) yield on the 60 s budget** and re-enter with the same step number |
| — | finish: tool_calls 1,339 / stop 27 / length 5; 32 text-only calls (2.3 %) | truncation share 0.4 % in the keith regime |
| e2e 142 s, queue 124 s, MTP 60 %, 57 preemptions, 249 tok/s | 142.2 s / 124 s / 59.8 % / 57 / 249 | from the /metrics parser on his final prom |

Per-call **e2e seconds** cannot come from transcripts (they carry only a
per-turn HH:MM:SS); they come from (a) the `requests.post` shim (client wall
time incl. redirect legs, tokens from `usage`, finish_reason, redirect count,
per game) and (b) the `/metrics` delta (server-side e2e/queue/inference/
prefill/decode means, TPOT, gen tok/s, MTP acceptance, preemptions, prefix hits).

## 4. The redirect / timeout finding (code-cited; no adaptation needed)

* The stock client is `requests.post(url, headers=…, json=…, timeout=T)` with
  no `allow_redirects` argument (`inference/agent/tool_agent.py:1302-1308`),
  so redirects are followed (requests default, `requests/api.py:48`).
* `T` = `_HarnessGameSession.request_timeout_seconds()` =
  min(analyzer_timeout **900 s**, remaining per-game seconds, remaining
  soft-deadline seconds) (`inference/framework/solver.py:227-244`). So the
  timeout is >= 900 s except in the last 15 min of a game, where it shrinks
  to the remaining time (down to 0.1 s) — those requests fail as
  `request_error` turns and the game ends. That is stock behaviour on Kaggle too.
* requests 2.34.2 (`.venv`): a **303 is re-issued as GET**
  (`sessions.py:rebuild_method`), and `Authorization` is **kept when the
  hostname is unchanged** (`should_strip_auth`). Modal's continuation
  (`303 -> poll URL`) therefore works with the stock bytes. The `timeout`
  applies per socket read, per leg — a generation may take longer than 900 s
  end to end across legs without tripping the client; the proxy's upstream
  timeout is 3,600 s (`modal_flashnext_serve.py:250`).
* Verified with the stock client in the dry run: the mock answers every 3rd
  completion with `303 -> /v1/_poll/<id>`; 181 of 542 calls were redirected,
  **181/181 poll legs carried the bearer**, the harness parsed every body.
* `/metrics` is forwarded by the proxy for any path with the bearer
  (`modal_flashnext_serve.py:_run_auth_proxy`); the runner GETs
  `<root>/metrics` before and after the wave and stores the raw text + delta.

Adaptations, both OUTSIDE the agent (bytes unchanged, sha asserted):
1. `HarnessSolver.analyzer_factory` — a documented solver field — builds the
   stock `ToolAgent` with exactly `_make_analyzer`'s arguments and tags the
   game thread with the game id. A test instantiates both paths in a
   subprocess for both arms and asserts identical fingerprints (provider,
   base_url, model_id, timeout 900, max_output_tokens, context budget
   31744/19968, yield 60, unlimited tool steps).
2. A `requests.post` observer shim: times the call, records status, redirect
   legs, `usage` tokens, finish_reason, reasoning/content chars, the payload's
   `max_tokens` and `timeout` — and re-raises/returns unchanged.

## 5. Verification done here

* `offkaggle/test_run_regime_wave.py`: **13/13 pass** (~12 s), incl. the
  keith-transcript reproduction, the /metrics parser on keith's prom, the
  factory-equivalence subprocess, and an end-to-end 2-game dry run.
* Full dry run, 25 games, concurrency 28, 25 s/game: **27.6 s wall, rc 0**,
  542 calls (21.7/game), 25/25 games acting (459 actions), no-tool-call share
  14.4 % (the mock's text-only branch), 181 redirected posts all
  authenticated, transcript calls == shim OK posts, `/metrics` delta 542 (the
  preflight completion correctly excluded by the before-snapshot),
  `score.json` written by the frozen scorer, summary 40 lines.
* Ctrl-C: exit 1.4 s after SIGINT, 25/25 `cancelled`, all artifacts written.
* Git: only the two new files under `offkaggle/`; `june_stock` and
  `taaf_scored_ref` clean.

## 6. Expected cost (RTX PRO 6000 ≈ $3.03/h GPU)

| step | GPU time | ≈ $ |
|---|---|---|
| cold start (image cached, weights on the Volume) | 10-20 min | 0.5-1.0 |
| one arm: preflight + 25 games (7,920 s cap ≈ 2.2 h) + scoring | 2.2-2.5 h | 6.7-7.6 |
| idle scale-down window if not stopped | 15 min | 0.8 |
| **one arm** | **≈ 2.7-3.0 h** | **≈ $8-9** |
| **both arms back to back (one cold start, one idle window)** | **≈ 5.2 h** | **≈ $16** |

Caveat: the $3.03/h is the GPU line only. The app also reserves `cpu=8` and
`memory=131072 MiB`, and Modal bills CPU and memory reservations per second
on top of the GPU; I could not verify the current rates offline — read the
usage dashboard after the smoke before budgeting the second arm. Wall-clock
per arm from launch: 2.5-3 h (the runner's own cap is 2.5 h from the start of
`bm.run`; games that hit it are marked `cancelled` and still scored).

## 7. Could not verify (needs the live endpoint / the smoke)

* The smoke in `scratchpad/flashnext_smoke.log` **failed on the Mac, not on
  the endpoint**: `SSL: CERTIFICATE_VERIFY_FAILED` from the python.org 3.14
  framework build's `urllib` (no CA bundle installed); the `.venv` Python
  3.12 verifies TLS fine with both `urllib` and `requests` (checked against
  pypi.org). The `modal run …::smoke` client should be run with certificates
  installed (`/Applications/Python 3.14/Install Certificates.command`) or with
  `SSL_CERT_FILE=$(.venv/bin/python -m certifi)`. No request reached the
  endpoint from that attempt, so cold start, `/arc3/identity`, the max-
  concurrency line and the redirect flow are all still unobserved.
* Modal's 303 poll URL host: if it is a different hostname, `requests` drops
  the bearer on the poll leg — harmless (the poll endpoint is Modal-managed,
  the proxy never sees it), but unobserved. The preflight prints the redirect
  legs of its own completion, and every shim record carries `redirect_codes`.
* `/metrics` through the proxy and Modal's 150 s ingress cap on a slow
  `/metrics` body: expected fine (tens of KB), unobserved.
* Throughput: keith's box ran 8 seqs / 5 GiB KV (≈ 3 running, 21 waiting,
  queue 124 s per request at 28 games). The same serving config on Modal
  should reproduce that queueing; a slower Volume read or preemption profile
  changes absolute latencies. Both arms see the same endpoint, so the A/B
  delta is fair; absolutes are not comparable to Kaggle.
* The mock exercises the harness's tool path with a trivial policy; real
  model behaviour (long thoughts, 60 s yields, context trimming, `length`
  finishes) is only observed live.

## 8. Observations worth carrying forward

* In keith's regime **49 % of turns end in `Yielded control: turn_time_budget`**
  (648/1,331) — the 60 s yield is the dominant turn terminator, not the
  action. Any config that shortens a call (e.g. `max_tokens=4096`) changes
  the yield/execute mix; the `TURNS` line makes that visible per arm.
* `finish_reason: length` was 5/1,371 (0.4 %) for keith. The flight arm sends
  `max_tokens=4096` on every call (recorded in the shim as `max_tokens`); its
  `length-finish share` is the first number to read.
* The harness's per-game cap cuts the request timeout to the remaining
  seconds; the last call of nearly every game fails with a read timeout and
  is counted server-side but not in the transcript — hence
  `/metrics requests >= transcript calls` (the test asserts that invariant).

## RESULT — keith arm, 2026-09-02 21:21→23:34 UTC (Modal RTX PRO 6000, $≈7)

Artifacts: offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/ (summary.txt,
results.json, telemetry.json, requests_shim.jsonl, metrics proms, harness transcripts).

| metric | Modal reproduction | keith V14 Kaggle commit (judge F4) |
|---|---|---|
| levels/game | **1.44** | 1.44 |
| local score | 6.40 | 6.76 |
| calls/game | 55.8 | 55 |
| turns/game | 54.5 (yielded 43%) | 53 (49% yielded) |
| reasoning chars/call mean / median | 3,173 / 1,962 | 3,406 / 2,206 |
| e2e per call | 139.9 s | 142 s |
| vLLM queue / inference | 120.9 s / 17.8 s | 124 s / 18 s |
| gen tok/s aggregate | 235 | 249 |
| MTP acceptance | 60.0% | 60% |
| preemptions | 142 | 57 |
| actions/game | 154 | ~148 |
| zero-level games | 2 | 6 |

Verdict: the public notebook's serving+analyzer regime is reproduced off-Kaggle with high
fidelity on every cadence/latency/throughput metric. The rig is a valid instrument for
single-knob regime experiments (read by telemetry; levels remain a noisy secondary).
24 client post errors (1.7%) all recovered by the stock retry; 585 redirected posts (Modal
proxy 303 flow) carried the bearer and completed.
