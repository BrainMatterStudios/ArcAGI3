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

## RESULT — flight arm, 2026-09-02 23:34→01:47 UTC (same server session; LAST ~35 MIN DEGRADED)

Artifacts: offkaggle/results/20260902T2334-flight/20260903-013417-regime-flight/. At 01:13 UTC the
Modal container was PREEMPTED ("Container terminated due to preemption") and the replacement
waited on RTX PRO 6000 capacity at the 128 GiB request; 48 client 500s, all in the last ~35 min.
Rig fix committed (96 GiB request, B200 fallback, 6 h cap). Read the matched-time marks, not the
final totals, for this arm.

### Single-knob comparison (same serving, same stock bytes; knob = analyzer window/output cap)

| metric | keith (32768 / no cap) | flight (24576 / 4096) |
|---|---|---|
| levels (25 games) — final | 36 (1.44/game) | 22 (0.88/game; degraded tail) |
| levels at matched +100 min | 32 | 20 |
| local score | 6.40 | 2.34 |
| calls/game | 55.8 | 79.5 |
| reasoning chars/call mean / median | 3,173 / 1,962 | 2,361 / 1,337 |
| finish=length calls | 6 (0.4%) | 38 (1.9%) |
| client e2e/call | 139.9 s | 97.9 s |
| prompt tok/call | 20,061 | 12,595 |
| actions/game | 154 | 189 |
| turns yielded on 60 s | 43% | 48% |
| zero-level games | 2 | 8 |

Matched-time trajectory (levels): flight LEADS for the first 30 min (10 vs 5 at +14 min; 14 vs 10
at +30) — more, faster calls clear the easy first levels sooner — then STALLS: 17 vs 22 at +50,
20 vs 31 at +90, 20 vs 32 at +98 (both pre-preemption). The capped regime converts the second
hour into actions (+23%) without levels; the uncapped regime keeps clearing.

Reading: the analyzer caps alone (24576/4096 vs 32768/none) move this base from 0.88 to 1.44
lv/game on identical serving — +0.56 lv/game, at the local MDE, but the telemetry shift
(+43% calls, −32% reasoning per call, 6× more length-truncated calls, +23% actions, 4× the
zero-level games) is large and consistent across every mark. Our Kaggle Flash-Next flight
(24576/4096, 22 seqs, no MTP) ALSO differed in serving; this arm shows the analyzer knob by
itself explains most of the regime difference. n=1 arm each; treat levels as secondary.

## PRE-REGISTERED — arm `keith_yield180` (launched 2026-09-03 ~02:00 UTC, before data)

Knob: LOCAL_ANALYZER_YIELD_SECONDS 60 → 180 on the keith base; everything else identical
(32768 / no cap, same serving profile, stock bytes). Hypothesis under test: the regime's gain
is per-call depth; the 60 s yield truncates 43% of the base's turns, so fewer interruptions
should give fewer, longer, more complete turns. Counter-hypothesis on record (08-30 session):
the 60 s yield does useful re-grounding and removing it hurts.
Reading rules (telemetry primary): yielded-turn share (base 43%) must fall below 20% for the
knob to have engaged; calls/game (base 55.8) expected to fall; reasoning chars/call (base
3,173 / 1,962) expected to rise; length-cut share and no-tool-call share reported. Levels
(base 36 / 1.44 per game, cross-boot local sd ≈ 0.3 lv on this base per the three public
commit runs 1.28–1.80): ≥ 48 levels (≥ +0.5 lv/game) = STEP candidate, carry to a Kaggle
counterbalanced pair after the quota reset; 30–47 = no step, knob neutral-to-positive,
report telemetry; < 30 = the yield's re-grounding matters, revert. One arm, n=1: any level
reading is provisional; the telemetry direction is the durable output.

**VOID — keith_yield180 launch of 01:51 UTC.** The container answering it was a **B200**
(/arc3/identity gpu row "NVIDIA B200, 183359 MiB"), reached via the fallback added after the
preemption; cadence at +10 min was 366 calls / 11 levels vs the base's 209 / 3 — a different
GPU regime, not the knob. Killed at +12 min; B200 fallback REMOVED (rig runs RTX PRO 6000 only;
if capacity is unavailable the arm waits or fails, never silently changes hardware). Relaunched
on the RTX PRO 6000 with the same pre-registration.

## RESULT — arm `keith_yield180` (RTX PRO 6000 verified via /arc3/identity; 02:02→04:15 UTC)

| metric | keith base (yield 60) | keith_yield180 |
|---|---|---|
| levels / lv per game | 36 / 1.44 | 34 / 1.36 |
| local score | 6.40 | 5.30 |
| paired per-game Δlv | — | −0.08 (sd 0.69; 4 up, 5 down) |
| turns yielded | 43.2% | 30.4% |
| calls/game · turns/game · calls/turn | 55.8 · 54.5 · 1.02 | 54.0 · 39.0 · 1.39 |
| reasoning chars/call mean / median | 3,173 / 1,962 | 3,321 / 2,011 |
| length-cut share | 0.4% | 0.4% |
| client e2e/call | 139.9 s | 144.9 s |
| vLLM queue / inference per request | 120.9 s / 17.8 s | 125.4 s / 18.4 s |
| actions/game | 154 | 133 |
| zero-level games | 2 | 4 |

Pre-registered verdict: the engagement criterion (yielded share < 20%) was NOT met (30%);
levels 34 fall in the 30–47 "no step, neutral" band. Reasoning per call did not change.

What the arm actually revealed (the durable output): per-call latency in this regime is
**queue-dominated** — ~125 s of the ~145 s per request is vLLM queue wait, ~18 s is inference
(3 running streams, ~21 waiting, set by the 5 GiB KV reservation ⇒ "3.21x" concurrency).
The 60 s yield therefore does not truncate reasoning at all; it simply returns the turn while
the request waits in queue, and the next turn re-issues it. Reasoning length is set by the
model, not by the yield. The regime's structure is: each game gets one full-context,
untruncated call every ~145 s; the "interruption" story is wrong, the "no truncation +
32k window" story (the flight-arm comparison) is the one the data supports.
Implication for the next knob: the queue is the throughput ceiling; the KV reservation
(5 GiB → e.g. 10–12 GiB of the ~15 GiB free after model load) would raise running streams
from ~3 to ~6–7 and roughly double calls/game WITHOUT truncation — untested in public; keith's
own V10 (28 seqs, no KV cap) thrashed at 114 tok/s, so there is a sweet spot to find. That is a
serving-profile arm (rig env override needed; ~$8 on the RTX PRO 6000).

## PRE-REGISTERED — arm `kv10` (serving knob; launched 2026-09-03 ~04:40 UTC, before data)

Knob: vLLM `--kv-cache-memory-bytes` 5 GiB → 10 GiB (rig override ARC3_KV_CACHE_MEMORY_BYTES,
identity endpoint must report profile `kv10-bf16-mtp3-c8-cg32-OVERRIDE`); analyzer arm = keith
(32768 / no cap), stock bytes, RTX PRO 6000 (verify gpu_rows before reading). Everything else
identical to the base. Hypothesis: the regime is queue-bound (125 s queue / 18 s inference,
~3 running streams from the 5 GiB reservation); doubling the reservation should roughly
double running streams (expected startup line ≈ "Maximum concurrency … ~6.4x") and cut queue
time, giving more untruncated calls per game. Risk on record: keith's V10 (28 seqs, no cap)
thrashed at 114 tok/s; and the 10 GiB must fit beside the 79 GiB model load (~15 GiB free) —
a boot failure or preemptions >> 53 is a valid negative.
Reading rules: ENGAGEMENT = vLLM queue time per request < 80 s (base 121 s) AND calls/game
> 70 (base 55.8); else the knob did not bind. Reasoning chars/call and length-cut share must
stay within ±15% of the base (else the regime changed in kind, not just rate). Levels (base
36): ≥ 48 = STEP candidate → Kaggle counterbalanced pair after the reset; 30–47 = no step
(throughput alone is not the lever, consistent with the +0.15 lv budget arithmetic); < 30 =
harmful (thrash/preemption) — check preemptions and TPOT. n=1; levels provisional.
VOID: kv10 launch 05:31Z preempted at +12 min (Modal 'Container terminated due to preemption'); relaunched 05:53Z

## RESULT — arm `kv10` (RTX PRO 6000 + profile kv10-…-OVERRIDE verified; 05:53→08:06 UTC)

| metric | keith base (KV 5 GiB, 3.21x) | kv10 (KV 10 GiB, 6.42x) |
|---|---|---|
| levels / lv per game | 36 / 1.44 | **43 / 1.72** |
| paired per-game Δlv | — | +0.28 (sd 1.22, SE 0.24; 6 up, 4 down) |
| local score | 6.40 | **10.47** (ft09 WON 6/6 = 100 pts; sb26 1→4; ar25 2→4) |
| zero-level games | 2 | 6 (bp35, cn04, m0r0, tn36 fell to 0; tn36 spent 1,132 actions) |
| calls/game · turns/game | 55.8 · 54.5 | 82.2 · 80.0 |
| reasoning chars/call mean / median | 3,173 / 1,962 | 3,050 / 1,810 (−4% / −8%) |
| length-cut share | 0.4% | 0.4% |
| client e2e/call | 139.9 s | 93.7 s |
| vLLM queue / inference | 120.9 s / 17.8 s | 65.5 s / 27.2 s |
| gen tok/s aggregate · TPOT | 235 · 12.5 ms | 342 · 18.8 ms |
| preemptions | 142 | 4 |
| actions/game | 154 | 243 |

Pre-registered verdict: ENGAGED (queue < 80 s, calls/game > 70), regime unchanged in kind
(reasoning within ±15%, truncation identical), levels 43 → **no step** (30–47 band): throughput
without truncation is a modest positive (+0.28 lv/game, 1.2 SE), consistent with the
loss-ledger's action-budget arithmetic (+0.15–0.27). The score jump (+64%) is concentrated in
three games and includes one full clear; the zero-level count tripled — more calls also feed
runaway games. Rider status: worth carrying as the FIRST counterbalanced Kaggle pair on top of
the byte-copy base (same GPU, same free memory ⇒ same 6.42x), not a reason to alter the base.
Modal cost this arm ≈ 2.5 h RTX PRO 6000 + the voided 12-min attempt.

## PRE-REGISTERED — arm `keith_retry` (fresh-mind level retry; written before launch, 09-03 ~10:15 UTC)

Knob: graft_retry installed in memory on the keith base (RETRY_ENABLE=1, K=3× human baseline,
ABS=200, COOLDOWN=150, MAX=2 per level); stock bytes sha unchanged; serving = keith profile
(KV 5 GiB) on the RTX PRO 6000 (verify identity). Hypothesis: stuck levels are one wrong
hypothesis pursued to the clock (loss-ledger); a level RESET plus explicit abandonment of the
stale note plus a forced alternative-hypothesis step clears more stuck levels; each cleared
level unlocks a fresh bucket. Counter-hypothesis on record: budget-based triggers cannot tell
"wrong hypothesis" from "slow correct execution"; resets cost actions and may cut a level that
was about to clear (08-27: depth-by-budget dead; 08-30: retry-replay accounted for 46% of
blown-level actions).
Reading rules: ENGAGEMENT = retries fired ≥ 15 across the 25 games (else the trigger did not
bind at K=3). Primary telemetry: levels cleared AFTER a retry fired on that level
([RETRY-CLEAR]) — ≥ 6 = the mechanism works; 1–5 = weak; 0 = dead. Secondary: actions in
never-clearing buckets (base ≈ 60%), zero-level games (base 2), calls/game and reasoning
per call must stay within ±15% of the base (else the regime changed in kind). Levels (base
36): ≥ 48 = STEP candidate → counterbalanced Kaggle pair on the byte-copy base after 09-05;
30–47 = no step (report [RETRY-CLEAR] count as the durable output); < 30 = harmful (resets
destroying progress) — count levels lost after a retry. n=1; levels provisional.
