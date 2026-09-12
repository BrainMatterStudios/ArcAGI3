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

## PRE-REGISTERED — arm `keith_retry` (fresh-mind level retry) — REVISED after the judge, before launch (09-03 ~10:50 UTC)

The first draft (K=3; retries ≥ 15; RETRY-CLEAR ≥ 6; levels ≥ 48) was withdrawn unlaunched:
the judge computed from the base arm that only 7 stuck buckets exceed 3× baseline, so those
thresholds were unreachable by construction. Also the rig was still deployed with the kv10
override; it is redeployed at KV 5 GiB and the runner now asserts profile + GPU in preflight.

Knob: graft_retry on the keith base, RETRY_K=2.5 (ABS 200, COOLDOWN 150, MAX 2), serving =
keith profile kv5 on the RTX PRO 6000 (asserted). Dose is NOTE-ONLY (the quoted "abandoned
hypothesis" is the labelled world-model note, updated on ~11% of turns; chat history kept) —
a stronger dose (quote the last reasoning tail / clear history) is the follow-up arm if this
one engages but does not flip levels.
Predicted trigger set from the base arm at K=2.5 (9 stuck buckets): sc25 L2, tn36 L3, sb26 L2,
cd82 L2, wa30 L2, sp80 L2, su15 L2 (the seven ≥ 3×) plus two more between 2.5× and 3×; plus
2 COLLISIONS — levels the base CLEARED past the threshold (tn36 L1 at 136 vs 96; r11l L1 at
101 vs 66) that this graft would reset first.
Reading rules: ENGAGEMENT = retries fired ≥ 6. PRIMARY = the per-(game, level) outcome table
on every retried level: cleared after retry ([RETRY-CLEAR]) vs not; ≥ 3 clears among ~9 fires
= mechanism works (carry to a stronger-dose arm and a Kaggle pair); 1–2 = weak; 0 = dead.
COLLISION COST reported in every band: retried (game, level) pairs that the base cleared and
this arm did not. Paired per-game Δlv vs the keith base and local score reported alongside
(a RESET can only lower a level's efficiency; the bucket is never refunded). Levels total is
NOT a verdict metric for this arm (25-game paired sd ≈ 1.2 lv; the mechanism touches ≤ 9
levels). Calls/game and reasoning/call must stay within ±15% of the base. n=1.

## RESULT — arm `keith_retry` (K=2.5, note-only dose; RTX PRO 6000 + kv5 asserted; 10:14→12:27 UTC)

| metric | keith base | keith_retry |
|---|---|---|
| retries fired | — | **5** (r11l L3 @133/51, sk48 L1 @154/61, cd82 L2 @21/8, sp80 L1 @182/39, sb26 L2 @95/28) |
| levels cleared after a retry | — | **0 / 5** |
| collision cost (retried level the base had cleared, lost here) | — | 1 (sp80 L1: base 1 → 0) |
| levels / lv per game | 36 / 1.44 | 40 / 1.60 (paired Δ +0.16, sd 0.92, SE 0.19; 8 up, 5 down — NOT a verdict metric) |
| local score | 6.40 | 8.62 |
| zero-level games | 2 | 4 |
| calls/game · actions/game | 55.8 · 154 | 51.7 · 119 |
| reasoning chars/call mean / median | 3,173 / 1,962 | 3,493 / 2,299 (+10% / +17%) |

Pre-registered verdict: ENGAGEMENT NOT MET (5 < 6): the stuck-bucket set differs draw to
draw (the judge's 9 predicted triggers came from the base draw; this draw produced 5). PRIMARY
= 0 of 5 retried levels cleared afterwards — the note-only fresh-mind dose did not flip any
stuck level in this draw; one retry destroyed a level the base cleared. The mechanism as
dosed is unsupported. Remaining variant on record (not run): stronger dose — clear chat
history / quote the last reasoning tail — but 0/5 with a note-only dose is not encouraging,
and the per-draw stuck set is itself unstable, so the trigger is chasing noise. Recommendation:
do not spend more on this lever unless Ahmed wants the stronger-dose arm explicitly.

## 2026-09-06 — 3-wall instrument arms (`keith_evid`, `keith_hypo`, `keith_up8`), `--draws`

Built for judge program item 1 (docs/research-2026-09-06/J-judge-0906.md); full write-up, test
counts and launch commands in `submission/_throughput_v1/EVID_STATUS.md`. NOT launched.

* `keith_evid` = keith + graft_evidence (EVID_ENABLE=1, MAX_ENTRIES 40, MAX_CHARS 1500, TRACE 1):
  object-level before/after diff + per-action trace + LEVEL CLEARED flag appended to every
  executed-action tool result. `keith_hypo` = keith + graft_hypo (HYPO_ENABLE=1): the
  hypothesis-enumeration + probe rule appended to every analyzer prompt. `keith_up8` = keith +
  MULTIMODAL_UPSCALE=8 (no graft; 512 px PNG, ~256 derived vision tokens vs 64).
* `--draws N` = taaf `n_passes`: each selected game played N times as independent runs
  (`<gid>_p0.._p<N-1>`); `results.json:games[*].draw/run_stem`, `telemetry.json:per_game` keyed by
  run stem, `per_game_draw` = per-(game, draw) levels; shim records carry `run_stem`.
  `--per-game-s` unchanged (default 7920).
* Telemetry: `evid_markers`, `evid_level_flags`, `hypo_markers`, `turn_levels`, `level_reached`,
  `wall_level`, `engagement` (`evid`/`hypo`, `_wall` = on the wall level; the judge's gate is
  >= 80 % of wall turns); summary `AID` and `DRAWS` lines, `evid`/`hypo`/`wall%` columns.
* Dry runs of all three arms with `--games cd82,dc22,lf52 --draws 2 --per-game-s 60` PASS
  (6 runs each; [EVID] on 350/350 executed turns, [HYPO] on 361/361 turns, UPSCALE 8 attested).

## fp8-KV boot smoke (09-06, judge item 3) — FAILED to boot

`--kv-cache-dtype fp8` on this pinned nightly (0.1.dev20073+g8e685d198, RadixArk NVFP4 + PLE patch):
EngineCore initialization failed ("vLLM exited rc=1 before ready"); the worker's root-cause line
was not captured (Modal's log tail had scrolled). The zero-byte ×2-streams idea is CLOSED unless a
future rider run captures the root cause from the start of the boot log (~$0.5). Rig redeployed at
the base profile (kv5, auto); KV dtype override stays available via ARC3_KV_CACHE_DTYPE.

## PRE-REGISTERED — 3-wall turn-capped instrument (launched 2026-09-06 ~18:35 UTC, before data)

Arms in order on ONE server session (base profile kv5, identity-gated): `keith` (base) →
`keith_evid` → `keith_hypo` → `keith_up8`; each `--games cd82,dc22,lf52 --draws 2 --concurrency 3
--max-calls 60 --per-game-s 1500` (6 runs/arm, two batches of 3; ≈$3/arm).
Walls: cd82 L2, dc22 L2, lf52 L2 (base pass rate 0/14 attempts that reached L2 across 5 runs).
Per-run VOID: request errors > 0, preemptions > 0, or length-finish > 1% (redraw once).
ATTEMPT counts only if the run reaches L2 with ≥ 30 calls left; PASS = level 3 reached.
Decision: ≥ 3/6 attempts pass AND base ≤ 1/6 → the 25-game wave ($9; rule ≥ 48 levels AND ≥ 3 of the
8 consistent walls passed); 2/6 → one more round on the same games; ≤ 1/6 with UPTAKE ≥ 30% and the
dc22 contradicted-movement / lf52 own-table-vs-click counts falling to ≤ 1 → EVID = RIDER candidate;
≤ 1/6 with low uptake → dead as a prompt aid. UP8: first-call prompt_tokens must be ≈ +192 over the
base's ~4,050 (≈ +0 = processor downscaled → arm void; ≫ → geometry differs → stop and read).
Realistic prior for a step: low (coverage ≈ 1 direct + 1 partial of the three walls).

## RESULT — base arm of the 3-wall instrument (16:34→17:09 UTC) + PRE-REGISTERED disambiguation

Base at conc 3 / 60 calls / 1500 s clock: dc22 L2 passed in BOTH draws (2/6 levels each; calls@L2 16,
27), cd82 p0 passed but VOID (length-finish 1.7%), cd82 p1 never reached L2, lf52 0/2. Attempts 4,
passes 2 valid (+1 void). In the five full-length runs these walls passed 0/14. Same bytes, same
server; differences: e2e 16 s vs 145 s (queue-free), yields 5.5% vs 43%, and the model is SHOWN
run_elapsed_seconds / time_remaining_seconds in every action result (tool_agent.py:1446,
solver.py:225) — a 1500 s clock instead of 7920 s. The instrument's null is therefore NOT ~0; the
aid arms are read against THIS base only (paired, same geometry).
DISAMBIGUATION ARM (queued behind the chain, before data): base at conc 3 / 60 calls but
--per-game-s 7920 (long clock, queue-free). Rule: dc22 passes ≥1/2 and total passes ≥2/4 → the
CADENCE (queue-free / no yields) carries the effect, the clock does not; dc22 0/2 and passes ≤1/4 →
the CLOCK CUE carries it → build a "clock compression" graft (report a compressed time_remaining to
the model; harness timing untouched) as the next single-knob arm on the full 25-game geometry.
Mixed (1 pass) → one more round each.

## RESULT — 3-wall instrument (5 arms, 16:34→19:51 UTC, ≈$15)

Walls cd82 L2 / dc22 L2 / lf52 L2, 2 draws each, conc 3, 60 calls, RTX PRO 6000 kv5 (identity OK all arms).
| arm | cd82 lv | dc22 lv | lf52 lv | valid passes | uptake | reasoning/call |
|---|---|---|---|---|---|---|
| base, 1500 s clock | 2(void) / 0 | **2 / 2** | 1 / 1 | 2 (+1 void) | — | 3040 |
| base, 7920 s clock | 1 / 1 | **2 / 2** | 1 / 1 | 2 (dc22 p0 reached L2 at call 32 → not an "attempt" but passed) | — | 3245 |
| evid | 0 / 2(void) | 2 / 0 | 1 / 1 | 1 (+1 void) | 7.8% | 3515 |
| hypo | 1 / 2(void) | 1 / 0 | 1(void) / 1 | 0 | 0.8% | 4533 |
| up8 (attested +201 tok) | 2(void) / 1 | 1 / 0 | 2 / 1 | 1 (+1 void) | — | 3346 |
Historical full-length base (5 runs, 55 calls/game, queued, 43% yields): these walls 0/14.
VERDICTS (pre-registered): EVID — not engaged (uptake < 30%), no pass advantage over the same-geometry
base → not a step; rider status needs the mechanism counts (deferred). HYPO — dead (uptake 0.8%,
worse). UP8 — live but no advantage → dead. CLOCK CUE — refuted (long clock passes dc22 2/2).
THE FINDING: at the queue-free cadence (e2e ~16-20 s, yields ~5%) the unmodified base passes dc22's
wall 4/4 within 60 calls; in the queued live regime (e2e ~145 s, yields 43-49%) it passed 0/4 with
the same call count. CORRECTION (tool_agent.py:1777-1785, 1904, 1969): a yield never discards a response — the 60 s turn
budget is checked between steps, so with 145 s calls EVERY turn ends after ONE call (calls/turn 1.02)
and the within-turn analysis→act tool loop never runs; queue-free, turns average 1.66 calls. The
mechanism is within-turn multi-step continuity, not wasted compute.
Cadence, not the clock and not any prompt aid, is what moved a deterministic wall.

## PRE-REGISTERED — "half-concurrency" geometry wave (launched 09-06 ~20:05 UTC, before data)

Hypothesis: the live loss is queue-induced yield waste. Live-legal fix with the SAME 9 h budget:
run games at concurrency 14 for 3,960 s each (110 games / 14 = 8 waves × 1.1 h = 8.8 h) instead
of 28 × 7,920 s (4 waves × 2.2 h). Expected: queue ≈ half (3.21x KV vs 14 waiting instead of 28),
e2e ≈ 70-80 s, yields ≈ 15-25%, productive calls per game ≥ the current 55 in half the time; and
74% of clears happen in the first hour anyway (loss-ledger-2).
Rig arm: `keith --concurrency 14 --per-game-s 3960 --games all` (two batches of 14 + 11, ≈2.2 h
wall, ≈$9), read against the 25-game base (36 levels, 1.44/game; 55.8 calls, 43% yields).
RULES: ENGAGEMENT = yielded share < 25% AND client e2e < 90 s. PRIMARY = levels: ≥ 48 → STEP
candidate → fly on Kaggle as a two-constant change of the flown v4 notebook (concurrency 28→14,
max_runtime_s_per_game 7920→3960) after a judge pass; 40–47 → positive-not-step, one more draw
(counterbalance: run it FIRST next session); < 40 → the cadence effect does not survive the
shorter per-game time → dead; per-game VOID rules as before (errors, preemptions, truncation >1%).
Secondary: zero-level games (base 2), calls/game, actions/game, reasoning/call within ±15%.

## PRE-REGISTERED — `keith_yield900` at the LIVE geometry (queued behind the half-concurrency wave)

Knob: LOCAL_ANALYZER_YIELD_SECONDS 60 → 900 (= analyzer_timeout) on the keith base, conc 28,
7920 s — a one-env-var live-legal change that lets a turn run several calls (analysis → act)
despite 145 s queued calls. The 180 s arm (34 lv) only allowed ~1 extra call (calls/turn 1.39);
900 s allows ~6. RULES: ENGAGEMENT = calls/turn ≥ 2.0 AND yielded share < 15%. PRIMARY levels
(base 36): ≥ 48 → STEP candidate (fly as a one-env-var change after a judge pass); 40–47 →
positive, redraw; < 40 → dead. Risk on record: long turns starve other games of the 3.21x
server (queue grows); read e2e, calls/game and zero-level games.

## RESULT — half-concurrency geometry wave (conc 14, 3960 s/game; 20:35→22:51 UTC)

32 levels (1.28/game) vs base 36; score 6.12 vs 6.40; zero-level games 6 vs 2; calls/game 56.8;
e2e 68 s (engaged on latency) but yields 37% (NOT engaged: < 25% required) and calls/turn 1.15 —
the 68 s call still exceeds the 60 s turn budget, so turns still end after one call. DEAD by rule.
Mechanistic reading: what the wall instrument exploited was calls faster than the turn budget
(16–20 s vs 60 s → 1.66 calls/turn), not shorter games. The live-legal version of that is the
turn budget itself (yield 900 s), now running at the live geometry.

## RESULT — `keith_yield900` (live geometry, turn budget 900 s; 22:51→01:05 UTC, ran LAST in session)

41 levels (1.64/game) vs base 36; score 8.46 vs 6.40; zero-level 1 vs 2; calls/game 51.5;
calls/turn **2.10** (engaged), yields **7%** (engaged); actions/game 102 vs 154; reasoning/call
within range; e2e 153 s (queue unchanged, as expected). Verdict by rule: **40–47 band = positive,
not a step; redraw FIRST in a fresh session** (order-effect control; this run was third in its
session). Live-legal as a one-env-var change (LOCAL_ANALYZER_YIELD_SECONDS=900) on the flown v4.
PRE-REGISTERED counterbalanced draw (launched ~01:15 UTC, first in a fresh session, ≈$9): ≥ 44
levels → pooled two-draw mean ≥ 42.5 (+6.5 over base, ~1.8 SE) → propose a live flight as a
CANDIDATE with its own rule (base 3.25/2.58; ≥3.9 positive candidate; 2.9–3.9 inconclusive;
<2.9 negative), Ahmed's go required; 36–43 → positive-but-unproven, keep on the rig; < 36 → the
first read was order/noise, dead.

## RESULT — yield900 counterbalanced draw (cold boot, first in session; 01:07→03:20 UTC) — VOID

Server died at ~+120 min (308 request errors, 3 games crashed; see app log line above for the
cause). Matched-time read before the failure: 30 levels at +120 min vs base 35 and the first
yield900 draw 40 — this draw was tracking BELOW the base, so the first draw's +5 is not
reproduced; but a void draw cannot settle it. Telemetry pre-failure: calls/turn 1.37, yields 4%,
actions/game 115 — engaged, same regime as draw 1. Verdict: yield900 = "positive-but-unproven";
pooled evidence (41 valid + a void 30-at-120) is consistent with no real gain. One more valid
counterbalanced draw is needed to keep or kill; the modal rig's preemption rate (3 of the last 8
long runs) makes each such draw a coin flip on completion.

## RESULT — yield900 counterbalanced draw #2 (cold boot, first/only arm; 07:03→09:17 UTC) — VALID

40 levels (1.60/game), score 7.90, zero-level 1, calls/game 54.2, calls/turn 2.05, yields 6.2%,
actions/game 124, e2e 146 s, 0 preemption/void. Two valid draws: 41 and 40 vs base 36 → pooled
+4.5 levels (+0.18 lv/game, ≈1.5 SE), score +23–32%, zero-level games 1 vs 2 in both, actions/game
−20–34%, regime engaged in both (2.05–2.10 calls/turn, 6–7% yields). Verdict by rule: 36–43 band =
POSITIVE, NOT A STEP (bar 48). It is the first lever in the campaign to replicate in the same
direction on two independent boots; it is a one-env-var change (LOCAL_ANALYZER_YIELD_SECONDS=900)
on the flown v4. Live expectation if the +25% score transfers: ≈ 2.9 → 3.6, below the one-draw
detectability threshold (+1.0). Decision on a slot is Ahmed's: EV-positive but unreadable live.

## PRE-REGISTERED — arm `keith_probe` (harness-enforced probe discipline on the 900 s regime; written 09-08 before the build finished)

Base for comparison: keith_yield900 rig draws 41 and 40 (25-game total sd across same-regime runs
≈ 2–3 levels; pooled null vs the 60 s base per loss-ledger-3). Knob: graft_probe — after 2
analysis-only python calls in a turn the harness REFUSES further analysis-only snippets (returns a
tool result demanding a ≤5-action test), cap 2 refusals/turn; informational one-liner on the turn
after a no-action turn. Everything else = keith_yield900. Run FIRST in a fresh session, live geometry.
ENGAGEMENT: refusals ≥ 1 per game on average AND acting-call-after-refusal share ≥ 50% AND share of
turns with ≥3 analysis-only calls < 5% (now 15%); if the model answers refusals with more
non-acting calls or no-tool replies, the lever is "not engaged" regardless of levels.
PRIMARY (levels, 25 games): ≥ 52 → step candidate (then a counterbalanced draw and ≥3 of the 12
never-passed walls before any slot); 45–51 → positive, counterbalanced redraw; < 45 → dead.
SECONDARY: wall actions/baseline median (now 0.72) — must rise toward ≥ 1.0 if the mechanism works;
yields/draw; zero-level games; length-finish ≤ 1% per run (VOID rules as before).

### BUILT 2026-09-08 — `keith_probe` ready (not launched; judge SHIP-WITH-FIXES applied)

Graft `submission/_throughput_v1/graft_probe.py` (14/14 tests on both bundles; real-engine dry run
`dry_run_probe.py` PASS 16/16 with a compliant and a stubborn brain; runner suite 26/26). Arm =
`keith_yield900` + `PROBE_ENABLE=1 PROBE_MAX_ANALYSIS=2 PROBE_MAX_PROBE=5 PROBE_MAX_REFUSALS=4
PROBE_NOTE_LINES=3` (differs from keith_yield900 by exactly those keys; `PROBE_*` scrubbed for every
other arm). Analysis-only = the harness's own `_ToolDispatchResult.step_executed` (payload); the pre-run
refusal predicate is an AST read for an `action()` call. Counters are per SPAN: a no-action turn carries
its analysis/refusal counts into the next turn on the same level (reset after an acting turn, a level
change or a new game), so the 4-refusal cap bounds the span (stubborn brain: exactly 4 refusals per game
over ~50 NOACT turns). GATE (judge amendment of the pre-registration above): refusals >= 1/game AND
acted-after-FIRST-refusal >= 50 % (turn-ending refusals count as non-acting) AND wall actions/baseline
median >= 0.9; secondary: spans with >= 3 executed analysis-only calls (leak split cap_lifted /
dead_branch / unparsable) and yields/draw. PRIMARY: levels vs the pooled six-draw base 39.33 (sd 2.34)
+ co-primary walls passed among the 12 six-draw-never-passed walls (bp35 L2, dc22 L2, g50t L2, lf52 L2,
lp85 L6, ls20 L2, r11l L3, sb26 L2, sp80 L2, tn36 L3, vc33 L4, wa30 L2). SAFETY: GAME_OVERs/run vs 0.87,
live-cap score vs 8.42/game. Summary lines `PROBE` / `PROBE-2ND` / `PROBE-PRIMARY` / `PROBE-SAFETY`.
READ (locked before launch, 09-08): PRIMARY levels >= 52 -> step candidate (counterbalanced redraw + >= 2 of the
12 walls before any slot); 45-51 -> positive, counterbalanced redraw; < 45, or ENGAGED with levels inside
39.33 +/- 2.34 -> dead. NOT ENGAGED -> the lever is unread regardless of levels (a mechanism failure, not a
score). VOID rules unchanged (request errors > 0, preemptions > 0, length-finish > 1 %); first in a fresh boot.
Judge's prior: weakly positive, +0 to +3 levels; the modal outcome is a null.

### RESULT 2026-09-08 — `keith_probe` ENGAGED, FLAT → DEAD (pre-registered read)

Wave `offkaggle/results/20260908T0752-keith_probe` (first in a fresh boot 07:52Z, identity ok: kv5 profile on
RTX PRO 6000, no override; 2.25 h; e2e 150 s, MTP 59 %, vLLM KV preemptions 121 — same 109–181 band as every
full-clock run of this regime; length-finish 0.5 %; one proxy 502 at +74 min + the usual end-of-clock read
timeouts; per-run VOID tags fire on every wave of this regime and are ignored as before). Cost ≈ $9.

ENGAGED = YES, strongly: refusals 66 (2.64/game, 23/25 games); acted after the FIRST refusal 56/64 = 87.5 %
(transcript read 96.9 %); wall actions/baseline median 1.00 (ledger-3: 0.72); spans with ≥ 3 executed
analysis-only calls 1.4 % (was 15 %; 10 of the 11 are dead-branch `if found: action(...)`, cap lifted 0);
analysis share of calls 32 % (was 49 %); turn-budget yields 0 (was 27–30/draw); calls/turn 1.68; actions
158/game (yield900 102, yield-60 154); NOACT turns 0. The harness reshaped the model's behaviour exactly as
designed.

PRIMARY: **41 levels** vs pooled six-draw base 39.33 (sd 2.34) → +1.7 (+0.71 sd) — inside the band →
**DEAD** under the locked read (engaged-but-flat). Paired per game vs the four Modal base draws
(M1 36, M4 40, Y1 41, Y2 40): mean +0.07, sd 0.74, se 0.15. Four games above their 4-draw max (cd82 2,
dc22 2, r11l 3, su15 2), one below its 4-draw min (tn36 0, a high-variance game). Co-primary: 2/12
never-passed walls passed (dc22 L2, r11l L3; target ≥ 3). Local score 7.44; live-cap 7.92/game (base 8.42).

SAFETY (corrected after the wave — runner bug, see below): GAME_OVERs 32 = 1.28/run vs yield900 Modal
0.74/run (17, 20) and yield-60 Modal 1.34/run — forced ≤ 5-action probes die about as often as the 60 s
regime did; the yield900 regime's lower death rate came from its longer analysis, which the probe removes.

Instrument bug found and fixed (commit after this block): `game_overs_from_events` pre-filtered lines on
`'"game_over": true'` (with a space) but the harness writes compact JSON, so the PROBE-SAFETY line read 0;
fixed to pre-filter on the key only; regression test `test_game_overs_from_events_counts_compact_json`;
a dated PROBE-SAFETY-CORRECTION line was appended to the wave's summary.txt (the original line is kept).

Reading: fourth replication that reshaping the loop's behaviour without changing what the model understands
does not move levels — patch 21 (×1.40 actions, −25 % levels), yield900 (2× calls/turn, 118 = 118),
probe discipline (analysis → action, wall ratio 0.72 → 1.00, +0.07/game). The 2/12 wall passes and 4-above-max
are within what a null draw produces (many capped games tie at 1, so exceeding the max is rare per game but
25 games give several chances); no redraw is bought for it — the pre-registration said dead and the
"two draws of +5 prove nothing" law applies with more force to one draw of +1.7.
Full report, exact refusal text, dry-run proof and the launch command: `submission/_throughput_v1/PROBE_STATUS.md`.

## PRE-REGISTERED — arm `keith_carry` (Track A1: compaction instead of eviction; written 2026-09-08 before launch)

Prerequisite read (docs/research-2026-09-08/R-reasoning-carriage-0908.md): prior-turn reasoning ALREADY reaches the
model in the stock regime (template keeps every `<think>` block; vLLM maps `reasoning` → `reasoning_content`; the
server's prompt-token deltas match tokenized reasoning to ~30 tokens on 1,268 request pairs). The plan's "reasoning
is wiped every second call" premise is withdrawn; what is lost is what the trimmer EVICTS (27 % of consecutive
requests evict; the window fills after ~10 turns). So A1 = compaction instead of eviction; reasoning carriage is
MEASURED, not re-injected.

Knob: graft_carry on the yield900 base (arm `keith_carry` = keith_yield900 + CARRY_ENABLE=1 CARRY_TARGET_FRACTION=0.5
CARRY_SUMMARY_CHARS=4800 CARRY_INPUT_CHARS=48000 CARRY_COMPACT_MAX_TOKENS=1500 CARRY_COMPACT_THINKING=0
CARRY_MIN_DROP_MSGS=2). When the trimmer must drop, the graft drops a chunk down to 50 % of the 31,744 budget and asks
the model (one extra no-tools call, thinking off, ≤ 1500 tokens) to fold the dropped turns into a compacted-knowledge
block that rides the system message of every later request; the stock trimmer still enforces the hard budget.
Expected ~7 compactions per 52-call game (each a queued ~150 s call ≈ 13 % of the clock). Everything else = keith_yield900.
Full build + tests: submission/_throughput_v1/CARRY_STATUS.md.

KILL TEST (first, fresh boot, ≈ $3): `--games cd82,dc22,lf52 --draws 2 --concurrency 3 --max-calls 60 --per-game-s 7920`
(the 09-06 3-wall instrument geometry with the long clock; that base read cd82 1/1, dc22 2/2, lf52 1/1 = 8 levels / 6
runs). READ: ENGAGED (compactions ≥ 1/run, failures ≤ 10 %, no request > 32,768 tokens, ≥ 40 % of calls carry the block)
AND levels ≥ 7 → the 25-game wave; ENGAGED but levels ≤ 6 → stop and read the blocks before spending more; NOT
ENGAGED → mechanism failure, fix before any wave. Also read the blocks themselves (transcript `SUMMARY:` sections):
concrete (coordinates/actions/results) vs generic, and whether refuted hypotheses are listed.

25-GAME WAVE (fresh boot, live geometry, ≈ $9, with a Monitor): ENGAGEMENT gate as above (CARRY line).
PRIMARY = levels vs the pooled six-draw base 39.33 (sd 2.34): ≥ 48 step candidate → counterbalanced redraw → live 3-draw
rule; 45–47 → counterbalanced redraw; ≤ 44, or ENGAGED with levels inside 39.33 ± 2.34 → dead (engaged-but-flat).
CO-PRIMARY = walls passed among the 12 six-draw-never-passed walls (target ≥ 3). SAFETY = GAME_OVERs/run vs 0.87 and
live-cap score/game vs 8.42. FIT-THE-CLOCK = (play calls × e2e) + (compactions × compaction e2e) ≤ 7,920 s per game,
read from the CARRY-SAFETY line and the CARRY line; an arm that does not fit is not live-eligible whatever it scores.
SECONDARY: prompt tok/call mean (base ≈ 20.3k) and max (must stay < 32,768), calls/game (base 51–54) — compactions
displace play calls; zero-level games (base 1–4); reasoning msgs carried per request (a new stock fact: expected several).
VOID rules unchanged (Modal container preemption; identity gate must print the kv5 profile on an RTX PRO 6000; vLLM
KV preemptions ~110–180 are a regime constant); first in a fresh boot. Judge's prior: the mechanism is the field's
converged method (Astra provider adapter, Tufa's own unfinished lever), but the compactor here is Flash-Next with
thinking off summarising itself; modal outcome unknown — this is the first read of the thesis on this brain.

### RESULT 2026-09-08 — `keith_carry` kill test: ENGAGED, levels BELOW the bar → NO WAVE (pre-registered read)

Wave `offkaggle/results/20260908T1843-keith_carry-kill` (fresh boot 20:43Z, identity ok: kv5 profile on RTX PRO 6000;
0.70 h, ~$2.5). Geometry = the 09-06 3-wall instrument with the long clock (cd82/dc22/lf52, 2 draws, conc 3, 60 calls,
7920 s), whose stock base read **8 levels / 6 runs** (cd82 1/1, dc22 2/2, lf52 1/1).

**MECHANISM: ENGAGED = YES, on every condition, with room to spare.** 44 compactions (7.33/game — the predicted ~7),
**0 failures of 44**, 0 requests over the 32,768 window (max 26,843), 287/360 calls (79.7 %) carried the block, first
block at call #13 (median). Per compaction: 19.7 dropped messages, 30.7k input chars, **13.0 s**, 1,260 completion
tokens, block 4,002 chars mean. Cost is negligible in this geometry (44 × 13 s over 6 runs vs 7,920 s/game).
The blocks themselves are **high quality**: concrete verified mechanics with coordinates, refuted hypotheses, tried
sequences (e.g. dc22 p1: "TOGGLE BLOCKED: clicking Blue T while the player stands on Solid Blue results in NO
movement", "when Blue moves A->B the destination becomes Dithered (hole)", "GAME OVER occurred when the player was
left on a dithered hole"). Knowledge capture is not the failure.

**PRIMARY: 6 levels / 6 runs vs base 8 / 6 runs** — cd82 [2, 0] vs [1, 1], dc22 [1, 1] vs [2, 2], lf52 [1, 1] vs [1, 1].
Pre-registered rule was "ENGAGED and levels >= 7 -> 25-game wave; ENGAGED but <= 6 -> stop and read the blocks before
spending more". **6 => STOP. The 25-game wave is NOT launched.** (Per-run levels here are 0-2 and 2 of 6 runs are VOID
by the standing rules — cd82_p0 length-finish 1.7 %, lf52_p0 request_errors 2 — so 6 vs 8 is inside this instrument's
noise; the honest statement is "did not clear the bar", not "carry is worse". The one real regression worth naming:
dc22's L2 wall, which the stock passes 4/4 in this queue-free geometry, was passed 0/2 here.)
SAFETY was fine: GAME_OVERs 0.83/run (base 0.87), live-cap score 1.59/game, fit-the-clock 966 s + compactions of 7,920 s.

**DIAGNOSIS (measured, and it is a design flaw in the arm, not in the mechanism):** the arm **under-fills the context
window**. Mean prompt tokens per call: **carry 15,147 = 47.7 % of the 31,744 budget**, vs stock **20,286 (63.9 %)** in
the 09-08 probe wave and **20,405 (64.3 %)** in the 09-07 yield900 wave. `CARRY_TARGET_FRACTION=0.5` drops to half the
budget and refills, so the agent holds ~4 recent turns where the stock rolls ~10, and the block (~1.3k tokens) does not
pay for the raw recency it displaces. The agent gained a good summary and lost about a quarter of its working context —
a wash at best on games whose wall needs precise current state (dc22).

**NEXT (not run; needs Ahmed's go on spend):** one single-knob variant `CARRY_TARGET_FRACTION=0.85` — compact only what
must be dropped, so the block is purely ADDITIVE and context fill matches or exceeds the stock's 64 %. Same kill test,
≈ $3. Rule: ENGAGED (unchanged) AND levels >= 8 (the base) -> 25-game wave; <= 7 -> A1 is dead as a step on this brain
and Track A moves to A2 (persistent workspace), which composes with the block rather than competing with it.
A second, cheaper rider if that runs: keep the last 2 turns verbatim regardless of the target.

## PRE-REGISTERED — `keith_carry` + `CARRY_TARGET_FRACTION=0.75` (variant kill test; written 2026-09-08 before launch)

Single knob on the arm read above: 0.5 -> **0.75**. Everything else identical (same graft, same flags, same geometry,
same games/draws), plus the 400-guard fix committed as 0538b5b (keep_last_user / no_user_after_trim /
rebuilt_from_last_user), which only fires where the pre-fix build produced a failed call.

WHY 0.75 AND NOT THE 0.85 FIRST PROPOSED: the diagnosis is context under-fill (47.7 % of budget vs the stock's 64 %),
but the knob also sets how OFTEN compaction fires. At 0.85 each compaction drops only ~4.7k tokens ≈ one history block,
so it would fire every 1-2 turns (~20-30 per run); at the live cadence (~150 s per queued call) that is ~47 % of the
7,920 s game clock and fails fit-the-clock whatever it scores. At 0.75 each compaction drops ~7.9k ≈ 2-3 blocks and
fires every ~2-3 turns (~12-15 per run, ~25 % of the live clock), while steady-state fill rises from ~50 % to ~87 % of
budget. 0.75 is the largest fill increase that keeps the arm live-eligible.

Launch: `--arm keith_carry --knob CARRY_TARGET_FRACTION=0.75 --games cd82,dc22,lf52 --draws 2 --concurrency 3
--max-calls 60 --per-game-s 7920`, fresh boot, ≈ $3. Recorded as `knob_overrides` (NOT the pinned arm env).

READS, locked before data:
* ENGAGEMENT (unchanged): compactions ≥ 1/run, failures ≤ 10 % of attempts, 0 requests over 32,768, ≥ 40 % of calls
  carry the block. Additionally CONFIRM THE KNOB BOUND: mean prompt tokens/call must rise above the 0.5 arm's 15,147
  (target ≈ 20k, the stock band) and compactions/run must land in 10-18; outside that the knob did not do what the
  arithmetic says and the level read is not interpretable.
* PRIMARY: levels over 6 runs vs the same-geometry stock base **8** (cd82 1/1, dc22 2/2, lf52 1/1) and vs the 0.5 arm's
  **6**. **≥ 8 → carry to the 25-game wave** (its own pre-registration above, gates ≥ 48 / 45-47 / ≤ 44, walls ≥ 3,
  fit-the-clock). **≤ 7 → A1 is DEAD as a step on this brain**: two configurations of the same lever, both engaged,
  neither above the base ⇒ Track A moves to A2 (persistent workspace), which composes with the block instead of
  competing with it for the window. No third dose of this knob.
* CO-PRIMARY (diagnostic, not a gate): dc22 L2, which the stock passes 4/4 in this queue-free geometry and the 0.5 arm
  passed 0/2. If 0.75 restores dc22 to ≥ 1/2 while total levels stay ≤ 7, the recency-displacement diagnosis is
  supported and the failure is the trade, not the block.
* SAFETY: GAME_OVERs/run vs 0.87; live-cap score/game; per-run VOID rules unchanged (request errors, preemptions,
  length-finish > 1 %) — 2 of 6 runs were VOID in the 0.5 arm, so read valid runs alongside the total.
* FIT-THE-CLOCK (live-eligibility, reported even though this geometry is queue-free): compactions/run × 150 s +
  play calls × 150 s must stay under 7,920 s in the live geometry.

### RESULT 2026-09-08 — `keith_carry` @ `CARRY_TARGET_FRACTION=0.75`: knob bound, levels 8 = base → WAVE (pre-registered read)

Wave `offkaggle/results/20260908T2012-keith_carry75-kill` (identity ok: kv5 on RTX PRO 6000; 0.67 h, ~$2.5; second arm
of this server session — see the caveat below). Same geometry and games as the 0.5 arm and the stock base.

**KNOB BOUND (the precondition): YES.** Mean prompt tokens/call **18,155 = 57.2 % of budget** (0.5 arm 15,147 = 47.7 %;
stock waves 20,286-20,405 = 64 %), so the under-fill is roughly two thirds closed. Compactions **11.50/run** (0.5 arm
7.33; predicted 12-15). Per compaction now 11.9 dropped messages / 17.8k input chars (0.5 arm: 19.7 / 30.7k) — smaller,
more frequent, as designed. ENGAGEMENT unchanged and clean: 69 compactions, **0 failures**, **0 requests over 32,768**
(max 26,827), 280/360 calls (77.8 %) carry the block, block 4,103 chars mean, 13.7 s each.

**PRIMARY: 8 levels / 6 runs — equal to the stock base's 8, up from the 0.5 arm's 6.**
Per (game, draw): cd82 **[2, 2]** (base [1,1], 0.5 arm [2,0]), dc22 **[0, 2]** (base [2,2], 0.5 arm [1,1]),
lf52 **[1, 1]** (base [1,1], 0.5 arm [1,1]). Rule was ">= 8 -> 25-game wave" ⇒ **the wave is earned.**
Secondary reads all move the same way: **0 VOID runs** (0.5 arm had 2), wave score 4.09 (0.5 arm 1.50), live-cap
4.18/game (0.5 arm 1.59), **WALL passes 3** of 4 attempts (0.5 arm 1; stock base 2), and **dc22's L2 — one of the 12
six-draw never-passed walls — was passed**. Efficiency is good where it clears (cd82 p1 43 actions vs a 55 baseline;
dc22 p1 45 vs 59). SAFETY: GAME_OVERs 0.67/run (base 0.87).

**HONEST WEIGHTING OF THIS RESULT.** (a) 8 vs 8 is a TIE with the base on levels, not a win; the bar was set at "not
worse than base" because the 0.5 arm was worse. The score/efficiency/wall-pass gains are real but are secondary reads
on 6 runs. (b) dc22 reads base [2,2] / 0.5 [1,1] / 0.75 [0,2] — high variance, so the earlier "monotonic decline on
dc22" reading is NOT supported; retracted. (c) This arm ran SECOND in its server session while the 0.5 arm ran FIRST;
the campaign's measured order effect is ~+0.25 levels for running first, i.e. if anything against this arm, so the
comparison is not flattered by order. (d) The 25-game wave's bar is >= 48 vs base 39.33 = +8.7 levels; a tie in the
kill test is weak evidence for clearing that, so the realistic prior on the wave is a null. It is run because the
pre-registration says so and because the secondary reads justify one $9 measurement, not because a step is expected.
(e) LIVE ECONOMICS, untested by this queue-free geometry: at the live cadence a compaction is a queued ~150 s call, so
11.5 compactions per 60 play calls means the agent trades roughly **8 of its ~52 play calls** for the block. Whether
the block is worth 8 play calls is exactly what the wave measures; the CARRY-SAFETY fit-the-clock line reports it.

**Launching the 25-game wave** under the pre-registration already written above (ENGAGED gate; PRIMARY >= 48 step
candidate / 45-47 counterbalanced redraw / <= 44 or engaged-and-flat dead; co-primary walls >= 3 of 12; safety; and
fit-the-clock), at `CARRY_TARGET_FRACTION=0.75`, **first in a fresh Modal boot** (the container from this kill test is
allowed to scale to zero first, per the standing order-control law).

### RESULT 2026-09-09 — `keith_carry` @ 0.75, 25-GAME WAVE: ENGAGED, FLAT → **DEAD** (pre-registered read)

Wave `offkaggle/results/20260908T2110-keith_carry75` (first in a FRESH boot 21:27Z, identity ok: kv5 on RTX PRO 6000;
2.24 h, ≈$9). Live geometry (25 games, conc 28, 7,920 s/game).

**ENGAGED = YES, cleanly:** 232 compactions (9.28/game), 2 failures of 234 (0.9 %), **0 requests over the 32,768
window** (max 31,304), 1,055/1,374 calls (76.8 %) carried the block, first block at call #13 (median), block 3,849
chars mean, 130.7 s each. Context fill 18,514 tok = **58 % of budget** (0.5 arm 48 %, stock 64 %) — the knob held at
scale. The mechanism did exactly what it was built to do, on 25 games.

**PRIMARY: 41 levels vs the pooled six-draw base 39.33 (sd 2.34) → +1.7 = +0.71 sd — INSIDE the band ⇒ DEAD**
(bands: ≥48 step candidate / 45-47 redraw / ≤44 or engaged-and-flat dead). Note the coincidence: the 09-08 probe
wave also landed on exactly 41 (+1.7, +0.71 sd). **CO-PRIMARY: 0 of the 12 six-draw never-passed walls passed**
(target ≥3), with all 12 present in the wave — a cleaner negative than the level total, because the walls are the
thing persistent knowledge was supposed to break.
Per game: ar25 4, ft09 5, lp85 4, re86 3, vc33 3, cd82 2, r11l 2, su15 2, tn36 2, tu93 2, and 12 games at 1;
zero-level games 3 (bp35, sc25, sk48 — the base's zero games, failing the same way).

**COSTS, measured:** requests/game 64.2 (55 play + 9.28 compaction) vs the stock's 52-55 — compaction is a **14 %
tax on the request budget**; vLLM KV **preemptions 386 vs the 110-180 regime constant** (2-3×, from the extra
concurrent requests, not from bigger contexts — fill is *below* the stock's); **GAME_OVERs 1.36/run vs the base's
0.87 (+56 %)**. The one gain is efficiency: live-cap score 9.47/game vs 8.42 (+12 %) — but per R-score-arithmetic the
entire efficiency term is worth ≤0.8 LB ever, while 98 % of the loss is levels, so it does not buy a step.
Fit-the-clock: 55 × 123 s + 9.28 × 131 s ≈ 7,980 s vs the 7,920 s cap — the arm is marginally OVER, i.e. not
live-eligible without trading away play calls.

**VERDICT: A1 (carry + compact) is DEAD as a step on this brain.** Two doses (target 0.5 and 0.75), both engaged,
neither outside the base band; the 0.75 dose fixed the 0.5 dose's regression and still bought nothing. This is the
**fifth engaged-and-flat replication** on this instrument (patch 21, yield900, probe discipline, carry 0.5, carry 0.75).
No redraw is bought: the pre-registration said dead, and the campaign's own law ("two draws of +5 prove nothing")
applies with more force to one draw of +1.7.

**WHAT THIS DOES AND DOES NOT KILL.** It kills "give the agent better *prose* knowledge that persists" — the blocks
were excellent (verified mechanics with coordinates, refuted hypotheses) and changed nothing. It does NOT directly
kill the executable half of the thesis (NVIDIA NOOA / Polyphony persist an *executable, verified* model and a
per-game toolkit, not a summary), which is A2/A3/A4. It does downgrade the prior on A2 (persistent workspace),
because A2 is 3-4 days of build on the same "the agent is model-starved" premise.
**RECOMMENDED NEXT (Ahmed's call, not started):** before spending 3-4 days on A2, run the reopen audit's #1 — the
**Stage-0 rerun on Flash-Next** (~1 day, ≈$5, `docs/research-2026-09-02/S0-stage0-kill-test.md` §1-3): given recorded
frontier transitions plus a backtest tool, can the deployed brain emit a backtest-green executable model at all? That
lane was killed on Qwen3.6/3.8-27B whose failure was thinking non-termination — a failure mode Flash-Next does not
have. It is decisive for the whole executable-model lane (A2 included): ≥2/3 green ⇒ build A2/protocol-lite with
evidence; 0/3 ⇒ A2 is dead by construction and Track A becomes A3/A4 (port NOOA / Polyphony as alternative loops)
plus Track D (Oct-1 absorption).

---

## 2026-09-09 — A3/A4 GATED OUT, TRACK A CLOSED, TRACK D IS NOW THE ONLY OPEN TRACK

**NOOA — DEAD on the clock contract.** Its loop assumes 1,200 s per turn, 5,000 env steps, 20 actions/turn and
*unlimited* turns. Our live geometry is 7,920 s per game and ~52 model calls. Not configurable around.

**Polyphony (`Mininglamp-AI/polyphony-arc-3`) — DEAD on COMPUTE, not on the clock.**
Full gate: `docs/research-2026-09-09/GATE-polyphony-clock.md`.
Licence **MIT** (read from LICENSE; the GitHub API's `NOASSERTION` is a detection quirk, same as NOOA — do not trust
that field). Loop shape is *literally our Stage-1*: a Python policy accepted only when it reproduces the real
transitions exactly, then searched for a plan. Built for deployable open-weight models (default Qwen3.6-27B).
Clock **passes**: `agent.py` — "TIME is the primary stop authority under a bounded wall-clock budget"; the old step
budget and stuck rule were removed; `--per-game-deadline-s` defaults to 1800 s, inside our 7,920 s.

It fails on decode tokens, and the failing metric is hardware-independent:

| | calls | completion tokens | tokens/call |
|---|---|---|---|
| Stage-1 green model (4 runs, mean) | 4 | **65,468** | **16,367** |
| stock, per WHOLE game | 64.2 | **83,094** | **1,293** |

A Stage-1 call is **12.7× fatter** than a stock call, and **one verified level model eats 79 % of an entire game's
decode budget** (best case 49 %, worst 142 %). Call *count* is fine (2-9 vs our ~52) — tokens per call is what kills
it, and long generation is intrinsic to writing a transition model. The incremental path does not rescue it: the
9-call run still burned 13,136 tokens per round. Their own competition config (`--parallel-nums 5
--per-game-deadline-s 14400` on `--tensor-parallel-size 8`) is ~6.4 GPU-hours per game against our 0.08.
Cannot be bought back: concurrency 28→5 gives 5.6× per-game compute but covers 5/28 of the games while the score
averages over all 110; the session cap leaves at most ~1.35× more wall clock.

**This is a different death from A1/A2.** Polyphony is not refuted — it is *priced out of our hardware*. Reopen only
if per-game compute changes by ~an order of magnitude. **New standing cost gate for any port: completion tokens per
game.** It is concurrency- and hardware-independent and it decides these questions in an hour, for free.

**TRACK D OPENED (the plan's §9 rule: A1 and A2 both flat ⇒ fall back to A3/A4 and D; A3/A4 now gone too).**
Built and committed 09-09, no spend: **`offkaggle/absorb_kernel.py`** — `stage` pulls a released kernel and lays down
a byte-identical push bundle plus `ATTEST.json`; `diff` compares the four serving-regime keys so a silent regime
change cannot slip through; `verify` re-checks a bundle against its attestation so after a graft you can say exactly
which hash moved. It never pushes. Verified rather than asserted: it reproduces `submission/_keith_copy/ATTEST.json`
(our live 3.25 base) and the hand-built `push_bothmounts/kernel-metadata.json` exactly, 13/13 tests, and a live
end-to-end run against the real Kaggle API on 09-09 returned the same two hashes recorded on 09-02 — which also
confirms **keith V14 is unchanged since**. Checklist: `docs/TRACK-D-absorption-checklist.md` (H+0..H+24, with the
cost gate placed *before* port enthusiasm and pre-registration required before reading any wave).

---

## PRE-REGISTERED — `keith` at CADENCE geometry (conc 6 / 1767 s), written 2026-09-09 BEFORE data

**Hypothesis.** A turn can hold a SECOND model call only if the call returns before the 60 s turn
budget. At the live cadence (conc 28, e2e 145 s) it never does: calls/turn 1.02, the within-turn
analysis→act loop never runs, and these walls fell 0/14. At conc 3 (e2e 16-20 s, calls/turn 1.66) the
UNMODIFIED base passed cd82/dc22/lf52 L2 **4/4** on matched call counts. Conc 14 (e2e 68 s) missed the
threshold by 8 s and read 32 vs 36 — consistent with the account, i.e. a near miss, not a refutation.

**Knob: geometry only. No graft. Stock bytes, `keith` arm, unchanged analyzer env.**

`--arm keith --games all --draws 1 --concurrency 6 --per-game-s 1767`

**Why 6 and why 1767.** Server throughput is fixed (~0.17 req/s; 3 running slots under the 5 GiB KV
reservation), so **calls/game ≈ 50 at EVERY concurrency** — lower concurrency does not buy calls, it
buys cadence in a shorter window. Setting `per_game_s = 32400 × conc / 110` keeps the 110-game run
inside the pinned notebook budget, so this arm is **live-legal as a two-constant change to the flown
v4 notebook**. conc 6 → est e2e ≈ 36 s (clear of the 60 s budget, unlike conc 9's ≈53 s) while keeping
6 requests in flight against 3 running slots so the GPU stays fed. 25-game wave ≈ 2.05 h, ≈$9.

**ENGAGEMENT (checked FIRST; failure ⇒ VOID, not dead — the mechanism never ran):**
* median e2e per call **< 60 s**, and
* **calls/turn ≥ 1.5** (live base 1.02; conc-3 reference 1.66), and
* calls/game **≥ 40** (guards the underfed-GPU confound at low concurrency; expected ≈50).

**PRIMARY — levels on the 25-game set, vs the pooled base 39.33, sd 2.34:**
* **≥ 48** → step candidate → propose a live flight (Ahmed's go required)
* **45–47** → positive, redraw counterbalanced in a fresh session before any claim
* **≤ 44** → DEAD by rule, even if engaged. Engaged-and-flat would be the sixth replication.

**CO-PRIMARY — walls.** The 12 never-passed walls, all 12 present: **≥ 3 passes** required for a step
claim. The conc-3 instrument's 4/4 on cd82/dc22/lf52 L2 predicts these specifically; report them named.

**SAFETY / diagnosis lines to record regardless:** e2e distribution, calls/turn, calls/game, yields %,
actions/game, GAME_OVERs/run, vLLM running/waiting and preemptions, zero-level games, wall clock.

**What each outcome means.** Engaged + ≥48 → cadence is the lever, it is live-legal, and adaptive
allocation and the level-1 laboratory build on top of it. Engaged + ≤44 → cadence does NOT convert on
the full geometry, the conc-3 wall result was wall-specific or small-n, and the campaign moves to the
model axis. Not engaged → void, re-run at conc 4 or 3.

**Queued behind it (DIAGNOSTIC, not deployable):** the same games at conc 6 with the per-game clock
left at 7,920 s, which multiplies calls/game ~4.5x by extending wave length. That arm separates
"budget converts" from "cadence converts". Only worth running if the deployable arm is engaged.

### ADDENDUM to the cadence pre-registration — written 2026-09-09 while the arm was still in preflight

Recovered from data we already had (`20260903T0553-kv10-keith` vs the same-session base
`20260902T2121-keith`), i.e. NOT from the arm now running. **The gates above are unchanged.**

| wave | levels /25 | calls/game | calls/turn |
|---|---|---|---|
| base | 36 | 55.80 | 1.02 |
| KV10 (double the KV reservation) | 43 | **82.16** | **1.03** |

**BUDGET DOES CONVERT, BUT SUBLINEARLY: +47% calls/game bought +19% levels** (elasticity ≈0.4).
That is the "does budget convert to levels" question, answered for free from an old wave. At this
elasticity, doubling levels/game needs roughly **5x** the calls, which is not purchasable on one GPU.
So the whole throughput/allocation family is worth tens of percent, not a step — consistent with
KV10's recorded verdict (+0.28 lv/game, no step) and with the 9-06 half-concurrency wave.

**Crucially, KV10's calls/turn stayed 1.03.** Doubling the running slots halves e2e to ~90 s, which is
STILL above the 60 s turn budget, so turns remained single-call. KV10 and conc-14 are therefore the
same null: both improved throughput, neither crossed the cadence threshold. The cadence account
retrodicts both.

**What this sharpens for the arm now running.** If conc 6 behaves as a pure throughput lever it should
land near 43 levels — which the pre-registered band already calls DEAD (≤44). To clear 48 it must do
something throughput alone cannot, which is exactly the claim under test: multi-call turns restoring
the within-turn analysis→act loop. The 4/4-vs-0/14 wall result is the only evidence that it is
structurally different rather than just faster. **This makes the co-primary wall count the more
informative read of the two.**

## RESULT — CADENCE arm (`keith`, conc 6 / 1767 s; 19:37→22:03 UTC, 146 min, ≈$9) — **DEAD by rule**

Artifacts `offkaggle/results/20260909-193735-regime-keith`. Read by
`offkaggle/read_cadence_arm.py`, which was written and committed BEFORE this data existed.

**ENGAGEMENT — PASSED on all three gates, cleanly, and better than projected:**

| gate | want | got |
|---|---|---|
| median e2e per call | < 60 s | **26.64 s** |
| calls per turn | ≥ 1.5 | **1.60** (base 1.02; conc-3 reference 1.66) |
| calls per game | ≥ 40 | **56.76** (base 55.80) |

The arm was also *healthier* than base: 1,434 requests vs 1,396, and **preemptions 4.1 % vs the
base's 10.2 %**. (Every run carries the runner's `VOID` flag, but that is the 3-wall instrument's
blanket rule — any wave-level preemption voids every run — and the base waves trip it harder. It is
not a confound against this arm.) **My mid-flight worry that conc 6 would underfeed the GPU was WRONG:
throughput held and calls/game slightly exceeded base.**

**PRIMARY: 34 levels vs pooled base 39.33 (sd 2.34) = −5.33 = −2.28 sd → DEAD.** Not flat — *worse*,
and outside 2 sd. Zero-level games 5 vs the base's 2.
**CO-PRIMARY: 1 of the 12 never-passed walls (dc22 L2), target ≥ 3.**

**CORRECTION I OWE (my error, in the pre-registration and in what I told Ahmed).** I wrote that the
conc-3 instrument "passed cd82/dc22/lf52 walls 4/4". The source line says only *"the unmodified base
passes **dc22's** wall 4/4 within 60 calls; in the queued live regime it passed 0/4"*. It was one wall,
dc22, across two clock conditions — not three walls. I generalised a single-wall result into three.
Note what that means for this read: **dc22 L2 is exactly the wall that DID pass here.** The specific
finding replicated. It simply did not generalise to the other 11 walls or to total levels.

**MECHANISM.** Actions were unchanged (156/game vs 154; 2.74 actions/call vs 2.76). What changed is
turns: **35.5 turns/game vs the base's 54.7, a 35 % cut**, because the same call budget was spent
1.60-at-a-time instead of 1.02. The agent bought within-turn deliberation and paid for it in turns,
and at most one acting tool call happens per turn.

**AN IDENTIFIABILITY LIMIT I SHOULD HAVE STATED UP FRONT.** clock = e2e × calls. You cannot hold both
the per-game clock and calls/game while varying cadence — fix calls and the clock shrinks (this arm,
7,920 → 1,767 s); fix the clock and call volume rises (confounded with budget). **So this arm is not a
single-knob test**: it changed cadence AND cut the wall clock 4.5×, and the harness stamps
`time_remaining_seconds` into every tool result as a pacing hint. A rushed-pacing explanation for the
−5.33 is live and unexcluded.

**VERDICT.** Cadence as a *live-legal* change is DEAD: the only live-legal way to buy it also cuts the
per-game clock, and the package reads −2.28 sd. Whether cadence *itself* is neutral or positive is
still open, and only separable by the non-deployable diagnostic arm (conc 6 at the unchanged 7,920 s
clock, ~9.2 h, ≈$40, which raises calls/game ~4.5× and so confounds with volume anyway — its value is
that it bounds the clock explanation, not that it isolates cadence).

Together with the KV10 elasticity (+47 % calls → +19 % levels), the throughput/cadence family is now
closed as a source of a step: every reachable point on the curve is worth tens of percent at best, and
the live-legal points are negative.

---

## PRE-REGISTERED — arm `keith_fx` (harness-computed dynamics), written 2026-09-10 BEFORE data

**The measurement it comes from** (`docs/research-2026-09-10/R-what-our-agent-actually-does.md`):
across **2,047 tool calls made on levels the run went on to CLEAR**, our agent forward-simulated a
next state **0 times (0.0 %)**, ran any search in 2.2 %, verified a prediction in 0.2 %, and touched
`transitions` — the harness's own before/after affordance — in 3.2 %. Median 9 lines of code, median
action-list length 1. **It is a reactive perceive-and-act loop.** Six experiments say we cannot talk
it into being anything else, and A2 proved we cannot tool it into one either (0 verifier calls in 358).

**So this arm does not ask.** The harness folds every real `(before, action, after)` triple into an
object-level effect table in pure python — objects matched by the stock `segment_layer`'s
translation-invariant hash — and appends a capped block to the user prompt. **Zero extra model calls.
No behaviour change is required for the information to arrive.**

**External convergence (found independently, after the graft was built):** PRO-LONG's harness appends
to a log after every action *automatically* — the agent never decides to write, only to read — and its
full-log vs no-log ablation reads 41.2 % vs 24.0 % pass@1. That is the same structural answer to A2's
failure: we removed the wrong side of the problem.

**Offline validation on real game data (before any spend):** replayed over recorded events, the table
recovers dc22's movement rule outright — `UP moves 1 obj by (-2,+0) [34/37]`, `RIGHT (+0,+2) [19/22]`,
`LEFT (+0,-2) [10/10]` — and g50t's `RIGHT/LEFT ±6 [44/84, 37/64]`. Appear/vanish is suppressed when
near-universal, because 18 of 25 games tick a HUD every action (the standing HUD law) and a
near-universal effect carries no information.

**Knob:** plain `keith` base + the `EFFECTS_*` keys + `graft_effects`. Nothing else differs.
`--arm keith_fx --games all --draws 1` at the live geometry (conc 28, 7,920 s).

**ENGAGEMENT (checked FIRST; failure ⇒ VOID, not dead).** Measured post-hoc by grepping the
transcripts' `[USER PROMPT]` sections for `Observed action effects`:
* the block appears in **≥ 60 %** of turns (it cannot appear before the first action, so 100 % is
  impossible), and
* in **≥ 50 %** of games the block carries **≥ 2 action lines** (a one-line table is not dynamics).

**PRIMARY — levels on 25 games vs pooled base 39.33, sd 2.34:** **≥ 48** step candidate / **45–47**
redraw / **≤ 44** DEAD by rule, engaged or not.
**CO-PRIMARY — ≥ 3 of the 12 six-draw never-passed walls**, all 12 present.
**SAFETY:** prompt tokens/call (expect ≈ +250 vs base 20,061), calls/game vs 55.8, GAME_OVERs/run vs
0.87, zero-level games vs 2, e2e/call, wall clock.

**What each outcome means.** Engaged + ≥48 → the first step of the campaign, and it says the gap was
an information-derivation gap the harness can close for free. Engaged + ≤44 → the **seventh**
engaged-and-flat replication, and the strongest possible evidence that supplying the missing
computation is not enough because the agent will not *act* on a model either; that would close the
"help it model" family completely and leave Track D as the only lane. Not engaged → void, fix the
block plumbing and re-run.

## RESULT — arm `keith_fx` (harness-computed dynamics; 08:19→10:35 UTC, ≈$9) — **DEAD by rule**

Artifacts `offkaggle/results/20260910-081906-regime-keith_fx`.

**ENGAGEMENT — PASSED, emphatically.** The block appeared in **97.2 % of turns** (gate ≥ 60 %) and
**25/25 games** carried a block with ≥ 2 action lines (gate ≥ 50 %). Spot-checked live mid-wave: on
dc22 at turn 10 the block read `RIGHT x9: moves 1 obj by (+0,+2) [8/9]`, `LEFT x7: (+0,-2) [7/7]`,
`DOWN x3: (+2,+0) [3/3]`, plus which MOUSE clicks were inert. **The information was correct, arrived
free, and arrived on essentially every turn.**

**PRIMARY: 35 levels vs pooled base 39.33 (sd 2.34) = −4.33 = −1.85 sd → DEAD by rule.**
**CO-PRIMARY: 1 of 12 never-passed walls (sb26 L2), target ≥ 3.**
Zero-level games **7 vs the base's 2**. actions/game 141 vs 154; actions/call 2.56 vs 2.76.

**PROCESS NOTE (my error, caught and corrected).** I first ran this through
`read_cadence_arm.py`, which applies the CADENCE arm's engagement gates (median e2e < 60 s,
calls/turn ≥ 1.5). Those are meaningless here — `keith_fx` runs at the LIVE geometry by design, so
147 s / 1.02 is the base condition, not a failure — and the script printed **VOID**. The correct
gates are this arm's own, written in its pre-registration, and they PASS. The script has since been
guarded so it refuses arms it was not written for.

**VERDICT — this closes the "help the agent model" family.** It was the strongest available version
of the idea: rather than ask the agent to build a model (A2: 0 verifier calls in 358), or direct it to
(0 models in 24 calls), the harness computed correct dynamics itself, for zero model calls, and put
them in front of the agent on 97 % of turns. Levels went **down**. This is the **seventh**
engaged-and-flat replication and the second that is engaged-and-*negative*.

Read together with the corpus measurement (0 of 2,047 successful calls forward-simulate) the
conclusion is sharper than "it did not help": **the agent does not act on a dynamics model even when
one is handed to it, correct and free.** The missing thing is not the information and not the
ability to derive it — it is the disposition to plan with it, and that is not reachable from the
harness side. Independent corroboration from the 09-10 sweep: Tycho measured the same shape, where
automatic repair produced models that *"reproduce observed transitions much more accurately, yet
reaches only 83.07"* against 88.49 — **world-model fidelity does not buy action quality.**

**Consequence for the queue.** `keith_fxs` (effects + sweep) is demoted: its effects half is now
measured dead. `keith_sweep` remains a *different* knob — it changes the ACTION distribution rather
than the prompt content, which is what AutumnBench actually diagnoses — but this result lowers its
prior materially and that should be said before it is run, not after.

---

## PRE-REGISTERED — `keith_budget3x` (pure budget knob), written 2026-09-10 BEFORE data

**The one unknown that bounds every remaining family.** Throughput levers, adaptive allocation and
triage all convert budget into levels through a single exchange rate, and we have measured that rate
**once**: the KV10 wave bought +47 % calls for +19 % levels, an elasticity of ≈0.4. Everything built
on it — including the triage simulation's headline — is an extrapolation from one point.

**Why this design and not the one I first specified.** My earlier "diagnostic arm" was conc 9 at the
unchanged clock. That changes TWO things at once: it triples calls/game *and* drops e2e from 145 s to
~53 s. The 09-10 cadence result (−2.28 sd) proved e2e is **not** inert, so that design would confound
budget with cadence and could not answer this question. Instead:

`--arm keith --games all --draws 1 --concurrency 28 --per-game-s 23760`

With 25 games at concurrency 28 **every game runs in parallel**, so e2e stays at the base 145 s and
the **only** thing that moves is the per-game clock. Pure budget knob, cadence held constant.
≈164 calls/game (2.9× the base 55.8), wall ≈6.6 h, ≈$27.

**NOT LIVE-LEGAL and not intended to be.** 3× the per-game clock cannot fit 110 games in the pinned
32,400 s notebook budget. This is a measurement, not a candidate. Nothing here can be flown.

**ENGAGEMENT (checked first; failure ⇒ VOID):** calls/game **≥ 140** (target ≈164; the arm must
actually receive the budget) and median e2e **within 120–175 s** (cadence genuinely unchanged).

**PRIMARY — levels on 25 games, read as an ELASTICITY, against pooled base 39.33:**
* **≥ 60 levels** → elasticity holds near 0.4 out to 3×. Budget converts. The triage/allocation family
  is a genuine route to a step and should be built; the aggressive policies in
  `docs/research-2026-09-10/R-triage-family-bounded.md` become credible rather than extrapolation.
* **46–59** → partial conversion (elasticity ~0.1–0.3). Triage worth roughly +10–30 %, worth building
  but not a step on its own.
* **≤ 45** → **budget does NOT convert at scale.** The entire throughput/allocation/triage family is
  capped at the low end, the KV10 point was the flat part of a curve that was already bending, and —
  taken with today's two engaged-and-negative results and the blocked model axis — **no harness lever
  remains**, which makes Track D absorption the whole campaign rather than the leading option.

**SAFETY / DIAGNOSIS to record regardless:** calls/game, e2e distribution, calls/turn, actions/game,
zero-level games, GAME_OVERs/run, preemptions, and the implied elasticity
`(levels/39.33 − 1) / (calls/55.8 − 1)` — which is the number this wave exists to produce.

**Note on what a NULL here would mean.** It would be the cleanest possible statement of our ceiling:
the agent, given three times the budget at identical cadence, does not go deeper. That is a
capability statement, not a harness one, and it would be worth more than a positive result on any
single graft.

### CORRECTION to the `keith_fx` write-up (same day, before the next arm reads out)

I wrote that "levels went **down**". That over-claims. The pre-registered verdict is **DEAD** — it did
not clear 48 — and that stands. But **"worse than base" is NOT established by one draw.**

Checked directly: within the wave, correlation between how rich a game's effect block ever got
(max action lines) and its final levels is **−0.086 over 25 games**, i.e. no dose-response gradient.
The bucketing is degenerate anyway — **24 of 25 games saw a 7+-line block** — so the arm delivered a
full-width table essentially everywhere and there is no dose variation to exploit. A −1.85 sd point
estimate on a single draw (p ≈ 0.06) is comfortably inside what a low draw produces.

**The defensible statements are:** the mechanism engaged on 97.2 % of turns with correct content; the
arm did not clear the bar; and there is no evidence the block itself is harmful. The same caveat
applies to the cadence arm's −2.28 sd (p ≈ 0.02, one draw) — stronger, but still one draw.

This does not change either verdict or the conclusion that the "help the agent model" family is
closed, because that conclusion rests on the *absence of gain* across four independent attempts
(A2 tool, A2 directed, fx free-and-correct, plus Tycho's external fidelity result), not on harm.

## RESULT — `keith_budget3x` — **VOID (instrument failure), with informative salvage**

Killed at +280 min. **17 of 25 games crashed with `OSError: [Errno 28] No space left on device`**,
all within a few minutes of +194 min. The wave cannot be read against its gates.

**ROOT CAUSE, and it is a finding in its own right.** `write_runtime_state` runs after **every**
action and re-serialises the **entire** history with `indent=2`. At ~17 KB per history entry that is
**O(n²) bytes written per game**:

| | actions | final state file | total bytes written |
|---|---|---|---|
| base game | 154 | 2.6 MB | 0.19 GB |
| this wave's crashed games | 371 | 6.2 MB | **1.12 GB** |
| tn36, the worst | 959 | 15.9 MB | **7.46 GB** |

Across 25 concurrent games at ~371 actions that is **~28 GB of write churn**, which is why ENOSPC
fired with 30 GB showing free — it is churn against APFS purgeable space, not steady-state storage.
**It scales with the square of the action count, so it is invisible at the live budget (154 actions)
and fatal at 3x.** Any future budget-increasing experiment must patch this first.

**SALVAGE — the numbers for the 17 games that ran are real, and they point one way.**

| | actions/game | levels/game |
|---|---|---|
| base | 154 | 1.57 |
| this wave, before the crash | **371 (2.4x)** | **1.71 (1.09x)** |

**Implied elasticity 0.06**, against the **0.4** measured by KV10 at +47 %. The per-game detail is
starker: **tn36 spent 959 actions and cleared ZERO levels**; sp80 524 → 0; bp35 498 → 0; wa30 785 → 1.
Meanwhile lp85 cleared 5 in 166 actions and re86 5 in 437. Extra budget flowed overwhelmingly into
games that were already stuck, exactly as the doomed-tail analysis predicted (62.5 % of actions land
on a level never cleared), and it did not convert.

**STATUS: this is NOT the answer to the pre-registered question, and must not be quoted as one.**
The games crashed at roughly half their intended clock and are not a clean 25-game read. But the
direction is strong enough to matter: if it holds, **elasticity decays hard between +47 % and +141 %**,
which caps the throughput / allocation / triage family at the low end and makes the aggressive triage
policies in `R-triage-family-bounded.md` extrapolation rather than opportunity.

**NEXT, in order:** (1) patch the O(n²) state write — cheap, and required before any budget arm;
(2) re-run this arm cleanly; (3) only then read the elasticity. Disk on this machine sits at 93 % on
the Data volume, so headroom must be confirmed before the re-run.

## SMOKE — `keith_budget` instrument test (3 games, conc 3, 1 h clock, ≈$5) — **PASS**

Run per Ahmed's 09-10 instruction to test small before spending again. Design: 3 games at
concurrency 3 collapses the queue (e2e 9.7–12.9 s vs the live 145 s), so a **1-hour** clock still
drives games into the high-action regime that broke the previous wave — the failure mode is
reproduced at a fifth of the cost.

`graft_compactstate: compact_json+batch_write(yes): OK`

| game | levels | actions | e2e | state |
|---|---|---|---|---|
| cd82 | 2 / 6 | 495 | 12.4 s | gave_up (clock) |
| dc22 | 0 / 6 | 1,091 | 12.9 s | gave_up (clock) |
| tn36 | 0 / 7 | **3,659** | 9.7 s | gave_up (clock) |

**INSTRUMENT: PASS. Zero ENOSPC, zero crashes, all three ran to their full clock, whole run
directory 101 MB.** tn36 reached **3,659 actions** — 24× the live base of 154, and nearly 4× the 959
that crashed the previous wave — with no disk pressure at all. The O(n²) write is fixed.

**SIGNAL (secondary, n=3, and cadence-confounded — the queue is collapsed here):** these three games
average 1.18 / 0.91 / 0.91 levels at base, ≈3.0 levels total. With **3–24× the actions** they produced
**2**. **tn36 spent 3,659 actions and cleared nothing.** That is a third independent pointer, after
KV10 (elasticity 0.4 at +47 %) and the void wave's salvage (0.06 at +141 %), that budget stops
converting well before 3×. It is not a verdict: n=3, hard games, and the low-latency confound.

## RESULT — `keith_budget` 3x clean (conc 28 / 23,760 s; 19:25→02:01 UTC, 6.6 h, ≈$27) — **BUDGET CONVERTS**

Instrument held: **zero ENOSPC, zero crashes**, 24 gave_up + **1 WON**, disk landed at 14 GB
(guard threshold 8 GB, never tripped). The `graft_compactstate` fix did its job.

**ENGAGEMENT — PASS on both gates.** calls/game **169.8** (gate ≥140, = **3.04×** the base 55.8);
median e2e **128.8 s** (gate 120–175 s, so cadence genuinely held and this is a pure budget knob).

**PRIMARY: 56 levels vs pooled base 39.33 (sd 2.34) = +16.67 = +7.12 sd → band PARTIAL (46–59).**

| | base | 3x arm | ratio |
|---|---|---|---|
| calls/game | 55.8 | 169.8 | 3.04× |
| actions/game | 154 | 618 | 4.01× |
| levels (25 games) | 39.33 | **56** | 1.42× |
| mean score/game | 6.40 | **13.76** | **2.15×** |
| zero-level games | 2 | 2 | — |

**IMPLIED ELASTICITY 0.21** (KV10 measured 0.40 at +47 %). So elasticity *decays* — 0.40 → 0.21 — but
budget **does** convert, substantially, out to 3×. The depth distribution moves right: games stuck at
one level fall 14 → 8, and 5–6-level games go 0 → 3.
**`ft09` WON OUTRIGHT: 6/6 levels, score 100.00, in 105 actions** — the first outright win in this
campaign's rig history.

## CORRECTION — I called this wrong twice yesterday, and it matters

After the void wave I wrote that its salvage "points one way", quoting an implied elasticity of
**0.06**, and then called the 3-game smoke a "third independent pointer" that budget stops converting
before 3×. **Both readings were wrong.** The clean number is **0.21 with the score more than doubling.**

Why the salvage misled: those games were killed by ENOSPC at roughly **half** their intended clock,
and at high budget the levels arrive **late** — the whole point of the extra budget is the back half of
the game. Reading a wave cut off mid-run as if it were a budget measurement was the error, and I
should have labelled it unreadable rather than directional. The smoke (n=3, hard games, collapsed
queue) never carried the weight I gave it either.

## WHAT IT DOES AND DOES NOT UNLOCK

**It is NOT live-legal and cannot be made so.** 3× the per-game clock needs 3× the total GPU-hours;
the Kaggle session is pinned at 32,400 s. This arm is a measurement, exactly as pre-registered.

**Redistribution inside the fixed budget captures only a fraction of it.** Re-running the triage
simulation with the now-**measured** 0.21 (every policy below sits inside the measured uplift range,
so this is no longer extrapolation): best net **+1.11** on a 9.12 base = **+12 %**. The reason is
structural — triage can only move budget between games, and the total is fixed, while the conversion
is sublinear.

**The finding that matters:** the agent is **not** at a capability ceiling on depth. Given 3× the calls
at identical cadence it goes markedly deeper, clears a game outright, and doubles the score. What
bounds us is total compute, not comprehension — which is the opposite of the conclusion the previous
six engaged-and-flat results were pointing toward, and it re-opens *anything that buys effective
calls per game* as the lever class worth attacking.

---

## PRE-REGISTERED — `keith_ctx12` (trade context for calls), written 2026-09-11 BEFORE data

**Where this comes from.** The 3x budget arm proved budget **converts** (169.8 calls/game → 56 levels,
+7.12 sd, elasticity **0.21**) but cannot fly: tripling the clock needs 3× the GPU-hours. Live,

```
calls/game = 32,400 x running_slots / (110 x service)        [concurrency cancels]
```

so the only live-legal lever is **running slots = KV / average sequence length**. Shrinking the
context window shortens sequences, fits more in the 5 GiB KV, drains the queue faster, and buys more
calls from the *same* GPU-hours.

**Six engaged-and-flat results say context CONTENT does not move levels. None of them tested
REMOVING it to buy calls.** That is the untested direction, and it is a one-env-var change.

**Knob:** `LOCAL_ANALYZER_CONTEXT_WINDOW` 32768 → **12288**. Nothing else differs from `keith`.
Live geometry otherwise (conc 28, 7,920 s). ≈2.2 h, ≈$9. **Live-legal**: it is one constant in the
flown notebook.

**Projection (to be falsified, not assumed):** budget 11,264, prompt ≈7,100, slots ≈8.5,
**≈140 calls/game (2.5×)**, and at elasticity 0.21 ≈**51.8 levels**. History drops to ≈7,285 tokens,
about 3 retained turns.

**MECHANISM GATE (checked FIRST; this is the part most likely to be wrong).** My slots model is a
projection, never measured. Required: **calls/game ≥ 90** (a real rise from 55.8) and **median prompt
tokens ≤ 9,000**. If calls do NOT rise, the slots model is wrong and the arm is VOID — that is a
cheap, valuable negative about how this server actually allocates KV, and it kills the whole
context-for-calls idea without any claim about levels.

**PRIMARY — levels on 25 games vs pooled base 39.33 (sd 2.34):** **≥ 48** step candidate / **45–47**
redraw / **≤ 44** dead.

**THE REAL RISK, stated up front.** Elasticity 0.21 was measured at *constant* context with a longer
clock. Here each call is **cheaper but dumber** — ~3 retained turns instead of ~10. If call quality
falls faster than call count rises, levels drop even though the mechanism engages. That outcome is
the informative one: it would mean context is load-bearing after all, which the six flat results
never actually tested.

**SAFETY:** calls/game, median + max prompt tokens, calls/turn, e2e, actions/game, zero-level games,
eviction share, GAME_OVERs/run.

**QUEUED:** `keith_ctx8` (window 8192, projected ≈221 calls/game and ≈63.7 levels) runs only if
ctx12's mechanism gate passes — a dose-response pair is worth far more than either point alone.

## RESULT — `keith_ctx12` (context 32768 → 12288; 07:58→10:12 UTC, ≈$9) — **DEAD, catastrophically**

**MECHANISM GATE: PASSED. The slots model was right.**

| | base | ctx12 |
|---|---|---|
| median prompt tokens | 20,061 | **6,674** |
| calls/game | 55.8 | **108.4 (1.94×)** |

Shrinking the window genuinely buys calls, almost exactly as projected. That part of the theory holds.

**PRIMARY: 12 levels vs pooled base 39.33 (sd 2.34) = −27.33 = −11.68 sd.** The worst read in the
campaign by a wide margin. **15 of 25 games cleared ZERO levels** (base 2).

**WHY — the pre-registered risk, and it is not subtle:**

| | base | ctx12 |
|---|---|---|
| actions/game | 154 | **29 (0.19×)** |
| **actions per call** | **2.76** | **0.27** |

Nearly twice the calls, and each one almost never acts. A 10× collapse in actions-per-call swamped
the 1.94× gain in calls.

**MECHANISM OF THE COLLAPSE (measured, not guessed).** At window 12,288 the context budget is 11,264.
Subtract the 3,170-token system prompt and a ~745-token user prompt and **~7,349 tokens remain for
history**. Observed completion tokens reach **10,439 on a single call** (mean 1,044). **One
long-reasoning turn does not fit in the entire history budget.** The trimmer then evicts constantly —
this is the same edge the A1 work already found, where the stock trim can strip back to the system
message alone when one turn's own assistant+tool pair exceeds the budget. The agent spends its extra
calls thrashing rather than acting.

## THE CORRECTION THIS FORCES ON MY OWN REASONING

I justified this arm with: "six engaged-and-flat results say context CONTENT does not move levels;
none tested REMOVING it." That framing was right about the gap and **wrong about the inference**.
Those six results shaped or added content **within a fixed, ample window**. This arm removed
**capacity**, and capacity turns out to be strongly load-bearing: −11.68 sd.

**Content is worth little; capacity is worth enormously.** Those are different claims and I had been
treating the first as evidence about the second.

**`keith_ctx8` IS CANCELLED, not queued.** Its window is smaller still, so the dose-response is
already answered in the direction that matters and running it would spend $9 to confirm a
catastrophe. The pre-registration said the pair was worth more than either point; that was true only
if the first point was survivable.

## WHERE THIS LEAVES THE LIVE-LEGAL PATH

Live, `calls/game = 32,400 × running_slots / (110 × service)`. Slots = KV ÷ average sequence length,
so there are exactly two ways to raise them:
* **shrink sequences** — measured here, **catastrophic**;
* **raise KV** — measured on the rig as +0.28 lv/game, and it **OOM'd on the Kaggle card** at 8/10 GiB.

**Both routes are now closed.** The 3× budget result (56 levels, ft09 won outright) is real and
remains **unreachable live**, because there is no live-legal way to buy the calls it needs. That is
the honest state: we know what would work and we cannot afford it.

---

## PRE-REGISTERED — `keith_ctx16`, written 2026-09-11 BEFORE data

**I over-closed yesterday and this corrects it.** After `keith_ctx12` read −11.68 sd I wrote that
"both routes to more live slots are now closed". That generalised **one point** into a route.

**The collapse at 12,288 has a MECHANISTIC threshold, not a smooth gradient.** Context budget minus
the 3,170-token system prompt and a ~745-token user prompt leaves the history budget. The longest
single observed completion is **10,439 tokens**:

| window | history budget | fits one max-length turn? | tested |
|---|---|---|---|
| 32,768 | 27,829 | yes | yes (base) |
| **16,384** | **11,445** | **yes** | **NO** |
| 12,288 | 7,349 | **no — thrashes** | yes (−11.68 sd) |
| 8,192 | 3,253 | no | no (cancelled) |

**16,384 sits on the other side of the threshold.** At 12,288 a single long-reasoning turn cannot fit
in the whole history budget, so the trimmer evicts constantly and actions/call fell 2.76 → 0.27.
At 16,384 that specific failure cannot occur. This is a different regime, not a milder dose.

**Knob:** `LOCAL_ANALYZER_CONTEXT_WINDOW` 32768 → **16384**, nothing else. Live geometry, live-legal,
≈2.2 h, ≈$9.

**MECHANISM GATE (first):** calls/game **≥ 80** (base 55.8) AND **actions/call ≥ 2.0** (base 2.76;
ctx12 collapsed to 0.27). **The second is the real test** — it is the quantity that failed last time,
and if it collapses again the threshold theory is wrong and the whole shrink-sequences route is
genuinely closed, which is worth knowing for $9.

**PRIMARY — levels vs pooled base 39.33 (sd 2.34):** ≥48 step / 45–47 redraw / ≤44 dead.
Projection ≈103 calls/game and ≈46 levels — *below* the bar on its own, which is expected and fine:
this arm exists to establish whether the route is alive, not to clear the bar alone.

**WHAT ELSE I WRONGLY CLOSED, recorded so it is not lost:**
1. **KV 6–6.5 GiB.** The Kaggle law says headroom is "≤1–2 GiB above the public 5 GiB" because 8 and
   10 GiB OOMed. **6 and 6.5 were never tried** and are inside the stated headroom.
2. **System-prompt trim.** 3,170 tokens re-sent on every call; cutting it raises history capacity and
   shortens sequences at the same time. Never tested.
3. **Service time.** `calls/game = 32,400 × slots / (110 × service)`. I only ever attacked slots.
   Service is 17.8 s, decode-dominated, and the regime pins `num_speculative_tokens=3`. Raising MTP
   attacks the denominator. **I never mentioned this axis at all.**

**Combinations multiply slots.** Projected: ctx16k + KV 6.5 GiB ≈ 134 calls/game ≈ **50.9 levels**
(clears); adding a 2k system-prompt trim ≈ 154 calls ≈ **53.8**. Each component is live-legal.

## RESULT — `keith_ctx16` (window 16384; 11:16→13:30 UTC, ≈$9) — **DEAD, and it refutes my threshold theory**

**MECHANISM GATE: calls PASS, actions/call FAIL — and actions/call was the stated real test.**

| | base 32768 | ctx16 | ctx12 |
|---|---|---|---|
| median prompt | 20,061 | 9,582 | 6,674 |
| calls/game | 55.8 | **81.8 (1.47×)** | 108.4 |
| **actions/call** | **2.76** | **1.35 (FAIL, gate 2.0)** | 0.27 |
| levels | 39.33 | **21** | 12 |

**PRIMARY: 21 levels vs 39.33 = −7.83 sd → DEAD.** Projection was ~46; it came in at 21.

**THE THRESHOLD THEORY IS REFUTED.** I predicted 16,384 would be a *different regime* because its
11,445-token history budget holds the longest observed turn (10,439) while 12,288's 7,349 does not.
It holds that turn **and still lost half its actions/call**:

| window | history budget | holds a max turn? | actions/call | levels |
|---|---|---|---|---|
| 32,768 | 27,829 | yes | 2.76 | 39.33 |
| 16,384 | 11,445 | **yes** | **1.35** | 21 |
| 12,288 | 7,349 | no | 0.27 | 12 |

Degradation tracks context **smoothly**. There is no safe window to sit at, and the mechanism is not
the eviction edge I hypothesised — it is that **context capacity is strongly load-bearing across its
whole range**.

**ROUTE A (shrink sequences) IS NOW PROPERLY CLOSED** — two points, a monotone gradient, and an
understood mechanism, rather than the single point I over-generalised from yesterday. Ahmed's
challenge is what produced this; the $9 bought a real closure instead of an inference.

**WHAT REMAINS GENUINELY UNTESTED (do not close these by inference):**
1. **KV 6–6.5 GiB** — inside the recorded Kaggle headroom ("≤1–2 GiB above the public 5"); only 8 and
   10 GiB OOMed. Worth ≈+20–30 % calls ≈ **+4 % levels** at elasticity 0.21. Real, but far short alone.
2. **Service time / MTP** — `calls/game = 32,400 × slots / (110 × service)`. Every arm so far attacked
   *slots*; nothing has attacked the **denominator**. Service is 17.8 s, decode-dominated, and the
   regime pins `num_speculative_tokens=3`. **Still completely untouched.**
3. **Drop reasoning from retained history** — shortens each turn without dropping turns, which is a
   different intervention from shrinking the window (that drops turns). Strong external evidence
   *against* it: OpenAI's retained-reasoning result was 13.3 → 38.3. Untested here nonetheless.

**A SECOND, SEPARATE FINDING worth keeping.** Six engaged-and-flat arms said context *content* does
not move levels. These two arms say context *capacity* moves them enormously — −7.83 sd at half the
window, −11.68 sd at a third. Those are different claims about the same resource, and the pair is now
measured rather than assumed.

## SIZING — the two remaining "buy calls" routes, closed by arithmetic on MEASURED data (2026-09-11, $0)

Rather than spend two more waves, both remaining routes were sized from telemetry already on disk.

**ROUTE C — service time / MTP. Measured, not assumed.** Pulled `vllm:spec_decode_*` from a wave's
own metrics: at MTP-3 the per-position acceptance is **76.0 % → 58.3 % → 45.4 %** (decay ratio 0.78),
giving **2.80 tokens per forward pass** against a theoretical ceiling of 4.00. Extrapolating that
measured decay, and charging a conservative 3 % extra cost per speculation position:

| MTP | tokens/forward | decode speedup | service | calls/game | levels |
|---|---|---|---|---|---|
| 3 (today) | 2.80 | 1.00× | 17.8 s | 50 | 38.4 |
| 5 | 3.43 | 1.16× | 15.6 s | 57 | 39.5 |
| 8 | 3.94 | 1.22× | 14.8 s | 60 | 39.9 |

**The ceiling settles it independently of my decay assumption:** to reach 48 levels on service alone,
service would have to fall to **7.7 s**, i.e. decode from 16.2 s to 6.1 s — a **2.6× decode speedup**,
which speculation cannot deliver. Even *perfect* acceptance at MTP-8 (physically impossible) reaches
only 50.8 levels.

**ROUTE B — KV headroom.** 6.0–6.5 GiB is inside the recorded Kaggle law and untried, worth
**+20–30 % calls ≈ 40–41 levels**. Real, and short.

**EVERY REMAINING LIVE-LEGAL COMBINATION:**

| configuration | calls/game | levels |
|---|---|---|
| base | 50 | 38.4 |
| KV 6.5 | 65 | 40.6 |
| MTP-5 | 57 | 39.5 |
| KV 6.5 + MTP-5 | 74 | 42.0 |
| **KV 6.5 + MTP-8 (most optimistic reading of both)** | 78 | **42.6** |

**The best live-legal case is 42.6 levels against a 48 bar — short by 5.4 — and it already stacks the
most favourable reading of both knobs**: KV at 6.5 GiB when 8 GiB OOMed on the Kaggle card, and MTP-8
when measured acceptance is already decaying 76 → 58 → 45 % per position.

**VERDICT: the whole "buy more calls per game" family is closed, on measured quantities rather than
inference.** The binding term is elasticity 0.21 — a 56 % call increase (the best available) buys
only 12 % more levels. The 3× budget result (56 levels, ft09 won outright) stays real and stays
unreachable, because there is no live-legal route to 3× the calls.

**Still genuinely open, and NOT closed by this:** dropping reasoning from retained history (shortens
each turn without dropping turns — a different intervention from shrinking the window, and the only
one of these that changes *composition* rather than *capacity*). External evidence runs against it
(OpenAI's retained-reasoning result was 13.3 → 38.3), which is exactly why it is worth one wave
rather than an assumption.

---

## PRE-REGISTERED — `keith_nr16` (same turns, half the tokens), written 2026-09-11 BEFORE data

**The diagnosis this is built on.** `keith_ctx16` bought calls (55.8 → 81.8) and died (−7.83 sd). The
measured cause was **turn depth**: a retained turn costs ~2,217 tokens and **~1,300 of those are the
assistant's reasoning**, so a 16,384 window keeps only **~5.2** turns against the base's **~12.6**.

**The intervention.** Strip `reasoning` from messages as they enter *retained* history. A turn then
costs ~917 tokens, and a 16,384 window keeps **~12.5** turns — the **same history depth as the 32,768
base at half the sequence length**. Same turns, twice the slots, twice the calls. The **current**
turn keeps its reasoning; only past turns lose the text, and their content, tool calls and tool
results all survive.

**Arm = `keith_ctx16`'s env (window 16384) + `graft_noreason`. Identical window to ctx16, so the pair
is a clean controlled comparison of exactly one thing: whether retained reasoning is worth its tokens.**

**COUNTER-EVIDENCE, recorded before the run because it is strong.** OpenAI attribute 13.3 % → 38.3 %
on the public set to **retained reasoning** plus compaction. Our own A1 probe proved the stock really
does carry reasoning on every retained turn (1,268 request pairs, slope 1.0), so this is a genuine
subtraction, not a no-op. If that result transfers, this arm removes the most valuable thing in the
context and should read badly. **Nobody has measured what retained reasoning is worth when its cost
is paid in calls.** That is the question.

**MECHANISM GATE (first):** calls/game **≥ 80** (ctx16 got 81.8) AND **actions/call ≥ 2.0**
(base 2.76; ctx16 collapsed to 1.35). Actions/call is again the real test — if turn depth was the
cause, restoring turns should restore it.

**PRIMARY — levels vs pooled base 39.33 (sd 2.34):** ≥48 step / 45–47 redraw / ≤44 dead.
Projection ≈103 calls/game ≈ **46 levels** — below the bar alone, as expected; KV 6.5 GiB would add
~+4 on top and is the queued follow-up **only if this arm's actions/call holds**.

**THE THREE-WAY READ, which is what makes this worth $9:**
* **actions/call recovers to ~2.7 and levels ≥ 45** → turn depth was the cause, retained reasoning is
  NOT worth its tokens for this model, and we have a live-legal lever for the first time.
* **actions/call recovers but levels stay low** → turns matter and reasoning matters too; the trade is
  a wash and the family closes.
* **actions/call stays ~1.35** → turn depth was NOT the cause of ctx16's collapse, my diagnosis is
  wrong, and something else about a small window breaks the agent.

## RESULT — `keith_nr16` (ctx16 + drop reasoning from retained history; ≈$9) — **DEAD, −8.26 sd, and my diagnosis was wrong**

**Both mechanism gates FAILED, and it is WORSE than the arm it was built to fix.**

Controlled pair — identical 16,384 window, only the graft differs:

| | median prompt | completion/call | e2e | calls/game | actions/call | levels |
|---|---|---|---|---|---|---|
| base (32,768) | 21,738 | 1,297 | 26.6 s | 56.8 | 2.74 | 39.33 |
| ctx16 | 9,582 | 1,118 | 90.1 s | 81.8 | 1.35 | 21 |
| **nr16** | **9,435** | **1,632** | **132.7 s** | **59.2** | **0.77** | **20** |

**THE STRUCTURAL MISTAKE, and it is the useful part.** I claimed this would give "the same turns at
half the sequence length". It did not, and it could not: **the trimmer always fills the window
budget.** nr16's median prompt (9,435) is essentially ctx16's (9,582). Dropping reasoning bought
**more turns**, not **shorter sequences** — exactly the failure mode I had already written down for
the graft *alone*, and then assumed the smaller window would prevent. It does not.

**Sequence length is set by the context window, full stop.** So dropping reasoning can never buy
slots, and the only lever on sequence length remains the window itself — which is measured
catastrophic. **This closes the composition route as well as the capacity route.**

**AND RETAINED REASONING IS LOAD-BEARING.** More reasoning-free turns did *worse* than fewer turns
with reasoning: actions/call **0.77 vs 1.35**. Stripped of its own past thinking the model **generated
46 % more tokens per call** (1,632 vs 1,118) — it re-derives what it can no longer see — which pushed
e2e from 90 s to **133 s** and *cut* calls/game from 81.8 to 59.2. The intervention paid for turns in
calls and got neither.

This is the **third** pre-registered outcome of the three I wrote down: actions/call did not recover,
so turn depth was **not** the cause of ctx16's collapse. It independently corroborates OpenAI's
retained-reasoning result (13.3 → 38.3 %) from the subtraction side, on an open model, at our budget.

**STATE OF THE "BUY CALLS" FAMILY — now closed on all four routes:**
| route | verdict |
|---|---|
| shrink the window | catastrophic, two points, smooth gradient |
| drop reasoning (composition) | **this arm** — cannot shorten sequences, and reasoning is load-bearing |
| raise KV 6–6.5 GiB | untried, sized at ≈40–41 levels |
| service time / MTP | sized from measured acceptance at ≈39.5–39.9 levels |

Best possible remaining combination ≈**42.6 levels against a 48 bar.** The 3× budget result
(56 levels, ft09 won outright) stays real and stays unreachable.

---

## CORRECTIONS 2026-09-12 — adversarial re-verification of every result above from 09-09 on (see `docs/research-2026-09-12/R-validation-audit-0912.md`)

Method: one refuting verifier per claim recomputing from the raw wave directories, plus instrument and
statistics skeptics; top-level spot checks reproduced the three highest-impact items. **Every level count
and primary number above reproduces. No verdict flips direction.** The following framing does not survive
and is retracted here rather than edited in place, so the record of what was believed stays intact.

**Statistics (touches every "sd" headline since 09-08).** The "pooled six-draw base 39.33 (sd 2.34)" is
`R-loss-ledger-3.md:3-5`: 36 (M1) / 40 (M4 = keith_retry, graft fired 5x) / 42 (Kaggle commit) / 41 (Y1) /
40 (Y2) / 37 (Kaggle commit). Half the pool is yield900; two members have no per-game data on disk. The
honest sd of a 25-game wave total, from per-game variance across the flat draws on disk, is **~3.2-3.5**
(n=6 sd 2.34 has 95 % CI 1.46-5.74; the "~6" in the NOTES is a mis-derivation). At sd 3.5 one wave resolves
±7.4 levels; two waves ±5.6; the 48 bar is +2.3 sd; the 45-47 band is inside noise; across ~15 arms
P(one spurious step candidate) ≈ 21 %. Robust verdicts: budget 56 (+4.4 sd, paired t +3.5), ctx12 12
(t −7.1), ctx16 21, nr16 20. Fragile: cadence 34 (−1.4 sd) and fx 35 (−1.2) are "not a step", NOT "worse
than base"; probe/carry 41 are flat and indistinguishable from +10-15 %. "Elasticity decays 0.40 → 0.21"
is a base-choice artifact: KV10 vs its own-session base 36 is 0.41 ± 0.32, vs the pool 0.20 ± 0.21; budget
within-wave 0.17 ± 0.07. No decay is demonstrated.

**Instrument.** (1) budget 3x: the container hit this rig's own 6 h lifetime cap at +297 min (it was 66 min
old at wave start from the smoke); 215 s outage, 24× HTTP 500; the post-restart tail ran at median e2e
104 s (below the 120 s cadence gate) → ~+156 surplus calls; `metrics_before.prom` is the smoke container so
the VLLM row (661 requests, e2e 181.9 s) is a two-container subtraction and unreadable. Bias ≤ +0.5 level.
(2) ctx12: the container DIED at +75-77 min (25 simultaneous HTTP 500, 398 s gap, new vLLM process at
+79.7 min; not the reaper, not idle scale-down; cause not in local files); ~464 s/game (5.9 %) lost; every
VLLM number and "preemptions 0" cover only the post-restart 40 % of calls. 11 of 12 clears were in by +43 min.
(3) ctx16 / nr16: 1,908 / 1,652 KV preemptions (0.9-1.1 per request vs 0.05-0.24 on every 32k wave) — a
scheduler thrash specific to the 16k window under kv5/c8; ≈17 calls/game and ≈25 % decode inflation; not
reported in either write-up. (4) The per-run VOID rule (request errors > 0, preemptions > 0) fires on 25/25
runs of every Modal wave because "errors" are the runner's own end-of-clock ReadTimeouts; it is inoperable
as written and has been waived by discretion (explicitly for cadence/fx/probe, silently for budget/ctx12/
ctx16/nr16). Replacement: VOID on any HTTP 5xx, on a `process_start_time_seconds` change between
before/after, on preemptions/request outside 0.02-0.35, or on any game's shim span < 0.99 × clock.

**Per-result retractions.**
* budget 3x: "first outright win in this campaign's rig history" is FALSE — ft09 won 6/6 in the KV10 wave
  (line 369 above; 108 actions, 7,686 s, inside a 1x clock). The table mixes the pooled base (levels) with
  the single lowest 09-02 draw (calls, actions, score, distributions): vs the Modal-pool means the ratios
  are 3.18× calls, 4.95× actions, 1.75× score; 5+-level games 0/1/1/1 → 3. **41 of 56 clears landed by
  7,920 s** (+0.71 sd; calls in the first 7,920 s = 53.0/game), so the base waves are consistent and the
  extra clock bought ~15 levels on the same games in the same boot. 56 sits in the pre-registration's own
  46-59 "partial" band; only 3 of the 12 never-passed walls fell (dc22, sb26, vc33).
* ctx12: mechanism was wrong — the stock trimmer budgets chars/3 of the JSON payload INCLUDING the base64
  image (`tool_agent.py:462-467`), so history room at 12k is ≈2,950 real tokens ≈ one turn, not 7,349; only
  2 of 2,709 completions exceed 7,349; 16 % of mid-game calls were sent with zero history and acted 0 % of
  the time. "base 20,061" is the mean prompt (median 21,421). Zero-level base is 1-7 across draws (mean
  3.25), not 2.
* ctx16: the "longest observed completion 10,439" premise was already false on disk (base 10,495, fx 10,673,
  evid 11,685, carry-kill 13,800; ctx16 itself 12,942). "Smooth gradient / no safe window" is over-claimed:
  actions/call is linear 32k→16k and collapses at 12k; three single draws cannot separate the shapes.
* nr16: the "base (32,768)" row (prompt 21,738 / 1,297 / e2e 26.6 s / 56.8 calls / 2.74 actions per call)
  is the 09-09 conc-6 CADENCE arm (34 levels) with 39.33 pasted in; the conc-28 base is e2e ≈145 s, median
  prompt 21,421, 55.8 calls, 2.76 actions/call. The graft restored 3.32 vs 2.72 retained assistant turns
  (not 12.5 vs 5.2); reasoning is ≈1/3 of a retained turn, not ≈60 %; "the trimmer always fills the
  window" holds only in estimator units (real prompts ≈58 % of the window). Because turns were never
  restored, "third pre-registered outcome: turn depth was NOT the cause" is unsupported — the arm is
  uninformative on turn depth. Confirmed: +46 % completion tokens on 25/25 games → −28 % calls.
* sizing: mixed anchors (calls from a modelled 50/game, levels from 55.8/39.33). Re-anchored, the best
  live-legal stack is 44.0 levels (shortfall 4.0), or 42.7 with the KV^0.56 scaling KV10 actually showed;
  "7.7 s / 2.6×" → 8.1-8.7 s / 2.3-2.5×; "the ceiling settles it" is false (perfect MTP-8 projects 48.8-52).
  The closure holds at elasticity 0.2 and fails at 0.4; KV10 cannot tell them apart.
* cadence: 24 of 25 games averaged 51.6 calls (tr87 ran alone in batch 5 at conc 1 with 181 calls); the
  −5.33 is concentrated in batch 1 (6 games, 4 levels); `time_remaining_seconds` appears ≈29 times in 25
  transcripts. "Throughput/cadence family closed as a step" was contradicted by budget 3x the next day.
* fx: "24/25 games saw a 7+-line block" → 16/25; the model quotes the effect table in 7.0 % of turns
  (thinking only, never in code); prompt cost +487 tokens/call vs +250 pre-registered.
* A2 (docs/research-2026-09-09): counts stand (0/358 verifier calls, and 0/359 in the control; 24/24
  length-capped inside `<think>`). "The failure is the LOOP, not the brain" does not stand: all 24 directed
  requests were fresh 2-message contexts (n_messages=2, has_tools=False), and the live directed prompt
  differed from the offline one (≈170- vs ≈600-word system, 12 vs 20 transitions, 60-cell lossy cap, 3 calls
  with one-line feedback vs 20). Pass-1 prompts (8.1-18.9k / 13.9-24.7k) were already inside the claimed
  Stage-1 window. Pass-2 GPU 5,857 s not 5,016. Cost denominator: one verified model = 88-89 % of a true
  stock game's decode (83,094 tokens/game was the carry75 arm; stock is 73.6-74.2k).
* R-what-our-agent-actually-does: ≥30 of the 2,047 calls forward-simulate with a hand-written transition
  function (tu93, m0r0, lp85, lf52, dc22) and several verify the prediction; `build_corpus.py:154` aligns
  calls to turn headers by index (wrong for the 35 % of turns with >1 call; 1,288 cleared-level calls dropped,
  4,573 continuation calls excluded); the classifier was never committed. Weak form stands.
* R-triage-family-bounded: AUC 0.94 is target leakage; on the incremental target no eval-legal feature beats
  ≈0.52; the keep-top-N table has no code; at elasticity 0.21 the in-range rows read −6 % to +9 %.
* PREREG-model-axis RESULT: 180.0 B params on disk (not 131 B); the 360 GB includes the 51.2 B-param PLE
  table that serving CPU-offloads; the served NVFP4 checkpoint keeps every attention/linear-attention tensor
  in bf16 (78 GiB non-PLE fits one 96 GB card); H100 is $3.95/h; a 2-4 GPU attention-only LoRA attempt is
  ≈$40-100. The lane is blocked on unproven tooling and the calendar, not on scale; M1/M2 were never run.

**Live.** Base draw #4 (sub 56133282, 09-10) = 2.73, in band; family n=6 mean 3.22 sd 0.78; identical-bytes
v4 sd 0.35 (n=3), yield900 sd 1.00 (n=3). The yield900 3-draw "dead" was decided by 0.01 (3.59 vs 3.6; 95 %
CI 2.68-4.50); the "118 = 118" off-Kaggle null cannot be recomputed (cache gone) and every same-regime
comparison on disk points +0.1-0.2 lv/game. Sub 55543514 (08-16, 1.29) is missing from the ledger.

**Rule ADOPTED by Ahmed 2026-09-12 and applied in code** (`run_regime_wave.py` LEDGER3_REFERENCE, `wall_reads`,
`metrics_delta`; `read_cadence_arm.py` constants): base = the 8 flat live-geometry draws [36, 34, 40, 41, 40, 41,
41, 35] mean 38.5, sd 3.5; two-wave mean ≥45 = step candidate, ≤42 = dead; one wave reading 31-46 REQUIRES a
counterbalanced redraw; VOID = any HTTP 5xx, a vLLM process restart between the metric snapshots, or
preemptions/request > 0.5; end-of-clock ReadTimeouts no longer void. Ahmed's other decisions the same day:
restart-at-the-wall graft YES (kill test first, Kaggle GPU quota only); yield900 draw #4 YES (one draw);
model-axis serving gate YES (Kaggle GPU quota).

---

## RESULTS 2026-09-12 — three Kaggle commit runs (plan steps 5, 2b, 2a); read under the corrected rule

Instrument: Kaggle GPU commit runs of the byte copy on the RTX PRO 6000 (25 public games, conc 28, 7,920 s,
one session at a time; outputs `scratchpad/kernel_out/<kernel>/`). Public-commit reference for this
instrument: keith V14 commit 36 levels / 55 calls/game / e2e 142 s / 6.76 pts (`REF keith V14 commit` line);
earlier Kaggle commit draws of our copy read 42 (base) and 37 (yield900). Single draws: ±7 levels resolvable.

### `arc3-keith-kv6` (KV 5 → 6.0 GiB, nothing else; 11:19→13:39Z) — **BOOTS, ALIVE, live-legal**
Model loading 81.8 GiB + 6.0 GiB KV fits (8 and 10 GiB OOMed on this card on 09-06). **38 levels** (dist
0:4 1:12 2:4 3:2 4:3), score 7.47 vs ref 6.76; **64.9 calls/game (+18 % vs 55)**, e2e 120 s (ref 142),
queue 100 s, inference 19.9 s, actions/call 2.73, preemptions 0.02/request, 1,391 gen tok/req. Consistent with
the audit's sizing (calls ∝ KV^0.56: 1.2× KV → +11-16 %) and with elasticity ~0.2 (+18 % calls → ~+4 %
levels, inside one-draw noise). This is submission #1 of the plan: kernel `arc3-keith-kv6` v1 COMPLETE,
byte copy + two changed lines. Next: 6.5 GiB commit only if Ahmed wants the boundary; otherwise fly 6.0.

### `arc3-keith-apc` (prefix caching ON, MTP OFF, KV 5 GiB; 14:49→17:09Z) — **closed by the pre-registered gate**
Boots (no `--mamba-cache-mode` needed; `enable_prefix_caching=True` in the engine args). **Prefix hit rate
15.7 %** (4.81 M / 30.7 M tokens) — below the 20 % kill line. **59.9 calls/game (+9 %)** — below the +10 %
net-calls gate, and that gain comes with MTP off (inference 37.8 s vs 19.9 s; queue 92 s). Preemptions
**0.54/request** (VOID flag under the new rule: APC pins blocks in the 5 GiB KV). 36 levels (= ref).
Model loading **74.3 GiB** without the MTP head (7.5 GiB less than with MTP). VERDICT: caching-without-MTP
at KV 5 GiB is not a lever. UNTESTED and cheap: the 7.5 GiB freed by dropping MTP means **APC + KV 10-12 GiB**
should fit on this card, which is exactly the regime where a prefix cache stops thrashing (hit rate ≫ 16 %,
preemptions ≪ 0.5). One commit run (~2.3 h) would settle it; not run without a go.

### `arc3-oracle-probe-fn` (oracle probe rebuilt on the flown Flash-Next base; 17:59→18:49Z) — **instrument FIXED, brain read NEGATIVE**
Both preflights passed (short: `[1,2,3]`; game-length: 7 actions, finish=stop). **All 15 clones valid** —
every clone executed the full 100-move budget (328 acting turns, 3.7-5.5 actions/turn), so the 08-29 "0 moves"
artefact (`preserve_thinking` instead of `enable_thinking`) is gone. Results: **guided 0/5, oracle 0/5,
control 0/5**; blocks delivered 2-4 of 5 (best: guided clone 4, 1 left, 4 delivered); 19 % of replies still
finish=length at 3,000 tokens but parsed. Guided clone 0 shows the limit-cycle pathology verbatim (`[3,5,4,4,5]`
repeated for 8 turns). Under the ORIGINAL pre-registration (`build_oracle_probe.py`: oracle < 50 % AND
guided < 50 % ⇒ "it cannot execute a correct handed-to-it model, even step-by-step"), this is the negative
branch: on wa30 L3 the served brain cannot turn either the full mechanics or per-turn instructions into a
100-move solution (engine-verified 82-move solution exists). Caveat stated once: the 100-move budget is the
level's own and leaves 18 moves of slack; one clone reached 4/5, so "cannot execute" and "cannot execute
within budget" are not separated. Plan step 2a's kill line (ORACLE < 50 % ⇒ every "harness supplies
dynamics, model plans" design is dead on this brain) **fires**: no effect-table planner is built. Optional
$0-slot / 50-min follow-up: the same kernel with the level budget lifted (ORACLE_PROBE_MAX_MOVES=250,
engine budget bypassed) to separate the two readings — only if Ahmed wants the distinction.
