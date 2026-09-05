# HANDOFF 2026-08-29 — the four packs are built; two A/B smokes are on the GPU

Supersedes `HANDOFF-2026-08-29-the-model-axis.md` (its strategy §1a/§3/§4 were retracted by
`docs/STRATEGY-2026-08-29-independent-review-path-to-6.md`; read that first).

## 0. State at 11:40 UTC

| thing | state |
|---|---|
| public LB | cstl 5.99 · us 1.74 (rank 291) · 12-draw Qwen3.8 series mean 1.41 |
| submission 55861885 (08-29 slot) | byte-identical v8 harvest, PENDING — today's slot is used |
| branch `winning/duck-patched` | `033d081`, **not pushed** (push gate) |
| kernel `arc3-tp-smoke` v1 | RUNNING since ~10:03 UTC — Pack 1 A/B (stock vs tp), ~4.6 h |
| kernel `arc3-tp2-smoke` v1 | RUNNING since ~10:50 UTC — Pack 2 A/B (tp vs tp+control), ~4.6 h |
| kernel `arc3-oracle-probe` v2 | RUNNING (previous session's diagnostic; not on the critical path) |
| flight arms | `submission/_duck38_flight/{tp1,tp2,tp24}/` built + validated, **not pushed, not submitted** |

## 1. What was built (all tests green, offline dry run 11/11)

All under `submission/_throughput_v1/`. Every seam is a monkeypatch on the anim bundle that plays at
eval (`submission/_inspect_replay/assets_build/ARC3-Inference`, byte-identical to the
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim` dataset; the framework half is archived at
`scratchpad/bundles/anim_20260807/`). Flags are read at call time; `*_ENABLE=0` is a pass-through.

| pack | module | flags (defaults) | what it changes |
|---|---|---|---|
| 1 throughput | `graft_throughput.py` | `TP_TRIM_LOW_WATER=0.5 TP_CONTEXT_WINDOW=24576 TP_YIELD_SECONDS=900 TP_TOOL_STEPS=8 TP_KEEP_NOTES_ON_GAME_OVER=1 TP_BATCH_CAP=10` | hysteresis trim (prefix survives turns), smaller window, no sub-call slices, notes survive GAME_OVER, ≤10 actions per tool call, 9 h time guard |
| 2 memory & control | `graft_control.py` | `TP2_SUMMARY TP2_PROBE(3 clicks) TP2_STALL(T1=10, T2=30 RESET, 2/level) TP2_STREAK(3) TP2_DIFF` | harness-owned summaries at cuts and level-ups (non-thinking call, rate-limited), level-start probe table in the prompt, HUD-aware stagnation directive + harness RESET, no-effect streak halt, diff summary on every action |
| 4 explorer fallback | `graft_explore.py` + `frontier_explorer.py` | `TP4_STALL_T3=30 TP4_BUDGET=800 TP4_ENDGAME_S=300` | after one harness RESET and 30 more stale actions, a just-explore frontier walk takes the level (≤800 engine actions, seconds of wall); endgame policy in the last 5 min; HARNESS NOTE back to the model |
| 3 serving | `docs/research-2026-08-29/R5-pack3-vllm024-boot-recipe.md` | — | vLLM 0.24 boots on Kaggle via the `jcole75/arc3-qwen36-runtime-wheels` house (CUDA-13.3 nvcc inside site-packages, `VLLM_USE_FLASHINFER_SAMPLER=0`); serving-lab4 builder in progress |

Tests: `.venv/bin/python submission/_throughput_v1/test_graft_throughput.py` (27), `test_graft_control.py`
(17), `test_graft_explore.py` (5), `test_frontier_explorer.py` (5). End-to-end: `dry_run.py` (mock brain +
real harness + real engine, 3 games, ~4 min) — it caught two real integration bugs today; run it after any
graft change.

Explorer alone (`bench_explorer.py`): L1 on 5/25 games in 200 actions, 10/25 in 800, 15/25 in 3,000;
on the LLM's 10 zero-level games it adds 2 (800) / 4 (3,000). Its value is unlocking depth.

## 2. How to read the smokes (pre-registered)

Fetch: `python3 submission/_tp_smoke/fetch_results.py tp_smoke_results.json stdout.log`
(set `TP_KERNEL_SLUG=arc3-tp2-smoke` for the second). The report cell prints `TP SMOKE READ: <verdict>`.

- **tp (Pack 1)**: PASS = tp actions/game ≥ 2.0× stock AND levels/game ≥ stock − 0.15 AND prefix hit ≥ 0.5;
  INCONCLUSIVE = actions ≥ 1.5× with levels within noise; FAIL otherwise.
- **tp2 (Pack 2)**: PASS = zero-level games ≤ tp − 3 (of 25) AND levels ≥ tp − 0.1; FAIL = levels < tp − 0.15
  or zero-level > tp + 2; else INCONCLUSIVE.
- Levels/game on 25 games is noisy (identical-bytes spread ≈ ±0.3 locally); actions/game and prefix-hit are
  near-deterministic. Read levels for *direction*, not magnitude.

## 2b. Pack 1 smoke READ (14:50 UTC): FAIL — and the cause is known

`arc3-tp-smoke` v1, 25 games × 7,920 s, stock then tp (yield 900 / tool steps 8 / cap 10 / 24k):

| phase | actions/game | levels/game | zero-level | score | requests | prefix hit | gen tok/s |
|---|---|---|---|---|---|---|---|
| stock | 47.4 | 1.00 | 8 | 3.86 | 1,213 | 0.29 | 331 |
| tp | 19.0 | 0.40 | 16 | 1.25 | 1,989 | 0.55 | 440 |

The mechanics worked (prefix hit ×1.9, decode +33%, +64% requests) but actions fell 60%. Two causes,
from the transcripts (`submission/_tp_smoke/results/transcripts/`):
1. **Bug (fixed, `0634261`)**: a deep hysteresis cut inside a long single turn removed the turn's only
   user message; vLLM rejected those requests with `400 "No user query found in messages"` (11 of 50
   calls on ar25, 6 of 53 on re86), each aborting the turn.
2. **The stock 60-s yield was doing useful work**: it re-grounds the model with a fresh prompt + board
   after every investigation call (41% of calls act in stock vs 18% under an 8-call in-turn loop).
   Longer yield / bounded tool steps are withdrawn.

`arc3-tp1b-smoke` (pushed ~15:00 UTC): stock vs **mech24** = hysteresis + 24k window + notes + time guard
with the STOCK yield and tool steps, batch cap 30. Same read rule. Flight arms rebuilt to this config.

## 2c. Pack 2 smoke READ (15:40 UTC): INCONCLUSIVE, leaning negative — confounded, re-run queued

`arc3-tp2-smoke` v1 ran Pack 2 on top of the *broken* Pack 1 base (trim bug + yield 900), so the pair is
fair but the absolute level is depressed:

| phase | actions | levels | zero-level | score | turns | requests |
|---|---|---|---|---|---|---|
| tp | 22.4 | 0.48 | 15 | 1.73 | 13.5 | 2,023 |
| tp2 | 22.1 | 0.36 | 16 | 0.83 | 8.1 | 1,570 |

Transcripts (ar25/ft09/re86): turns took ~55% longer (570–680 s vs 360–430 s); the extra requests were
**~18 synchronous summary calls per game, ~70 s each at concurrency 28**, and the carried notes grew from
~270 to ~1,300 chars. The stall directive, harness RESET and streak halt never fired on these games (no
cost, no effect). The batch aggregate dropped the diff (fixed, `84f510c`).

Changes (`84f510c`): summaries now run in a background thread, ≤300 tokens, ≥240 s apart, ~120 words.
`arc3-tp2b-smoke` (pushed ~15:50 UTC): mech24 vs mech24+control on the corrected Pack 1 base.

## 2d. tp1b READ (19:40 UTC): FAIL — hysteresis starves recency

| phase | actions | levels | zero-level | score | turns | prefix hit | gen tok/s |
|---|---|---|---|---|---|---|---|
| stock | 64.6 | 1.04 | 9 | 4.90 | 18.3 | 0.28 | 330 |
| mech24 (hysteresis 24k/50%, stock yield) | 20.8 | 0.60 | 12 | 2.21 | 9.7 | 0.54 | 471 |

Transcripts: yield 60 s in effect, zero 400s, no batch-cap hits. After each 50% cut the model has 1–3
turns of history (ka59: `history_messages` 3–8 on 23 of 41 turns; stock 12–26) and it re-investigates:
turns ending in yield-without-action 32/41 vs 20/47 in stock. **Recent context is load-bearing for this
model; "we are time-starved, not context-starved" (R4 §4) was wrong as a design premise.** Throughput
gains that cost recency are net negative.

Consequences: Pack 1 in the flight arms is reduced to notes-survive-GAME_OVER + time guard. Two arms queued
behind `tp2b`: **tp2c** = stock vs stock+control (the clean Pack 2 read) and **tp1c** = stock vs gentle
hysteresis (43k window, 25% cuts, post-cut context ≥ stock's steady state).

Instrument note: the two stock phases agreed on levels (1.00 / 1.04) but differed 37% on actions/game
(47 vs 65) — read levels and zero-level counts, not actions.


## 2e. tp2b READ (20:20 UTC): INCONCLUSIVE, slightly positive on the (bad) mech24 base

| phase | actions | levels | zero-level | score | turns | requests |
|---|---|---|---|---|---|---|
| mech24 | 23.4 | 0.44 | 15 | 1.30 | 9.4 | 1,867 |
| mech24 + control (async summaries) | 22.9 | 0.48 | 14 | 1.59 | 6.5 | 1,657 |

Control is no longer costing throughput (requests down, summaries off the critical path) and reads
neutral-to-positive on levels. The clean read is `arc3-tp2c-smoke` (stock vs stock+control, running,
~00:30 UTC). `arc3-tp1c-smoke` (stock vs 43k/25% hysteresis) pushed 20:25 UTC.

## 3. Decision tree for the 08-30 slot (00:01 UTC)

1. tp PASS or INCONCLUSIVE-with-levels-up → push `submission/_duck38_flight/tp1` as a commit
   (`cd submission/_duck38_flight/tp1 && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000`,
   ~1.3 h: boot + the inert 3-game 60-min smoke hook), wait COMPLETE, then submit that version at 00:01 UTC
   via the gated runner pattern in `scripts/submit_v8harvest_20260829.py` (adapt slug/svid; keep the
   one-shot window + identity re-attest + slot race guards). **Submission is Ahmed's call — stop before it.**
2. tp2 PASS as well → fly `tp2` instead of `tp1` (it contains Pack 1).
3. tp FAIL → do not fly; diagnose from `stdout.log` (per-game actions/levels/turns, prefix hit) before
   touching thresholds. The v8 harvest (1.65) remains the fallback draw.
4. Never fly an arm whose commit did not COMPLETE; an ERROR costs no slot but a silent stock run poisons the read.

## 4. Next builds (in order)

1. Pack 3: `submission/_serving_lab4/` (being written) — boot 0.24, churn-vs-stable prefix regimes at
   conc 28, MTP nst=2 with prefix caching (greedy-identical battery + acceptance from `/metrics`). Push after
   the smokes free the GPU. Adopt only under the README's rules.
2. `tp4` smoke (`build_tp_smoke.py tp4`, kernel `arc3-tp4-smoke`) once tp2 reads.
3. Pack 2 follow-ups if tp2 is INCONCLUSIVE: HUD-masked `board_changed` in the compact result; expectation
   field on batches; re-probe on level-up (`TP2_PROBE_EACH_LEVEL`).

## 5. Laws added today

- A wrapper that looks up its "stock" through an attribute on the *currently bound* method breaks as soon as
  another graft wraps the same seam. Close over the stock at install time (module hook), and run
  `dry_run.py` — unit tests with fakes did not catch it.
- The solver filters `RESET` out of the model-facing `valid_actions`; gate harness resets on the engine's
  `available_actions` ids, never on the prompt list.
- A per-cell HUD mask misses multi-digit counters; extend to whole border rows/columns, and treat
  border-only changes as "no new state".
- Summaries must be rate-limited (≥2,000 chars, ≥90 s apart per game): an un-throttled cut hook made 738
  calls in one dry run.
- Kaggle allows ≥2 concurrent GPU kernels; the weekly quota (30 h per the older memory, 60 h per the last
  handoff — unverified) is the real bottleneck. Check remaining quota in the web UI before the third push.

## 6. 08-30 pooled-stock evidence (75 stock plays across 3 kernels) — the keep/kill base

Analysis over the three stock phases (tp, tp1b, tp2c; per-game levels in `submission/_tp_smoke/results*/`):

- **Volatility dominates**: same bytes draw ar25 at 1/2/4 levels, ft09 0–4, re86 0–3, vc33 0–3.
  Single stock draw ≈ 3.9–4.9 local; **best-of-3 = 7.81 local (1.48 levels/game)**. 12/25 games always
  clear ≥1 level, 10 flaky, only 3 never (dc22, g50t, sk48).
- **The budget currency is LLM turns**: ~16 turns/game in 2.2 h (~500 s/turn at conc 28). Cleared levels
  cost 5–64 actions. corr(actions, levels) = 0.32 across the 75 plays.
- Implications, in order: (1) serving speed (turns/game) is the one throughput lever the recency finding
  does not kill → serving-lab4 (RUNNING 08-30 ~09:00 UTC) measures vLLM 0.24 + MTP + prefix-regime;
  (2) per-game volatility means flaky games are won by better *draws by default* — mechanisms that
  stabilise the good draw (carry notes across GAME_OVER, avoid known-noop repeats) matter more than new
  capabilities; (3) any future lever read needs levels, never actions.

08-30 slot: submission 55886429 (byte-identical harvest draw #3) landed 08:52 UTC. 08-29 draw scored 1.39.
tp1c v1 died on a Kaggle infra mount failure (competition wheelhouse absent at t+5s); v2 re-pushed, QUEUED.

## 7. 08-30 midday — Flash-Next is the live bet; Pack 5 built from forensics

- **R7 forensics** (docs/research-2026-08-29/R7): modal stock failure = analysis-paralysis (63% of wall
  in zero-action model calls; 47% of calls yield without acting) + notes-channel amnesia (4/16 games carry
  ZERO notes because assistant text is empty; the world model sits in hidden reasoning). Goal hallucination:
  not observed. Action volume alone: refuted (dc22/sc25 exceeded baseline actions, cleared nothing).
- **Pack 5 (`graft_emission.py`)**: wm-from-reasoning harvest + act-floor (force the tool call after 3
  analysis-only calls). `arc3-tp5-smoke` = stock vs stock+emission, QUEUED behind serving-lab4.
- **R6 model scan**: top pick Qwen3.8-Flash-Next NVFP4 (125B MoE, 6B active). `submission/_flashnext_gate/`
  built (aa741f0): single-GPU vLLM-dev boot via sonpham's public 3-part package, PLE tables (104GB) in host
  RAM, their exact argv. **Their own package lock records duck-harness GCP mean 9.63 on the 25 public games**
  (our stock: 4.1–4.9 local) — same loop shape, bigger sparse brain. License prize-compatible (Qwen
  Community 1.0). Gate: boots AND >=450 tok/s conc-28 AND qwen3_xml parse >=95%.
- GPU queue order when a session frees: **flashnext-gate first**, then tp1c retry (fresh slug arc3-tp1c2).
- tp1c died twice to the same Kaggle competition-mount flake (metadata identical to working kernels).

## 8. 08-30 evening — Pack 5 PASS, Flash-Next boots, tp5em armed for 08-31 (Ahmed's call)

- **tp5 smoke PASS** (first passing harness read): stock 0.88 lv/9 zero -> +emission 1.04 lv/6 zero,
  local 3.50 -> 4.40; forensics' named victims flipped (tn36 0->2, tr87/m0r0/wa30/bp35 0->1).
- **Flight arm** arc3-duck38-tp5em committed COMPLETE (svid 346075311, hash 070598bf == local).
  Gated runner `scripts/submit_tp5em_20260831.py` mock-tested. **Fallback** harvest #4 runner also ready.
  Neither launches without Ahmed (approval gate); default if silent = harvest.
- **Flash-Next gate v4: BOOTS on Kaggle** (their exact config, 740 s; cu13-nvcc fix). conc-28 412-480
  tok/s vs 27B 280-297; per-session +39-45%; parse 1.0. Formal gate FAIL at the 450 stable bar —
  recorded; proceeding to the decisive 25-game stock-harness smoke (builder being written:
  submission/_flashnext_smoke/). Failure ladder that got here: pidfd_getfd (seccomp) -> arch "12.0f"
  rejected -> image nvcc predates SM 12.0 -> jcole75 cu13 wheels.
- serving-lab4 CLOSED Pack 3 for the 27B: 0.24 boots (KV x4.7) but tok/s unchanged at conc 28;
  MTP acceptance 0.72-0.77 yet +2.5% at conc 28 and 19/20 greedy mismatches.

## 9. 08-30 21:00 UTC — FLASH-NEXT FIRST FLIGHT ARMED (Ahmed-approved)

Ahmed approved the Flash-Next submission for the 08-31 00:01 UTC slot. Runner
`scripts/submit_flashnext_20260831.py` launched 20:59 UTC (in-session background task), counting down.
Arm: arc3-flashnext-flight v1 (svid 346125566, hash 08f83048…) — STOCK duck harness served by
Qwen3.8-Flash-Next NVFP4; commit smoke SMOKE-OK (boot 980 s, vc33 2 levels in a 40-min box).
Reading rule pre-registered in the submission message (new distribution; >=1.45 positive; >1.74 re-banks;
<0.5 = suspect serving, pull the kernel log). tp5em (svid 346075311) and harvest #4 runners stay armed
but UNLAUNCHED — candidates for 09-01 depending on the flashnext read.

## 10. 08-31 — FLASH-NEXT FIRST FLIGHT SUBMITTED

Submission **55902917**, 00:17:17 UTC (runner recovered from an external kill mid-settle: marker released
per the no-submission rule, relaunched detached). 90-min watch clean. Score lands ~09:30 UTC.
Read against the pre-registered rule in the submission message. Next-slot candidates by outcome:
>=1.45 → iterate the Flash-Next harness (efficiency: per-level action waste is the known drag);
in-band → second flashnext draw AND/OR tp5em (svid 346075311, armed); <0.5 → pull the kernel log
(serving failure) and fly tp5em while diagnosing.

## 11. 08-31 09:22 UTC — FLASH-NEXT FIRST FLIGHT: 1.88, NEW ALL-TIME BEST

Submission 55902917 scored **1.88 public** — re-banks the best (prior 1.74; 27B 12-draw mean 1.41).
Pre-registered read: model-swap POSITIVE live. Rank 291 -> **249**/2651. LB top moved again overnight:
cstl 7.51 (!), Franzen 4.05. Single draw; the arm's mean is unknown but the smoke projected ~2.2.
09-01 runner prepared: `scripts/submit_flashnext_20260901.py` (byte-identical redraw #2, same svid
346125566) — mock-tested, NOT launched (Ahmed's call). Parallel build queue: (a) flashnext + Pack 5
emission (compose the two passing reads — needs a tp5-on-flashnext smoke first), (b) efficiency drag
(per-level action waste), (c) serving tune (max_num_seqs 22 vs conc 28; context 24k).

## 12. 08-31 14:53 UTC — tuned A/B read: TUNED LOSES; redraw recommended for 09-01

Stock-vs-tuned on Flash-Next (25 games each, one boot):
| phase | levels | zero | score | actions |
|---|---|---|---|---|
| stock | 1.20 | 5 | 4.35 | 240 |
| tuned (TP5+TP6+TP7+keep-notes+90s timeout) | 0.92 | 8 | 2.89 | 257 |

The composed pack is REJECTED (worse on every aggregate; ft09 2->0, tr87 3->0 vs partial wins
tn36/bp35 0->1, lp85 2->3). Note also stock's second independent draw (1.20/4.35) confirms the
first (1.16/3.87) — the stock arm's local mean is ~1.18 lv / ~4.1.
Lessons: (a) composing four prompt-side interventions at once was the same mistake as Pack 1 —
attribution impossible, and added per-turn prompt mass plausibly harms this model; (b) next reads
must be single-variable (TP7 deaths-protocol alone is the highest-prior candidate); (c) stock
Flash-Next remains the best flyable arm.
09-01 slot recommendation: byte-identical stock redraw (scripts/submit_flashnext_20260901.py,
mock-tested, UNLAUNCHED — Ahmed's call).

## 13. 08-31 15:20 UTC — THE SERVING-CONFIG GAP: public V31 lane (2.66 LB) adopted

Ahmed surfaced a public notebook claiming >2: `romantamrazov/arc-real-agi-solution`
(team **The AGI Boys — rank 22, LB 2.66**, verified in the full leaderboard CSV).
Forensics on its exact bytes (pulled, diffed):

- **Agent code is byte-identical to June stock** (diff vs `scratchpad/bundles/june_stock`
  = 0 lines on tool_agent/prompts/solver). The prompt/tool policy is NOT the field's edge.
- Its bundle is **Tufa's own official share** (`keithtyser/taaf-duck-qwen38-serving-v1`,
  frozen from `jeroencottaar/taaf-kaggle-source-share`, 08-17, ARC3-Inference @ aa69123).
- Model = the SAME checkpoint we fly (`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`,
  official Qwen/Qwen3.8-27B-FP8).
- **The whole delta vs our duck38-v12 is serving flags.** Our v12 patch swapped only the
  model identity into the OLD June serving script: `VLLM_MAX_MODEL_LEN = 65536`, bf16 KV,
  no MTP, no async scheduling, transformers from the wheelhouse. The 2.66 stack serves with:
  `--kv-cache-dtype fp8`, `--max-model-len 262144`,
  `--speculative-config {"method":"mtp","num_speculative_tokens":3}` (the repack's NATIVE
  mtp.* tensors, validated at boot), `--async-scheduling`, `--no-enable-chunked-prefill`
  (V31-only, unproven delta), transformers 5.8.0 + hf-hub 1.5.0 overlay, and a 15s-period
  server health watchdog with staged restart. Their own comment: "Fallback = exact
  MTP3+async serving that produced 2.66 LB."
- ANALYZER_CONTEXT_WINDOW is 32768 in BOTH — the agent-side trim is unchanged; the win is
  pure serving throughput/latency (fp8 KV halves KV memory at concurrency 28; MTP-3 +
  async-scheduling raise decode throughput; we are prefill/throughput-bound per R8).
- Conclusion (hard evidence, closes a 2-week miss): **our 1.4-vs-2.66 gap on the 27B was
  serving config, not harness policy.** The 08-17 audit's "field 2.76 via newer public duck
  bundles" was this lane; we never adopted it.

**Staged (blocked on Ahmed's push — auto-mode classifier denies `kaggle kernels push`):**
- `submission/_v31_copy/` — byte-copy kernel (`ahmedmobasher86/arc3-v31-copy`), same
  sources/docker pin/machine shape. A GPU commit = boot proof + offline 25-game read of the
  2.66 stack, no slot cost.
- `scripts/submit_v31copy_20260901.py` — gated runner for the 09-01 00:01Z slot; refuses to
  run until EXPECTED_HASH / EXPECTED_SCRIPT_VERSION_ID are attested from the commit.
  Reading rule pre-registered in MESSAGE: >=2.2 adoption confirmed / 1.5-2.2 partial /
  <1.3 boot failure.
- TP8 durable-timeout graft (`submission/_throughput_v1/graft_durable.py` + tests, all 7
  suites + dry_run green) — queued behind the adoption; next single-variable A/B belongs ON
  the V31 stack, not on Flash-Next, if tonight confirms.

Model re-scan (scratchpad/modelscan2/REPORT.md): Flash-Next NVFP4 stays best-fit for the
box; one in-family upgrade (primitive-ai mixed NVFP4-FP8 v2, +13% throughput, quality tied)
worth an A/B later; FP8 Flash-Next closed by arithmetic; 49k context fits (24 KiB/token).
NOTE the lane implication: the 27B+MTP3 stack at 2.66 evidence-class now outranks
Flash-Next's 1.88 anchor as the base to build on.

## 14. 08-31 20:00 UTC — v31-copy commit read + servebench decomposition (both COMPLETE)

**arc3-v31-copy commit (svid 346312727, 2h12m, RTX Pro 6000):** offline 25-game read
**local mean 4.50, median 1.82, 28 levels = 1.12 lv/game, 7 zero-level, 2,733 actions.**
Above 27B-stock (0.84-1.04 lv) AND above Flash-Next stock local (~4.1 mean). Server log
confirms the scored args: kv_cache_dtype=fp8 + mtp num_spec_tokens=3 + async_scheduling +
chunked prefill ON — i.e. the V22-fallback path, the exact "produced 2.66 LB" config.
Local->LB ratios (1.7-2.06) project **2.2-2.65 LB**. Runner
`scripts/submit_v31copy_20260901.py` fully attested (hash 73f1dbbc, svid 346312727),
--mock green end-to-end incl. submit_gated honesty gate. ARMED-READY; needs Ahmed's go.

**arc3-servebench27 (5-config sweep, no games):**
- HARD: `--no-enable-chunked-prefill` CANNOT boot this model ("Chunked prefill is required
  for mamba cache mode 'align'" — the 27B is hybrid/mamba-style too). The public "V31"
  primary always failed; every 2.66-class run was the V22 args. Flag closed.
- HARD (vLLM's own log): fp8 KV doubles KV capacity 199,136 -> 398,272 tokens (48.8 GiB).
  bf16 KV at concurrency 28 with 30-50k prompts is massively over-subscribed -> queueing.
- MTP3 vs fp8KV-only: +13% turns/min, long-prompt latency 326s -> 221s (-32%), 0/8 greedy
  divergence. Async on top: +2% (noise, n~42).
- INSTRUMENT CAVEAT (recorded so nobody trusts the wrong number): the A-baseline
  "8.71 turns/min" is INVALID as a comparison — payload chars/token was miscalibrated
  (~1.7 not 3.3), so l20k/xl28k buckets were really ~40k/~56k tokens and A (65536 cap)
  rejected 56 of them with HTTP 400, completing only cheap requests. A-vs-rest turns/min
  compares different request mixes. B/C/D (0 errors each) are mutually valid.
- Flash-Next transfer implication unchanged: fp8-KV relief is 27B-specific (Flash-Next KV
  is tiny); candidate transfers are max-num-seqs 22->28, async, MTP (native head), each
  needing its own load-gen bench on the sonpham dev build.

## 15. 08-31 21:00 UTC — FRESH EXPLORATION CONSOLIDATED (R9+R10+R11): the path-to-7 plan

Three independent agents (trace forensics, field intel, schema-traces mining) converged.
Full reports: docs/research-2026-08-31/R9-path-to-7-gap-analysis.md, R10-field-intel-7plus.md,
R11-schema-traces-mining.md.

ARITHMETIC (R9, formula reproduced against all 25 games): LB 7.0 requires local ~12-14
(local->LB ~0.5) = efficiency-at-cap AND ~2.1 levels/game. Action volume already suffices
(2,733 emitted vs 2,542 for a 2-level baseline profile) — actions land in wrong buckets.

THE CONVERGED METHOD (R10: entire >=78% public tier, repos public since July; R11: Zanette-lab
traces, verified 95.35%/98.98% mean RHAE): executable world model + validate-against-log
before acting + search the MODEL (never the live game) + abort batch on first mispredicted
frame + persistent notes/suggestion memory spine. BUT (R11): frontier runs cost 5.9h/game +
4.4M output tokens on 1M context — wholesale copy impossible in a 2.2h/game 32k-context box;
and full backtest-green world models will break at 27B (Opus needed ~190 edits/game).
Adopt mechanisms, scoped down. cstl (7.51) = two anonymous SWEs, no public trail; Milestone 2
forces prize-track open-sourcing by SEPT 30 — build absorption capacity.

RANKED BUILD PLAN (single-variable A/Bs on the proven 27B V22 stack; grade by MECHANISM
METRICS — acting-turn share, validated-plan share, actions-per-level — NOT local mean,
which does not transfer across harnesses per three verified self-reports):
1. TP9 TURN-PIPELINE REPAIR (R9 #1, our-stack-specific, invisible to external lanes, gates
   everything): 48% of turns idle (60s yield discards whole turns; 13 vLLM read-timeouts;
   r11l 6,000s deterministic livelock). Persist/resume truncated turns, livelock detector
   (repeated-output hash -> perturbation), client-timeout retry. ~2x acting turns.
2. TP10 MEMORY SPINE (R11 minimal A/B + R10 D3 + review's 43% zero-level memory gap):
   notes.md persisted and re-injected every turn (4KB cap), `suggestion` note-to-self echoed
   back verbatim, honest last-turn accounting ("committed N — executed M; MISPREDICTED").
   Pure plumbing, no new 27B skill, fits 32k.
3. TP11 FIRST-ATTEMPT DISCIPLINE (R9 #2): probe budget vs baseline estimate; mandatory
   distilled replay plan after GAME_OVER; stop grinding buckets past ~2x baseline
   (sp80 kept 0.07 of 4.76 pts DESPITE clearing).
4. TP12 SCOPED MODEL-AND-SEARCH (R9 #3 / R10 D1-D2, scoped per R11): abort-on-surprise
   batches with heuristic expectations; solve-in-sandbox nudge for exactly-solvable games
   (sc25 = Lights-Out, hand-clicked 114 actions). NOT full backtest-green models at 27B.
5. Global budget scheduler (park stuck games; freed decode raises all streams).
6. FINE-TUNE (Ahmed's question, R11 verdict): method-lite LoRA viable — CoT stripped but
   2.37M chars of genuine Claude narration + commit reasons; ~5,400 turn examples. Teaches
   protocol/format, not frontier coding. Sequence AFTER TP10 lands; behind the serving gate
   (no fine-tune has ever been served+scored); dataset UNLICENSED (email authors); preserve
   mtp.* tensors through merge+requant.

IN FLIGHT: v31-copy runner armed (pid 57314, fires 09-01 00:01Z, svid 346312727, projection
2.2-2.65 LB). Ops note: disk was at 100% — agent purged uv/pip caches for ~12GB.

## 16. 09-01 10:06 UTC — v31-copy adoption scored: 1.71 (PARTIAL TRANSFER band)

Submission 55927189 (byte-copy of the public V31/V22 stack, svid 346312727) scored
**1.71** — pre-registered band 1.5-2.2 = partial transfer. Not adoption-confirmed
(<2.2), below our 1.88 all-time best (Flash-Next), well below the copy source's 2.66.

Reading (skeptical, in order of likelihood):
1. **The 2.66 was a MAX-statistic, not a mean.** The AGI Boys' 2.66 is their best of 48
   submissions; their notebook title claims only ">2". With CV 0.17-0.20 a config whose
   single-draw mean is ~1.9-2.1 produces a 2.66 max over dozens of draws. Our 1.71 single
   draw is fully consistent with that same distribution's mean. The adoption likely
   transferred CORRECTLY and the 2.66 expectation was my mis-read of a max as a mean —
   the reading rule should have been written against their DISTRIBUTION, not their best.
2. Draw variance pure and simple: 1.71 vs an expected ~2.2 is within ~2 sd.
3. Serving divergence in the scored rerun cannot be ruled in or out — the platform law
   holds (kernels output returns the COMMIT run; no channel out of a scored rerun), so
   there is no rerun log to diff. The commit run's server args were verified correct.
Local->LB transfer this flight: 4.50 -> 1.71 = 0.38 (historical 0.49-0.59) — another
data point that local means over-predict, consistent with R10's public-25 findings.

Implications:
- The 27B V22-serving stack draws ~1.7-2.1/draw — roughly AT our Flash-Next level, not
  above it. Serving adoption bought variance-band parity, not a step. The step must come
  from the HARNESS lane (TP9/TP10 A/Bs, staged and awaiting push) — consistent with R9/R10:
  the 4-7.5 tier is harness work on the same models.
- Leaderboard 10:06Z: cstl 7.51, Lord Han Solo 4.99, Tufa 4.71, THK 4.45, sonpham 4.42,
  Franzen 4.05; new 3.4-3.9 entrants overnight. Top-17 bar keeps rising.
- 09-02 slot options: (a) push TP9/TP10 A/Bs today on GPU (no slot cost), submit whichever
  arm wins its mechanism read on the V22 stack; (b) if neither is ready/clean, redraw the
  v31-copy (grow n on the new base). Decision after the A/B reads; Ahmed's call on pushes.

## 17. 09-02 09:23 UTC — TP9 flight scored 1.72: local lever gains are NOT reaching the hidden set

**Live draws, same 27B V22 stack:** stock 1.71 (sub 55927189) -> TP9 1.72 (sub 55950252).
Flat, despite TP9's same-boot A/B win. Pre-registered band 1.7-2.2 = inconclusive single
draw. But the PAIR is informative beyond the pre-registration:

**The A/B evidence (local, public 25, same boot each):**
| kernel | phase | score | levels | zero | actions |
|---|---|---|---|---|---|
| ab-tp9 | stock | 4.27 | 1.04 | 8 | 2,380 |
| ab-tp9 | TP9 | 6.42 | 1.56 (+50%) | 3 | 2,924 |
| ab-tp10 | stock | 7.14 | 1.36 | 8 | 3,635 |
| ab-tp10 | TP10 | 7.24 | 1.48 (+9%) | 5 | 3,042 |
Boot variance: identical stock bytes drew 4.27 and 7.14 -> SAME-BOOT LAW (only same-boot
comparisons are readable locally). Flight commit read: 4.55 / 1.20lv / 6 zero, median 2.77.

**Reading (skeptical):** a +50% local level gain producing +0.01 live is consistent with
(a) draw variance masking a modest true live gain (sd ~0.3; a true ~2.0 mean drawing 1.72
is within 1sd), and/or (b) the R10-verified pattern that public-25 gains do not transfer
(85 of ~110 hidden games are unseen; their stall/timer profile may differ). NOT consistent
with a large live gain. The lever is mechanism-verified but live-unproven.

**Leaderboard 09:23Z:** cstl 7.51, Lord Han Solo 4.99, Tufa 4.71, sonpham 4.52, THK 4.45,
Hieu Vy 4.11, Franzen 4.05; 3.85 tier thickening. Our 1.88 (Flash-Next) remains our best.

**Recommendation for 09-03 (Ahmed's call):**
1. Today on GPU (no slot): TP9+TP10 combined vs TP9-alone same-boot A/B (both levers have
   independent positive local reads; composition now licensed).
2. Tonight's slot: EITHER redraw the TP9 flight (grows live n on the lever, per the
   pre-registration) OR fly combined if its A/B wins decisively. Default if unreachable:
   TP9 redraw (byte-identical, svid 346574220 already attested).
3. The deeper question the flat pair raises: the binding constraint live may be the
   UNSEEN-game comprehension gap, not the mechanics TP9 fixes — the schema-traces
   protocol lane (scoped model-and-search, TP12) and the Sept-30 open-sourcing absorption
   remain the step-change candidates.

## 18. 09-02 10:04 UTC — Review of §13–§17: the same-boot instrument had no order control

A fresh-session review of this whole log found that every same-boot A/B (`arc3-v22-ab-*`)
ran stock in phase 1 and the graft in phase 2 with no stock-vs-stock control; the "+50%"
TP9 read in §17 is therefore confounded with phase order (warm vLLM prefix cache/graphs),
and the flight commit's 4.55 vs stock 4.50 was a warning the reading missed. Two control
kernels were built, judged (SHIP-WITH-FIXES, fixes applied) and pushed 10:04Z:
`arc3-v22-aa` (stock, stock) and `arc3-v22-ba-tp9` (tp9, stock). Design, estimators,
reading rules and tonight's slot decision tree are pre-registered in
**docs/EXPERIMENT-2026-09-02-ab-instrument-control.md** — that file supersedes §17's
recommendation. The amended fresh-session prompt (fact 3 corrected) is in
docs/HANDOFF-2026-09-02-fresh-session-prompt.md. Ledger stale entries filled from the API.

## 19. 09-02 16:17 UTC — Step-change search done; tonight = Flash-Next exact redraw; keith V14 copy queued for the quota reset

Search deliverable: docs/SEARCH-2026-09-02-step-change-ranking.md (reports + judge in
docs/research-2026-09-02/). #1 = byte-copy of keithtyser V14 (Flash-Next serving regime on stock
code; forks drew 2.80/3.22/3.38 live; expected ≈2.7). Push refused: weekly GPU quota exhausted;
`scripts/push_keith_copy_when_free.sh` (pid in logs/push_keith_copy.log) retries every 30 min and
will start the commit at the reset (expected Sat 09-05 00:00 UTC) → attest per J-judge.md → fly 09-05
with Ahmed's go. Ahmed approved the plan 16:15Z: the 09-03 slot is the Flash-Next EXACT REDRAW
(`scripts/submit_flashnext_20260903.py`, armed 16:17Z, fires 00:01Z, svid 346125566, message carries
the pre-registered rule: ≥2.4 weakens the serving-gap hypothesis; 1.5–2.4 in-band; <1.2 pull the log).
09-04: redraw of a committed version or hold — decide after the 09-03 read.

## 20. 09-03 09:26 UTC — Flash-Next exact redraw scored 1.94 (in-band; new best by 0.06); Modal regime axis closed at ~+0.3 lv

Sub 55970756 (byte-identical to 55902917's 1.88) → **1.94**. Pre-registered band 1.5–2.4 =
in-band: our slow-serving Flash-Next profile draws ~1.9 (n=2: 1.88, 1.94), the tier's public
profile (keithtyser V14 forks) draws 2.8–3.4. Serving-gap hypothesis stands.
Overnight on the Modal RTX PRO 6000 rig (offkaggle/REGIME_WAVE_STATUS.md): base regime
reproduced (1.44 lv/game, telemetry == the public commit run); flight analyzer caps → 0.88
(the caps ARE the gap); yield 180 s → 1.36 (neutral, queue-bound); KV 10 GiB → 1.72 (+0.28 ±
0.24, no step; first full win ft09; zero-level games 2→6). Stage-0 (27B on H100): frontier
protocol lane DEAD by rule (1/3 green; thinking never terminates on hard games).
Plan: byte-copy of keith V14 commits automatically at the Sat 00:00Z quota reset → attest →
fly 09-05 (Ahmed's go). Then the first counterbalanced Kaggle pair = kv10 on that base.
Original candidates still standing: thinking-termination control; runaway-game containment.
09-04 slot: no committed version worth a slot beyond redraws — Ahmed's call (hold or redraw).

## 21. 09-04 09:26 UTC — Flash-Next redraw #3 scored 1.78 (in-band); our slow profile is pinned at ~1.87

Sub 55996715 → 1.78. The Flash-Next-under-our-profile distribution is now n=3: 1.88, 1.94, 1.78
(mean 1.87, sd 0.08 — tighter than the 27B family's 0.17-0.30). The public-profile forks sit at
2.8–3.4; the gap is the analyzer caps + serving regime, reproduced on Modal (0.88 vs 1.44
lv/game on the same server). 09-03 Modal program closed: yield neutral, KV10 +0.28 (no step),
fresh-mind retry unsupported (5 fires, 0/5 clears), Stage-0 lane dead. Next: the byte-copy of
keithtyser V14 commits automatically at Sat 09-05 00:00 UTC (scripts/push_keith_copy_when_free.sh)
→ attest per J-judge.md → fly 09-05 on Ahmed's go. Then kv10 as the first counterbalanced pair.

## 22. 09-05 23:24 UTC — keith V14 byte-copy FLEW (sub 56042273) after a mount-layout fight

The 00:13Z auto-push (v1) and a re-push (v2) both ERRORed at t=7s: on this account's GPU sessions
Kaggle sometimes mounts inputs in the SHORT layout (/kaggle/input/<slug>, no competitions/ or
datasets/<owner>/ prefixes) — the 08-15 lesson again. v3 (wait for the long-form path) proved the
mount never appears; v4 resolves the competition root in either layout and waits for a non-empty
wheels dir (install cell + one path in the run cell are the ONLY deviations from the public V14).
Both of the account's GPU sessions were held by rsna-knee-train kernels; Ahmed stopped one
(standing rule now: ONE GPU session at a time, never stop non-ARC kernels). v4 commit
(svid 347562879, hash 90efdebb…) COMPLETE 23:14Z: profile kv5, "105,202 tokens / 3.21x" line
identical to keith's, 243 tok/s, 0 restarts, 25/25 audit, LOCAL 42 levels (1.68/game) score 10.42
(keith's own commit: 1.44 / 6.76). Runner auto-attested + submitted 23:24Z with Ahmed's explicit go.
READING RULE (pre-registered in the message): baseline family n=3 mean 1.87; field draws
2.80/3.22/3.38; >=2.4 transfer confirmed (new base); 1.9–2.4 partial (diff telemetry); <1.9 not
byte-equivalent in effect (pull the commit log). Score ~08:30–10:30Z 09-06.
Next: kv10 rider as the first counterbalanced pair on this base (Modal: +0.28 lv, first full win).
