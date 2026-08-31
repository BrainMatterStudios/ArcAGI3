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
