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
