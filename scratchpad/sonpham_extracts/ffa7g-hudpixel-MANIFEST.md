# ffa7g-hudpixel — dedicated per-cell HUD pixel model (prototype, offline-verified)

**Status: implementation complete, offline-verified, NOT yet live/GCP-tested.** Patch applies
`-p1` clean to a fresh `../baseline-v12/src/ARC3-Inference` copy, all 7 touched files compile,
and the two new classes pass a standalone dry-run test (see Verification below). No GCP spend
yet — a subset smoke-test is the natural next step before any full-25 A/B against `ffa7gn`.
Derives from `../ffa7g/` (full-frame + ACTION7 + goal-guidance + no-impact band ∪ HUD code-model
+ state graph), which itself derives from `../baseline-v12/` — per the harness rules, one
combined patch on a fresh copy of the frozen baseline (diffed against pristine, not chained on
top of `ffa7g`'s own patch), same convention `ffa7g` itself used.

## Motivation

Read the actual `ffa7gnh`/`ffa7gnhb` run transcripts (25 games × 2 passes, 50 total) to
answer "did the existing LLM HUD code-model (`_HudCodeModel`) ever actually get used?"
**No.** `grep -c "def advance_hud"` across all 50 transcripts returns zero. Every recorded
`no_impact_source` in both passes was `band` or `exact` — never `model` or `model+band`.
The code path was implemented, sandboxed correctly, and validated in an offline dry-run
(36/36 catch rate once combined with the band), but it was never exercised in a real
scored run, and the harness's own conclusion at the time ("the code model didn't add")
was really measuring "an already-overloaded generalist agent never remembers to also
volunteer this optional side-task" — not "code-based HUD modeling doesn't work."

Root cause, read directly from `inference/agent/tool_agent.py`'s
`_extract_scientist_note`/`_extract_hud_model_code`: `hud_model` is extracted from a
**free-form fenced code block the agent may or may not spontaneously write** inside its
normal tool-call response — it competes with puzzle-solving, planning, and the knowledge
ledger for the same turn's attention, with no dedicated call of its own and no forcing
function. `_HudCodeModel` itself is also **binary at the whole-model level**: `record()`
increments a single `fails` counter and drops the *entire* model (`stale = True`) after
just 2 consecutive mispredictions anywhere, even if 95% of its claimed cells are still
correct — discarding good partial signal rather than degrading gracefully.

**Correction to an initial hypothesis, resolved by reading `_StateGraph.set_mask` directly**
(not by re-paraphrasing the MANIFEST prose, which reads ambiguously on this point): the
state graph's canonical-state mask is **already live/non-monotonic** — `set_mask`'s own
docstring is explicit that it deliberately avoids a monotonic union of the band specifically
to prevent permanently over-masking transient over-flags, and re-canonicalizes whenever the
mask changes in either direction. There is no monotonic-union bug to fix here. The only real
change needed is what feeds into `model_bar` (the verified-cells argument already passed to
both no-impact detection and `_StateGraph.observe()`), swapping the old binary
`_HudCodeModel.active` gate for `_HudPixelModel`'s richer, per-cell rolling-trust set.

## Design

1. **`_HudPixelModel`** (replaces `_HudCodeModel`): same sandboxed-`exec` `advance_hud(frame,
   action) -> frame` interface and the same `_verify_hud_model` diffing (`bar_cells` claimed,
   `model_ok` whether they all matched) — but instead of one global `fails`/`stale` gate, each
   *cell* the model has ever claimed gets its own rolling window of pass/fail observations
   (same `window`/`threshold`/`warmup` shape as `_HousekeepingBand`, for consistency). A cell
   is "trusted" (included in `model_bar`) once it has enough observations and recent accuracy
   clears the threshold; it drops out again on its own if the model starts missing it — no
   whole-model reset needed, and a newly-registered function starts every cell's trust from
   scratch (old accuracy isn't evidence for new code).
2. **Dedicated periodic call** (new, host-triggered, not agent-volunteered): every
   `HUD_MODEL_REFRESH_ACTIONS` real actions (env-configurable), the harness fires its own
   standalone completion — same server, isolated context, no tool schema, system prompt
   narrowly scoped to "here are the last N (frame, action, next-frame) transitions, write or
   refine `advance_hud`, or say there's no evidence of a HUD element yet." This is the actual
   fix for "never once attempted": the model doesn't have to remember to volunteer it inside an
   already-crowded turn, because it's the *only* thing being asked in that call.
3. **Wiring**: `model_bar` passed to `_HousekeepingBand`-adjacent no-impact logic and to
   `_StateGraph.observe()` becomes `_HudPixelModel.trusted_cells()` instead of the old
   `model.active`-gated `bar_cells`. Everything downstream (mask union, canonical-state
   hashing, `no_impact_source` reporting) is unchanged — it already treats `model_bar` as "a
   set of extra verified cells to union in," which is exactly what the new model still
   provides, just computed per-cell instead of all-or-nothing.

## Files touched (7, same footprint as `../ffa7g/` since this patch is diffed vs pristine
baseline and carries `ffa7g`'s own changes forward)
`inference/agent/action_names.py`, `inference/agent/frame_mode.py`,
`inference/agent/prompts.py` (new `HUD_SUBAGENT_SYSTEM_PROMPT`, used only by the dedicated
call, never injected into the main agent's system prompt), `inference/agent/python_tool_sandbox.py`,
`inference/agent/runtime_state.py`, `inference/agent/tool_agent.py` (new
`ToolAgent.refresh_hud_model()`; removed the now-meaningless "stale, rewrite it" nudge in
`_compact_action_result`), `inference/framework/solver.py` (`_HudCodeModel` → `_HudPixelModel`,
`_verify_hud_model` → `_verify_hud_model_percell`, new `_hud_model_refresh_actions()` /
`_hud_model_min_transitions()` env readers, new `_HarnessGameSession._recent_hud_transitions()`,
scheduling + level-transition-reset wired into `_execute_action`).

New env vars: `HUD_MODEL_REFRESH_ACTIONS` (default 8; 0 disables the dedicated call, falling
back to agent-volunteered-only), `HUD_MODEL_MIN_TRANSITIONS` (default 3; minimum recorded
transitions before the first dedicated call).

## Verification

**Offline (done):** `harnesses/ffa7g-hudpixel/test_hud_pixel_model.py` — standalone dry-run
(no `taaf`/repo deps; the two new pure-Python classes extracted and exercised directly), same
spirit as `../ffa7g/`'s ls20 replay table:
- Per-cell trust converges correctly (a synthetic "moves counter" cell becomes trusted exactly
  at `warmup`, not before).
- **Safety property holds at per-cell granularity**: a real-gameplay cell the model never
  claims is NEVER masked, even while it's actively changing — same "union only verified
  claims, never subtract" guarantee `_HudCodeModel` had, now per-cell instead of whole-model.
- **Non-monotonic self-correction**: a cell the model starts consistently mispredicting (e.g. a
  counter that resets on a new level) drops OUT of trust within the rolling `window`, with no
  whole-model reset and the registered function left untouched.
- Re-registering different code resets per-cell trust to empty (old accuracy isn't evidence for
  new code).
- A real bug was CAUGHT by writing this test before shipping: the first draft of
  `HUD_SUBAGENT_SYSTEM_PROMPT` described `advance_hud(frame: list[str], ...)`, but the actual
  harness interface (unchanged from `_HudCodeModel`) is string-based (`frame` is the
  newline-joined `.ascii` text) — the test's synthetic `advance_hud` failed until this was
  fixed to match. Exactly the kind of mismatch that would have silently produced zero
  registrations in a live run (the LLM's list-shaped code would `exec` fine but crash inside
  `predict()`'s `try/except`, silently returning `None` forever).

**Patch mechanics:** `-p1 --dry-run` and a real apply both clean against a fresh
`../baseline-v12/src/ARC3-Inference` copy; all 7 touched files pass `py_compile`.

**Live smoke test #1 (2026-07-30, RedHatAI FP8, 7-game subset, concurrency 8,
`HUD_MODEL_REFRESH_ACTIONS=4`): confirmed the scheduling mechanism fires correctly, caught a
real timeout bug before it could waste the full run.** `refresh_hud_model` fired on schedule
(direct evidence in the startup log), proving the host-triggered call itself works. But every
single attempt failed with `HTTPConnectionPool ... Read timed out (read timeout=30.0)` — a
naive hardcoded 30s timeout on the dedicated call, while the main analyzer loop has NO timeout
by default (`LOCAL_ANALYZER_TIMEOUT=0` -> `None`) specifically because it can legitimately
queue behind other concurrent games' completions on a saturated 8-concurrency server. Every
attempt was timing out before ever getting a response, so `advance_hud` could never register
in this run regardless of how many times it fired — killed the run early (would have burned
the full ~2h budget for zero signal). **Fixed**: `_HUD_MODEL_TIMEOUT_S` (env `HUD_MODEL_TIMEOUT_S`,
default 120s) gives the dedicated call the same queueing headroom as the main loop. Bundle
rebuilt, patch regenerated, standalone test re-verified, smoke test relaunched.

**Live smoke test #2 (same config, relaunched with the timeout fix): timed out again at the
new 120s, but a direct SSH check of vLLM's own `/metrics` at that moment showed
`num_requests_running=7, num_requests_waiting=0` — zero queue depth.** Not a queueing problem
at all: `refresh_hud_model` was silently inheriting `_chat_completion`'s default
`max_tokens`/`thinking` from the SAME `ToolAgent` instance config used for real game-playing
turns -- this harness's validated-best config for that job is uncapped output + thinking ON,
which is exactly why full-25 runs need a 132-minute budget per game. A narrow HUD-modeling
call has no business generating on that scale; it wasn't stuck, it was still genuinely
thinking. **Fixed properly this time**: `_chat_completion` gained optional `max_tokens`/
`thinking` override parameters (defaulting to the old shared-config behavior for every other
caller, so the main loop is unaffected), and `refresh_hud_model` now passes
`max_tokens=_HUD_MODEL_MAX_TOKENS` (env `HUD_MODEL_MAX_TOKENS`, default 2048) and
`thinking=False` explicitly — this call needs pattern-matching + code, not extended reasoning,
and Qwen-family `enable_thinking=false` is the dominant lever for generation length (see
[[laguna-model-swap]]). Bundle rebuilt again, patch regenerated, standalone test re-verified,
smoke test relaunched a third time.

**Lesson for next time reusing a shared LLM-calling helper for a narrower side-task: check
every setting it silently inherits, not just the ones that obviously matter (tools/messages) —
output length and reasoning-mode are just as capable of silently blowing a request's budget as
an actual server queue, and a naive symptom-level fix (bump the timeout) does not surface that
until the SECOND live attempt.**

**Live smoke test #3 (same config, relaunched with the max_tokens/thinking fix) — completed
cleanly, 2h12m, mean score 3.07 (median 2.78) on the 7-game subset.** No crashes, no
tracebacks, no `SERVER FAILED`/`FAILED`/`TEARDOWN_FAILED` markers — the new code is safe to run
end to end. Per-game: bp35 0.44, ka59 0.00, r11l 4.76, sb26 2.78, sk48 2.78, tn36 10.71, wa30
0.00 (5/7 games completed at least one level). Nothing here regresses vs. the historical
scoring range on this subset for other candidates.

**The dedicated call mechanism is CONFIRMED WORKING end to end for the first time ever in this
project** (searched all 7 transcripts for the real `key: value` no-quote rendering, not the
system-prompt prose that mentions the same substrings — the JSON-shaped grep from earlier
sessions gives false negatives here): `hud_model: checking` appears for 4 of 7 games (ka59 x3,
sb26 x105, tn36 x20, wa30 x61) — direct proof the dedicated call fired, the model wrote valid
Python, `_extract_hud_model_code` extracted it, and `_HudPixelModel.register()` accepted it
with no error. **But no game ever reached `hud_model: active` or `no_impact_source: model`/
`model+band`** — registered code never got any cell's rolling accuracy above the 90% threshold
within the `warmup=3` window. So: the delivery pipe (schedule → ask → extract → register →
verify) now genuinely works: what doesn't yet work is the model reliably writing a per-cell
predictor accurate enough to earn trust. Plausible causes, not yet distinguished: (a) `bp35`,
`r11l`, `sk48` never got the model to register anything at all in this short a run (may need
more than a handful of transitions, or these games' HUD elements genuinely aren't learnable
this way); (b) for the 4 that DID register, the predictions may be close-but-not-pixel-exact
(a single mismatched cell anywhere in a claimed row keeps that whole row's cells below
threshold); (c) `max_tokens=2048` with `thinking=False` may be too tight for the model to both
reason about the transitions AND emit a correct function in one shot.

**Not yet done:** distinguish those causes (e.g. log the actual generated code/response
content for the dedicated call, currently invisible since it's deliberately isolated from the
main transcript — a real telemetry gap), try loosening `HUD_MODEL_MAX_TOKENS`/warmup/threshold,
and only then consider a full-25 A/B against the `ffa7gn` baseline. As a prototype-validation
milestone this is a clean stopping point: the new architecture (per-cell trust + dedicated
call) is proven to run safely and to exercise the mechanism it was built to fix, even though
tuning it to actually reach "active" status is unfinished.

## Full-25 run (`g4run-hudpixel-ffa7gn-20260730-2242`, 3h/game budget, concurrency 36,
`HUD_MODEL_TIMEOUT_S=300`)

First attempt at this run (concurrency 28, 120s timeout, launched 22:11) hit the same
queueing-headroom problem the subset smoke test's attempt #1 hit, just at larger scale: the
dedicated call couldn't get serviced inside 120s even though it's a cheap bounded request,
because at concurrency 28 it's competing for the same vLLM slots as 25 games' own turns
simultaneously (not true in the 7-game subset). Stopped and relaunched with `max-num-seqs`
raised 28→36 and `HUD_MODEL_TIMEOUT_S` raised to 300s (see `gcp/v12hudpixel_ffa7gn_startup.sh`
and `gcp/launch_hudpixel_ffa7gn.sh`).

**Completed cleanly: 3h0m49s, mean score 1.18 (median 0.00), 2367 total actions, 1.31M
tokens.** For direct comparison, the RedHatAI FP8 `ffa7gn` baseline (no hudpixel, standard
132-min/game budget, no dedicated call) scored 2.46 and 1.39 across its own two passes (mean
1.925) — see `g4run-qwenquant-ffa7gn-redhatai-fp8-p{1,2}-20260729-0855`. A single 1.18 run
falls inside the documented noise band for this scoring metric (see
[[score-metric-is-noise-dominated]], ~0.45–2.67 as the 95% range on a 25-game mean) — **this
result cannot distinguish "hudpixel hurts" from ordinary run-to-run variance**, and note the
3h budget bought essentially no extra actions over the 2h12m baseline (2367 vs ~2500) since
the harness is throughput-capped per game (see [[arc3-throughput-ceiling]]), not time-capped —
the extra wall-clock headroom was mostly unused.

**`hud_model` never reached `active` in any of the 25 games.** Grepped all 25 transcripts for
the real `key: value` rendering: `checking` appears 91 times across the run, `active` zero
times. Same outcome as the subset smoke test, now confirmed at 3.5x the wall-clock budget and
with the queueing/timeout infrastructure genuinely fixed (raising concurrency+timeout changed
nothing about the underlying finding) — the bottleneck is not infrastructure, it's the
dedicated call's ability to write a per-cell-accurate predictor, per the three not-yet-
distinguished causes above.

**Bug found while grepping for status strings: the system prompt addendum (`prompts.py`,
the `GAME_OVERVIEW_ADDENDUM`-style HUD section) still describes the OLD `_HudCodeModel`
mechanism verbatim** — it tells the agent to write `advance_hud(frame, action) -> frame`
itself in a `HUD model:` section, and says a misprediction produces `hud_model: stale`. That
string was never updated when `_HudPixelModel` replaced `_HudCodeModel`; `stale` is not a
real status the new code ever emits (only `active`/`checking`, see `solver.py:1680`) — it's
dead documentation baked into every single turn's context (1574 matches for `hud_model:
stale` across the 25 transcripts, 100% of them inside the repeated system-prompt text, not
runtime output). Harmless to correctness (the old prose-fallback path still works and feeds
the same `_HudPixelModel` instance either way), but it's misleading to the agent about what
outcome to expect and burns a few dozen tokens of context every turn on a stale explanation.
Should be rewritten to describe the dedicated-call + per-cell-trust design if this line of
work continues.

**Conclusion:** the hudpixel prototype is validated as safe to run at full scale (no crashes,
no infra failures after the concurrency/timeout fix) but has not yet demonstrated the actual
mechanism working — no cell in any game across 32 total game-attempts (7 subset + 25 full)
has ever earned trust. This is not a scoring regression (the 1.18 mean is noise, not signal)
but it is also not yet a win: the `no_impact` statistical band remains the only thing actually
firing in every game. Next step, if pursued further, is the "not yet done" list above — most
importantly, making the dedicated call's actual generated code/response visible for
inspection, since right now the mechanism's failure mode is a black box.
