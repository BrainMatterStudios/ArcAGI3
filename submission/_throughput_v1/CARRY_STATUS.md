# CARRY — compaction instead of eviction (Track A1, built 2026-09-08, NOT launched)

Plan: docs/PLAN-2026-09-08-revised-plan-to-7plus.md §6 A1 ("carry + compact"). Prerequisite read:
docs/research-2026-09-08/R-reasoning-carriage-0908.md — **the "carry" half was already true**: the stock keeps
`reasoning` on every persisted assistant message, this vLLM build maps it onto `reasoning_content`, and the
Flash-Next template renders `<think>…</think>` for every assistant message (`preserve_thinking` undefined ⇒ true).
Verified on 947 within-turn + 321 cross-turn request pairs of two rig waves: the server's prompt-token delta
equals tokenized reasoning + note + tool result within ~30 tokens (slope 0.999–1.000). Re-injecting reasoning
would double-count it. What the stock LOSES is what the trimmer evicts (27 % of consecutive requests on the rig
evicted; the window fills after ~10 turns; older turns survive only as the one-line World-model note).

Files (new unless marked; no stock byte edited — the runner still asserts the june_stock tree sha `74ab6910…`):

| file | what |
|---|---|
| `submission/_throughput_v1/graft_carry.py` | the graft (`install()` -> `carry: OK`, `status()`), master flag `CARRY_ENABLE` |
| `submission/_throughput_v1/test_graft_carry.py` | 12 tests (fake session + the REAL python sandbox; both bundles) |
| `submission/_throughput_v1/dry_run_carry.py` | real-engine dry run (june_stock + pinned taaf + arcengine, vc33/ls20/sb26, 14k window) |
| `offkaggle/run_regime_wave.py` (edited) | arm `keith_carry`, `[HARNESS CARRY]` extractor, `carry` telemetry + gate, 4 summary lines, mock compaction reply |
| `offkaggle/test_run_regime_wave.py` (edited) | arm invariants, in-memory install, extractor on a CARRY transcript, `keith_carry` dry run end to end |
| `docs/research-2026-09-08/R-reasoning-carriage-0908.md` (+ `reasoning_delta_probe.py`, the template copy) | the prerequisite read |
| `offkaggle/REGIME_WAVE_STATUS.md` (appended) | pre-registration |

## 1. Mechanism (all flags read at call time; `CARRY_ENABLE=0` = byte-identical stock)

Hooks (`ToolAgent`, identical seams in both bundles): `_trim_messages_for_context` (called at turn start, before
every request, on the overflow retry, and by `_persistent_history_messages` at turn end), `_chat_completion`
(telemetry only), `analyze` (per-turn state). Per trim:

1. `messages[0]` (system) is rebuilt from `self._system_prompt` + the current compacted block (single copy;
   the system message is never persisted — `_persistent_history_messages` keeps `trimmed[1:]`).
2. If the stock estimate ≤ `context_budget_tokens` (31,744): stock trim (a no-op).
3. Else: evict with the STOCK `_drop_oldest_history_block` until the estimate ≤ `CARRY_TARGET_FRACTION` [0.5] ×
   budget, collecting the dropped messages (≥ `CARRY_MIN_DROP_MSGS` [2], else stock).
4. **Compaction call** (one extra request; `agent_mod.requests.post`, so the shim records it; no tools;
   `enable_thinking` per `CARRY_COMPACT_THINKING` [0]; `max_tokens` `CARRY_COMPACT_MAX_TOKENS` [1500];
   timeout min(`CARRY_TIMEOUT_S` [600], the turn's request timeout); skipped with < 30 s of game left):
   system = `COMPACT_SYSTEM` (fixed headings: MECHANICS VERIFIED / HYPOTHESES REFUTED / GOAL MODEL / KEY OBJECTS AND
   COORDINATES / TRIED ON THIS LEVEL / OPEN QUESTIONS / CURRENT PLAN / EARLIER LEVELS; "keep every item from the
   previous block that is still true; merge"), user = level + previous block + the dropped turns rendered as
   `[TURN STATE]` (state lines + the model's own note; boilerplate stripped) / `[YOUR REASONING]` (head 3,500 +
   tail 1,500 chars) / `[YOUR NOTE]` / `[YOUR CODE]` (1,500) / `[TOOL RESULT]` (1,000), whole text capped at
   `CARRY_INPUT_CHARS` [48,000] by cutting the middle. Reply → the new block (capped `CARRY_SUMMARY_CHARS` [4,800]).
   Any failure (HTTP ≥ 400, empty, exception) → eviction proceeds without a block change (`compaction_failures`).
5. Stock trim of `[system+block, *remaining]` enforces the hard budget.

Geometry at the live budget: system + tools ≈ 4.7k est tokens; target 0.5 ⇒ after a compaction ~11k est of
history stays (~4 turns), refill takes ~6–7 turns ⇒ **~7 compactions per 52-call game**, each a queued call
(~150 s at the live cadence ⇒ ~13 % of the clock). At 0.65 it would be ~11 (the reason for 0.5).

Markers (`[HARNESS CARRY]` sections; the extractor treats them as their own sections, META→THINKING adjacency intact):
`[CARRY-CALL] game=<stem> turn=<n> req=<i> msgs=<N> reasoning_msgs=<R> reasoning_chars=<C> summary_chars=<S>
prompt_tokens=<P> completion_tokens=<Q>` (one per model call, written before the call's META) and
`[CARRY-COMPACT] game=<stem> turn=<n> dropped_msgs=<k> input_chars=<c> summary_chars=<s> prompt_tokens=<p>
completion_tokens=<q> e2e_s=<t> ok=1|0 [err=<why>]` + `SUMMARY:` + the block.

## 2. Test results (2026-09-08)

`test_graft_carry.py` — **12/12 on BOTH bundles** (june_stock and the anim bundle):
01 install/seams; 02 flag-off byte-identical to stock on a budget where the stock trimmer evicts (transcript bytes,
request messages, persisted history; no `_carry` state, no post); 03 prior-turn reasoning is in the next request and
measured (`[CARRY-CALL] … reasoning_msgs=1`); 04 compaction fires: payload has no tools, `enable_thinking` False,
`max_tokens` 1500, the dropped reasoning reached the compactor, `(none yet)` on the first, the previous block on the
next, the block rides the next request's system message exactly once and the dropped turn is gone, never persisted,
marker + `SUMMARY:` in the transcript, token accounting includes the call; 05 every request's estimate ≤ budget and a
20k-char reply is capped to `CARRY_SUMMARY_CHARS`; 06 failures (ConnectionError / HTTP 500 / empty) fall back to
eviction, the turn still acts, later compactions recover; 07 skips (`small_drop`, `overflow_path`, `no_time`);
08 a new game gets a fresh block; 09 exceptions inside the graft return the stock result; 10 status/env parsing;
11 rendering of dropped turns (boilerplate stripped, caps); 12 window counters (`prompt_over_window`).

`dry_run_carry.py` (real engine, 3 games, 14k window) — **PASS 12/12**, 8 s: 90 calls, 28 compactions (9–10 per
game), 0 failures; markers == graft counters == mock; one `[CARRY-CALL]` per call; compaction request had no tools,
thinking off, max_tokens 1500; the dropped turns' reasoning reached the compactor every time; first compaction per
game saw `(none yet)`, the other 25 saw the previous block; the block rode 81 later requests, never duplicated; the
stock estimate of every request ≤ budget (max 11,870 of 12,976); every request carried the retained reasoning.

`offkaggle/test_run_regime_wave.py` — **29/30** (new: `test_carry_arm_installs_graft_in_memory`,
`test_extractor_reads_carry_markers`, `test_dry_run_keith_carry_arm_end_to_end`; the 30th,
`test_game_overs_from_events_counts_compact_json`, is a pytest-fixture test that the script runner could not call —
given a default in this commit). Runner dry run (`--arm keith_carry --knob LOCAL_ANALYZER_CONTEXT_WINDOW=9000
--max-calls 14 --per-game-s 120`): 30 compactions / 3 games, ENGAGED = YES, the block in every prompt log.

## 3. Runner arm, telemetry and the reads

`keith_carry` = `KEITH_YIELD900_ENV` + `CARRY_ENABLE=1 CARRY_TARGET_FRACTION=0.5 CARRY_SUMMARY_CHARS=4800
CARRY_INPUT_CHARS=48000 CARRY_COMPACT_MAX_TOKENS=1500 CARRY_COMPACT_THINKING=0 CARRY_MIN_DROP_MSGS=2`;
`ARM_GRAFTS["keith_carry"] = ("graft_carry",)`; `CARRY_` scrubbed for every other arm (tested).

Summary lines: `CARRY` (compactions/game, failures, per-compaction dropped msgs / input chars / e2e / tokens, block
chars, **ENGAGED gate**), `CARRY-WINDOW` (prompt tok/call mean+max from usage, over-32,768 count, calls carrying
the block, first block call #, reasoning msgs/chars carried per request), `CARRY-PRIMARY` (levels vs 39.33 sd
2.34 + the 12 never-passed walls), `CARRY-SAFETY` (GAME_OVERs/run vs 0.87, live-cap vs 8.42, fit-the-clock).

**ENGAGEMENT** = compactions ≥ 1.0/game AND failures ≤ 10 % of attempts AND `prompt_tokens > 32768` never AND
≥ 40 % of calls carry the block.

## 3b. Defect found in the first live run and fixed (2026-09-08, commit after the kill test)

The kill test produced one `400 ... "No user query found in messages."` in 157 calls (lf52_p0, action 63). The
chat template raises that when no `role == "user"` message survives a trim (tool results ride as `role: "tool"`
and are rendered as `<tool_response>`, which the template explicitly does not count).

**It is a stock defect, reproduced on the unmodified bundle with the graft never imported**: with one turn's own
assistant+tool pairs over the budget (10 calls x 8k reasoning + 4k tool results), the stock drop loop plus
`_drop_until_first_user_message` returns the system message alone. Prior stock waves never hit it (0 in 75 runs
of three waves) because they average 1.0-2.1 calls/turn; this arm reaches it sooner because the system message
now also carries the compacted block (~1.3-1.6k tokens of headroom).

Fix (graft only; the stock bundle is untouched): (a) the drop-to-target loop stops before evicting the last real
user message (`keep_last_user`); (b) every enabled-path return goes through a guard that, if no user message
survived, re-trims with the stock-sized system message (`no_user_after_trim`) and, failing that, rebuilds the
request as system-with-block + the turn's own user prompt (`rebuilt_from_last_user`). Verified on all three
severities: a sendable request that keeps the block and fits the budget; the realistic path (8 calls, 4k
reasoning) is unchanged. Regression test 13 in `test_graft_carry.py` asserts both the stock behaviour and the
recovery. NOTE for reading the wave: this makes the arm differ from stock in a second, smaller way — it converts
a request the stock would fail into a shorter valid one. It fires only where the stock would have 400'd.

## 4. Known limitations

1. No game has been played against the real model; the block's QUALITY (does Flash-Next write a useful, honest
   summary of its own turns with thinking off?) is the experiment. Read the `SUMMARY:` blocks in the transcripts.
2. Each compaction is a queued call (~150 s live): ~7/game ≈ 13 % of the clock; the fit-the-clock line reports
   play calls × e2e; add compaction e2e × count.
3. `--max-calls` counts only regular `_chat_completion` calls, not compaction posts (the shim counts both).
4. The block is cleared only on a new game; level transitions keep it (the compaction prompt is told the level and
   asks for an EARLIER LEVELS section). The stock still clears its own one-line note on a level change.
5. The retained user prompts still carry ~700 tokens of identical boilerplate each (~7k of the window); slimming
   history prompts is a separate rider, not part of this arm.
6. Not stacked with any other graft.
7. Block truncation: 1 of 12 compactions in the kill test hit BOTH caps at once (`summary_chars` 4,800 and
   `completion_tokens` 1,500), so that block was cut mid-sentence and is fed forward as prior knowledge.
   Median block is 3,921 chars, so this is an edge case; if it exceeds ~15 % in the wave, raise
   `CARRY_COMPACT_MAX_TOKENS` as a separate single-knob change, not mid-arm.

## 5. Launch commands (NOT run; token read from `~/.config/arc3/vllm_token` into memory only)

Kill test first (fresh boot; the 3-wall instrument geometry with a long clock, base 8 levels / 6 runs):
```
cd /Users/ahmed/Documents/ArcAGI3
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_carry --games cd82,dc22,lf52 --draws 2 \
    --concurrency 3 --max-calls 60 --per-game-s 7920 \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --out offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_carry-kill
```
Then, only if ENGAGED and levels ≥ 7 of the base's 8, the 25-game wave (fresh boot, live geometry):
```
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_carry --games all \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --out offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_carry
```
