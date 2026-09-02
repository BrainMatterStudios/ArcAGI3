# Stage-0 kill test — Qwen3.8-27B-FP8 vs frontier-recorded level-0 transitions

Pre-registered question: given the frontier's recorded level-0 transitions for a hard ARC-AGI-3 game plus a backtest tool,
can Qwen3.8-27B (vLLM FP8, thinking on, T=0.6/top_p=0.95) produce a backtest-green executable world model?
KILL RULE (fixed before data): green on >= 2 of 3 games {sk48, tn36, cn04} within <= 20 model calls per game, else the lane is dead.

Work dir: `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/stage0/`
(`transitions.py`, `backtest.py`, `llm.py`, `stage0.py`, `<game>_transitions.json`, `runs/<game>[_0b]/{prompt0.txt,cand_NN.py,bt_NN.json,resp_NN.json,log.jsonl,summary.json}`, `stage0.log`, `stage0b.log`).

## 1. Data extraction (transitions.py)
Source: HF `schema-harness/arc-agi-3-schema-traces` events.jsonl (`turn_started` grid at env_step 0 = entry grid; `action_taken`
events carry action/x/y, resulting grid, `level_up`/`dead`/`win` flags). Level 0 = step 0 up to and including the first `level_up`.

| game | frontier model | level-0 transitions | of which RESET | clicks |
|---|---|---|---|---|
| sk48 | claude-fable-5 | 42 (indices 0..41) | 1 (index 27) | 0 |
| tn36 | claude-fable-5 | 12 | 0 | 12 |
| cn04 | claude-opus-4-8 | 18 | 0 | 0 |

Click convention verified from data: x = column, y = row (tn36 click (26,43) changes cap cells at row 42, cols 25-27).

## 2. Backtest instrument (backtest.py)
Contract as actually used by the frontier files (NOT the `(state, grid, flags)` order in the task text):
stateful `init_state(entry)` + `predict(state, grid, action, x, y) -> (grid, flags, state)`, or stateless `step(grid, action, x, y) -> (grid, flags)`;
`ENTRY_GRID` injected as a module global. Scoring: teacher-forced replay; grid compared cell-for-cell on non-terminal steps, flags on
every step; on the terminal (level_up) step only flags (the post-level-up grid is the next level's board); RESET steps skipped with the
model state re-initialised on the post-reset grid (`reinit`); exceptions/timeouts/crashes = mismatch with traceback. Candidates run in a
subprocess with a 180 s timeout. Stricter than the frontier harness in one way: step 0 IS checked (the frontier harness skipped it as
"no prior grid").

### Instrument validation (mandatory gate) — PASSED
| game | frontier file | result |
|---|---|---|
| sk48 | `world_model_v5.py` (final) and `snapshots/cleared_level_1..7.py` | **41/41 green** (1 reset skipped) |
| sk48 | `snapshots/cleared_level_0.py` | 40/41 — only the final `level_up` flag missing (the frontier had not yet encoded the goal when it cleared L0; its own harness read 40/40 only after level-1 edits) |
| tn36 | `snapshots/cleared_level_0.py` and `world_model_v5.py` | **12/12 green** |
| cn04 | `snapshots/cleared_level_0.py` and `world_model_v5.py` | **18/18 green** |
Reset policy check: `continue` (state carried across RESET) breaks sk48 at index 28 (budget bar), `reinit` is green -> `reinit` is the
harness semantic. Mock dry-run of the full loop (`stage0.py --mock`, model returns the frontier's final file): green in 1 call on all 3 games.

## 3. Endpoint constraints discovered (changed the harness, not the kill rule)
- The Modal endpoint kills ANY HTTP request at ~300 s (`modal-http: internal error: function execution timed out`), streaming included.
  At ~55 tok/s that caps a single request at ~15k output tokens. The first attempt (max_tokens=16000) ended `finish=length` with the
  whole budget spent in thinking and no file (sk48 call 1: 288 s, 16000 tokens, 27.5k chars of reasoning spent re-counting hex rows).
- Per the coordinator's instruction the cap was raised to max_tokens = min(40000, 65536 - prompt - 512). To honour that under the 300 s
  wall, one logical model call is produced as a CHAIN of `/v1/completions` segments (<= 10k tokens each, ~180 s) on token ids: the chat
  template is applied via `/tokenize` (its generation prompt already ends in `<think>\n`), each segment's output ids are appended and
  generation continues until EOS or the 40k budget. Verified coherent across 6 chained segments. Counted as ONE model call.
- Stop rule added (coordinator): two consecutive calls ending `finish=length` with no code -> stop the game, failure mode
  "cannot emit code within a 40k-token call". Per-call prompt/completion tokens logged from the API usage field.
- Prompt window: system + data + the last K rounds of (candidate, feedback) (K=2, or 1 if the initial prompt > 25k tokens).

## 4. Arm A — pre-registered encoding (hex rows; first N transitions as full grids, rest as per-row changed runs; prompt <= ~24k)
Prompt sizes (vLLM tokenizer): sk48 21.5k (4 full grids + 37 diffs), tn36 19.7k (8 full grids + 3 diffs), cn04 6.8k (all 18 full).

| game | green? | best matched/total | calls | wall | prompt tok | completion tok | what happened |
|---|---|---|---|---|---|---|---|
| sk48 | NO | 0/41 (no candidate ever emitted) | 4 | 39.2 min | 87,083 | 127,567 | calls 1,3,4: 40,000 tokens of thinking, no file (tail of call 4's reasoning is a half-written `predict()` with frame/track/riding-block state — close to the frontier's mechanics, but still drafting inside `<think>` when the budget ended; 114 count/recount passages). Call 2: EOS emitted INSIDE the think block after 7.6k tokens (no `</think>`, empty answer). Stopped by the 2x-length rule. |
| tn36 | **YES** | **12/12** | 2 | 24.6 min | 39,617 | 79,189 | call 1: 40k thinking, no file. Call 2: 39.2k tokens, GREEN. Candidate is a genuine mechanics model (connected components of color 1 = toggle slots; click toggles slot 1<->5; row-1 budget bar spends one 9->3 per click; `level_up` on a click when no 1s remain). Different hypothesis from the frontier's (caps/stems/run button) but reproduces all 12 transitions. |
| cn04 | NO | 0/18 (no candidate ever emitted) | 2 | 24.1 min | 13,947 | 80,000 | both calls: 40,000 tokens of thinking, no file; reasoning never reached code ("I can't fully determine the e-shape's movement rules... make my best guess"); 67 count/recount passages. Stopped by the 2x-length rule. |

**Verdict under the kill rule (Arm A): 1 of 3 green -> the lane is DEAD as pre-registered.**
Dominant failure mode: the model cannot get from data to an emitted file inside a 40k-token generation; a large share of the
thinking is spent re-counting characters in 64-char hex rows / reconciling column indices.

## 5. Arm B — Stage-0b, coordinate-explicit encoding (NOT pre-registered; coordinator-requested follow-up)
Same contract, backtest, chained 40k cap, stop rules. Encoding: entry grid as width-2 integers with a column ruler on top and row labels
at left (identical consecutive rows collapsed); every transition as ONLY its changed cells grouped by row, `rROW: cCOL:old->new`, with
contiguous columns sharing the same old->new collapsed to `cA-B:old->new` (deviation from the literal `(r,c): old->new` list, which cost
47k tokens on sk48; the grouped form is 32.8k). Steps with > 200 changed cells get the full grid (2 such in sk48, 1 in cn04).
Prompt sizes: sk48 32.8k, tn36 7.8k, cn04 17.4k tokens. History window K=1 for sk48 (prompt > 25k), K=2 otherwise.

| game | green? | best matched/total | calls | wall | prompt tok | completion tok | what happened |
|---|---|---|---|---|---|---|---|
| sk48 | NO | 0/41 (no candidate ever emitted) | 4 | 31.0 min | 132,687 | 99,415 | effective cap was ~32k/call (65536 - 32.8k prompt - 512). Calls 1,3,4: budget exhausted in thinking (32,141 / 32,022 / 31,216 tokens), no file; call 2: EOS inside the think block after 4k tokens. Stopped by the 2x-length rule. |
| tn36 | **YES** | **12/12** | 3 | 16.7 min | 25,923 | 53,607 | call 1: 40k thinking, no file. Call 2: 884-token reply, 6/12 (toggled the wrong cells at transition 1). Call 3: 12.7k tokens, GREEN. Same mechanics family as Arm A's solution (color-1 components as toggle slots, Chebyshev-1 click hit test on the slot centre, row-1 budget bar, level_up = click on the 9-band once no 1s remain). |
| cn04 | NO | 0/18 (no candidate ever emitted) | 2 | 24.2 min | 35,035 | 80,000 | both calls: 40,000 tokens of thinking, no file. Stopped by the 2x-length rule. |

**Verdict under the kill rule (Arm B): 1 of 3 green -> DEAD again.** The coordinate-explicit encoding did not rescue sk48 or cn04;
it made tn36 faster (16.7 vs 24.6 min, 54k vs 79k completion tokens) but tn36 was already green under Arm A.

## 6. Overall verdict
Both arms: green only on tn36 (12 transitions, click-toggle mechanic). sk48 (42 transitions, crane/rope/riding-block physics with a
reset) and cn04 (18 transitions, key/lock lattice with rotation) never produced a single candidate file in either encoding: every
call either exhausted the 32-40k output budget inside `<think>` or emitted EOS mid-thought. **The lane is dead as pre-registered
(needs >= 2 of 3).** The failure is not "wrong model, iterate": on the two hard games the model never reaches the emit step at all.

## 7. Things I could not verify / caveats
- Only ONE draw per game per arm (T=0.6). The tn36 green in both arms and the 0/41, 0/18 in both arms are consistent, but n=1 per cell.
- Endpoint: Modal's ~300 s per-request kill is a deployment setting I did not change; chained continuation re-tokenizes nothing
  (server-returned token ids are re-fed), but a chained generation is not bit-identical to a single uninterrupted one (sampling RNG
  restarts per segment). It was verified coherent on a 6-segment test and on these runs (reasoning continues mid-sentence).
- The stop-rule label "within a 40k-token call" is literally true for tn36/cn04; for sk48 under Arm B the cap was ~32k because of the
  32.8k prompt.
- sk48 `cleared_level_0.py` is 40/41 on my instrument (missing terminal flag); the frontier's later snapshots and final file are 41/41,
  which is what validates the instrument. I did not re-derive transitions from the offline `arc_agi` engine; the recorded grids were
  used as ground truth (the frontier's own harness scored 40/40 on the same steps, so they are internally consistent).
- The tn36 candidates generalize differently from the frontier's model (they treat all color-1 components as toggles, ignore the
  cap/stem program semantics); they are green on the 12 recorded transitions only. Not tested on unseen tn36 actions.
- The aborted first attempt (16k cap, non-chained; sk48 call 1 288 s / 16000 tokens / no code) and the Modal-timeout attempt are kept
  in `runs/sk48_aborted_16k/` and `runs/sk48_aborted_modal_timeout/`, `stage0_aborted_16k.log`.
