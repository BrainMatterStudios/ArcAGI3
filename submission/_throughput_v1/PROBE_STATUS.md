# PROBE — harness-enforced probe discipline (built 2026-09-08, judge fixes applied 09-08, NOT launched)

loss-ledger-3 candidate #1 (docs/research-2026-09-08/R-loss-ledger-3.md): under the 900 s turn budget
the call mix did not change (analysis-only 49 %), 84 turns / 75 runs burn 5-7 analysis calls and yield
with nothing executed, ANALYSIS-PARALYSIS is 33 % of stuck tails, wall-level exploration is 0.72x the
human baseline, and appended prompt blocks are ignored (graft_hypo uptake 0.8 %). So the discipline is
enforced by the harness: at most 2 analysis-only python calls per span; the next snippet that would
again execute nothing is not run and gets a tool result demanding a <= 5-action test instead.

Files (new unless marked; no stock byte edited — the runner still asserts the june_stock tree sha
`74ab691052406c22…` after the in-memory install):

| file | what |
|---|---|
| `submission/_throughput_v1/graft_probe.py` | the graft (`install()` -> `probe: OK`, `status()`), master flag `PROBE_ENABLE` |
| `submission/_throughput_v1/test_graft_probe.py` | 14 tests (fake session + the REAL python sandbox; both bundles) |
| `submission/_throughput_v1/dry_run_probe.py` | real-engine dry run, two brains (compliant / stubborn), june_stock + pinned taaf + arcengine |
| `offkaggle/run_regime_wave.py` (edited) | arm `keith_probe`, PROBE telemetry, gate / primary / safety reads, 4 summary lines, `--mock-analysis-calls` |
| `offkaggle/test_run_regime_wave.py` (edited) | arm invariant, flag scrub, extractor + aggregate on a PROBE transcript, in-memory install, `keith_probe` dry run |
| `offkaggle/REGIME_WAVE_STATUS.md` (appended) | pointer under the 09-08 pre-registration |

## 0. Judge fixes applied (SHIP-WITH-FIXES, 2026-09-08)

1. `PROBE_MAX_REFUSALS` default 2 -> **4** (graft, arm env, tests, docs).
2. **Span semantics**: the analysis-only count AND the refusal count are no longer reset by a turn that
   executed nothing. A NOACT turn carries both into the next turn on the SAME level of the same game
   (`carried_turns`); the span resets only after a turn that executed an action (at once, when that
   turn ends), on a level change (`_level_of`: the live session's `levels_completed`, else the runtime
   frame's level; unknown level = same level) or on a new game (runtime dir). A model that answers
   refusals by yielding cannot buy a fresh budget with a new turn; the cap bounds the whole span.
3. New counters: `first_refusal_followups` / `acted_after_first_refusal` (did the very next python
   call after the FIRST refusal of a span execute an action; a refusal that was the turn's last call
   is settled as a NON-acting follow-up), `refusal_turn_ending`, `carried_turns`, and the
   `turns_ge3_analysis` leak split into `leak_cap_lifted` (the leaking 3rd analysis-only snippet had
   no `action()` call — only the lifted cap, `PROBE_MAX_REFUSALS=0`, or a budget >= 3 let it run),
   `leak_dead_branch` (it CONTAINED an `action()` call but executed none) and `leak_unparsable`.
   `refusal_cap` skips now count only snippets that would have been refused.
4. Runner gate/reads rewritten (§5): ENGAGEMENT = refusals >= 1/game AND acted-after-first-refusal
   >= 50 % (graft counters, turn-ending refusals in the denominator) AND wall actions/baseline median
   >= 0.9; secondary = spans with >= 3 executed analysis-only calls (leak split) and yields/draw;
   PRIMARY = levels vs the pooled six-draw base 39.33 (sd 2.34) with delta, co-primary = walls passed
   among the 12 six-draw-never-passed walls (list from loss-ledger-3 NOTES Q1); SAFETY = GAME_OVERs/run
   vs 0.87 and live-cap score/game vs 8.42.
5. Dry run extended with a stubborn brain that never acts: proves the 4-refusal cap and the carry.

## 1. How "analysis-only" is detected — the payload, with evidence

The stock `ToolAgent._run_python_tool` (june_stock `inference/agent/tool_agent.py:1453-1588`) runs the
snippet in the sandbox subprocess and returns

```
_ToolDispatchResult(content=<JSON string the model reads>, step_executed=step_executed)
step_executed = any(bool(item.get("executed")) for item in action_results)      # tool_agent.py:1580
```

where `action_results` is the list of per-`action()` payloads the sandbox collected
(`python_tool_sandbox.py` `action()` appends `reply["action_result"]`; the host keeps its own copy in
`host_action_results`), and each payload is the solver's `step_env` result compacted by
`_compact_action_result` (`executed`, `executed_actions`, `executed_count`, `stop_reason`, ...). The
analyzer loop ends the turn on the first dispatch whose `step_executed` is True (`tool_agent.py:1953`).
So `step_executed` IS the harness's own answer to "did this python call execute a game action", and the
graft counts an analysis-only call as **a python call whose `_ToolDispatchResult.step_executed` is
False**. Three consequences, all asserted in test 03 on the real sandbox:

* `print(len(history))` -> `step_executed=False` -> analysis-only (stdout shows it ran);
* `action(['BOGUS'])` -> the solver's `_error_payload` (`solver.py:544`, `"executed": False`) ->
  `step_executed=False` -> analysis-only **even though the code calls action()**;
* `x = 1/0` -> error payload -> analysis-only.

The graft never re-parses code for the count. Code is inspected in exactly one place: the refusal must
decide BEFORE the snippet runs (the refused snippet is not executed), when no payload exists yet. There
`code_calls_action()` does `ast.parse` and refuses iff the tree holds no `Call` whose func is the name
`action` (attribute `.action(...)` also counts as a call). A snippet that mentions `action()` anywhere,
even in a branch that will not run, is always executed and then counted from its payload (and, if it
is the 3rd analysis-only call of the span, counted as `leak_dead_branch`); a snippet that does not parse
is left to the stock syntax-error path and counts as analysis-only afterwards (`leak_unparsable`).

## 2. The mechanism, per span (all flags read at call time)

`analyze()` = one turn; a SPAN = consecutive turns on one level up to and including the first that
executes an action. `_run_python_tool` wrapper:

1. **Refuse** when `analysis_calls >= PROBE_MAX_ANALYSIS` [2] and `refusals < PROBE_MAX_REFUSALS` [4]
   and the code has no `action()` call: the snippet is not run; the tool result goes out through the
   same rendering a python error takes (`self._render_tool_payload({"tool": "python", "error": TEXT},
   truncate_fields=("stdout","error","result"))`, `step_executed=False`), so `analyze()` appends it as
   the `{"role": "tool"}` message, `_render_tool_result_display` shows the bare text under
   `[TOOL RESULT: python]`, and the persisted history carries it forward.
2. **Deadlock cap**: after `PROBE_MAX_REFUSALS` refusals in one span the graft stops refusing (skip
   counter `refusal_cap`); the stock loop proceeds and the 3rd executed analysis-only call is counted
   as `turns_ge3_analysis` + `leak_cap_lifted`.
3. **NOACT**: a turn that ends with `step_executed=False` and is neither a `retryable_failure` (the
   solver retries the same step) nor a `stop_requested` yield (the run is ending; skip counter) is
   recorded, its counters carry, and the next `_build_user_prompt` for the same game gets ONE line
   prepended: `Previous turn executed no action after N analysis calls.` (+ ` (R refused)` when R > 0;
   N and R are the span's running counts; <= 160 chars), consumed once. Informational only.
4. **Follow-up reads** settle on the next python call of the span or at the turn's end: `after ANY
   refusal` (`calls_after_refusal` / `acting_calls_after_refusal`) and `after the FIRST refusal of the
   span` (`first_refusal_followups` / `acted_after_first_refusal`); a refusal that ends a turn is
   counted in `refusal_turn_ending` and settled as non-acting.

Exact refusal text (`graft_probe.REFUSAL_TEXT`; `{n}` = executed analysis-only calls in the span,
`{k}` = `PROBE_MAX_PROBE`; the note lines come from the carried note's `open_questions`, then
`current_plan` (the stock maps the model's `Next test:` there) then `world_model` (where the stock puts
`Hypothesis:`), split on `;` / sentence ends / list numbering, up to `PROBE_NOTE_LINES` [3] lines of
<= 140 chars; whole text hard-capped at 900 chars):

```
Analysis budget for this turn is spent (2 analysis-only calls). Only a snippet that executes a game action is accepted now: run a <=5-action test of your leading hypothesis with action([...]) and read the result. Untested hypotheses in your notes:
- does SPACE open the door
- is the edge bar a timer.
- Plan: test it.
```
(`<=` is ASCII on purpose: the content is `json.dumps`-escaped for the model, `≤` would reach it as
`≤`.) With no labelled note: `Untested hypotheses in your notes: (none recorded - state one now and test it)`.

Transcript markers (harness's own `_append_transcript_section`, label `[HARNESS PROBE]`):
`[PROBE-REFUSE] game=<run stem> turn=<analysis_step> analysis_calls=<n> refusal=<k>/<max>` (inside
the refused call's `[TOOL CALL: python]` section, right before its `[TOOL RESULT]`) and
`[PROBE-NOACT] game=… turn=… analysis_calls=<n> refusals=<k> reason=<yield|no_capture>` (right after
the turn's `[ANALYZER STATUS]`). `status()`: `refusals, turns_with_refusal, noact_turns, carried_turns,
analysis_calls_total, acting_calls_total, calls_after_refusal, acting_calls_after_refusal,
first_refusal_followups, acted_after_first_refusal, refusal_turn_ending, turns_total,
turns_ge3_analysis, leak_cap_lifted, leak_dead_branch, leak_unparsable` (+ the two shares,
`errors, skips, per_game` keyed by run stem `<gid>_p<draw>`). Flag off (`PROBE_ENABLE=0`): no state
object, no markers, byte-identical transcript, prompts and persisted history (test 02, a turn shaped
A A A X where the flag-on graft would refuse the 3rd call).

## 3. Test results (2026-09-08, after the fixes)

`test_graft_probe.py` — **14/14 on BOTH bundles** (`GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference`
and the default anim bundle):

* 01 install OK, seams `_probe_stock` on `analyze`, `_run_python_tool`, `_build_user_prompt`
* 02 flag-off byte-identical (transcript bytes, prompts, persisted history; the 3rd snippet RAN; no `_probe` attr; counters 0)
* 03 analysis-only from the payload: acts / pure analysis / `action(['BOGUS'])` failed action / python error; the AST
  predicate; the three leak buckets (`leak_cap_lifted`, `leak_dead_branch` for `if False: action(...)`, `leak_unparsable`)
* 04 refusal fires exactly at the 3rd analysis-only call, not at the 1st/2nd; refused snippet's sentinel absent from the
  result and no `stdout` key; payload shape `{"tool","error"}`; exact text incl. the 3 note lines; marker `refusal=1/4`;
  display = bare text; a 3rd call WITH `action()` is never refused; first-refusal follow-up 1/1 acted
* 05 deadlock cap: A A R R R R then the 7th snippet runs (`refusal_cap` skip 1), `turns_with_refusal` once,
  `calls_after_refusal` 4/0 acting, first-refusal follow-up = the 2nd refusal (non-acting), `leak_cap_lifted` 1;
  `PROBE_MAX_REFUSALS=0` never refuses; `=2` lifts after two; `PROBE_MAX_ANALYSIS=0` refuses the first call
* 06 through `analyze()`: file order TOOL CALL -> `[PROBE-REFUSE]` -> `[TOOL RESULT: python]\nAnalysis budget…`; the 4th
  request carries the refusal as a tool message; the action executes; an acting turn closes the span (A A X next turn, no
  refusal); `carried_turns` 0
* 07 NOACT: marker `reason=no_capture` once; the notice is the FIRST line of exactly the next `[USER PROMPT]` and absent
  after; a pre-call yield gives `analysis_calls=0` / `reason=yield`; `stop_requested` yield and `request_error` are NOT NOACT
* 08 status shape incl. the new counters, env parsing (garbage -> default, negative -> 0), per-game keys == `_PER_GAME_KEYS`
* 09 exception inside the graft -> stock result returned, `errors` counted (both seams)
* 10 note extraction (numbering, `;`, dashes, caps 140/900 chars, `(none recorded…)`, `PROBE_NOTE_LINES=0`), NOACT line <= 160
* 11 a new game (runtime dir) gets fresh state; per_game keyed by run stem
* 12 **carry**: A A (NOACT) -> the next turn's first analysis-only snippet is refused at once
  (`turn=2 analysis_calls=2 refusal=1/4`, `carried_turns` 1); an acting turn resets; refusals carry too
  (A A R R NOACT -> next turn refuses at 3/4 and 4/4, then the cap lifts: `leak_cap_lifted` 1)
* 13 **level change resets the span** (fresh budget on the new level, notice still rides); unknown level = same level
* 14 **turn-ending refusal**: A A R under a 3-step cap -> `refusal_turn_ending` 1, follow-ups settled non-acting
  (1/0 and first 1/0, share 0.0); the carried next turn's action is not credited to it; a request error right after
  a refusal leaves the follow-up pending for the retried step

`dry_run_probe.py` (june_stock + pinned taaf + REAL arcengine, vc33/ls20/sb26, yield 900 s) — **PASS 16/16**, ~25 s:

* compliant brain (tool steps 4, 24 actions/game): 75 turns, 75 refusals == `[PROBE-REFUSE]` markers == refusal tool
  results == requests in which the mock saw the text; sentinel results 0; turn 1 of every game = A A R R -> NOACT
  (`noact_turns` 3 = `carried_turns` 3 = `refusal_turn_ending` 3), its first-refusal follow-up settled non-acting; every
  later pattern turn (A A S X) = one refusal followed by an action: `acted_after_first_refusal` 69/72 (the 3 misses are
  the turn-ending ones), `acting_calls_after_refusal` 69/75; leaks 0; notices 3 on exactly the next prompt; errors 0.
* stubborn brain (tool steps 6, 15 s/game, analysis-only forever): 0 actions in 150 turns; **exactly 4 refusals per
  game** (`refusal=4/4` marker once per game) although each game burned ~50 NOACT turns — the cap carried across every
  NOACT turn (`carried_turns` 147, a NOACT marker with `analysis_calls=332`); `leak_cap_lifted` 3 = one per span;
  `refusal_turn_ending` 3 (the 4th refusal ended turn 1), first-refusal follow-ups 3/0 acted; `refusal_cap` skips 872;
  147 NOACT markers, 147 notices; errors 0.

`offkaggle/test_run_regime_wave.py` — **26/26** (3 new: `test_probe_arm_installs_graft_in_memory`,
`test_extractor_reads_probe_markers_and_ledger_call_types`, `test_dry_run_keith_probe_arm_end_to_end`; 2
extended: arm invariant incl. the never-passed wall list and the base mean/sd recomputed from the six draws, flag
scrub). Regression: `test_graft_retry` 19/19, `test_graft_evidence` 17/17, `test_graft_hypo` 9/9 (june_stock).

## 4. Dry-run proof — a refusal followed by an action on the real engine

`scratchpad/tp_dry_run/probe-093954-compliant/transcripts/vc33-5430563c_p0.txt` (the model-facing tool
message is the JSON `{"tool": "python", "error": "<this text>"}`; the transcript shows its display):

```
[TOOL CALL: python]
<tool_call>
<function=python>
<parameter=code>
print('SENTINEL-REFUSED-SNIPPET')
</parameter>
</function>
</tool_call>
[HARNESS PROBE]
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=3 analysis_calls=2 refusal=1/4
[TOOL RESULT: python]
Analysis budget for this turn is spent (2 analysis-only calls). Only a snippet that executes a game action is accepted now: run a <=5-action test of your leading hypothesis with action([...]) and read the result. Untested hypotheses in your notes:
- does SPACE open the door
- is the edge bar a timer.
- Plan: test it.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
...
[TOOL CALL: python]
<parameter=code>
prefs = [v for v in valid_actions if str(v).upper() in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]
step = len(history)
if prefs:
    r = action(prefs[step % len(prefs)])
...
print('acted', r.get('executed'))
</parameter>
[TOOL RESULT: python]
acted True
[ANALYZER STATUS]
...
step_executed: True
message: Step executed.
```

and the carried NOACT phase (turn 1 = A A R R under a 4-step cap; the notice on turn 2; the stubborn
run's marker showing the carried count):

```
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=1 analysis_calls=2 refusal=1/4
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=1 analysis_calls=2 refusal=2/4
message: No action(...) call was captured.
[PROBE-NOACT] game=vc33-5430563c_p0 turn=1 analysis_calls=2 refusals=2 reason=no_capture
--- analysis_step=2 | action=0 | ... | tool-agent ---
[USER PROMPT]
Previous turn executed no action after 2 analysis calls (2 refused).
                                  (stubborn run, ~50 turns later, still refusals=4)
[PROBE-NOACT] game=vc33-5430563c_p0 turn=51 analysis_calls=332 refusals=4 reason=no_capture
```

## 5. Runner arm, telemetry and the three reads

`keith_probe` = `KEITH_YIELD900_ENV` (keith V14 + `LOCAL_ANALYZER_YIELD_SECONDS=900`) + `PROBE_ENABLE=1
PROBE_MAX_ANALYSIS=2 PROBE_MAX_PROBE=5 PROBE_MAX_REFUSALS=4 PROBE_NOTE_LINES=3`; `ARM_GRAFTS["keith_probe"]
= ("graft_probe",)`; `PROBE_` in `GRAFT_ENV_PREFIXES` (scrubbed from the shell for every other arm; tested).
The invariant test asserts the arm differs from `keith_yield900` by exactly the five PROBE keys.

Per run (`telemetry.json: per_game[stem].probe`; definitions `TELEMETRY_DEFINITIONS["probe"|"call_types"|"probe_gate"|"probe_span"]`):
markers (`refusals`, `noact_turns`, `noact_notices`, `turns_with_refusal`); the ledger's `q3.py ctype` call
classes from the transcript (`N`, `E`, `X` = code contains `\baction\(`, `A`) plus `R` = refused;
`turns_ge3_analysis` (>= 3 `A`-class calls before the first `X`, the ledger's 15 %; refused excluded) and
the strict `turns_ge3_nonacting`; `acting_after_refusal` and `acted_after_first_refusal` from the
transcript (turn-ending refusal = non-acting follow-up); `probe.graft` = the graft's own payload-based
counters for the run (all of §2 incl. the leak split); `yields_turn_time_budget`; `wall_actions_ratio` =
`actions_per_level[levels_completed] / base_actions_per_level[levels_completed]` (loss-ledger-3 q7);
`game_overs` = action rows with `game_over: true` in the run's `artifacts/<stem>_events.jsonl`
(ledger q2); `live_cap_score` = the ledger's `score()` (100 · Σ (l+1)·min(1.15, (b/a)²) / Σ 1..n). Offline
engine only for baselines-based reads (None on Kaggle).

Pooled (`aggregate.probe`) and the four summary lines:

* `PROBE` — **ENGAGEMENT gate** (`gate.engaged`): refusals >= 1/game AND `graft.acted_after_first_refusal_share`
  >= 50 % (turn-ending refusals in the denominator; transcript read printed beside it) AND wall
  actions/baseline median >= 0.9 (ledger-3 0.72).
* `PROBE-2ND` — spans with >= 3 executed analysis-only calls (graft; target < 5 %, ledger-3 15 %) with the
  leak split `cap_lifted / dead_branch / unparsable`, the transcript's A-class and strict reads, carried
  turns, turn_time_budget yields per draw (ledger-3 27-30, target < 20), NOACT turns/notices, call mix.
* `PROBE-PRIMARY` — levels per draw vs the pooled six-draw base **39.33 (sd 2.34)** (Y1 41, Y2 40, YK 37,
  M1 36, M4 40, K 42) with the delta in levels and in sd (25-game waves only; >= 52 step candidate, 45-51
  redraw, < 45 dead) and the co-primary **walls passed among the 12 six-draw-never-passed walls** —
  `NEVER6_WALLS = bp35 L2, dc22 L2, g50t L2, lf52 L2, lp85 L6, ls20 L2, r11l L3, sb26 L2, sp80 L2, tn36 L3,
  vc33 L4, wa30 L2` (loss-ledger-3 NOTES Q1; passed = `levels_completed >= wall` in any draw; target >= 3).
* `PROBE-SAFETY` — GAME_OVERs per run vs the yield900 base 0.87 and live-cap score per game vs 8.42.

Requested dry run (`--dry-run --arm keith_probe --games tu93,ft09,cd82 --concurrency 3 --max-calls 12`;
the mock answers the first 3 python calls of each turn with analysis-only snippets, default for a probe
arm, recorded as `results.json:mock_analysis_calls`):

```
  PROBE  refusals 7 (2.33/game; gate >= 1) games 3 | acted-after-FIRST-refusal graft 6/7 (85.7%; gate >= 50.0%; turn-ending refusals 1 count as non-acting) transcript 6/7 (85.7%) | after ANY refusal graft 6/7 (85.7%) transcript 6/7 | wall actions/baseline median 0.05 (n=3, under 1x 3; gate >= 0.9, ledger-3 0.72) | ENGAGED = NO {refusals_per_game+, acted_after_first_refusal+, wall_actions_ratio-} | grafts {'graft_probe': 'probe: OK'} | flags {'PROBE_ENABLE': '1', 'PROBE_MAX_ANALYSIS': '2', 'PROBE_MAX_PROBE': '5', 'PROBE_MAX_REFUSALS': '4', 'PROBE_NOTE_LINES': '3'}
  PROBE-2ND spans >=3 executed analysis-only (graft) 0/9 (0.0%; target < 5.0%, ledger-3 15.0%) leak: cap_lifted 0 dead_branch 0 unparsable 0 | transcript turns >=3 A-class 0/9 (0.0%) strict incl. refused/errors 88.9% | carried turns 0 | turn_time_budget yields 0 (0.0/draw; ledger-3 27-30, target < 20) | NOACT turns 0 (notices 0) | call mix {'A': 18, 'R': 7, 'X': 6, 'N': 5} analysis share 50.0% (ledger-3 49.0%)
  PROBE-PRIMARY levels 0 over 3 runs / 1 draw(s) = 0.0/draw vs pooled six-draw base 39.33 (sd 2.34) -> delta -39.3 (-16.81 sd; 25-game waves only; >= 52 step candidate, 45-51 redraw, < 45 dead) | never-passed walls 0/12 passed [] (present in wave: 0; co-primary target >= 3)
  PROBE-SAFETY GAME_OVERs 0 (0.00/run over 3 runs; yield900 base 0.87) | live-cap score 0.00 (0.00/game over 3 runs; yield900 base 8.42/game)
  run               lv/n    act  calls turns reas_mean  len%  notool%  e2e_s   score  retry probe wall/b  evid  hypo  wall%  upt%  c@L2 att void  state
  cd82-fb555c5d_p0    0/6      2    12     3      1541      -    8.3%    0.3    0.00    0/0   3/0   0.04     0     0   0.0%  0.0%  None   -    -  gave_up
```
(7 refusals, one of them the last call of a run under `--max-calls 12` — settled as a non-acting
turn-ending follow-up, hence 6/7. The three `stop_requested` yields at the cap are skipped, not NOACT.
ENGAGED = NO here only because a 12-call mock takes 2 actions against baselines of 20-40: the wall ratio
read works. The end-to-end test also runs the same mock knob on the stock `keith_yield900` arm and asserts
the sentinel snippet RUNS there — the refusal is the graft, not the mock.)

## 6. Known limitations

1. **No game has been played against the real model.** The compliant mock obeys every refusal it can; the
   live acted-after-first-refusal share, and whether probing turns into levels, is the experiment.
2. The pre-run predicate is static: a snippet whose `action()` sits in a branch that never runs (or that
   errors before reaching it) is executed and counted from its payload (`leak_dead_branch` when it is the
   span's 3rd); it is never refused. The model cannot be refused for a snippet that tries to act and fails.
3. A refused call still costs a model call (~150 s at the live cadence, ~130 s of it queue): with the
   cap at 4 a span can burn 2 analysis + 4 refusals (6 calls, ~15 min) before the loop proceeds. The
   stubborn dry run shows the worst case: the harness cannot make a model act, it can only stop
   subsidising analysis.
4. The ledger's ">= 3 analysis-only calls" is code-based and counted executed calls only (there were no
   refusals); the strict transcript read will look worse than the base whenever the model needs the
   refusal to act. The secondary read uses the graft's executed-analysis count per span.
5. The NOACT notice is one line in the turn's own prompt and stays in the persisted history like any
   prompt text (not stripped the way `[HYPO]` is); refusal texts (~300-450 chars each) also stay in the
   history until the stock trimmer drops the turn.
6. Note lines are only as good as the carried note: the stock extractor collapses whitespace and maps
   `Hypothesis:` onto `world_model`, so the split is heuristic; a model that never writes a labelled note
   gets `(none recorded - state one now and test it)`.
7. Level detection for the span reads the live session's `levels_completed` (the seam graft_retry uses);
   without a session it falls back to the runtime frame's level, and with neither it treats the next turn
   as the same level (carry). A harness-issued level RESET keeps `levels_completed`, so a span survives a
   RESET on the same level — by design (the level is still uncleared).
8. `wall_actions_ratio`, `live_cap_score` need `base_actions_per_level` (offline engine only); on Kaggle
   they are None and the gate's wall condition cannot be read. The Kaggle notebook was not touched.
9. Not stacked with `graft_evidence` / `graft_hypo` in any test (single-lever arm). If stacked, the
   refusal returns before `graft_evidence`'s wrapper sees a payload in either install order.
10. GAME_OVER counting reads `artifacts/<stem>_events.jsonl`; a run whose events file is missing reads None
    and is excluded from the per-run mean (the summary prints the run count).

## 7. Launch command (NOT run; token read from `~/.config/arc3/vllm_token` into memory only)

Pre-registration: offkaggle/REGIME_WAVE_STATUS.md, "PRE-REGISTERED — arm `keith_probe`" (09-08) as amended
by the judge (§0.4 above). Base for comparison: the pooled six-draw base 39.33 (sd 2.34) levels. **First in
a fresh session, live geometry** (25 games, concurrency 28, 7920 s/game), no `--knob`:

```
cd /Users/ahmed/Documents/ArcAGI3
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_probe --games all \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --out offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_probe
```

Read `summary.txt`: `IDENTITY … ok=True`, `VLLM … preemptions 0`, length-finish <= 1 % (else VOID); then
`PROBE … ENGAGED = YES/NO` (refusals >= 1/game, acted-after-first-refusal >= 50 %, wall ratio >= 0.9);
`PROBE-2ND` for the leak split and yields; `PROBE-PRIMARY` for levels vs 39.33 (>= 52 step candidate ->
counterbalanced draw; 45-51 redraw; < 45 or engaged-but-flat -> dead) and the never-passed walls
(co-primary >= 3 of 12); `PROBE-SAFETY` for GAME_OVERs/run vs 0.87 and live-cap score vs 8.42/game.
