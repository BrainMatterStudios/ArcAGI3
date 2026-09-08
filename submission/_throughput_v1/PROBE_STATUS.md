# PROBE — harness-enforced probe discipline (built 2026-09-08, NOT launched)

loss-ledger-3 candidate #1 (docs/research-2026-09-08/R-loss-ledger-3.md): under the 900 s turn budget
the call mix did not change (analysis-only 49 %), 84 turns / 75 runs burn 5-7 analysis calls and yield
with nothing executed, ANALYSIS-PARALYSIS is 33 % of stuck tails, wall-level exploration is 0.72x the
human baseline, and appended prompt blocks are ignored (graft_hypo uptake 0.8 %). So the discipline is
enforced by the harness: at most 2 analysis-only python calls per turn; the next snippet that would
again execute nothing is not run and gets a tool result demanding a <= 5-action test instead.

Files (new unless marked; no stock byte edited — the runner still asserts the june_stock tree sha
`74ab691052406c22…` after the in-memory install):

| file | what |
|---|---|
| `submission/_throughput_v1/graft_probe.py` | the graft (`install()` -> `probe: OK`, `status()`), master flag `PROBE_ENABLE` |
| `submission/_throughput_v1/test_graft_probe.py` | 11 tests (fake session + the REAL python sandbox; both bundles) |
| `submission/_throughput_v1/dry_run_probe.py` | real-engine dry run (june_stock + pinned taaf + arcengine, vc33/ls20/sb26, mock brain) |
| `offkaggle/run_regime_wave.py` (edited) | arm `keith_probe`, PROBE telemetry + pre-registered gate, `PROBE`/`PROBE-WALL` summary lines, `--mock-analysis-calls` |
| `offkaggle/test_run_regime_wave.py` (edited) | arm-difference invariant, flag scrub, extractor on a PROBE transcript, in-memory install, `keith_probe` dry run |
| `offkaggle/REGIME_WAVE_STATUS.md` (appended) | pointer under the 09-08 pre-registration |

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
even in a branch that will not run, is always executed and then counted from its payload; a snippet
that does not parse is left to the stock syntax-error path (more useful to the model than a refusal)
and counts as analysis-only afterwards.

## 2. The mechanism, per turn (all flags read at call time)

`analyze()` = one turn. The wrapper resets the counters when a turn starts. `_run_python_tool` wrapper:

1. **Refuse** when `analysis_calls >= PROBE_MAX_ANALYSIS` [2] and `refusals < PROBE_MAX_REFUSALS` [2]
   and the code has no `action()` call: the snippet is not run; the tool result goes out through the
   same rendering a python error takes (`self._render_tool_payload({"tool": "python", "error": TEXT},
   truncate_fields=("stdout","error","result"))`, `step_executed=False`), so `analyze()` appends it as
   the `{"role": "tool"}` message, `_render_tool_result_display` shows the bare text under
   `[TOOL RESULT: python]`, and the persisted history carries it forward.
2. **Deadlock cap**: after `PROBE_MAX_REFUSALS` refusals in one turn the graft stops refusing (skip
   counter `refusal_cap`); the stock loop proceeds and a 3rd executed analysis-only call is counted as
   `turns_ge3_analysis` (enforcement leaked).
3. **NOACT**: a turn that ends with `step_executed=False` and is neither a `retryable_failure` (the
   solver retries the same step) nor a `stop_requested` yield (the run is ending; skip counter) is
   recorded; the next `_build_user_prompt` for the same game gets ONE line prepended:
   `Previous turn executed no action after N analysis calls.` (+ ` (R refused)` when R > 0; <= 160
   chars), consumed once. Informational only.

Exact refusal text (`graft_probe.REFUSAL_TEXT`; `{n}` = executed analysis-only calls this turn,
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
`[PROBE-REFUSE] game=<run stem> turn=<analysis_step> analysis_calls=<n> refusal=<k>/<max>` (lands
inside the refused call's `[TOOL CALL: python]` section, right before its `[TOOL RESULT]`) and
`[PROBE-NOACT] game=… turn=… analysis_calls=<n> refusals=<k> reason=<yield|no_capture>` (right after
the turn's `[ANALYZER STATUS]`). `status()`: `refusals, turns_with_refusal, noact_turns,
analysis_calls_total, acting_calls_total, calls_after_refusal, acting_calls_after_refusal,
acting_after_refusal_share, turns_total, turns_ge3_analysis, errors, skips, per_game` (keyed by the
run stem `<gid>_p<draw>` from the transcript path). Flag off (`PROBE_ENABLE=0`): no state object, no
markers, byte-identical transcript, prompts and persisted history (test 02, a turn shaped A A A X where
the flag-on graft would refuse the 3rd call).

## 3. Test results (2026-09-08)

`test_graft_probe.py` — **11/11 on BOTH bundles** (`GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference`
and the default anim bundle):

* 01 install OK, seams `_probe_stock` on `analyze`, `_run_python_tool`, `_build_user_prompt`
* 02 flag-off byte-identical (transcript bytes, prompts, persisted history; the 3rd snippet RAN; no `_probe` attr; counters 0)
* 03 analysis-only from the payload: acts / pure analysis / `action(['BOGUS'])` failed action / python error; the AST predicate
* 04 refusal fires exactly at the 3rd analysis-only call, not at the 1st/2nd; refused snippet's sentinel absent from the
  result and no `stdout` key; payload shape `{"tool","error"}`; exact text incl. the 3 note lines; marker line; display =
  bare text; a 3rd call WITH `action()` is never refused; counters (1 refusal, 1/1 acting after)
* 05 deadlock cap: A A R R then the 5th snippet runs (`refusal_cap` skip), `turns_with_refusal` once, `calls_after_refusal`
  2/0 acting, `turns_ge3_analysis` 1; `PROBE_MAX_REFUSALS=0` never refuses; `PROBE_MAX_ANALYSIS=0` refuses the first call
* 06 through `analyze()`: file order TOOL CALL -> `[PROBE-REFUSE]` -> `[TOOL RESULT: python]\nAnalysis budget…`; the 4th
  request carries the refusal as a tool message; the action executes; per-turn reset on the next `analyze()` (A A X, no refusal)
* 07 NOACT: marker `reason=no_capture` once; the notice is the FIRST line of exactly the next `[USER PROMPT]` and absent
  after; a pre-call yield gives `analysis_calls=0` / `reason=yield`; `stop_requested` yield and `request_error` are NOT NOACT
* 08 status shape, env parsing (garbage -> default, negative -> 0), per-game counters
* 09 exception inside the graft -> stock result returned, `errors` counted (both seams)
* 10 note extraction (numbering, `;`, dashes, caps 140/900 chars, `(none recorded…)`, `PROBE_NOTE_LINES=0`), NOACT line <= 160
* 11 a new game (runtime dir) gets fresh state; per_game keyed by run stem

`dry_run_probe.py` (june_stock + pinned taaf + REAL arcengine, vc33/ls20/sb26, mock brain, yield 900 s,
tool steps 6) — **PASS 10/10**, 8 s: 75 turns, 78 refusals (75 turns with a refusal + the 2nd refusal on
each game's NOACT turn), `[PROBE-REFUSE]` markers 78 == refusal tool results 78 == requests in which the
mock saw the text 78; sentinel results 0 (the refused snippet never ran); acting-after-refusal 72/72 on
the phase-2 turns; `refusal_cap` skips 6 (2 per NOACT turn); 3 NOACT turns, 3 notices, each on exactly
the next prompt; graft counters == transcript; errors 0.

`offkaggle/test_run_regime_wave.py` — **26/26** (3 new: `test_probe_arm_installs_graft_in_memory`,
`test_extractor_reads_probe_markers_and_ledger_call_types`, `test_dry_run_keith_probe_arm_end_to_end`; 2
extended: arm invariant, flag scrub). Regression: `test_graft_retry` 19/19, `test_graft_evidence` 17/17,
`test_graft_hypo` 9/9 (june_stock).

## 4. Dry-run proof — a refusal followed by an action on the real engine

`scratchpad/tp_dry_run/probe-091053/transcripts/vc33-5430563c_p0.txt`, turn 3 (the model-facing tool
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
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=3 analysis_calls=2 refusal=1/2
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

and the NOACT phase of the same file (turn 1 = A A R R A A under tool steps 6, then the notice):

```
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=1 analysis_calls=2 refusal=1/2
[PROBE-REFUSE] game=vc33-5430563c_p0 turn=1 analysis_calls=2 refusal=2/2
message: No action(...) call was captured.
[PROBE-NOACT] game=vc33-5430563c_p0 turn=1 analysis_calls=4 refusals=2 reason=no_capture
--- analysis_step=2 | action=1 | 09:06:57 | tool-agent ---
[USER PROMPT]
Previous turn executed no action after 4 analysis calls (2 refused).
```

## 5. Runner arm and telemetry

`keith_probe` = `KEITH_YIELD900_ENV` (keith V14 + `LOCAL_ANALYZER_YIELD_SECONDS=900`) + `PROBE_ENABLE=1
PROBE_MAX_ANALYSIS=2 PROBE_MAX_PROBE=5 PROBE_MAX_REFUSALS=2 PROBE_NOTE_LINES=3`; `ARM_GRAFTS["keith_probe"]
= ("graft_probe",)`; `PROBE_` added to `GRAFT_ENV_PREFIXES` (scrubbed from the shell for every other arm;
test asserts a stale `PROBE_*` never reaches `keith_yield900`). The invariant test asserts the arm differs
from `keith_yield900` by exactly the five PROBE keys.

Telemetry per run (`telemetry.json: per_game[stem].probe`, definitions in `TELEMETRY_DEFINITIONS["probe"]`
/ `["call_types"]` / `["probe_gate"]`):

* `refusals`, `noact_turns`, `noact_notices` — marker counts; `turns_with_refusal`.
* `call_types` — the ledger's `q3.py ctype` classes reproduced from the transcript (`N` no tool call,
  `E` error without `action(` in the code, `X` code contains `\baction\(`, `A` otherwise) plus `R` =
  refused. NOTE the ledger reads the CODE, the graft reads the PAYLOAD: an `action()` that failed is `X`
  for the ledger and analysis-only for the graft. Both numbers are reported; `analysis_call_share`
  (`A`/calls) is the ledger's 49 %.
* `turns_ge3_analysis` = turns with >= 3 `A`-class calls before the first `X` (the ledger's 15 %;
  refused calls excluded — they were not executed) — the pre-registered gate number;
  `turns_ge3_nonacting` = >= 3 of ANY non-`X` class before the first `X` (refusals, errors, no-tool
  replies included) — the strict read. Under this arm the strict share is structurally high whenever the
  model complies only after a refusal (A A R X = 3 non-acting calls): read the gate on the first, report both.
* `acting_after_refusal` — the next python call after a `[PROBE-REFUSE]` in the same turn is `X`
  (transcript, code-based); `probe.graft` carries the graft's own payload-based counters for the run
  (`acting_calls_after_refusal / calls_after_refusal`) — the summary prints both; the gate uses the graft read.
* `yields_turn_time_budget` — turns whose status says `turn_time_budget` (the ledger's "yields": 84 on 75 runs).
* `wall_actions`, `wall_baseline`, `wall_actions_ratio` = `actions_per_level[levels_completed] /
  base_actions_per_level[levels_completed]` on an unfinished run — exactly loss-ledger-3 `q7.py`
  ("wall-level actions/baseline", median 0.72). `game_rows` now records `baselines` from
  `game_run.base_actions_per_level` (offline engine only; None in submission mode -> ratio None).

Pooled (`aggregate.probe`): totals and per-game rates, pooled shares, `graft` pooled counters, `call_types`,
`yields_per_draw` (= total / draws), `wall_actions_ratio {n, median, under_1x}`, and `gate` = the
pre-registration in REGIME_WAVE_STATUS.md 09-08 (`refusals >= 1/game`, `acting-after-refusal >= 50 %`,
`turns_ge3_analysis < 5 %` -> `engaged`) plus the ledger's two secondary reads (`yields < 20/draw`,
`wall ratio median >= 1.0`), each with `ok`. Summary lines `PROBE` and `PROBE-WALL`; per-run columns
`probe` (refusals/noact) and `wall/b`.

Requested dry run (`--dry-run --arm keith_probe --games tu93,ft09,cd82 --concurrency 3 --max-calls 12`;
the mock answers the first 3 python calls of each turn with analysis-only snippets, default for a probe
arm, recorded as `results.json:mock_analysis_calls`):

```
  PROBE  refusals 7 (2.33/game; gate >= 1) games 3 | acting-after-refusal graft 6/6 (100.0%) transcript 6/6 (100.0%; gate >= 50.0%) | turns >=3 analysis-only 0/9 (0.0%; gate < 5.0%, ledger-3 15.0%) strict incl. refused/errors 88.9% | NOACT turns 0 (notices 0) | call mix {'A': 18, 'R': 7, 'X': 6, 'N': 5} analysis share 50.0% (ledger-3 49.0%) | grafts {'graft_probe': 'probe: OK'} | flags {'PROBE_ENABLE': '1', 'PROBE_MAX_ANALYSIS': '2', 'PROBE_MAX_PROBE': '5', 'PROBE_MAX_REFUSALS': '2', 'PROBE_NOTE_LINES': '3'}
  PROBE-WALL actions/baseline median 0.05 (n=3, under 1x 3; ledger-3 0.72, target >= 1) | turn_time_budget yields 0 (-/draw; ledger-3 27-30, target < 20) | ENGAGED (pre-registered: refusals, after-refusal, >=3-analysis) = YES {refusals_per_game+, acting_after_refusal+, turns_ge3_analysis+, yields_per_draw-, wall_actions_ratio-}
  run               lv/n    act  calls turns reas_mean  len%  notool%  e2e_s   score  retry probe wall/b  evid  hypo  wall%  upt%  c@L2 att void  state
  cd82-fb555c5d_p0    0/6      2    12     3      1432      -    8.3%    0.3    0.00    0/0   3/0   0.04     0     0   0.0%  0.0%  None   -    -  gave_up
```
(7 refusals, 6 calls after a refusal: the 7th refusal was a run's last call under `--max-calls 12`. The
three `stop_requested` yields at the cap are skipped, not NOACT. The end-to-end test also runs the same
mock knob on the stock `keith_yield900` arm and asserts the sentinel snippet RUNS there — the refusal is the
graft, not the mock.)

## 6. Known limitations

1. **No game has been played against the real model.** The mock complies with every refusal; the live
   acting-after-refusal share, and whether probing turns into levels, is the experiment.
2. The pre-run predicate is static: a snippet whose `action()` sits in a branch that never runs (or that
   errors before reaching it) is executed and counted from its payload; it is never refused. The model
   cannot be refused for a snippet that tries to act and fails — by design.
3. A refused call still costs a model call (~150 s at the live cadence, ~130 s of it queue): the lever
   buys action discipline, not calls. Two refusals + two analysis calls per turn is the worst case
   before the loop proceeds (`PROBE_MAX_REFUSALS`).
4. The ledger's ">= 3 analysis-only calls" is code-based and counted executed calls only (there were no
   refusals); the strict transcript read will look worse than the base whenever the model needs the
   refusal to act. The gate is pre-registered on the executed-analysis read.
5. The NOACT notice is one line in the turn's own prompt and stays in the persisted history like any
   prompt text (not stripped the way `[HYPO]` is); refusal texts (~300-450 chars each) also stay in the
   history until the stock trimmer drops the turn.
6. Note lines are only as good as the carried note: the stock extractor collapses whitespace and maps
   `Hypothesis:` onto `world_model`, so the split is heuristic (`;`, sentence ends, numbering); a model that
   never writes a labelled note gets `(none recorded - state one now and test it)`.
7. The `[PROBE-NOACT]` marker is written after the turn's `[ANALYZER STATUS]`, inside that section's body
   for the runner's parser; `[PROBE-REFUSE]` sits inside the refused `[TOOL CALL]` body. Both are matched
   at line start with the field shape; a `[PROBE-REFUSE]` in model prose is not counted (tested).
8. `wall_actions_ratio` needs `base_actions_per_level`, which the engine exposes OFFLINE only (the rig);
   it is None on Kaggle. The Kaggle notebook was not touched; this arm exists for the off-Kaggle rig only.
9. Not stacked with `graft_evidence` / `graft_hypo` in any test (single-lever arm). If stacked, the
   refusal returns before `graft_evidence`'s wrapper sees a payload in either install order.

## 7. Launch command (NOT run; token read from `~/.config/arc3/vllm_token` into memory only)

Pre-registration: offkaggle/REGIME_WAVE_STATUS.md, "PRE-REGISTERED — arm `keith_probe`" (09-08). Base
for comparison: the keith_yield900 rig draws 41 / 40 levels. **First in a fresh session, live geometry**
(25 games, concurrency 28, 7920 s/game), no `--knob`:

```
cd /Users/ahmed/Documents/ArcAGI3
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_probe --games all \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --out offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_probe
```

Read `summary.txt`: `IDENTITY … ok=True`, `VLLM … preemptions 0`, length-finish <= 1 % (else VOID); then
the `PROBE` line — ENGAGEMENT = refusals >= 1/game AND acting-after-refusal (graft read) >= 50 % AND
turns >= 3 analysis-only < 5 %; secondary `PROBE-WALL` — wall actions/baseline median toward >= 1.0 (now
0.72) and turn_time_budget yields < 20/draw (now 27-30). PRIMARY = levels: >= 52 step candidate (then a
counterbalanced draw and >= 3 of the 12 never-passed walls); 45-51 positive, redraw; < 45 or
engaged-but-flat -> dead (the model probes and still fails: comprehension, not cadence).
