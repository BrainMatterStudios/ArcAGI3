# EVID / HYPO / UP8 — the 3-wall instrument arms (built 2026-09-06, NOT launched)

Judge program item 1 (docs/research-2026-09-06/J-judge-0906.md): three single-lever arms on
the keith V14 base for the turn-capped 3-wall test (cd82 L2, dc22 L2, lf52 L2; 2 draws per arm;
`--per-game-s 1500`). The judge's tail reads say the walls lose on EVIDENCE HANDLING (the
post-clear frame diffed as level data, misread batched logs, a click at (16,25) vs the model's
own (16,32), false memory of when an object appeared), not on hypothesis starvation.

Files (new unless marked):

| file | what |
|---|---|
| `submission/_throughput_v1/graft_evidence.py` | EVIDENCE-INTEGRITY AID (`install()` -> `evidence: OK`, `status()`), master flag `EVID_ENABLE` |
| `submission/_throughput_v1/graft_hypo.py` | HYPOTHESIS-ENUMERATION + PROBE RULE (`hypo: OK`), master flag `HYPO_ENABLE` |
| `submission/_throughput_v1/test_graft_evidence.py` | 14 tests (install, flag-off bytes, diff kinds, coordinates, trace, level flag, caps, fallback, analyze() end to end) |
| `submission/_throughput_v1/test_graft_hypo.py` | 7 tests (install, flag-off, once per call, absent after win, cap, fallback, analyze() end to end) |
| `submission/_throughput_v1/dry_run_evidence.py` | real-engine dry run (june_stock + pinned taaf + arcengine, mock brain replaying the sb26 win trace) |
| `offkaggle/run_regime_wave.py` (edited) | arms `keith_evid` / `keith_hypo` / `keith_up8`, `--draws N`, run-stem tagging, aid telemetry + engagement, `AID`/`DRAWS` summary lines |
| `offkaggle/test_run_regime_wave.py` (edited) | arm-difference invariant, flag-leak scrub, extractor, upscale probe, in-memory installs, `keith_evid --draws 2` dry run |
| `offkaggle/REGIME_WAVE_STATUS.md` (appended) | pointer section |

No stock byte was edited: the runner still asserts the june_stock tree sha `74ab691052406c22…`
after the in-memory install (probe test), and every dry run prints `stock agent sha 74ab69105240
(== june_stock pin)`.

## 1. What the evidence aid emits, and where

**Seam (verified on june_stock `inference/agent/tool_agent.py`).** An `action(...)` batch has one
path: `analyze()` -> `_dispatch_tool()` -> `_run_python_tool(state_path, arguments)` ->
`_ToolDispatchResult(content=<JSON string from _render_tool_payload>, step_executed)`. `analyze()`
appends `dispatch.content` verbatim as the `{"role": "tool"}` message the model reads, renders it
into the transcript with `_render_tool_result_display` (which shows the `stdout` field), and keeps
it in `_history_messages` for later turns. The graft wraps `ToolAgent._run_python_tool`: before the
stock call it notes `len(session.history_entries)` (the live `_HarnessGameSession` reached through
`self._step_env_callback.__self__`; fallback `load_runtime_state(state_path)`), after it diffs the
entries appended during the call, and appends the block to the payload's `stdout` (created when the
snippet printed nothing), re-dumped with the stock `indent=2`. `EVID_ENABLE=0` returns the stock
object untouched (test 02: byte-identical content).

**Frames and levels.** `solver._execute_action` appends one `HistoryEntry(action_display,
Frame(grid, step=action_count, level=_level_number(game)))` per executed action and
`_level_number = levels_completed + 1` (capped at `number_of_levels`), so a level clear inside a
batch is a strict rise of `Frame.level` between consecutive frames — and the frame returned by the
clearing action already shows the NEXT level. That is exactly the cd82 failure in the judge's
evidence: at action 28 (`MOUSE(row=4, col=43), SPACE`, `level_completed: True`) the model printed
rows 33-44 of `current_frame.ascii` — the level-2 board — and read it as level-1 evidence
(offkaggle/results/20260902T2121-keith/…/transcripts/cd82-fb555c5d_p0.txt, lines ~4120-4160).

**A real block from the real-engine dry run** (sb26, the harness replaying the recorded win trace;
`scratchpad/tp_dry_run/evid-evid-175509/transcripts/sb26-7fbdac44_p0.txt`, tool result at
transcript line ~2985). The clear happened on the 3rd action of the call, so the diff is split at
the boundary and the cross-level pair is never diffed:

```
[TOOL RESULT: python]
replay 54 1 lvl 4 cleared True

[EVID] harness object diff for actions 55-57 (3 executed in this call); coords are (row,col), 0-based: row = line index of `.ascii` from the top, col = char index from the left, the same row/col MOUSE takes
LEVEL CLEARED after action 3 (SPACE) — the frames after it belong to the NEXT level (level 4); do not diff them against this level.
level 3 diff, before action 1 -> after action 2: 33 cells changed, 6 object changes
  MOVED O/orange size 16: (57,9)-(60,12) -> (22,42)-(25,45) (d row -35, col +33)
  MOVED g/gray size 4: (23,43)-(24,44) -> (58,10)-(59,11) (d row +35, col -33)
  RESHAPED c/charcoal size 600->612 at (54,0)-(63,63) -> (54,0)-(63,63) [edge]
  RESHAPED c/charcoal size 188->176 at (20,16)-(27,47) -> (20,16)-(27,47)
  RESHAPED g/gray size 58->57 at (53,0)-(53,57) -> (53,0)-(53,56) [edge]
  RESHAPED G/dark gray size 6->7 at (53,58)-(53,63) -> (53,57)-(53,63) [edge]
level 4 start frame (after action 3): 41 non-background objects
TRACE per action: 1 MOUSE(row=58, col=10): 20 cells changed | 2 MOUSE(row=23, col=43): 53 cells changed | 3 SPACE: LEVEL CLEARED (frame now level 4)
```

A real single-action block from the runner dry run on one of the walls (dc22, `keith_evid` arm,
`…/regime_dry_0906/20260906-175822-regime-keith_evid-dry/transcripts/dc22-fdcac232_p0.txt`):

```
[EVID] harness object diff for action 1 (1 executed in this call); coords are (row,col), 0-based: row = line index of `.ascii` from the top, col = char index from the left, the same row/col MOUSE takes
diff, before action 1 -> after action 1: 9 cells changed, 4 object changes
  MOVED N/light green size 4: (40,10)-(41,11) -> (42,10)-(43,11) (d row +2, col +0)
  RESHAPED G/dark gray size 576->577 at (54,0)-(62,63) -> (54,0)-(63,63) [edge]
  RESHAPED W/white size 64->63 at (63,0)-(63,63) -> (63,1)-(63,63) [edge]
  RESHAPED g/gray size 32->32 at (38,8)-(43,13) -> (38,8)-(43,13)
```
(DOWN moved the 2x2 light-green agent two rows down; the two `[edge]` lines are the HUD bars.)

Entry kinds: `MOVED` (same color + shape at a new place, with the delta), `APPEARED`, `VANISHED`,
`RECOLORED` (identical cells, new color), `RESHAPED` (same color, overlapping cells, different
shape — HUD bars shrinking); `[edge]` = bbox touches the border. Components covering
>= `EVID_BG_FRACTION` [0.25] of the grid are background and never listed. Multi-action calls get the
`TRACE per action` line: the single component that moved after action k (tracked by color+size across
the batch), else the changed-cell count, `no change`, or `LEVEL CLEARED`. GAME_OVER / run-complete /
stopped-early notes come from the stock `last_action_result`.

Knobs (read at call time): `EVID_ENABLE` (master), `EVID_MAX_ENTRIES` [40], `EVID_MAX_CHARS` [1500,
whole block; header + LEVEL CLEARED flag + trace survive a cut], `EVID_TRACE` [1], `EVID_TRACE_MAX`
[12], `EVID_CONNECTIVITY` [4 = the sandbox's `segment_layer`, so `size` equals the `pixels` the
model sees; 8 available], `EVID_BG_FRACTION` [0.25]. Cost measured on the dry run: 57,431 chars over
96 executed tool results = 598 chars (~150-200 tokens) per executed turn, cap ~400 tokens.

`status()`: `diffs_emitted`, `level_flags`, `chars_added`, `traces_emitted`, `errors`, `skips`,
`per_game`. Any exception inside the wrapper returns the stock result (tests 10; counted in `errors`).

## 2. Coordinate convention — proof

* `inference/utils/grid_utils.py:format_grid_ascii` emits one line per grid row and one char per
  column; `runtime_state.Frame.ascii` is that string. So `.ascii.splitlines()[row][col]` is cell
  `grid[row][col]`. Test 04 builds a frame with one blue cell at (2,7), asserts
  `lines[2][7] == ARC_COLOR_CHARS[9]` and that the aid reports `APPEARED b/blue size 1 at (2,7)`.
* MOUSE: the prompt says "`row` is vertical position, `col` is horizontal position"
  (prompts.py:34); `solver._normalize_actions` clamps `row`/`col` and sends `{"x": column, "y": row}`
  to the engine (solver.py:528-534); `_model_mouse_action_data` maps back `row = y, col = x`
  (solver.py:121-125); `_format_action_display` renders `MOUSE(row=.., col=..)`. Test 04 asserts
  `_model_mouse_action_data({"x": 7, "y": 2}) == {"row": 2, "col": 7}`.
* The sandbox segmentation's `boundary` points are `[row, col]` (segmentation.py docstring;
  prompts.py:43). The aid uses 4-connectivity by default so its object sizes match those nodes.
* The block header states the convention every time.

## 3. The hypothesis rule (graft_hypo)

Wraps `ToolAgent._build_user_prompt`; appends the fixed block (`graft_hypo.HYPO_BLOCK`, 875 chars
<= 900) once per `analyze()` call while the run is playing (absent once `previous_step_summary.run_complete`,
the session's `game_run.state != "playing"`, or the engine reports `won`). The inline follow-up
prompts inside a turn are built by `analyze()` itself, so the block never repeats within a turn
(test 07 through `analyze()` with a text-only first reply). The turn after a level clear keeps it —
the new level is the next wall and the engagement gate counts wall turns.

```
[HYPO] Hypothesis discipline for this uncleared level (harness rule, every turn):
1. List >=3 candidate mechanics / goal predicates (what would make this level count as cleared), each consistent with EVERY transition observed on this level so far; with no transitions yet, derive them from the layout. For each, name one observation that would falsify it.
2. Pick the cheapest probe (<=3 actions) whose outcome differs between at least two candidates. Write the predicted outcome per candidate, then run that probe FIRST as one action([...]) batch, before any longer plan.
3. If the previous batch was expected to clear the level and did not, state which predicate that falsified, drop it, and re-derive the goal from the transitions; do not repeat the same plan with small variations.
4. Judge what changed from a computed before/after diff of the frames, never from memory.
```
`status()`: `blocks_injected`, `errors`, `skips`, `per_game`.

## 4. UPSCALE 8 (keith_up8)

The env key is `MULTIMODAL_UPSCALE`, read at call time by
`inference/agent/vision_context.py:current_grid_image_upscale()` (default 16; keith's env sets 4);
`frame_to_png_data_url` renders the 64x64 frame at 1 px/cell and resizes NEAREST to
`cols*scale x rows*scale`, attached as the `image_url` part of every user message
(`ToolAgent._build_user_message`). keith_up8 = keith + `MULTIMODAL_UPSCALE=8` and nothing else
(arm-difference test). The runner asserts `current_grid_image_upscale() == env` in `verify_imports`
and records `results.json:vision` = the PNG the stock code produces: keith `[256, 256]` px, keith_up8
`[512, 512]` px (upscale probe test + the dry-run summary `UPSCALE=8 ([512, 512] px, ~256 vision tok/img derived)`).

Vision tokens: 64 per image at upscale 4 is MEASURED — re-running the judge's
`scratchpad/judge_0906/recon_tokens.py` on the 25 real keith calls, real tokenizer + chat template
with 64 tokens per `<|image_pad|>`, matches vLLM's reported `prompt_tokens` at median ratio 0.986.
256 at upscale 8 is DERIVED from the same geometry ((512/32)^2 with 16-px patches x 2x2 merge); the
Qwen3.8 `preprocessor_config.json` is not in the local HF cache, so the patch/merge values and any
`max_pixels` cap at 512 px are not verified here — the first real call on the up8 arm will show it in
the shim's `prompt_tokens` (expect ~+192 per image).

## 5. Tests and dry runs (all run 2026-09-06; bundle = june_stock)

* `test_graft_evidence.py` — **14/14** (`GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference`):
  install/seam; flag-off byte-identical; MOVED/APPEARED/VANISHED/RECOLORED/RESHAPED with exact
  coordinates, background never listed, 4 vs 8 connectivity; coordinate convention (§2); batch trace
  incl. no-single-mover and `EVID_TRACE`/`EVID_TRACE_MAX`; level flag + split (clear first/middle/last,
  cross-level pair never diffed); entry and char caps (flag survives); wrapper injects into `stdout`
  and the transcript renderer shows it, counters; level flag + GAME_OVER note through the wrapper;
  exception fallback (both before and after the stock call); runtime-state fallback without a session;
  `inject()` shapes; status; `analyze()` end to end (marker once in the transcript's TOOL RESULT and
  in the kept history).
* `test_graft_hypo.py` — **7/7**: install; flag-off identical; once per prompt (turn 0, after a
  clear); absent after run_complete / not playing / won; <= 900 chars; fallback; `analyze()` end to end
  (follow-up prompt does not repeat it).
* `dry_run_evidence.py` (real arcengine, sb26 replay + vc33/ls20 cycling, 60 actions/game):
  `--graft both` **PASS 11/11**, `--graft evid` **PASS 10/10**, `--graft hypo` **PASS 6/6**.
  96 executed-action tool results -> 96 `[EVID]` blocks (== `diffs_emitted`), 3 LEVEL CLEARED flags on
  sb26 (== the three clears, all inside batches; levels 3/8 reached), 35 traces, 0 errors; 96 turns ->
  96 `[HYPO]` blocks (== `blocks_injected` == prompts).
* `offkaggle/test_run_regime_wave.py` — **23/23** (arm envs, pins, extractor incl. the
  new aid/engagement/level extraction on a synthetic transcript, identity gate, upscale probe
  (256 vs 512 px, 64 vs 256 derived), in-memory installs for keith_evid/keith_hypo (only the intended
  method rebound, sha holds, run-stem tag `cd82-fb555c5d_p1` for index 3 of 2 games), flight/keith_retry
  dry runs, and `keith_evid --games cd82,lf52 --draws 2` end to end).
* Runner dry runs, the exact shape asked for (`--games cd82,dc22,lf52 --draws 2 --per-game-s 60`),
  artifacts under `<scratchpad>/regime_dry_0906/`:
  * `keith_evid`: 6 runs (`cd82/dc22/lf52 _p0/_p1`), `[EVID] 350 (58.3/run)`, engagement wall
    100.0 % (350/350) — every executed turn on the wall level carried the aid; `DRAWS 2 | levels per
    (game, draw): {'cd82-fb555c5d': [0, 0], 'dc22-fdcac232': [0, 0], 'lf52-271a04aa': [0, 0]}`.
  * `keith_hypo`: `[HYPO] 361 (60.2/run)`, engagement wall 100.0 % (361/361).
  * `keith_up8`: `UPSCALE=8 ([512, 512] px, ~256 vision tok/img derived)`, no graft, no markers.
  (Levels are 0 because the mock brain cycles actions; the mock counts prompt tokens as chars/4, so
  its token numbers carry no information about the aids' cost.)

## 6. Runner additions (offkaggle/run_regime_wave.py)

* Arms: `keith_evid` = keith + `EVID_ENABLE=1 EVID_MAX_ENTRIES=40 EVID_MAX_CHARS=1500 EVID_TRACE=1`;
  `keith_hypo` = keith + `HYPO_ENABLE=1`; `keith_up8` = keith + `MULTIMODAL_UPSCALE=8`. Each differs
  from `keith` by exactly its own keys (test). `install_env` scrubs every `RETRY_*/EVID_*/HYPO_*`
  key from the shell for every arm. Grafts install in memory after `verify_imports` (identity gate
  untouched: `/arc3/identity` profile + RTX PRO 6000 still required, `--skip-preflight` still warns).
* `--draws N` (default 1) = taaf `Benchmark.n_passes`; taaf plays pass 0 then pass 1, the solver
  names runs `<gid>_p<draw>`; `game_rows` adds `draw` / `run_stem` (draw = index // n_games), the
  analyzer factory tags the game thread with the run stem so `requests_shim.jsonl` carries
  `run_stem`, and per-draw scores are read from the frozen scorer's `trial_scores[".../pass-k"]`.
  `--per-game-s N` is the per-run runtime cap (default unchanged, 7920 = public geometry).
* Telemetry per run: `evid_markers`, `evid_level_flags`, `hypo_markers`, `turn_levels` (from each
  turn's first `Current state: step N, level L` line), `level_reached`, `wall_level`
  (`levels_completed + 1`, None if won), `engagement` = {evid: aided executed turns / executed turns,
  hypo: aided turns / turns, `_wall` variants on the wall level}; `per_game_draw` = per-(game, draw)
  levels; summary `AID` / `DRAWS` lines and `evid` / `hypo` / `wall%` columns.

## 7. Launch commands (NOT run; token read from `~/.config/arc3/vllm_token` into memory only)

```
cd /Users/ahmed/Documents/ArcAGI3
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_evid --games cd82,dc22,lf52 --draws 2 --per-game-s 1500 \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 --out offkaggle/results
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_hypo --games cd82,dc22,lf52 --draws 2 --per-game-s 1500 \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 --out offkaggle/results
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_up8  --games cd82,dc22,lf52 --draws 2 --per-game-s 1500 \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 --out offkaggle/results
```
Read `summary.txt`: the `DRAWS` line gives levels per (game, draw) — the judge's rule counts a
wall-attempt as passed when a run reaches level 3 on these games (their modal wall is L2); the `AID`
line gives the engagement gate (>= 80 % of wall turns). Six runs at 1500 s with concurrency 28 run in
parallel: ~25-30 min wall per arm on the queue-free endpoint.

## 8. Unverified / limitations

1. **No game has been played against the real model.** Engagement is attested on the mock; the
   effect on cd82/dc22/lf52 L2 is the experiment.
2. **256 vision tokens at upscale 8 is derived, not measured** (§4); 64 at upscale 4 is measured.
3. The stock `step_env` stops a batch at `level_completed`, so within ONE `action([...])` call no
   frame follows a clear; frames after a clear inside one tool result arise only when the snippet
   calls `action()` again (the "misread batched logs" case). The flag fires in both cases (3/3 on the
   real engine); the split with post-clear frames is unit-tested (tests 06, 09) and the real-engine
   transcript shows the split form (old-level diff + next-level start frame).
4. The aid costs ~150-200 tokens per executed turn inside a 31,744-token budget the harness trims
   by dropping the oldest turns, so it shortens the carried history by roughly one turn in ten.
   `EVID_MAX_CHARS` is the lever.
5. HUD bars show up as `RESHAPED … [edge]` on nearly every action (the model's system prompt already
   teaches edge bars); background-sized components are suppressed by `EVID_BG_FRACTION`.
6. `RESHAPED` can list a same-size shape change (e.g. `size 32->32`) when a frame around a mover
   re-forms; it is informative but adds lines on busy games — the entry cap handles it.
7. HYPO engagement is 100 % by construction (a fixed block on every prompt); the gate is trivially
   met — the wave reads only whether the walls fall.
8. `wall_level` uses `levels_completed + 1` from the benchmark row; when a run is cancelled
   mid-turn the last turn may be missing its prompt-level line (counted as level None, excluded from
   the wall share).
9. The Kaggle notebook was not touched; these arms exist for the off-Kaggle rig only.
