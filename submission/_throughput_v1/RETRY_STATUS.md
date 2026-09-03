# RETRY — fresh-mind level retry graft (built 2026-09-03, NOT launched)

One harness-side experiment: when a level's action bucket runs past K x its
human baseline without clearing, the harness issues a level RESET through the
normal action path, blanks the carried level-scoped world-model note, and
hands the model one "FRESH MIND" prompt block that quotes the abandoned
hypothesis and asks for two alternatives tested cheaply.

Files (all new; no tracked file edited except the two offkaggle runner files):

| file | what |
|---|---|
| `submission/_throughput_v1/graft_retry.py` | the graft (`install()` -> `retry: OK`, `status()`) |
| `submission/_throughput_v1/test_graft_retry.py` | 17 unit tests (mock session, both bundles) |
| `submission/_throughput_v1/dry_run_retry.py` | real-engine dry run, `--mode retry-only` (june_stock) / `--mode all-packs` (graft stack) |
| `offkaggle/run_regime_wave.py` | arm `keith_retry`, `install_grafts`, `[RETRY]`/`[RETRY-CLEAR]` telemetry, summary line, `--knob` |
| `offkaggle/test_run_regime_wave.py` | arm-difference invariant, extractor, in-memory install probe, keith_retry dry run |

## 1. Trigger source — is a human baseline available at runtime? YES (offline)

Verified in the june_stock bundle + the sha-pinned taaf framework:

* `ToolAgent.analyze()` receives `step_env=session.step_env` every turn
  (solver.py:294-303); `step_env.__self__` is the live `_HarnessGameSession`
  (the same seam graft_control uses for its stagnation RESET).
* `session.game.game_run.actions_per_level[idx]` = actions accumulated in the
  level bucket (taaf game.py:566 increments it for every action, RESET and
  retries included); `idx = session.game.current_state.levels_completed`.
* `session.game.game_run.base_actions_per_level` = the per-level HUMAN
  BASELINES. taaf game_api.py:231 copies `arcengine environment_info.baseline_actions`
  at `start_game()`; offline over `environment_files` it is populated —
  checked live: tu93 `[19, 16, 34, 42, 123, 80, 14, 23, 111]`, vc33 L1 = 7,
  ls20 L1 = 22. It is `None` in submission mode (game_api.py:234-240), in
  which case the graft falls back to `RETRY_ABS` [200].
* `inference/utils/rearc_baselines.py` (re_arc metadata) is an eval-side
  helper, not used by the agent at runtime — not needed.
* `session.game.current_state.available_actions` lists RESET (id 0) when the
  engine allows it; `session._execute_action(RESET, batch_index=1,
  batch_size=1, generated_tokens=0)` is the exact call the harness's own
  GAME_OVER auto-reset uses (solver.py:663-665): it goes through
  `game.execute_action` (history + `actions_per_level` + invariants), writes
  the runtime state, and appends the viewer event.

Trigger rule (all env read at call time): fire when
`actions_on_level >= ceil(RETRY_K x baseline)` (K=3; `>= RETRY_ABS` when no
baseline), the level is not cleared, fewer than `RETRY_MAX` [2] retries fired
on it, and `actions_on_level - bucket_count_after_last_retry >= RETRY_COOLDOWN`
[150]. Guards: run state `playing`; engine not GAME_OVER/WIN (the harness
auto-reset is pending there and is never touched); last engine action not
already RESET; RESET in `available_actions`; `should_stop()` false. `RETRY_ENABLE=0`
= pure pass-through (asserted byte-identical, see §3).

## 2. The exact prompt block (<= 1,200 chars incl. a <= 600-char quote)

Appended once to the next user prompt after a retry (this one from the real-engine
dry run, vc33, `scratchpad/tp_dry_run/retry-115259/transcripts/vc33-5430563c_p0.txt`):

```
FRESH MIND (harness level retry 1/2): the harness RESET this level at action 8 because it was not cleared after 7 actions (human baseline ~7, limit 1x). The board is back at the start of level 1; the world model carried for this level was cleared on purpose. The hypothesis you were pursuing is now treated as FAILED:
<<<
Plan: cycle actions.
>>>
Do not resume it. State TWO alternative hypotheses that explain the transitions in `history` differently (another goal, another meaning of the objects, another effect of the actions), pick the cheapest to test, and test it with at most 5 actions in one `action([...])` batch before committing to anything longer.
```

Template (`graft_retry.FRESH_MIND_BLOCK`): `{budget}` is `human baseline ~B, limit Kx`
or `limit N actions` (abs fallback); `{quote}` = `World model: … | Goal model: … |
Action model: … | Plan: …` from the carried note, whitespace-collapsed, cut to 600
chars, or `(no world-model note was recorded)`. The block is hard-capped at
1,200 chars (test 09 checks the worst case). Alongside it the graft blanks
`world_model, goal_model, action_model, recent_findings, open_questions,
current_plan` (the same set stock wipes on a level transition,
tool_agent.py:1113-1126) and keeps `cross_level_notes`. Optional
`RETRY_CLEAR_HISTORY=1` [default 0] also drops `_history_messages`.

Budget: ~300 tokens on a 31,744-token context budget; the harness trims older
history with `preserve_recent=1`, so the latest user prompt (with the block) is
never cut. Dry-run prompt tokens/call: 19,121 mean.

Transcript markers, written with the harness's own
`tool_agent._append_transcript_section` under a `[HARNESS RETRY]` section:

```
[HARNESS RETRY]
[RETRY] game=vc33-5430563c level=1 actions=7 baseline=7 threshold=7 retry=1/2 action_num=8
[HARNESS RETRY]
[RETRY-CLEAR] game=<id> level=<n> actions=<bucket> retries=<k>
```

`status()` exposes `retries_fired`, `levels_cleared_after_retry`, `retry_log`
(game, level, actions, baseline, threshold, retry, action_num), `clear_log`, `skips`.

## 3. Test results (2026-09-03)

`test_graft_retry.py` — 17/17 on BOTH bundles
(`GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference` and the
default anim bundle):

* 01 install OK, seams `_retry_stock`
* 02 flag-off byte-identical: stock `analyze` vs wrapper with `RETRY_ENABLE=0`
  on a mock game 999 actions past the trigger — identical transcript bytes,
  identical prompts, no RESET, no state object
* 03 fires at exactly 3 x 20 = 60 (59 -> no, 60 -> yes; RESET lands in the bucket -> 61)
* 04 abs fallback (no baseline) at exactly `RETRY_ABS`; 05 fractional K / ceil
* 06 cooldown honoured (149 no / 150 yes), max 2 per level, skips counted
* 07 budget is per level, not per game
* 08 FRESH MIND present exactly once on the next prompt, absent after; stale
  note gone from the stock block; cross-level note kept
* 09 block <= 1,200 chars, quote <= 600 chars
* 10 suppression is level-scoped (later level's note untouched); 11 history clear knob
* 12 RESET through `_execute_action` (batch 1/1, generated_tokens=0), recorded
  in history, runtime state rewritten, stock analyze sees the post-RESET
  `action_num` and refreshed `valid_actions`
* 13 guards: GAME_OVER, last action RESET, RESET unavailable, should_stop,
  run not playing, RETRY_MAX=0, no session
* 14 `[RETRY-CLEAR]` once per retried level that clears, counted; unretried
  clears not counted
* 15 block re-armed when the request carrying it dies (retryable_failure, no reasoning)
* 16 game change resets state; 17 status shape

`dry_run_retry.py --mode retry-only` (june_stock + pinned taaf + REAL arcengine,
vc33/ls20/sb26, mock brain, K=1/ABS=15/cooldown 10/max 2) — PASS 11/11, 21 s:
6 retries (2 per game), baselines reachable, RESETs via `_execute_action` and in
`game_run.history`, `[RETRY]` count == retries == FRESH MIND prompts == 6,
`sum(actions_per_level) == len(history)`, no crash.

`dry_run_retry.py --mode all-packs` (graft_retry installed FIRST, then
`dry_run.py`'s Packs 1-10 stacked on top, anim bundle) — dry_run's own 13 checks
PASS and 4 retries fired under the full stack (no interaction regression).
`dry_run.py` itself was not edited (it has no graft parameter).

`offkaggle/test_run_regime_wave.py` — 17/17 (5 new/extended): arms differ from
`keith` by exactly `{RETRY_ENABLE, RETRY_K, RETRY_ABS, RETRY_COOLDOWN, RETRY_MAX}`;
stock arms carry no RETRY flag and `install_env` scrubs stale RETRY_* from the
shell; extractor counts markers (and ignores `[RETRY]` inside model text);
subprocess probe: `install_grafts("keith_retry")` rebinds `analyze`/`_build_user_prompt`
in memory while `assert_stock_tree()` still hashes to the june_stock pin, the
factory-built ToolAgent keeps the keith fingerprint (31,744 budget, server-default
max_output, 60 s yield); end-to-end `--dry-run --arm keith_retry`.

## 4. Dry-run proof of the runner arm

```
.venv/bin/python offkaggle/run_regime_wave.py --dry-run --arm keith_retry --games tu93,ft09,vc33 \
    --per-game-s 14 --wave-cap-s 60 --out <scratch> --knob RETRY_K=0.2 --knob RETRY_ABS=4 --knob RETRY_COOLDOWN=3
```
(`--knob` shrinks the thresholds so the trigger fires inside a 14-second mock
game; it is recorded in `results.json:knob_overrides` and flagged in the summary.)

```
REGIME WAVE  arm=keith_retry  status=done  dry_run=True  games=3
  stock agent sha 74ab69105240 (== june_stock pin) | framework f68b6850b242 | pkls pinned
  ARM KNOB  CONTEXT_WINDOW=32768  MAX_OUTPUT=0  (yield 60 s, ...); harness-reported {'max_output_tokens': 'server default', 'context_budget_tokens': '31744', 'yield_seconds': '60.0'}
  RETRY  fired 6 (2.00/game) | levels cleared after retry 0 | games with retry 3 | grafts {'graft_retry': 'retry: OK'} | flags {'RETRY_ENABLE': '1', 'RETRY_K': '0.2', 'RETRY_ABS': '4', 'RETRY_COOLDOWN': '3', 'RETRY_MAX': '2'}
  KNOB OVERRIDES (not the pinned arm env): {'RETRY_K': '0.2', 'RETRY_ABS': '4', 'RETRY_COOLDOWN': '3'}
  game            lv/n    act  calls turns reas_mean  len%  notool%  e2e_s   score  retry  state
  ft09-0d8bbf25    0/6     32    34    31      1179      -   11.8%    0.3    0.00    2/0  gave_up
  tu93-0768757b    0/9     31    35    30      1175      -   17.1%    0.3    0.00    2/0  gave_up
  vc33-5430563c    0/7     31    35    30      1373      -   14.3%    0.3    0.00    2/0  gave_up
```
`telemetry.json` carries `aggregate.retries_total / retries_per_game /
retry_clears_total / games_with_retry`, per-game `retries_fired / retry_clears`,
and `grafts.status.graft_retry` (the graft's own `retry_log`, `skips`).

## 5. Known limitations

1. **Mid-level RESET semantics on the hosted/competition API are unverified by
   this work.** The rig runs OFFLINE with `ONLY_RESET_LEVELS=true`, where RESET
   restarts the current level (verified). The 08-27 memory note says RESET is
   swallowed in competition mode for the post-WIN "new play" path
   (api.py:316-334); a mid-level RESET's live behaviour must be checked against
   the hosted API before any Kaggle use. This experiment is scoped to the
   off-Kaggle rig.
2. The graft cannot tell "wrong hypothesis" from "right hypothesis, slow
   execution": at K=3 a level whose human baseline is 7 fires at 21 actions.
   `RETRY_K` / `RETRY_ABS` are the levers; the retry costs 1 action plus the
   progress on the level.
3. The carried chat history is kept by default (`RETRY_CLEAR_HISTORY=0`), so the
   model still sees its own earlier reasoning; only the labelled note is
   blanked and the block asks it to abandon the hypothesis. Whether wiping
   history helps is a follow-up knob, not tested for effect.
4. The quote is the harness's labelled note, which is stale on ~89% of turns
   (loss ledger); when the model never wrote a labelled note the block says so.
5. The `[RETRY]` marker is written before the turn header (the check runs at
   the top of `analyze`), so it lands at the tail of the previous turn's
   section in the transcript; the extractor anchors on the line shape, not
   the section.
6. Not a stall detector: it fires on budget, not on repetition/novelty
   (TP2's stagnation RESET and TP9's livelock breaker cover those).
7. No effect measurement yet — zero games played against the model. The
   loss-ledger pre-registration for this class of lever applies: read
   levels/game and games with stuck-level actions < baseline against the
   stock keith draws (1.04-1.36 lv/game).

## 6. Launch command for the rig (NOT run)

```
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_retry \
    --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
    --games all --out offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_retry
```
Token: `~/.config/arc3/vllm_token` (read into memory only). Pinned arm env =
`keith` + `RETRY_ENABLE=1 RETRY_K=3 RETRY_ABS=200 RETRY_COOLDOWN=150 RETRY_MAX=2`;
do not pass `--knob` for the scored read.
