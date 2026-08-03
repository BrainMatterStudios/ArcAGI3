# duck-patched — base duck + two verified harness fixes

The plain duck (our 0.92–1.26 floor) with two in-memory patches applied at the notebook's
customization hook. Model, sampling parameters, concurrency, per-game budget, game list and
submission path are all **unchanged** — the only behavioural deltas are below.

## Patch 1 — ACTION7 round trip (bug fix)

`inference/agent/action_names.py:7-17` maps `ACTION1..6` + `RESET` only. The consequence,
executed against our own bundle:

```
ACTION5  -> model: SPACE   -> back to engine: ACTION5
ACTION6  -> model: MOUSE   -> back to engine: ACTION6
ACTION7  -> model: ACTION7 -> back to engine: None      <-- broken
```

`to_model_action` passes `ACTION7` through to the model (`.get(raw, raw)` default), but
`to_engine_action` returns `None` because `ACTION7` is neither a key nor a value of the map.
`solver.py:521-526` then rejects it: **"Unknown action at index N: 'ACTION7'"**. The model is
offered a legal action it can never execute.

**6 of the 25 public games use ACTION7** — ar25, bp35, lf52, sb26, sk48, su15 (grep
`environment_files/`). ACTION7 has existed in the engine since arc-agi 0.9.2, so this is a
long-standing bug, not version skew.

## Patch 2 — animation metadata

`taaf.game.GameState.animation_frames` (game.py:173) already exposes the intermediate frames of
a multi-frame action; only the final frame ever reaches the model. Patch 2a wraps
`_HarnessGameSession._execute_action` to attach a summary; patch 2b forwards it through
`ToolAgent._compact_action_result`, which is what the model actually sees.

Added fields, only when an action produced an animation (so single-frame actions cost nothing):

| field | meaning |
|---|---|
| `animation_frame_count` | number of intermediate frames |
| `animation_changed` | anything differed from the final frame |
| `animation_only_changed` | motion visible **only** mid-animation — something moved and returned or moved through. This is the signal the model cannot otherwise recover. |
| `animation_changed_cell_count` | distinct cells involved |
| `animation_changed_bbox` | `[x0, y0, x1, y1]` of that motion |

No images, no raw frames — roughly 20–40 tokens per animated action.

## Patch 12 — compaction-on-evict + LLM-free plan queue (`TAAF_COMPACT`, default ON)

Research 2026-08-02 idea 4 / plan B4. Two context-lifecycle repairs, one env switch
(`TAAF_COMPACT=0` disables both at call time):

- **12a compaction-on-evict.** The bundle silently drops oldest history on token-budget
  overflow (`_drop_oldest_history_block`) and at the 30-assistant-turn cap
  (`_keep_recent_history_turns`); both channels converge in `_persistent_history_messages`,
  which the patch wraps to diff what left persistent memory each turn. Every
  `TAAF_COMPACT_EVERY` (8) evicted messages, ONE bounded LLM call (prompt ≤
  `TAAF_COMPACT_PROMPT_CHARS`, reply ≤ `TAAF_COMPACT_RESPONSE_TOKENS`, timeout
  `TAAF_COMPACT_TIMEOUT_S`, ≤ `TAAF_COMPACT_MAX_CALLS` per game) folds them into a pinned
  store — `facts / action_effects / failed_hypotheses / open_questions` — injected into
  every later user prompt. `failed_hypotheses` merges mechanically (old ∪ new) so refuted
  ideas are not retried after eviction. Any failure falls back to the stock silent drop.
- **12b plan queue.** The model may end its text with `{"plan_queue": [...]}` (schema is
  appended to the system prompt). The harness drains it one action per analysis step with
  zero LLM calls through the solver's own `step_env` (its `board_changed` is already
  HUD-masked by patch 8). First violation — rejected action, GAME_OVER, unexpected level
  change, or a per-step `expect` note contradicted — aborts the remainder and the model
  gets a compact report in its next prompt. Aborted plans are never re-accepted verbatim.
  Cap `TAAF_COMPACT_QUEUE_MAX` (10) steps.

Diagnostics accumulate in `COMPACT_DIAGNOSTICS` (evictions_seen, compactions_done,
compaction_tokens, queue_plans, queue_steps_executed, queue_aborts, llm_calls_saved, …),
printed to the kernel log at exit. Tests: `test_compact_queue.py` (23, incl. a stub-brain
E2E through the real play loop and a subprocess apply-check against the scored bundle
bytes at `scratchpad/taaf_scored_ref`).

## Not patched — RESET restriction

The public fork this work draws on restricts RESET to GAME_OVER. **Our bundle already does the
equivalent, more thoroughly:** `solver._engine_action_names` strips RESET from the model's
choices unconditionally (solver.py:116-117), and `_execute_auto_reset()` fires automatically on
GAME_OVER (solver.py:278-282). `verify_reset_already_handled()` asserts both still hold, so if
upstream ever changes we find out in the log instead of silently regressing.

## Verification

```
.venv/bin/python -m pytest submission/_duck_patched/test_duck_patches.py -q     # 13 passed
.venv/bin/python submission/_duck_patched/build_duck_patched.py                 # rebuild notebook
```

The build script fails loudly if any anchor cell in the upstream notebook has moved. Every code
cell in the output was checked to compile, and the patch cell was executed against the real
harness:

```
[duck-patch] patch1 action7: OK
[duck-patch] patch2a animation-producer: OK
[duck-patch] patch2b animation-forward: OK
[duck-patch] patch3 reset: NOT NEEDED (RESET stripped from choices + auto-reset on GAME_OVER)
ACTION7 round trip: ACTION7
valid actions keep ACTION7: ['UP', 'MOUSE', 'ACTION7']
engine accepts it: ACTION7
```

## Push

Interactive kernels only get a P100, which cannot serve Qwen3.6-27B-FP8, so the vLLM setup and
the benchmark run are both gated on `TRUE_SUBMISSION` (the commit lands a dummy parquet). The
RTX 6000 flag must be passed on the CLI — `machine_shape` in the metadata is inert.

```
kaggle kernels push -p submission/_duck_patched --accelerator NvidiaRtxPro6000
```

## What is NOT claimed

The 1.47 public fork's score is *associated* with its patches, not proven caused by them. Patch 1
is a confirmed bug fix with a known blast radius (24% of public games); patch 2 is a plausible
but unmeasured improvement. Treat the pair as one experiment.
