# Pack 2 — Memory & Control Graft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the share of zero-level plays (43% on the Qwen3.8 wave) by making the loop keep its state, probe before it reasons, and notice when it is stuck — without relying on the model volunteering `World model:` lines.

**Architecture:** One graft module `submission/_throughput_v1/graft_control.py` (installed after `graft_throughput`, sharing its flag style) patches four seams: (A) the trimmer's cut event and the level-up note wipe → a harness-owned summary call; (B) `_HarnessGameSession.play` → a deterministic level-start probe whose table rides the user prompt; (C) `_HarnessGameSession._execute_action` + `ToolAgent._build_user_prompt` → HUD-aware frame hashing, a stagnation tracker with a directive tier and a RESET tier; (D) `_HarnessGameSession.step_env` → a no-effect streak halt inside one tool call; (E) `_execute_action` payload + `_compact_action_result` → a compact diff summary. Every behaviour has a `TP2_*` flag read at call time; `TP2_ENABLE=0` is a pass-through.

**Tech Stack:** as Pack 1 (Python 3.12, unittest, anim bundle at `submission/_inspect_replay/assets_build/ARC3-Inference`, `arcengine` from `.venv` for ActionInput/GameAction).

**Evidence:** R2 §§1-6 (OpenAI retained-state ×3; AERA forced-probe 0/5→5/5; AVO/Retrodict stagnation tiers; OPINE dead-signatures; Retrodict `[DIFF]`), R4 §1.2/§4 (no stagnation detector, notes wiped, 33/481 turns with a world model, 91–1,304 actions on L1 for zero-level plays).

---

## Flags

| flag | default | meaning |
|---|---|---|
| `TP2_ENABLE` | `1` | master switch |
| `TP2_SUMMARY` | `1` | harness-owned summary at trim cuts and level-ups (non-thinking call, `TP2_SUMMARY_MAX_TOKENS`=600) |
| `TP2_PROBE` | `1` | level-start probe at game start; `TP2_PROBE_CLICKS`=3 salient components; `TP2_PROBE_EACH_LEVEL`=0 |
| `TP2_STALL` | `1` | stagnation supervisor; `TP2_STALL_T1`=10 actions without a new masked-frame state → directive; `TP2_STALL_T2`=40 → harness RESET (max `TP2_STALL_RESETS_PER_LEVEL`=2) |
| `TP2_STREAK` | `1` | halt a tool call's batch after `TP2_STREAK_N`=3 consecutive no-effect actions |
| `TP2_DIFF` | `1` | diff summary on every action result |

## Shared primitives (Task 1)

```python
def grid_hash(grid, mask=None) -> str            # sha1 of the grid with masked cells zeroed
class HudMask:                                    # online: per-cell change counts over transitions
    def observe(self, before, after) -> None
    def mask(self) -> set[tuple[int,int]] | None  # cells changing in > 60% of >= 8 transitions
def diff_summary(before, after, mask=None) -> dict  # changed, changed_ex_hud, bbox, colors_added, colors_removed
def components(grid, background=None) -> list[dict] # 4-connected same-colour components: color, cells, bbox, area
def salient_clicks(grid, k) -> list[tuple[int,int]] # centers of small, rare-colour, non-background components
```

## Tasks

### Task 1: primitives + tests
Files: `submission/_throughput_v1/graft_control.py`, `submission/_throughput_v1/test_graft_control.py`.
Tests: hash stable/changes; HudMask flags a cell that changes every transition after 8 obs and not a static cell; diff_summary counts/bbox/colours on a 4x4 toy grid; components on a toy grid (two blobs + background); salient_clicks prefers the small rare blob and returns row/col inside it.

### Task 2: (E) diff summary
Seams: wrap `_HarnessGameSession._execute_action` — read `previous_grid` before, `new grid` after (via `_grid_from_state(self.game.current_state)`), attach `payload["diff"]`; feed `HudMask.observe` (stored on the session as `_tp2_hud`); wrap `ToolAgent._compact_action_result` to copy `diff` through.
Tests: fake session with a `game` stub whose `execute_action` swaps grids; assert `payload["diff"]["changed"]` and that compact carries it.

### Task 3: (D) no-effect streak halt
Seam: wrap `_HarnessGameSession.step_env`; count consecutive executed results with `board_changed False and frame_count <= 1` (use `diff.changed_ex_hud == 0` when available); when the streak reaches N inside one tool call, return `_error_payload("no_effect_streak: ...")` for further calls until `begin_tool_call` resets it (hook: wrap `ToolAgent._run_python_tool` to reset `session._tp2_streak = 0` via `self._step_env_callback.__self__`).
Tests: fake session; three no-change payloads then the fourth call returns executed False with the streak error; reset clears.

### Task 4: (C) stagnation supervisor
State on the session: `_tp2_seen` (set of masked hashes), `_tp2_since_new` (actions since a new state), `_tp2_since_level` (actions since the level changed), `_tp2_resets_this_level`. Updated in the `_execute_action` wrapper. Directive injection: wrap `ToolAgent._build_user_prompt` to append a `STAGNATION` block when `since_new >= T1` (text lists the count and the untested-action guidance). RESET tier: in the `play` loop we cannot easily interpose per turn, so do it in the `_build_user_prompt` wrapper's caller: wrap `ToolAgent.analyze` — before delegating, if the session says `since_new >= T2` and `resets_this_level < cap`, call `session._execute_action(RESET)` (level reset; `ONLY_RESET_LEVELS=true` at eval) and set `_tp2_since_new = 0`, `_tp2_last_reset_note` for the prompt. The agent reaches its session through `step_env.__self__` (the bound callback passed into analyze).
Tests: fake session + fake analyzer; simulate N identical hashes → directive present in prompt; at T2 a RESET is executed once and the counter clears.

### Task 5: (B) level-start probe
Seam: wrap `_HarnessGameSession.play`: before delegating, if the game is playing and `action_count == 0`, run the probe: for each of ACTION1..ACTION5 present in `self.game.current_state.available_actions`, execute once and record `diff`; if ACTION6 present, click `salient_clicks(grid, k)`; stop early on level_completed/game_over (auto-reset handled by the stock loop). Store a table text on `self.analyzer._tp2_probe = {"level": L, "text": ...}`. `_build_user_prompt` wrapper appends the text while `current_level == L`.
Tests: fake session whose `_execute_action` records calls and returns synthetic diffs; assert action order, click count, table text content and that it disappears after a level change.

### Task 6: (A) harness-owned summary
Seams: (1) in `graft_throughput.trim`, expose a hook `graft_throughput.ON_CUT = None` called with `(self, dropped_messages)` when a cut happens (Pack 1 change: 3 lines, tested); (2) Pack 2 sets `ON_CUT` to summarise: build a plain transcript of the dropped assistant text + tool results (cap 14k chars), POST a non-thinking completion (`chat_template_kwargs {"enable_thinking": false}`, `max_tokens` 600, temperature 0.2) with a fixed summary system prompt asking for exactly the seven labelled lines; parse with `_extract_scientist_note` and merge into `_summarized_knowledge` (existing carry channel). (3) wrap `_update_summarized_knowledge_from_step_summary`: on `level_transition`, summarise the whole current history with the level-boundary prompt into `cross_level_notes` + `action_model`, then wipe the level-specific keys.
Tests: mock `requests.post` (as in `test_effort_medium`) returning a canned summary; assert the merge and that a failed request leaves notes untouched (fail-open).

### Task 7: smoke builder generalisation + Pack 2 A/B
Modify `submission/_tp_smoke/build_tp_smoke.py` to accept `--arm tp2` producing kernel `arc3-tp2-smoke` with phases `("tp", TP_ENABLE=1, TP2_ENABLE=0)` vs `("tp2", TP_ENABLE=1, TP2_ENABLE=1)`, both grafts embedded. Read: zero-level games (tp2 ≤ tp − 3 of 25) and mean levels (≥ tp) as PASS; levels < tp − 0.15 FAIL.

### Task 8: flight arm `arc3-duck38-tp2` (after PASS) — same shape as `_duck38_tp1` with both grafts.
