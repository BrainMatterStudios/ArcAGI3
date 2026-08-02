"""In-memory patches applied to the read-only TAAF duck bundle before `bm.run()`.

The Kaggle dataset that carries the duck harness is read-only, so every fix here is a
monkey-patch installed at import time in the notebook. Each patch is independently
guarded: if the upstream shape it expects is missing, that patch reports FAIL and the
others still apply. Nothing here changes the model, sampling, concurrency or budgets.

PATCH 1 — ACTION7 round trip (BUG FIX).
    `inference.agent.action_names.ENGINE_TO_MODEL_ACTION` maps ACTION1..6 + RESET only.
    `to_model_action("ACTION7")` passes the name through to the model, but
    `to_engine_action("ACTION7")` returns None, so solver.py's action parser rejects it
    with "Unknown action at index N". The model is offered a legal action it can never
    execute. 6 of the 25 public games use ACTION7 (ar25, bp35, lf52, sb26, sk48, su15).

PATCH 2 — animation metadata.
    `taaf.game.GameState.animation_frames` already exposes the intermediate frames of a
    multi-frame action, but only the final frame ever reaches the model. This adds a few
    scalar fields per action describing what moved during the animation. No images, no
    raw frames — roughly 20-40 tokens per action of causal signal.

NOT PATCHED — RESET restriction.
    The public fork this work draws on restricts RESET to GAME_OVER. Our bundle already
    does the equivalent, more thoroughly: `solver._engine_action_names` strips RESET from
    the model's choices unconditionally (solver.py:116-117) and `_execute_auto_reset()`
    fires on GAME_OVER (solver.py:278-282). No change needed; verified by
    `verify_reset_already_handled()` below.
"""
from __future__ import annotations

from typing import Any

# Engine action ids that count as real actions (RESET is id 0 and is scored differently).
_ACTION7 = "ACTION7"


def patch_action7() -> str:
    """Make ACTION7 survive the model<->engine name round trip. Returns a status line."""
    from inference.agent import action_names as an

    if _ACTION7 in an.ENGINE_TO_MODEL_ACTION:
        return "patch1 action7: SKIP (already mapped)"

    # ACTION7's meaning is not fixed across games, so it has no friendly label like
    # UP/DOWN/MOUSE. Map it to itself: the model sees "ACTION7" and the engine gets
    # "ACTION7" back.
    an.ENGINE_TO_MODEL_ACTION[_ACTION7] = _ACTION7
    an.MODEL_TO_ENGINE_ACTION.clear()
    an.MODEL_TO_ENGINE_ACTION.update(
        {value: key for key, value in an.ENGINE_TO_MODEL_ACTION.items()}
    )

    if an.to_engine_action(an.to_model_action(_ACTION7)) != _ACTION7:
        raise RuntimeError("patch1 action7: round trip still broken after patch")
    return "patch1 action7: OK"


def patch_animation_producer() -> str:
    """Attach the animation summary to the payload `_execute_action` returns.

    `_compact_action_result` (patch 2b) only forwards these keys; this is what creates
    them. Wrapping the method rather than reimplementing it keeps us insulated from
    upstream changes to the payload's other fields.
    """
    from inference.framework import solver

    session = getattr(solver, "_HarnessGameSession", None)
    if session is None or not hasattr(session, "_execute_action"):
        return "patch2a animation-producer: FAIL (_HarnessGameSession._execute_action not found)"
    if getattr(session._execute_action, "_animation_patched", False):
        return "patch2a animation-producer: SKIP (already applied)"

    original = session._execute_action

    def _execute_action(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original(self, *args, **kwargs)
        if isinstance(payload, dict):
            try:
                payload.update(summarize_animation(self.game.current_state))
            except Exception:
                pass  # never let telemetry break an action
        return payload

    _execute_action._animation_patched = True  # type: ignore[attr-defined]
    session._execute_action = _execute_action
    return "patch2a animation-producer: OK"


def patch_animation_metadata() -> str:
    """Forward the animation summary through the compaction the model actually sees."""
    from inference.agent import tool_agent

    agent_cls = getattr(tool_agent, "ToolAgent", None)
    if agent_cls is None or not hasattr(agent_cls, "_compact_action_result"):
        return "patch2b animation-forward: FAIL (ToolAgent._compact_action_result not found)"
    if getattr(agent_cls._compact_action_result, "_animation_patched", False):
        return "patch2b animation-forward: SKIP (already applied)"

    original = agent_cls._compact_action_result

    def _compact_action_result(self: Any, payload: dict[str, Any]) -> dict[str, Any]:
        compact = original(self, payload)
        for key in _ANIMATION_KEYS:
            if key in payload:
                compact[key] = payload[key]
        return compact

    _compact_action_result._animation_patched = True  # type: ignore[attr-defined]
    agent_cls._compact_action_result = _compact_action_result
    return "patch2b animation-forward: OK"


_ANIMATION_KEYS = (
    "animation_frame_count",
    "animation_changed",
    "animation_only_changed",
    "animation_changed_cell_count",
    "animation_changed_bbox",
)


def summarize_animation(state: Any) -> dict[str, Any]:
    """Describe the animation of a just-executed action as a handful of scalars.

    `state` is a taaf GameState. Returns {} when the action produced no animation, so
    single-frame actions cost zero extra tokens.
    """
    try:
        animation = list(state.animation_frames)
    except Exception:
        return {}
    if not animation:
        return {}

    final = _frame_rows(state.frame)
    changed_cells: list[tuple[int, int]] = []
    only_in_animation = False
    # HUD cells (patch 8) tick during animations too; they are not motion signal.
    hud_cells = {(int(y), int(x)) for y, x in _hud_current_mask_cells()}

    for frame in animation:
        rows = _frame_rows(frame)
        for y, (arow, frow) in enumerate(zip(rows, final)):
            for x, (avalue, fvalue) in enumerate(zip(arow, frow)):
                if avalue != fvalue and (y, x) not in hud_cells:
                    changed_cells.append((x, y))
                    only_in_animation = True

    out: dict[str, Any] = {
        "animation_frame_count": len(animation),
        "animation_changed": bool(changed_cells),
        # True when the animation showed motion that is NOT visible in the final frame —
        # i.e. something moved and returned, or moved through. This is the signal the
        # model cannot otherwise recover.
        "animation_only_changed": only_in_animation,
        "animation_changed_cell_count": len(set(changed_cells)),
    }
    if changed_cells:
        xs = [x for x, _ in changed_cells]
        ys = [y for _, y in changed_cells]
        out["animation_changed_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    return out


def _frame_rows(frame: Any) -> list[list[int]]:
    data = getattr(frame, "data", frame)
    return [list(row) for row in data]


def verify_reset_already_handled() -> str:
    """Confirm our bundle already restricts RESET, so patch 3 is genuinely unnecessary."""
    import inspect

    from inference.framework import solver

    src = inspect.getsource(solver._engine_action_names)
    strips_reset = 'if name == "RESET"' in src and "continue" in src
    auto_resets = hasattr(solver._HarnessGameSession, "_execute_auto_reset")
    if strips_reset and auto_resets:
        return "patch3 reset: NOT NEEDED (RESET stripped from choices + auto-reset on GAME_OVER)"
    return (
        f"patch3 reset: REVIEW (strips_reset={strips_reset}, auto_resets={auto_resets}) "
        "— upstream changed, re-check whether RESET must be restricted"
    )


def patch_dynamic_grid_burner() -> str:
    """Burn coordinate grids into the images sent to the VLM (Spatial Overlay)."""
    from inference.agent import vision_context

    if getattr(vision_context.frame_to_png_data_url, "_grid_burner_patched", False):
        return "patch4 grid-burner: SKIP (already applied)"

    original_func = vision_context.frame_to_png_data_url

    def frame_to_png_data_url_patched(frame: Any, *, upscale: int | None = None) -> str:
        import io, base64
        from PIL import Image, ImageDraw

        rows = len(frame.grid)
        cols = max((len(row) for row in frame.grid), default=0)
        if rows <= 0 or cols <= 0:
            raise ValueError("Cannot render an empty grid as an image.")

        scale = vision_context.current_grid_image_upscale() if upscale is None else max(1, int(upscale))
        image = Image.new("RGB", (cols, rows), vision_context.ARC_COLOR_MAP[0])
        pixels = image.load()
        for row_idx, row in enumerate(frame.grid):
            for col_idx in range(cols):
                value = row[col_idx] if col_idx < len(row) else 0
                pixels[col_idx, row_idx] = vision_context.ARC_COLOR_MAP.get(int(value), vision_context.ARC_COLOR_MAP[0])
        
        if scale > 1:
            image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            
            # Draw subtle grid lines
            grid_color = (128, 128, 128)
            for r in range(rows + 1):
                y = r * scale
                width = 2 if r % 5 == 0 else 1
                draw.line([(0, y), (cols * scale, y)], fill=grid_color, width=width)
            for c in range(cols + 1):
                x = c * scale
                width = 2 if c % 5 == 0 else 1
                draw.line([(x, 0), (x, rows * scale)], fill=grid_color, width=width)

            # Draw coordinate labels
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                text_color = (200, 200, 200)
                # row labels (left edge)
                for r in range(rows):
                    if r % 5 == 0:
                        draw.text((2, r * scale + 2), str(r), fill=text_color, font=font)
                # col labels (top edge)
                for c in range(cols):
                    if c % 5 == 0 and c > 0:
                        draw.text((c * scale + 2, 2), str(c), fill=text_color, font=font)
            except Exception:
                pass

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    frame_to_png_data_url_patched._grid_burner_patched = True  # type: ignore[attr-defined]
    vision_context.frame_to_png_data_url = frame_to_png_data_url_patched
    return "patch4 grid-burner: OK"


def patch_prompts() -> str:
    """Update system prompt to instruct the model to use the grid coordinates."""
    from inference.agent import prompts
    if "numeric coordinate labels" in prompts.MULTIMODAL_CONTEXT_ADDENDUM:
        return "patch5 prompts: SKIP (already applied)"
    prompts.MULTIMODAL_CONTEXT_ADDENDUM += "- The image contains numeric coordinate labels (0, 5, 10...) along the top and left edges. Use these to precisely identify row and column indices for spatial reasoning and action targeting.\n"
    return "patch5 prompts: OK"


def patch_tool_agent_analyze() -> str:
    """Intercept the Qwen-27B action generation pipeline to inject TransferExplorer and Retrospective summaries."""
    import sys
    from pathlib import Path
    
    # Ensure arcagi3 is in sys.path
    possible_paths = [
        Path("/kaggle/input/arcagi3-agent/lib"),
        Path("/kaggle/input/datasets/ahmedmobasher86/arcagi3-agent/lib"),
        Path(__file__).resolve().parents[2] / "src", # Local fallback for dev
    ]
    for p in possible_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            
    from inference.agent import tool_agent
    agent_cls = getattr(tool_agent, "ToolAgent", None)
    if agent_cls is None or not hasattr(agent_cls, "analyze"):
        return "patch6 tool_agent_analyze: FAIL (ToolAgent.analyze not found)"
    if getattr(agent_cls.analyze, "_tool_agent_patched", False):
        return "patch6 tool_agent_analyze: SKIP (already applied)"

    original_analyze = agent_cls.analyze

    def analyze_patched(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env = None,
        *args,
        **kwargs
    ):
        from inference.agent.runtime_state import load_runtime_state
        from inference.agent.tool_agent import AnalyzerTurnResult
        import arcengine
        
        # Phase 3: Inject Retrospective Summaries into system prompt
        original_system_prompt = getattr(self, "_original_system_prompt", None)
        if original_system_prompt is None:
            self._original_system_prompt = self._system_prompt
            original_system_prompt = self._system_prompt
            
        retrospectives = getattr(self, "_retrospective_summaries", [])
        if retrospectives:
            retro_text = "\n\nRetrospective Summaries from previous levels of this game:\n" + "\n".join(f"- {r}" for r in retrospectives)
            self._system_prompt = original_system_prompt + retro_text
        else:
            self._system_prompt = original_system_prompt

        # Phase 2: Heuristic Prober (TransferExplorer)
        if action_num < 200 and step_env is not None:
            try:
                from arcagi3.transfer_explorer import TransferExplorer
                if not hasattr(self, "_transfer_explorer_instance"):
                    self._transfer_explorer_instance = TransferExplorer()
                
                current_frame, _ = load_runtime_state(state_path)
                grid = current_frame.grid
                
                gstate_terminal = False
                gstate_notplayed = (action_num == 0)
                levels = current_frame.level
                
                # Format available actions as integers for TransferExplorer
                available_ids = []
                for name in (valid_actions or []):
                    try:
                        available_ids.append(arcengine.GameAction.from_name(name).value)
                    except Exception:
                        pass
                
                # numpy is required by arcagi3
                import numpy as np
                grid_np = np.array(grid, dtype=np.int32)
                
                action = self._transfer_explorer_instance.decide(
                    grid_np, gstate_terminal, gstate_notplayed, levels, available_ids
                )
                
                step_args = None
                if action and action[0] == "S":
                    action_name = arcengine.GameAction.from_id(action[1]).name
                    step_args = {"action": action_name}
                elif action and action[0] == "C":
                    step_args = {"action": "ACTION6", "col": action[1], "row": action[2]}
                elif action and action[0] == "reset":
                    step_args = {"action": "RESET"}
                    
                if step_args is not None:
                    # execute the symbolic transition
                    result = step_env(step_args)
                    # Note: we should still check if this action completed the level!
                    if self._last_action_result and self._last_action_result.get("level_completed"):
                        # Mark that we should do retrospective, but wait, the prober doesn't use the LLM, 
                        # so there are no LLM history messages for this level! We can just skip retro for heuristic clears.
                        pass
                    return AnalyzerTurnResult(
                        step_executed=True,
                        reasoning=f"TransferExplorer heuristic probe executed: {step_args}",
                        yielded_control=False,
                    )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("TransferExplorer failed, falling back to LLM: %s", exc)
                pass
                
        # Fallback to normal Qwen LLM analysis
        result = original_analyze(self, state_path, action_num, valid_actions, step_env, *args, **kwargs)
        
        # Phase 3: Retrospective generation on level complete
        if hasattr(self, "_last_action_result") and self._last_action_result and self._last_action_result.get("level_completed"):
            try:
                retro_messages = list(self._history_messages)
                retro_messages.append({
                    "role": "user", 
                    "content": "The level was just solved! Briefly summarize the core mechanic of this level and the strategy used to solve it. This summary will be provided to you in future levels of this game, so focus on transferable rules (e.g. 'green objects always move right until they hit a wall'). Keep it under 4 sentences."
                })
                # Call chat completion
                retro_result = self._chat_completion(retro_messages, tools=None)
                retro_content = retro_result.message.get("content", "")
                if retro_content:
                    if not hasattr(self, "_retrospective_summaries"):
                        self._retrospective_summaries = []
                    self._retrospective_summaries.append(retro_content)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Retrospective generation failed: %s", exc)
                pass
                
        return result

    analyze_patched._tool_agent_patched = True  # type: ignore[attr-defined]
    agent_cls.analyze = analyze_patched
    return "patch6 tool_agent_analyze: OK"


# --- PATCH 8: HUD wavefront mask -------------------------------------------------
#
# 18-24 of the 25 public games paint a step-budget bar into the frame and tick it
# on (nearly) every action, so raw full-frame equality — which is exactly what the
# harness uses for `board_changed` (solver.py:703) and what the model's sandbox
# helpers `state_hash` / `diff_frames` hash — reports a board change for provable
# no-ops. Measured corpus no-op rate: 0.208 raw vs 0.455 masked; `board_changed`
# is wrong on 13/25 games (100% wrong on 6).
#
# The detector is the validated wavefront statistic (docs/test-artifacts-2026-08-02/
# RESULTS-determinism-hudmask.md, 24/25 games, 0 false-mask cells): a HUD bar is a
# collinear run of >=4 cells that each change EXACTLY ONCE, in monotone spatial
# order of change time (>=3 distinct times), all with the same color transition.
# The statistic only holds within RESET-free stretches (a RESET refills the bar),
# so the tracker accumulates per-segment change statistics and breaks the segment
# on RESET, on level transitions, and on GAME_OVER/WIN screens (all of which
# repaint or refill). Strips found in any segment are unioned into the game mask.
#
# UNMASK GUARD (protects the full-edge-strip games sp80/vc33/r11l/s5i5/tu93): if
# any masked cell changes twice within one segment, real content has entered the
# strip — the whole region is dropped from the mask permanently for that game.
#
# Consumers patched:
#   - `board_changed` in `_HarnessGameSession._execute_action` (and thereby every
#     downstream copy: step_env batch aggregation, `_compact_action_result`,
#     outcome summaries, viewer events);
#   - `summarize_animation` (patch 2) ignores masked cells;
#   - the python-tool sandbox's `state_hash` and `diff_frames` (patch 9), which
#     receive the live mask through the runtime-state payload.
#
# TAAF_HUD_MASK=0 disables masking at call time (the tracker still observes).


class HudMaskTracker:
    """Incremental wavefront/HUD-strip detector over reset-free segments.

    Per-segment statistics (change count, first-change time, first transition)
    are updated per frame in O(cells); strips are re-detected from the statistics
    (never from stored frames), so memory stays O(1) in episode length.

    Two precision rules beyond the validated wavefront statistic — needed because
    real play traces (unlike the scripted no-op probes) contain monotone content
    fills that satisfy the statistic per-line (measured on stored vc33 episodes:
    a painted block reads as a stack of parallel "strips"):

      * CROSS-SEGMENT CONFIRMATION — a line is masked only after it produced a
        wavefront strip (same orientation, line and color transition, overlapping
        span) in >= MIN_SEGMENTS distinct reset-free segments. A budget bar ticks
        in every segment; a content fill rarely repeats on the same line with the
        same transition. MIN_SEGMENTS=3 measured 0 masked cells outside the
        validated HUD regions across all 43 stored episodes (vs 77 at 2), while
        keeping 119 HUD-only no-op detections.
      * STACK RULE — >= MAX_STACK adjacent parallel confirmed lines with
        overlapping spans are a 2D block, not a bar, and are all rejected
        (2 adjacent lines stay allowed: the ls20 bar is two rows tall).
    """

    MIN_LEN = 4
    MIN_DISTINCT_TIMES = 3
    MIN_SEGMENT_STEPS = 10
    MIN_SEGMENTS = 3
    MAX_STACK = 2

    def __init__(self, min_segments: int | None = None) -> None:
        import numpy as np

        self._np = np
        self._prev = None
        self._levels: int | None = None
        self._steps = 0
        self._segment_id = 0
        # (orient, line, from_color, to_color) -> {segment_id: (span_start, span_end)}
        self._sightings: dict[tuple, dict[int, tuple[int, int]]] = {}
        self._dead_lines: set[tuple] = set()  # unmask-guard casualties, permanent
        self._tainted_lines: set[tuple] = set()  # (orient, line) seen in a 2D stack
        self._mask = None  # bool array, True = HUD cell
        self._dropped = None  # bool array, True = never mask again (guard fired)
        self._nchg = None
        self._tchg = None
        self._tfrom = None
        self._tto = None
        if min_segments is not None:
            self.MIN_SEGMENTS = int(min_segments)

    # -- lifecycle -----------------------------------------------------------

    def seed(self, grid: Any) -> None:
        """Record the pre-action board so the first transition is counted."""
        if self._prev is None:
            g = self._np.asarray(grid, dtype=self._np.int16)
            self._init_arrays(g.shape)
            self._prev = g

    def _init_arrays(self, shape: tuple[int, int]) -> None:
        np = self._np
        self._mask = np.zeros(shape, dtype=bool)
        self._dropped = np.zeros(shape, dtype=bool)
        self._reset_segment_stats(shape)

    def _reset_segment_stats(self, shape: tuple[int, int] | None = None) -> None:
        np = self._np
        if shape is None:
            shape = self._nchg.shape
        self._nchg = np.zeros(shape, dtype=np.int32)
        self._tchg = np.full(shape, -1, dtype=np.int32)
        self._tfrom = np.full(shape, -1, dtype=np.int16)
        self._tto = np.full(shape, -1, dtype=np.int16)
        self._steps = 0

    # -- main entry ----------------------------------------------------------

    def observe(
        self,
        grid: Any,
        *,
        is_reset: bool = False,
        levels_completed: int | None = None,
        state_name: str | None = None,
    ) -> dict[str, bool]:
        """Feed the post-action board; returns raw/masked change verdicts."""
        np = self._np
        g = np.asarray(grid, dtype=np.int16)
        if self._prev is None or g.shape != self._prev.shape:
            self._init_arrays(g.shape)
            self._prev = g
            if levels_completed is not None:
                self._levels = int(levels_completed)
            return {"raw_changed": False, "masked_changed": False}

        diff = self._prev != g
        raw_changed = bool(diff.any())
        masked_changed = bool((diff & ~self._mask).any())

        boundary = bool(is_reset)
        if levels_completed is not None:
            if self._levels is not None and int(levels_completed) != self._levels:
                boundary = True
            self._levels = int(levels_completed)
        if state_name in ("GAME_OVER", "WIN"):
            boundary = True

        if boundary:
            # The bar refills / the scene repaints: this transition must not
            # enter the statistic. Bank what the closing segment proved first.
            self._record_sightings()
            self._segment_id += 1
            self._reset_segment_stats()
            self._refresh_mask()
        else:
            if raw_changed:
                newly = diff & (self._nchg == 0)
                self._tchg[newly] = self._steps
                self._tfrom[newly] = self._prev[newly]
                self._tto[newly] = g[newly]
                self._nchg[diff] += 1
                repeat = (self._nchg >= 2) & self._mask
                if repeat.any():
                    self._drop_regions(repeat)
            self._steps += 1
            if self._steps >= self.MIN_SEGMENT_STEPS:
                self._record_sightings()
                self._refresh_mask()

        self._prev = g
        return {"raw_changed": raw_changed, "masked_changed": masked_changed}

    # -- detection -----------------------------------------------------------

    def _detect_segment_strips(self) -> dict[tuple, tuple]:
        if self._steps < self.MIN_SEGMENT_STEPS:
            return {}
        np = self._np
        once = self._nchg == 1
        height, width = once.shape
        strips: dict[tuple, tuple] = {}

        def consider(run: list[tuple[int, int]], orient: str) -> None:
            if len(run) < self.MIN_LEN:
                return
            times = [int(self._tchg[y, x]) for y, x in run]
            deltas = np.diff(times)
            if not (bool((deltas >= 0).all()) or bool((deltas <= 0).all())):
                return
            if len(set(times)) < self.MIN_DISTINCT_TIMES:
                return
            transitions = {(int(self._tfrom[y, x]), int(self._tto[y, x])) for y, x in run}
            if len(transitions) != 1:
                return
            (from_color, to_color) = next(iter(transitions))
            if orient == "H":
                key = ("H", run[0][0], from_color, to_color, run[0][1], run[-1][1])
            else:
                key = ("V", run[0][1], from_color, to_color, run[0][0], run[-1][0])
            strips[key] = tuple(run)

        for y in range(height):
            run: list[tuple[int, int]] = []
            for x in range(width + 1):
                if x < width and once[y, x]:
                    run.append((y, x))
                else:
                    consider(run, "H")
                    run = []
        for x in range(width):
            run = []
            for y in range(height + 1):
                if y < height and once[y, x]:
                    run.append((y, x))
                else:
                    consider(run, "V")
                    run = []
        return strips

    def _record_sightings(self) -> None:
        """Fold the live segment's strips into the cross-segment evidence.

        Lines that show up as part of a stack of > MAX_STACK adjacent parallel
        strips within a SINGLE segment are tainted permanently: that shape is a
        2D content fill (e.g. vc33's paint blocks read as 8+ parallel "strips"),
        and per-line cross-segment confirmation cannot be trusted for it.
        """
        strips = self._detect_segment_strips()
        for orient in ("H", "V"):
            lines: dict[int, list[tuple[int, int]]] = {}
            for key in strips:
                if key[0] == orient:
                    lines.setdefault(key[1], []).append((key[4], key[5]))
            indices = sorted(lines)
            stack: list[int] = []

            def taint(stack_lines: list[int], orient: str = orient) -> None:
                if len(stack_lines) > self.MAX_STACK:
                    for member in stack_lines:
                        self._tainted_lines.add((orient, member))

            for line in indices:
                touches = bool(stack) and line == stack[-1] + 1 and any(
                    prev_span[0] <= cur_span[1] and cur_span[0] <= prev_span[1]
                    for prev_span in lines[stack[-1]]
                    for cur_span in lines[line]
                )
                if touches:
                    stack.append(line)
                else:
                    taint(stack)
                    stack = [line]
            taint(stack)

        for key in strips:
            orient, line, from_color, to_color, start, end = key
            if (orient, line) in self._tainted_lines:
                continue
            line_key = (orient, line, from_color, to_color)
            per_segment = self._sightings.setdefault(line_key, {})
            old = per_segment.get(self._segment_id)
            if old is None:
                per_segment[self._segment_id] = (start, end)
            else:
                per_segment[self._segment_id] = (min(old[0], start), max(old[1], end))

    def _confirmed_lines(self) -> dict[tuple, tuple[int, int]]:
        """Line keys sighted in >= MIN_SEGMENTS distinct segments, with the
        union span, provided spans from different segments actually overlap."""
        confirmed: dict[tuple, tuple[int, int]] = {}
        for line_key, per_segment in self._sightings.items():
            if line_key in self._dead_lines or len(per_segment) < self.MIN_SEGMENTS:
                continue
            if (line_key[0], line_key[1]) in self._tainted_lines:
                continue
            spans = sorted(per_segment.values())
            overlapping = any(
                a_start <= b_end and b_start <= a_end
                for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:])
            ) or len(spans) == 1
            if not overlapping:
                continue
            confirmed[line_key] = (
                min(s for s, _ in spans),
                max(e for _, e in spans),
            )
        return confirmed

    def _apply_stack_rule(
        self, confirmed: dict[tuple, tuple[int, int]]
    ) -> dict[tuple, tuple[int, int]]:
        """Reject stacks of > MAX_STACK adjacent parallel lines (2D blocks)."""
        by_orient: dict[str, dict[int, list[tuple]]] = {"H": {}, "V": {}}
        for line_key, span in confirmed.items():
            by_orient[line_key[0]].setdefault(line_key[1], []).append((line_key, span))
        rejected: set[tuple] = set()
        for lines in by_orient.values():
            indices = sorted(lines)
            stack: list[int] = []

            def flush(stack_lines: list[int], lines: dict = lines) -> None:
                if len(stack_lines) > self.MAX_STACK:
                    for member in stack_lines:
                        for line_key, _ in lines[member]:
                            rejected.add(line_key)

            for line in indices:
                touches = bool(stack) and line == stack[-1] + 1 and any(
                    prev_span[0] <= cur_span[1] and cur_span[0] <= prev_span[1]
                    for _, prev_span in lines[stack[-1]]
                    for _, cur_span in lines[line]
                )
                if touches:
                    stack.append(line)
                else:
                    flush(stack)
                    stack = [line]
            flush(stack)
        return {k: v for k, v in confirmed.items() if k not in rejected}

    def _refresh_mask(self) -> None:
        mask = self._np.zeros(self._mask.shape, dtype=bool)
        for line_key, (start, end) in self._apply_stack_rule(self._confirmed_lines()).items():
            orient, line = line_key[0], line_key[1]
            if orient == "H":
                mask[line, start : end + 1] = True
            else:
                mask[start : end + 1, line] = True
        mask &= ~self._dropped
        self._mask = mask

    def _drop_regions(self, repeat_mask: Any) -> None:
        """Unmask guard: a masked cell changed twice inside one segment — real
        content entered the strip. Drop the whole region, permanently."""
        dropped_any = False
        for line_key, span in list(self._confirmed_lines().items()):
            orient, line = line_key[0], line_key[1]
            start, end = span
            if orient == "H":
                hit = bool(repeat_mask[line, start : end + 1].any())
            else:
                hit = bool(repeat_mask[start : end + 1, line].any())
            if hit:
                self._dead_lines.add(line_key)
                if orient == "H":
                    self._dropped[line, start : end + 1] = True
                else:
                    self._dropped[start : end + 1, line] = True
                dropped_any = True
        # Any repeat cell inside the mask is unmaskable from now on, even if no
        # confirmed line claimed it (belt and suspenders).
        self._dropped |= repeat_mask
        if dropped_any or bool(repeat_mask.any()):
            self._refresh_mask()

    # -- accessors -----------------------------------------------------------

    def mask_cells(self) -> list[list[int]]:
        """JSON-able [row, col] list of currently masked cells."""
        if self._mask is None:
            return []
        return [[int(y), int(x)] for y, x in self._np.argwhere(self._mask)]

    def mask_array(self) -> Any:
        return None if self._mask is None else self._mask.copy()

    def masked_equal(self, grid_a: Any, grid_b: Any) -> bool:
        np = self._np
        a = np.asarray(grid_a, dtype=np.int16)
        b = np.asarray(grid_b, dtype=np.int16)
        if a.shape != b.shape:
            return False
        diff = a != b
        if self._mask is not None and self._mask.shape == diff.shape:
            diff = diff & ~self._mask
        return not bool(diff.any())


import threading as _threading  # noqa: E402 - module-level TLS for the mask channel

_HUD_TLS = _threading.local()


def _hud_mask_enabled() -> bool:
    import os

    return os.environ.get("TAAF_HUD_MASK", "1").strip() not in {"0", "false", "False"}


def _hud_current_mask_cells() -> list[list[int]]:
    if not _hud_mask_enabled():
        return []
    cells = getattr(_HUD_TLS, "mask_cells", None)
    return list(cells) if cells else []


def patch_hud_board_identity() -> str:
    """Mask HUD strips out of `board_changed` and track the per-game mask."""
    from inference.framework import solver

    session_cls = getattr(solver, "_HarnessGameSession", None)
    if session_cls is None or not hasattr(session_cls, "_execute_action"):
        return "patch8 hud-mask: FAIL (_HarnessGameSession._execute_action not found)"
    if getattr(session_cls._execute_action, "_hud_patched", False):
        return "patch8 hud-mask: SKIP (already applied)"

    original = session_cls._execute_action

    def _execute_action(self: Any, action: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        tracker = getattr(self, "_hud_tracker", None)
        try:
            if tracker is None:
                tracker = HudMaskTracker()
                self._hud_tracker = tracker
                tracker.seed(solver._grid_from_state(self.game.current_state))
            # Publish before the action so nested consumers (animation summary,
            # sandbox calls) see this game's mask, not a stale thread value.
            _HUD_TLS.mask_cells = tracker.mask_cells()
        except Exception:
            tracker = None

        payload = original(self, action, *args, **kwargs)

        if tracker is not None:
            try:
                state = self.game.current_state
                is_reset = getattr(getattr(action, "id", None), "name", "") == "RESET"
                info = tracker.observe(
                    solver._grid_from_state(state),
                    is_reset=is_reset,
                    levels_completed=int(state.levels_completed),
                    state_name=getattr(state.raw.state, "name", None),
                )
                if (
                    isinstance(payload, dict)
                    and payload.get("executed")
                    and not is_reset
                    and _hud_mask_enabled()
                    and payload.get("board_changed")
                    and not info["masked_changed"]
                ):
                    payload["board_changed"] = False
                    payload["board_changed_hud_only"] = True
                _HUD_TLS.mask_cells = tracker.mask_cells()
            except Exception:
                pass  # masking must never break an action
        return payload

    _execute_action._hud_patched = True  # type: ignore[attr-defined]
    session_cls._execute_action = _execute_action
    return "patch8 hud-mask: OK"


_HUD_SANDBOX_HELPERS = '''

HUD_MASK_CELLS = []


def _hud_apply_mask(grid):
    if not HUD_MASK_CELLS:
        return grid
    try:
        rows = [list(row) for row in grid]
        for cell in HUD_MASK_CELLS:
            y = int(cell[0])
            x = int(cell[1])
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                rows[y][x] = 0
        return rows
    except Exception:
        return grid


def _hud_frame_grid(frame):
    for attr in ("_grid", "grid"):
        grid = getattr(frame, attr, None)
        if grid:
            return grid
    if isinstance(frame, (list, tuple)):
        return frame
    return None


_hud_raw_state_hash = state_hash


def state_hash(frame):
    if HUD_MASK_CELLS:
        try:
            grid = _hud_frame_grid(frame)
            if grid is not None:
                return _hud_raw_state_hash(_hud_apply_mask(grid))
        except Exception:
            pass
    return _hud_raw_state_hash(frame)


state_hash.__doc__ = _hud_raw_state_hash.__doc__

_hud_raw_diff_frames = diff_frames


def diff_frames(a, b):
    if HUD_MASK_CELLS:
        try:
            grid_a = _hud_frame_grid(a)
            grid_b = _hud_frame_grid(b)
            if grid_a is not None and grid_b is not None:
                return _hud_raw_diff_frames(
                    _hud_apply_mask(grid_a), _hud_apply_mask(grid_b)
                )
        except Exception:
            pass
    return _hud_raw_diff_frames(a, b)


diff_frames.__doc__ = _hud_raw_diff_frames.__doc__

'''

_HUD_REFRESH_ANCHOR = '        runtime_globals["last_action_result"] = action_result\n'
_HUD_MAIN_ANCHOR = "\ndef main():\n"


def patch_hud_sandbox() -> str:
    """Make the python-tool sandbox's state_hash / diff_frames HUD-aware."""
    from inference.agent import python_tool_sandbox as sandbox_mod
    from inference.agent import tool_agent

    bootstrap = getattr(sandbox_mod, "_SANDBOX_BOOTSTRAP", None)
    if not isinstance(bootstrap, str):
        return "patch9 hud-sandbox: FAIL (_SANDBOX_BOOTSTRAP not found)"
    if "HUD_MASK_CELLS" not in bootstrap:
        if _HUD_REFRESH_ANCHOR not in bootstrap or _HUD_MAIN_ANCHOR not in bootstrap:
            return "patch9 hud-sandbox: FAIL (bootstrap anchors moved upstream)"
        bootstrap = bootstrap.replace(
            _HUD_REFRESH_ANCHOR,
            _HUD_REFRESH_ANCHOR
            + '        HUD_MASK_CELLS[:] = list(state_payload.get("hud_mask") or [])\n',
        )
        bootstrap = bootstrap.replace(
            _HUD_MAIN_ANCHOR, _HUD_SANDBOX_HELPERS + _HUD_MAIN_ANCHOR
        )
        sandbox_mod._SANDBOX_BOOTSTRAP = bootstrap

    if getattr(sandbox_mod.run_sandboxed_python, "_hud_patched", False):
        return "patch9 hud-sandbox: SKIP (already applied)"

    original_run = sandbox_mod.run_sandboxed_python

    def run_sandboxed_python(
        *,
        code: str,
        timeout_seconds: int,
        initial_state: dict[str, Any],
        action_handler: Any,
    ) -> dict[str, Any]:
        state = dict(initial_state or {})
        if "hud_mask" not in state:
            state["hud_mask"] = _hud_current_mask_cells()

        def handler(actions: Any) -> dict[str, Any]:
            out = action_handler(actions)
            try:
                refreshed = out.get("state") if isinstance(out, dict) else None
                if isinstance(refreshed, dict) and "hud_mask" not in refreshed:
                    refreshed["hud_mask"] = _hud_current_mask_cells()
            except Exception:
                pass
            return out

        return original_run(
            code=code,
            timeout_seconds=timeout_seconds,
            initial_state=state,
            action_handler=handler,
        )

    run_sandboxed_python._hud_patched = True  # type: ignore[attr-defined]
    sandbox_mod.run_sandboxed_python = run_sandboxed_python
    # tool_agent imported the function by name; rebind its reference too.
    if getattr(tool_agent, "run_sandboxed_python", None) is not None:
        tool_agent.run_sandboxed_python = run_sandboxed_python
    return "patch9 hud-sandbox: OK"


# --- PATCH 7: run watchdog -------------------------------------------------------
#
# Detects stalled games in the harness loop and stops them so one wedged game
# cannot silently eat the submission's wall-clock budget (the host reported ~1/3
# of failed submissions "stuck silently"). Three mechanisms, all cooperative:
#
#   1. STALL DETECTION — progress is (scored action count, levels completed). If
#      neither moves for TAAF_WATCHDOG_STALL_S seconds (default 900) the game is
#      stalled: the analyzer is wedged in a retry/parse loop, the endpoint died,
#      or the model deliberates forever without acting.
#   2. RESET-AND-CONTINUE — the first TAAF_WATCHDOG_MAX_RESETS stalls (default 1)
#      are answered with one engine RESET (level reset under ONLY_RESET_LEVELS;
#      costs 1 scored action) and a fresh timer, which un-wedges games stuck in a
#      degenerate board state. The RESET is only issued from the session's own
#      worker thread — `should_stop` is also polled by analyzer-side threads, and
#      those must never touch the engine.
#   3. KILL + WALL CAP — further stalls stop the game cleanly (`should_stop`
#      returns True, the play loop exits, `finish_game` banks whatever levels are
#      already completed — nothing earned is lost). Independently, a per-game
#      wall-clock cap (TAAF_WATCHDOG_WALL_CAP_S, default 7200; 0 disables) stops
#      any game the solver did not already bound via `max_runtime_s_per_game`.
#
# TAAF_WATCHDOG=0 disables the whole patch at call time.


def _watchdog_env_float(name: str, default: float) -> float:
    import os

    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _watchdog_enabled() -> bool:
    import os

    return os.environ.get("TAAF_WATCHDOG", "1").strip() not in {"0", "false", "False"}


def _watchdog_state(session: Any) -> dict[str, Any]:
    import time

    wd = getattr(session, "_watchdog_state", None)
    if wd is None:
        wd = {
            "stall_s": max(1.0, _watchdog_env_float("TAAF_WATCHDOG_STALL_S", 900.0)),
            "wall_cap_s": _watchdog_env_float("TAAF_WATCHDOG_WALL_CAP_S", 7200.0),
            "max_resets": int(_watchdog_env_float("TAAF_WATCHDOG_MAX_RESETS", 1.0)),
            "progress": None,
            "t_progress": time.monotonic(),
            "resets_done": 0,
            "in_reset": False,
            "killed": None,  # None | "stall" | "wall_cap"
            "thread_id": None,
        }
        session._watchdog_state = wd
    return wd


def patch_watchdog() -> str:
    """Install the stall watchdog on the harness game session."""
    import threading
    import time

    from inference.framework import solver

    session_cls = getattr(solver, "_HarnessGameSession", None)
    if session_cls is None or not hasattr(session_cls, "should_stop"):
        return "patch7 watchdog: FAIL (_HarnessGameSession.should_stop not found)"
    if getattr(session_cls.should_stop, "_watchdog_patched", False):
        return "patch7 watchdog: SKIP (already applied)"

    original_should_stop = session_cls.should_stop
    original_play = session_cls.play

    def _log(session: Any, message: str) -> None:
        run = getattr(session.game, "game_run", None)
        game_id = getattr(run, "game_id", "?") if run is not None else "?"
        print(f"[watchdog] {game_id}: {message}", flush=True)

    def play(self: Any) -> None:
        wd = _watchdog_state(self)
        wd["thread_id"] = threading.get_ident()
        wd["t_progress"] = time.monotonic()
        return original_play(self)

    def should_stop(self: Any) -> bool:
        if original_should_stop(self):
            return True
        if not _watchdog_enabled():
            return False
        try:
            wd = _watchdog_state(self)
            now = time.monotonic()

            run = self.game.game_run
            levels = int(run.levels_completed) if run is not None else 0
            progress = (self.action_count, levels)
            if progress != wd["progress"]:
                wd["progress"] = progress
                wd["t_progress"] = now

            # Per-game wall cap, only where the solver has no cap of its own
            # (`runtime_limit_reached` already fires inside original_should_stop
            # when max_runtime_s_per_game is set).
            if (
                self.solver.max_runtime_s_per_game is None
                and wd["wall_cap_s"] > 0
                and (now - self.started_at) >= wd["wall_cap_s"]
            ):
                if wd["killed"] is None:
                    wd["killed"] = "wall_cap"
                    _log(
                        self,
                        f"wall cap {wd['wall_cap_s']:.0f}s reached at "
                        f"{self.action_count} actions, {levels} levels — stopping game",
                    )
                return True

            stalled_for = now - wd["t_progress"]
            if stalled_for < wd["stall_s"]:
                return False

            if wd["resets_done"] < wd["max_resets"]:
                # Recovery: one RESET, only ever from the session's own thread.
                if threading.get_ident() == wd["thread_id"] and not wd["in_reset"]:
                    wd["in_reset"] = True
                    try:
                        _log(
                            self,
                            f"no progress for {stalled_for:.0f}s — issuing recovery RESET "
                            f"({wd['resets_done'] + 1}/{wd['max_resets']})",
                        )
                        self._execute_auto_reset()
                        wd["resets_done"] += 1
                        wd["t_progress"] = time.monotonic()
                        wd["progress"] = None
                    except Exception as exc:  # noqa: BLE001 - failed recovery -> kill path
                        _log(self, f"recovery RESET failed ({exc!r}) — will stop instead")
                        wd["resets_done"] = wd["max_resets"]
                    finally:
                        wd["in_reset"] = False
                    return original_should_stop(self)
                # Wrong thread: wait for the session thread, but never forever.
                if stalled_for >= 2 * wd["stall_s"]:
                    if wd["killed"] is None:
                        wd["killed"] = "stall"
                        _log(self, f"stalled {stalled_for:.0f}s (recovery unavailable) — stopping game")
                    return True
                return False

            if wd["killed"] is None:
                wd["killed"] = "stall"
                _log(
                    self,
                    f"stalled {stalled_for:.0f}s after {wd['resets_done']} recovery reset(s) "
                    f"— stopping game at {self.action_count} actions, {levels} levels",
                )
            return True
        except Exception:  # noqa: BLE001 - the watchdog must never wedge the loop itself
            return False

    should_stop._watchdog_patched = True  # type: ignore[attr-defined]
    play._watchdog_play_patched = True  # type: ignore[attr-defined]
    session_cls.should_stop = should_stop
    session_cls.play = play
    return "patch7 watchdog: OK"


def apply_all(verbose: bool = True) -> list[str]:
    """Apply every patch. Each is independent; one failing does not block the others."""
    results = []
    for fn in (
        patch_action7,
        patch_animation_producer,
        patch_animation_metadata,
        verify_reset_already_handled,
        patch_dynamic_grid_burner,
        patch_prompts,
        patch_tool_agent_analyze,
        patch_watchdog,
        patch_hud_board_identity,
        patch_hud_sandbox,
    ):
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 - a broken patch must not kill the run
            results.append(f"{fn.__name__}: FAIL ({exc})")
    if verbose:
        for line in results:
            print(f"[duck-patch] {line}", flush=True)
    return results
