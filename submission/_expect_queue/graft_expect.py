"""Expect-queue graft — per-step expectation checks + halt-and-return-control
on batched actions.

Motivation (measured, docs/RESULTS-2026-08-22-testing-campaign.md row 1):
replaying 652 batches against a halt rule saved 14.53% of batched actions
(m0r0 26%, sk48 25%, dc22 22%) — but 72% of halted remainders still had
SOME effect, so the right shape is HALT-AND-RETURN-CONTROL (truncate the
rest of the batch and hand the model a compact diff for a cheap re-plan),
never hard-discard-and-pretend.

Three mechanisms behind one flag (EXPECT_QUEUE, default on):

1. `expect` transport: each action object passed to the sandbox
   ``action(actions)`` may carry an optional ``expect`` list of up to
   ``MAX_EXPECT_CELLS`` cells ``[row, col, color]`` that the settled
   post-step frame must show.
2. Host-side per-step verification inside the batch executor: after each
   executed step, if the step's expect cells mismatch the post-step frame,
   or the step was a strict no-op (post-frame bit-identical to pre-frame),
   the REMAINING actions are not executed and the tool result reports
   ``stop_reason`` (``expect_mismatch`` / ``no_op``) plus a compact
   ``stop_detail`` diff ("step K halted the batch: ...; remaining N actions
   not executed"). Executed steps are never rolled back. Steps that complete
   a level (or end the run / hit game over) never halt via this graft — the
   stock terminal breaks fire first.
3. Prompt addendum: probe-then-commit contract; batches of more than
   ``EXPECT_BATCH_GUIDANCE`` actions "must" include expects — guidance the
   model is asked to follow, NOT hard-enforced (fail-open: expect-less
   batches still execute, and still halt on strict no-ops).

No-op rule shipped: FULL-FRAME identity — the stock ``board_changed``
computation (``previous_grid != _grid_from_state(new_state)``,
solver.py:703). No HUD band-rule exists in the June stock (the HUD
knowledge is prompt text only, prompts.py:29/95), so a HUD tick counts as
"changed" and suppresses the no-op halt. That is the conservative
direction: HUD noise can only cause FEWER halts, never a false halt.

Envelope note: this graft can only TRUNCATE batches — it removes engine
steps and returns control early. It adds no sleeps, retries, or decode
obligations beyond a bounded prompt addendum, and cannot extend
``max_runtime_s_per_game`` or any timeout.

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference — same pin the
_yield_carryover / _truthful_telemetry grafts were built against):

- solver.py:588-661 — ``_HarnessGameSession.step_env``: the batch loop this
  graft re-executes with halt checks added. Stock per-step breaks:
  should_stop (:605-607), invalid_action (:608-613), action_error
  (:615-626), run_complete/game_over/level_completed (:630-638);
  aggregation :643-659; viewer flush :660. The ported loop reproduces all
  of them via the SAME session helpers (``_normalize_actions``,
  ``_error_payload``, ``_terminal_payload``, ``_execute_action``,
  ``write_viewer_payload``), then adds expect/no-op checks AFTER the stock
  terminal breaks.
- solver.py:667-734 — ``_execute_action``; :703 is the full-frame identity
  bit (``board_changed``) the no-op halt keys on; :93-98
  ``_grid_from_state`` supplies the post-step frame for expect checks.
- solver.py:491-542 — ``_normalize_actions`` reads only
  ``action``/``row``/``col`` from raw dicts, so an extra ``expect`` key is
  tolerated (ignored) by every stock path.
- solver.py:297 — ``step_env=self.step_env`` is bound per analyze call via
  attribute lookup, so a class-attribute patch reaches every session;
  single-namespace (grep: no by-name import of ``step_env`` anywhere).
- tool_agent.py:1378-1414 — ``ToolAgent._normalize_python_actions`` DROPS
  every key except action/row/col; the transport patch re-attaches a
  validated ``expect`` per action (malformed expects raise ValueError into
  the sandbox, same feedback channel as the legacy MOUSE x/y error at
  :1405-1406).
- tool_agent.py:1495-1545 — the sandbox ``action(actions)`` handler:
  :1499 normalizes (our expect survives), :1529 forwards
  ``{"actions": normalized}`` to the session's ``step_env``.
- python_tool_sandbox.py:269-307 (inside the ``_SANDBOX_BOOTSTRAP`` string
  literal, :21-398) — a SECOND normalizer runs in the sandbox SUBPROCESS
  and strips unknown keys before anything reaches the host. The graft
  patches the module-level bootstrap string by anchored replacement
  (unique ``entry["col"]`` block), adding validated ``expect``
  pass-through. Validation lives sandbox-side because a raise there
  surfaces to the model as a normal Python error, whereas a host-side
  ``action_handler`` exception collapses to the generic "action failed in
  sandbox host." (python_tool_sandbox.py:540-547).
  ``run_sandboxed_python`` reads ``_SANDBOX_BOOTSTRAP`` at call time from
  its own module global (:459) and tool_agent imports only the function
  (:32), so patching the module attribute covers every call.
- tool_agent.py:1416-1451 — ``_compact_action_result`` passes
  ``stop_reason``/``stop_detail`` (:1442-1445) into the model-visible tool
  result, so the halt diff reaches the model with zero extra patching.
- tool_agent.py:184-193 — ``_terminal_action_reason`` keys on
  run_complete/game_over/level_completed/done flags, NOT stop_reason, so
  ``expect_mismatch``/``no_op`` results are never misread as terminal.
- tool_agent.py:350-359 — ``_build_system_prompt`` reads module globals at
  call time; :941 is its ONLY call site (``ToolAgent.__init__``), resolved
  as a module global — patching ``agent_mod._build_system_prompt`` covers
  it (single namespace).

Fail-open invariants:
- The stock-equivalent calls inside the ported loop
  (``_execute_action``, ``_normalize_actions``, ...) are never wrapped in
  graft try/except — they behave exactly as stock, including exception
  semantics.
- Every graft-added check (expect extraction, cell comparison, no-op test,
  prompt append) sits inside try/except; any error means "no halt / stock
  prompt".
- If extraction fails before any action executes, or the request is a
  single expect-less action, the wrapper delegates to the ORIGINAL
  ``step_env`` untouched.
- EXPECT_QUEUE=0 disables at install time (no patches) and at call time
  (installed wrappers become pure pass-throughs).
- The only deliberate new exception is the ValueError for a malformed
  ``expect`` (model-facing feedback inside the sandbox); it can only fire
  when the flag is on AND the model actually supplied an ``expect``.
"""

from __future__ import annotations

import os
from typing import Any

MAX_EXPECT_CELLS = 16
GRID_MAX_INDEX = 63
EXPECT_BATCH_GUIDANCE = 3
_DETAIL_CELL_CAP = 4

STOP_REASON_EXPECT = "expect_mismatch"
STOP_REASON_NOOP = "no_op"

EXPECT_ADDENDUM_MARKER = "Batch expect contract (probe-then-commit):"
EXPECT_ADDENDUM = (
    f"\n\n{EXPECT_ADDENDUM_MARKER}\n"
    "- Each action object passed to `action(actions)` may include an optional `expect` list of up to "
    f"{MAX_EXPECT_CELLS} cells, each `[row, col, color]`, that the settled post-step frame must show.\n"
    "- After each step of a batch the harness verifies the frame: if a step's expect cells mismatch, or the "
    "step changed nothing at all (strict no-op), the REMAINING actions of the batch are NOT executed and the "
    "result reports `stop_reason` (`expect_mismatch` / `no_op`) with a compact diff in `stop_detail`. "
    "Executed steps are never rolled back; steps that complete a level never halt the batch this way.\n"
    "- Probe-then-commit: probe with single actions until the mechanics are understood, then commit batches "
    f"with expects. Batches of more than {EXPECT_BATCH_GUIDANCE} actions must include `expect` cells on at "
    "least their later steps, chosen to discriminate (cells your world model predicts will change).\n"
    "- A truncated batch is cheap re-planning, not failure: read `executed_count`, `stop_detail`, and the "
    "fresh frame, then re-plan from the actual state.\n"
)


def _enabled() -> bool:
    return os.environ.get("EXPECT_QUEUE", "1").strip() not in {"0", "false", "False"}


# --- sandbox bootstrap surgery (transport through the subprocess) -----------

_BOOTSTRAP_ANCHOR = (
    '            if "col" in item:\n'
    '                entry["col"] = item.get("col")\n'
)
_BOOTSTRAP_EXPECT_BLOCK = (
    '            if "expect" in item:\n'
    "                _expect_value = item.get(\"expect\")\n"
    "                if not isinstance(_expect_value, (list, tuple)) or not (\n"
    f"                    1 <= len(_expect_value) <= {MAX_EXPECT_CELLS}\n"
    "                ):\n"
    "                    raise ValueError(\n"
    f'                        f"Action {{index}}: `expect` must be a list of 1..{MAX_EXPECT_CELLS} '
    '[row, col, color] cells."\n'
    "                    )\n"
    "                _expect_cells = []\n"
    "                for _cell_index, _cell in enumerate(_expect_value, start=1):\n"
    "                    if not isinstance(_cell, (list, tuple)) or len(_cell) != 3:\n"
    "                        raise ValueError(\n"
    '                            f"Action {index}: `expect` cell {_cell_index} must be [row, col, color]."\n'
    "                        )\n"
    "                    try:\n"
    "                        _row, _col, _color = int(_cell[0]), int(_cell[1]), int(_cell[2])\n"
    "                    except (TypeError, ValueError):\n"
    "                        raise ValueError(\n"
    '                            f"Action {index}: `expect` cell {_cell_index} must contain integers."\n'
    "                        )\n"
    f"                    if not (0 <= _row <= {GRID_MAX_INDEX} and 0 <= _col <= {GRID_MAX_INDEX}) or _color < 0:\n"
    "                        raise ValueError(\n"
    '                            f"Action {index}: `expect` cell {_cell_index} out of range '
    f'(row/col 0..{GRID_MAX_INDEX}, color >= 0)."\n'
    "                        )\n"
    "                    _expect_cells.append([_row, _col, _color])\n"
    '                entry["expect"] = _expect_cells\n'
)


# --- expect validation (transport seam) -------------------------------------


def _validate_expect(value: Any, index: int) -> list[list[int]]:
    """Validated ``[[row, col, color], ...]`` or a model-facing ValueError."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Action {index}: `expect` must be a list of [row, col, color] cells."
        )
    if len(value) == 0:
        raise ValueError(f"Action {index}: `expect` must not be empty (omit it instead).")
    if len(value) > MAX_EXPECT_CELLS:
        raise ValueError(
            f"Action {index}: `expect` lists at most {MAX_EXPECT_CELLS} cells (got {len(value)})."
        )
    cells: list[list[int]] = []
    for cell_index, cell in enumerate(value, start=1):
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            raise ValueError(
                f"Action {index}: `expect` cell {cell_index} must be [row, col, color]."
            )
        try:
            row, col, color = (int(cell[0]), int(cell[1]), int(cell[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Action {index}: `expect` cell {cell_index} must contain integers."
            ) from exc
        if not (0 <= row <= GRID_MAX_INDEX and 0 <= col <= GRID_MAX_INDEX):
            raise ValueError(
                f"Action {index}: `expect` cell {cell_index} row/col must be within 0..{GRID_MAX_INDEX}."
            )
        if color < 0:
            raise ValueError(
                f"Action {index}: `expect` cell {cell_index} color must be a non-negative integer."
            )
        cells.append([row, col, color])
    return cells


def _expects_from_arguments(arguments: dict[str, Any]) -> list[Any]:
    """Per-action expect lists aligned with the requested-actions order.

    Mirrors ``_normalize_actions``'s raw-list construction (solver.py:494-514):
    batch form reads each item's ``expect``; single form reads the top-level
    ``expect`` key.
    """
    raw = arguments.get("actions")
    if isinstance(raw, list):
        return [item.get("expect") if isinstance(item, dict) else None for item in raw]
    return [arguments.get("expect")]


# --- host-side per-step verification ----------------------------------------


def _expect_mismatches(
    grid: tuple[tuple[int, ...], ...], expect: Any
) -> list[dict[str, int]] | None:
    """Mismatched cells vs the post-step grid, or None when unchecked.

    Lenient by design at this layer (validation already happened at the
    transport seam): malformed cells are skipped rather than halting.
    """
    if not isinstance(expect, (list, tuple)) or not expect:
        return None
    mismatches: list[dict[str, int]] = []
    checked = 0
    for cell in expect[:MAX_EXPECT_CELLS]:
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            continue
        try:
            row, col, want = (int(cell[0]), int(cell[1]), int(cell[2]))
            got = int(grid[row][col])
        except (TypeError, ValueError, IndexError):
            continue
        checked += 1
        if got != want:
            mismatches.append({"row": row, "col": col, "want": want, "got": got})
    if checked == 0:
        return None
    return mismatches


def _mismatch_summary(mismatches: list[dict[str, int]]) -> str:
    shown = ", ".join(
        f"[{cell['row']},{cell['col']}] got {cell['got']} want {cell['want']}"
        for cell in mismatches[:_DETAIL_CELL_CAP]
    )
    extra = len(mismatches) - _DETAIL_CELL_CAP
    if extra > 0:
        shown += f" (+{extra} more)"
    return shown


def _halt_verdict(
    session: Any,
    solver_mod: Any,
    payload: dict[str, Any],
    expect: Any,
    step_index: int,
    batch_size: int,
) -> tuple[str | None, str | None, list[dict[str, int]] | None]:
    """(stop_reason, stop_detail, mismatches) for one executed step.

    Called ONLY after the stock terminal breaks (run_complete / game_over /
    level_completed) have been given precedence — a level-completing step can
    never reach this check.
    """
    remaining = batch_size - step_index
    mismatches: list[dict[str, int]] | None = None
    if expect is not None:
        grid = solver_mod._grid_from_state(session.game.current_state)
        mismatches = _expect_mismatches(grid, expect)
    if mismatches:
        detail = (
            f"step {step_index} halted the batch: expect mismatch at "
            f"{_mismatch_summary(mismatches)}; remaining {remaining} actions not executed"
            if remaining > 0
            else (
                f"step {step_index}: expect mismatch at {_mismatch_summary(mismatches)}; "
                "batch already complete (0 remaining)"
            )
        )
        return STOP_REASON_EXPECT, detail, mismatches
    if remaining > 0 and payload.get("executed") and not payload.get("board_changed"):
        # Strict no-op: full-frame identity (solver.py:703). A HUD tick makes
        # board_changed True and suppresses this halt — conservative direction.
        detail = (
            f"step {step_index} halted the batch: strict no-op (post-step frame bit-identical "
            f"to pre-step frame); remaining {remaining} actions not executed"
        )
        return STOP_REASON_NOOP, detail, mismatches
    return None, None, mismatches


def _run_batch_with_expects(
    session: Any,
    solver_mod: Any,
    arguments: dict[str, Any],
    expects: list[Any],
) -> dict[str, Any]:
    """Faithful port of the stock batch loop (solver.py:588-661) with the
    graft's halt checks added after the stock terminal breaks."""
    requested_actions, error = session._normalize_actions(arguments)
    if error is not None or requested_actions is None:
        return session._error_payload(error or "Could not parse action request.")
    if session.should_stop() or solver_mod._is_engine_game_over(session.game):
        return session._terminal_payload(requested_actions)

    executed_payloads: list[dict[str, Any]] = []
    total_reward = 0.0
    stop_reason: str | None = None
    stop_detail: str | None = None
    expect_mismatch: list[dict[str, int]] | None = None
    batch_size = len(requested_actions)
    requested_displays = [
        solver_mod._format_action_display(action.id.name, dict(action.data))
        for action in requested_actions
    ]

    for batch_index, action in enumerate(requested_actions, start=1):
        if session.should_stop():
            stop_reason = "stopped"
            break
        if action.id.value not in session.game.current_state.available_actions:
            message = (
                f"{solver_mod._format_action_display(action.id.name, dict(action.data))} "
                "is not valid right now."
            )
            if executed_payloads:
                stop_reason = "invalid_action"
                break
            return session._error_payload(message)

        try:
            payload = session._execute_action(
                action,
                batch_index=batch_index,
                batch_size=batch_size,
                flush_viewer_payload=False,
            )
        except Exception as exc:
            if executed_payloads:
                stop_reason = "action_error"
                break
            return session._error_payload(f"{type(exc).__name__}: {exc}")
        executed_payloads.append(payload)
        total_reward += float(payload.get("reward", 0.0) or 0.0)

        # Stock terminal breaks take precedence: a level-completing (or
        # run-ending) step NEVER halts via expect/no-op.
        if payload.get("run_complete"):
            stop_reason = "run_complete"
            break
        if payload.get("game_over"):
            stop_reason = "game_over"
            break
        if payload.get("level_completed"):
            stop_reason = "level_completed"
            break

        # Graft halt checks — guarded: any error means "no halt".
        try:
            expect = expects[batch_index - 1] if batch_index - 1 < len(expects) else None
            halt_reason, halt_detail, mismatches = _halt_verdict(
                session, solver_mod, payload, expect, batch_index, batch_size
            )
        except Exception:  # noqa: BLE001 — verification must never break the step
            halt_reason, halt_detail, mismatches = None, None, None
        if mismatches:
            expect_mismatch = mismatches
        if halt_reason is not None:
            stop_reason = halt_reason
            stop_detail = halt_detail
            break

    if not executed_payloads:
        return session._error_payload("No action was executed.")

    final_payload = dict(executed_payloads[-1])
    final_payload["reward"] = total_reward
    final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
    final_payload["batched"] = batch_size > 1
    final_payload["requested_count"] = batch_size
    final_payload["executed_count"] = len(executed_payloads)
    final_payload["requested_actions"] = requested_displays
    final_payload["executed_actions"] = [
        str(item.get("action_display") or item.get("action_name") or "")
        for item in executed_payloads
    ]
    final_payload["board_changed"] = any(
        bool(item.get("board_changed")) for item in executed_payloads
    )
    final_payload["stopped_early"] = len(executed_payloads) < batch_size
    if stop_reason is not None:
        final_payload["stop_reason"] = stop_reason
    if stop_detail is not None:
        final_payload["stop_detail"] = stop_detail
    if expect_mismatch is not None:
        final_payload["expect_mismatch"] = expect_mismatch
    session.write_viewer_payload()
    return final_payload


def install() -> str:
    if not _enabled():
        return "expect_queue: SKIP (EXPECT_QUEUE=0)"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"expect_queue: SKIP (solver module missing: {exc!r})"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"expect_queue: SKIP (tool_agent module missing: {exc!r})"
    try:
        from inference.agent import python_tool_sandbox as sandbox_mod
    except Exception as exc:  # noqa: BLE001
        return f"expect_queue: SKIP (python_tool_sandbox module missing: {exc!r})"

    # Presence gates — every seam symbol, fail toward stock on any mismatch.
    session_cls = getattr(solver_mod, "_HarnessGameSession", None)
    if session_cls is None:
        return "expect_queue: SKIP (missing _HarnessGameSession)"
    original_step_env = getattr(session_cls, "step_env", None)
    if original_step_env is None:
        return "expect_queue: SKIP (missing _HarnessGameSession.step_env)"
    for name in (
        "_execute_action",
        "_normalize_actions",
        "_error_payload",
        "_terminal_payload",
        "should_stop",
        "write_viewer_payload",
    ):
        if getattr(session_cls, name, None) is None:
            return f"expect_queue: SKIP (missing _HarnessGameSession.{name})"
    for name in ("_grid_from_state", "_is_engine_game_over", "_format_action_display"):
        if getattr(solver_mod, name, None) is None:
            return f"expect_queue: SKIP (missing solver.{name})"
    tool_agent_cls = getattr(agent_mod, "ToolAgent", None)
    original_normalize = getattr(tool_agent_cls, "_normalize_python_actions", None)
    if original_normalize is None:
        return "expect_queue: SKIP (ToolAgent._normalize_python_actions missing)"
    original_build_prompt = getattr(agent_mod, "_build_system_prompt", None)
    if original_build_prompt is None:
        return "expect_queue: SKIP (_build_system_prompt missing)"
    original_bootstrap = getattr(sandbox_mod, "_SANDBOX_BOOTSTRAP", None)
    if not isinstance(original_bootstrap, str):
        return "expect_queue: SKIP (_SANDBOX_BOOTSTRAP missing)"
    bootstrap_already = 'entry["expect"]' in original_bootstrap
    if not bootstrap_already and original_bootstrap.count(_BOOTSTRAP_ANCHOR) != 1:
        return "expect_queue: SKIP (sandbox bootstrap anchor moved)"
    if getattr(original_step_env, "_expect_queue_patched", False):
        return "expect_queue: SKIP (already applied)"

    # --- transport: keep `expect` alive through the sandbox normalizer ------

    def normalize_with_expect(self: Any, value: Any) -> list[dict[str, Any]]:
        normalized = original_normalize(self, value)
        if not _enabled():
            return normalized
        try:
            if isinstance(value, (list, tuple)):
                items = list(value)
            else:
                items = [value]
        except Exception:  # noqa: BLE001 — coercion failure => stock output
            return normalized
        for index, (entry, item) in enumerate(zip(normalized, items), start=1):
            if isinstance(item, dict) and "expect" in item:
                # ValueError here is deliberate model-facing feedback in the
                # sandbox — same channel as the stock legacy-MOUSE error.
                entry["expect"] = _validate_expect(item.get("expect"), index)
        return normalized

    # --- batch executor with halt checks ------------------------------------

    def step_env_with_expects(self: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        if not _enabled():
            return original_step_env(self, arguments)
        try:
            expects = _expects_from_arguments(arguments)
            raw_actions = arguments.get("actions")
            batch_length = len(raw_actions) if isinstance(raw_actions, list) else 1
            wants_graft_path = batch_length > 1 or any(
                expect is not None for expect in expects
            )
        except Exception:  # noqa: BLE001 — extraction failure => stock
            return original_step_env(self, arguments)
        if not wants_graft_path:
            return original_step_env(self, arguments)
        return _run_batch_with_expects(self, solver_mod, arguments, expects)

    # --- prompt addendum (call-time flagged, injected exactly once) ---------

    def build_system_prompt_with_expect(*args: Any, **kwargs: Any) -> str:
        prompt = original_build_prompt(*args, **kwargs)
        try:
            if _enabled() and EXPECT_ADDENDUM_MARKER not in prompt:
                prompt += EXPECT_ADDENDUM
        except Exception:  # noqa: BLE001 — addendum failure => stock prompt
            pass
        return prompt

    step_env_with_expects._expect_queue_patched = True  # type: ignore[attr-defined]
    session_cls.step_env = step_env_with_expects
    tool_agent_cls._normalize_python_actions = normalize_with_expect
    agent_mod._build_system_prompt = build_system_prompt_with_expect
    if not bootstrap_already:
        sandbox_mod._SANDBOX_BOOTSTRAP = original_bootstrap.replace(
            _BOOTSTRAP_ANCHOR, _BOOTSTRAP_ANCHOR + _BOOTSTRAP_EXPECT_BLOCK, 1
        )
    # Single-namespace by design: step_env is only bound via ``self.step_env``
    # (solver.py:297), _normalize_python_actions via ``self.`` (:1499), and
    # _build_system_prompt via the tool_agent module global (:941).
    install.originals = {  # type: ignore[attr-defined]
        "step_env": original_step_env,
        "_normalize_python_actions": original_normalize,
        "_build_system_prompt": original_build_prompt,
        "_SANDBOX_BOOTSTRAP": original_bootstrap,
    }
    return "expect_queue: OK"
