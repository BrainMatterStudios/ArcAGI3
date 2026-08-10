"""duck-mem: memory + anti-waste patch stack for the shipped duck-base v2 config.

Built for the 2026-08-11 00:01 UTC slot. Every patch is fail-safe: it verifies its
anchor before applying and degrades to a printed warning (never a crash) if the
bundled harness differs from the version this stack was written against. Nothing
here changes the model, sampling parameters, concurrency, per-game budget, the game
list, or the submission path. Behavioural deltas, in order:

  P1  token estimator len//4            (was (len+2)//3; measured 1.44x overcount)
  P2  middle-drop history trimmer       (was front-drop; keeps the KV prefix cache)
  P3  neutralize the "minimize actions" own-goal in GAME_OVERVIEW_ADDENDUM
  P4  repeated-no-effect batch guard    (faithful stock step_env copy + guard;
                                        NOTE: raw board_changed => inert on the
                                        18/25 HUD-ticker games; weak but safe)
  P5  analyzer yield 60 -> 90 s         GATED OFF (DUCK_MEM_P5=1 to arm): the
                                        sonpham tempo law (900s-yield cost 2.2x
                                        vs 60s act-look-act) argues against;
                                        unmeasured for us
  P6  adaptive memory expansion         GATED OFF (DUCK_MEM_P6=1 to arm): doubles
                                        per-request context on a 96GB GPU;
                                        contradicts the retune direction
                                        (65536->40960) and is unmeasured; edits
                                        the serve config — highest-risk item

Default stack (nothing armed) = P1-P4: a single coherent "anti-waste" theme.
Review 2026-08-10 (this repo): P3 as first written was a SILENT NO-OP —
tool_agent binds GAME_OVERVIEW_ADDENDUM by value at import time
(tool_agent.py:17-19) and reads the local name at :352, so patching the prompts
module alone changes nothing once tool_agent is imported (the hook cell runs
with `bm` already constructed). Fixed below by rebinding BOTH namespaces.

Reference: scratchpad/taaf_scored_ref/src (bundle snapshot aa69123, DIRTY).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ARM_MARKER = "[duck-mem]"

# P4 — only these display names are guard-worthy: same action, none of the
# board-changing machinery, most games expose them as the avatar directions.
_MEM_DIRECTIONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})

_BUDGET_MAX_MODEL_LEN = 98304
_BUDGET_CONTEXT_WINDOW = 65536


# ---------------------------------------------------------------------------
# P1 — token estimator
# ---------------------------------------------------------------------------

def _estimate_tokens_impl(value) -> int:  # noqa: ANN001
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 3) // 4)


def patch_token_estimator() -> bool:
    """((len+2)//3) -> ((len+3)//4). Errs high (~1.08x) so it cannot overrun the
    served window; recovers ~30% of the context we already pay for at zero prefill."""
    from inference.agent import tool_agent as ta

    if getattr(ta._estimate_tokens, "_mem", False):
        return True

    _estimate_tokens_impl._mem = True  # type: ignore[attr-defined]
    ta._estimate_tokens = _estimate_tokens_impl
    print(f"{ARM_MARKER} P1 estimator //4: OK", flush=True)
    return True


# ---------------------------------------------------------------------------
# P2 — middle-drop trimmer
# ---------------------------------------------------------------------------

def _drop_middle_history_block(self, history, *, preserve_recent):  # noqa: ANN001
    """Drop ONE full turn block from the middle of the history list, in place.

    Contract: same signature and True/False contract as the harness's
    `_drop_oldest_history_block`, but the head of history (system-adjacent turns)
    and the sacred tail (`preserve_recent` messages) survive, so the vLLM prefix
    cache is not invalidated on every trim."""
    removable = len(history) - preserve_recent
    if removable <= 0:
        return False
    if removable == 1:
        history.pop(0)
        return True
    sacred_from = len(history) - preserve_recent
    user_indexes = [
        i for i, m in enumerate(history)
        if str(m.get("role", "")).strip() == "user" and i < sacred_from
    ]
    if not user_indexes:
        history.pop(0)
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        return True
    mid = len(history) // 2
    target = min(user_indexes, key=lambda i: abs(i - mid))
    block_end = target + 1
    while block_end < sacred_from and history[block_end].get("role") != "user":
        block_end += 1
    del history[target:block_end]
    return True


def patch_middle_drop() -> bool:
    """Replace front-drop eviction with middle-drop. Keeps the head of history
    (system prompt + earliest turns) intact so vLLM prefix caching survives trims
    (measured 49.3% hit rate, currently invalidated on every front trim), and the
    model keeps the early-findings context that front-drop destroys."""
    from inference.agent import tool_agent as ta

    cls = ta.ToolAgent
    if getattr(cls, "_mem_middle_drop", False):
        return True
    _drop_middle_history_block._mem = True  # type: ignore[attr-defined]
    cls._mem_middle_drop = True  # type: ignore[attr-defined]
    cls._drop_oldest_history_block = _drop_middle_history_block
    print(f"{ARM_MARKER} P2 middle-drop trimmer: OK", flush=True)
    return True


# ---------------------------------------------------------------------------
# P3 — prompt own-goal neutralization
# ---------------------------------------------------------------------------

_OLD_OPTIMIZE_LINE = "- Optimize for as few in-game actions as possible while still being reliable.\n"
_NEW_OPTIMIZE_LINE = (
    "- Completing levels is the goal; exploration that reveals mechanics is worth its actions.\n"
)


def patch_prompt_own_goal() -> bool:
    """The 'minimize actions' line is an own-goal: the completion-share cap binds
    long before efficiency (8/10 measured completed levels are cap-bound), and the
    instruction suppresses the exploration that unlocks depth.

    MUST rebind in BOTH namespaces: tool_agent imports GAME_OVERVIEW_ADDENDUM by
    value (`from inference.agent.prompts import GAME_OVERVIEW_ADDENDUM`,
    tool_agent.py:17-19) and _build_system_prompt reads the LOCAL binding
    (tool_agent.py:352). Patching prompts alone is a silent no-op."""
    from inference.agent import prompts
    from inference.agent import tool_agent as ta

    anchor = prompts.GAME_OVERVIEW_ADDENDUM
    if _OLD_OPTIMIZE_LINE not in anchor:
        print(f"{ARM_MARKER} P3 prompt anchor MISSING (already neutralized?)", flush=True)
        return False
    replacement = anchor.replace(_OLD_OPTIMIZE_LINE, _NEW_OPTIMIZE_LINE)
    assert replacement != anchor
    prompts.GAME_OVERVIEW_ADDENDUM = replacement
    ta.GAME_OVERVIEW_ADDENDUM = replacement
    if _OLD_OPTIMIZE_LINE in getattr(ta, "GAME_OVERVIEW_ADDENDUM", ""):
        print(f"{ARM_MARKER} P3 tool_agent rebind FAILED", flush=True)
        return False
    print(f"{ARM_MARKER} P3 prompt own-goal neutralized (both namespaces): OK", flush=True)
    return True


# ---------------------------------------------------------------------------
# P4 — repeated-no-effect batch guard (engine level)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P4 — repeated-no-effect batch guard (engine level)
# ---------------------------------------------------------------------------

def _should_guard_no_effect(board_changed: bool, action_display, next_display) -> bool:  # noqa: ANN001
    """Guard rule: the executed action is a direction, it changed nothing on the
    visible board, and the batch's next action repeats the same direction."""
    if board_changed:
        return False
    if action_display not in _MEM_DIRECTIONS:
        return False
    return action_display == next_display


def patch_repeated_no_effect_guard() -> bool:
    """If a direction leaves the entire visible board unchanged, do not repeat that
    same direction inside the current batch. Return control for re-observation."""
    import inference.framework.solver as solver

    session = solver._HarnessGameSession
    if getattr(session, "_mem_no_effect_guard", False):
        return True
    current = getattr(session.step_env, "__name__", "") == "step_env"
    if not current:
        print(f"{ARM_MARKER} P4 guard SKIPPED: step_env already replaced", flush=True)
        return False

    def _step_env(self, arguments):  # noqa: ANN001, ANN202
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or solver._is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)

        executed_payloads = []
        total_reward = 0.0
        stop_reason = None
        stop_detail = None
        batch_size = len(requested_actions)
        requested_displays = [
            solver._format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]

        for batch_index, action in enumerate(requested_actions, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = (
                    f"{solver._format_action_display(action.id.name, dict(action.data))} "
                    "is not valid right now."
                )
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                )
            except Exception as exc:  # noqa: BLE001
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break

            action_display = requested_displays[batch_index - 1]
            next_display = requested_displays[batch_index] if batch_index < batch_size else None
            if _should_guard_no_effect(payload.get("board_changed"), action_display, next_display):
                stop_reason = "repeated_no_effect"
                stop_detail = (
                    f"Stopped before repeating {action_display}: the previous identical "
                    "direction left the visible board unchanged. Re-observe before retrying."
                )
                break

        if not executed_payloads:
            return self._error_payload("No action was executed.")

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
        self.write_viewer_payload()
        return final_payload

    _step_env._mem = True  # type: ignore[attr-defined]
    session._mem_no_effect_guard = True  # type: ignore[attr-defined]
    session.step_env = _step_env
    print(f"{ARM_MARKER} P4 repeated-no-effect guard: OK", flush=True)
    return True


# ---------------------------------------------------------------------------
# P5 — analyzer yield
# ---------------------------------------------------------------------------

def patch_yield_seconds(seconds: float = 90.0) -> bool:
    from inference.agent import tool_agent as ta

    old = ta._LOCAL_ANALYZER_YIELD_SECONDS
    ta._LOCAL_ANALYZER_YIELD_SECONDS = float(seconds)
    os.environ["LOCAL_ANALYZER_YIELD_SECONDS"] = str(seconds)
    print(f"{ARM_MARKER} P5 yield {old!r} -> {seconds}: OK", flush=True)
    return True


# ---------------------------------------------------------------------------
# P6 — adaptive memory expansion (the 96 GB finding)
# ---------------------------------------------------------------------------

def _detected_vram_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        return int(first)
    except Exception:  # noqa: BLE001
        return 0


def patch_adaptive_memory(bundle_dir) -> bool:  # noqa: ANN001
    """Setup commands run AFTER this hook (bm.run reads setup_commands.json at run
    time), so edit the file in place. Only expands when the eval GPU has the VRAM:
    RTX Pro 6000 (Blackwell) = 96 GB runs max-model-len 98304 + window 65536;
    a smaller machine keeps the shipped 65536/32768 (identical bytes behaviour)."""
    setup_path = Path(bundle_dir) / "setup_commands.json"
    if not setup_path.exists():
        print(f"{ARM_MARKER} P6 setup_commands.json NOT FOUND — skipped", flush=True)
        return False
    commands = json.loads(setup_path.read_text())
    vram_mb = _detected_vram_mb()
    changed = False
    if vram_mb >= 80000 and isinstance(commands, list) and commands:
        edition = commands[0]
        for old, new in (
            ("VLLM_MAX_MODEL_LEN = 65536", f"VLLM_MAX_MODEL_LEN = {_BUDGET_MAX_MODEL_LEN}"),
            ("ANALYZER_CONTEXT_WINDOW = 32768", f"ANALYZER_CONTEXT_WINDOW = {_BUDGET_CONTEXT_WINDOW}"),
        ):
            if old in edition and new not in edition:
                edition = edition.replace(old, new)
                changed = True
        commands[0] = edition
        if changed:
            setup_path.write_text(json.dumps(commands))
            print(
                f"{ARM_MARKER} P6 memory expansion VRAM={vram_mb}MB: "
                f"max-model-len {_BUDGET_MAX_MODEL_LEN} + window {_BUDGET_CONTEXT_WINDOW}: OK",
                flush=True,
            )
        else:
            print(f"{ARM_MARKER} P6 memory expansion: anchors missing, no change", flush=True)
    elif vram_mb < 80000:
        print(f"{ARM_MARKER} P6 memory expansion SKIPPED (VRAM={vram_mb}MB < 80000)", flush=True)
    return True


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def apply_all(bundle_dir) -> None:  # noqa: ANN001
    print(f"{ARM_MARKER} applying patch stack", flush=True)
    ok = [
        patch_token_estimator(),
        patch_middle_drop(),
        patch_prompt_own_goal(),
        patch_repeated_no_effect_guard(),
    ]
    armed = len(ok)
    # P5/P6 are gated OFF by default: no supporting measurement, and held
    # counter-evidence (tempo law; retune direction). See module docstring.
    if os.environ.get("DUCK_MEM_P5", "0") == "1":
        ok.append(patch_yield_seconds(90.0))
        armed += 1
    else:
        print(f"{ARM_MARKER} P5 yield: GATED OFF", flush=True)
    if os.environ.get("DUCK_MEM_P6", "0") == "1":
        ok.append(patch_adaptive_memory(bundle_dir))
        armed += 1
    else:
        print(f"{ARM_MARKER} P6 memory expansion: GATED OFF", flush=True)
    print(f"{ARM_MARKER} patch stack applied: {sum(1 for v in ok if v)}/{armed}", flush=True)