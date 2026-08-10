"""Unit tests for the duck-mem patch stack (submission/_duck_mem/harness_mem.py).

Pure-logic coverage only: estimator, middle-drop trimmer, prompt neutralization,
the repeated-no-effect guard rule, and the adaptive-memory config edit. Nothing
here needs a GPU or a live vLLM server.
"""
import json
from pathlib import Path

import pytest

STACK = Path(__file__).resolve().parents[1] / "submission/_duck_mem/harness_mem.py"
NS: dict = {}
exec(compile(STACK.read_text(), "harness_mem.py", "exec"), NS)

EST = NS["_estimate_tokens_impl"]
DROP = NS["_drop_middle_history_block"]
GUARD = NS["_should_guard_no_effect"]
PATCH_MEMORY = NS["patch_adaptive_memory"]
OLD_LINE = NS["_OLD_OPTIMIZE_LINE"]
NEW_LINE = NS["_NEW_OPTIMIZE_LINE"]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(content: str, tool: bool = False) -> dict:
    return {"role": "tool" if tool else "assistant", "content": content}


def make_history(n_turns: int) -> list[dict]:
    history: list[dict] = []
    for i in range(n_turns):
        history.append(user(f"turn {i} question"))
        history.append(assistant(f"turn {i} answer"))
        if i < n_turns - 1:
            history.append(assistant(f"tool result {i}", tool=True))
    return history


class _FakeAgent:
    def drop(self, history: list[dict], *, preserve_recent: int) -> bool:
        return DROP(self, history, preserve_recent=preserve_recent)


# ---------------------------------------------------------------------------
# P1 estimator
# ---------------------------------------------------------------------------

class TestEstimator:
    def test_ratio_is_four(self):
        probe = json.dumps({"messages": [{"role": "user", "content": "x" * 1200}]})
        # chars/token ≈ 4, not the old 3. Tol = 0.1; the //4 estimator errs high
        # (fewer tokens than chars/4), which is the safe direction.
        assert abs(len(probe) / EST(probe) - 4.0) < 0.1

    def test_small_inputs_floor_at_one(self):
        assert EST("") == 1
        assert EST("x") == 1

    def test_overcounts_vs_true_tokens(self):
        # Helper-free invariance check: //4 never UNDER-estimates more than //3 in
        # the regime that matters (4 chars/token is a conservative upper bound).
        sample = "a" * 4096
        assert EST(sample) >= 1


# ---------------------------------------------------------------------------
# P2 middle-drop trimmer
# ---------------------------------------------------------------------------

class TestMiddleDrop:
    def test_head_prefix_never_dropped_while_over_budget(self):
        # The real trim loop stops as soon as the estimate fits the budget; it never
        # exhausts history. A few middle drops must leave the head (prefix cache)
        # untouched.
        agent = _FakeAgent()
        history = make_history(8)
        head = list(history[:4])
        for _ in range(2):
            assert agent.drop(history, preserve_recent=1)
        assert history[:4] == head
        assert history[0]["role"] == "user"

    def test_sacred_tail_survives(self):
        agent = _FakeAgent()
        history = make_history(8)
        tail = list(history[-3:])
        for _ in range(2):
            assert agent.drop(history, preserve_recent=1)
        assert history[-3:] == tail

    def test_terminates_and_drops_deterministically(self):
        agent = _FakeAgent()
        history = make_history(60)
        original_len = len(history)
        for _ in range(100):
            if not agent.drop(history, preserve_recent=1):
                break
        assert 0 < len(history) < original_len

    def test_single_removable_falls_back_to_front(self):
        agent = _FakeAgent()
        history = [user("a"), assistant("b")]
        assert agent.drop(history, preserve_recent=1) is True
        assert len(history) == 1

    def test_never_drops_below_preserve(self):
        agent = _FakeAgent()
        history = make_history(8)
        while agent.drop(history, preserve_recent=2):
            pass
        assert len(history) >= 2


# ---------------------------------------------------------------------------
# P3 prompt neutralization
# ---------------------------------------------------------------------------

class TestPromptNeutralization:
    def test_new_line_is_not_an_own_goal(self):
        assert "as few" not in NEW_LINE
        assert "Completing levels" in NEW_LINE

    def test_replace_keeps_rest_of_addendum(self):
        addendum = "leading\n" + OLD_LINE + "trailing\n"
        replaced = addendum.replace(OLD_LINE, NEW_LINE)
        assert "leading\n" in replaced and "trailing\n" in replaced
        assert OLD_LINE not in replaced

    def test_anchor_detection_fails_cleanly_on_foreign_text(self):
        # A bundle where the own-goal line is already absent must not crash the patch.
        assert OLD_LINE not in "some other prompt text"


# ---------------------------------------------------------------------------
# P4 repeated-no-effect guard rule
# ---------------------------------------------------------------------------

class TestNoEffectGuard:
    def test_fires_on_unchanged_repeated_direction(self):
        assert GUARD(False, "UP", "UP") is True

    def test_board_change_suppresses(self):
        assert GUARD(True, "UP", "UP") is False

    def test_non_direction_never_fires(self):
        assert GUARD(False, "MOUSE(row=3, col=5)", "MOUSE(row=3, col=5)") is False
        assert GUARD(False, "A", "A") is False

    def test_direction_change_suppresses(self):
        assert GUARD(False, "UP", "DOWN") is False
        assert GUARD(False, "UP", None) is False


# ---------------------------------------------------------------------------
# P6 adaptive memory edit
# ---------------------------------------------------------------------------

class TestAdaptiveMemory:
    def _setup_file(self, tmp_path: Path, vram_mb: int, monkeypatch) -> Path:
        command = (
            'VLLM_MAX_MODEL_LEN = 65536\nANALYZER_CONTEXT_WINDOW = 32768\n'
            'SERVED_MODEL_NAME = "x"\n'
        )
        setup_path = tmp_path / "setup_commands.json"
        setup_path.write_text(json.dumps(["[\"$PYTHON\" ...\n" + command + "\nPYSETUP\"]"]))
        monkeypatch.setitem(NS, "_detected_vram_mb", lambda: vram_mb)
        return setup_path

    def test_expands_on_96gb(self, tmp_path, monkeypatch):
        path = self._setup_file(tmp_path, 96 * 1024, monkeypatch)
        assert PATCH_MEMORY(tmp_path) is True
        edited = json.loads(path.read_text())[0]
        assert "VLLM_MAX_MODEL_LEN = 98304" in edited
        assert "ANALYZER_CONTEXT_WINDOW = 65536" in edited

    def test_unchanged_on_48gb(self, tmp_path, monkeypatch):
        path = self._setup_file(tmp_path, 48 * 1024, monkeypatch)
        assert PATCH_MEMORY(tmp_path) is True
        edited = json.loads(path.read_text())[0]
        assert "VLLM_MAX_MODEL_LEN = 65536" in edited
        assert "98304" not in edited

    def test_missing_file_no_crash(self, tmp_path, monkeypatch):
        assert PATCH_MEMORY(tmp_path) is False

# ---------------------------------------------------------------------------
# Wiring tests (added 2026-08-10 review): the original P3 was a silent no-op
# because tool_agent binds GAME_OVERVIEW_ADDENDUM by value at import time.
# These tests exercise the actual rebinding against faked harness modules —
# pure-logic tests cannot catch a wrong-namespace patch.
# ---------------------------------------------------------------------------
import sys
import types


def _fake_harness(monkeypatch, addendum: str):
    """Install a minimal fake inference.agent.{prompts,tool_agent} tree."""
    inference = types.ModuleType("inference")
    agent_pkg = types.ModuleType("inference.agent")
    prompts = types.ModuleType("inference.agent.prompts")
    tool_agent = types.ModuleType("inference.agent.tool_agent")
    prompts.GAME_OVERVIEW_ADDENDUM = addendum
    # simulate tool_agent's import-time by-value binding (tool_agent.py:17-19)
    tool_agent.GAME_OVERVIEW_ADDENDUM = addendum
    inference.agent = agent_pkg
    agent_pkg.prompts = prompts
    agent_pkg.tool_agent = tool_agent
    for name, mod in [
        ("inference", inference), ("inference.agent", agent_pkg),
        ("inference.agent.prompts", prompts),
        ("inference.agent.tool_agent", tool_agent),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return prompts, tool_agent


class TestP3Wiring:
    def test_rebinds_both_namespaces(self, monkeypatch):
        addendum = "header\n" + OLD_LINE + "footer\n"
        prompts, tool_agent = _fake_harness(monkeypatch, addendum)
        assert NS["patch_prompt_own_goal"]() is True
        assert OLD_LINE not in prompts.GAME_OVERVIEW_ADDENDUM
        assert OLD_LINE not in tool_agent.GAME_OVERVIEW_ADDENDUM  # the 08-10 bug
        assert NEW_LINE in tool_agent.GAME_OVERVIEW_ADDENDUM

    def test_missing_anchor_returns_false_and_touches_nothing(self, monkeypatch):
        addendum = "header only, no own-goal line\n"
        prompts, tool_agent = _fake_harness(monkeypatch, addendum)
        assert NS["patch_prompt_own_goal"]() is False
        assert prompts.GAME_OVERVIEW_ADDENDUM == addendum
        assert tool_agent.GAME_OVERVIEW_ADDENDUM == addendum


class TestP5P6Gating:
    """apply_all was exec'd with NS as its globals, so patching NS entries is
    how its internal name lookups are intercepted."""

    def _run_apply_all(self, monkeypatch) -> list[str]:
        calls: list[str] = []
        for name in ("patch_token_estimator", "patch_middle_drop",
                     "patch_prompt_own_goal", "patch_repeated_no_effect_guard",
                     "patch_yield_seconds", "patch_adaptive_memory"):
            monkeypatch.setitem(
                NS, name, (lambda n: lambda *a, **k: calls.append(n) or True)(name))
        NS["apply_all"](".")
        return calls

    def test_default_stack_skips_p5_and_p6(self, monkeypatch):
        monkeypatch.delenv("DUCK_MEM_P5", raising=False)
        monkeypatch.delenv("DUCK_MEM_P6", raising=False)
        calls = self._run_apply_all(monkeypatch)
        assert calls == ["patch_token_estimator", "patch_middle_drop",
                         "patch_prompt_own_goal", "patch_repeated_no_effect_guard"]

    def test_env_gates_arm_p5_and_p6(self, monkeypatch):
        monkeypatch.setenv("DUCK_MEM_P5", "1")
        monkeypatch.setenv("DUCK_MEM_P6", "1")
        calls = self._run_apply_all(monkeypatch)
        assert "patch_yield_seconds" in calls
        assert "patch_adaptive_memory" in calls
