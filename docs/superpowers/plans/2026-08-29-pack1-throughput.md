# Pack 1 — Throughput Graft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise actions-per-game of the shipped duck38-v12 harness ≥2.5× with plumbing-only changes (no prompt-semantics change), measured A/B at eval geometry on a Kaggle GPU commit before it takes a submission slot.

**Architecture:** One runtime graft module (`submission/_throughput_v1/graft_throughput.py`) monkeypatches five seams of the anim-bundle `ToolAgent` (context trimming, init-time budgets, note wipe on GAME_OVER, per-tool-call action cap) plus a pure time-guard helper for the notebook. Every behaviour is behind a call-time env flag so one kernel can run a stock phase and a graft phase in the same vLLM boot. A smoke-kernel builder (`submission/_tp_smoke/`) layers the graft onto the tracked v12 notebook with two 25-game phases and per-phase telemetry; a flight builder (`submission/_duck38_tp1/`) produces the scored notebook.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), `unittest`, the anim bundle at `submission/_inspect_replay/assets_build/ARC3-Inference` (importable as `inference.*`), Kaggle CLI (`kaggle kernels push --accelerator NvidiaRtxPro6000`).

**Evidence this plan rests on:** `docs/research-2026-08-29/R4-harness-throughput-audit.md` §0–§3 (prefix-hit 0–20%, prefill:decode 12–15:1, 60-s yield < 118-s call, 26% action-less wall, notes wiped on GAME_OVER, 20–140-action blind batches).

---

## Seams (anim bundle, `inference/agent/tool_agent.py`)

| seam | line | what the graft does |
|---|---|---|
| `ToolAgent._trim_messages_for_context` | 2056 | hysteresis: when over budget, drop oldest blocks until ≤ budget × `TP_TRIM_LOW_WATER` |
| `ToolAgent.__init__` | ~1040 | after stock init, override `_context_budget_tokens`, `_yield_seconds`, `_tool_steps` from `TP_*` env |
| `ToolAgent._update_summarized_knowledge_from_step_summary` | 1343 | keep notes on `game_over`; still wipe on `level_transition` / `run_complete` |
| `ToolAgent._run_python_tool` + `_normalize_python_actions` | 1693 / 1614 | per-tool-call action budget (`TP_BATCH_CAP`); the (N+1)th requested action raises `ValueError` inside the sandbox |
| notebook run cell | — | `time_guard_per_game_s()` shrinks `bm.solver.max_runtime_s_per_game` only if setup ate the margin |

Flags (all read at call time; `TP_ENABLE=0` makes every patch a pass-through):

| flag | default | meaning |
|---|---|---|
| `TP_ENABLE` | `1` | master switch |
| `TP_TRIM_LOW_WATER` | `0.5` | fraction of budget to drop to when over budget (`1.0` = stock behaviour) |
| `TP_CONTEXT_WINDOW` | `24576` | replaces `LOCAL_ANALYZER_CONTEXT_WINDOW` for the budget (`0` = leave stock) |
| `TP_YIELD_SECONDS` | `900` | replaces the 60-s yield (`-1` = leave stock) |
| `TP_TOOL_STEPS` | `8` | bounds calls per turn (`-1` = leave stock, `0` = unlimited) |
| `TP_KEEP_NOTES_ON_GAME_OVER` | `1` | `0` = stock wipe |
| `TP_BATCH_CAP` | `10` | requested actions per `python` tool call (`0` = unlimited) |

---

### Task 1: Graft skeleton, install(), flag helpers

**Files:**
- Create: `submission/_throughput_v1/graft_throughput.py`
- Create: `submission/_throughput_v1/test_graft_throughput.py`

- [ ] **Step 1: Write the failing test file**

```python
"""Offline validation for the throughput graft (no GPU/network/Kaggle).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_throughput.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_throughput as tp  # noqa: E402

FLAGS = ("TP_ENABLE", "TP_TRIM_LOW_WATER", "TP_CONTEXT_WINDOW", "TP_YIELD_SECONDS",
         "TP_TOOL_STEPS", "TP_KEEP_NOTES_ON_GAME_OVER", "TP_BATCH_CAP")


def _clear_flags() -> None:
    for name in FLAGS:
        os.environ.pop(name, None)


class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        cls.install_status = tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "throughput: OK")
        self.assertEqual(tp.install(), "throughput: SKIP (already applied)")

    def test_02_defaults(self) -> None:
        self.assertTrue(tp.enabled())
        self.assertEqual(tp.trim_low_water(), 0.5)
        self.assertEqual(tp.context_window(), 24576)
        self.assertEqual(tp.yield_seconds(), 900.0)
        self.assertEqual(tp.tool_steps(), 8)
        self.assertTrue(tp.keep_notes_on_game_over())
        self.assertEqual(tp.batch_cap(), 10)

    def test_03_master_switch_off(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        self.assertFalse(tp.enabled())
        self.assertEqual(tp.trim_low_water(), 1.0)      # stock
        self.assertEqual(tp.context_window(), 0)        # stock
        self.assertEqual(tp.yield_seconds(), -1.0)      # stock
        self.assertEqual(tp.tool_steps(), -1)           # stock
        self.assertFalse(tp.keep_notes_on_game_over())
        self.assertEqual(tp.batch_cap(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python submission/_throughput_v1/test_graft_throughput.py`
Expected: `ModuleNotFoundError: No module named 'graft_throughput'`

- [ ] **Step 3: Write the module skeleton**

```python
"""Throughput graft (Pack 1, 2026-08-29) — plumbing-only changes that raise
actions-per-game of the anim-bundle ToolAgent. No prompt text changes.

Evidence: docs/research-2026-08-29/R4-harness-throughput-audit.md.

Seams (anim bundle inference/agent/tool_agent.py):
- _trim_messages_for_context (:2056) — drop-one-block-per-turn invalidates
  the vLLM prefix cache every call (hit 0-20%). Hysteresis: when over budget,
  cut to TP_TRIM_LOW_WATER x budget in one go so the prefix survives turns.
- __init__ — _context_budget_tokens / _yield_seconds / _tool_steps are set
  from module constants read at import; override them per instance from
  TP_CONTEXT_WINDOW / TP_YIELD_SECONDS / TP_TOOL_STEPS.
- _update_summarized_knowledge_from_step_summary (:1343) — wipes the carried
  notes on game_over; keep them (TP_KEEP_NOTES_ON_GAME_OVER).
- _run_python_tool (:1693) + _normalize_python_actions (:1614) — per tool
  call action budget (TP_BATCH_CAP); the (cap+1)th requested action raises
  inside the sandbox, ending the batch with a readable error.

Fail-open: every patched seam checks its flag at call time and falls back to
stock behaviour on TP_ENABLE=0; graft logic errors fall through to stock.
"""
from __future__ import annotations

import os
import threading
from typing import Any

_tls = threading.local()
_STATE = {"installed": False}

DEFAULT_TRIM_LOW_WATER = 0.5
DEFAULT_CONTEXT_WINDOW = 24576
DEFAULT_YIELD_SECONDS = 900.0
DEFAULT_TOOL_STEPS = 8
DEFAULT_BATCH_CAP = 10
BATCH_CAP_MESSAGE = (
    "action batch cap reached: at most {cap} actions per python tool call. "
    "Observe the results so far, then call the tool again for more actions."
)


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP_ENABLE", "1").lower() not in {"0", "false", "no", "off"}


def trim_low_water() -> float:
    if not enabled():
        return 1.0
    try:
        value = float(_env("TP_TRIM_LOW_WATER", str(DEFAULT_TRIM_LOW_WATER)))
    except ValueError:
        return DEFAULT_TRIM_LOW_WATER
    return min(1.0, max(0.05, value))


def context_window() -> int:
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))))
    except ValueError:
        return DEFAULT_CONTEXT_WINDOW


def yield_seconds() -> float:
    if not enabled():
        return -1.0
    try:
        return float(_env("TP_YIELD_SECONDS", str(DEFAULT_YIELD_SECONDS)))
    except ValueError:
        return DEFAULT_YIELD_SECONDS


def tool_steps() -> int:
    if not enabled():
        return -1
    try:
        return int(_env("TP_TOOL_STEPS", str(DEFAULT_TOOL_STEPS)))
    except ValueError:
        return DEFAULT_TOOL_STEPS


def keep_notes_on_game_over() -> bool:
    if not enabled():
        return False
    return _env("TP_KEEP_NOTES_ON_GAME_OVER", "1").lower() not in {"0", "false", "no", "off"}


def batch_cap() -> int:
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP_BATCH_CAP", str(DEFAULT_BATCH_CAP))))
    except ValueError:
        return DEFAULT_BATCH_CAP


def install() -> str:
    if _STATE["installed"]:
        return "throughput: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"throughput: SKIP (tool_agent module missing: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "throughput: SKIP (missing ToolAgent)"
    _STATE["installed"] = True
    return "throughput: OK"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python submission/_throughput_v1/test_graft_throughput.py`
Expected: 3 tests OK.

- [ ] **Step 5: Commit**

```bash
git add submission/_throughput_v1
git commit -q -m "feat(throughput): graft skeleton with call-time flags"
```

---

### Task 2: Hysteresis trimming

**Files:**
- Modify: `submission/_throughput_v1/graft_throughput.py`
- Modify: `submission/_throughput_v1/test_graft_throughput.py`

- [ ] **Step 1: Add the failing tests**

```python
class _AgentMixin:
    def _agent(self):
        return self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")

    @staticmethod
    def _turn(i: int, chars: int = 600) -> list[dict]:
        return [
            {"role": "user", "content": f"u{i} " + ("x" * chars)},
            {"role": "assistant", "content": f"a{i} " + ("y" * chars), "tool_calls": [{"id": f"c{i}", "function": {"name": "python", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "t" * chars},
        ]


class TrimTests(_AgentMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def _messages(self, turns: int) -> list[dict]:
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(turns):
            msgs.extend(self._turn(i))
        return msgs

    def test_10_under_budget_is_untouched(self) -> None:
        agent = self._agent()
        msgs = self._messages(2)
        self.assertEqual(agent._trim_messages_for_context(msgs), msgs)

    def test_11_over_budget_cuts_to_low_water(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = self._messages(12)   # ~ 12 * 600 tokens
        out = agent._trim_messages_for_context(msgs)
        est = agent._estimate_request_input_tokens(out)
        self.assertLessEqual(est, 2000 + 700)          # <= low water + one block
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")
        self.assertEqual(out[-1], msgs[-1])

    def test_12_stable_prefix_across_turns(self) -> None:
        """After a cut, appending turns must not move the prefix until the
        budget is exceeded again."""
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = agent._trim_messages_for_context(self._messages(12))
        first_user = msgs[1]
        for i in range(12, 15):
            msgs = agent._trim_messages_for_context(msgs + self._turn(i))
            if agent._estimate_request_input_tokens(msgs) <= 4000:
                self.assertIs(msgs[1], first_user)

    def test_13_stock_when_disabled(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        agent = self._agent()
        agent._context_budget_tokens = 4000
        out = agent._trim_messages_for_context(self._messages(12))
        est = agent._estimate_request_input_tokens(out)
        self.assertGreater(est, 3000)   # stock stops right under budget
        self.assertLessEqual(est, 4000)

    def test_14_preserve_recent_and_extra_safety_honoured(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = self._messages(12)
        out = agent._trim_messages_for_context(msgs, preserve_recent=3, extra_safety_tokens=1000)
        self.assertLessEqual(agent._estimate_request_input_tokens(out), 3000)
        self.assertEqual(out[-3:], msgs[-3:])
```

- [ ] **Step 2: Run to verify they fail** — `test_11`, `test_12`, `test_14` FAIL (stock leaves the estimate just under budget).

- [ ] **Step 3: Implement**

Add to `graft_throughput.py`:

```python
def _patch_trim(agent_mod: Any, cls: Any) -> None:
    stock_trim = cls._trim_messages_for_context

    def trim(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        low = trim_low_water()
        if low >= 1.0 or not messages:
            return stock_trim(self, messages, tools=tools, preserve_recent=preserve_recent,
                              extra_safety_tokens=extra_safety_tokens)
        try:
            system_message = messages[0]
            history = list(messages[1:])
            preserve_recent = max(0, preserve_recent)
            budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
            estimate = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            if estimate <= budget:
                return [system_message, *self._drop_until_first_user_message(history)]
            target = max(1, int(budget * low))
            while history and estimate > target:
                if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                    break
                estimate = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            history = self._drop_until_first_user_message(history)
            return [system_message, *history]
        except Exception:  # noqa: BLE001 — fail open to stock
            return stock_trim(self, messages, tools=tools, preserve_recent=preserve_recent,
                              extra_safety_tokens=extra_safety_tokens)

    trim._tp_stock = stock_trim
    cls._trim_messages_for_context = trim
```

and call `_patch_trim(agent_mod, cls)` inside `install()` before setting `installed`.

Note `test_14`: stock drops until under budget; preserve_recent=3 keeps the last three messages — the assertion `out[-3:] == msgs[-3:]` holds for both.

- [ ] **Step 4: Run tests** — all OK.
- [ ] **Step 5: Commit** — `git commit -q -m "feat(throughput): hysteresis context trimming"`

---

### Task 3: Init-time budget overrides (window / yield / tool steps)

- [ ] **Step 1: Failing tests**

```python
class InitOverrideTests(_AgentMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def test_20_defaults_applied(self) -> None:
        agent = self._agent()
        self.assertEqual(agent._context_budget_tokens, 24576 - agent._reply_reserve_tokens - agent._request_safety_margin_tokens)
        self.assertEqual(agent._yield_seconds, 900.0)
        self.assertEqual(agent._tool_steps, 8)

    def test_21_stock_when_disabled(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        agent = self._agent()
        stock_window = self.agent_mod._LOCAL_ANALYZER_CONTEXT_WINDOW
        self.assertEqual(agent._context_budget_tokens, max(1024, stock_window - agent._reply_reserve_tokens - agent._request_safety_margin_tokens))
        self.assertEqual(agent._yield_seconds, None if self.agent_mod._LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(self.agent_mod._LOCAL_ANALYZER_YIELD_SECONDS))

    def test_22_zero_tool_steps_means_unlimited(self) -> None:
        os.environ["TP_TOOL_STEPS"] = "0"
        self.assertIsNone(self._agent()._tool_steps)

    def test_23_yield_zero_disables(self) -> None:
        os.environ["TP_YIELD_SECONDS"] = "0"
        self.assertIsNone(self._agent()._yield_seconds)
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**

```python
def _patch_init(cls: Any) -> None:
    stock_init = cls.__init__

    def init(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        try:
            window = context_window()
            if window > 0:
                self._context_budget_tokens = max(
                    1024, window - self._reply_reserve_tokens - self._request_safety_margin_tokens)
            ys = yield_seconds()
            if ys >= 0:
                self._yield_seconds = None if ys == 0 else float(ys)
            ts = tool_steps()
            if ts >= 0:
                self._tool_steps = None if ts == 0 else max(1, ts)
        except Exception:  # noqa: BLE001
            pass

    init._tp_stock = stock_init
    cls.__init__ = init
```

Call from `install()`.

- [ ] **Step 4: Run — OK.**  **Step 5: Commit** — `feat(throughput): per-instance window/yield/tool-step overrides`

---

### Task 4: Keep notes across GAME_OVER

- [ ] **Step 1: Failing tests**

```python
class NotesTests(_AgentMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def _agent_with_notes(self):
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "walls block moves"
        agent._summarized_knowledge["current_plan"] = "go right"
        agent._summarized_knowledge["cross_level_notes"] = "keep"
        return agent

    def test_30_game_over_keeps_notes(self) -> None:
        agent = self._agent_with_notes()
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "walls block moves")
        self.assertEqual(agent._summarized_knowledge["current_plan"], "go right")

    def test_31_level_transition_still_wipes(self) -> None:
        agent = self._agent_with_notes()
        agent._last_step_summary = {"level_transition": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "keep")

    def test_32_stock_when_disabled(self) -> None:
        os.environ["TP_KEEP_NOTES_ON_GAME_OVER"] = "0"
        agent = self._agent_with_notes()
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
```

- [ ] **Step 2: Run — `test_30` FAILS.**
- [ ] **Step 3: Implement**

```python
def _patch_notes(cls: Any) -> None:
    stock = cls._update_summarized_knowledge_from_step_summary

    def update(self):
        if not keep_notes_on_game_over():
            return stock(self)
        try:
            summary = self._last_step_summary
            if not summary:
                return None
            if summary.get("level_transition") or summary.get("run_complete"):
                return stock(self)
            return None   # game_over (or nothing): keep every carried note
        except Exception:  # noqa: BLE001
            return stock(self)

    update._tp_stock = stock
    cls._update_summarized_knowledge_from_step_summary = update
```

- [ ] **Step 4: Run — OK.**  **Step 5: Commit** — `feat(throughput): keep carried notes across GAME_OVER`

---

### Task 5: Per-tool-call action cap

- [ ] **Step 1: Failing tests**

```python
class BatchCapTests(_AgentMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def test_40_cap_truncates_within_one_tool_call(self) -> None:
        agent = self._agent()
        tp.begin_tool_call()
        out = agent._normalize_python_actions(["UP"] * 25)
        self.assertEqual(len(out), 10)
        with self.assertRaises(ValueError) as ctx:
            agent._normalize_python_actions(["UP"])
        self.assertIn("action batch cap reached", str(ctx.exception))

    def test_41_budget_resets_per_tool_call(self) -> None:
        agent = self._agent()
        tp.begin_tool_call()
        agent._normalize_python_actions(["UP"] * 10)
        tp.begin_tool_call()
        self.assertEqual(len(agent._normalize_python_actions(["DOWN"] * 3)), 3)

    def test_42_unlimited_when_disabled(self) -> None:
        os.environ["TP_BATCH_CAP"] = "0"
        agent = self._agent()
        tp.begin_tool_call()
        self.assertEqual(len(agent._normalize_python_actions(["UP"] * 25)), 25)

    def test_43_run_python_tool_resets_budget(self) -> None:
        """_run_python_tool is the seam that calls begin_tool_call; drive it
        with an empty code string (early return) and check the budget reset."""
        agent = self._agent()
        tp.begin_tool_call()
        agent._normalize_python_actions(["UP"] * 10)
        agent._run_python_tool(Path("/nonexistent/state.json"), {"code": ""})
        self.assertEqual(len(agent._normalize_python_actions(["UP"] * 2)), 2)
```

- [ ] **Step 2: Run — FAIL (`begin_tool_call` missing).**
- [ ] **Step 3: Implement**

```python
def begin_tool_call() -> None:
    _tls.remaining = batch_cap()


def _remaining() -> int | None:
    cap = batch_cap()
    if cap <= 0:
        return None
    remaining = getattr(_tls, "remaining", None)
    if remaining is None:
        remaining = cap
        _tls.remaining = remaining
    return remaining


def _patch_batch_cap(cls: Any) -> None:
    stock_normalize = cls._normalize_python_actions
    stock_run = cls._run_python_tool

    def normalize(self, value):
        normalized = stock_normalize(self, value)
        remaining = _remaining()
        if remaining is None:
            return normalized
        if remaining <= 0:
            raise ValueError(BATCH_CAP_MESSAGE.format(cap=batch_cap()))
        if len(normalized) > remaining:
            normalized = normalized[:remaining]
        _tls.remaining = remaining - len(normalized)
        return normalized

    def run(self, state_path, arguments):
        begin_tool_call()
        return stock_run(self, state_path, arguments)

    normalize._tp_stock = stock_normalize
    run._tp_stock = stock_run
    cls._normalize_python_actions = normalize
    cls._run_python_tool = run
```

Note: `_run_python_tool(..., {"code": ""})` returns before touching `state_path`, so the test needs no runtime state file. The ValueError propagates through `_handle_action` → sandbox → the model sees it as the tool error.

- [ ] **Step 4: Run — OK.**  **Step 5: Commit** — `feat(throughput): per-tool-call action cap`

---

### Task 6: Time guard helper

- [ ] **Step 1: Failing tests**

```python
class TimeGuardTests(unittest.TestCase):
    def test_50_no_change_when_margin_holds(self) -> None:
        self.assertEqual(tp.time_guard_per_game_s(7920.0, setup_elapsed_s=630.0, games=110, concurrency=28), 7920.0)

    def test_51_shrinks_when_setup_ate_margin(self) -> None:
        got = tp.time_guard_per_game_s(7920.0, setup_elapsed_s=1200.0, games=110, concurrency=28)
        self.assertLess(got, 7920.0)
        self.assertAlmostEqual(got, (32400.0 - 1200.0 - 240.0) / 4, places=1)

    def test_52_never_grows(self) -> None:
        self.assertEqual(tp.time_guard_per_game_s(3600.0, setup_elapsed_s=0.0, games=4, concurrency=28), 3600.0)
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**

```python
def time_guard_per_game_s(stock_per_game_s: float, *, setup_elapsed_s: float, games: int,
                          concurrency: int, total_budget_s: float = 32400.0,
                          margin_s: float = 240.0) -> float:
    """Shrink the per-game box only if setup + waves would overrun the 9 h box."""
    try:
        waves = max(1, -(-int(games) // max(1, int(concurrency))))
        fits = (total_budget_s - setup_elapsed_s - margin_s) / waves
        return float(min(stock_per_game_s, max(600.0, fits)))
    except Exception:  # noqa: BLE001
        return float(stock_per_game_s)
```

- [ ] **Step 4: Run — OK.**  **Step 5: Commit** — `feat(throughput): time guard helper`

---

### Task 7: A/B smoke kernel builder (stock vs graft, 25 games, eval geometry)

**Files:**
- Create: `submission/_tp_smoke/build_tp_smoke.py`
- Create: `submission/_tp_smoke/kernel-metadata.json` (copy of `_v8_smoke/kernel-metadata.json` with id/title `arc3-tp-smoke`, code_file `arc3-tp-smoke.ipynb`)
- Create: `submission/_tp_smoke/validate_tp_smoke.py`
- Create: `submission/_tp_smoke/fetch_results.py` (copy of `_v8_smoke/fetch_results.py` with `TP_KERNEL_SLUG`/`arc3-tp-smoke`)

Design (mirrors `build_v8_smoke.py`): load `submission/_duck38_v12/arc3-duck38-v12.ipynb`; insert the GPU-assert cell after the imports cell; replace the smoke cell with a two-phase version; insert the graft cell (embeds `graft_throughput.py`, calls `install()`, hard-asserts `"throughput: OK"`); insert a telemetry cell; replace the run block with the phase loop that sets `TP_ENABLE` per phase and scrapes vLLM `/metrics` at each phase boundary; append a report cell.

Phases:
```python
SMOKE_PHASES = [
    ("stock", GAMES_25, 7920, {"TP_ENABLE": "0"}),
    ("tp",    GAMES_25, 7920, {"TP_ENABLE": "1"}),
]
```
`GAMES_25` = the 25 env_name strings from `environment_files/*/metadata.json` (`<id>-<hash>` folder names). `soft_end` = start + 5.2 h. Kernel budget ≈ 10 min boot + 2 × 2.2 h.

Telemetry per phase (written to `tp_smoke_results.json`): for each `bm.game_runs` entry: `game_id, levels, actions (len(history)), final_score, actions_per_level, state, solver_note`; `turns = max analysis_step` (from the session seam already used by v8 telemetry: wrap `_HarnessGameSession.play` to record `analysis_step` at exit); vLLM `/metrics` counters before/after each phase: `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`, `vllm:request_success_total`; phase wall seconds. Report cell prints a two-row table: phase, games, mean actions/game, mean levels/game, mean score, prefill:gen ratio, prefix hit %, requests, gen tok/s.

- [ ] **Step 1: Write `validate_tp_smoke.py`** asserting: notebook has the graft cell with `install()` + assert, `SMOKE_PHASES` two entries with `TP_ENABLE` 0/1, `run_as_submission` branch untouched, metrics scrape present.
- [ ] **Step 2: Write the builder** (code in the executing session; follow `build_v8_smoke.py` structure).
- [ ] **Step 3: Build + validate**: `.venv/bin/python submission/_tp_smoke/build_tp_smoke.py && .venv/bin/python submission/_tp_smoke/validate_tp_smoke.py` → `OK`.
- [ ] **Step 4: Dry-import test**: `.venv/bin/python -c "import json;nb=json.load(open('submission/_tp_smoke/arc3-tp-smoke.ipynb'));[compile(''.join(c['source']).replace('await ','_=') ,'nb','exec') for c in nb['cells'] if c['cell_type']=='code']"` (syntax check per cell; `await` at top level is replaced for the compile check only).
- [ ] **Step 5: Commit** — `feat(tp-smoke): A/B smoke kernel builder`

---

### Task 8: Push, monitor, read

- [ ] **Step 1: Push**: `cd submission/_tp_smoke && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000`
- [ ] **Step 2: Monitor** with a background loop on `kaggle kernels status ahmedmobasher86/arc3-tp-smoke` every 15 min (ScheduleWakeup / Monitor; expect ~5 h).
- [ ] **Step 3: Fetch**: `python3 submission/_tp_smoke/fetch_results.py tp_smoke_results.json stdout.log`
- [ ] **Step 4: Read** (pre-registered): PASS = phase `tp` mean actions/game ≥ 2.0× `stock` AND mean levels/game ≥ stock − 0.15 AND prefix-hit ≥ 50%. INCONCLUSIVE = actions ≥ 1.5× with levels within noise. FAIL = levels/game < stock − 0.15 or actions < 1.5×. Write the read to `submission/_tp_smoke/results/READ.md`.

---

### Task 9: Flight notebook (only after a PASS)

**Files:** `submission/_duck38_tp1/build_duck38_tp1.py`, `kernel-metadata.json` (id `arc3-duck38-tp1`), `validate_duck38_tp1.py`

- v12 base + ONE inserted cell before the run cell: embeds `graft_throughput.py`, sets every `TP_*` flag explicitly, `install()` behind `assert "throughput: OK"`, purges `EFFORT_*`/`EXPLORER*` env, applies `time_guard_per_game_s` to `bm.solver.max_runtime_s_per_game` using `time.time() - NOTEBOOK_START_EPOCH`.
- Build, validate (12 cells, hash printed), push a COMMIT (`--accelerator NvidiaRtxPro6000`), wait COMPLETE, record svid.
- **STOP before `kaggle competitions submit`** — submission is Ahmed's call (push gate).
