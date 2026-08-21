"""Offline validation for the yield_carryover graft (no GPU/network/Kaggle).

Part A — unit tests of the digest builder on synthetic slice conversations.
Part B — integration against the June stock bundle: install() onto the real
``ToolAgent`` seams, drive real ``analyze`` calls with a stubbed
``_chat_completion`` (real sandbox, real yield machinery via
``_yield_seconds``), and assert digest carryover, single injection, the
slice-3 cap, flags-off byte-identity, and fail-open.

Run:  .venv/bin/python submission/_yield_carryover/test_yield_carryover.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(
    os.environ.get(
        "GRAFT_TEST_BUNDLE",
        str(_HERE.parents[1] / "scratchpad/bundles/june_stock/src/ARC3-Inference"),
    )
)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_carryover
from graft_carryover import (
    DIGEST_HEADER,
    FINAL_SLICE_INSTRUCTION,
    build_slice_digest,
    install,
)


def _clear_flags() -> None:
    os.environ.pop("YIELD_CARRYOVER", None)
    os.environ.pop("YIELD_SLICE_CAP", None)


def _tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }


class UnitDigestBuilder(unittest.TestCase):
    HEAD = "The slice prompt starts here"

    def _messages(self) -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": f"{self.HEAD} and continues with state."},
            {
                "role": "assistant",
                "reasoning": "first thoughts " + "x" * 700,
                "tool_calls": [_tool_call("c1", "print('probe-alpha')")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"stdout": "probe-alpha ran"}'},
            {
                "role": "assistant",
                "reasoning": "final thoughts " + "y" * 700,
                "tool_calls": [_tool_call("c2", "z = 2\n" + "#pad\n" * 100)],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "R" * 900},
        ]

    def test_digest_contains_code_results_and_reasoning_tail(self) -> None:
        digest = build_slice_digest(self._messages(), self.HEAD)
        self.assertIn(DIGEST_HEADER, digest)
        self.assertIn("probe-alpha", digest)  # code of call 1
        self.assertIn("probe-alpha ran", digest)  # result of call 1
        self.assertIn("[1] python:", digest)
        self.assertIn("[2] python:", digest)
        self.assertIn("Last reasoning tail:", digest)
        # tail comes from the LAST assistant reasoning, bounded to 500 chars
        tail = digest.split("Last reasoning tail: ", 1)[1].splitlines()[0]
        self.assertLessEqual(len(tail), 500)
        self.assertTrue(tail.endswith("y" * 40))
        self.assertNotIn("first thoughts", digest)

    def test_bounds_code_200_result_400_total_6000(self) -> None:
        digest = build_slice_digest(self._messages(), self.HEAD)
        for line in digest.splitlines():
            if line.startswith("[") and "python:" in line:
                code = line.split("python: ", 1)[1]
                self.assertLessEqual(len(code), 200 + 20)  # +truncation marker
            if line.strip().startswith("result:"):
                self.assertLessEqual(len(line), 400 + 30)
        self.assertLessEqual(len(digest), 6000)
        # messages before the boundary never leak in
        self.assertNotIn("old turn", digest)
        self.assertNotIn("old answer", digest)

    def test_empty_slice_or_missing_marker_yields_empty(self) -> None:
        messages = self._messages()[:4]  # nothing after the slice prompt
        self.assertEqual(build_slice_digest(messages, self.HEAD), "")
        self.assertEqual(build_slice_digest(self._messages(), "no-such-marker"), "")
        self.assertEqual(build_slice_digest(None, self.HEAD), "")
        self.assertEqual(build_slice_digest(self._messages(), ""), "")

    def test_parts_list_user_content_matches_marker(self) -> None:
        messages = self._messages()
        messages[3] = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{self.HEAD} multimodal variant"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        }
        digest = build_slice_digest(messages, self.HEAD)
        self.assertIn("probe-alpha", digest)


class IntegrationJuneBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: PLC0415

        cls.agent_mod = agent_mod
        cls.Frame = Frame
        cls.HistoryEntry = HistoryEntry
        cls.write_runtime_state = staticmethod(write_runtime_state)
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    def _state_path(self) -> Path:
        import tempfile  # noqa: PLC0415

        tmp = Path(tempfile.mkdtemp(prefix="yc_test_"))
        state_path = tmp / "tool_runtime_state.json"
        frame = self.Frame(grid=((0, 1), (1, 0)), step=0, level=1)
        self.write_runtime_state(state_path, current_frame=frame, history=[self.HistoryEntry(action="", frame=frame)])
        return state_path

    def _make_agent(self, script, *, yield_after_s: float = 0.05, sleep_s: float = 0.08):
        """Real ToolAgent; only the network call is stubbed. Each scripted
        completion sleeps past the yield budget so the slice ends yielded."""
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        agent._yield_seconds = yield_after_s
        requests_log: list[list[dict]] = []
        result_cls = self.agent_mod._ChatCompletionResult

        def fake_chat(messages, *, tools=None, request_timeout_seconds=None):
            requests_log.append(json.loads(json.dumps(messages)))
            time.sleep(sleep_s)
            message = script.pop(0)
            return result_cls(message=message, finish_reason="tool_calls", usage=None)

        agent._chat_completion = fake_chat  # instance attr; class analyze stays patched
        return agent, requests_log

    @staticmethod
    def _texts(messages: list[dict]) -> str:
        chunks = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.extend(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return "\n".join(chunks)

    def _run_slice(self, agent, state_path, analysis_step):
        return agent.analyze(
            state_path,
            0,
            valid_actions=["ACTION1"],
            step_env=None,
            analysis_step=analysis_step,
            request_timeout_seconds=30.0,
            should_stop=lambda: False,
        )

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "yield_carryover: OK")
        self.assertEqual(install(), "yield_carryover: SKIP (already applied)")

    def test_02_resume_injects_digest_exactly_once_and_cap_at_slice_3(self) -> None:
        state_path = self._state_path()
        script = [
            {"reasoning": "slice one thinking tail-ALPHA", "tool_calls": [_tool_call("c1", "print('marker-snippet-71')")]},
            {"reasoning": "slice two thinking", "tool_calls": [_tool_call("c2", "print('second-probe')")]},
            {"reasoning": "slice three thinking", "tool_calls": [_tool_call("c3", "print('third-probe')")]},
        ]
        agent, requests_log = self._make_agent(script)

        r1 = self._run_slice(agent, state_path, analysis_step=7)
        self.assertTrue(r1.yielded_control)
        self.assertFalse(r1.step_executed)
        self.assertNotIn(DIGEST_HEADER, self._texts(requests_log[0]))  # slice 1: no injection

        r2 = self._run_slice(agent, state_path, analysis_step=7)  # resume -> slice 2
        self.assertTrue(r2.yielded_control)
        slice2_text = self._texts(requests_log[1])
        self.assertEqual(slice2_text.count(DIGEST_HEADER), 1)  # injected exactly once
        self.assertIn("marker-snippet-71", slice2_text)  # slice-1 code carried
        self.assertIn("tail-ALPHA", slice2_text)  # slice-1 reasoning tail carried
        self.assertNotIn(FINAL_SLICE_INSTRUCTION, slice2_text)  # cap not yet

        r3 = self._run_slice(agent, state_path, analysis_step=7)  # resume -> slice 3
        self.assertTrue(r3.yielded_control)
        slice3_text = self._texts(requests_log[2])
        self.assertEqual(slice3_text.count(FINAL_SLICE_INSTRUCTION), 1)  # cap fires at slice 3
        self.assertEqual(slice3_text.count(DIGEST_HEADER), 1)
        self.assertIn("second-probe", slice3_text)  # digest now from slice 2

    def test_03_new_turn_resets_state_no_injection(self) -> None:
        state_path = self._state_path()
        script = [
            {"reasoning": "t1", "tool_calls": [_tool_call("c1", "print('turn-a-probe')")]},
            {"reasoning": "t2", "tool_calls": [_tool_call("c2", "print('turn-b-probe')")]},
        ]
        agent, requests_log = self._make_agent(script)
        self._run_slice(agent, state_path, analysis_step=1)
        self._run_slice(agent, state_path, analysis_step=2)  # NEW analysis_step = new turn
        text = self._texts(requests_log[1])
        self.assertNotIn(DIGEST_HEADER, text)
        self.assertNotIn(FINAL_SLICE_INSTRUCTION, text)

    def test_04_flags_off_bytes_identical_prompt_and_passthrough(self) -> None:
        os.environ["YIELD_CARRYOVER"] = "0"
        os.environ["YIELD_SLICE_CAP"] = "0"
        try:
            state_path = self._state_path()
            script = [
                {"reasoning": "t1", "tool_calls": [_tool_call("c1", "print('off-probe')")]},
                {"reasoning": "t2", "tool_calls": [_tool_call("c2", "print('off-probe-2')")]},
            ]
            agent, requests_log = self._make_agent(script)
            self._run_slice(agent, state_path, analysis_step=3)
            self._run_slice(agent, state_path, analysis_step=3)  # resume
            self.assertNotIn(DIGEST_HEADER, self._texts(requests_log[1]))
            self.assertNotIn(FINAL_SLICE_INSTRUCTION, self._texts(requests_log[1]))
            # patched _build_user_prompt output is byte-identical to stock
            frame = self.Frame(grid=((0, 1), (1, 0)), step=0, level=1)
            original = install.originals["_build_user_prompt"]
            patched_out = agent._build_user_prompt(0, valid_actions=["ACTION1"], current_frame=frame)
            stock_out = original(agent, 0, valid_actions=["ACTION1"], current_frame=frame)
            self.assertEqual(patched_out, stock_out)
        finally:
            _clear_flags()

    def test_05_fail_open_digest_crash_still_completes(self) -> None:
        state_path = self._state_path()
        script = [
            {"reasoning": "t1", "tool_calls": [_tool_call("c1", "print('crash-probe')")]},
            {"reasoning": "t2", "tool_calls": [_tool_call("c2", "print('crash-probe-2')")]},
        ]
        agent, requests_log = self._make_agent(script)
        with mock.patch.object(graft_carryover, "build_slice_digest", side_effect=RuntimeError("boom")):
            r1 = self._run_slice(agent, state_path, analysis_step=9)
            r2 = self._run_slice(agent, state_path, analysis_step=9)
        self.assertTrue(r1.yielded_control)
        self.assertTrue(r2.yielded_control)
        self.assertNotIn(DIGEST_HEADER, self._texts(requests_log[1]))  # no digest, no crash

    def test_06_empty_yield_slice_keeps_previous_digest(self) -> None:
        state_path = self._state_path()
        script = [
            {"reasoning": "t1", "tool_calls": [_tool_call("c1", "print('keeper-probe')")]},
            {"reasoning": "t2", "tool_calls": [_tool_call("c2", "print('slice2')")]},
            {"reasoning": "t3", "tool_calls": [_tool_call("c3", "print('slice3')")]},
        ]
        agent, requests_log = self._make_agent(script)
        self._run_slice(agent, state_path, analysis_step=4)  # slice 1 builds digest
        # slice 2: yield fires before ANY request (empty slice) — shrink budget
        agent._yield_seconds = 0.0000001
        r2 = self._run_slice(agent, state_path, analysis_step=4)
        self.assertTrue(r2.yielded_control)
        agent._yield_seconds = 0.05
        self._run_slice(agent, state_path, analysis_step=4)  # slice 3: digest still slice-1's
        self.assertIn("keeper-probe", self._texts(requests_log[-1]))

    def test_07_digest_token_cost_bounded(self) -> None:
        # worst-case digest stays within the priced envelope (~1.5k tokens at len/4)
        big = [
            {"role": "user", "content": "HEADMARK padded prompt"},
        ]
        for i in range(20):
            big.append({"role": "assistant", "reasoning": "r" * 1000, "tool_calls": [_tool_call(f"c{i}", "c" * 1000)]})
            big.append({"role": "tool", "tool_call_id": f"c{i}", "content": "o" * 2000})
        digest = build_slice_digest(big, "HEADMARK")
        self.assertLessEqual(len(digest), 6000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
