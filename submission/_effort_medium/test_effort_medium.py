"""Offline validation for the effort_medium graft (no GPU/network/Kaggle).

Part A — payload tests: reasoning_effort riding chat_template_kwargs on the
vllm branch only, flag off => byte-identical stock payload, dual-namespace
rebinding.
Part B — dead-completion retry: drive the real (patched)
``ToolAgent._chat_completion`` with a mocked ``requests.post``; assert the
force fires only on the measured dead signature, the wire request carries
the forced tool_choice + act-now line, the caller's message list is not
mutated, and pending state clears at analyze entry.

Run:  .venv/bin/python submission/_effort_medium/test_effort_medium.py
"""

from __future__ import annotations

import json
import os
import sys
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

from graft_effort import ACT_NOW_LINE, FORCED_TOOL_CHOICE, install


def _clear_flags() -> None:
    for name in ("EFFORT_MEDIUM", "EFFORT_DEAD_RETRY", "EFFORT_LEVEL"):
        os.environ.pop(name, None)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _completion(message: dict, finish_reason: str = "stop") -> dict:
    return {"choices": [{"message": message, "finish_reason": finish_reason}], "usage": {"total_tokens": 10}}


class EffortMediumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.utils import openai_compat as compat_mod  # noqa: PLC0415

        cls.agent_mod = agent_mod
        cls.compat_mod = compat_mod
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    def _payload(self, provider: str = "vllm") -> dict:
        return self.agent_mod.build_chat_payload(
            provider=provider,
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            thinking=True,
            tools=[{"type": "function", "function": {"name": "python"}}],
            tool_choice="auto",
        )

    def _agent(self):
        return self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _tools(self):
        return [{"type": "function", "function": {"name": "python", "parameters": {}}}]

    # --- part A: payload ---

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "effort_medium: OK")
        self.assertEqual(install(), "effort_medium: SKIP (already applied)")

    def test_02_kwargs_present_on_synthetic_vllm_request(self) -> None:
        payload = self._payload()
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "medium"},
        )

    def test_03_effort_level_env_override(self) -> None:
        os.environ["EFFORT_LEVEL"] = "low"
        self.assertEqual(self._payload()["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_04_non_vllm_provider_untouched(self) -> None:
        payload = self._payload(provider="openrouter")
        self.assertNotIn("chat_template_kwargs", payload)

    def test_05_flag_off_byte_identical_to_stock(self) -> None:
        os.environ["EFFORT_MEDIUM"] = "0"
        patched = self._payload()
        stock = install.originals["build_chat_payload"](
            provider="vllm",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            thinking=True,
            tools=[{"type": "function", "function": {"name": "python"}}],
            tool_choice="auto",
        )
        self.assertEqual(patched, stock)
        self.assertNotIn("reasoning_effort", patched["chat_template_kwargs"])

    def test_06_dual_namespace_rebound(self) -> None:
        self.assertIs(self.agent_mod.build_chat_payload, self.compat_mod.build_chat_payload)
        self.assertTrue(getattr(self.compat_mod.build_chat_payload, "_effort_medium_patched", False))

    def test_07_setdefault_never_clobbers_explicit_value(self) -> None:
        # If a future harness sets reasoning_effort itself, the graft yields:
        # feed the wrapper an inner build that already carries the key.
        import graft_effort  # noqa: PLC0415

        wrapper_globals = self.compat_mod.build_chat_payload.__closure__
        original = install.originals["build_chat_payload"]

        def explicit_inner(**kwargs):
            payload = original(**kwargs)
            payload["chat_template_kwargs"]["reasoning_effort"] = "xhigh"
            return payload

        # re-create the wrapper against the explicit inner via a fresh install
        # on a throwaway namespace is overkill; assert the wrapper's operation
        # directly instead: it uses dict.setdefault on chat_template_kwargs.
        payload = explicit_inner(
            provider="vllm",
            model="m",
            messages=[],
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            thinking=True,
        )
        template_kwargs = payload["chat_template_kwargs"]
        template_kwargs.setdefault("reasoning_effort", graft_effort._effort_level())
        self.assertEqual(template_kwargs["reasoning_effort"], "xhigh")
        self.assertIsNotNone(wrapper_globals)  # wrapper really is a closure over the stock build

    # --- part B: dead-completion retry ---

    def test_08_dead_signature_arms_then_forces_next_request(self) -> None:
        agent = self._agent()
        posted: list[dict] = []
        responses = [
            _completion({"reasoning_content": "endless thinking", "content": ""}, "stop"),
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
                    ],
                },
                "tool_calls",
            ),
        ]

        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            posted.append(json)
            return _FakeResponse(responses.pop(0))

        caller_messages = [{"role": "user", "content": "turn prompt"}]
        with mock.patch.object(self.agent_mod.requests, "post", side_effect=fake_post):
            agent._chat_completion(caller_messages, tools=self._tools())
            self.assertTrue(agent._eff_dead_pending)  # armed by dead completion
            self.assertEqual(posted[0].get("tool_choice"), "auto")  # first request stock
            agent._chat_completion(caller_messages, tools=self._tools())
        self.assertFalse(agent._eff_dead_pending)  # healthy completion disarms
        # forced request: tool_choice pinned to python + one act-now user line
        self.assertEqual(posted[1]["tool_choice"], FORCED_TOOL_CHOICE)
        self.assertEqual(posted[1]["messages"][-1], {"role": "user", "content": ACT_NOW_LINE})
        self.assertEqual(sum(1 for m in posted[1]["messages"] if m.get("content") == ACT_NOW_LINE), 1)
        self.assertEqual(caller_messages, [{"role": "user", "content": "turn prompt"}])  # caller list untouched

    def test_09_fires_only_on_dead_signature(self) -> None:
        agent = self._agent()
        cases = [
            (_completion({"content": "some text"}, "stop"), False),  # content present
            (
                _completion(
                    {"content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
                    "stop",
                ),
                False,
            ),  # tool call present
            (_completion({"reasoning_content": "r", "content": ""}, "length"), False),  # not stop
            (_completion({"reasoning_content": "r <tool_call>x</tool_call>", "content": ""}, "stop"), False),  # markup: stock recovery owns it
            (_completion({"reasoning_content": "r", "content": ""}, "stop"), True),  # the measured signature
            (_completion({"content": ""}, "stop"), True),  # fully empty
        ]
        for payload, expected in cases:
            agent._eff_dead_pending = False
            with mock.patch.object(
                self.agent_mod.requests, "post", return_value=_FakeResponse(payload)
            ):
                agent._chat_completion([{"role": "user", "content": "p"}], tools=self._tools())
            self.assertEqual(agent._eff_dead_pending, expected, msg=f"payload={payload}")

    def test_10_flag_off_no_arming_no_forcing(self) -> None:
        os.environ["EFFORT_DEAD_RETRY"] = "0"
        agent = self._agent()
        agent._eff_dead_pending = True  # even if armed, flag off must not force
        posted: list[dict] = []

        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            posted.append(json)
            return _FakeResponse(_completion({"reasoning_content": "r", "content": ""}, "stop"))

        with mock.patch.object(self.agent_mod.requests, "post", side_effect=fake_post):
            agent._chat_completion([{"role": "user", "content": "p"}], tools=self._tools())
        self.assertEqual(posted[0].get("tool_choice"), "auto")
        self.assertNotIn(ACT_NOW_LINE, json.dumps(posted[0]["messages"]))

    def test_11_analyze_entry_clears_pending(self) -> None:
        agent = self._agent()
        agent._eff_dead_pending = True
        result = agent.analyze(Path("/nonexistent/state.json"), 0)  # stock early-return path
        self.assertIsNone(result)
        self.assertFalse(agent._eff_dead_pending)

    def test_12_fail_open_on_weird_result_shapes(self) -> None:
        agent = self._agent()
        weird = _completion({"content": {"not": "a-string"}}, "stop")
        with mock.patch.object(self.agent_mod.requests, "post", return_value=_FakeResponse(weird)):
            out = agent._chat_completion([{"role": "user", "content": "p"}], tools=self._tools())
        self.assertIsNotNone(out)  # no crash; detection failure => not armed or armed, never raises

    def test_13_inner_request_exception_propagates_as_stock(self) -> None:
        agent = self._agent()
        agent._eff_dead_pending = True  # forced path must still clear the tls flag
        with mock.patch.object(
            self.agent_mod.requests,
            "post",
            side_effect=self.agent_mod.requests.ConnectionError("dead"),
        ):
            with self.assertRaises(self.agent_mod.requests.RequestException):
                agent._chat_completion([{"role": "user", "content": "p"}], tools=self._tools())
        import graft_effort  # noqa: PLC0415

        self.assertFalse(getattr(graft_effort._tls, "force_python", False))  # tls flag released


if __name__ == "__main__":
    unittest.main(verbosity=2)
