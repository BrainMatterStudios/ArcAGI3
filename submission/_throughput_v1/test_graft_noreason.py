#!/usr/bin/env python3
"""Tests for graft_noreason. The bar: reasoning leaves retained history, and NOTHING else
about the message list changes -- order, roles, content, tool_calls and tool results all
survive, because a malformed history is how you lose a wave without noticing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graft_noreason as gn  # noqa: E402


class StripTests(unittest.TestCase):
    def setUp(self):
        gn._STATE["stripped"] = 0
        gn._STATE["messages_seen"] = 0

    def test_removes_reasoning_from_assistant_messages(self):
        out = gn._strip([{"role": "assistant", "content": "c", "reasoning": "long think"}])
        self.assertNotIn("reasoning", out[0])
        self.assertEqual(out[0]["content"], "c")
        self.assertEqual(gn._STATE["stripped"], 1)

    def test_leaves_user_and_tool_messages_untouched(self):
        msgs = [{"role": "user", "content": "u"},
                {"role": "tool", "tool_call_id": "1", "name": "python", "content": "{}"}]
        self.assertEqual(gn._strip(list(msgs)), msgs)
        self.assertEqual(gn._STATE["stripped"], 0)

    def test_preserves_tool_calls_on_a_stripped_assistant(self):
        tc = [{"id": "1", "name": "python", "arguments": "{}"}]
        out = gn._strip([{"role": "assistant", "reasoning": "r", "tool_calls": tc}])
        self.assertEqual(out[0]["tool_calls"], tc)
        self.assertNotIn("reasoning", out[0])

    def test_null_content_becomes_empty_string_when_nothing_else_remains(self):
        out = gn._strip([{"role": "assistant", "content": None, "reasoning": "r"}])
        self.assertEqual(out[0]["content"], "")

    def test_order_and_length_are_preserved(self):
        msgs = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b", "reasoning": "r"},
                {"role": "tool", "content": "c"},
                {"role": "user", "content": "d"}]
        out = gn._strip(list(msgs))
        self.assertEqual([m["role"] for m in out], [m["role"] for m in msgs])
        self.assertEqual(len(out), 4)

    def test_does_not_mutate_the_input_messages(self):
        original = {"role": "assistant", "content": "c", "reasoning": "r"}
        gn._strip([original])
        self.assertIn("reasoning", original, "must copy, not mutate the caller's dict")

    def test_assistant_without_reasoning_is_passed_through_unchanged(self):
        m = {"role": "assistant", "content": "c"}
        self.assertIs(gn._strip([m])[0], m)


class InstallTests(unittest.TestCase):
    def test_install_binds_and_is_idempotent(self):
        REPO = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"))
        from inference.agent import tool_agent as mod
        before = mod.ToolAgent._persistent_history_messages
        gn._STATE["installed"] = False
        msg = gn.install()
        try:
            self.assertTrue(msg.endswith(": OK"), msg)
            self.assertIsNot(mod.ToolAgent._persistent_history_messages, before)
            self.assertIn("SKIP", gn.install())
        finally:
            mod.ToolAgent._persistent_history_messages = before
            gn._STATE["installed"] = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
