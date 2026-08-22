"""Offline validation for the playbook_rider graft (no GPU/network/Kaggle).

Part A — unit tests of the menu classifier and block table against the
frame-0 menus recorded in scratchpad/testing_20260822/dispatch.json.
Part B — integration against the June stock bundle: install() onto the real
``ToolAgent._build_user_prompt`` seam and assert per-archetype injection,
exactly-once semantics, unknown-menu no-op, flags-off byte-identity,
fail-open on malformed playbook data, and the <=120-token block budget.

Run:  .venv/bin/python submission/_playbook_rider/test_playbook_rider.py
"""

from __future__ import annotations

import math
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

import graft_playbook
from graft_playbook import (
    PLAYBOOK,
    PLAYBOOK_HEADER,
    classify_menu,
    install,
    playbook_block,
)


def _clear_flags() -> None:
    os.environ.pop("PLAYBOOK_PRIOR", None)


def _approx_tokens(text: str) -> int:
    """Conservative token estimate: max(words, ceil(chars/4))."""
    return max(len(text.split()), math.ceil(len(text) / 4))


class UnitClassifier(unittest.TestCase):
    def test_click_family_menus(self) -> None:
        # dispatch.json frame-0 menus of the CLICK games (+RESET as served raw)
        for menu in ([6], [0, 6], [6, 7], [5, 6, 7], ["ACTION6"], ["MOUSE"]):
            self.assertEqual(classify_menu(menu), "CLICK", menu)

    def test_avatar_family_menus(self) -> None:
        # ls20/tu93/tr87 {1,2,3,4}; g50t/re86/wa30 {1,2,3,4,5}
        for menu in (
            [1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
            ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
        ):
            self.assertEqual(classify_menu(menu), "AVATAR", menu)

    def test_mixed_menus(self) -> None:
        # bp35 {3,4,6,7}; also the movement+click superset menus (cd82 etc.)
        for menu in ([3, 4, 6, 7], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 7]):
            self.assertEqual(classify_menu(menu), "MIXED", menu)

    def test_unknown_menus_get_no_archetype(self) -> None:
        # unconfident menus must yield None (triage defaults these to CLICK
        # as a never-kill safety; a prompt prior must NOT)
        for menu in ([], [0], [5], [7], [5, 7], ["WEIRD"], None, 42):
            self.assertIsNone(classify_menu(menu), menu)

    def test_blocks_carry_the_playbook_numbers(self) -> None:
        click = playbook_block("CLICK")
        self.assertIn("~18-37", click)
        self.assertIn("~131", click)
        self.assertIn("same cell", click)
        avatar = playbook_block("AVATAR")
        self.assertIn("~33-70", avatar)
        self.assertIn("~237", avatar)
        mixed = playbook_block("MIXED")
        self.assertIn("~102", mixed)
        self.assertIn("~11 times", mixed)
        for block in (click, avatar, mixed):
            self.assertTrue(block.startswith(PLAYBOOK_HEADER))
            self.assertNotIn("you should", block.lower())  # statistics, not orders

    def test_block_token_budget_under_120(self) -> None:
        for archetype, block in PLAYBOOK.items():
            tokens = _approx_tokens(block)
            print(f"playbook block {archetype}: {len(block)} chars, ~{tokens} tokens")
            self.assertLessEqual(tokens, 120, (archetype, tokens))

    def test_malformed_playbook_entries_yield_no_block(self) -> None:
        self.assertEqual(playbook_block(None), "")
        self.assertEqual(playbook_block("NOSUCH"), "")
        with mock.patch.dict(graft_playbook.PLAYBOOK, {"CLICK": None}):
            self.assertEqual(playbook_block("CLICK"), "")
        with mock.patch.dict(graft_playbook.PLAYBOOK, {"CLICK": "   "}):
            self.assertEqual(playbook_block("CLICK"), "")


class IntegrationJuneBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

        cls.agent_mod = agent_mod
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    def _make_agent(self):
        return self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )

    def _prompt(self, agent, valid_actions, action_num=0):
        return agent._build_user_prompt(
            action_num, valid_actions=valid_actions, current_frame=None
        )

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "playbook_rider: OK")
        self.assertEqual(install(), "playbook_rider: SKIP (already applied)")

    def test_02_correct_block_per_archetype(self) -> None:
        cases = {
            "CLICK": (["ACTION6"], "~18-37"),
            "AVATAR": (["ACTION1", "ACTION2", "ACTION3", "ACTION4"], "~33-70"),
            "MIXED": (["ACTION3", "ACTION4", "ACTION6", "ACTION7"], "~102"),
        }
        for archetype, (menu, marker) in cases.items():
            agent = self._make_agent()
            prompt = self._prompt(agent, menu)
            self.assertEqual(prompt.count(PLAYBOOK_HEADER), 1, archetype)
            self.assertIn(marker, prompt, archetype)
            self.assertIn(playbook_block(archetype), prompt, archetype)
            # appended at the END: the head of the prompt (carryover's 80-char
            # slice-boundary marker) is byte-identical to stock
            stock = install.originals["_build_user_prompt"](
                agent, 0, valid_actions=menu, current_frame=None
            )
            self.assertEqual(prompt[:80], stock[:80], archetype)
            self.assertTrue(prompt.startswith(stock), archetype)

    def test_03_injected_exactly_once_per_game(self) -> None:
        agent = self._make_agent()
        first = self._prompt(agent, ["ACTION6"])
        self.assertEqual(first.count(PLAYBOOK_HEADER), 1)
        # later turns (and yield-resume slices) rebuild the prompt: no block,
        # byte-identical to stock — prefix-cache stable
        for action_num in (0, 1, 7):
            later = self._prompt(agent, ["ACTION6"], action_num=action_num)
            stock = install.originals["_build_user_prompt"](
                agent, action_num, valid_actions=["ACTION6"], current_frame=None
            )
            self.assertNotIn(PLAYBOOK_HEADER, later)
            self.assertEqual(later, stock)

    def test_04_unknown_menu_no_injection_and_never_retried(self) -> None:
        agent = self._make_agent()
        stock = install.originals["_build_user_prompt"](
            agent, 0, valid_actions=["ACTION5"], current_frame=None
        )
        first = self._prompt(agent, ["ACTION5"])
        self.assertEqual(first, stock)  # byte-identical: no injection
        # a later readable menu must NOT trigger a late injection
        later = self._prompt(agent, ["ACTION6"], action_num=3)
        self.assertNotIn(PLAYBOOK_HEADER, later)

    def test_05_flag_off_byte_identical(self) -> None:
        os.environ["PLAYBOOK_PRIOR"] = "0"
        try:
            agent = self._make_agent()
            patched = self._prompt(agent, ["ACTION6"])
            stock = install.originals["_build_user_prompt"](
                agent, 0, valid_actions=["ACTION6"], current_frame=None
            )
            self.assertEqual(patched, stock)
            # flag off must not consume the once-per-game attempt
            self.assertFalse(getattr(agent, "_playbook_prior_done", False))
        finally:
            _clear_flags()

    def test_06_fail_open_on_malformed_playbook_data(self) -> None:
        agent = self._make_agent()
        stock = install.originals["_build_user_prompt"](
            agent, 0, valid_actions=["ACTION6"], current_frame=None
        )
        with mock.patch.dict(graft_playbook.PLAYBOOK, {"CLICK": {"broken": True}}):
            prompt = self._prompt(agent, ["ACTION6"])
        self.assertEqual(prompt, stock)  # byte-identical, no crash
        # a classifier crash is also fail-open
        agent = self._make_agent()
        with mock.patch.object(
            graft_playbook, "classify_menu", side_effect=RuntimeError("boom")
        ):
            prompt = self._prompt(agent, ["ACTION6"])
        self.assertEqual(prompt, stock)

    def test_07_injected_prompt_token_overhead(self) -> None:
        agent = self._make_agent()
        prompt = self._prompt(agent, ["ACTION6"])
        stock = install.originals["_build_user_prompt"](
            agent, 0, valid_actions=["ACTION6"], current_frame=None
        )
        overhead = _approx_tokens(prompt) - _approx_tokens(stock)
        print(f"first-prompt overhead: {len(prompt) - len(stock)} chars, ~{overhead} tokens")
        self.assertLessEqual(overhead, 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
