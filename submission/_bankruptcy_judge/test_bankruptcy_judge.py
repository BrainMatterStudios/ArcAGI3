"""Offline validation for the bankruptcy_judge graft (no GPU/network/Kaggle).

Covers: install/idempotency + flags-off install SKIP (subprocess); the
per-action evidence recorder on the REAL ``_HarnessGameSession.step_env``
seam (changed_px / level_delta / no-op tuples); the TUNED trigger (action
branch per archetype window, the stall OR-branch, cooldowns, per-level and
per-session caps, carried-model and evidence prerequisites); fresh-context
judge-call construction (exact [system, user] messages, no history leakage,
temp 1.0, no tools, max_tokens, reasoning_effort=medium); ruling parsing
(cited vs uncited vs KEEP vs garbage); rebuild application (REBUILT +
REJECTED:<old> + open_questions = discriminating tests); contradiction
ledger rendering and its reset on level transition; flags-off = byte-
identical prompts; fail-open on judge network errors and broken sessions;
and byte-parity of the embedded judge instruction with the falsifier file.

Run:  .venv/bin/python submission/_bankruptcy_judge/test_bankruptcy_judge.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
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

import graft_judge
from graft_judge import (
    LEDGER_MARKER,
    REBUILT_MARKER,
    build_judge_user_message,
    classify_menu,
    install,
    parse_judge_ruling,
)


def _clear_flags() -> None:
    for name in ("JUDGE_LEDGER", "JUDGE_CALL", "JUDGE_MAX_TOKENS"):
        os.environ.pop(name, None)


def _grid(fill: int, size: int = 4, marks: dict[tuple[int, int], int] | None = None):
    rows = [[fill] * size for _ in range(size)]
    for (row, col), color in (marks or {}).items():
        rows[row][col] = color
    return rows


class _ScriptedState:
    def __init__(
        self,
        arcengine,
        grid,
        *,
        levels_completed: int = 0,
        just_won_level: bool = False,
        state=None,
    ) -> None:
        self.raw = SimpleNamespace(state=state or arcengine.GameState.NOT_FINISHED)
        self.available_actions = [0, 1, 2, 3, 4, 5, 6]
        self.levels_completed = levels_completed
        self.just_won_level = just_won_level
        self.won = False
        self.frame = SimpleNamespace(data=grid)


class _ScriptedGame:
    """Duck-typed taaf Game: each execute_action pops the next scripted state
    (or keeps the current one when the queue is exhausted = strict no-op)."""

    def __init__(self, arcengine, initial: _ScriptedState, queue: list[_ScriptedState]):
        self._state = initial
        self._queue = list(queue)
        self.game_run = SimpleNamespace(
            state="playing", history=[], solver_analysis_html=None, game_id="fake"
        )
        self.number_of_levels = 5
        self.executed: list = []

    @property
    def current_state(self):
        return self._state

    def execute_action(self, action, *, generated_tokens=0, uncached_input_tokens=0):
        self.executed.append(action)
        self.game_run.history.append(action)
        if self._queue:
            self._state = self._queue.pop(0)
        return self._state


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._payload = {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 111, "completion_tokens": 55},
        }

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


REJECT_CITED = """CONTRADICTIONS:
- EXHIBIT A claims each click recolors one tile, but #12 and #14 show clicks changing 36 px each.
- EXHIBIT A predicts progress from repeated clicks, but transitions #15-#19 have level_delta 0.
VERDICT: REJECT
HYPOTHESES:
H1: clicks toggle a fixed neighborhood of tiles (lights-out) | test: click (10,10) and count changed cells
H2: a tile must be selected before placement takes effect | test: click the palette at (3,60) then click (10,10)
H3: progress is gated by a hidden order of activations | test: click the same two tiles in reversed order
"""

REJECT_UNCITED = """CONTRADICTIONS:
- none
VERDICT: REJECT
HYPOTHESES:
H1: something else entirely | test: press ACTION5
H2: a different guess | test: click (1,1)
H3: a third guess | test: press ACTION1
"""

KEEP_RULING = """CONTRADICTIONS:
- none
VERDICT: KEEP
"""


class BankruptcyJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        import arcengine  # noqa: PLC0415

        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415

        cls.arcengine = arcengine
        cls.agent_mod = agent_mod
        cls.solver_mod = solver_mod
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    # --- fixtures -----------------------------------------------------------

    def _session(self, game):
        tmp = Path(tempfile.mkdtemp(prefix="bj_test_"))
        return self.solver_mod._HarnessGameSession(
            solver=SimpleNamespace(
                max_runtime_s_per_game=None,
                max_actions_per_game=None,
                soft_time_remaining_seconds=lambda: None,
                job_dir=None,
                label="test",
                # consumed by the anim bundle's solver; June stock ignores it
                animation_awareness=False,
            ),
            game=game,
            analyzer=SimpleNamespace(generated_tokens=0, total_tokens=0),
            game_index=0,
            pass_index=0,
            state_path=tmp / "tool_runtime_state.json",
            transcript_path=tmp / "transcript.txt",
            analysis_html_relpath="x.html",
            stop_event=threading.Event(),
            viewer_data_path=tmp / "viewer.json",
        )

    def _game(self, queue_grids, initial_grid=None, **state_kwargs_by_index):
        arc = self.arcengine
        initial = _ScriptedState(arc, initial_grid or _grid(0))
        queue = []
        for index, grid in enumerate(queue_grids):
            kwargs = state_kwargs_by_index.get(f"s{index}", {})
            queue.append(_ScriptedState(arc, grid, **kwargs))
        return _ScriptedGame(arc, initial, queue)

    def _agent(self, *, menu=(0, 6), evidence=None, world_model="clicking a tile recolors only that tile"):
        """ToolAgent + fake bound session carrying an evidence log."""
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        fake_session = SimpleNamespace(
            game=SimpleNamespace(
                current_state=SimpleNamespace(available_actions=list(menu))
            ),
            _judge_evidence=list(evidence or []),
        )
        agent._step_env_callback = types.MethodType(lambda self, args: {}, fake_session)
        if world_model:
            agent._summarized_knowledge["world_model"] = world_model
        return agent, fake_session

    @staticmethod
    def _evidence(n=6, level=1, changed=37, noop_every=None, start=0):
        rows = []
        for k in range(n):
            noop = noop_every is not None and (k % noop_every == noop_every - 1)
            rows.append(
                {
                    "i": start + k,
                    "action": "ACTION6",
                    "coords": [10 + k, 20],
                    "changed_px": 0 if noop else changed,
                    "level": level,
                    "level_delta": 0,
                }
            )
        return rows

    def _prompt(self, agent, action_num, *, level=1, summary=None):
        return agent._build_user_prompt(
            action_num,
            valid_actions=["ACTION6"],
            current_frame=SimpleNamespace(level=level, step=action_num),
            history_entries=None,
            previous_step_summary=summary,
        )

    # --- 01 installation ----------------------------------------------------

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "bankruptcy_judge: OK")
        self.assertEqual(install(), "bankruptcy_judge: SKIP (already applied)")

    def test_02_flags_off_install_skips_in_fresh_process(self) -> None:
        env = dict(os.environ, JUDGE_LEDGER="0", JUDGE_CALL="0")
        env["PYTHONPATH"] = os.pathsep.join([str(_HERE), str(_BUNDLE)])
        out = subprocess.run(
            [sys.executable, "-c", "import graft_judge; print(graft_judge.install())"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("bankruptcy_judge: SKIP (JUDGE_LEDGER=0 and JUDGE_CALL=0)", out.stdout)

    # --- 03 evidence recorder on the real session seam ----------------------

    def test_03_evidence_recorder_real_step_env(self) -> None:
        changed = _grid(0, marks={(1, 1): 3})
        game = self._game(
            [
                changed,                                  # step 1: 1 px change
                [list(row) for row in changed],           # step 2: strict no-op
                _grid(9),                                 # (blocked by expect no-op halt
            ]                                             #  if that graft were present)
        )
        session = self._session(game)
        session.step_env({"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}]})
        evidence = session._judge_evidence
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["i"], 0)
        self.assertEqual(evidence[0]["changed_px"], 1)
        self.assertEqual(evidence[0]["level_delta"], 0)
        self.assertEqual(evidence[1]["changed_px"], 0)  # strict no-op captured
        self.assertEqual(evidence[1]["action"], "ACTION2")

        # level completion => level_delta 1 and payload-level recorded
        game2 = self._game(
            [_grid(7)], s0={"levels_completed": 1, "just_won_level": True}
        )
        session2 = self._session(game2)
        session2.step_env({"actions": [{"action": "ACTION1"}]})
        self.assertEqual(session2._judge_evidence[0]["level_delta"], 1)
        self.assertEqual(session2._judge_evidence[0]["level"], 2)

    def test_04_mouse_coords_recorded(self) -> None:
        game = self._game([_grid(0, marks={(2, 2): 5})])
        session = self._session(game)
        session.step_env({"actions": [{"action": "MOUSE", "row": 2, "col": 3}]})
        entry = session._judge_evidence[0]
        self.assertEqual(entry["action"], "ACTION6")
        self.assertIsInstance(entry["coords"], list)
        self.assertEqual(len(entry["coords"]), 2)

    # --- trigger ------------------------------------------------------------

    def test_05_action_branch_fires_at_tuned_click_window(self) -> None:
        agent, session = self._agent(menu=(0, 6), evidence=self._evidence(30))
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            for action_num in (0, 10, 20, 30, 44):
                self._prompt(agent, action_num)
            self.assertEqual(post.call_count, 0, "no fire below A_click=45")
            self._prompt(agent, 45)
            self.assertEqual(post.call_count, 1, "asl=45 must fire for CLICK")
        state = agent._judge_state_v1
        self.assertEqual(state["archetype"], "CLICK")
        self.assertEqual(state["fires_session"], 1)
        self.assertEqual(state["rulings"][0]["reason"], "actions")
        # fire resets the window
        self.assertEqual(state["level_start_action"], 45)

    def test_06_avatar_window_is_40(self) -> None:
        agent, _ = self._agent(menu=(0, 1, 2, 3, 4), evidence=self._evidence(30))
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            self._prompt(agent, 0)
            self._prompt(agent, 39)
            self.assertEqual(post.call_count, 0)
            self._prompt(agent, 40)
            self.assertEqual(post.call_count, 1)
        self.assertEqual(agent._judge_state_v1["archetype"], "AVATAR")

    def test_07_stall_branch_fires_at_22_blocks(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(4))
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            for _ in range(22):  # blocks 0..21 — bsl reaches 21, below T_stall
                self._prompt(agent, 5)
            self.assertEqual(post.call_count, 0)
            self._prompt(agent, 5)  # block 22: bsl=22 => stall fire
            self.assertEqual(post.call_count, 1)
        self.assertEqual(agent._judge_state_v1["rulings"][0]["reason"], "stall")

    def test_08_cooldowns_and_caps(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(30))
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            self._prompt(agent, 0)                        # window zero-point
            self._prompt(agent, 45)                       # fire 1 (level 1)
            self.assertEqual(post.call_count, 1)
            self._prompt(agent, 91)                       # asl>=45 again but block
            self.assertEqual(post.call_count, 1)          # cooldown (8 blocks) holds
            for _ in range(6):
                self._prompt(agent, 140)
            self.assertEqual(post.call_count, 1)
            self._prompt(agent, 140)                      # 8th block after fire 1
            self.assertEqual(post.call_count, 2)          # fire 2 (level cap reached)
            for _ in range(20):
                self._prompt(agent, 400)                  # eligible but cap_level=2
            self.assertEqual(post.call_count, 2)
            for _ in range(30):
                self._prompt(agent, 500, level=2)         # new level resets level cap
            self.assertEqual(post.call_count, 3)          # fire 3 = session cap
            for _ in range(30):
                self._prompt(agent, 900, level=3)
            self.assertEqual(post.call_count, 3, "cap_session=3 is hard")

    def test_09_no_fire_without_carried_model_or_evidence(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(30), world_model="")
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            self._prompt(agent, 60)
            self.assertEqual(post.call_count, 0, "no carried model => no judge")
        agent2, _ = self._agent(menu=(0, 6), evidence=[])
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(KEEP_RULING)
        ) as post:
            self._prompt(agent2, 60)
            self.assertEqual(post.call_count, 0, "no evidence => no judge")

    # --- judge call construction --------------------------------------------

    def test_10_fresh_context_no_history_leakage(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(6))
        agent._history_messages = [
            {"role": "user", "content": "SENTINEL_HISTORY_MUST_NOT_LEAK"},
            {"role": "assistant", "content": "SENTINEL_ASSISTANT"},
        ]
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResponse(KEEP_RULING)

        with mock.patch.object(self.agent_mod.requests, "post", side_effect=fake_post):
            self._prompt(agent, 0)  # window zero-point
            self._prompt(agent, 45)
        payload = captured["payload"]
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        messages = payload["messages"]
        self.assertEqual(len(messages), 2, "fresh context is exactly [system, user]")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], agent._system_prompt)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("BANKRUPTCY JUDGE", messages[1]["content"])
        self.assertIn("EXHIBIT A — carried working world model:", messages[1]["content"])
        self.assertIn("EXHIBIT B — evidence digest:", messages[1]["content"])
        self.assertIn("clicking a tile recolors only that tile", messages[1]["content"])
        self.assertIn("changed_px=37", messages[1]["content"])
        blob = json_dumps_safe(payload)
        self.assertNotIn("SENTINEL_HISTORY_MUST_NOT_LEAK", blob)
        self.assertNotIn("SENTINEL_ASSISTANT", blob)
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["chat_template_kwargs"]["reasoning_effort"], "medium")

    # --- ruling parsing + application ---------------------------------------

    def test_11_parser_cited_vs_uncited_vs_keep(self) -> None:
        cited = parse_judge_ruling(REJECT_CITED)
        self.assertEqual(cited["verdict"], "REJECT")
        self.assertTrue(cited["cited"])
        self.assertEqual(len(cited["hypotheses"]), 3)
        self.assertEqual(cited["hypotheses"][0]["n"], "1")
        self.assertIn("lights-out", cited["hypotheses"][0]["mechanic"])
        self.assertIn("click (10,10)", cited["hypotheses"][0]["test"])
        uncited = parse_judge_ruling(REJECT_UNCITED)
        self.assertEqual(uncited["verdict"], "REJECT")
        self.assertFalse(uncited["cited"])
        keep = parse_judge_ruling(KEEP_RULING)
        self.assertEqual(keep["verdict"], "KEEP")
        garbage = parse_judge_ruling("total nonsense with no format at all")
        self.assertIsNone(garbage["verdict"])
        self.assertEqual(parse_judge_ruling("")["verdict"], None)

    def test_12_cited_reject_rebuilds_world_model(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(30))
        with mock.patch.object(
            self.agent_mod.requests, "post", return_value=_FakeResponse(REJECT_CITED)
        ):
            self._prompt(agent, 0)  # window zero-point
            self._prompt(agent, 45)
        knowledge = agent._summarized_knowledge
        self.assertTrue(knowledge["world_model"].startswith(REBUILT_MARKER))
        self.assertIn("REJECTED: clicking a tile recolors only that tile", knowledge["world_model"])
        self.assertIn("lights-out", knowledge["world_model"])
        self.assertIn("discriminating tests", knowledge["open_questions"].lower())
        self.assertIn("click (10,10)", knowledge["open_questions"])
        self.assertTrue(agent._judge_state_v1["rulings"][0]["applied"])
        # the rebuilt model is what the NEXT prompt renders
        prompt = self._prompt(agent, 46)
        self.assertIn(REBUILT_MARKER, prompt)

    def test_13_uncited_reject_and_keep_change_nothing(self) -> None:
        for ruling_text in (REJECT_UNCITED, KEEP_RULING):
            agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(30))
            with mock.patch.object(
                self.agent_mod.requests, "post", return_value=_FakeResponse(ruling_text)
            ):
                self._prompt(agent, 0)  # window zero-point
                self._prompt(agent, 45)
            self.assertEqual(
                agent._summarized_knowledge["world_model"],
                "clicking a tile recolors only that tile",
                f"ruling must be ignored: {ruling_text[:30]!r}",
            )
            self.assertFalse(agent._judge_state_v1["rulings"][0]["applied"])

    # --- contradiction ledger -----------------------------------------------

    def test_14_ledger_renders_and_resets_on_level_transition(self) -> None:
        evidence = self._evidence(9, level=1, noop_every=3)  # 3 no-ops of 9
        agent, session = self._agent(menu=(0, 6), evidence=evidence)
        # a recorded expect-queue stop event on level 1
        agent._compact_action_result(
            {"executed": True, "level": 1, "stop_reason": "expect_mismatch",
             "stopped_early": True, "requested_count": 4, "executed_count": 2}
        )
        prompt = self._prompt(agent, 9)
        self.assertIn(LEDGER_MARKER, prompt)
        self.assertIn("3 of the last 9 actions on this level changed zero pixels", prompt)
        self.assertIn("1 action batch(es) halted early on an expect mismatch", prompt)
        self.assertIn("without a level-up", prompt)
        # stated facts sit INSIDE the carried world-model block
        self.assertLess(
            prompt.index("Working world model carried from earlier turns:"),
            prompt.index(LEDGER_MARKER),
        )
        self.assertLess(prompt.index(LEDGER_MARKER), prompt.index("end of world model"))
        # level transition: evidence is level-1, events reset => ledger gone
        prompt2 = self._prompt(agent, 10, level=2)
        self.assertNotIn(LEDGER_MARKER, prompt2)
        self.assertEqual(agent._judge_state_v1["expect_events"], {})

    def test_15_ledger_absent_when_nothing_to_state(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(6))  # zero no-ops
        prompt = self._prompt(agent, 6)
        self.assertNotIn(LEDGER_MARKER, prompt)

    def test_16_ledger_tracks_noops_without_expect_queue(self) -> None:
        # loose integration: no expect graft, no stop events — strict no-ops
        # from the evidence log alone still produce the ledger.
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(6, noop_every=2))
        prompt = self._prompt(agent, 6)
        self.assertIn(LEDGER_MARKER, prompt)
        self.assertIn("3 of the last 6 actions", prompt)
        self.assertNotIn("expect mismatch", prompt.split(LEDGER_MARKER, 1)[1].split("\n")[0])

    # --- flags + fail-open ---------------------------------------------------

    def test_17_flags_off_byte_identical_prompt(self) -> None:
        evidence = self._evidence(9, noop_every=3)
        agent, _ = self._agent(menu=(0, 6), evidence=evidence)
        os.environ["JUDGE_LEDGER"] = "0"
        os.environ["JUDGE_CALL"] = "0"
        with mock.patch.object(self.agent_mod.requests, "post") as post:
            patched_prompt = self._prompt(agent, 60)
            self.assertEqual(post.call_count, 0)
        stock_prompt = install.originals["_build_user_prompt"](
            agent,
            60,
            valid_actions=["ACTION6"],
            current_frame=SimpleNamespace(level=1, step=60),
            history_entries=None,
            previous_step_summary=None,
        )
        self.assertEqual(patched_prompt, stock_prompt)

    def test_18_benign_state_is_byte_identical_even_with_flags_on(self) -> None:
        # No no-ops, no events, below trigger => the graft adds nothing.
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(4))
        patched_prompt = self._prompt(agent, 10)
        stock_prompt = install.originals["_build_user_prompt"](
            agent,
            10,
            valid_actions=["ACTION6"],
            current_frame=SimpleNamespace(level=1, step=10),
            history_entries=None,
            previous_step_summary=None,
        )
        self.assertEqual(patched_prompt, stock_prompt)

    def test_19_judge_network_failure_is_fail_open(self) -> None:
        agent, _ = self._agent(menu=(0, 6), evidence=self._evidence(30))
        with mock.patch.object(
            self.agent_mod.requests,
            "post",
            side_effect=self.agent_mod.requests.RequestException("connection refused"),
        ):
            self._prompt(agent, 0)  # window zero-point
            prompt = self._prompt(agent, 45)  # must not raise
        self.assertIn("clicking a tile recolors only that tile", prompt)
        outcome = agent._judge_state_v1["rulings"][0]
        self.assertFalse(outcome["applied"])
        self.assertIn("connection refused", outcome["error"])
        # the failed fire still consumed the cooldown — no hammering
        with mock.patch.object(self.agent_mod.requests, "post") as post:
            self._prompt(agent, 46)
            self.assertEqual(post.call_count, 0)

    def test_20_broken_session_is_fail_open(self) -> None:
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        agent._step_env_callback = None  # no session bound at all
        agent._summarized_knowledge["world_model"] = "model"
        prompt = self._prompt(agent, 60)  # must not raise, must not fire
        self.assertIn("model", prompt)

    # --- misc units ----------------------------------------------------------

    def test_21_classify_menu_matches_triage_rule(self) -> None:
        self.assertEqual(classify_menu([0, 6]), "CLICK")
        self.assertEqual(classify_menu([0, 1, 2, 3, 4]), "AVATAR")
        self.assertEqual(classify_menu([0, 1, 2, 3, 4, 5]), "AVATAR")
        self.assertEqual(classify_menu([0, 1, 6]), "MIXED")
        self.assertEqual(classify_menu([0, 5, 6]), "CLICK")
        self.assertIsNone(classify_menu([0]))
        self.assertEqual(classify_menu(None), "CLICK")

    def test_22_embedded_instruction_matches_falsifier_file(self) -> None:
        on_disk = (_HERE / "falsifier" / "judge_instruction.txt").read_text().strip()
        self.assertEqual(graft_judge.JUDGE_INSTRUCTION, on_disk)
        framed = build_judge_user_message("WM", "DIGEST")
        self.assertTrue(framed.startswith(on_disk))
        self.assertIn("EXHIBIT A — carried working world model:\n---\nWM\n---", framed)
        self.assertIn("EXHIBIT B — evidence digest:\n---\nDIGEST\n---", framed)
        self.assertTrue(framed.endswith("Deliver your ruling now, in the exact format specified."))


def json_dumps_safe(payload) -> str:
    try:
        return json.dumps(payload)
    except TypeError:
        return str(payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
