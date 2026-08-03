"""Tests for patch 13 (mechanic-archetype playbook in the system prompt).

Layers:
  1. UNIT — patch applies/skips/declines cleanly; call-time env gating; the text
     is archetype-level (no game ids — the hidden set is different games) and
     within the ~350-token budget.
  2. INJECTION — the playbook lands in a REAL ToolAgent's composed system prompt
     (agents are constructed after apply_all), and only when enabled.
  3. SCORED-BUNDLE — the patch must APPLY on the scored bundle bytes at
     scratchpad/taaf_scored_ref in a subprocess (law: the drifted _adopt tree is
     NOT the eval bytes), and the text must flow into a real ToolAgent there.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_playbook.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import _PLAYBOOK_TEXT  # noqa: E402

# A distinctive fragment that can only come from the playbook.
MARKER = "PRECONDITION HUNT"


def _apply() -> None:
    result = duck_patches.patch_mechanic_playbook()
    assert "OK" in result or "SKIP" in result, result


# ---------------------------------------------------------------------------------
# UNIT
# ---------------------------------------------------------------------------------


def test_patch_applies_and_is_idempotent():
    first = duck_patches.patch_mechanic_playbook()
    assert first in ("patch13 playbook: OK", "patch13 playbook: SKIP (already applied)")
    second = duck_patches.patch_mechanic_playbook()
    assert second == "patch13 playbook: SKIP (already applied)"


def test_playbook_is_archetype_level_not_game_specific():
    """The hidden eval set is DIFFERENT games: any game id or memorized solution
    in the text would be dead weight at best and misleading at worst."""
    lowered = _PLAYBOOK_TEXT.lower()
    for game_id in (
        "cn04", "dc22", "g50t", "lf52", "ls20", "m0r0", "sk48", "tr87", "wa30",
        "ft09", "tu93", "sb26", "vc33",
    ):
        assert game_id not in lowered, f"playbook leaks game id {game_id!r}"
    # No memorized coordinates or solutions either.
    assert "37,7" not in _PLAYBOOK_TEXT and "(37, 7)" not in _PLAYBOOK_TEXT


def test_playbook_token_budget():
    """~250-350 tokens. At the usual ~4 chars/token this means <= ~1400 chars;
    the bundle's own conservative estimator ((len+2)//3) must stay near budget."""
    n = len(_PLAYBOOK_TEXT)
    assert n <= 1400, f"playbook too long: {n} chars (~{n // 4} tokens at 4 chars/tok)"
    assert n >= 700, "playbook suspiciously short — did the text get truncated?"


def test_playbook_covers_the_four_diagnosed_gaps():
    for fragment in (
        "PRECONDITION HUNT",
        "SETBACKS MAY BE TOOLS",
        "LEGENDS ARE SPECS",
        "COUPLED ENTITIES",
    ):
        assert fragment in _PLAYBOOK_TEXT, f"missing heuristic {fragment!r}"


def test_patch_declines_cleanly_when_builder_missing(monkeypatch):
    from inference.agent import tool_agent

    monkeypatch.setattr(tool_agent, "_build_system_prompt", None)
    msg = duck_patches.patch_mechanic_playbook()
    assert msg.startswith("patch13 playbook: FAIL"), msg


# ---------------------------------------------------------------------------------
# INJECTION — real composed system prompts
# ---------------------------------------------------------------------------------


def test_playbook_lands_in_module_system_prompt_when_enabled(monkeypatch):
    from inference.agent import tool_agent

    _apply()
    monkeypatch.delenv("TAAF_PLAYBOOK", raising=False)  # default ON
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert MARKER in prompt
    assert prompt.rstrip().endswith(_PLAYBOOK_TEXT.rstrip())


def test_playbook_env_gating_is_call_time(monkeypatch):
    from inference.agent import tool_agent

    _apply()
    monkeypatch.setenv("TAAF_PLAYBOOK", "0")
    assert MARKER not in tool_agent._build_system_prompt(tool_output_tokens=1000)
    monkeypatch.setenv("TAAF_PLAYBOOK", "1")
    assert MARKER in tool_agent._build_system_prompt(tool_output_tokens=1000)
    monkeypatch.setenv("TAAF_PLAYBOOK", "false")
    assert MARKER not in tool_agent._build_system_prompt(tool_output_tokens=1000)


def test_playbook_lands_in_real_tool_agent_prompt(monkeypatch):
    """The path production takes: ToolAgent.__init__ composes _system_prompt."""
    from inference.agent.tool_agent import ToolAgent

    _apply()
    monkeypatch.delenv("TAAF_PLAYBOOK", raising=False)
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    assert MARKER in agent._system_prompt
    assert agent._system_prompt.count(MARKER) == 1, "playbook must appear exactly once"

    monkeypatch.setenv("TAAF_PLAYBOOK", "0")
    agent_off = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    assert MARKER not in agent_off._system_prompt


def test_playbook_stacks_with_plan_queue_guidance(monkeypatch):
    """Patches 12b and 13 wrap the same builder; both texts must coexist and a
    re-apply of either must not double-wrap (marker forwarding)."""
    from inference.agent import tool_agent

    assert duck_patches.patch_plan_queue() in (
        "patch12b plan-queue: OK",
        "patch12b plan-queue: SKIP (already applied)",
    )
    _apply()
    monkeypatch.setenv("TAAF_PLAYBOOK", "1")
    monkeypatch.setenv("TAAF_COMPACT", "1")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert MARKER in prompt and "PLAN QUEUE" in prompt
    # Idempotency across the whole chain: markers must be visible on the outermost.
    builder = tool_agent._build_system_prompt
    assert getattr(builder, "_playbook_patched", False)
    assert getattr(builder, "_plan_queue_patched", False)


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE validation (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------


def test_scored_bundle_has_the_wrapped_symbol():
    source = (SCORED_REF / "inference/agent/tool_agent.py").read_text()
    assert "def _build_system_prompt" in source, (
        "scored bundle lost _build_system_prompt — re-validate patch13"
    )


@pytest.mark.skipif(not SCORED_REF.is_dir(), reason="scored bundle bytes absent")
def test_patch13_applies_on_scored_bundle_bytes():
    """Apply patch 13 on the scored bytes in a clean subprocess; the playbook
    must reach a real ToolAgent's system prompt there, and only when enabled."""
    script = f"""
import os, sys
sys.path.insert(0, {str(SCORED_REF)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

r = duck_patches.patch_mechanic_playbook()
assert r == "patch13 playbook: OK", r

from inference.agent.tool_agent import ToolAgent
os.environ["TAAF_PLAYBOOK"] = "1"
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
assert {MARKER!r} in agent._system_prompt, "playbook missing from scored-bundle agent"
os.environ["TAAF_PLAYBOOK"] = "0"
agent_off = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
assert {MARKER!r} not in agent_off._system_prompt, "TAAF_PLAYBOOK=0 must disable"
print("SCORED-PLAYBOOK-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-PLAYBOOK-OK" in proc.stdout
