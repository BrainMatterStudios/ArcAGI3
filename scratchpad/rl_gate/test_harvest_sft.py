"""Sanity tests for harvest_sft.py against real captured episodes.

Run:  .venv/bin/python -m pytest scratchpad/rl_gate/test_harvest_sft.py -q
"""
import json
import re
from pathlib import Path

import pytest

import harvest_sft as H

SB26 = "k3_batch1_sb26"
DATA_URL = re.compile(r"^data:image/[a-z]+;base64,[A-Za-z0-9+/=]+$")


@pytest.fixture(scope="module")
def sb26():
    return H.harvest_episode(SB26)


def test_sb26_boundaries(sb26):
    """Full-win episode: 8 completed levels, samples span exactly levels 1..8."""
    assert sb26["completed_levels"] == [1, 2, 3, 4, 5, 6, 7, 8]
    levels = {s["meta"]["level"] for s in sb26["samples"]}
    assert levels == set(range(1, 9))
    assert all(s["meta"]["level_won"] for s in sb26["samples"])
    assert not sb26["warnings"]
    # every reconstructed turn matches the event-log turn count
    assert len(sb26["samples"]) == sb26["n_turns_events"] == 24


def test_sample_well_formedness(sb26):
    for s in sb26["samples"]:
        msgs = s["messages"]
        assert msgs[0]["role"] == "system"
        # legal role sequencing for Qwen chat: tool only after assistant-with-tool_calls
        # (or another tool), assistant never first after system without a user turn
        assert msgs[1]["role"] == "user"
        for prev, cur in zip(msgs, msgs[1:]):
            if cur["role"] == "tool":
                assert prev["role"] in ("assistant", "tool")
                if prev["role"] == "assistant":
                    assert prev.get("tool_calls"), "tool result without tool_calls"
        # target is an assistant message containing the acted-on python tool call
        tgt = s["target"]
        assert tgt["role"] == "assistant" and tgt.get("tool_calls")
        assert "action(" in H.code_of(tgt)
        # image parts preserved exactly as data-urls
        for m in msgs:
            if isinstance(m.get("content"), list):
                for p in m["content"]:
                    if p["type"] == "image_url":
                        assert DATA_URL.match(p["image_url"]["url"])
        assert s["meta"]["images"] == H.count_images(msgs) > 0


def test_token_cap_flagging(sb26):
    """No silent drops: samples over 32768 tokens must carry the flag (count them)."""
    over = [s for s in sb26["samples"] if s["meta"]["tokens"] > H.MAX_TOKENS]
    for s in over:
        assert "over_32768_tokens" in s["meta"]["flags"]
    under = [s for s in sb26["samples"] if s["meta"]["tokens"] <= H.MAX_TOKENS]
    for s in under:
        assert "over_32768_tokens" not in s["meta"]["flags"]


def test_outputs_on_disk():
    out = H.OUT / "all_wins.jsonl"
    if not out.exists():
        pytest.skip("run harvest_sft.py first")
    wins = [json.loads(l) for l in out.open()]
    assert wins and all(s["meta"]["level_won"] for s in wins)
    stats = json.loads((H.OUT / "stats.json").read_text())
    assert stats["total_samples_wins"] == len(wins)
