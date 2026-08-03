"""Generator invariants: loadability, solvability, determinism, holdout split
integrity, corpus schema.

Run from repo root:
    .venv/bin/python -m pytest tools/synthgen/tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families import FAMILIES, sample_game, solution_trace
from render_game import render_game_py, write_env_dir
from validate import replay, validate_game

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_V3 = REPO_ROOT / "submission" / "_sft_k3" / "corpus_v3"

SEED = 424242


# ---------------------------------------------------------------------------
# sampling determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_sampling_is_deterministic(family):
    a = sample_game(family, 0, SEED)
    b = sample_game(family, 0, SEED)
    assert a == b
    assert render_game_py(a) == render_game_py(b)


@pytest.mark.parametrize("family", FAMILIES)
def test_different_indices_differ(family):
    a = sample_game(family, 0, SEED)
    b = sample_game(family, 1, SEED)
    assert a["game_id"] != b["game_id"]
    assert a["levels"] != b["levels"]


# ---------------------------------------------------------------------------
# loadability + solvability + determinism through the real local_wrapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def game_batch(tmp_path_factory):
    """One validated game per family, generated once for the module."""
    root = tmp_path_factory.mktemp("games")
    out = {}
    for family in FAMILIES:
        spec = sample_game(family, 0, SEED)
        write_env_dir(spec, root)
        out[family] = spec
    return root, out


@pytest.mark.parametrize("family", FAMILIES)
def test_game_loads_solves_deterministically(game_batch, family):
    root, specs = game_batch
    report = validate_game(root, specs[family])
    assert report["loaded"]
    assert report["solved"], report
    assert report["level_boundaries_ok"], report
    assert report["no_game_over"], report
    assert report["deterministic"], report


def test_env_dir_layout_matches_public_games(game_batch):
    root, specs = game_batch
    for spec in specs.values():
        base, version = spec["game_id"].split("-")
        d = root / base / version
        assert (d / f"{base}.py").exists()
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["game_id"] == spec["game_id"]
        sol = json.loads((d / "solution.json").read_text())
        assert sol["trace"], "empty solution trace"


def test_wrong_actions_do_not_win(game_batch):
    """A trivially wrong trace (spam ACTION1 or corner clicks) must not win."""
    root, specs = game_batch
    for family, spec in specs.items():
        n = len(solution_trace(spec))
        if family == "click":
            bad = [{"id": "ACTION6", "x": 0, "y": 0}] * min(n, 5)
        else:
            bad = [{"id": "ACTION1"}] * min(n, 5)
        records = replay(root, spec["game_id"], bad)
        assert records[-1]["state"] != "WIN"
        assert records[-1]["score"] < len(spec["levels"])


# ---------------------------------------------------------------------------
# corpus schema + family holdout split integrity
# ---------------------------------------------------------------------------

try:
    import corpus  # requires the editable `inference` harness + corpus_v3
except Exception:  # noqa: BLE001
    corpus = None
needs_corpus = pytest.mark.skipif(
    corpus is None or not CORPUS_V3.exists(),
    reason="corpus_v3 reference or harness package not present",
)


@pytest.fixture(scope="module")
def small_corpus(game_batch, tmp_path_factory):
    if corpus is None:
        pytest.skip("corpus builder unavailable")
    root, _specs = game_batch
    # generate_batch-style manifest for the module's games
    reports = []
    for family in FAMILIES:
        spec = sample_game(family, 0, SEED)
        reports.append(validate_game(root, spec))
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in reports))
    out = tmp_path_factory.mktemp("corpus")
    stats = corpus.build_corpus(root, out, holdout_family="push", seed=1, with_tokens=False)
    return out, stats


@needs_corpus
def test_holdout_split_integrity(small_corpus):
    out, stats = small_corpus
    rows = {
        name: [json.loads(l) for l in (out / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "val", "holdout_transfer")
    }
    train_val = rows["train"] + rows["val"]
    # 1. the held-out family never leaks into train/val
    assert all(r["meta"]["family"] != "push" for r in train_val)
    # 2. holdout file contains only the held-out family
    assert rows["holdout_transfer"]
    assert all(r["meta"]["family"] == "push" for r in rows["holdout_transfer"])
    # 3. split is by game: no episode overlap between train and val
    train_games = {r["meta"]["episode"] for r in rows["train"]}
    val_games = {r["meta"]["episode"] for r in rows["val"]}
    assert not (train_games & val_games)


@needs_corpus
def test_corpus_record_schema(small_corpus):
    out, _stats = small_corpus
    rows = []
    for name in ("train", "val", "holdout_transfer"):
        rows += [json.loads(l) for l in (out / f"{name}.jsonl").read_text().splitlines()]
    assert rows
    ref_system = json.loads(
        (CORPUS_V3 / "train.jsonl").open().readline()
    )["messages"][0]["content"]
    for r in rows:
        assert sorted(r.keys()) == ["messages", "meta", "target"]
        msgs = r["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == ref_system  # byte-identical system prompt
        assert msgs[1]["role"] == "user"
        # user messages are [text, image] pairs with a data-url PNG
        for m in msgs:
            if m["role"] == "user":
                assert isinstance(m["content"], list) and len(m["content"]) == 2
                assert m["content"][0]["type"] == "text"
                assert m["content"][0]["text"].endswith("Current grid image:")
                assert m["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        # target: assistant message with one python tool call, dict arguments
        t = r["target"]
        assert t["role"] == "assistant"
        assert sorted(t.keys()) == ["content", "reasoning_content", "role", "tool_calls"]
        (tc,) = t["tool_calls"]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "python"
        assert isinstance(tc["function"]["arguments"], dict)
        assert "action(" in tc["function"]["arguments"]["code"]
        # tool responses parse and carry the sandbox stdout envelope
        for m in msgs:
            if m["role"] == "tool":
                payload = json.loads(m["content"])
                assert payload["tool"] == "python"
                assert payload["returncode"] == 0
                assert "stdout" in payload
        # meta split axes
        for key in ("game", "level", "family", "episode", "teacher", "n_actions_turn"):
            assert key in r["meta"]


@needs_corpus
def test_corpus_renders_through_kernel_path(small_corpus):
    """The exact code path the SFT kernel runs at collate time: normalize +
    chat-template render with the target as a clean suffix."""
    transformers = pytest.importorskip("transformers")  # noqa: F841
    import sys

    sft_dir = str(REPO_ROOT / "submission" / "_sft_k3")
    if sft_dir not in sys.path:
        sys.path.insert(0, sft_dir)
    from sft_common import normalize_sample, render
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CORPUS_V3 / "tokenizer_bundle"))
    tools = json.loads((CORPUS_V3 / "tools.json").read_text())
    out, _stats = small_corpus
    rows = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    for r in rows[:3]:
        msgs, tgt = normalize_sample(r)
        full, target = render(tok, msgs, tgt, tools)
        assert full.endswith(target)
        assert len(target) > 0
