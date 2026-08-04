"""v2 invariants: new-family game validity, episode revision arcs, narrator
determinism/variety, corpus schema + gates.

Run from repo root:
    .venv/bin/python -m pytest tools/synthgen/tests -q
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from families_v2 import V2_FAMILIES, ALL_FAMILIES, sample_game_v2, make_sim
from render_v2 import render_game_py_v2, write_env_dir_v2
from episodes import build_episode, episode_actions
from validate import replay, validate_game
from validate_v2 import validate_episode
from narrate import draw_target_budget, render_turn

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_V3 = REPO_ROOT / "submission" / "_sft_k3" / "corpus_v3"

SEED = 424243


# ---------------------------------------------------------------------------
# sampling determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", V2_FAMILIES)
def test_v2_sampling_is_deterministic(family):
    a = sample_game_v2(family, 0, SEED)
    b = sample_game_v2(family, 0, SEED)
    assert a == b
    assert render_game_py_v2(a) == render_game_py_v2(b)


@pytest.mark.parametrize("family", V2_FAMILIES)
def test_v2_different_indices_differ(family):
    a = sample_game_v2(family, 0, SEED)
    b = sample_game_v2(family, 1, SEED)
    assert a["game_id"] != b["game_id"]
    assert a["levels"] != b["levels"]


# ---------------------------------------------------------------------------
# engine acceptance: clean solution + scripted episode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v2_batch(tmp_path_factory):
    root = tmp_path_factory.mktemp("games_v2")
    out = {}
    for family in V2_FAMILIES:
        spec = sample_game_v2(family, 0, SEED)
        write_env_dir_v2(spec, root)
        out[family] = spec
    return root, out


@pytest.mark.parametrize("family", V2_FAMILIES)
def test_v2_game_loads_solves_deterministically(v2_batch, family):
    root, specs = v2_batch
    report = validate_game(root, specs[family])
    assert report["loaded"]
    assert report["solved"], report
    assert report["level_boundaries_ok"], report
    assert report["no_game_over"], report
    assert report["deterministic"], report


@pytest.mark.parametrize("family", V2_FAMILIES)
def test_v2_episode_replays_to_win(v2_batch, family):
    root, specs = v2_batch
    spec = specs[family]
    ep = build_episode(spec, 0)
    rep = validate_episode(root, spec, ep)
    assert rep["episode_ok"], rep


def test_episode_deterministic():
    spec = sample_game_v2("replay", 1, SEED)
    a = build_episode(spec, 0)
    b = build_episode(spec, 0)
    assert episode_actions(a) == episode_actions(b)
    assert a["clean"] == b["clean"]


def test_wrong_actions_do_not_win_v2(v2_batch):
    root, specs = v2_batch
    for family, spec in specs.items():
        bad = [{"id": "ACTION1"}] * 5
        records = replay(root, spec["game_id"], bad)
        assert records[-1]["state"] != "WIN"
        assert records[-1]["score"] < len(spec["levels"])


# ---------------------------------------------------------------------------
# revision-arc structure
# ---------------------------------------------------------------------------


def _first_noncleanepisode(family):
    for idx in range(12):
        spec = sample_game_v2(family, idx, SEED)
        ep = build_episode(spec, 0)
        if not ep["clean"]:
            return spec, ep
    raise AssertionError("no non-clean episode in 12 tries")


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_revision_arc_present(family):
    spec, ep = _first_noncleanepisode(family)
    phases = [t["phase"] for lvl in ep["levels"] for t in lvl]
    assert "revise" in phases, phases
    sems = [t["sem"] for lvl in ep["levels"] for t in lvl]
    # a committed wrong prediction exists somewhere before the revision
    assert any(s.get("wrong_pred") for s in sems)
    # the revision turn names the abandoned hypothesis
    rev = next(s for s in sems if s.get("revision"))
    assert rev["revision"]["frm"] != rev["revision"]["to"]
    # wrong-hypothesis turns are marked as such
    assert any(s["hyp"]["kind"] == "wrong" for s in sems)


def test_clean_fraction_mixed():
    """Across many games, both clean and revision episodes occur."""
    flags = []
    for family in ("replay", "carry"):
        for idx in range(8):
            spec = sample_game_v2(family, idx, SEED)
            flags.append(build_episode(spec, 0)["clean"])
    assert any(flags) and not all(flags)


def test_sim_matches_solver_win():
    """Family sims accept their own reference solutions."""
    for family in V2_FAMILIES:
        spec = sample_game_v2(family, 2, SEED)
        for li, acts in enumerate(spec["solution_levels"]):
            sim = make_sim(spec, li)
            for a in acts:
                sim.step(5 if a["id"] == "ACTION5" else int(a["id"][-1]))
            assert sim.won, (family, li)


# ---------------------------------------------------------------------------
# narrator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_sem():
    spec = sample_game_v2("replay", 0, SEED)
    ep = build_episode(spec, 0)
    turn = ep["levels"][0][0]
    return spec, turn["sem"]


def test_narrator_deterministic(sample_sem):
    spec, sem = sample_sem
    a = render_turn(spec, sem, random.Random(7), 1200)
    b = render_turn(spec, sem, random.Random(7), 1200)
    assert a == b


def test_narrator_seed_varies_phrasing(sample_sem):
    spec, sem = sample_sem
    a = render_turn(spec, sem, random.Random(1), 1200)[0]
    b = render_turn(spec, sem, random.Random(2), 1200)[0]
    assert a != b


def test_narrator_budget_scales(sample_sem):
    spec, sem = sample_sem
    short = render_turn(spec, sem, random.Random(3), 150)[0]
    long = render_turn(spec, sem, random.Random(3), 3000)[0]
    assert len(long) > len(short) * 2


def test_budget_draw_distribution():
    rng = random.Random(0)
    draws = sorted(draw_target_budget(rng) for _ in range(4000))
    med = draws[len(draws) // 2]
    assert 1500 < med < 2300
    assert draws[0] >= 150 and draws[-1] <= 4600


# ---------------------------------------------------------------------------
# corpus v2 (needs the editable harness + corpus_v3)
# ---------------------------------------------------------------------------

try:
    import corpus_v2  # noqa: F401
    import validate_v2 as vv2
    _corpus_ok = True
except Exception:  # noqa: BLE001
    _corpus_ok = False
needs_corpus = pytest.mark.skipif(
    not _corpus_ok or not CORPUS_V3.exists(),
    reason="corpus_v3 reference or harness package not present")


@pytest.fixture(scope="module")
def small_corpus_v2(tmp_path_factory):
    if not _corpus_ok:
        pytest.skip("corpus builder unavailable")
    root = tmp_path_factory.mktemp("g")
    reports = []
    for family in ("replay", "mirror", "rules"):
        for idx in range(2):
            spec = sample_game_v2(family, idx, SEED)
            write_env_dir_v2(spec, root)
            rep = validate_game(root, spec)
            ep = build_episode(spec, 0)
            rep.update(vv2.validate_episode(root, spec, ep))
            rep["ok"] = rep["ok"] and rep["episode_ok"]
            reports.append(rep)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reports))
    out = tmp_path_factory.mktemp("c")
    stats = corpus_v2.build_corpus_v2(root, out, holdout_family="mirror",
                                      seed=1, with_tokens=True)
    return out, stats


@needs_corpus
def test_v2_holdout_split_integrity(small_corpus_v2):
    out, _stats = small_corpus_v2
    gate = vv2.gate_split_integrity(out, "mirror")
    assert gate["ok"], gate


@needs_corpus
def test_v2_corpus_record_schema(small_corpus_v2):
    out, _stats = small_corpus_v2
    rows = []
    for name in ("train", "val", "holdout_transfer"):
        rows += [json.loads(l) for l in (out / f"{name}.jsonl").read_text().splitlines()]
    assert rows
    ref_system = json.loads(
        (CORPUS_V3 / "train.jsonl").open().readline())["messages"][0]["content"]
    for r in rows:
        assert sorted(r.keys()) == ["messages", "meta", "target"]
        msgs = r["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == ref_system
        t = r["target"]
        assert t["role"] == "assistant"
        assert sorted(t.keys()) == ["content", "reasoning_content", "role", "tool_calls"]
        assert len(t["reasoning_content"]) > 100
        assert t["content"].startswith("World model:")
        (tc,) = t["tool_calls"]
        assert tc["function"]["name"] == "python"
        assert "action(" in tc["function"]["arguments"]["code"]
        for m in msgs:
            if m["role"] == "tool":
                payload = json.loads(m["content"])
                assert payload["tool"] == "python"
                assert payload["returncode"] == 0
        for key in ("game", "level", "family", "episode", "phase", "hyp",
                    "clean_episode", "qwen_target_text"):
            assert key in r["meta"]


@needs_corpus
def test_v2_revision_metadata_present(small_corpus_v2):
    out, _stats = small_corpus_v2
    rows = []
    for name in ("train", "val", "holdout_transfer"):
        rows += [json.loads(l) for l in (out / f"{name}.jsonl").read_text().splitlines()]
    non_clean = [r for r in rows if not r["meta"]["clean_episode"]]
    assert any(r["meta"]["revised"] for r in non_clean)
    assert any(r["meta"]["phase"] == "misstep" for r in non_clean)


@needs_corpus
def test_v2_gate_functions_run(small_corpus_v2):
    """Gate machinery runs and reports the right shape (thresholds are only
    meaningful on the full corpus; here we assert structure + kernel render)."""
    out, _stats = small_corpus_v2
    rows = [r for split in vv2._load_rows(out).values() for r in split]
    g_len = vv2.gate_length_distribution(rows)
    assert set(g_len["checks"]) == {"mean", "p10", "p90"}
    g_ov = vv2.gate_overlap(rows, n_pairs=50)
    assert g_ov["cross_game"]["n"] > 0
    g_enc = vv2.gate_kernel_encode(out, sample_n=4)
    assert g_enc["suffix_render_ok"].split("/")[0] == g_enc["suffix_render_ok"].split("/")[1]
    assert all(c["untruncatable"] == 0 for c in g_enc["caps"].values())


@needs_corpus
def test_v2_renders_through_kernel_path(small_corpus_v2):
    transformers = pytest.importorskip("transformers")  # noqa: F841
    import sys

    sft_dir = str(REPO_ROOT / "submission" / "_sft_k3")
    if sft_dir not in sys.path:
        sys.path.insert(0, sft_dir)
    from sft_common import normalize_sample, render
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CORPUS_V3 / "tokenizer_bundle"))
    tools = json.loads((CORPUS_V3 / "tools.json").read_text())
    out, _stats = small_corpus_v2
    rows = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    for r in rows[:3]:
        msgs, tgt = normalize_sample(r)
        full, target = render(tok, msgs, tgt, tools)
        assert full.endswith(target)
        assert len(target) > 0
