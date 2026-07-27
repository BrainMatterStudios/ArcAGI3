"""Offline test for the geodesic efficiency post-pass.

No GPU / no duck LLM needed: we feed the *recorded* duck trajectories
(``/tmp/repdone/artifacts/*_events.jsonl`` == ``session.viewer_events``) into the
post-pass, replay them live on ``Arcade(OFFLINE)``, and confirm:

  (a) the post-pass builds a compressed replay sequence,
  (b) the live replay reaches the duck's completed-level depth,
  (c) per-game ``max(duck, replay)`` reproduces hybrid_live_replay.py's +38%,
  (d) a simulated post-pass exception falls back to the duck score, no crash.

Run standalone:
    PATH="$HOME/.local/bin:$PATH" ENVIRONMENTS_DIR=<envdir> \
        .venv/bin/python inference/framework/test_geodesic_postpass.py
Or with pytest (auto-skips if artifacts/env files are absent).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # standalone run without pytest installed
    class _PytestStub:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def deco(fn):
                    return fn
                return deco

    pytest = _PytestStub()  # type: ignore

from inference.framework import geodesic_postpass as gp

ART = Path(os.environ.get("GEODESIC_ART_DIR", "/tmp/repdone/artifacts"))
ENVDIR = Path(
    os.environ.get("ENVIRONMENTS_DIR")
    or "/Users/ahmed/Documents/ArcAGI3/environment_files"
)

# Ground truth reproduced from scratchpad/hybrid_live_replay.py in this env.
EXPECTED_DUCK_MEAN = 1.7809
EXPECTED_HYBRID_MEAN = 2.4576
EXPECTED_GAIN_PCT = 38.0
EXPECTED_FAITHFUL = 16
EXPECTED_POSITIVE = 8


# --------------------------------------------------------------------------- #
# Scorer -- byte-identical to hybrid_live_replay.py (verified to reproduce the
# duck's 1.781 mean). Depth-weighted squared efficiency, all-levels denominator.
# --------------------------------------------------------------------------- #
def _level_score(baseline_i: float, actions_i: int) -> float:
    if actions_i <= 0:
        return 0.0
    return min((baseline_i / actions_i) ** 2 * 100.0, 100.0)


def _play_score(completed, per_level_actions, baseline, total_levels) -> float:
    num = 0.0
    for j in range(completed):
        b = baseline[j] if j < len(baseline) else baseline[-1]
        num += (j + 1) * _level_score(b, per_level_actions[j])
    den = sum(range(1, total_levels + 1))
    return num / den if den else 0.0


def _replay_with_per_level(env, seq, total_levels, action_cap):
    """Replay + capture per-level action counts (for offline scoring)."""
    import arcengine

    obs = env.reset()
    live_lc = int(getattr(obs, "levels_completed", 0) or 0)
    per_level = [0] * (total_levels + 2)
    steps = 0
    for tok in seq:
        if steps >= action_cap:
            break
        if tok[0] == "R":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(arcengine.GameAction.from_id(tok[1]))
        else:
            obs = env.step(arcengine.GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        steps += 1
        cur = min(live_lc, total_levels + 1)
        per_level[cur] += 1
        live_lc = max(live_lc, int(getattr(obs, "levels_completed", 0) or 0))
    return live_lc, per_level


def _artifacts_available() -> bool:
    return ART.is_dir() and bool(glob.glob(str(ART / "*_events.jsonl"))) and ENVDIR.is_dir()


def _compute_portfolio():
    """Run the module's post-pass logic over every recorded game; return rows."""
    from arc_agi import Arcade, OperationMode

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENVDIR))

    gid_map = {}
    for p in sorted(glob.glob(str(ART / "*_p0_events.jsonl"))):
        full = os.path.basename(p).split("_p0_")[0]
        gid_map[full.split("-")[0]] = full

    rows = []
    for gid in sorted(gid_map):
        full = gid_map[gid]
        events = [json.loads(l) for l in open(ART / f"{full}_p0_events.jsonl")]
        vd = json.load(open(ART / f"{full}_p0_viewer_data.json"))
        meta = json.load(open(glob.glob(str(ENVDIR / gid / "*" / "metadata.json"))[0]))
        baseline = meta["baseline_actions"]
        total_levels = len(baseline)
        duck_score = vd["final_score"]

        env = arc.make(gid)
        obs = env.reset()
        s0 = gp.frame_to_hash(obs.frame)

        seq, K, reachable = gp.build_replay_sequence(events, s0)
        if K == 0 or not reachable or not seq:
            rows.append(dict(gid=gid, K=K, duck=duck_score, replay_lc=0,
                             faithful=False, hybrid=duck_score, gain=0.0, seq_len=len(seq)))
            continue

        env2 = arc.make(gid)
        live_lc, per_level = _replay_with_per_level(env2, seq, total_levels, len(seq) + 5)
        faithful = live_lc >= K
        replay_score = _play_score(min(live_lc, K), per_level, baseline, total_levels)
        hybrid = max(duck_score, replay_score)
        rows.append(dict(gid=gid, K=K, duck=duck_score, replay_lc=live_lc,
                         faithful=faithful, hybrid=hybrid, gain=hybrid - duck_score,
                         seq_len=len(seq)))
    return rows


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _artifacts_available(), reason="recorded artifacts/env files absent")
def test_postpass_reproduces_38pct():
    rows = _compute_portfolio()
    n = len(rows)
    assert n >= 25, f"expected >=25 games, got {n}"

    # (a) compressed replay play produced for games the duck advanced in
    produced = [r for r in rows if r["K"] > 0 and r["seq_len"] > 0]
    assert produced, "no replay sequences produced"

    # (b) replay reaches the duck's completed depth on the faithful subset
    faithful = sum(1 for r in rows if r["faithful"])
    assert faithful == EXPECTED_FAITHFUL, f"faithful={faithful} != {EXPECTED_FAITHFUL}"

    # (c) per-game max(duck, replay) reproduces the +38% mean
    duck_mean = sum(r["duck"] for r in rows) / n
    hybrid_mean = sum(r["hybrid"] for r in rows) / n
    gain_pct = 100.0 * (hybrid_mean - duck_mean) / duck_mean
    positive = sum(1 for r in rows if r["gain"] > 1e-9)

    assert abs(duck_mean - EXPECTED_DUCK_MEAN) < 0.02, duck_mean
    assert abs(hybrid_mean - EXPECTED_HYBRID_MEAN) < 0.02, hybrid_mean
    assert abs(gain_pct - EXPECTED_GAIN_PCT) < 1.0, gain_pct
    assert positive == EXPECTED_POSITIVE, positive
    # additive invariant: hybrid never below duck for any game
    assert all(r["hybrid"] >= r["duck"] - 1e-9 for r in rows)


def test_failsafe_on_exception():
    """A post-pass failure returns cleanly and never raises (duck score kept)."""

    class _BoomArcade:
        def make(self, *a, **k):
            raise RuntimeError("simulated engine failure")

    class _FakeGame:
        _arcade = _BoomArcade()
        env_name = "tu93"
        game_id = "tu93"
        _competition_scorecard = None
        _scorecard_id = "sc-x"

    res = gp.run_geodesic_postpass(_FakeGame(), [{"type": "action", "board": [[0]]}])
    assert res["status"] == "error"
    assert "simulated engine failure" in res["error"]


def test_disabled_flag(monkeypatch):
    monkeypatch.setenv("TAAF_GEODESIC_POSTPASS", "0")
    assert gp.postpass_enabled() is False
    res = gp.run_geodesic_postpass(object(), [])
    assert res["status"] == "disabled"


def test_build_sequence_pure():
    """Pure builder: no milestone -> empty seq, K=0, reachable."""
    events = [
        {"type": "action", "action_name": "ACTION4", "action_display": "RIGHT",
         "board": [[1]], "level_completed": False},
    ]
    s0 = gp.board_hash([[0]])
    seq, K, reachable = gp.build_replay_sequence(events, s0)
    assert K == 0 and seq == [] and reachable is True


if __name__ == "__main__":
    if not _artifacts_available():
        raise SystemExit(f"artifacts/env not found (ART={ART}, ENVDIR={ENVDIR})")
    rows = _compute_portfolio()
    n = len(rows)
    duck_mean = sum(r["duck"] for r in rows) / n
    hybrid_mean = sum(r["hybrid"] for r in rows) / n
    gain_pct = 100.0 * (hybrid_mean - duck_mean) / duck_mean
    faithful = sum(1 for r in rows if r["faithful"])
    positive = sum(1 for r in rows if r["gain"] > 1e-9)
    print(f"games                = {n}")
    print(f"duck mean            = {duck_mean:.4f} (expect {EXPECTED_DUCK_MEAN})")
    print(f"HYBRID mean          = {hybrid_mean:.4f} (expect {EXPECTED_HYBRID_MEAN})")
    print(f"gain                 = {gain_pct:+.1f}% (expect +{EXPECTED_GAIN_PCT}%)")
    print(f"faithful replays     = {faithful}/{n} (expect {EXPECTED_FAITHFUL})")
    print(f"games w/ positive gain= {positive} (expect {EXPECTED_POSITIVE})")
    assert all(r["hybrid"] >= r["duck"] - 1e-9 for r in rows), "ADDITIVE INVARIANT VIOLATED"
    print("ADDITIVE INVARIANT held: hybrid >= duck for every game.")
