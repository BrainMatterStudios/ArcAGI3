#!/usr/bin/env python3
"""Offline validation of the banking graft against the REAL engine (no GPU).

Layers:
  1. prune_winning_trace unit checks (synthetic traces).
  2. Engine integration on sb26 with the recorded 142-action win trace
     (docs/test-artifacts-2026-08-02/sb26_win_trace.json), under
     ONLY_RESET_LEVELS=true (competition parity):
       play -> WIN, post-WIN RESET must open a FRESH play, replay the pruned
       plan with per-step grid/level verification -> WIN again, and the
       scorecard's card score must equal the max over the two plays with the
       replay's action count strictly smaller.
  3. Seam drift guard: verify_seam() against BOTH bundles' solver sources.

Run:  .venv/bin/python submission/_duck38_v12_bank/validate_offline.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLES = Path(
    "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
    "f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad/bundles"
)

os.environ["ONLY_RESET_LEVELS"] = "true"  # competition parity

sys.path.insert(0, str(Path(__file__).parent))


def bundle_paths(name: str) -> list[str]:
    root = BUNDLES / name / "src"
    out = []
    for repo in sorted(root.iterdir(), reverse=True):
        for cand in (repo / "src", repo):
            if cand.is_dir():
                out.append(str(cand))
    return out


def check_prune_units() -> None:
    import arcengine
    from graft_bank import BankingPlanError, TraceStep, prune_winning_trace

    A = arcengine.GameAction.ACTION1
    R = arcengine.GameAction.RESET
    W = arcengine.GameState.WIN
    P = arcengine.GameState.NOT_FINISHED

    g0 = ((0,),)
    g1 = ((1,),)
    g2 = ((2,),)
    g3 = ((3,),)

    def step(aid, grid, lv, st=P):
        return TraceStep(aid, {}, grid, lv, st)

    # no-op (same grid, same level) pruned; level-advance kept
    trace = [step(A, g0, 0), step(A, g1, 0), step(A, g2, 1), step(A, g3, 2, W)]
    plan = prune_winning_trace(trace, g0, 2)
    assert [s.grid for s in plan] == [g1, g2, g3], plan
    # RESET voids the pending level segment
    trace = [step(A, g1, 0), step(R, g0, 0), step(A, g2, 1, P), step(A, g3, 2, W)]
    plan = prune_winning_trace(trace, g0, 2)
    assert [s.grid for s in plan] == [g2, g3], plan
    # non-WIN ending raises
    try:
        prune_winning_trace([step(A, g1, 0)], g0, 1)
        raise AssertionError("expected BankingPlanError")
    except BankingPlanError:
        pass
    # trailing actions after last level advance raise
    try:
        prune_winning_trace([step(A, g1, 1, P), step(A, g2, 1, W)], g0, 1)
    except BankingPlanError:
        pass
    else:
        raise AssertionError("expected trailing-action BankingPlanError")
    print("PASS prune unit checks")


def check_engine_integration() -> None:
    """Full banking flow through the COMPETITION-parity localhost server —
    the same arc_agi REST app (with the competition RESET guard and per-play
    scorecard accounting) the Kaggle gateway runs. The plain OFFLINE wrapper
    is NOT sufficient: its play accounting differs (first play never opens a
    card under ONLY_RESET_LEVELS), discovered 2026-08-17."""
    import arc_agi
    import arcengine
    from graft_bank import TraceStep, _grid_from_frame_raw, prune_winning_trace

    sys.path.insert(0, str(REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"))
    from taaf.competition_arcade import CompetitionArcadeServer

    server = CompetitionArcadeServer(
        game_ids=["sb26-7fbdac44"],
        environments_dir=str(REPO / "environment_files"),
    ).start()
    client = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=server.base_url,
        arc_api_key=server.api_key,
    )
    card_id = client.open_scorecard()
    env = client.make("sb26-7fbdac44", scorecard_id=card_id)
    assert env is not None

    actions = json.load(open(REPO / "docs/test-artifacts-2026-08-02/sb26_win_trace.json"))

    def to_engine(a: dict):
        aid = getattr(arcengine.GameAction, a["name"])
        data = {k: v for k, v in a.items() if k != "name"}
        return aid, data

    resp = env.reset()  # competition-standard game open
    initial_grid = _grid_from_frame_raw(resp)

    trace: list[TraceStep] = []

    def record(aid, data):
        r = env.step(aid, data=data)
        trace.append(
            TraceStep(aid, data, _grid_from_frame_raw(r), int(r.levels_completed), r.state)
        )
        return r

    # Prunable content, duck-style: a long exploration segment voided by a
    # mid-level RESET. The curated win trace alone hits the 100-point
    # efficiency cap on both plays; the waste segment drags the ORIGINAL
    # play below the cap so the replay's uplift is measurable (real duck
    # play is exactly this shape: heavy exploration before the win path).
    for _ in range(20):
        for a in actions[:3]:
            record(*to_engine(a))
    record(arcengine.GameAction.RESET, {})  # level reset: voids everything above

    for a in actions:
        resp = record(*to_engine(a))
        if resp.state == arcengine.GameState.WIN:
            break
    assert trace and trace[-1].state == arcengine.GameState.WIN, (
        f"recorded trace did not reach WIN (last state {trace[-1].state})")
    original = sum(1 for s in trace if s.action_id != arcengine.GameAction.RESET)
    levels = int(resp.levels_completed)

    plan = prune_winning_trace(trace, initial_grid, levels)
    print(f"engine: WIN, {original} non-RESET actions -> pruned plan {len(plan)} actions")
    assert len(plan) < original, "pre-RESET junk segment must be voided"

    # post-WIN RESET must open a FRESH play under ONLY_RESET_LEVELS=true
    resp = env.step(arcengine.GameAction.RESET, data={})
    assert int(resp.levels_completed) == 0, resp.levels_completed
    assert resp.state != arcengine.GameState.WIN
    print("engine: post-WIN RESET opened a fresh play (levels=0)")

    for i, s in enumerate(plan, 1):
        resp = env.step(s.action_id, data=dict(s.action_data))
        assert _grid_from_frame_raw(resp) == s.grid, f"frame divergence at {i}/{len(plan)}"
        assert int(resp.levels_completed) == s.levels_completed, f"level divergence at {i}"
    assert resp.state == arcengine.GameState.WIN, resp.state
    print(f"engine: replay of {len(plan)} actions reached WIN with zero divergence")

    # Server-side scorecard (the SAME accounting the gateway runs): the card
    # must show 2 plays, the replay play with fewer actions, and the official
    # aggregate score must be the max over per-play scores.
    from arc_agi.scorecard import EnvironmentScorecard

    mgr = server._arcade.scorecard_manager
    sc = mgr.scorecards[card_id]
    card = sc.cards[env.environment_info.game_id]
    assert card.total_plays >= 2, f"expected >=2 plays, got {card.total_plays}"
    print(f"server: card plays={card.total_plays} actions-per-play={card.actions} "
          f"levels-per-play={card.levels_completed}")
    assert card.actions[-1] < card.actions[0], f"replay should use fewer actions: {card.actions}"
    assert card.levels_completed[-1] == card.levels_completed[0], card.levels_completed

    agg = EnvironmentScorecard.from_scorecard(sc, list(server._arcade.available_environments))
    (game_scores,) = [e for e in agg.environments if e.id == env.environment_info.game_id]
    per_play = [r.score for r in game_scores.runs]
    assert game_scores.score == max(per_play), (game_scores.score, per_play)
    orig, replay = game_scores.runs[0], game_scores.runs[-1]
    # The curated sb26 trace is so far above baseline that BOTH plays hit the
    # 100 play-cap (the 115 per-level cap lets efficient levels subsidize the
    # junked level — measured 2026-08-17). The uplift mechanism is therefore
    # asserted at LEVEL granularity, where real duck wins live (3-15 range):
    assert all(b >= a for a, b in zip(orig.level_scores, replay.level_scores)), (
        orig.level_scores, replay.level_scores)
    assert replay.level_scores[0] > orig.level_scores[0], (
        "junked level must improve", orig.level_scores[0], replay.level_scores[0])
    print(f"server: per-play scores {per_play} -> game score {game_scores.score} (max)")
    print(f"server: junked-level score {orig.level_scores[0]:.2f} -> {replay.level_scores[0]:.2f} "
          f"on the banked replay; all levels non-regressing")
    server.stop()


def check_seam_both_bundles() -> None:
    import hashlib
    import importlib
    import inspect

    for name in ("june", "anim"):
        saved = list(sys.path)
        try:
            for p in bundle_paths(name):
                sys.path.insert(0, p)
            for mod in [m for m in list(sys.modules) if m.startswith("inference")]:
                del sys.modules[mod]
            solver_mod = importlib.import_module("inference.framework.solver")
            src = inspect.getsource(solver_mod.HarnessSolver._play_one)
            digest = hashlib.blake2b(src.encode()).hexdigest()
            from graft_bank import STOCK_PLAY_ONE_SRC_HASH
            assert digest == STOCK_PLAY_ONE_SRC_HASH, f"{name}: seam drift {digest[:16]}"
            print(f"PASS seam pin matches {name} bundle _play_one")
        finally:
            sys.path[:] = saved
            for mod in [m for m in list(sys.modules) if m.startswith("inference")]:
                del sys.modules[mod]


def main() -> None:
    for p in bundle_paths("june"):
        sys.path.append(p)
    check_prune_units()
    check_engine_integration()
    check_seam_both_bundles()
    print("ALL OFFLINE VALIDATION PASSED")


if __name__ == "__main__":
    main()
