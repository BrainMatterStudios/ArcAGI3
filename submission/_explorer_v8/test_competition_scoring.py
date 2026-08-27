"""Does a grinder/specialist win CONVERT TO SCORE? — the authoritative gate.

Standing law: scorecard mechanics are only valid through
``taaf.competition_arcade.CompetitionArcadeServer`` — the real
``arc_agi.server.create_app(..., competition_mode=True)`` REST app, i.e. the
same accounting the Kaggle gateway runs. Neither the grinder's internal
counters nor the harness's own ``GameRun`` bookkeeping is evidence about score.

WHY THIS FILE EXISTS (smoke #2, kernel arc3-v8-smoke v2, 2026-08-26):
ft09 was cracked by the early specialist probe (6/6 levels, 1233 engine
actions) and banked (75-action replay), yet the run summary printed
``state=gave_up level=0/6 score=0.00``. The harness's ``GameRun`` counts only
actions executed through ``Game.execute_action``; the grinder steps
``game.env`` directly, so ``GameRun`` never sees the win. The same kernel's
log also carries the reconciliation the framework itself performs:

    R11.12: score mismatch for ft09-0d8bbf25: framework=0.0000, engine=100.0000

``framework`` is ``GameRun._compute_final_score()`` (LLM actions only);
``engine`` is the arc_agi scorecard attached to the env by
``GameAPI._start_game`` (``arcade.make(..., scorecard_id=...)``), which counts
every step through that env — the grinder's included. These tests pin that
down on the competition server rather than on a warning string.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python -m pytest -q -s \
          submission/_explorer_v8/test_competition_scoring.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

import pytest  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(ROOT, "submission/_search_core"),
           os.path.join(ROOT, "submission/_explorer_floor"),
           os.path.join(ROOT, "submission/_duck38_v12_bank"),
           os.path.join(ROOT, "submission/_adopt/taaf-src/src/"
                              "tufa-arc-agi-framework/src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import graft_explorer as v7  # noqa: E402
import graft_explorer_v8 as v8  # noqa: E402
import taaf.game  # noqa: E402
import taaf.game_api  # noqa: E402
from taaf.competition_arcade import CompetitionArcadeServer  # noqa: E402

ENV_DIR = os.path.join(ROOT, "environment_files")


class _Solver:
    max_actions_per_game = None

    @staticmethod
    def soft_time_remaining_seconds():
        return None


class GrinderSession:
    """The subset of _HarnessGameSession the graft touches, wrapped around a
    REAL taaf GameAPI whose env is bound to a competition scorecard."""

    def __init__(self, game):
        self.game = game
        self.solver = _Solver()
        self.stop_event = threading.Event()
        self.action_count = 0
        self.analysis_step = 0
        self.wrote_state = 0

    def runtime_limit_reached(self):
        return False

    def write_runtime_state(self):
        self.wrote_state += 1


@pytest.fixture(autouse=True)
def _clean_run_state():
    saved = {k: os.environ.get(k) for k in list(os.environ) if k.startswith("EXPLORER")}
    saved_grind, saved_poll = v7._grind, v7._maybe_grind
    v7._RUN_T0 = None
    v7._GRIND_WALL_SPENT[0] = 0.0
    v8._PROBE_GATE = threading.BoundedSemaphore(2)
    try:
        import graft_bank

        graft_bank._BankingKillSwitch.tripped = False
    except Exception:  # noqa: BLE001
        pass
    yield
    for k in [k for k in list(os.environ) if k.startswith("EXPLORER")]:
        del os.environ[k]
    os.environ.update({k: v for k, v in saved.items() if v is not None})
    v7._grind, v7._maybe_grind = saved_grind, saved_poll
    v7._RUN_T0 = None
    v7._GRIND_WALL_SPENT[0] = 0.0


def engine_score(game):
    """The AUTHORITATIVE number: the arc_agi scorecard the env writes to.

    Same object ``GameAPI._finish_game`` reconciles against (R11.12), and the
    same accounting the competition gateway keeps for the scored run.

    NOTE a real competition-mode property, reproduced here: reading a live
    scorecard is 403 FORBIDDEN (``GET /api/scorecard/<id>`` is refused while
    the run is open — you cannot poll your own score mid-competition). The
    card is only readable by CLOSING it, which is exactly what
    ``_CompetitionScorecard.finish_run`` does at teardown
    (taaf/game_api.py:79-86)."""
    card = game._competition_scorecard.finish_run()
    assert card is not None, "closing the competition scorecard returned nothing"
    engine_game_id = game.env.environment_info.game_id
    env_list = card.find_environment(engine_game_id)
    assert env_list is not None and env_list.runs, f"no runs for {engine_game_id}"
    # THE authoritative field is the ENVIRONMENT-level score, which is the MAX
    # over that environment's plays (``runs``). Reading runs[0] reads the FIRST
    # play — for a grinder crack that is the search play, which scores ~3.5
    # while the banked replay play scores ~100. Measured on this server:
    #   runs = 2, runs[0].score = 3.51, environment score = 100.0
    return env_list, card


def run_grinder_on_competition_server(stem: str, *, budget_s: float = 2000.0,
                                     via: str = "early"):
    """Play ONE game on the competition REST arcade, driven only by the v8
    grinder (no LLM), and return (game, engine run, framework score)."""
    with CompetitionArcadeServer(game_ids=[stem], environments_dir=ENV_DIR) as server:
        game_id = server.exposed_game_ids[0]
        game = taaf.game_api.GameAPI(env_name=game_id, arcade_spec=server.arcade_spec)
        session_state = taaf.game.RunSession(record_intermediate_states=False)
        game.start_game(session_state)
        session = GrinderSession(game)

        os.environ["EXPLORER"] = "1"
        os.environ["EXPLORER_V8"] = "1"
        os.environ["EXPLORER_GRIND_TIME_S"] = str(int(budget_s))
        os.environ["EXPLORER_OWNED_TIME_S"] = str(int(budget_s))
        os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = str(int(budget_s * 2))
        v7._RUN_T0 = time.monotonic()

        t0 = time.time()
        if via == "early":
            outcome = v8.early_specialist_probe(session)
            xs = v7._session_state(session)
        else:
            # the STALL-triggered generic path: tu93 has no specialist, its
            # crack comes from the nbfs_macros lane inside a normal engagement
            xs = v7._session_state(session)
            xs["worker_thread"] = threading.get_ident()
            v8._grind_v8(session, xs, 1)
            outcome = f"stall_grind:{xs['diag'].get('v8_bank_result')}"
        wall = time.time() - t0

        env, card = engine_score(game)
        framework = game.game_run._compute_final_score()
        print(f"\n    [{stem}] outcome={outcome} wall={wall:.1f}s "
              f"grinder_actions={xs['diag']['grinder_actions']}")
        print(f"    [{stem}] ENGINE scorecard (AUTHORITATIVE): score={env.score} "
              f"levels={env.levels_completed}/{env.level_count} "
              f"actions={env.actions} resets={env.resets} "
              f"plays={len(env.runs)} completed={env.completed}")
        for i, r in enumerate(env.runs):
            print(f"        play[{i}] score={r.score:.3f} "
                  f"levels={r.levels_completed} actions={r.actions} "
                  f"apl={list(getattr(r, 'actions_per_level', None) or [])}")
        print(f"    [{stem}] FRAMEWORK GameRun: score={framework} "
              f"levels={game.game_run.levels_completed} "
              f"actions={sum(game.game_run.actions_per_level)}")
        return game, env, framework, xs, outcome


# ==========================================================================
# The gate: a specialist win must be worth points on the competition server
# ==========================================================================

def test_J_ft09_specialist_win_scores_on_the_competition_server():
    game, env, framework, xs, outcome = run_grinder_on_competition_server("ft09")
    assert outcome.startswith("engaged(ft09_gf2):game_won"), outcome
    assert len(xs["grind_unlocked_levels"]) == 6, xs["grind_unlocked_levels"]
    assert xs["diag"].get("games_banked_by_grinder") == 1, xs["diag"]
    # THE assertion: the scored quantity, not the grinder's own counter.
    assert env.score > 0, (
        f"specialist win did not convert to score on the competition server: "
        f"{env.score}")
    assert env.levels_completed == 6, env.levels_completed
    assert env.completed is True
    assert env.score >= 90.0, f"expected a near-perfect banked replay, got {env.score}"
    # the banked replay is a SEPARATE play and the card takes the max
    assert len(env.runs) >= 2, f"banking should open a second play, got {len(env.runs)}"
    assert min(r.score for r in env.runs) < env.score, [r.score for r in env.runs]
    # ... and the framework mirror is the thing that reads 0, which is why it
    # must never be used as the score read (smoke #2's misgrade).
    print(f"    [ft09] VERDICT engine={env.score} vs framework={framework}")


@pytest.mark.slow
def test_J_tu93_generic_crack_scores_on_the_competition_server():
    """tu93 has no specialist: it is cracked by the generic nbfs_macros lane
    (84 687 engine actions offline). Over HTTP this is slow — it is the second
    half of the gate, so it runs, but it is marked slow."""
    os.environ["EXPLORER_V8_EARLY_MAX_ACTIONS"] = "24"
    game, env, framework, xs, outcome = run_grinder_on_competition_server(
        "tu93", budget_s=4000.0, via="stall")
    assert xs["diag"].get("games_won_by_grinder") == 1, (outcome, xs["diag"])
    assert env.score > 0, f"tu93 crack did not convert to score: {env.score}"
    assert env.levels_completed == 9, env.levels_completed
    print(f"    [tu93] VERDICT engine={env.score} vs framework={framework}")


def test_J_the_framework_mirror_is_not_the_scored_quantity():
    """Pin the seam itself, so no future read confuses the two accountings.

    GameRun counts only actions passed through Game.execute_action
    (taaf/game.py:497-573 — history/actions_per_level/levels_completed are all
    updated there and nowhere else). The grinder steps game.env directly
    (graft_explorer_v8.GuardedEnv.step -> env.step), which is the SAME env the
    arc_agi scorecard is attached to by GameAPI._start_game
    (taaf/game_api.py:204: arcade.make(..., scorecard_id=self._scorecard_id)).
    So: engine sees everything, framework sees only the LLM."""
    game, env, framework, xs, _ = run_grinder_on_competition_server("ft09")
    assert framework == 0.0, framework          # no LLM action was ever taken
    assert sum(game.game_run.actions_per_level) == 0
    assert game.game_run.levels_completed == 0
    assert env.score >= 90.0                     # engine: the win, scored
    assert env.actions >= 1233                   # engine counted the grind


# ==========================================================================
# The probe tax, measured on the authoritative accounting instead of assumed
# ==========================================================================

def ft09_level1_plan():
    """ft09's minimal level-1 token sequence, derived offline (no HTTP)."""
    import types

    import search_core as sc
    import specialists
    from arc_agi import Arcade, OperationMode

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    env = client.make(sc.discover_games(ENV_DIR)["ft09"])
    session = GrinderSession(types.SimpleNamespace(
        env=env, game_run=types.SimpleNamespace(game_id="ft09", state="playing")))
    xs = v7._session_state(session)
    genv = v8.GuardedEnv(env, session, xs, t0=time.monotonic(),
                         action_cap=10 ** 6, time_cap_s=1e9, owned_time_cap_s=1e9)
    prebuilt = v8._warm_and_detect(xs, genv, "ft09", sc, specialists)
    res = specialists.solve_level(prebuilt[1], prebuilt[3], 1, 60.0)
    assert res.get("solved"), res
    return list(res["handle"].path)


def play_ft09_l1(with_probe: bool, plan, extra_raw_actions: int = 0):
    """One competition-server play of ft09 level 1 through the SCORED path,
    optionally preceded by a declined (detect-only) early probe."""
    import arcengine

    with CompetitionArcadeServer(game_ids=["ft09"], environments_dir=ENV_DIR) as server:
        game_id = server.exposed_game_ids[0]
        game = taaf.game_api.GameAPI(env_name=game_id, arcade_spec=server.arcade_spec)
        game.start_game(taaf.game.RunSession(record_intermediate_states=False))
        session = GrinderSession(game)
        os.environ["EXPLORER"] = "1"
        os.environ["EXPLORER_V8"] = "1"
        v7._RUN_T0 = time.monotonic()
        probe_actions = 0
        if extra_raw_actions:
            # simulate a NON-CRACKING grind: raw env steps, the way the
            # grinder makes them, without any post-WIN reset to open a play
            for _ in range(extra_raw_actions):
                game.env.step(arcengine.GameAction.RESET, data={})
            probe_actions = extra_raw_actions
        if with_probe:
            os.environ["EXPLORER_V8_EARLY_ENGAGE"] = "0"     # detect only
            v8.early_specialist_probe(session)
            probe_actions = v7._session_state(session)["diag"]["grinder_actions"]
        for tok in plan:                                     # the SCORED path
            if tok[0] == "C":
                action = arcengine.ActionInput(
                    id=arcengine.GameAction.ACTION6,
                    data={"x": int(tok[1]), "y": int(tok[2])})
            else:
                action = arcengine.ActionInput(
                    id=arcengine.GameAction.from_id(int(tok[1])), data={})
            game.execute_action(action)
        env, _ = engine_score(game)
        print(f"    [ft09 {'PROBE' if with_probe else ('GRIND%d' % extra_raw_actions if extra_raw_actions else 'CONTROL')}] "
              f"probe_actions={probe_actions} plan={len(plan)} -> "
              f"engine score={env.score:.4f} levels={env.levels_completed} "
              f"plays={len(env.runs)} "
              f"play_scores={[round(r.score, 2) for r in env.runs]}")
        return env, probe_actions


def test_J_probe_tax_measured_on_the_competition_server():
    """A/B the ONLY thing that can sink this lever: does a DECLINED probe cost
    score on a game the LLM then completes level 1 of?

    Arm A: 89-action detect-only probe, then ft09's minimal L1 plan.
    Arm B: the same plan, no probe.
    If probe actions were billed into the LLM's play, A collapses to
    (43/(4+89))^2 x 100 / 21 = 1.02; if the accounting isolates them, A == B."""
    plan = ft09_level1_plan()
    assert len(plan) <= 8, plan
    control, _ = play_ft09_l1(False, plan)
    probed, probe_actions = play_ft09_l1(True, plan)
    assert probe_actions > 50, probe_actions        # the probe really ran
    print(f"    [TAX] control={control.score:.4f} probed={probed.score:.4f} "
          f"delta={probed.score - control.score:+.4f} "
          f"(naive billed-together prediction: "
          f"{(43 / (len(plan) + probe_actions)) ** 2 * 100 / 21:.4f})")
    assert control.score > 0, control.score
    # The measured verdict — asserted, not assumed:
    # MEASURED VERDICT (asserted so a future change to the accounting breaks
    # the build): the probe's actions ARE billed into the same play as the
    # LLM's, so the tax is real. It matches the naive prediction, which means
    # nothing isolates it — max-over-plays does NOT save a non-cracking probe,
    # because the probe's resets are level_resets and the competition guard
    # (api.py:316-334) swallows the one reset that would open a fresh play.
    tax = control.score - probed.score
    assert tax > 0, "expected a real tax; the accounting must have changed"
    assert probed.score == pytest.approx(
        (43 / (len(plan) + probe_actions)) ** 2 * 100 / 21, rel=0.05), probed.score
    assert 3.0 < tax < 4.8, f"tax {tax} outside the pre-registered 100/W bound"


def test_J_non_cracking_grind_is_billed_into_the_llm_play():
    """The same mechanism at grind scale. A grind that does NOT crack cannot
    open a fresh play, so every action it spends is charged to the level the
    LLM later completes — which is why the value case rests entirely on
    cracks that BANK (a post-WIN reset is the only reset that opens a play)."""
    plan = ft09_level1_plan()
    control, _ = play_ft09_l1(False, plan)
    ground, n = play_ft09_l1(False, plan, extra_raw_actions=1000)
    print(f"    [GRIND TAX] control={control.score:.4f} after {n} grind "
          f"actions={ground.score:.4f} (delta {ground.score - control.score:+.4f})")
    assert ground.score < control.score / 10, (control.score, ground.score)
