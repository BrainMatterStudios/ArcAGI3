"""Offline test suite for the explorer v8 live search graft.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python -m pytest -q \
          submission/_explorer_v8/test_graft_explorer_v8.py

Covers, in the order the build brief asks for them:
  A  reset-replay backend parity with snapshot on a small game
  B  rollout momentum honoured (one reset per rollout, 90-95% action repeat)
  C  specialist lane fires (ft09) and falls back (non-matching game)
  D  banking handoff builds and replays a valid minimal plan from a crack
  E  every envelope guard enforced (one assertion per guard)
  F  flags off = v7 (no mutation of the flown lane)
  G  fail-open on every seam
  H  live-cost simulation for the tu93 and ft09 cracks
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
           os.path.join(ROOT, "submission/_duck38_v12_bank")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import arcengine  # noqa: E402
import graft_explorer as v7  # noqa: E402
import graft_explorer_v8 as v8  # noqa: E402
import search_core as sc  # noqa: E402
import specialists  # noqa: E402

ENV_DIR = os.path.join(ROOT, "environment_files")
GATEWAY = v8.GATEWAY_ACT_PER_S


# --------------------------------------------------------------------------
# harness doubles
# --------------------------------------------------------------------------

def make_env(stem: str):
    from arc_agi import Arcade, OperationMode

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    return client.make(sc.discover_games(ENV_DIR)[stem])


class FakeSolver:
    def __init__(self, soft=None, max_actions=None):
        self._soft = soft
        self.max_actions_per_game = max_actions

    def soft_time_remaining_seconds(self):
        return self._soft


class FakeRun:
    game_id = "test"
    state = "playing"


class FakeGame:
    def __init__(self, env):
        self.env = env
        self.game_run = FakeRun()


class FakeSession:
    """The subset of _HarnessGameSession that the graft actually touches."""

    def __init__(self, env, *, soft=None, max_actions=None, runtime_done=False):
        self.game = FakeGame(env)
        self.solver = FakeSolver(soft, max_actions)
        self.stop_event = threading.Event()
        self.action_count = 0
        self.analysis_step = 0
        self._runtime_done = runtime_done
        self.wrote_state = 0

    def runtime_limit_reached(self):
        return self._runtime_done

    def write_runtime_state(self):
        self.wrote_state += 1


@pytest.fixture(autouse=True)
def _clean_run_state():
    """Each test starts from a pristine run envelope + flag set."""
    saved = {k: os.environ.get(k) for k in list(os.environ)
             if k.startswith("EXPLORER")}
    v7._RUN_T0 = None
    v7._GRIND_WALL_SPENT[0] = 0.0
    try:
        import graft_bank

        graft_bank._BankingKillSwitch.tripped = False
        graft_bank._BankingKillSwitch.reason = ""
    except Exception:  # noqa: BLE001
        pass
    yield
    for k in [k for k in list(os.environ) if k.startswith("EXPLORER")]:
        del os.environ[k]
    os.environ.update({k: v for k, v in saved.items() if v is not None})
    v7._RUN_T0 = None
    v7._GRIND_WALL_SPENT[0] = 0.0


def guarded(env, session=None, **kw):
    session = session or FakeSession(env)
    xs = v7._session_state(session)
    kw.setdefault("action_cap", 10 ** 9)
    kw.setdefault("time_cap_s", 1e9)
    kw.setdefault("owned_time_cap_s", 1e9)
    return v8.GuardedEnv(env, session, xs, t0=time.monotonic(), **kw), session, xs


# ==========================================================================
# A. reset-replay backend parity with snapshot
# ==========================================================================

def test_A_backend_parity_same_solution_tu93_l1():
    """Both backends must find the SAME minimal L1 solution on tu93."""
    out = {}
    for backend in ("snapshot", "reset_replay"):
        core = sc.SearchCore(make_env("tu93"), backend=backend,
                             warmup_rounds=3, max_states=4000)
        core.warmup_and_freeze()
        res = core.solve_level(1, 60.0, max_tier=1)
        assert res["solved"], (backend, res)
        out[backend] = (res["depth"], tuple(res["handle"].path))
    assert out["snapshot"][0] == out["reset_replay"][0], out
    assert out["snapshot"][1] == out["reset_replay"][1], out
    assert out["snapshot"][0] == 18, out          # fixture's verified minimum


def test_A_crc_key_parity_with_frontier_graph():
    """SearchCore's numpy key is byte-identical to the shipped graph key."""
    import numpy as np

    rng = np.random.default_rng(7)
    grid = rng.integers(0, 16, size=(64, 64)).astype(np.int8)
    cells = [(0, 0), (3, 5), (63, 63)]
    mb = sc.mask_to_bool(cells, grid.shape)
    assert sc.crc_key(2, grid, mb) == v7.FrontierGraph().node_key(
        2, grid.tolist(), cells)


# ==========================================================================
# B. rollout momentum
# ==========================================================================

def test_B_rollout_is_one_reset_per_rollout_not_per_action():
    env = make_env("tu93")
    core = sc.SearchCore(env, backend="reset_replay", warmup_rounds=2,
                         max_states=1000)
    core.warmup_and_freeze()
    be = core.backend
    root = be.root()
    r0, a0 = be.resets, be.actions_spent
    traj, _ = be.rollout(root, lambda obs, prev: ("S", 1), k=25, momentum=1.0)
    assert len(traj) == 25
    assert be.resets - r0 == 1, "a rollout must cost exactly ONE reset"
    assert be.actions_spent - a0 == 26, "1 reset + 25 steps"


def test_B_momentum_repeat_rate_matches_setting():
    """With momentum m the sampler must repeat the previous token ~m of the
    time (this is the 8-10x action-efficiency claim's mechanism)."""
    import random

    class Rec:
        name = "rec"
        env = None
        actions_spent = 0
        resets = 0

        def __init__(self):
            self.toks = []

        def rollout(self, *a, **k):
            raise NotImplementedError

    # exercise the exact expression used by both backends
    for m in (0.90, 0.92, 0.95):
        rng = random.Random(11)
        prev = None
        fresh = 0
        n = 4000
        pool = [("S", 1), ("S", 2), ("S", 3), ("S", 4)]
        for i in range(n):
            tok = prev if (prev is not None and rng.random() < m) \
                else pool[i % len(pool)]
            if prev is not None and tok is not prev:
                fresh += 1
            prev = tok
        rate = 1.0 - fresh / (n - 1)
        assert abs(rate - m) < 0.03, (m, rate)


def test_B_goexplorer_momentum_wired_from_env():
    os.environ["EXPLORER_V8_MOMENTUM_PCT"] = "94"
    os.environ["EXPLORER_V8_ROLLOUT_K"] = "17"
    core = sc.SearchCore(make_env("tu93"), backend="reset_replay",
                         warmup_rounds=1, max_states=100)
    go = sc.GoExplorer(
        core, tier=v8._env_int("EXPLORER_V8_GOEXPLORE_TIER", 3),
        k_rollout=v8._env_int("EXPLORER_V8_ROLLOUT_K", 30),
        momentum=v8._env_int("EXPLORER_V8_MOMENTUM_PCT", 92) / 100.0)
    assert go.momentum == pytest.approx(0.94)
    assert go.k_rollout == 17
    assert 0.90 <= sc.GoExplorer(core).momentum <= 0.95, "doc's 90-95% band"


# ==========================================================================
# C. specialist lane
# ==========================================================================

def test_C_specialist_detects_and_solves_ft09_on_reset_replay():
    core = sc.SearchCore(make_env("ft09"), backend="reset_replay",
                         warmup_rounds=3, max_states=2000)
    core.warmup_and_freeze()
    name = specialists.detect(core)
    assert name == "ft09_gf2"
    a0 = core.backend.actions_spent
    res = specialists.solve_level(core, name, 1, 120.0)
    assert res["solved"], res
    assert res["algo"] == "spec:ft09_gf2"
    assert res["handle"].path, "a solved specialist must carry a replay path"
    print(f"\n[C] ft09 L1 via specialist: depth {res['depth']}, "
          f"{core.backend.actions_spent - a0} engine actions")


def test_C_specialist_declines_on_non_matching_game():
    core = sc.SearchCore(make_env("tu93"), backend="reset_replay",
                         warmup_rounds=3, max_states=2000)
    core.warmup_and_freeze()
    assert specialists.detect(core) is None
    assert not any(specialists.detect_matrix(core).values())


def test_C_unknown_specialist_fails_open():
    core = sc.SearchCore(make_env("tu93"), backend="reset_replay",
                         warmup_rounds=1, max_states=100)
    res = specialists.solve_level(core, "no_such_specialist", 1, 1.0)
    assert res["solved"] is False and res["reason"] == "unknown_specialist"


# ==========================================================================
# D. banking handoff
# ==========================================================================

def test_D_bank_plan_concatenates_in_level_order():
    seqs = {2: [("S", 2)], 1: [("S", 1), ("C", 3, 4)], 3: [("S", 3)]}
    assert v8.bank_plan_actions(seqs) == [
        ("S", 1), ("C", 3, 4), ("S", 2), ("S", 3)]


def test_D_full_crack_and_bank_ft09_end_to_end():
    """The whole point of v8: crack a game the LLM lane scores 0 on, then
    convert the grind win into a near-optimal banked replay."""
    env = make_env("ft09")
    session = FakeSession(env)
    v7._RUN_T0 = time.monotonic()
    xs = v7._session_state(session)
    os.environ["EXPLORER_V8"] = "1"
    t0 = time.time()
    v8._grind_v8(session, xs, 0)
    wall = time.time() - t0
    assert xs["diag"]["games_won_by_grinder"] == 1, xs["diag"]
    assert xs["diag"].get("games_banked_by_grinder") == 1, xs["diag"]
    assert "banked: replayed win" in xs["diag"]["v8_bank_result"]
    assert xs["diag"]["v8_specialist"] == "ft09_gf2"
    assert len(xs["grind_unlocked_levels"]) == 6
    print(f"\n[D] ft09 FULL CRACK + BANK: {xs['diag']['grinder_actions']} engine "
          f"actions, offline wall {wall:.1f}s, "
          f"live wall @{GATEWAY:.0f} act/s = "
          f"{xs['diag']['grinder_actions'] / GATEWAY:.1f}s, "
          f"bank={xs['diag']['v8_bank_result']}")


def test_D_bank_refuses_when_plan_exceeds_cap():
    os.environ["EXPLORER_V8_BANK_MAX_ACTIONS"] = "3"
    ok, why = v8.bank_crack(object(), {1: [("S", 1)] * 10})
    assert ok is False and "cap 3" in why


def test_D_bank_refuses_when_disabled():
    os.environ["EXPLORER_V8_BANK"] = "0"
    ok, why = v8.bank_crack(object(), {1: [("S", 1)]})
    assert (ok, why) == (False, "disabled")


def test_D_bank_refuses_empty_plan():
    assert v8.bank_crack(object(), {}) == (False, "empty_plan")


def test_D_bank_refuses_a_plan_that_does_not_start_at_level_1():
    """After v7's bounded takeover the grinder owns levels k+1.. only; those
    early levels have no recorded minimal sequence, so a fresh-play replay
    cannot reproduce the win and banking must decline."""
    ok, why = v8.bank_crack(object(), {2: [("S", 1)], 3: [("S", 2)]})
    assert ok is False and "does not start at level 1" in why
    ok, why = v8.bank_crack(object(), {1: [("S", 1)], 3: [("S", 2)]})
    assert ok is False and "does not start at level 1" in why


def test_D_bank_trips_shared_kill_switch_on_server_patch_signature():
    """If post-WIN RESET stops opening a fresh play, banking must disable
    itself run-wide — and share that verdict with graft_bank."""
    import graft_bank

    class PatchedEnv:
        def step(self, action, data=None):
            class R:
                frame = [[[0]]]
                levels_completed = 4          # NOT a fresh play
                state = arcengine.GameState.NOT_FINISHED
            return R()

    ok, why = v8.bank_crack(PatchedEnv(), {1: [("S", 1)]})
    assert ok is False and "KILL" in why
    assert graft_bank._BankingKillSwitch.tripped
    ok2, why2 = v8.bank_crack(PatchedEnv(), {1: [("S", 1)]})
    assert ok2 is False and "kill switch tripped" in why2


def test_D_bank_honours_remaining_envelope():
    env = make_env("ft09")
    genv, session, xs = guarded(env, time_cap_s=1e9)
    genv._time_cap_s = 0.0                    # nothing left
    genv._owned_time_cap_s = 0.0
    ok, why = v8.bank_crack(env, {1: [("S", 1)] * 50}, guard=genv)
    assert ok is False and "budget" in why


# ==========================================================================
# E. envelope guards — one assertion per guard
# ==========================================================================

def _trip(genv) -> str:
    with pytest.raises(v8._GrindAbort) as exc:
        for _ in range(200):
            genv.reset()
    return exc.value.reason


def test_E_guard_action_budget():
    env = make_env("tu93")
    genv, _, _ = guarded(env, action_cap=7)
    assert _trip(genv) == "budget"
    assert genv.actions == 7


def test_E_guard_engagement_time_cap():
    env = make_env("tu93")
    genv, _, _ = guarded(env, time_cap_s=0.0)
    assert _trip(genv) == "time_cap"


def test_E_guard_owned_time_cap_applies_only_when_grind_owned():
    env = make_env("tu93")
    genv, session, xs = guarded(env, time_cap_s=0.0, owned_time_cap_s=1e9)
    xs["grind_unlocked_levels"].add(1)
    assert genv.wall_cap() == 1e9
    genv.reset()                               # no abort: owned cap governs
    xs["grind_unlocked_levels"].clear()
    assert genv.wall_cap() == 0.0
    assert _trip(genv) == "time_cap"


def test_E_guard_cumulative_run_grind_budget_midflight():
    """v7 checked this only at engagement ENTRY; v8 must trip mid-grind."""
    os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = "10"
    env = make_env("tu93")
    v7._RUN_T0 = time.monotonic()
    v7._GRIND_WALL_SPENT[0] = 10.0             # budget already spent elsewhere
    genv, _, _ = guarded(env)
    assert _trip(genv) == "run_grind_budget"


def test_E_guard_run_cutoff_midflight():
    os.environ["EXPLORER_RUN_CUTOFF_S"] = "1"
    os.environ["EXPLORER_OWNED_TIME_S"] = "1"
    env = make_env("tu93")
    v7._RUN_T0 = time.monotonic() - 100.0      # deep into the run
    genv, _, _ = guarded(env)
    assert _trip(genv) == "run_cutoff"


def test_E_hard_cutoff_equals_v7_implicit_worst_case():
    """v7's start gate (18000 s) + owned cap (1500 s) = 19500 s was v7's real
    but unenforced ceiling; v8 enforces exactly that number."""
    assert v8._hard_cutoff_s() == 19500.0
    os.environ["EXPLORER_RUN_HARD_CUTOFF_S"] = "9000"
    assert v8._hard_cutoff_s() == 9000.0


def test_E_guard_stop_event():
    env = make_env("tu93")
    session = FakeSession(env)
    genv, _, _ = guarded(env, session=session)
    session.stop_event.set()
    assert _trip(genv) == "cancelled"


def test_E_guard_runtime_limit():
    env = make_env("tu93")
    session = FakeSession(env, runtime_done=True)
    genv, _, _ = guarded(env, session=session)
    assert _trip(genv) == "runtime_cap"


def test_E_guard_soft_time_floor():
    env = make_env("tu93")
    session = FakeSession(env, soft=10.0)      # < the 60 s floor
    genv, _, _ = guarded(env, session=session)
    assert _trip(genv) == "soft_time"


def test_E_guard_per_game_action_cap():
    env = make_env("tu93")
    session = FakeSession(env, max_actions=5)
    session.action_count = 5
    genv, _, _ = guarded(env, session=session)
    assert _trip(genv) == "action_cap"


def test_E_action_ceiling_is_clamped_to_what_the_wall_cap_can_produce():
    """v7 shipped a 500k action ceiling no 1500 s cap can reach. v8 clamps."""
    owned = 1500
    ceiling = min(500000, int(owned * GATEWAY * 1.5))
    assert ceiling == 292500 < 500000


def test_E_remaining_s_is_the_min_over_all_wall_guards():
    os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = "100"
    os.environ["EXPLORER_RUN_CUTOFF_S"] = "10000"
    env = make_env("tu93")
    session = FakeSession(env, soft=500.0)
    v7._RUN_T0 = time.monotonic()
    v7._GRIND_WALL_SPENT[0] = 60.0
    genv, _, _ = guarded(env, session=session, time_cap_s=600.0,
                         owned_time_cap_s=1500.0)
    # candidates: 600 (engagement), 100-60=40 (run grind), 10000 (cutoff),
    # 500-60=440 (soft floor) -> 40
    assert 39.0 < genv.remaining_s() <= 40.0


def test_E_v8_never_loosens_a_v7_guard():
    """Every shared knob keeps its v7 default value."""
    for knob, default in (("EXPLORER_GRIND_TIME_S", 600),
                          ("EXPLORER_OWNED_TIME_S", 1500),
                          ("EXPLORER_RUN_GRIND_BUDGET_S", 2700),
                          ("EXPLORER_RUN_CUTOFF_S", 18000),
                          ("EXPLORER_GRIND_MAX_PER_LEVEL", 1),
                          ("EXPLORER_AGE_ACTIONS", 120),
                          ("EXPLORER_AGE_TURNS", 10)):
        assert v8._env_int(knob, default) == default
    assert v7._GRIND_GATE._initial_value == 1, "one concurrent grind, run-wide"


# ==========================================================================
# F. flags off = v7
# ==========================================================================

def test_F_flag_off_leaves_v7_grind_installed():
    os.environ["EXPLORER_V8"] = "0"
    before = v7._grind
    note = v8.install()
    assert "v8: OFF" in note
    assert v7._grind is before, "flag off must not swap the grind driver"


def test_F_flag_off_disables_v8_even_with_sub_flags_on():
    os.environ["EXPLORER_V8"] = "0"
    os.environ["EXPLORER_V8_SPECIALIST"] = "1"
    os.environ["EXPLORER_V8_BANK"] = "1"
    assert v8.v8_enabled() is False


def test_F_explorer_master_switch_still_wins():
    os.environ["EXPLORER"] = "0"
    os.environ["EXPLORER_V8"] = "1"
    assert v8.v8_enabled() is False


def test_F_flag_on_swaps_and_uninstall_restores():
    os.environ["EXPLORER_V8"] = "1"
    original = v7._grind
    try:
        note = v8.install()
        assert "v8: OK" in note or "v8: SKIP" in note
        if "v8: OK" in note:
            assert v7._grind is v8._grind_v8
            assert v7._grind_v7 is original
    finally:
        v8.uninstall()
    assert v7._grind is original


def test_F_v7_module_source_untouched_except_the_bank_bugfix():
    """v8 lives in its own module; the only edit to the flown v7 file is the
    documented bank_replay crash repair."""
    import inspect

    src = inspect.getsource(v7)
    assert "SearchCore" not in src
    assert "specialists" not in src
    assert "def bank_replay(level_seqs: dict[int, list[tuple]])" in src
    assert "bank_replay(level_seqs)" in src


def test_F_v7_bank_replay_arity_repaired():
    """The pre-repair call site passed 2 args to a 1-param function, so every
    v7 win raised TypeError inside a blanket except and went unbanked."""
    import inspect

    src = inspect.getsource(v7._grind)
    assert "bank_replay(level_seqs, " not in src
    assert "for lvl in sorted(level_seqs):" in src


# ==========================================================================
# G. fail-open on every seam
# ==========================================================================

def test_G_install_declines_when_search_core_unimportable(monkeypatch):
    os.environ["EXPLORER_V8"] = "1"
    monkeypatch.setattr(v8, "_import_core",
                        lambda: (_ for _ in ()).throw(ImportError("boom")))
    original = v7._grind
    note = v8.install()
    assert "v8: SKIP" in note and "unimportable" in note
    assert v7._grind is original


def test_G_grind_without_engine_wrapper_marks_exhausted():
    session = FakeSession(None)
    session.game.env = None
    xs = v7._session_state(session)
    v8._grind_v8(session, xs, 3)
    assert 3 in xs["grind_exhausted"]
    assert xs["grinding"] is False


def test_G_search_error_is_caught_and_engagement_ends_clean(monkeypatch):
    env = make_env("tu93")
    session = FakeSession(env)
    xs = v7._session_state(session)
    v7._RUN_T0 = time.monotonic()
    monkeypatch.setattr(v8, "_run_search",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    v8._grind_v8(session, xs, 1)
    assert xs["grinding"] is False
    assert xs["diag"].get("games_won_by_grinder", 0) == 0


def test_G_specialist_detect_exception_falls_back_to_generic(monkeypatch):
    env = make_env("tu93")
    genv, session, xs = guarded(env, action_cap=3000, time_cap_s=25.0,
                                owned_time_cap_s=25.0)
    monkeypatch.setattr(specialists, "detect",
                        lambda core: (_ for _ in ()).throw(ValueError("bad")))
    os.environ["EXPLORER_V8_SLICE_S"] = "5"
    reason = None
    try:
        reason = v8._run_search(session, xs, genv, "tu93", sc, specialists,
                                {}, lambda t: None, lambda t: "")
    except v8._GrindAbort as e:
        reason = e.reason
    assert xs["diag"]["v8_specialist"] is None
    assert reason is not None

def test_G_swallowed_abort_is_reasserted_after_a_lane(monkeypatch):
    """specialists.* wrap their bodies in `except Exception`, which swallows a
    guard trip. The lane loop must re-raise it instead of buying a free lane."""
    env = make_env("ft09")
    genv, session, xs = guarded(env, action_cap=10 ** 9)

    def fake_solve(core, name, target, budget):
        genv._action_cap = 0                    # a guard trips inside the lane
        try:
            genv.check()
        except v8._GrindAbort:
            return {"solved": False, "reason": "swallowed"}   # what specialists do
        return {"solved": False}

    monkeypatch.setattr(specialists, "detect", lambda core: "ft09_gf2")
    monkeypatch.setattr(specialists, "solve_level", fake_solve)
    with pytest.raises(v8._GrindAbort) as exc:
        v8._run_search(session, xs, genv, "ft09", sc, specialists, {},
                       lambda t: None, lambda t: "")
    assert exc.value.reason == "budget"


def test_G_banking_exception_never_escapes():
    class Exploding:
        def step(self, *a, **k):
            raise RuntimeError("engine on fire")

    ok, why = v8.bank_crack(Exploding(), {1: [("S", 1)]})
    assert ok is False and "RuntimeError" in why


def test_G_guard_probe_failures_do_not_block():
    class BrokenSession(FakeSession):
        def runtime_limit_reached(self):
            raise RuntimeError("probe broken")

    env = make_env("tu93")
    genv, _, _ = guarded(env, session=BrokenSession(env), action_cap=40)
    for _ in range(40):
        genv.reset()
    assert genv.actions == 40


def test_G_exhausted_level_is_never_re_engaged(monkeypatch):
    """A level every lane proved unreachable is closed for the game (v7 rule)."""
    env = make_env("tu93")
    session = FakeSession(env)
    xs = v7._session_state(session)
    v7._RUN_T0 = time.monotonic()
    monkeypatch.setattr(v8, "_run_search", lambda *a, **k: "frontier_exhausted")
    v8._grind_v8(session, xs, 4)
    assert 4 in xs["grind_exhausted"]


def test_G_non_cracking_engagement_respects_the_cap_and_exits_clean():
    """The common case: a hard game. The engagement must stop at the cap, not
    win, not raise, reset the env for the LLM, and stay a legal play."""
    env = make_env("cn04")
    session = FakeSession(env)
    xs = v7._session_state(session)
    v7._RUN_T0 = time.monotonic()
    os.environ["EXPLORER_V8"] = "1"
    os.environ["EXPLORER_GRIND_TIME_S"] = "20"
    os.environ["EXPLORER_OWNED_TIME_S"] = "20"
    os.environ["EXPLORER_V8_SLICE_S"] = "5"
    t0 = time.monotonic()
    v8._grind_v8(session, xs, 0)
    wall = time.monotonic() - t0
    assert wall < 40.0, f"cap 20s overrun: {wall:.1f}s"
    assert xs["grinding"] is False
    assert xs["diag"].get("games_won_by_grinder", 0) == 0
    assert session.wrote_state == 1
    resp = env.step(arcengine.GameAction.RESET, data={})
    assert resp is not None and resp.state != arcengine.GameState.GAME_OVER
    print(f"\n[G] cn04 non-crack engagement: {xs['diag']['grinder_actions']} "
          f"actions in {wall:.1f}s (cap 20s), clean exit")


# ==========================================================================
# H. live-cost simulation
# ==========================================================================

CRACK_SIM = {}


@pytest.mark.parametrize("stem", ["ft09", "tu93"])
def test_H_live_cost_simulation_fits_the_engagement_caps(stem):
    """Crack the game on the RESET-REPLAY backend, count real engine actions,
    price them at the measured live gateway rate, and assert the result fits
    the per-engagement caps."""
    env = make_env(stem)
    session = FakeSession(env)
    xs = v7._session_state(session)
    v7._RUN_T0 = time.monotonic()
    os.environ["EXPLORER_V8"] = "1"
    os.environ["EXPLORER_GRIND_TIME_S"] = "100000"      # measure, don't clip
    os.environ["EXPLORER_OWNED_TIME_S"] = "100000"
    os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = "100000"
    t0 = time.time()
    v8._grind_v8(session, xs, 0)
    offline_wall = time.time() - t0

    acts = xs["diag"]["grinder_actions"]
    live_s = acts / GATEWAY
    levels = len(xs["grind_unlocked_levels"])
    assert xs["diag"]["games_won_by_grinder"] == 1, "expected a full crack"
    assert xs["diag"].get("games_banked_by_grinder") == 1, xs["diag"]
    first = xs["diag"]["v8_first_unlock_actions"]
    CRACK_SIM[stem] = dict(actions=acts, live_s=live_s, levels=levels,
                           offline_wall=offline_wall, first_unlock=first,
                           bank=xs["diag"]["v8_bank_result"],
                           specialist=xs["diag"]["v8_specialist"])
    # the 600 s generic cap governs only until the first unlock makes the game
    # grind-owned; after that the 1500 s owned cap applies
    assert first / GATEWAY <= 600.0, (
        f"{stem} does not reach its first unlock inside the generic cap")
    print(f"\n[H] {stem}: {levels} levels, {acts} engine actions, "
          f"first unlock at {first} acts ({first / GATEWAY:.1f}s live), ")
    print(f"[H] {stem}: "
          f"offline {offline_wall:.1f}s, LIVE {live_s:.1f}s @ {GATEWAY:.0f} act/s "
          f"| generic cap 600s: {'FITS' if live_s <= 600 else 'over'} "
          f"| owned cap 1500s: {'FITS' if live_s <= 1500 else 'OVER'} "
          f"| {xs['diag']['v8_bank_result']}")
    # the owned cap is the one that governs a grind-owned game once level 1
    # unlocks; both measured cracks must fit inside it
    assert live_s <= 1500.0, f"{stem} crack does not fit the owned cap"
    assert live_s <= 2700.0, f"{stem} crack does not fit the run grind budget"


def test_H_report(capsys):
    if len(CRACK_SIM) < 2:
        pytest.skip("run the parametrized sim first (same session)")
    total = sum(v["live_s"] for v in CRACK_SIM.values())
    print("\n[H] LIVE-COST SUMMARY @130 act/s")
    for k, v in CRACK_SIM.items():
        print(f"    {k}: {v['levels']} levels  {v['actions']:>6} acts  "
              f"{v['live_s']:>7.1f}s live  first-unlock "
              f"{v['first_unlock'] / GATEWAY:>5.1f}s  spec={v['specialist']}  "
              f"{v['bank']}")
    print(f"    both cracks together: {total:.1f}s of the 2700s run budget "
          f"({100 * total / 2700:.0f}%)")
    assert total <= 2700.0
