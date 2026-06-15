"""Unit tests for C7 WorldModelPolicy (wm_policy.py).

Checks:
  1. make_policy() returns HybridPolicy by default (env unset or '0').
  2. make_policy() returns WorldModelPolicy when ARCAGI3_WORLDMODEL=1.
  3. Lockstep identity test (D2): WorldModelPolicy with cfg.enabled=False produces
     byte-identical action token streams to HybridPolicy on all 8 local games.
  4. Proxy properties (.gs, .phase, .mm) are accessible after a few steps.
  5. reset_all() resets WMState cleanly.
  6. _predict_key returns bytes (not None) on a simple move in a game with motion model.
  7. --agent wm choice passes through runner without crashing (integration smoke test).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")
ALL_GAMES = ("navg", "btnc", "push", "maze", "clickbig", "navgc", "collect", "switchdoor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(game_id: str):
    import logging

    from arc_agi import Arcade, OperationMode

    logger = logging.getLogger("t")
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=GAMES_DIR, logger=logger)
    return client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")


def _run_policy(pol, env, budget: int = 500) -> list:
    """Drive pol through env for up to budget steps; return list of tokens."""
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P

    tokens = []
    obs = env.reset()
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(
            grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        tokens.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        n += 1
    return tokens


# ---------------------------------------------------------------------------
# 1. make_policy selector
# ---------------------------------------------------------------------------

def test_make_policy_default_returns_hybrid(monkeypatch):
    monkeypatch.delenv("ARCAGI3_WORLDMODEL", raising=False)
    from arcagi3.policy import HybridPolicy
    from arcagi3.wm_policy import make_policy

    pol = make_policy(seed=0)
    assert isinstance(pol, HybridPolicy)


def test_make_policy_env1_returns_wm(monkeypatch):
    monkeypatch.setenv("ARCAGI3_WORLDMODEL", "1")
    from arcagi3.wm_policy import WorldModelPolicy, make_policy

    pol = make_policy(seed=0)
    assert isinstance(pol, WorldModelPolicy)


def test_make_policy_env0_returns_hybrid(monkeypatch):
    monkeypatch.setenv("ARCAGI3_WORLDMODEL", "0")
    from arcagi3.policy import HybridPolicy
    from arcagi3.wm_policy import make_policy

    pol = make_policy(seed=0)
    assert isinstance(pol, HybridPolicy)


# ---------------------------------------------------------------------------
# 2. WorldModelPolicy construction and proxy properties
# ---------------------------------------------------------------------------

def test_wm_construction_and_proxies():
    from arcagi3.policy import HybridPolicy
    from arcagi3.wm_policy import WMConfig, WorldModelPolicy

    cfg = WMConfig(enabled=False)
    pol = WorldModelPolicy(cfg=cfg, seed=42)
    assert isinstance(pol.base, HybridPolicy)
    # Phase/mm/gs accessible before any steps
    assert pol.phase == "probe"
    assert pol.mm is None
    assert pol.gs is None  # no game started yet


def test_wm_reset_all():
    from arcagi3.wm_policy import WMConfig, WMState, WorldModelPolicy

    pol = WorldModelPolicy(cfg=WMConfig(enabled=False), seed=0)
    pol.s.level = 5
    pol.s.plan_aborts = 10
    pol.reset_all()
    assert pol.s.level == -1
    assert pol.s.plan_aborts == 0
    assert isinstance(pol.s, WMState)


# ---------------------------------------------------------------------------
# 3. D2 Lockstep identity test — THE KEYSTONE SAFETY NET
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ALL_GAMES)
def test_wm_passthrough_equals_hybrid(game_id):
    """With cfg.enabled=False, WorldModelPolicy produces byte-identical tokens to HybridPolicy.

    This is the D2 lockstep identity test from the blueprint — mechanically proves the
    wrapper adds nothing when planning is off. Must pass BEFORE any planning logic fires.
    """
    from arcagi3.policy import HybridPolicy
    from arcagi3.wm_policy import WMConfig, WorldModelPolicy

    # budget: enough to get through the game or demonstrate equivalence
    budget = 600

    env_hp = _make_env(game_id)
    env_wm = _make_env(game_id)

    hp = HybridPolicy(seed=0)
    wm = WorldModelPolicy(cfg=WMConfig(enabled=False), seed=0)

    tokens_hp = _run_policy(hp, env_hp, budget)
    tokens_wm = _run_policy(wm, env_wm, budget)

    assert tokens_hp == tokens_wm, (
        f"Token mismatch for {game_id}:\n"
        f"  HybridPolicy:       {tokens_hp[:10]}\n"
        f"  WorldModelPolicy:   {tokens_wm[:10]}"
    )


# ---------------------------------------------------------------------------
# 4. predict_key produces bytes on a simple move after motion model is built
# ---------------------------------------------------------------------------

def test_predict_key_returns_bytes_after_probe():
    """_predict_key should return bytes (not None) once the motion model is warm."""
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.wm_policy import WMConfig, WorldModelPolicy

    pol = WorldModelPolicy(cfg=WMConfig(enabled=False), seed=0)
    env = _make_env("navg")

    obs = env.reset()
    # run until probe phase is done (phase != 'probe')
    n = 0
    while n < 200:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(
            grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        if pol.phase != "probe" and pol.mm is not None and pol.mm.ok:
            # Try predicting a key for the first available simple move
            aids = list(pol.mm.deltas.keys())
            if aids:
                result = pol._predict_key(grid, ("S", aids[0]))
                assert result is not None, "_predict_key returned None when motion model is ready"
                assert isinstance(result, bytes), f"_predict_key returned {type(result)}, expected bytes"
                break
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        n += 1


# ---------------------------------------------------------------------------
# 5. Integration: --agent wm smoke test (default env -> HybridPolicy path)
# ---------------------------------------------------------------------------

def test_runner_agent_wm_smoke(monkeypatch):
    """--agent wm with ARCAGI3_WORLDMODEL unset should still use HybridPolicy and win navg."""
    monkeypatch.delenv("ARCAGI3_WORLDMODEL", raising=False)
    from arcagi3.runner import run_game

    r = run_game("navg", GAMES_DIR, budget=600, agent_name="wm")
    assert r.won, f"runner --agent wm (default-off) failed navg: {r}"


def test_runner_agent_wm_worldmodel_on(monkeypatch):
    """--agent wm with ARCAGI3_WORLDMODEL=1 uses WorldModelPolicy and still wins navg."""
    monkeypatch.setenv("ARCAGI3_WORLDMODEL", "1")
    from arcagi3.runner import run_game

    r = run_game("navg", GAMES_DIR, budget=600, agent_name="wm")
    assert r.won, f"runner --agent wm (WorldModelPolicy enabled) failed navg: {r}"


# ---------------------------------------------------------------------------
# 6. reactive path is unaffected
# ---------------------------------------------------------------------------

def test_runner_reactive_unchanged(monkeypatch):
    """--agent reactive always uses HybridPolicy regardless of ARCAGI3_WORLDMODEL."""
    monkeypatch.setenv("ARCAGI3_WORLDMODEL", "1")
    from arcagi3.runner import run_game

    r = run_game("navg", GAMES_DIR, budget=600, agent_name="reactive")
    assert r.won, f"runner --agent reactive failed navg with ARCAGI3_WORLDMODEL=1: {r}"


# ---------------------------------------------------------------------------
# 7. WorldModelPolicy enabled=True (no components) still equals HybridPolicy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ("navg", "btnc", "maze", "collect"))
def test_wm_enabled_no_components_equals_hybrid(game_id):
    """WorldModelPolicy with planning ON but no components is still identical to HybridPolicy.

    With self.goal=None (no goal inference handle), _goal_ready() returns True but
    _get_goal_hyp() reads base.gi, so the base's real GoalInference runs. However since
    base.gi needs multiple level-ups to get confident goals, and our games need hundreds of
    steps, planning will never fire in the short window — but more importantly with cfg.enabled=True
    and no external component handles wired, the decide() path MUST still be equivalent
    for the default games at short budgets.

    NOTE: this test uses a shorter budget (200 steps) to be fast; the keystone D2 test at
    budget=600 above is the authoritative one.
    """
    from arcagi3.policy import HybridPolicy
    from arcagi3.wm_policy import WMConfig, WorldModelPolicy

    budget = 200

    env_hp = _make_env(game_id)
    env_wm = _make_env(game_id)

    hp = HybridPolicy(seed=0)
    # enabled=True, but no external component handles: goal/planner both None
    wm = WorldModelPolicy(cfg=WMConfig(enabled=True, use_planner=True, use_goal_inference=True,
                                        min_goal_conf=0.99),  # near-impossible gate
                          seed=0)

    tokens_hp = _run_policy(hp, env_hp, budget)
    tokens_wm = _run_policy(wm, env_wm, budget)

    assert tokens_hp == tokens_wm, (
        f"Token mismatch for {game_id} (enabled, no components, high gate):\n"
        f"  HybridPolicy: {tokens_hp[:10]}\n"
        f"  WorldModelPolicy: {tokens_wm[:10]}"
    )
