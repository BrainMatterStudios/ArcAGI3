"""Phase-0 firewall: WinningExplorer must be byte-identical to TransferExplorer when model-search
is off OR has no confident plan (the stub today always defers). Enforces the mission's hard rule:
'If model search has no confident plan, behavior must be byte-identical to TransferExplorer.'
"""
import logging

import pytest

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.winning_explorer import WinningExplorer
from arcengine import GameAction, GameState

logging.basicConfig(level=logging.ERROR)


def _make_collect():
    from arc_agi import Arcade, OperationMode
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir="src/arcagi3/games", logger=logging.getLogger("fw"))
    return client.make(game_id="collect", scorecard_id="sc-fw")


def _step(env, token):
    if token[0] == "reset":
        return env.reset()
    if token[0] == "S":
        return env.step(GameAction.from_id(token[1]))
    return env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]})


def _record_transfer(budget=150):
    eng = TransferExplorer(seed=0)
    eng.reset_all()
    env = _make_collect()
    obs = env.reset()
    trace = []
    for _ in range(budget):
        args = dict(
            grid=P.to_grid(obs.frame),
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        token = eng.decide(**args)
        trace.append((args, token))
        obs = _step(env, token)
    return trace


def test_winning_byte_identical_when_disabled():
    """enable_model_search=False -> byte-identical to TransferExplorer at every step (the firewall)."""
    trace = _record_transfer()
    w = WinningExplorer(seed=0, enable_model_search=False)
    w.reset_all()
    for i, (args, expected) in enumerate(trace):
        got = w.decide(**args)
        assert got == expected, f"divergence at step {i}: {got} != {expected}"
    assert w._model_fires == 0
    assert w._fallback_actions == len(trace)


def test_clickonly_defers_to_transfer():
    """On click-only games (no directional actions, e.g. lp85) the model loop never acts even with
    model search enabled — protecting transfer's click-signature domain."""
    trace = _record_transfer()
    w = WinningExplorer(seed=0, enable_model_search=True)
    w.reset_all()
    for args, _expected in trace:
        clickonly = dict(args, available=[6])              # force click-only availability
        assert w._model_decide(clickonly["grid"], [6]) is None
    assert w._model_fires == 0
