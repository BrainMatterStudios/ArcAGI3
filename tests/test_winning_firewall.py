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


@pytest.mark.parametrize("enable", [True, False])
def test_winning_byte_identical_to_transfer(enable):
    """Replaying TransferExplorer's exact input sequence into a fresh WinningExplorer yields the
    identical action token at every step — model-search stub defers, so no divergence is possible."""
    trace = _record_transfer()
    w = WinningExplorer(seed=0, enable_model_search=enable)
    w.reset_all()
    for i, (args, expected) in enumerate(trace):
        got = w.decide(**args)
        assert got == expected, f"divergence at step {i}: WinningExplorer {got} != TransferExplorer {expected}"
    # With model search on, the firewall must have deferred every step (stub returns None).
    if enable:
        assert w._model_fires == 0
        assert w._fallback_actions == len(trace)
