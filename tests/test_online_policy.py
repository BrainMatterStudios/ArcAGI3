"""Phase A no-regression + fail-safe tests for OnlineLearningExplorer (GraphRanker).

Firewall: with the model disabled, the composed policy MUST be byte-identical to a bare
SalienceExplorer driven the same way (the 0.33 floor is untouched). Plus fail-safe on
malformed input, and a smoke win with the model enabled.
"""
import logging
import os

import numpy as np

os.environ.setdefault("ARC_API_KEY", "local-dev")

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402
from arcagi3.online_explorer import OnlineLearningExplorer  # noqa: E402

GAMES = os.path.join(os.path.dirname(__file__), "..", "src", "arcagi3", "games")


def _drive(policy, game, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES,
                    logger=logging.getLogger("t"))
    env = client.make(game_id=game, scorecard_id="sc")
    obs = env.reset()
    tokens, n = [], 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        g = P.to_grid(obs.frame)
        tok = policy.decide(g, obs.state == GameState.GAME_OVER,
                            obs.state == GameState.NOT_PLAYED,
                            int(obs.levels_completed or 0), list(obs.available_actions or []))
        tokens.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        n += 1
    return tokens, obs.state == GameState.WIN, int(obs.levels_completed or 0)


def test_disabled_model_is_byte_identical_to_base():
    """The banked-0.33 firewall: model off -> exactly the SalienceExplorer token stream."""
    base = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    base_tokens, base_won, _ = _drive(base, "navg", 4000)

    ole = OnlineLearningExplorer(seed=0, trust_threshold=3, border_mask=2)
    ole.model.usable = False  # force the fallback path
    ole_tokens, ole_won, _ = _drive(ole, "navg", 4000)

    assert ole.n_overrides == 0
    assert base_won == ole_won
    assert base_tokens == ole_tokens


def test_failsafe_on_malformed_grids():
    """decide() must never raise and always return a valid token tuple."""
    ole = OnlineLearningExplorer(seed=0)
    for grid in (np.zeros((64, 64), dtype=np.int8),
                 np.zeros((1, 1), dtype=np.int8),
                 np.full((64, 64), 99, dtype=np.int8),
                 np.zeros((64, 64), dtype=np.float32)):
        tok = ole.decide(grid, False, False, 0, [1, 2, 3, 4, 5, 6])
        assert isinstance(tok, tuple) and tok[0] in ("S", "C", "reset")


def test_enabled_model_still_wins_btnc():
    """With the model enabled, the policy must still solve a quick local game (no breakage)."""
    ole = OnlineLearningExplorer(seed=0, trust_threshold=3, border_mask=2, allow_cpu=True)
    assert ole.model.usable, "torch should be usable on CPU in dev (allow_cpu=True)"
    _, won, levels = _drive(ole, "btnc", 4000)
    assert won, f"online failed btnc (levels={levels})"
