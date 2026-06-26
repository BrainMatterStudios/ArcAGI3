"""Firewall + routing tests for ValueGuidedExplorer."""

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
import numpy as np

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.value_guided_explorer import ValueGuidedExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks = []
    for _ in range(steps):
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(
            P.to_grid(obs.frame),
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        toks.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
    return toks


def test_value_guided_off_is_byte_identical():
    base = _drive(TransferExplorer(**CFG), "push", 400)
    off = _drive(ValueGuidedExplorer(enable_value_guidance=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_value_guided_prefers_higher_scored_equal_depth_frontier():
    pol = ValueGuidedExplorer(enable_value_guidance=True, **CFG)
    pol.nodes = {}
    pol.root_key = b"root"
    root = pol._observe(b"root", [(("S", 1), 0), (("S", 2), 0)])
    left = pol._observe(b"left", [(("S", 3), 0)])
    right = pol._observe(b"right", [(("S", 4), 0)])
    root.edges = {("S", 1): (b"left", 0.0), ("S", 2): (b"right", 0.0)}
    left.edges = {}
    right.edges = {}

    class DummyModel:
        def score(self, features):
            return 0.5

    pol.value_model = DummyModel()
    pol._mu = np.zeros(7)
    pol._sigma = np.ones(7)
    pol._score_key = lambda key: {b"left": 0.1, b"right": 0.9}.get(key)

    path = pol._path_to_frontier(b"root", 0)

    assert path == [("S", 2)]
