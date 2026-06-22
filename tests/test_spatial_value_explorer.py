"""Firewall + behavior tests for SpatialValueExplorer (per-game value-CNN)."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.spatial_value_explorer import SpatialValueExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset(); toks = []; lv = 0
    for _ in range(steps):
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        toks.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        lv = max(lv, int(obs.levels_completed or 0))
    return toks, lv


def test_valuecnn_off_byte_identical():
    base, _ = _drive(SalienceExplorer(**CFG), "push", 400)
    off, _ = _drive(SpatialValueExplorer(enable_value_cnn=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_valuecnn_on_solves_local_multilevel():
    pol = SpatialValueExplorer(enable_value_cnn=True, train_steps=20, **CFG)
    _, lv = _drive(pol, "btnc", 4000)
    assert lv >= 1   # trains a value net post-first-reward, still makes progress
