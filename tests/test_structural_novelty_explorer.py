"""Firewall test for StructuralNoveltyExplorer: enable_struct=False -> byte-identical to v6."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.structural_novelty_explorer import StructuralNoveltyExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset(); toks = []
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
    return toks


def test_struct_off_byte_identical():
    base = _drive(SalienceExplorer(**CFG), "push", 400)
    off = _drive(StructuralNoveltyExplorer(enable_struct=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_struct_on_completes_local_game():
    toks = _drive(StructuralNoveltyExplorer(enable_struct=True, **CFG), "navg", 6000)
    assert len(toks) > 10
