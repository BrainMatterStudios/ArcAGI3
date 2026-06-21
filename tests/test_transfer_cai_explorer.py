"""Firewall + behavior tests for TransferCAIExplorer (combo of two additive levers)."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.transfer_cai_explorer import TransferCAIExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset(); toks = []; levels = 0
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
        levels = max(levels, int(obs.levels_completed or 0))
    return toks, levels


def test_combo_off_byte_identical():
    base, _ = _drive(SalienceExplorer(**CFG), "push", 400)
    off, _ = _drive(TransferCAIExplorer(enable_transfer=False, enable_cai=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_combo_on_solves_local_click_game():
    _, lv = _drive(TransferCAIExplorer(enable_transfer=True, enable_cai=True, noop_k=4, **CFG), "btnc", 4000)
    assert lv >= 1   # makes real progress (btnc is a fast multi-level click game)
