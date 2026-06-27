"""Firewall + behavior tests for ChainMacroExplorer (within-game solution-sequence replay)."""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.chain_macro_explorer import ChainMacroExplorer

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


def test_macro_off_byte_identical_to_transfer():
    """Firewall: enable_macro=False -> byte-identical to the banked TransferExplorer."""
    base, _ = _drive(TransferExplorer(**CFG), "push", 400)
    off, _ = _drive(ChainMacroExplorer(enable_macro=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def _obj_grid(color=7, at=(10, 20)):
    g = np.zeros((64, 64), dtype=np.int8)
    r, c = at
    g[r:r + 2, c:c + 2] = color
    return g


def test_macro_pick_clicks_matching_object():
    pol = ChainMacroExplorer(enable_macro=True, seed=0)
    pol.bg = 0
    pol._cur_grid = _obj_grid(7, (10, 20))
    pol.macro = [(7,)]; pol.macro_pos = 0; pol._macro_clicked = set()
    tok = pol._macro_pick()
    assert tok == ("C", 20, 10)        # click the color-7 object's centroid (col=20, row=10)
    assert pol.macro_pos == 1
    assert pol._macro_pick() is None   # the only matching object is already clicked


def test_chain_records_effective_clicks_only():
    pol = ChainMacroExplorer(enable_macro=True, seed=0)
    pol.bg = 0
    pol._prev_grid = _obj_grid(7, (10, 20))
    pol._record(b"k", ("C", 20, 10), b"k2", 0.0, [], False)   # effective (key changed)
    assert pol._level_ops == [(7,)]
    pol._record(b"k2", ("C", 20, 10), b"k2", 0.0, [], False)  # no-op (same key) -> not recorded
    assert pol._level_ops == [(7,)]


def test_macro_on_runs_and_progresses_local_click_game():
    _, lv = _drive(ChainMacroExplorer(enable_macro=True, **CFG), "btnc", 4000)
    assert lv >= 1
