from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks = []
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


def test_augment_off_is_byte_identical():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base = _drive(SalienceExplorer(**cfg), "push", 400)
    off = _drive(HistoryAugmentedExplorer(augment=False, **cfg), "push", 400)
    assert off == base and len(base) > 50


def test_key_unaugmented_when_no_counts():
    import numpy as np
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    base = SalienceExplorer(seed=0)
    base.bg = 0
    base.vt.update(grid)
    assert pol._key(grid) == base._key(grid)   # no counts -> identical to banked key


def test_key_changes_with_counter():
    import numpy as np
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    k0 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 1}    # a visited-object counter
    k1 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 2}
    k2 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 5}    # 5 % 4 == 1 == same residue as count 1
    k5 = pol._key(grid)
    assert k0 != k1 and k1 != k2 and k1 == k5   # mod 4: counts 1 and 5 collapse
