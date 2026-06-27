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


import numpy as np


def _g(avatar_rc, tile_rc=None):
    g = np.zeros((64, 64), dtype=np.int8)
    if tile_rc is not None:
        g[tile_rc] = 7                 # a static glyph, color 7
    ar, ac = avatar_rc
    g[ar, ac] = 9                      # avatar, color 9 (drawn last -> occludes tile if same cell)
    return g


def test_occlusion_aware_visit_counter():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    sig = (7, 20, 20, 20, 20)          # (color, r0, c0, r1, c1) of the tile at (20,20)
    # frame 0: avatar at (20,21), tile visible at (20,20) -> remember tile, no visit yet
    pol._update_history(_g((20, 21), tile_rc=(20, 20)))
    # frame 1: avatar moves onto (20,20) -> tile occluded; visit++ (edge-triggered arrival)
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 1
    # frame 2: avatar stays on (20,20) -> NO additional count (not a new arrival)
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 1
    # frame 3: avatar leaves to (20,21); tile reappears
    pol._update_history(_g((20, 21), tile_rc=(20, 20)))
    # frame 4: avatar steps back onto the tile -> visit++ again
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 2


def _gm(head, body, tile, anim):
    g = np.zeros((64, 64), dtype=np.int8)
    if tile is not None:
        g[tile] = 7
    if anim is not None:
        g[anim] = 5
    g[head] = 12
    g[body] = 9
    return g


def test_multicolor_avatar_body_occlusion_counts():
    # head(12)+body(9) move as one; the BODY occludes the tile on arrival -> visit must still count,
    # and the independently-moving animation color (5) must NOT be learned as avatar.
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0
    pol._update_history(_gm((10, 10), (11, 10), (12, 10), (0, 0)))   # tile visible at (12,10)
    pol._update_history(_gm((11, 10), (12, 10), None, (0, 1)))       # body lands on tile (occluded); anim drifts
    assert pol._avatar_colors == {12, 9}                              # both avatar colors, anim(5) excluded
    assert pol._counts.get((7, 12, 10, 12, 10)) == 1                  # body-occlusion visit counted
    assert all(sig[0] not in (12, 9) for sig in pol._counts)          # no avatar-color self-counts
