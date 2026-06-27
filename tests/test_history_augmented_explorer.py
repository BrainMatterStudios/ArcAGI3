import numpy as np
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
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    base = SalienceExplorer(seed=0)
    base.bg = 0
    base.vt.update(grid)
    assert pol._key(grid) == base._key(grid)


def test_key_changes_with_counter():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    k0 = pol._key(grid)
    pol._counts = {((20, 20, 7),): 1}
    k1 = pol._key(grid)
    pol._counts = {((20, 20, 7),): 2}
    k2 = pol._key(grid)
    pol._counts = {((20, 20, 7),): 5}   # 5 % 4 == 1 == residue of count 1
    k5 = pol._key(grid)
    assert k0 != k1 and k1 != k2 and k1 == k5


def _gd(tile=True, occ=None):
    """A 2-cell static glyph at (20,20)=7,(20,21)=8; `occ` = cells an occluder (color 9) covers."""
    g = np.zeros((64, 64), dtype=np.int8)
    if tile:
        g[20, 20] = 7
        g[20, 21] = 8
    for cell in (occ or []):
        g[cell] = 9
    return g


GLYPH_SIG = ((20, 20, 7), (20, 21, 8))


def test_disappearance_visit_counter():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0
    # A mobile occluder (2-cell color-9 block) slides up its column ONTO the static glyph and off again.
    pol._update_history(_gd(True, occ=[(25, 20), (25, 21)]))        # f0: glyph seen=1; occluder below
    pol._update_history(_gd(True, occ=[(23, 20), (23, 21)]))        # f1: glyph seen=2; occluder moved (mobile)
    pol._update_history(_gd(False, occ=[(20, 20), (20, 21)]))       # f2: mobile occluder covers glyph -> 1
    assert pol._counts.get(GLYPH_SIG) == 1
    pol._update_history(_gd(True, occ=[(18, 20), (18, 21)]))        # f3: glyph reappears; occluder above
    pol._update_history(_gd(True, occ=[(18, 20), (18, 21)]))        # f4: glyph seen=2 again (occluder paused, still mobile)
    pol._update_history(_gd(False, occ=[(20, 20), (20, 21)]))       # f5: mobile occluder covers glyph again -> 2
    assert pol._counts.get(GLYPH_SIG) == 2
    assert list(pol._counts.keys()) == [GLYPH_SIG]                  # the mobile occluder is never counted


def test_moving_object_not_counted():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0

    def gmov(pos):
        g = np.zeros((64, 64), dtype=np.int8)
        g[pos] = 9
        return g

    pol._update_history(gmov((5, 5)))
    pol._update_history(gmov((5, 6)))   # moved -> previous never confirmed static
    pol._update_history(gmov((5, 7)))   # moved again
    assert pol._counts == {}            # a thing that never stays put is never counted


def test_vanish_into_background_not_counted():
    """A confirmed-static glyph that simply disappears leaving BACKGROUND (the avatar leaving its spawn
    cell; a HUD digit blanking) is NOT a visit. A visit requires a MOVING object to cover the cells --
    that is what 'the avatar stepped onto it' means. This kills the ls20 spawn-block + HUD pollution.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    pol.bg = 0
    pol._update_history(_gd(True))    # f0 glyph seen=1
    pol._update_history(_gd(True))    # f1 glyph seen=2 (confirmed static)
    pol._update_history(_gd(False))   # f2 glyph gone, nothing covers the cells (-> background)
    assert pol._counts == {}


def test_static_glyph_replaced_by_static_not_counted():
    """A confirmed-static glyph replaced by a DIFFERENT static thing (a HUD digit changing value, no
    motion) is NOT a visit -- the replacement never moved onto it, so no mobile occluder is present.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    pol.bg = 0

    def g(color):
        a = np.zeros((64, 64), dtype=np.int8)
        a[20, 20] = color
        a[20, 21] = color
        return a

    pol._update_history(g(7))   # f0 seen=1
    pol._update_history(g(7))   # f1 seen=2 confirmed
    pol._update_history(g(5))   # f2 same cells, different color, but it simply APPEARED (never moved)
    assert pol._counts == {}


def test_paused_then_moving_object_not_counted():
    """The real ls20 failure mode: the avatar MOVES, then PAUSES >=2 frames (blocked by a wall),
    then moves away. Stability alone confirms it 'static' and counts its disappearance as a visit —
    but it moved before pausing, so mobility memory must keep it excluded the whole time.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0

    def gmov(pos):
        g = np.zeros((64, 64), dtype=np.int8)
        g[pos] = 9
        return g

    pol._update_history(gmov((5, 5)))   # appears
    pol._update_history(gmov((5, 10)))  # MOVED 5 cells -> ever_moved
    pol._update_history(gmov((5, 10)))  # paused (static, frame 2)
    pol._update_history(gmov((5, 10)))  # paused (static, frame 3 -> stability would 'confirm')
    pol._update_history(gmov((20, 20)))  # moves away -> its (5,10) cluster disappears
    assert pol._counts == {}            # a mover, even when paused, is never a 'visit'


def test_decide_runs_when_augmenting():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    pol = HistoryAugmentedExplorer(augment=True, **cfg)
    _drive(pol, "navg", 300)
    assert isinstance(pol._counts, dict)   # ran end-to-end without error
