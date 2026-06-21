from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer, _Node
from arcagi3.tour_explorer import TourExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    """Drive a reactive policy on a local OFFLINE game; return (tokens, final_levels)."""
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks, levels = [], 0
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


def test_v6_mode_is_byte_identical_to_banked():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base, _ = _drive(SalienceExplorer(**cfg), "push", 400)
    v6, _ = _drive(TourExplorer(frontier_mode="v6", **cfg), "push", 400)
    assert v6 == base and len(base) > 50


def _mk(pol, key, cands_with_tiers, edges):
    n = _Node(key, cands_with_tiers)
    n.edges = dict(edges)
    pol.nodes[key] = n
    return n


def _toy_graph(pol):
    # R(no untried) -> A(1 untried, depth1, disc1), R -> B(2 untried, depth1, disc2) -> C(1 untried, depth2, disc3)
    _mk(pol, b"R", [(("S", 1), 0), (("S", 2), 0)],
        {("S", 1): (b"A", 0.0), ("S", 2): (b"B", 0.0)})
    _mk(pol, b"A", [(("S", 1), 0), (("S", 3), 0)], {("S", 1): (b"R", 0.0)})
    _mk(pol, b"B", [(("S", 1), 0), (("S", 3), 0), (("S", 4), 0), (("S", 5), 0)],
        {("S", 1): (b"R", 0.0), ("S", 5): (b"C", 0.0)})
    _mk(pol, b"C", [(("S", 1), 0), (("S", 6), 0)], {("S", 1): (b"B", 0.0)})
    pol._disc_order = {b"R": 0, b"A": 1, b"B": 2, b"C": 3}


def test_yield_picks_max_untried_nearest_frontier():
    pol = TourExplorer(frontier_mode="yield")
    _toy_graph(pol)
    # depth-1 frontiers A(1 untried) and B(2 untried) -> pick B; path R->B = [("S",2)]
    assert pol._path_to_frontier(b"R", 9) == [("S", 2)]


def test_dfs_picks_most_recently_discovered_frontier():
    pol = TourExplorer(frontier_mode="dfs")
    _toy_graph(pol)
    # frontiers A(disc1) and C(disc3) -> pick C (most recent); path R->B->C = [("S",2),("S",5)]
    assert pol._path_to_frontier(b"R", 9) == [("S", 2), ("S", 5)]


def test_modes_preserve_coverage_on_local_game():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    _, lv_v6 = _drive(SalienceExplorer(**cfg), "navg", 8000)
    _, lv_y = _drive(TourExplorer(frontier_mode="yield", **cfg), "navg", 8000)
    _, lv_d = _drive(TourExplorer(frontier_mode="dfs", **cfg), "navg", 8000)
    assert lv_v6 > 0 and lv_y >= lv_v6 and lv_d >= lv_v6   # no coverage loss on the toy


def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        TourExplorer(frontier_mode="bogus")
