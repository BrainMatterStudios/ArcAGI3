"""Firewall + behavior tests for GoExploreExplorer."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.go_explore_explorer import GoExploreExplorer

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


def test_goexplore_off_byte_identical():
    base, _ = _drive(SalienceExplorer(**CFG), "push", 400)
    off, _ = _drive(GoExploreExplorer(enable_goexplore=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_goexplore_on_runs_and_archives():
    pol = GoExploreExplorer(enable_goexplore=True, **CFG)
    toks, lv = _drive(pol, "navg", 2000)
    assert len(toks) > 10 and len(pol.archive) > 1  # runs + builds a cell archive (KILLED: weaker than salience)
