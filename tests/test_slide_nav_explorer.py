"""Tests for SlideNavExplorer (Increment 2): firewall + the push-safety gate.

- enable_slide=False -> byte-identical action trace to SalienceExplorer (the 0.33 firewall).
- on the sokoban toy (push), the online gate must DETECT push and DELEGATE (never engage
  spatial nav) -> no sokoban regression (the failure that killed the un-gated spatial.py).
- on a pure-nav toy, it should ENGAGE nav mode and still clear levels.
"""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.slide_nav_explorer import SlideNavExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset(); toks, levels = [], 0
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


def test_slide_off_byte_identical():
    base, _ = _drive(SalienceExplorer(**CFG), "push", 400)
    off, _ = _drive(SlideNavExplorer(enable_slide=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_slide_gates_off_push_sokoban():
    # push (sokoban) needs thousands of actions to solve; the point here is the SAFETY GATE:
    # a non-avatar object moves with the avatar (push) -> must DELEGATE, never engage spatial nav.
    pol = SlideNavExplorer(enable_slide=True, **CFG)
    _drive(pol, "push", 600)
    assert pol.mode == "delegate"


def test_slide_engages_on_pure_nav():
    pol = SlideNavExplorer(enable_slide=True, **CFG)
    _, lv = _drive(pol, "navg", 6000)
    assert pol.mode == "nav"        # clean avatar, walls, no push -> engage spatial nav
    assert lv >= 1
