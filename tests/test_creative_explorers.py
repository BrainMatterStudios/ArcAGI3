"""Firewall + behavior tests for the three creative-tournament candidates.

Each candidate subclasses SalienceExplorer. With its feature OFF it must produce a
byte-identical action trace to the banked v6 (the 0.33-floor firewall). With it ON it must
still drive local games without error and (for transfer) actually learn a reward signature.
"""

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.prior_explorer import PriorExplorer
from arcagi3.relational_explorer import RelationalExplorer
from arcagi3.transfer_relational_explorer import TransferRelationalExplorer

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
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


def _base(game, steps=400):
    return _drive(SalienceExplorer(**CFG), game, steps)[0]


def test_transfer_off_byte_identical():
    base = _base("push")
    off, _ = _drive(TransferExplorer(enable_transfer=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_prior_off_byte_identical():
    base = _base("push")
    off, _ = _drive(PriorExplorer(enable_prior=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_relational_off_byte_identical():
    base = _base("push")
    off, _ = _drive(RelationalExplorer(enable_relational=False, **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_combo_off_byte_identical():
    base = _base("push")
    off, _ = _drive(
        TransferRelationalExplorer(enable_transfer=False, enable_relational=False, **CFG),
        "push", 400)
    assert off == base and len(base) > 50


def test_combo_on_completes_local_games():
    _, lv = _drive(TransferRelationalExplorer(enable_transfer=True, enable_relational=True,
                                              **CFG), "btnc", 4000)
    assert lv >= 1


def test_transfer_on_runs_and_learns_on_click_game():
    # btnc is a click game with multiple levels -> a click level-up should be learned.
    pol = TransferExplorer(enable_transfer=True, **CFG)
    _, levels = _drive(pol, "btnc", 4000)
    assert levels >= 1
    # after at least one level-up the rewarding signature set must be populated
    assert pol.reward_click_sig or pol.reward_simple


def test_prior_on_completes_local_click_game():
    _, levels = _drive(PriorExplorer(enable_prior=True, **CFG), "btnc", 4000)
    assert levels >= 1


def test_relational_on_completes_local_nav_game():
    _, levels = _drive(RelationalExplorer(enable_relational=True, **CFG), "navg", 8000)
    assert levels >= 1
