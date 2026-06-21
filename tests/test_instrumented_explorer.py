from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.instrumented_explorer import InstrumentedExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    """Drive a reactive policy on a local OFFLINE game; return the list of returned tokens."""
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


def test_instrumented_is_byte_identical_to_banked():
    # FIREWALL: same seed + config -> identical action sequence as the banked explorer.
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base = _drive(SalienceExplorer(**cfg), "push", 400)
    inst = _drive(InstrumentedExplorer(**cfg), "push", 400)
    assert inst == base and len(base) > 50


def test_tags_populate_sensibly():
    pol = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2)
    _drive(pol, "push", 400)
    assert pol.tags["fresh_local_test"] > 0
    assert pol.tags["plan_replay_walk"] + pol.tags["new_plan_to_frontier"] > 0  # push has walks
    assert isinstance(pol.plan_invalidations, int) and pol.plan_invalidations >= 0
    assert all(isinstance(n, int) and n > 0 for n in pol.new_plan_len)


def test_tags_are_deterministic():
    a = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2); _drive(a, "push", 300)
    b = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2); _drive(b, "push", 300)
    assert dict(a.tags) == dict(b.tags) and a.plan_invalidations == b.plan_invalidations
