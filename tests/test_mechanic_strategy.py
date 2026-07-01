"""The reactive MechanicSolverStrategy (portfolio-ready) must solve sb26 via the decide() interface, and
abstain safely on a non-archetype game (no crash, no false level-up) — the additive-play safety property.
"""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.mechanic_strategy import MechanicSolverStrategy


def _drive(prefix, budget=200):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=f"strat-{prefix}")
    obs = env.reset()
    pol = MechanicSolverStrategy()
    n = 0
    while n < budget:
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, gstate_terminal=(obs.state is GameState.GAME_OVER),
                         gstate_notplayed=(obs.state is GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        if obs.state is GameState.WIN or int(obs.levels_completed or 0) >= 1:
            break
    return int(obs.levels_completed or 0)


def test_reactive_strategy_solves_sb26():
    assert _drive("sb26") >= 1, "reactive pattern-match strategy should solve sb26 L0"


def test_reactive_strategy_abstains_on_non_archetype():
    # tu93 is not a pattern-match game: the strategy must not crash and must not falsely complete a level
    lv = _drive("tu93", budget=120)
    assert lv == 0, f"strategy should abstain on tu93 (no false level-up), got {lv}"
