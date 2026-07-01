"""The reactive GrabDragStrategy (portfolio-ready, source-free) must solve wa30 via the decide() interface
(learn avatar by probing -> perceive roles -> plan -> execute) and abstain safely on a non-grab-drag game."""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.grabdrag_strategy import GrabDragStrategy


def _drive(prefix, budget=400):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=f"gd-{prefix}")
    obs = env.reset()
    pol = GrabDragStrategy()
    for _ in range(budget):
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
        if obs.state is GameState.WIN or int(obs.levels_completed or 0) >= 1:
            break
    return int(obs.levels_completed or 0)


def test_reactive_grabdrag_solves_wa30():
    assert _drive("wa30") >= 1, "reactive grab-drag strategy should solve wa30 L0"


def test_reactive_grabdrag_abstains_on_non_archetype():
    # tu93 is not grab-drag: must not crash, must not falsely complete a level
    assert _drive("tu93", budget=200) == 0
