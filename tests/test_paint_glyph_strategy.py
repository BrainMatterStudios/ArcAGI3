"""Reactive PaintStrategy (re86) and GlyphStrategy (sc25) must solve their games via decide() and abstain
safely on a non-matching game (no crash, no false level-up)."""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.paint_strategy import PaintStrategy
from arcagi3.glyph_strategy import GlyphStrategy


def _drive(pol, prefix, budget=300):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=f"pg-{prefix}")
    obs = env.reset()
    for _ in range(budget):
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, gstate_terminal=(obs.state is GameState.GAME_OVER),
                         gstate_notplayed=(obs.state is GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        if obs.state is GameState.WIN or int(obs.levels_completed or 0) >= 1:
            break
    return int(obs.levels_completed or 0)


def test_paint_solves_re86():
    assert _drive(PaintStrategy(), "re86") >= 1


def test_glyph_solves_sc25():
    assert _drive(GlyphStrategy(), "sc25") >= 1


def test_paint_abstains_on_non_match():
    assert _drive(PaintStrategy(), "tu93", budget=150) == 0


def test_glyph_abstains_on_non_match():
    assert _drive(GlyphStrategy(), "tu93", budget=150) == 0
