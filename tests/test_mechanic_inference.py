"""Tests for the mechanic-inference probe (Increment 1 of the puzzle-solving program).

Validate the fixed-ontology detectors on local toys with KNOWN mechanics: avatar detection,
action-dependent directional deltas, and the signature interaction (push on the sokoban toy).
"""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.mechanic_inference import infer_mechanics

GAMES_DIR = "src/arcagi3/games"


def _capture(game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); traj = []; prev = 0
    bg = P.detect_background(P.to_grid(obs.frame))
    for _ in range(steps):
        g = P.to_grid(obs.frame); st = obs.state
        tok = pol.decide(g, gstate_terminal=(st == GameState.GAME_OVER),
                         gstate_notplayed=(st == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = env.reset(); continue
        elif tok[0] == "S":
            nobs = env.step(GameAction.from_id(tok[1]))
        else:
            nobs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        ng = P.to_grid(nobs.frame); lv = int(nobs.levels_completed or 0)
        traj.append((g, tok, ng, float(lv - prev))); prev = lv; obs = nobs
    return traj, bg


def test_avatar_detected_on_nav_game():
    traj, bg = _capture("navg", 1000)
    rep = infer_mechanics(traj, bg)
    assert rep.avatar_color is not None
    # arrows produce distinct, action-dependent directional deltas
    assert len(rep.avatar_action_deltas) >= 3
    assert rep.movement_type in ("step", "slide")


def test_push_interaction_detected_on_sokoban():
    traj, bg = _capture("push", 1500)
    rep = infer_mechanics(traj, bg)
    assert rep.avatar_color is not None
    assert "push" in rep.interactions   # sokoban: avatar pushes a block (non-avatar mover)


def test_block_interaction_on_maze():
    traj, bg = _capture("maze", 1500)
    rep = infer_mechanics(traj, bg)
    assert rep.avatar_color is not None
    assert "block" in rep.interactions   # walls stop the avatar
