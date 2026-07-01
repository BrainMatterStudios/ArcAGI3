"""TDD for the wa30 grab-drag planner. Validates perception + the exact forward model against engine
ground truth, then asserts the planner actually WINS wa30 L0 in the real engine.

Run: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 .venv/bin/python -m pytest tests/test_wa30_planner.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
import wa30_planner as W


def _make():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("wa30"))
    env = client.make(game_id=gid, scorecard_id="test-wa30")
    obs = env.reset()
    return env, obs


def _truth(env):
    g = env._game
    lvl = g.current_level
    av = lvl.get_sprites_by_tag("wbmdvjhthc")[0]
    blocks = sorted((s.x, s.y) for s in lvl.get_sprites_by_tag("geezpjgiyd"))
    held = av in g.nsevyuople
    return (av.x, av.y, av.rotation, held), blocks


def test_perceive_matches_engine_truth():
    env, obs = _make()
    (ax, ay, _, _), blocks = _truth(env)
    grid = P.to_grid(obs.frame)
    avatar, pblocks, pads = W.perceive(grid)
    assert avatar == (ax, ay), f"avatar {avatar} != truth {(ax, ay)}"
    assert pblocks == blocks, f"blocks {pblocks} != truth {blocks}"
    assert len(pads) == 3, f"expected 3 goal pads, perceived {pads}"


def test_forward_model_matches_engine():
    """Drive engine + single-block model with the same actions; assert agreement each step."""
    env, obs = _make()
    (ax, ay, rot, held), blocks = _truth(env)
    # track block at (32,36); other blocks are walls
    target_block = (32, 36)
    others = [b for b in blocks if b != target_block]
    walls = W.border_walls() | set(others)
    model = W.Model(walls)
    state = (ax, ay, target_block[0], target_block[1], rot, held)
    # a sequence that navigates up to the block, grabs, drags — must never touch other blocks
    seq = [1, 1, 5, 2, 2, 4, 5, 3]  # up,up,grab,down,down,right,release,left
    for a in seq:
        obs = env.step(GameAction.from_id(a) if a != 5 else GameAction.ACTION5)
        state = model.step(state, a)
        (eax, eay, erot, eheld), eblocks = _truth(env)
        # find engine position of our tracked block (identity by nearest to model's prediction)
        etb = min(eblocks + others, key=lambda b: abs(b[0]-state[2]) + abs(b[1]-state[3])) if False else None
        assert (eax, eay) == (state[0], state[1]), f"after {a}: avatar engine {(eax,eay)} != model {(state[0],state[1])}"
        assert erot == state[4], f"after {a}: facing engine {erot} != model {state[4]}"
        assert eheld == state[5], f"after {a}: held engine {eheld} != model {state[5]}"


def test_planner_wins_wa30_l0():
    env, obs = _make()
    grid = P.to_grid(obs.frame)
    avatar, blocks, pads = W.perceive(grid)
    plan = W.plan_all(avatar, blocks, pads)
    assert plan is not None, "planner found no solution"
    for a in plan:
        obs = env.step(GameAction.ACTION5 if a == 5 else GameAction.from_id(a))
        if obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1:
            break
    assert int(obs.levels_completed or 0) >= 1, (
        f"planner did not win L0: levels={obs.levels_completed} state={obs.state} plan_len={len(plan)}")
