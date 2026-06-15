"""Verify the submission adapter (submission/my_agent.py) end-to-end.

Drives MyAgent.choose_action / is_done through the offline engine exactly like the
official ARC-AGI-3-Agents framework does (one action per call, action carries its own
data), confirming the reactive adapter + arcagi3 import path are wired correctly.
"""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "local-dev")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "submission"))
GAMES = str(ROOT / "src" / "arcagi3" / "games")


def _drive(game_id, budget=4000):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from my_agent import MyAgent

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES,
                    logger=logging.getLogger("t"))
    env = client.make(game_id=game_id, scorecard_id="sc")
    agent = MyAgent(game_id=game_id)
    obs = env.reset()
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        if agent.is_done(agent.frames, obs):
            break
        act = agent.choose_action(agent.frames, obs)
        if act.is_complex():
            d = act.action_data
            obs = env.step(GameAction.ACTION6, data={"x": int(d.x), "y": int(d.y)})
        else:
            obs = env.step(act)
        n += 1
    return obs, n


# SalienceExplorer is the submission policy: more thorough on real games (11 vs 9 levels
# head-to-head, no regressions) at the cost of action-efficiency on local nav/sokoban.
# Budgets below reflect its measured win profile (navg 3249, btnc 11, push 13411 actions).
def test_my_agent_wins_navg():
    obs, n = _drive("navg", budget=6000)
    assert obs.state.name == "WIN"


def test_my_agent_wins_btnc():
    obs, n = _drive("btnc", budget=4000)
    assert obs.state.name == "WIN"


def test_my_agent_wins_push():
    obs, n = _drive("push", budget=16000)
    assert obs.state.name == "WIN"
