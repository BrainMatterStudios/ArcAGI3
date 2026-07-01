"""Deployment-path validation: a solution found by the general search must REPLAY through the strict reactive
decide() interface (the eval execution path) and solve at the clean scored-play cost. This is the 'clean replay'
half of the search-replay paradigm (the dirty search play is discarded by max-over-runs scoring)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from general_solve import solve


class _Replay:
    def __init__(self, plan): self.gs = None; self.plan = list(plan); self.i = 0
    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        if gstate_terminal: return ("reset",)
        if self.i >= len(self.plan): return ("S", int(available[0]) if available else 5)
        t = self.plan[self.i]; self.i += 1; return t


def test_found_solution_replays_reactively():
    toks = solve("dc22", verbose=False)["tokens"]
    assert toks, "search must find a dc22 solution"
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("dc22"))
    env = client.make(game_id=gid, scorecard_id="dc22-replay-test"); obs = env.reset()
    pol = _Replay(toks); won = False
    for _ in range(len(toks) + 5):
        tok = pol.decide(P.to_grid(obs.frame), gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        if tok[0] == "reset": obs = env.reset()
        elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
        else: obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        if obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1:
            won = True; break
    assert won, "found solution must replay to a win through the reactive interface"
