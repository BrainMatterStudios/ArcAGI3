"""GeneralSearchStrategy deploys search-replay over the reactive decide() interface: it drives an affordance
BFS by emitting reset/step tokens (dirty search play), caches the winning sequence, and replays it clean.
Validates the full eval-path deployment on dc22 (a genuine multi-step interleaved click+move solution)."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.general_search_strategy import GeneralSearchStrategy


def test_reactive_search_replay_solves_dc22():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("dc22"))
    env = c.make(game_id=gid, scorecard_id="gss-dc22-test"); obs = env.reset()
    pol = GeneralSearchStrategy()
    for _ in range(40000):
        tok = pol.decide(P.to_grid(obs.frame), gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        if tok[0] == "reset": obs = env.reset()
        elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
        else: obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        if pol.mode == "done" and int(obs.levels_completed or 0) >= 1:
            return
    assert False, "reactive search+replay should solve dc22"
