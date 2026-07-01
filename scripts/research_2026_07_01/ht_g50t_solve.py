"""g50t SOLVED — clone-recorder. Verify the solution from a fresh reset.

Mechanic: 2 phases. Phase 0 you record a ghost path; A5 banks it and resets you.
The banked ghost replays its moves in lockstep with your phase-1 moves; once its
recording is exhausted it STAYS PUT. Ghost ends on the button (37,7) -> relay ->
slides the gate at (13,37) away -> opens the down-passage so the real avatar can
reach the goal at player-cell (43,49).  WIN predicate (safkknjslo):
    whftgckbcu.x+1 == dzxunlkwxt.x and whftgckbcu.y+1 == dzxunlkwxt.y
"""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

SOLUTION = [4,4,4,4,5, 2,2,2,2, 2,2,2, 4,4,4,4,4, 2,2,2,2, 3]

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith('g50t'))

def run(actions, verbose=False):
    env = c.make(game_id=gid, scorecard_id='x')
    obs = env.reset()
    assert obs.levels_completed == 0
    saw_win = False
    for a in actions:
        obs = env.step(GameAction.from_id(a))
        # pump busy animation frames
        frames = 0
        while (env._game.vgwycxsxjz.jqpwhiraaj or env._game.qgzorkgosv or env._game.hctlyapjnq) \
              and obs.state == GameState.NOT_FINISHED and frames < 400:
            obs = env.step(GameAction.from_id(5))
            frames += 1
        if obs.state == GameState.WIN:
            saw_win = True
        if obs.levels_completed >= 1 and not saw_win:
            saw_win = True
    return obs, saw_win

if __name__ == "__main__":
    print("SOLUTION (%d actions):" % len(SOLUTION), SOLUTION)
    for trial in range(3):
        obs, saw_win = run(SOLUTION)
        print(f"trial {trial}: levels_completed={obs.levels_completed} state={obs.state.name} saw_win={saw_win}")
    assert obs.levels_completed >= 1, "NOT SOLVED"
    print("\n*** g50t level 0 SOLVED: levels_completed >= 1 from fresh reset ***")
