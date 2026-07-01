"""Faithful g50t harness: replay action lists via the real engine, auto-pumping
busy animation frames. Provides state probes for search."""
import os
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

_c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
_GID = next(e.game_id for e in _c.get_environments() if e.game_id.startswith('g50t'))

FILLER = 5  # ignored while busy

def new_env():
    env = _c.make(game_id=_GID, scorecard_id='x')
    env.reset()
    return env

def busy(env):
    return bool(env._game.vgwycxsxjz.jqpwhiraaj) or env._game.qgzorkgosv or env._game.hctlyapjnq

def pump(env, max_frames=400):
    """Advance filler frames until not busy or terminal."""
    obs = env._last_obs if hasattr(env,'_last_obs') else None
    frames = 0
    while busy(env) and frames < max_frames:
        obs = env.step(GameAction.from_id(FILLER))
        frames += 1
        if obs.state != GameState.NOT_FINISHED:
            break
    env._last_obs = obs
    return obs, frames

def do(env, a):
    """Apply one logical action a in {1,2,3,4,5}, then settle animations."""
    obs = env.step(GameAction.from_id(a))
    env._last_obs = obs
    if obs.state != GameState.NOT_FINISHED:
        return obs
    o2, _ = pump(env)
    return o2 if o2 is not None else obs

def replay(actions):
    env = new_env()
    obs = env._last_obs = None
    for a in actions:
        obs = do(env, a)
        if obs.state != GameState.NOT_FINISHED:
            break
    if obs is None:
        obs, _ = pump(env)
    return env, obs

def probe(env):
    g = env._game
    ctrl = g.vgwycxsxjz
    return dict(
        px=ctrl.dzxunlkwxt.x, py=ctrl.dzxunlkwxt.y,
        phase=ctrl.rlazdofsxb,
        nghost=len(ctrl.rloltuowth),
        gate=[(gt.x, gt.y, gt.is_visible) for gt in ctrl.uwxkstolmf],
        timer=g.twyixucrqi.x,
        dead=ctrl.dzxunlkwxt.pddqxjztas,
    )

if __name__ == "__main__":
    env, obs = replay([])
    print("fresh", obs.state, probe(env))
