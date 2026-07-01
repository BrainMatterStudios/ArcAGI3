"""The mechanic-solver wired into the default PortfolioPolicy must (1) crack an archetype game (sb26) via
its own max-over-plays play, and (2) NOT regress a normal game (transfer_s0 still banks its levels). Uses a
small level_stall_limit so the portfolio rotates to the appended mechanic play quickly.
"""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.portfolio_policy import PortfolioPolicy


def _drive(prefix, stall, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=f"pf-{prefix}")
    obs = env.reset()
    pol = PortfolioPolicy(seed=0, level_stall_limit=stall)
    best = 0
    for _ in range(budget):
        pol._last_full_reset = bool(getattr(obs, "full_reset", False))
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
        best = max(best, int(obs.levels_completed or 0))
        if obs.state is GameState.WIN:
            break
    return best


def test_portfolio_cracks_sb26_via_mechanic_play():
    # coverage strategies score 0 on sb26; with a moderate stall each plateaus fast and the portfolio
    # rotates through to the appended mechanic play, which solves L0. (At eval, stall=20000 + huge budget
    # gives the same outcome; small stall here just reaches the last play cheaply.)
    assert _drive("sb26", stall=300, budget=5000) >= 1


def test_portfolio_no_regression_on_normal_game():
    # DEFAULT stall: transfer_s0 (strategy 0) runs long enough to bank tu93 levels; the appended mechanic
    # play is last and only abstains here -> the floor is preserved (append can't touch strategy 0).
    assert _drive("tu93", stall=20000, budget=1500) >= 1
