"""Generalization harness for the human-like general-solver research.

Runs a reactive policy on offline games under the STRICT eval contract (the policy sees ONLY the frame grid,
available actions, game state, and levels_completed -- no engine introspection), and measures ACTIONS-TO-SOLVE
per level (the RHAE currency: score ~ (human_actions/agent_actions)^2). Supports a leave-mechanics-out split so
we can test blind generalization: develop on some games, measure on games the policy has never been tuned for.

This is paradigm-neutral: any candidate agent that implements `decide(grid, gstate_terminal, gstate_notplayed,
levels, available) -> token` (token in {("reset",), ("S",id), ("C",x,y)}) can be measured here.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

ALL_GAMES = ["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0",
             "r11l","re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]


@dataclass
class GameResult:
    game: str
    levels_solved: int = 0
    total_actions: int = 0
    actions_per_level: list = field(default_factory=list)   # actions spent to reach each new level
    wall_s: float = 0.0
    error: str = ""


def run_policy(policy, game: str, budget: int = 3000, seed: int = 0) -> GameResult:
    """Play one game with a policy under the strict frames+feedback contract; measure actions-to-each-level."""
    t0 = time.time()
    try:
        client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
        env = client.make(game_id=gid, scorecard_id=f"gen-{game}")
        obs = env.reset()
    except Exception as e:  # noqa: BLE001
        return GameResult(game=game, error=f"init:{e}")

    res = GameResult(game=game)
    last_level = int(obs.levels_completed or 0)
    actions_since_level = 0
    for _ in range(budget):
        # STRICT contract: hand the policy only what the eval provides.
        grid = P.to_grid(obs.frame)
        try:
            token = policy.decide(
                grid,
                gstate_terminal=(obs.state == GameState.GAME_OVER),
                gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                levels=int(obs.levels_completed or 0),
                available=list(obs.available_actions or []),
            )
        except Exception as e:  # noqa: BLE001
            res.error = f"decide:{e}"; break

        if token[0] == "reset":
            obs = env.reset()
        elif token[0] == "S":
            obs = env.step(GameAction.from_id(int(token[1])))
        elif token[0] == "C":
            obs = env.step(GameAction.ACTION6, data={"x": int(token[1]), "y": int(token[2])})
        else:
            res.error = f"bad token {token!r}"; break
        res.total_actions += 1
        actions_since_level += 1

        lvl = int(obs.levels_completed or 0)
        if lvl > last_level:
            res.actions_per_level.append(actions_since_level)
            res.levels_solved = lvl
            last_level = lvl
            actions_since_level = 0
        if obs.state == GameState.WIN:
            break
    res.wall_s = time.time() - t0
    return res


def measure(policy_factory, games=None, budget: int = 3000, seed: int = 0, verbose: bool = True):
    """Run a policy (fresh instance per game) across games; return results + a summary."""
    games = games or ALL_GAMES
    results = []
    for g in games:
        r = run_policy(policy_factory(seed), g, budget=budget, seed=seed)
        results.append(r)
        if verbose:
            apl = ",".join(map(str, r.actions_per_level)) or "-"
            print(f"  {g:>6}  L{r.levels_solved}  acts={r.total_actions:>5}  per-level=[{apl}]  {r.wall_s:.1f}s"
                  + (f"  ERR {r.error}" if r.error else ""))
    solved = [r for r in results if r.levels_solved >= 1]
    if verbose:
        print(f"SUMMARY: solved {len(solved)}/{len(results)} games at >=L1; "
              f"median actions-to-L1 among solved = "
              f"{int(np.median([r.actions_per_level[0] for r in solved])) if solved else '-'}")
    return results


class RandomPolicy:
    """Baseline: uniform over available simple actions + occasional random click. Quantifies blind-play cost."""
    def __init__(self, seed=0):
        self.gs = None
        self.rng = np.random.default_rng(seed)

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        if gstate_terminal:
            return ("reset",)
        avail = list(available)
        simple = [a for a in avail if a in (1, 2, 3, 4, 5)]
        if 6 in avail and self.rng.random() < 0.15:
            return ("C", int(self.rng.integers(0, 64)), int(self.rng.integers(0, 64)))
        if simple:
            return ("S", int(self.rng.choice(simple)))
        return ("reset",)


if __name__ == "__main__":
    import sys
    games = sys.argv[1:] or ["sb26", "wa30", "tu93", "cd82", "ls20"]
    print("Random-policy baseline (quantifies how expensive blind exploration is under RHAE):")
    measure(lambda s: RandomPolicy(s), games=games, budget=2000)
