# scripts/discovery_bakeoff.py
"""Phase 1 — 3-way model-discovery bake-off on ls20 (black-box).

Drives each engine via the decide() contract against the offline ls20 env (capture_levelup
pattern), logging per-level cleared/actions and the (A_h/A_m)^2 efficiency. ls20 source is
touched ONLY by the A_h grader (bakeoff_metrics.human_baseline_actions).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py [budget]
"""
from __future__ import annotations
import logging, sys, time
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.bakeoff_metrics import efficiency, transfer_slope, human_baseline_actions
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.spatial_value_explorer import SpatialValueExplorer

logging.basicConfig(level=logging.ERROR)


def _retry(fn, tries=5, delay=1.0, what=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                msg = f"step failed after {tries} tries"
                if what:
                    msg += f": {what}"
                raise RuntimeError(msg) from e
            time.sleep(delay * (i + 1))


@dataclass
class LevelResult:
    level: int
    cleared: bool
    actions: int
    a_h: int | None
    eff: float


@dataclass
class EngineResult:
    name: str
    levels_cleared: int
    levels: list[LevelResult] = field(default_factory=list)
    @property
    def slope(self) -> float:
        return transfer_slope([lr.actions for lr in self.levels if lr.cleared])


def make_ls20():
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("bo"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    return client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["bakeoff"]))


def run_engine(name: str, engine, budget: int, max_level: int = 4) -> EngineResult:
    """Drive `engine.decide(...)`; charge each environment-altering action to the current level.

    max_level is the bake-off depth target (harness-level, not an engine constant) — ls20 has more levels;
    we compare engines over L1..L4.
    """
    env = make_ls20()
    obs = _retry(env.reset)
    res = EngineResult(name=name, levels_cleared=0)
    actions_this_level = 0
    cur_level = int(obs.levels_completed or 0)
    a_h_cache: dict[int, int | None] = {}
    n = 0
    while n < budget and cur_level < max_level:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        token = engine.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=cur_level,
            available=list(obs.available_actions or []),
        )
        if token[0] == "reset":
            obs = _retry(env.reset)
        elif token[0] == "S":
            aid = token[1]
            obs = _retry(lambda: env.step(GameAction.from_id(aid)), what=f"{name} L{cur_level} {token}")
            actions_this_level += 1
            n += 1
        else:  # "C"
            x, y = token[1], token[2]
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": x, "y": y}), what=f"{name} L{cur_level} {token}")
            actions_this_level += 1
            n += 1
        new_level = int(obs.levels_completed or 0)
        if new_level > cur_level:
            a_h = a_h_cache.setdefault(cur_level, human_baseline_actions(level=cur_level))
            res.levels.append(LevelResult(cur_level, True, actions_this_level, a_h,
                                          efficiency(a_h or 0, actions_this_level)))
            res.levels_cleared = new_level
            cur_level = new_level
            actions_this_level = 0
    return res


def print_table(results: list[EngineResult]) -> None:
    print(f"\n{'engine':<22}{'levels':<8}{'L1 act':<8}{'L1 eff':<8}{'slope':<8}", flush=True)
    for r in results:
        l1 = next((x for x in r.levels if x.level == 0), None)
        print(f"{r.name:<22}{r.levels_cleared:<8}"
              f"{(l1.actions if l1 else '-'):<8}{(round(l1.eff,3) if l1 else '-'):<8}"
              f"{round(r.slope,2):<8}", flush=True)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    engines = [
        ("salience(#2)", SalienceExplorer(seed=0)),
        ("spatial_value(#3)", SpatialValueExplorer(seed=0)),
    ]
    results = []
    for name, eng in engines:
        if hasattr(eng, "reset_all"):
            eng.reset_all()
        print(f"running {name}...", flush=True)
        results.append(run_engine(name, eng, budget))
    print_table(results)


if __name__ == "__main__":
    main()
