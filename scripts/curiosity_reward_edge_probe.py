"""Gate 1 (decisive): does reward-free curiosity stumble a first reward edge on a wall game
where the baseline produces ZERO? If both wall games stay at zero, the bet has failed."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game
from scripts.reward_edge_probe import summarize_reward_paths

logging.basicConfig(level=logging.ERROR)


def gate1_summary(rows):
    unlocked = sum(1 for _g, base, cur in rows if base == 0 and cur > 0)
    return {"unlocked": unlocked, "pass": unlocked >= 1}


def positive_edges(game_id: str, budget: int, enable_curiosity: bool, seed: int = 0) -> int:
    eng = ObjectCuriosityExplorer(seed=seed, enable_object_curiosity=enable_curiosity)
    eng.reset_all()
    env = make_game(game_id)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level, available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1])); actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    summary = summarize_reward_paths(eng, eng.root_key)
    return int(summary["positive_edges"])


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["ls20", "re86"]
    rows = []
    for game in games:
        base = positive_edges(game, budget, enable_curiosity=False)
        cur = positive_edges(game, budget, enable_curiosity=True)
        rows.append((game, base, cur))
        print(f"[{game:8}] base_pos_edges={base} curiosity_pos_edges={cur}", flush=True)
    summary = gate1_summary(rows)
    print("-" * 80, flush=True)
    print(f"unlocked={summary['unlocked']}", flush=True)
    print("GATE 1: " + ("PASS" if summary["pass"] else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
