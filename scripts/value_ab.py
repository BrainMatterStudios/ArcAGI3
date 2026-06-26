"""Decisive A/B gate: TransferExplorer vs ValueGuidedExplorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.value_guided_explorer import ValueGuidedExplorer
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game


def summarize_rows(rows):
    return {
        "improved": sum(1 for _game, base, test in rows if test > base),
        "regressed": sum(1 for _game, base, test in rows if test < base),
    }


def run_policy(policy, game: str, budget: int):
    env = make_game(game)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        token = policy.decide(
            P.to_grid(obs.frame),
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level,
            available=list(obs.available_actions or []),
        )
        if token[0] == "reset":
            obs = env.reset()
        elif token[0] == "S":
            obs = env.step(GameAction.from_id(token[1]))
            actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]})
            actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    return level


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["tu93", "vc33", "ls20", "re86", "wa30"]
    rows = []
    for game in games:
        base = run_policy(TransferExplorer(seed=0), game, budget)
        test = run_policy(ValueGuidedExplorer(seed=0), game, budget)
        rows.append((game, int(base), int(test)))
        print(f"[{game:8}] transfer={base} value={test}", flush=True)
    summary = summarize_rows(rows)
    print("-" * 80, flush=True)
    print(f"improved={summary['improved']} regressed={summary['regressed']}", flush=True)
    print(
        "GO-RULE: "
        + ("PASS" if summary["improved"] >= 1 and summary["regressed"] == 0 else "FAIL"),
        flush=True,
    )


if __name__ == "__main__":
    main()
