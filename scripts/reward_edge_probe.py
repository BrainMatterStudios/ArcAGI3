"""Reward-edge viability probe for Phase B RVH No-T.

Counts how much true reward signal exists in the banked explorer's observed graph before any
learned value model is introduced.
"""

from __future__ import annotations

from collections import deque
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game

logging.basicConfig(level=logging.ERROR)


def _nodes_of(graph_like):
    return getattr(graph_like, "nodes", graph_like)


def summarize_reward_paths(graph_like, root: bytes) -> dict[str, int | None]:
    wm = _nodes_of(graph_like)
    positive_edges = []
    positive_nodes = set()
    rewarding_actions = set()
    for src, node in wm.items():
        for _action, (dst, reward) in node.edges.items():
            if reward > 0:
                positive_edges.append((src, dst))
                positive_nodes.add(dst)
                rewarding_actions.add(_action)

    shortest = None
    seen = {root}
    q = deque([(root, 0)])
    while q:
        key, dist = q.popleft()
        if key in positive_nodes:
            shortest = dist
            break
        node = wm.get(key)
        if node is None:
            continue
        for _action, (dst, _reward) in node.edges.items():
            if dst not in seen:
                seen.add(dst)
                q.append((dst, dist + 1))

    return {
        "states": len(wm),
        "positive_edges": len(positive_edges),
        "reachable_positive_nodes": len(positive_nodes.intersection(seen)) if shortest is not None else 0,
        "shortest_reward_distance": shortest,
        "rewarding_actions": len(rewarding_actions),
    }


def probe_game(game_id: str, budget: int, seed: int = 0) -> dict[str, int | None | str]:
    eng = TransferExplorer(seed=seed)
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
            levels=level,
            available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
            actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
            actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    summary = summarize_reward_paths(eng, eng.root_key)
    summary.update({
        "game": game_id,
        "levels": level,
        "actions": actions,
    })
    return summary


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["tu93", "vc33", "ls20", "re86"]
    rows = [probe_game(game, budget) for game in games]
    for row in rows:
        print(
            f"[{row['game']:8}] levels={row['levels']} actions={row['actions']} "
            f"states={row['states']} positive_edges={row['positive_edges']} "
            f"reward_nodes={row['reachable_positive_nodes']} shortest={row['shortest_reward_distance']} "
            f"rewarding_actions={row['rewarding_actions']}",
            flush=True,
        )
    total_pos = sum(int(row["positive_edges"]) for row in rows)
    total_states = sum(int(row["states"]) for row in rows)
    print("-" * 80, flush=True)
    print(f"TOTAL states={total_states} positive_edges={total_pos}", flush=True)


if __name__ == "__main__":
    main()
