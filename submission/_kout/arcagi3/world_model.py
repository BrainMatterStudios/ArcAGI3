"""World model: a directed graph of observed states and action transitions.

Nodes are state keys (hash bytes of the masked grid). Edges record, for each (state,
action) actually taken, the resulting state and the reward (change in levels_completed).
Assumes (locally) deterministic dynamics: state + action -> same next state. The graph
drives exploration (find nearest unexplored frontier) and exploitation (replay action
sequences that lead to reward).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Optional

# An Action is a small hashable token the agent maps to a GameAction:
#   ("S", action_id)          simple action (1..5, 7)
#   ("C", x, y)               complex click (ACTION6 at x,y)
Action = tuple


@dataclass
class Node:
    key: bytes
    candidate_actions: tuple[Action, ...] = ()  # full action set proposed at this state
    edges: dict[Action, tuple[bytes, float]] = field(default_factory=dict)  # action -> (next_key, reward)
    terminal: bool = False  # GAME_OVER reached here (only RESET escapes)
    visits: int = 0

    def untried(self) -> list[Action]:
        return [a for a in self.candidate_actions if a not in self.edges]


class WorldModel:
    def __init__(self) -> None:
        self.nodes: dict[bytes, Node] = {}

    def observe(self, key: bytes, candidates: tuple[Action, ...], terminal: bool = False) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key=key, candidate_actions=candidates, terminal=terminal)
            self.nodes[key] = node
        else:
            # merge any newly-proposed candidates (keep order, dedup)
            if candidates and candidates != node.candidate_actions:
                seen = set(node.candidate_actions)
                merged = list(node.candidate_actions) + [c for c in candidates if c not in seen]
                node.candidate_actions = tuple(merged)
            node.terminal = node.terminal or terminal
        node.visits += 1
        return node

    def record(self, key: bytes, action: Action, next_key: bytes, reward: float) -> None:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key=key)
            self.nodes[key] = node
        node.edges[action] = (next_key, reward)

    def has_untried(self, key: bytes) -> bool:
        n = self.nodes.get(key)
        return bool(n and not n.terminal and n.untried())

    def reward_action(self, key: bytes) -> Optional[Action]:
        """If this state has a known action that yielded positive reward, return it."""
        n = self.nodes.get(key)
        if not n:
            return None
        best, best_r = None, 0.0
        for a, (_nk, r) in n.edges.items():
            if r > best_r:
                best, best_r = a, r
        return best

    def path_to_frontier(self, start: bytes, max_depth: int = 100000) -> Optional[list[Action]]:
        """BFS over known edges to the nearest non-terminal node with untried actions.

        Returns the action sequence from `start` to that frontier node (empty list if
        `start` itself is a frontier), or None if none reachable.
        """
        if self.has_untried(start):
            return []
        visited = {start}
        # queue of (key, path)
        q: deque[tuple[bytes, list[Action]]] = deque([(start, [])])
        while q:
            key, path = q.popleft()
            if len(path) > max_depth:
                continue
            node = self.nodes.get(key)
            if not node:
                continue
            for action, (nk, _r) in node.edges.items():
                if nk in visited:
                    continue
                visited.add(nk)
                npath = path + [action]
                if self.has_untried(nk):
                    return npath
                q.append((nk, npath))
        return None

    def path_between(self, start: bytes, goal: bytes) -> Optional[list[Action]]:
        """Shortest known action path from start to goal, or None."""
        if start == goal:
            return []
        visited = {start}
        q: deque[tuple[bytes, list[Action]]] = deque([(start, [])])
        while q:
            key, path = q.popleft()
            node = self.nodes.get(key)
            if not node:
                continue
            for action, (nk, _r) in node.edges.items():
                if nk in visited:
                    continue
                if nk == goal:
                    return path + [action]
                visited.add(nk)
                q.append((nk, path + [action]))
        return None

    def __len__(self) -> int:
        return len(self.nodes)
