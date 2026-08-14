"""T0 planner: Dijkstra over the exact transition graph (design doc §2 Layer 4).

Goals, in priority order:
  1. WIN replay — if a level-up (state, action) edge has been observed and its
     source is reachable, go execute it (shortest path).
  2. Frontier — otherwise head for the cheapest (state, untried-action) pair,
     where "cheapest" prices both distance and the action prior:
     cost = dist(state) + 1 + prior_penalty * rank(action in the state's menu).
     Menus arrive pre-ranked by the battery's REACTIVE/click-target priors
     (CLICK dominates 19/25 games), so rank 0 is the best-prior probe.

RESET awareness: every node gets a virtual RESET edge to the level root at
cost 1 (a RESET is one scored action) unless a real RESET edge from that node
was already observed. Fatal pairs are never traversed; nondeterministic edges
were purged by escalation and are never offered.

The planner is pure: it never touches an env. Model-proposed edges are
verified by executing them — the agent aborts and replans on any mismatch.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.graph import ActionKey, LevelGraph, NodeKey, RESET_ACTION

MenuFn = Callable[[NodeKey], list[ActionKey]]

INF = float("inf")


@dataclass
class Plan:
    """A verified-executable prefix plus one probe action at the end.

    `expected[i]` is the state the graph predicts after `actions[i]`; the
    final (frontier) action has no prediction and `expected` is one shorter
    than `actions` for kind="frontier". kind="none" means no reachable goal.
    """

    kind: str  # "win" | "frontier" | "none"
    actions: list[ActionKey] = field(default_factory=list)
    expected: list[NodeKey] = field(default_factory=list)
    cost: float = INF
    target: NodeKey | None = None


def dijkstra(
    graph: LevelGraph, start: NodeKey, *, reset_cost: float = 1.0
) -> tuple[dict[NodeKey, float], dict[NodeKey, tuple[NodeKey, ActionKey]]]:
    """Shortest known-cost to every reachable node over deterministic
    non-fatal edges plus the virtual RESET edge. Returns (dist, parent)."""
    adj = graph.adjacency()
    dist: dict[NodeKey, float] = {start: 0.0}
    parent: dict[NodeKey, tuple[NodeKey, ActionKey]] = {}
    counter = 0  # heap tie-breaker: keys are not comparable across types
    heap: list[tuple[float, int, NodeKey]] = [(0.0, counter, start)]
    root = graph.root
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist.get(u, INF):
            continue
        edges = list(adj.get(u, ()))
        # virtual reset: only when no real RESET edge from u was observed
        if root is not None and u != root \
                and not any(a == RESET_ACTION for a, _ in edges) \
                and (u, RESET_ACTION) not in graph.fatal:
            edges.append((RESET_ACTION, root))
        for action, v in edges:
            cost = reset_cost if action == RESET_ACTION else 1.0
            nd = d + cost
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = (u, action)
                counter += 1
                heapq.heappush(heap, (nd, counter, v))
    return dist, parent


def _path_to(
    graph: LevelGraph,
    parent: dict[NodeKey, tuple[NodeKey, ActionKey]],
    start: NodeKey,
    goal: NodeKey,
) -> tuple[list[ActionKey], list[NodeKey]]:
    actions: list[ActionKey] = []
    expected: list[NodeKey] = []
    node = goal
    while node != start:
        prev, action = parent[node]
        actions.append(action)
        expected.append(node)
        node = prev
    actions.reverse()
    expected.reverse()
    return actions, expected


def plan(
    graph: LevelGraph,
    current: NodeKey,
    menu_of: MenuFn,
    *,
    prior_penalty: float = 0.3,
    greedy_rank_max: int = 3,
    fatal_retry_cap: int = 2,
) -> Plan:
    """Best next plan from `current`.

    Fast path: if the current node itself offers an untried action of rank
    <= greedy_rank_max, probe it immediately — at prior_penalty <= 0.3 no
    other node can beat cost 1 + 0.3*3 = 1.9, since reaching it costs >= 2.
    This shortcut carries most exploration steps and keeps Dijkstra calls
    rare on fresh-state chains.
    """
    cur_untried = graph.untried(current, menu_of(current))
    if cur_untried:
        menu = menu_of(current)
        best = min(cur_untried, key=menu.index)
        if menu.index(best) <= greedy_rank_max:
            return Plan(kind="frontier", actions=[best], expected=[],
                        cost=1 + prior_penalty * menu.index(best),
                        target=current)

    dist, parent = dijkstra(graph, current)

    # goal 1: observed level-up edge
    best_win: tuple[float, NodeKey, ActionKey] | None = None
    for src, action in graph.win_edges:
        d = dist.get(src, INF)
        if d < INF and (best_win is None or d + 1 < best_win[0]):
            best_win = (d + 1, src, action)
    if best_win is not None:
        cost, src, action = best_win
        actions, expected = _path_to(graph, parent, current, src)
        return Plan(kind="win", actions=actions + [action], expected=expected,
                    cost=cost, target=src)

    # goal 2: cheapest reachable frontier, priced with the action prior
    best: tuple[float, float, NodeKey, ActionKey] | None = None
    for key in graph.nodes:
        d = dist.get(key, INF)
        if d == INF:
            continue
        if best is not None and d + 1 > best[0]:
            continue  # cannot beat the incumbent even at rank 0
        menu = menu_of(key)
        untried = graph.untried(key, menu)
        if not untried:
            continue
        action = min(untried, key=menu.index)
        cost = d + 1 + prior_penalty * menu.index(action)
        if best is None or (cost, d) < (best[0], best[1]):
            best = (cost, d, key, action)
    if best is None:
        # goal 3 (last resort): retry a never-succeeded fatal pair. Deaths
        # in these games are mostly timer-driven (masked budget bar), so one
        # death is weak evidence; bounded retries keep a single unlucky
        # death from walling off a corridor. Reached only when every
        # ordinary frontier is exhausted.
        best_fatal: tuple[float, NodeKey, ActionKey] | None = None
        for (src, action), deaths in graph.fatal.items():
            if deaths >= fatal_retry_cap or graph.edges.get((src, action)):
                continue
            d = dist.get(src, INF)
            if d < INF and (best_fatal is None or d + 1 < best_fatal[0]):
                best_fatal = (d + 1, src, action)
        if best_fatal is not None:
            cost, src, action = best_fatal
            actions, expected = _path_to(graph, parent, current, src)
            return Plan(kind="frontier", actions=actions + [action],
                        expected=expected, cost=cost, target=src)
        return Plan(kind="none")
    cost, _, key, action = best
    actions, expected = _path_to(graph, parent, current, key)
    return Plan(kind="frontier", actions=actions + [action], expected=expected,
                cost=cost, target=key)


if __name__ == "__main__":  # standalone self-check on a synthetic graph
    import numpy as np

    from engineered.graph import simple_key

    g = LevelGraph("test", 0)
    f = np.zeros((2, 2))
    for k in ("r", "a", "b"):
        g.ensure_node(k, f, [1, 2])
    g.set_root("r")
    a1, a2 = simple_key(1), simple_key(2)
    g.observe("r", a1, "a")
    g.observe("a", a1, "b")
    g.observe("b", a1, "r")  # cycle, all of a1 tried everywhere
    menu = {"r": [a1, a2], "a": [a1], "b": [a1]}
    p = plan(g, "b", lambda k: menu[k])
    # only frontier is a2 at r; b->r via observed edge costs 1
    assert p.kind == "frontier" and p.actions == [a1, a2], p
    # strand b: kill the cycle edge -> virtual RESET is the only way home
    g2 = LevelGraph("test", 0)
    for k in ("r", "a", "b"):
        g2.ensure_node(k, f, [1, 2])
    g2.set_root("r")
    g2.observe("r", a1, "a")
    g2.observe("a", a1, "b")
    g2.observe("b", a1, "b")  # self-loop; no way forward
    p2 = plan(g2, "b", lambda k: menu[k])
    assert p2.kind == "frontier" and p2.actions == [RESET_ACTION, a2], p2
    assert p2.cost == 2 + 0.3  # RESET 1 + probe 1 + rank-1 prior
    print("planner self-check OK")
