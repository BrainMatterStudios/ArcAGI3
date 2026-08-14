"""Planner: Dijkstra over exact + rule-predicted edges (design doc §2 Layer 4).

Stage 1b (T0) planned over observed edges only; its measured failure mode was
the blind frontier sweep — every untried (state, action) pair probed at cost 1,
including the thousands a gated rule could already predict. Stage 2a adds the
effect model:

Goals, in priority order:
  1. WIN replay / commit mode — if a level-up (state, action) edge has been
     observed and its source is reachable via observed OR predicted edges, go
     execute it (shortest path; predicted hops cost 1 + penalty and are
     verified on execution). Once this goal exists the planner never explores
     again — this is where the (b/a)^2 bill gets paid.
  2. Frontier, re-prioritized by the effect model:
     (a) cheapest UNPREDICTED untried pair — genuinely unknown outcome,
         maximum information: cost = dist + 1 + prior_penalty * menu rank.
     (b) cheapest NOVEL predicted state — a state no one has visited, reached
         by chaining gated-rule predictions up to max_pred_depth hops beyond
         the observed set. Executing the path verifies every predicted edge
         (mismatch -> demotion) and is where rule-guided exploration finds
         wins without sweeping.
     (c) only when (a) and (b) are exhausted: the remaining predicted untried
         pairs, cheapest first — the T0 sweep order, kept for completeness
         (a wrong "null"/known-successor prediction cannot hide a win
         forever; it is merely probed last).
  3. Bounded retry of never-succeeded fatal pairs (timer deaths), then none.

RESET awareness unchanged: virtual RESET edge to the level root at cost 1.
The planner is pure: it never touches an env. Every model-proposed edge is
verified by executing it — the agent aborts, demotes and replans on mismatch.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Hashable, Protocol

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.graph import ActionKey, LevelGraph, NodeKey, RESET_ACTION

MenuFn = Callable[[NodeKey], list[ActionKey]]

INF = float("inf")


class EffectsPort(Protocol):
    """What the planner needs from the effect model (agent supplies it)."""

    @property
    def active(self) -> bool: ...          # any gated rule at all?
    max_pred_depth: int
    max_pred_nodes: int

    def classify(self, key: NodeKey, akey: ActionKey):
        """Prediction for an untried action at a REAL node: an object with
        .kind in {"edge", "null"} (+ .key/.frame/.penalty for "edge"),
        or None when no gated rule speaks."""
        ...

    def chain(self, key: Hashable, frame, avail) -> list:
        """[(akey, key, frame, penalty)] successors of a predicted frame."""
        ...


@dataclass
class Plan:
    """A verified-executable prefix plus one probe action at the end.

    `expected[i]` is the state predicted after `actions[i]`; for probe plans
    (kinds "frontier") the final action has no prediction and `expected` is
    one shorter than `actions`. For fully predicted plans (win replay or
    novel-predicted targets) `expected` covers every action. `predicted[i]`
    marks steps that traverse a rule-predicted (unverified) edge — a mismatch
    there must demote the rule that proposed it. kind="none": no goal.
    """

    kind: str  # "win" | "frontier" | "none"
    actions: list[ActionKey] = field(default_factory=list)
    expected: list[NodeKey] = field(default_factory=list)
    predicted: list[bool] = field(default_factory=list)
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


def _predicted_expansion(
    graph: LevelGraph,
    dist: dict[NodeKey, float],
    effects: EffectsPort,
    menu_of: MenuFn,
) -> tuple[dict[NodeKey, float],
           dict[NodeKey, tuple[NodeKey, ActionKey]],
           set[NodeKey]]:
    """Dijkstra continuation into rule-predicted space.

    Seeds: predicted edges from untried pairs at reachable REAL nodes whose
    predicted successor is NOT a reachable real node. Expansion: T1 chains on
    predicted frames, depth- and node-capped. Returns (pred_dist,
    pred_parent, novel) where novel = predicted keys absent from the graph.
    """
    pred_dist: dict[NodeKey, float] = {}
    pred_parent: dict[NodeKey, tuple[NodeKey, ActionKey]] = {}
    meta: dict[NodeKey, tuple] = {}  # key -> (frame, avail, depth)
    heap: list[tuple[float, int, NodeKey]] = []
    counter = 0

    for key, node in graph.nodes.items():
        d = dist.get(key, INF)
        if d == INF:
            continue
        menu = menu_of(key)
        for akey in graph.untried(key, menu):
            p = effects.classify(key, akey)
            if p is None or p.kind != "edge":
                continue
            if p.key in dist:
                continue  # already reachable via observed edges
            nd = d + 1.0 + p.penalty
            if nd < pred_dist.get(p.key, INF):
                pred_dist[p.key] = nd
                pred_parent[p.key] = (key, akey)
                meta[p.key] = (p.frame, node.avail, 1)
                counter += 1
                heapq.heappush(heap, (nd, counter, p.key))

    popped = 0
    while heap and popped < effects.max_pred_nodes:
        d, _, u = heapq.heappop(heap)
        if d > pred_dist.get(u, INF):
            continue
        popped += 1
        frame, avail, depth = meta[u]
        if depth >= effects.max_pred_depth:
            continue
        for akey, k2, f2, pen in effects.chain(u, frame, avail):
            if k2 in dist:
                continue
            nd = d + 1.0 + pen
            if nd < pred_dist.get(k2, INF):
                pred_dist[k2] = nd
                pred_parent[k2] = (u, akey)
                meta[k2] = (f2, avail, depth + 1)
                counter += 1
                heapq.heappush(heap, (nd, counter, k2))

    novel = {k for k in pred_dist if k not in graph.nodes}
    return pred_dist, pred_parent, novel


def _mixed_path(
    graph: LevelGraph,
    parent: dict[NodeKey, tuple[NodeKey, ActionKey]],
    pred_parent: dict[NodeKey, tuple[NodeKey, ActionKey]],
    start: NodeKey,
    goal: NodeKey,
) -> tuple[list[ActionKey], list[NodeKey], list[bool]]:
    """Path start -> goal whose suffix runs through predicted space."""
    suffix_a: list[ActionKey] = []
    suffix_e: list[NodeKey] = []
    node = goal
    while node in pred_parent:
        prev, action = pred_parent[node]
        suffix_a.append(action)
        suffix_e.append(node)
        node = prev
    suffix_a.reverse()
    suffix_e.reverse()
    prefix_a, prefix_e = _path_to(parent, start, node)
    actions = prefix_a + suffix_a
    expected = prefix_e + suffix_e
    predicted = [False] * len(prefix_a) + [True] * len(suffix_a)
    return actions, expected, predicted


def plan(
    graph: LevelGraph,
    current: NodeKey,
    menu_of: MenuFn,
    *,
    prior_penalty: float = 0.3,
    greedy_rank_max: int = 3,
    fatal_retry_cap: int = 2,
    effects: EffectsPort | None = None,
) -> Plan:
    """Best next plan from `current` (see module docstring for the goal
    ladder). With effects=None this is exactly the Stage-1 T0 planner.

    Fast path: if the current node itself offers an untried, UNPREDICTED
    action of rank <= greedy_rank_max, probe it immediately — at
    prior_penalty <= 0.3 no other probe can beat cost 1 + 0.3*3 = 1.9, since
    reaching any other node costs >= 2. Predicted pairs are excluded: probing
    what a gated rule already knows is the Stage-1b waste this tier removes.
    """
    use_fx = effects is not None and effects.active

    def unpredicted(key: NodeKey, untried: list[ActionKey]) -> list[ActionKey]:
        if not use_fx:
            return untried
        return [a for a in untried if effects.classify(key, a) is None]

    # commit-mode precedence: the greedy probe shortcut is only legal while
    # NO win edge is known — otherwise exploration would preempt the replay
    # (the "stop exploring once the win is known" contract).
    if not graph.win_edges:
        cur_menu = menu_of(current)
        cur_untried = unpredicted(current, graph.untried(current, cur_menu))
        if cur_untried:
            best = min(cur_untried, key=cur_menu.index)
            if cur_menu.index(best) <= greedy_rank_max:
                return Plan(kind="frontier", actions=[best], expected=[],
                            predicted=[False],
                            cost=1 + prior_penalty * cur_menu.index(best),
                            target=current)

    dist, parent = dijkstra(graph, current)

    pred_dist: dict[NodeKey, float] = {}
    pred_parent: dict[NodeKey, tuple[NodeKey, ActionKey]] = {}
    novel: set[NodeKey] = set()
    if use_fx:
        pred_dist, pred_parent, novel = _predicted_expansion(
            graph, dist, effects, menu_of)

    # goal 1: observed level-up edge, reachable via observed or predicted path
    # (win_edges is a set — iterate sorted so tie-breaks are hash-seed-free)
    best_win: tuple[float, NodeKey, ActionKey, bool] | None = None
    for src, action in sorted(graph.win_edges, key=repr):
        d_obs = dist.get(src, INF)
        d_pred = pred_dist.get(src, INF)
        d = min(d_obs, d_pred)
        if d < INF and (best_win is None or d + 1 < best_win[0]):
            best_win = (d + 1, src, action, d_pred < d_obs)
    if best_win is not None:
        cost, src, action, via_pred = best_win
        if via_pred:
            actions, expected, predicted = _mixed_path(
                graph, parent, pred_parent, current, src)
        else:
            actions, expected = _path_to(parent, current, src)
            predicted = [False] * len(actions)
        return Plan(kind="win", actions=actions + [action],
                    expected=expected, predicted=predicted + [False],
                    cost=cost, target=src)

    # goal 2a: cheapest reachable UNPREDICTED frontier pair
    best: tuple[float, float, NodeKey, ActionKey] | None = None
    for key in graph.nodes:
        d = dist.get(key, INF)
        if d == INF:
            continue
        if best is not None and d + 1 > best[0]:
            continue  # cannot beat the incumbent even at rank 0
        menu = menu_of(key)
        untried = unpredicted(key, graph.untried(key, menu))
        if not untried:
            continue
        action = min(untried, key=menu.index)
        cost = d + 1 + prior_penalty * menu.index(action)
        if best is None or (cost, d) < (best[0], best[1]):
            best = (cost, d, key, action)

    # goal 2b: cheapest NOVEL predicted state (rule-guided exploration).
    # `novel` is a set — deterministic repr tie-break, not hash order.
    best_novel: tuple[float, NodeKey] | None = None
    for k in novel:
        d = pred_dist[k]
        if best_novel is None or (d, repr(k)) < (best_novel[0],
                                                 repr(best_novel[1])):
            best_novel = (d, k)

    if best is not None and (best_novel is None or best[0] <= best_novel[0]):
        cost, _, key, action = best
        actions, expected = _path_to(parent, current, key)
        return Plan(kind="frontier", actions=actions + [action],
                    expected=expected,
                    predicted=[False] * (len(actions) + 1),
                    cost=cost, target=key)
    if best_novel is not None:
        cost, key = best_novel
        actions, expected, predicted = _mixed_path(
            graph, parent, pred_parent, current, key)
        return Plan(kind="frontier", actions=actions, expected=expected,
                    predicted=predicted, cost=cost, target=key)

    # goal 2c: predicted-but-untried pairs, cheapest first (T0 completeness:
    # a wrong "null"/known-successor prediction cannot hide a win forever)
    if use_fx:
        best_c: tuple[float, float, NodeKey, ActionKey] | None = None
        for key in graph.nodes:
            d = dist.get(key, INF)
            if d == INF:
                continue
            if best_c is not None and d + 1 > best_c[0]:
                continue
            menu = menu_of(key)
            untried = graph.untried(key, menu)
            if not untried:
                continue
            action = min(untried, key=menu.index)
            cost = d + 1 + prior_penalty * menu.index(action)
            if best_c is None or (cost, d) < (best_c[0], best_c[1]):
                best_c = (cost, d, key, action)
        if best_c is not None:
            cost, _, key, action = best_c
            actions, expected = _path_to(parent, current, key)
            return Plan(kind="frontier", actions=actions + [action],
                        expected=expected,
                        predicted=[False] * (len(actions) + 1),
                        cost=cost, target=key)

    # goal 3 (last resort): retry a never-succeeded fatal pair. Deaths in
    # these games are mostly timer-driven (masked budget bar), so one death
    # is weak evidence; bounded retries keep a single unlucky death from
    # walling off a corridor. Reached only when every frontier is exhausted.
    best_fatal: tuple[float, NodeKey, ActionKey] | None = None
    for (src, action), deaths in graph.fatal.items():
        if deaths >= fatal_retry_cap or graph.edges.get((src, action)):
            continue
        d = dist.get(src, INF)
        if d < INF and (best_fatal is None or d + 1 < best_fatal[0]):
            best_fatal = (d + 1, src, action)
    if best_fatal is not None:
        cost, src, action = best_fatal
        actions, expected = _path_to(parent, current, src)
        return Plan(kind="frontier", actions=actions + [action],
                    expected=expected,
                    predicted=[False] * (len(actions) + 1),
                    cost=cost, target=src)
    return Plan(kind="none")


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

    # effects: a fake port that predicts a2 at r -> novel state "n"
    class _FX:
        active = True
        max_pred_depth = 3
        max_pred_nodes = 100

        class _P:
            kind = "edge"
            key = "n"
            frame = None
            penalty = 0.25

        def classify(self, key, akey):
            return self._P() if (key, akey) == ("r", a2) else None

        def chain(self, key, frame, avail):
            return []

    p3 = plan(g, "b", lambda k: menu[k], effects=_FX())
    # a2@r is now predicted -> no unpredicted frontier; goal 2b routes to the
    # novel predicted state "n" through r
    assert p3.kind == "frontier" and p3.actions == [a1, a2], p3
    assert p3.predicted == [False, True] and p3.target == "n", p3
    print("planner self-check OK")
