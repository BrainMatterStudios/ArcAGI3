"""T0 exact transition-memory graph over masked states (design doc §2 Layer 3).

One `LevelGraph` per (game, level). Nodes are masked-state keys from the
Stage-1a perception layer (`Perception.state_key` bytes — hashable and
tick-stable); edges are (state, action) -> successor with observed counts.

Three properties the design doc requires, all first-class here:

  * Markov-violation detector: the same (state, action) observed with a
    DIFFERENT successor flags the source state and escalates its identity
    locally — future visits to that raw state are keyed by (previous
    effective state, arriving action, raw state), i.e. a one-step history
    split. Evidence recorded under the pre-split identity is purged (it is
    mixed-context and would poison exact planning); fatal knowledge is kept
    (conservative: avoiding a once-lethal pair costs at most one detour).

  * GAME_OVER-persistent memory: nothing in this module wipes on death.
    A lethal (state, action) is recorded in `fatal` and the graph survives
    the RESET — actions accumulate across attempts of a level (memory
    `arcagi3-action-accounting-measured`), so knowledge must too.

  * RESET awareness: the level root is recorded, and observed RESET edges
    are ordinary edges (cost 1 like everything else — a RESET is one scored
    action). The planner adds a virtual reset edge for nodes where RESET has
    not been observed yet.

Keys are `Hashable` (bytes for plain states, tuples for escalated ones);
actions are `ActionKey = (action_id, x, y)` with x = y = -1 for non-clicks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Iterable, NamedTuple

import numpy as np

# (action_id, x, y); x = y = -1 for non-click actions. RESET is id 0.
ActionKey = tuple[int, int, int]
RESET_ACTION: ActionKey = (0, -1, -1)

NodeKey = Hashable


def click_key(x: int, y: int) -> ActionKey:
    return (6, int(x), int(y))


def simple_key(action_id: int) -> ActionKey:
    return (int(action_id), -1, -1)


@dataclass
class Node:
    """One observed masked state: representative raw frame + action menu
    ingredients. The frame is the RAW settled frame (menus need real colors);
    identity always comes from the masked key, never from this array."""

    key: NodeKey
    frame: np.ndarray
    avail: tuple[int, ...] = ()


class ObserveResult(NamedTuple):
    new_edge: bool
    violation: bool  # this observation contradicted earlier evidence


@dataclass
class LevelGraph:
    """Exact transition memory for one level of one game."""

    stem: str
    level: int
    nodes: dict[NodeKey, Node] = field(default_factory=dict)
    edges: dict[tuple[NodeKey, ActionKey], Counter] = field(default_factory=dict)
    tried: dict[NodeKey, set[ActionKey]] = field(default_factory=dict)
    # pair -> observed death count. A Counter, not a set: timer-driven deaths
    # (the masked step-budget bar) can strike any pair once; a pair that has
    # never SUCCEEDED stays out of planning but must remain retriable or a
    # single unlucky death walls off a corridor (measured on tu93 L2).
    fatal: Counter = field(default_factory=Counter)
    win_edges: set[tuple[NodeKey, ActionKey]] = field(default_factory=set)
    violated: set[NodeKey] = field(default_factory=set)
    root: NodeKey | None = None
    n_transitions: int = 0
    violation_events: int = 0
    death_contradictions: int = 0

    # -- nodes -------------------------------------------------------------

    def ensure_node(self, key: NodeKey, frame: np.ndarray,
                    avail: Iterable[int]) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key=key, frame=np.asarray(frame, dtype=np.int16).copy(),
                        avail=tuple(sorted(a for a in avail if a)))
            self.nodes[key] = node
        return node

    def set_root(self, key: NodeKey) -> None:
        if self.root is None:
            self.root = key

    # -- identity escalation ----------------------------------------------

    @staticmethod
    def _raw(key: NodeKey) -> NodeKey:
        """The raw masked-state component of a (possibly composite) key."""
        if isinstance(key, tuple) and len(key) == 4 and key[0] == "ctx":
            return key[3]
        return key

    def effective_key(self, raw_key: NodeKey,
                      context: tuple[NodeKey, ActionKey] | None) -> NodeKey:
        """Plain key normally; a history-1 composite once the raw state is
        flagged Markov-violating and an arrival context is known.

        The context's state component is normalized to its RAW key: composite
        contexts would nest ("ctx" of "ctx" of ...), and that measurably
        exploded 46 real tu93 states into 771 never-merging nodes. One level
        of history, no more — deeper non-Markov structure shows up as repeat
        violations and is a dispatch signal, not something T0 should chase."""
        if raw_key in self.violated and context is not None:
            return ("ctx", self._raw(context[0]), context[1], raw_key)
        return raw_key

    def _escalate(self, key: NodeKey) -> None:
        """Split identity for `key`: future visits are context-keyed.

        Purges BOTH directions of stale evidence recorded against the
        pre-split identity:
          * outgoing edges + tried-marks (mixed-context — would poison exact
            planning; fatal pairs are KEPT, see module docstring);
          * incoming successor entries pointing at `key` — future arrivals
            land on context keys, and leaving plain-`key` successors in
            place makes every re-observed incoming edge look like a fresh
            violation and cascade the escalation backwards through the graph
            (measured: 16 real tu93 events snowballed to 692).
        """
        self.violated.add(self._raw(key))
        for edge_key in [k for k in self.edges if k[0] == key]:
            del self.edges[edge_key]
        self.tried.pop(key, None)
        for edge_key, succ in list(self.edges.items()):
            if key in succ:
                del succ[key]
                if not succ:
                    del self.edges[edge_key]
                    self.tried.get(edge_key[0], set()).discard(edge_key[1])

    # -- observation -------------------------------------------------------

    def observe(self, src: NodeKey, action: ActionKey, dst: NodeKey | None, *,
                game_over: bool = False, level_up: bool = False) -> ObserveResult:
        """Record one executed transition. `dst` is None only for GAME_OVER
        (the death screen is not a state). Returns whether the edge was new
        and whether it contradicted prior evidence."""
        self.n_transitions += 1
        self.tried.setdefault(src, set()).add(action)
        if game_over:
            # Dying where we previously moved is NOT treated as an identity
            # violation: in these games death is overwhelmingly driven by the
            # step-budget bar — deliberately masked OUT of identity — so it
            # is timer state, not board state. Splitting identity cannot
            # model a counter; it only shatters the graph. Death is recorded
            # as evidence: the pair keeps its successful successors (if any)
            # and a PURE-fatal pair (no success ever) never enters planning.
            contradiction = (src, action) in self.edges or (src, action) in self.win_edges
            new = (src, action) not in self.fatal
            self.fatal[(src, action)] += 1
            if contradiction:
                self.death_contradictions += 1
            return ObserveResult(new_edge=new, violation=False)
        if level_up:
            new = (src, action) not in self.win_edges
            self.win_edges.add((src, action))
            if (src, action) in self.fatal:
                self.death_contradictions += 1
            return ObserveResult(new_edge=new, violation=False)
        assert dst is not None
        succ = self.edges.setdefault((src, action), Counter())
        new = not succ
        contradiction = bool(succ) and dst not in succ
        succ[dst] += 1
        if contradiction:
            # same (state, action), different successor: genuine Markov
            # violation of the masked identity -> escalate locally
            self.violation_events += 1
            self._escalate(src)
            # the current observation's own context is unknown-mixed too;
            # it is intentionally dropped with the purge
        return ObserveResult(new_edge=new, violation=contradiction)

    # -- queries -----------------------------------------------------------

    def deterministic_successor(self, src: NodeKey,
                                action: ActionKey) -> NodeKey | None:
        """The successor if exactly one has ever been observed, else None
        (a contradicted pair is unusable for exact planning — though
        escalation normally purges it before this is asked)."""
        succ = self.edges.get((src, action))
        if succ is not None and len(succ) == 1:
            return next(iter(succ))
        return None

    def out_edges(self, src: NodeKey) -> Iterable[tuple[ActionKey, NodeKey]]:
        """Deterministic outgoing edges. A pair with successes stays usable
        even if it has also died once (timer deaths — see observe); a
        PURE-fatal pair has no successor entry and never appears."""
        for (s, action), succ in self.edges.items():
            if s == src and len(succ) == 1:
                yield action, next(iter(succ))

    def untried(self, src: NodeKey, menu: Iterable[ActionKey]) -> list[ActionKey]:
        """Menu actions never executed at `src` (fatal pairs are tried)."""
        done = self.tried.get(src, set())
        return [a for a in menu if a not in done]

    def adjacency(self) -> dict[NodeKey, list[tuple[ActionKey, NodeKey]]]:
        """Deterministic non-fatal edges grouped by source (planner input) —
        one O(E) pass instead of per-node scans."""
        adj: dict[NodeKey, list[tuple[ActionKey, NodeKey]]] = {}
        for (src, action), succ in self.edges.items():
            if len(succ) == 1:
                adj.setdefault(src, []).append((action, next(iter(succ))))
        return adj

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "fatal": len(self.fatal),
            "win_edges": len(self.win_edges),
            "violated_states": len(self.violated),
            "violation_events": self.violation_events,
            "death_contradictions": self.death_contradictions,
            "transitions": self.n_transitions,
        }


if __name__ == "__main__":  # standalone self-check on a synthetic game
    g = LevelGraph("test", 0)
    a1, a2 = simple_key(1), simple_key(2)
    g.ensure_node("s0", np.zeros((2, 2)), [1, 2])
    g.set_root("s0")
    for _ in range(3):
        assert not g.observe("s0", a1, "s1").violation
    assert g.deterministic_successor("s0", a1) == "s1"
    # violation: same (state, action), different successor
    r = g.observe("s0", a1, "s2")
    assert r.violation and "s0" in g.violated
    assert g.deterministic_successor("s0", a1) is None  # purged
    # escalated identity separates contexts
    k = g.effective_key("s0", ("sX", a2))
    assert k == ("ctx", "sX", a2, "s0")
    assert not g.observe(k, a1, "s1").violation
    assert g.deterministic_successor(k, a1) == "s1"
    # death persistence: fatal is remembered, nothing wiped
    g.observe("s1", a2, None, game_over=True)
    assert ("s1", a2) in g.fatal and g.deterministic_successor(k, a1) == "s1"
    print("graph self-check OK:", g.stats())
