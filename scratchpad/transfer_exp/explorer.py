"""Graph explorer for ARC-AGI-3, with a pluggable cross-level knowledge carrier.

Built to answer ONE question: does knowledge learned on level 1 reduce the cost of
clearing later levels? The published implementations all discard everything at a
level boundary and both sets of authors name that as their main open problem, so
the carrier is the part under test -- everything else is a faithful-enough
reimplementation of the published method to make the comparison meaningful.

Design follows Rudakov, Shock & Cowley (arXiv 2512.24156):
  * node identity = hash of the frame with HUD/counter pixels masked out
  * action space reduced to one candidate click per 4-connected colour component
  * five priority tiers, escalated GLOBALLY (tier 0 exhausted everywhere before
    tier 1 is touched anywhere) -- their ablation shows the tier gate is what pays
  * navigation by WALKING the graph toward the nearest frontier, never by reset
  * dead edges (no frame change) pruned permanently at that node

Deliberately NOT used: environment cloning. It works offline and is meaningless in
competition mode, where the env is an HTTP wrapper around a server-side guid, so
building on it would produce a search that passes every local test and silently
aliases at submission.
"""
from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'ideas'))
import hud_mask as _hud

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_SALIENT_MIN = 6          # colours 6..15 read as chromatic/foreground
_MEDIUM = (2, 32)         # a "button-like" component side length
_MAX_COMPONENTS = 64      # cap candidate clicks per node
_SUSPICIOUS_THRESHOLD = 3  # confirmations before trusting a back-to-start edge


def frame_np(obs) -> np.ndarray:
    """Settled frame as a 2D int array. The API returns a stack; the last is settled."""
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


# ---------------------------------------------------------------------------
# the thing under test
# ---------------------------------------------------------------------------

@dataclass
class Knowledge:
    """What an explorer may carry across a level boundary.

    COLD arm constructs a fresh one at every level; WARM arm carries it forward.
    Nothing here is level-specific -- these are claims about the GAME.
    """
    hud_mask: np.ndarray | None = None       # pixels that change on every action
    dead_simple: set[int] = field(default_factory=set)   # non-click ids that never did anything
    simple_tries: dict[int, int] = field(default_factory=dict)
    simple_hits: dict[int, int] = field(default_factory=dict)
    colour_tries: dict[int, int] = field(default_factory=dict)   # click effectiveness by colour
    colour_hits: dict[int, int] = field(default_factory=dict)

    def colour_weight(self, colour: int) -> float:
        """Laplace-smoothed effectiveness, used as a sampling weight.

        Weighting the CHOICE rather than nudging a tier is deliberate: measured
        click-effect rates sit between 2% and 50%, so the earlier coarse tier
        adjustment never fired and the carrier was inert -- both arms ran the same
        policy and produced bit-identical trajectories.
        """
        t = self.colour_tries.get(colour, 0)
        h = self.colour_hits.get(colour, 0)
        return (h + 0.5) / (t + 1.0)

    def simple_weight(self, aid: int) -> float:
        t = self.simple_tries.get(aid, 0)
        h = self.simple_hits.get(aid, 0)
        return (h + 0.5) / (t + 1.0)

    def note_simple(self, aid: int, changed: bool) -> None:
        self.simple_tries[aid] = self.simple_tries.get(aid, 0) + 1
        if changed:
            self.simple_hits[aid] = self.simple_hits.get(aid, 0) + 1
            self.dead_simple.discard(aid)
        elif self.simple_tries[aid] >= 8 and self.simple_hits.get(aid, 0) == 0:
            self.dead_simple.add(aid)

    def note_click(self, colour: int, changed: bool) -> None:
        self.colour_tries[colour] = self.colour_tries.get(colour, 0) + 1
        if changed:
            self.colour_hits[colour] = self.colour_hits.get(colour, 0) + 1

    def summary(self) -> dict:
        return {
            "hud_pixels": int(self.hud_mask.sum()) if self.hud_mask is not None else 0,
            "dead_simple": sorted(self.dead_simple),
            "colours_seen": len(self.colour_tries),
        }


# ---------------------------------------------------------------------------
# per-level search state
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    edges: dict = field(default_factory=dict)     # action_key -> {"result", "target", "tier"}
    order: list = field(default_factory=list)


class GraphExplorer:
    def __init__(self, knowledge: Knowledge, seed: int = 0, use_knowledge: bool = True,
                 game_id: str | None = None):
        self.game_id = game_id
        self.k = knowledge
        self.use_knowledge = use_knowledge
        self.rng = random.Random(seed)
        self.reset_level_state()

    def reset_level_state(self) -> None:
        """Graph is always per-level. Only `Knowledge` may cross a boundary."""
        self.nodes: dict[str, _Node] = {}
        self.succ: dict[str, set[str]] = {}
        self.pred: dict[str, set[str]] = {}
        self.active_tier = 0
        self.dist: dict[str, int] = {}
        self.next_hop: dict[str, tuple] = {}
        self._dist_dirty = True
        self.level_first_hash: str | None = None
        self.suspicious: dict[tuple, int] = {}

    # -- perception ---------------------------------------------------------

    def _masked(self, grid: np.ndarray) -> np.ndarray:
        if self.use_knowledge and self.k.hud_mask is not None and self.k.hud_mask.shape == grid.shape:
            g = grid.copy()
            g[self.k.hud_mask] = 16
            return g
        return grid

    def node_id(self, grid: np.ndarray) -> str:
        g = self._masked(grid)
        # E1: strip the HUD before hashing. 18/25 games tick a step-budget bar into
        # the frame before the legality check, so without this the node id changes on
        # every action, the graph never merges, and the search degenerates to a tree.
        # Measured collapse from masking on affected games: 16.14x (lf52 83.8x).
        if self.game_id is not None:
            g = _hud.mask_frame(g, self.game_id)
        return hashlib.blake2b(
            np.ascontiguousarray(g, dtype=np.int8).tobytes(),
            digest_size=16,
            person=repr(g.shape).encode()[:16],
        ).hexdigest()

    def _components(self, grid: np.ndarray):
        """(colour, y, x) candidate clicks -- one random interior pixel per component.

        Random interior rather than centroid: a centroid can fall outside a concave
        or ring-shaped component.
        """
        out = []
        for col in np.unique(grid):
            lab, n = ndimage.label(grid == col, structure=_S4)
            if n == 0:
                continue
            objs = ndimage.find_objects(lab)
            for idx in range(1, n + 1):
                sl = objs[idx - 1]
                if sl is None:
                    continue
                ys, xs = np.nonzero(lab[sl] == idx)
                if not len(ys):
                    continue
                j = self.rng.randrange(len(ys))
                y = int(ys[j] + sl[0].start)
                x = int(xs[j] + sl[1].start)
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                out.append((int(col), y, x, h, w))
                if len(out) >= _MAX_COMPONENTS:
                    return out
        return out

    @staticmethod
    def _tier(colour: int, h: int, w: int) -> int:
        salient = colour >= _SALIENT_MIN
        medium = _MEDIUM[0] <= h <= _MEDIUM[1] and _MEDIUM[0] <= w <= _MEDIUM[1]
        if salient and medium:
            return 0
        if medium:
            return 1
        if salient:
            return 2
        return 3

    # -- graph --------------------------------------------------------------

    def _ensure(self, nid: str, grid: np.ndarray, avail) -> _Node:
        if nid in self.nodes:
            return self.nodes[nid]
        node = _Node()
        for aid in avail:
            if aid == 0 or aid == 6:
                continue
            # A directional/simple action is always tier 0: movement control is the
            # cheapest thing to establish and the reference puts arrows first.
            tier = 3 if (self.use_knowledge and aid in self.k.dead_simple) else 0
            node.edges[("S", aid)] = {
                "result": 0, "target": None, "tier": tier,
                "w": self.k.simple_weight(aid) if self.use_knowledge else 1.0,
            }
        if 6 in avail:
            for colour, y, x, h, w in self._components(grid):
                node.edges[("C", colour, y, x)] = {
                    "result": 0, "target": None, "tier": self._tier(colour, h, w),
                    "w": self.k.colour_weight(colour) if self.use_knowledge else 1.0,
                }
        node.order = list(node.edges)
        self.rng.shuffle(node.order)
        self.nodes[nid] = node
        self._dist_dirty = True
        return node

    def _open_edges(self, nid: str):
        node = self.nodes.get(nid)
        if node is None:
            return []
        return [k for k in node.order
                if node.edges[k]["result"] == 0 and node.edges[k]["tier"] <= self.active_tier]

    def _rebuild_distances(self) -> None:
        """Reverse BFS seeded simultaneously from every frontier node."""
        frontier = [n for n in self.nodes if self._open_edges(n)]
        self.dist = {n: 0 for n in frontier}
        self.next_hop = {}
        dq = deque(frontier)
        while dq:
            cur = dq.popleft()
            for prev in self.pred.get(cur, ()):
                if prev not in self.dist:
                    self.dist[prev] = self.dist[cur] + 1
                    node = self.nodes.get(prev)
                    if node:
                        for k, e in node.edges.items():
                            if e["result"] == 1 and e["target"] == cur:
                                self.next_hop[prev] = k
                                break
                    dq.append(prev)
        self._dist_dirty = False

    def _escalate(self) -> bool:
        if self.active_tier >= 4:
            return False
        self.active_tier += 1
        self._dist_dirty = True
        return True

    # -- policy -------------------------------------------------------------

    def choose(self, nid: str, grid: np.ndarray, avail):
        """Greedy-local, then walk to the nearest frontier, under a global tier gate."""
        self._ensure(nid, grid, avail)
        for _ in range(5):                      # at most 5 tier escalations
            open_here = self._open_edges(nid)
            if open_here:
                # Sample by learned effectiveness rather than uniformly. This is the
                # only place carried knowledge can change behaviour, so it is the
                # whole experiment: identical algorithm in both arms, differing only
                # in whether the weights survived the level boundary.
                edges = self.nodes[nid].edges
                weights = [max(1e-6, edges[k].get("w", 1.0)) for k in open_here]
                return self.rng.choices(open_here, weights=weights, k=1)[0], "probe"
            if self._dist_dirty:
                self._rebuild_distances()
            hop = self.next_hop.get(nid)
            if hop is not None and self.nodes[nid].edges[hop]["result"] == 1:
                return hop, "walk"
            if not self._escalate():
                return None, "exhausted"
        return None, "exhausted"

    def record(self, nid: str, key, new_nid: str, changed: bool, grid_after: np.ndarray) -> None:
        node = self.nodes[nid]
        edge = node.edges.get(key)
        if edge is None:
            return
        if not changed:
            edge["result"] = -1                  # dead at this node, permanently
        else:
            # A transition back to the level's opening frame is the signature of a
            # hidden reset. Do not commit it until confirmed; the reference lost its
            # official run to exactly this edge becoming a perpetual frontier target.
            if self.level_first_hash is not None and new_nid == self.level_first_hash:
                c = self.suspicious.get((nid, key), 0) + 1
                self.suspicious[(nid, key)] = c
                if c < _SUSPICIOUS_THRESHOLD:
                    return
            edge["result"] = 1
            edge["target"] = new_nid
            self.succ.setdefault(nid, set()).add(new_nid)
            self.pred.setdefault(new_nid, set()).add(nid)
        self._dist_dirty = True

    # -- knowledge updates --------------------------------------------------

    def learn(self, key, changed: bool) -> None:
        if key[0] == "S":
            self.k.note_simple(key[1], changed)
        else:
            self.k.note_click(key[1], changed)


def learn_hud_mask(env_step, obs, probes: int = 6):
    """Pixels that change on EVERY probing action are counter/HUD, not board.

    Empirical rather than the reference's edge-geometry rule, which its own authors
    flag as breaking when the counter is drawn into the scene rather than at an edge --
    and the scored set is 55 unseen games.
    """
    before = frame_np(obs)
    always = np.ones(before.shape, dtype=bool)
    cur = obs
    used = 0
    seen_change = False
    for i in range(probes):
        avail = [a for a in (cur.available_actions or []) if a not in (0, 6)]
        if not avail:
            break
        prev = frame_np(cur)
        # Cycle through DIFFERENT actions: probing one blocked action repeatedly
        # yields no change at all and ANDs the mask to empty on every game.
        cur = env_step(avail[i % len(avail)])
        used += 1
        delta = frame_np(cur) != prev
        if delta.any():
            seen_change = True
            always &= delta
    if not seen_change:
        return np.zeros(before.shape, dtype=bool), cur, used
    # Guard: a mask swallowing a fifth of the board is not a counter, it is the game.
    if always.sum() > 0.2 * always.size:
        always = np.zeros_like(always)
    return always, cur, used
