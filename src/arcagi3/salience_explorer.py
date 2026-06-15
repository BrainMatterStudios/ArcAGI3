"""SalienceExplorer — our own reimplementation of the published hierarchical salience-tiered
graph-exploration algorithm (arXiv 2512.24156 "just-explore", Algorithm 1).

This is a clean-room reimplementation of the *algorithm as described in the paper* (NOT a
copy of their code): a pure object-graph explorer with no motion model, which is why it
avoids our hybrid's sokoban-fragility. Reactive one-action-per-call interface (same as
HybridPolicy) so it drops into run_reactive / the submission adapter. The banked reactive
agent is untouched; this is a separate policy measured head-to-head.

Algorithm 1 (hierarchical action selection), per state node, at salience threshold p:
  1) if a known action here produced reward, take it (exploit);
  2) else if this node has an untested action with tier <= p, take one UNIFORMLY AT RANDOM
     among the lowest such tier;
  3) else move along the shortest known path to the nearest reachable node that still has
     an untested action with tier <= p;
  4) else raise p and recurse; if p exhausted, RESET to root (bounded), then stop.
State id = object-structure hash with frequently-changing (status-bar/counter) cells masked.
Actions are salience-tiered: simple actions tier 0; clicks by object salience (0..9).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P

SIMPLE_IDS = [1, 2, 3, 4, 5]
MAX_TIER = 9


class _Node:
    __slots__ = ("key", "cands", "tier", "edges", "terminal", "visits")

    def __init__(self, key, cands_with_tiers, terminal=False):
        self.key = key
        self.cands = tuple(a for a, _t in cands_with_tiers)
        self.tier = {a: t for a, t in cands_with_tiers}
        self.edges: dict = {}  # action -> (next_key, reward)
        self.terminal = terminal
        self.visits = 0

    def untried_le(self, p):
        return [a for a in self.cands if a not in self.edges and self.tier.get(a, 0) <= p]

    def has_untried_le(self, p):
        return not self.terminal and any(
            a not in self.edges and self.tier.get(a, 0) <= p for a in self.cands)

    def reward_action(self):
        best, br = None, 0.0
        for a, (_nk, r) in self.edges.items():
            if r > br:
                best, br = a, r
        return best


class SalienceExplorer:
    def __init__(self, max_click_targets: int = 96, seed: int = 0,
                 max_stuck_resets: int = 200, trust_threshold: int = 3,
                 border_mask: int = 2) -> None:
        self.max_click_targets = max_click_targets
        self.rng = np.random.default_rng(seed)
        self.max_stuck_resets = max_stuck_resets
        # border_mask > 0 enables the dynamic-border (HUD/progress-bar) mask: cells within
        # this many rows/cols of the grid edge that have EVER changed are dropped from the
        # state key. Monotonic bottom-edge progress bars (re86/wa30) change each cell only
        # once, so the cell-frequency VolatilityTracker never catches them -> every state is
        # forever-unique -> graph explodes (re86 1.1 act/state). Masking the dynamic edge band
        # restores state revisits without touching the interior play area. 0 == off. Default 2
        # catches 2-wide edge bars (sc25 right-edge cols 62-63, sk48) that band=1 half-masks;
        # band=2 measured 33 levels @30k (strict superset of band=1's 32, +sk48, no regressions).
        self.border_mask = max(0, int(border_mask))
        # trust_threshold > 1 enables suspicious-transition filtering: a NEW transition that
        # conflicts with an already-recorded edge (the signature of animation/frame noise on
        # real games) must repeat this many times before it overwrites the trusted edge. The
        # first observation of any edge, and any reward-bearing transition, is trusted at once
        # (so deterministic games are not slowed). trust_threshold == 1 == original behaviour.
        self.trust_threshold = max(1, int(trust_threshold))
        self.reset_all()

    # expose .gs.wm-like length for the runner's states_seen (duck-typing)
    @property
    def gs(self):
        return self

    @property
    def wm(self):
        return self.nodes

    def reset_all(self):
        self.vt = P.VolatilityTracker()
        self.nodes: dict[bytes, _Node] = {}
        self.bg: int | None = None
        self.root_key: bytes | None = None
        self.active_group = 0
        self.plan: list = []
        self._expect: bytes | None = None
        self.prev_key: bytes | None = None
        self.prev_action = None
        self.prev_levels = 0
        self.expect_reset = False
        self.stuck_resets = 0
        self.pending: dict = {}  # (key, action) -> (candidate_next_key, count) for suspicion filter

    def _candidates(self, grid, available):
        cands = []
        for aid in SIMPLE_IDS:
            if aid in available:
                cands.append((("S", aid), 0))
        if 6 in available:
            for x, y, prio in P.salient_click_targets(grid, max_targets=self.max_click_targets,
                                                      coarse_grid_step=8):
                cands.append((("C", int(x), int(y)), int(prio)))
        return cands

    def _key(self, grid):
        m = self.vt.mask()
        bm = self._border_mask()
        if bm is not None:
            m = m | bm
        if m.any():
            grid = grid.copy()
            grid[m] = self.bg if self.bg is not None else 0
        return P.object_state_key(grid, background=self.bg)

    def _border_mask(self):
        """Edge cells (within border_mask of the grid edge) that have ever changed -> HUD."""
        b = self.border_mask
        if b <= 0 or self.vt.changes is None or self.vt.steps < self.vt.min_steps:
            return None
        edge = np.zeros(self.vt.shape, dtype=bool)
        edge[:b] = edge[-b:] = True
        edge[:, :b] = edge[:, -b:] = True
        return edge & (self.vt.changes > 0)

    def _observe(self, key, cands, terminal=False):
        n = self.nodes.get(key)
        if n is None:
            n = _Node(key, cands, terminal)
            self.nodes[key] = n
        else:
            n.terminal = n.terminal or terminal
        n.visits += 1
        return n

    def _path_to_frontier(self, start, p):
        if start not in self.nodes:
            return None
        if self.nodes[start].has_untried_le(p):
            return []
        seen = {start}
        q = deque([(start, [])])
        while q:
            k, path = q.popleft()
            node = self.nodes.get(k)
            if not node:
                continue
            for a, (nk, _r) in node.edges.items():
                if nk in seen:
                    continue
                seen.add(nk)
                np_ = path + [a]
                nn = self.nodes.get(nk)
                if nn is not None and nn.has_untried_le(p):
                    return np_
                q.append((nk, np_))
        return None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        self.vt.update(grid)
        if self.bg is None:
            self.bg = P.detect_background(grid)
        cur = self._key(grid)

        if gstate_terminal or gstate_notplayed:
            if gstate_terminal and self.prev_action is not None and self.prev_key is not None:
                self._record(self.prev_key, self.prev_action, cur, 0.0,
                             self._candidates(grid, available), terminal=True)
            self.prev_action = None
            self.expect_reset = True
            self.plan = []
            return ("reset",)

        if self.root_key is None:
            self.root_key = cur
            self._observe(cur, self._candidates(grid, available))

        if self.expect_reset:
            self.expect_reset = False
            self.prev_action = None

        if self.prev_action is not None and self.prev_key is not None:
            reward = float(levels - self.prev_levels)
            self._record(self.prev_key, self.prev_action, cur, reward,
                         self._candidates(grid, available), terminal=False)
            if reward > 0:
                self.active_group = 0  # re-prioritise high salience after a level-up

        self.prev_levels = levels
        action = self._choose(cur)
        self.prev_key = cur
        self.prev_action = None if action[0] == "reset" else action
        return action

    def _record(self, key, action, next_key, reward, cands, terminal):
        node = self.nodes.get(key) or self._observe(key, cands)
        existing = node.edges.get(action)
        if (self.trust_threshold <= 1 or reward > 0 or existing is None
                or existing[0] == next_key):
            # trust at once: filtering off, reward-bearing, first observation, or consistent
            node.edges[action] = (next_key, reward)
            self.pending.pop((key, action), None)
        else:
            # conflict with a trusted edge -> require the new target to repeat before overwriting
            pk = (key, action)
            cand, cnt = self.pending.get(pk, (next_key, 0))
            cand, cnt = (next_key, cnt + 1) if cand == next_key else (next_key, 1)
            if cnt >= self.trust_threshold:
                node.edges[action] = (next_key, reward)
                self.pending.pop(pk, None)
            else:
                self.pending[pk] = (cand, cnt)
        self._observe(next_key, cands, terminal=terminal)
        self._expect = next_key if self.plan else None

    def _choose(self, cur):
        node = self.nodes.get(cur)
        if node is None:
            return self._random(cur)
        # 1) exploit reward
        ra = node.reward_action()
        if ra is not None:
            return ra
        # 2) active plan replay
        if self.plan:
            if self._expect is not None and cur != self._expect:
                self.plan = []
            else:
                return self.plan.pop(0)
        # 3) hierarchical tier exploration
        g = self.active_group
        while g <= MAX_TIER:
            local = node.untried_le(g)
            if local:
                self.active_group = g
                mp = min(node.tier.get(a, 0) for a in local)
                choices = [a for a in local if node.tier.get(a, 0) == mp]
                return choices[int(self.rng.integers(0, len(choices)))]
            path = self._path_to_frontier(cur, g)
            if path:
                self.active_group = g
                self.plan = path
                return self.plan.pop(0)
            g += 1
        # 4) exhausted from here -> bounce off root
        if self.root_key is not None and cur != self.root_key and self.stuck_resets < self.max_stuck_resets:
            self.stuck_resets += 1
            self.expect_reset = True
            self.plan = []
            return ("reset",)
        return self._random(cur)

    def _random(self, cur):
        node = self.nodes.get(cur)
        cands = node.cands if node else (("S", 1),)
        return cands[int(self.rng.integers(0, len(cands)))]

    def __len__(self):
        return len(self.nodes)
