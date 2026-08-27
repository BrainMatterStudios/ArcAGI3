"""SearchCore — stages 1-4 of the measured search-suite plan
(docs/RESEARCH-2026-08-23-searchcore-and-multirole.md §A).

Stage 3 adds:
  run-length macros   repeat a basic action while the masked frame keeps
                      changing (cap 12), EMITTING every intermediate state
                      as a real node (the probe's mnbfs swallowed them and
                      falsely exhausted ls20-L2 — measured completeness bug).
  ignition probes     when NO single action changes the root frame (sc25
                      class), probe click+click and click+move pairs (~b^2,
                      only at inert roots) and seed nbfs from any pair that
                      ignites the frame.

Stage 4 adds:
  go-explore          archive scheduler: cells selected with weight
                      1/sqrt(visits+1), return-then-explore momentum
                      rollouts (0.92 action-repeat), shorter-trajectory
                      replacement; exact masked-hash primary cell with a
                      coarse component-multiset tier auto-switched past
                      ~10k cells.
  portfolio racer     successive halving across {nbfs, nbfs+macros,
                      go-explore} on novel-states-per-100-actions, with
                      frame-0 archetype opening dispatch (playbook rule:
                      {6}-only=CLICK / {1,2,3,4}(+5)=AVATAR / else MIXED).
  (T3 pitch-lattice   already present in the stage-2 generator — verified,
                      nothing added.)

Two API-compatible backends over one algorithm layer:

  SNAPSHOT      copy.deepcopy(env) expansion on the offline engine
                (verified deterministic, ~0.8 ms/copy, ~11.6k act/s).
  RESET-REPLAY  the live-lane cost model: testing an action at depth d
                costs 1 reset + d replay actions + 1 action.  Rollouts are
                amortized: ONE reset per rollout, momentum action-repeat.

Algorithm layer (backend-agnostic):

  nbfs          novelty-preferred-but-COMPLETE best-first: states that make a
                new (y, x, color) atom true go to the fast queue; non-novel
                states are DEFERRED, never dropped (measured 29 levels at
                120 s/game vs BFS 26, IW(1)-strict 8).

Click generation is 4-tiered with learned pruning (tiers cumulative,
escalated only on true frontier exhaustion):

  T1  component centroids, UNCAPPED, snap-to-own-color (the probe's 16-cap
      alone cost dc22 three levels).
  T2  stride-4 interior lattice of LARGE components (cn04-class: clicks land
      inside big regions and position matters).
  T3  autocorrelation pitch-lattice cell centers (su15's real 16x14 grid at
      4 px pitch; r11l).
  T4  every cell that has EVER differed from the reference frame in any
      observed frame; cells that never changed in any frame are pruned
      (per-cell never-changed memory), subsampled, hard-capped.

plus a DeadClickMemory: a click position with zero observed effect across
>= K distinct states is pruned from every later candidate list.

State identity: the static per-game HUD table of the probe is REPLACED by
the learned VolatilityMask (imported from the shipped graft, not forked),
warmed up on throwaway transitions and FROZEN before any search uses its
cells for signatures (doc §A law: a mask that keeps learning mid-search
makes signatures drift and corrupts the dedup).

Transposition keys use the FrontierGraph (level, h, w, crc32) format —
`crc_key` is unit-tested byte-identical to FrontierGraph.node_key — while
the snapshot hot loop dedups on the full masked bytes (zero collision risk;
crc32 alone gives ~5% birthday collision odds at 20k states).

REPAIRS 2026-08-25 (full2700 falsifier regressions vs the probes):
  1. dead-click pruning scoped to bulk tiers 3/4 only — k=3 on component
     targets collapsed sb26 L1 to a 320-state exhaustion (probe: depth 9/10).
  2. tier3 = native 4-px raster fallback when pitch autocorrelation fails
     (it measurably NEVER fires on real lattice games); estimator now
     requires a genuine peak instead of smallest-p>=0.90.
  3. warmup mask validation: single-action twin fan-out unmasks
     action-dependent (reactive) cells — su15's only click feedback lives in
     border row 63 and the slow-tick rule had masked it (33-state collapse).
  4. solve_level escalates click tiers on state_cap, not just exhaustion
     (r11l's tier-2 cap-blow starved the tier-3 lattice that solves it).
  5. portfolio: per-level lane ROTATION with growing slices + move-to-front
     replaces winner-commitment; lane timeout is no longer terminal (ls20:
     goexplore burned 2557 s on L2 while nbfs — 3 probe levels — never ran).
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import sys
import time
import zlib
from collections import deque
from typing import Any, Callable, Iterable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPLORER_DIR = os.path.join(os.path.dirname(_HERE), "_explorer_floor")
if _EXPLORER_DIR not in sys.path:
    sys.path.insert(0, _EXPLORER_DIR)

from graft_explorer import (  # noqa: E402  (shipped machinery, reused not forked)
    FrontierGraph,
    VolatilityMask,
    _background_color,
    _components,
)

import specialists  # noqa: E402  (stage 6: mechanic-class specialist tier)

ROOT = os.path.dirname(os.path.dirname(_HERE))


# --------------------------------------------------------------------------
# frame helpers
# --------------------------------------------------------------------------

def settled(obs) -> np.ndarray:
    """Last layer of the frame stack (animation runs to 61 layers on g50t)."""
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


def lv(obs) -> int:
    return int(obs.levels_completed or 0)


def mask_to_bool(mask_cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for y, x in mask_cells or []:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            m[y, x] = True
    return m


def masked_grid(grid: np.ndarray, mask_bool: np.ndarray | None) -> np.ndarray:
    if mask_bool is None or not mask_bool.any():
        return grid
    return np.where(mask_bool, 0, grid)


def crc_key(level: int, grid: np.ndarray, mask_bool: np.ndarray | None) -> tuple:
    """FrontierGraph.node_key-compatible (level, h, w, crc32), numpy-fast.

    Byte-identical to FrontierGraph.node_key (unit-tested): masked cells
    flattened to 0, cells serialized row-major as (value & 0xFF).
    """
    g = masked_grid(grid, mask_bool)
    b = np.ascontiguousarray(g, dtype=np.uint8).tobytes()
    h, w = g.shape
    return (int(level), h, w, zlib.crc32(b))


def masked_bytes(level: int, grid: np.ndarray, mask_bool: np.ndarray | None) -> tuple:
    """Dedup identity for the seen-set: 128-bit blake2b of the masked bytes
    (collision odds ~1e-29 at 100k states; 32x lighter than raw bytes)."""
    g = masked_grid(grid, mask_bool)
    b = np.ascontiguousarray(g, dtype=np.uint8).tobytes()
    return (int(level), hashlib.blake2b(b, digest_size=16).digest())


# --------------------------------------------------------------------------
# 4-tier click generator + learned pruning
# --------------------------------------------------------------------------

class DeadClickMemory:
    """A click position with zero observed effect across >= K distinct source
    states (and no effect EVER) is dead for the rest of the game.

    STATIC-CELL VETO (measured 2026-08-23): pruning on no-op counts alone
    false-kills stateful targets — vc33 L4's winning click no-ops in early
    states and only fires later; K=3 alone cost the level (3 vs the probe's
    4). When an ever-changed map is supplied, only positions whose cell has
    NEVER changed in any observed frame may be declared dead."""

    def __init__(self, k: int = 3):
        self.k = int(k)
        self.stats: dict[tuple[int, int], dict[str, Any]] = {}

    def record(self, xy: tuple[int, int], state_key: Any, changed: bool) -> None:
        st = self.stats.setdefault(xy, {"effect": 0, "noop": set()})
        if changed:
            st["effect"] += 1
        else:
            st["noop"].add(state_key)

    def is_dead(self, xy: tuple[int, int],
                ever_changed: np.ndarray | None = None) -> bool:
        st = self.stats.get(xy)
        if not (st and st["effect"] == 0 and len(st["noop"]) >= self.k):
            return False
        if ever_changed is not None:
            x, y = xy
            if 0 <= y < ever_changed.shape[0] and 0 <= x < ever_changed.shape[1] \
                    and ever_changed[y, x]:
                return False   # cell carries dynamic content: never prune
        return True

    def prune(self, targets: list[tuple[int, int]],
              ever_changed: np.ndarray | None = None) -> list[tuple[int, int]]:
        return [t for t in targets if not self.is_dead(t, ever_changed)]


class ChangeMemory:
    """Per-cell EVER-changed-in-any-frame memory (game-scoped).

    A cell whose value has matched the reference frame in every observed
    frame has never carried dynamic content; T4 prunes it."""

    def __init__(self):
        self.ref: np.ndarray | None = None
        self.ever_changed: np.ndarray | None = None

    def observe(self, grid: np.ndarray) -> None:
        if self.ref is None or self.ref.shape != grid.shape:
            self.ref = grid.copy()
            self.ever_changed = np.zeros(grid.shape, dtype=bool)
            return
        self.ever_changed |= grid != self.ref


def _estimate_pitch(g: np.ndarray, axis: int, p_min: int = 3, p_max: int = 16,
                    threshold: float = 0.90, peak_margin: float = 0.03
                    ) -> tuple[int, int] | None:
    """(pitch, phase) along `axis`, or None.

    REPAIRED 2026-08-25 (full2700 falsifier regression): the original rule
    (smallest p whose shifted self-agreement clears 0.90) can NEVER fire on
    the real lattice games — measured agreement profiles on su15/r11l/vc33
    roots are monotone-decreasing in p (background dominates; no peak at the
    true 4-px pitch), so it returned None on one axis (su15 y) or a spurious
    p=3 (vc33: 436 junk targets). Now the argmax-agreement p must clear the
    threshold AND stand `peak_margin` above BOTH neighbors — a genuine peak.
    Real lattice games fail this (correctly) and get the tier3 raster
    fallback instead; synthetic gridline frames still resolve exactly."""
    n = g.shape[axis]
    ps = list(range(p_min, min(p_max, n // 2) + 1))
    if not ps:
        return None

    def score(p: int) -> float:
        if axis == 0:
            return float(np.mean(g[p:, :] == g[:-p, :]))
        return float(np.mean(g[:, p:] == g[:, :-p]))

    scores = {p: score(p) for p in ps}
    best = min(ps, key=lambda p: (-scores[p], p))
    if scores[best] < threshold:
        return None
    for nb in (best - 1, best + 1):
        if nb in scores and scores[best] - scores[nb] < peak_margin:
            return None                       # no real peak: reject
    diffs = np.any(np.diff(g, axis=axis) != 0, axis=1 - axis)
    boundaries = np.nonzero(diffs)[0] + 1
    if len(boundaries) == 0:
        return None
    votes = np.bincount(boundaries % best, minlength=best)
    return best, int(np.argmax(votes))


class ClickGenerator:
    """Cumulative-tier (x, y) targets; every tier deduped against earlier
    output (manhattan <= 1) and filtered through the DeadClickMemory."""

    T1_MIN, T1_MAX = 2, 100          # probe's compact band, cap REMOVED
    T2_STRIDE = 4
    T4_STRIDE = 2
    T4_CAP = 512
    LATTICE_CAP = 400
    RASTER_PITCH = 4                 # native ARC-3 cell raster (64x64 @4px)

    def __init__(self, dead: DeadClickMemory | None = None):
        self.dead = dead or DeadClickMemory()

    @staticmethod
    def _dedup(cands: Iterable[tuple[int, int]], out: list[tuple[int, int]],
               radius: int = 1) -> None:
        for x, y in cands:
            if all(abs(x - u) + abs(y - v) > radius for u, v in out):
                out.append((x, y))

    def _comps(self, rows: list) -> tuple[int, list[dict]]:
        bg = _background_color(rows)
        return bg, [c for c in _components(rows) if c["color"] != bg]

    def tier1(self, rows: list) -> list[tuple[int, int]]:
        """Uncapped compact-component centroids, snapped to own color."""
        bg, comps = self._comps(rows)
        comps = [c for c in comps if self.T1_MIN <= c["size"] <= self.T1_MAX]
        comps.sort(key=lambda c: c["size"])
        out: list[tuple[int, int]] = []
        for c in comps:
            x, y = c["centroid"]
            if not (0 <= y < len(rows) and 0 <= x < len(rows[0])):
                continue
            if rows[y][x] != c["color"]:
                r0, c0, r1, c1 = c["bbox"]
                best = None
                for yy in range(r0, r1 + 1):
                    for xx in range(c0, c1 + 1):
                        if rows[yy][xx] == c["color"]:
                            d = abs(yy - y) + abs(xx - x)
                            if best is None or d < best[0]:
                                best = (d, xx, yy)
                if best is None:
                    continue
                x, y = best[1], best[2]
            self._dedup([(x, y)], out, radius=2)
        return out

    def tier2(self, rows: list, out: list[tuple[int, int]]) -> None:
        """Stride-4 interior of LARGE components (size > T1_MAX)."""
        bg, comps = self._comps(rows)
        for c in comps:
            if c["size"] <= self.T1_MAX:
                continue
            r0, c0, r1, c1 = c["bbox"]
            for yy in range(r0 + 1, r1, self.T2_STRIDE):
                for xx in range(c0 + 1, c1, self.T2_STRIDE):
                    if rows[yy][xx] == c["color"]:
                        self._dedup([(xx, yy)], out)

    def tier3(self, grid: np.ndarray, out: list[tuple[int, int]]) -> None:
        """Pitch-lattice cell centers.

        REPAIRED 2026-08-25: when autocorrelation fails on either axis (the
        measured case on EVERY real lattice game — su15/r11l profiles are
        monotone in p), fall back to the native 4-px cell raster: exactly
        the configuration the 2026-08-23 probe used to unlock su15 L1
        (depth 7, branching 225) and r11l L1 (depth 3, branching 256).
        Measured at the roots: 224/256 raster clicks change su15's frame,
        256/256 change r11l's; the old tier3 contributed 0 live targets."""
        h, w = grid.shape
        py = _estimate_pitch(grid, axis=0)
        px = _estimate_pitch(grid, axis=1)
        if py is not None and px is not None:
            (p_y, o_y), (p_x, o_x) = py, px
            ys = list(range(o_y + p_y // 2, h, p_y))
            xs = list(range(o_x + p_x // 2, w, p_x))
        else:
            p = self.RASTER_PITCH
            ys = list(range(p // 2, h, p))
            xs = list(range(p // 2, w, p))
        cands = [(x, y) for y in ys for x in xs][: self.LATTICE_CAP]
        self._dedup(cands, out)

    def tier4(self, grid: np.ndarray, change: ChangeMemory,
              out: list[tuple[int, int]]) -> None:
        """Ever-changed cells only (never-changed-any-frame cells pruned),
        stride-subsampled, capped."""
        if change.ever_changed is None or not change.ever_changed.any():
            return
        ys, xs = np.nonzero(change.ever_changed)
        cands = [(int(x), int(y)) for y, x in zip(ys, xs)
                 if y % self.T4_STRIDE == 0 and x % self.T4_STRIDE == 0]
        added = 0
        for cand in cands:
            if added >= self.T4_CAP:
                break
            before = len(out)
            self._dedup([cand], out)
            added += len(out) - before

    def targets(self, grid: np.ndarray, tier: int,
                change: ChangeMemory | None = None) -> list[tuple[int, int]]:
        """REPAIRED 2026-08-25: DeadClickMemory pruning is scoped to the BULK
        tiers (3/4 lattice-raster cells) only. Applied to T1/T2 component
        targets it false-kills stateful UI clicks that no-op until a
        prerequisite is met and whose own cell never changes (the
        ever-changed veto cannot protect them) — measured on sb26: 15/19
        centroid targets dead after one 60 s audition, collapsing L1 to a
        320-state exhaustion vs 7011+ states with pruning off (the probe,
        which never pruned, reached depth 9/10 of the full crack). The
        probe parity rule: component targets are never pruned; pruning
        exists to control the 256-512-target raster tiers, where every
        effective cell is protected by effect counts or the veto."""
        rows = grid.tolist()
        out = self.tier1(rows)
        if tier >= 2:
            self.tier2(rows, out)
        n_comp = len(out)
        if tier >= 3:
            self.tier3(grid, out)
        if tier >= 4 and change is not None:
            self.tier4(grid, change, out)
        ever = change.ever_changed if change is not None else None
        return out[:n_comp] + self.dead.prune(out[n_comp:], ever)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class Handle:
    """Backend-opaque state handle. Algorithms read obs/depth/path only."""

    __slots__ = ("obs", "depth", "path", "env")

    def __init__(self, obs, depth: int, path: list[tuple], env=None):
        self.obs = obs
        self.depth = depth
        self.path = path
        self.env = env


def _apply(env, tok: tuple):
    from arcengine import GameAction

    if tok[0] == "C":
        return env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
    return env.step(GameAction.from_id(int(tok[1])))


class SnapshotBackend:
    """copy.deepcopy(env) expansion (offline engine; deterministic, ~0.8 ms)."""

    name = "snapshot"

    def __init__(self, env):
        self.env = env
        self.actions_spent = 0   # engine steps (incl. resets), for reporting
        self.resets = 0
        self.copies = 0

    def root(self) -> Handle:
        obs = self.env.reset()
        self.resets += 1
        self.actions_spent += 1
        return Handle(obs, 0, [], env=self.env)

    def children(self, handle: Handle, tokens: list[tuple], consume: bool = False):
        """Yield (token, child Handle | None). With consume=True the LAST
        token reuses the parent env in place (probe's measured optimization;
        the parent handle must not be expanded again afterwards)."""
        n = len(tokens)
        for i, tok in enumerate(tokens):
            if consume and i == n - 1:
                child_env = handle.env
            else:
                child_env = copy.deepcopy(handle.env)
                self.copies += 1
            obs = _apply(child_env, tok)
            self.actions_spent += 1
            if obs is None:
                yield tok, None
                continue
            yield tok, Handle(obs, handle.depth + 1, handle.path + [tok], env=child_env)

    def rollout(self, handle: Handle, choose: Callable[[Any, tuple | None], tuple | None],
                k: int, momentum: float = 0.90, rng=None,
                stop: Callable[[Any], bool] | None = None,
                snap: Callable[[Any, int], bool] | None = None,
                ) -> tuple[list[tuple], Handle]:
        """One snapshot restore, k momentum-repeat steps. Returns (trajectory
        of (tok, obs, Handle|None), final Handle). `snap(obs, depth)` asks the
        algorithm whether this state is worth materializing (deepcopy);
        `stop(obs)` ends the rollout early (level-up / game-over)."""
        rng = rng or random.Random(0)
        env = copy.deepcopy(handle.env)
        self.copies += 1
        obs, depth, path = handle.obs, handle.depth, list(handle.path)
        traj: list[tuple] = []
        prev_tok = None
        for _ in range(k):
            tok = prev_tok if (prev_tok is not None and rng.random() < momentum) \
                else choose(obs, prev_tok)
            if tok is None:
                break
            prev_tok = tok
            obs = _apply(env, tok)
            self.actions_spent += 1
            depth += 1
            path.append(tok)
            h = None
            if obs is not None and snap is not None and snap(obs, depth):
                h = Handle(obs, depth, list(path), env=copy.deepcopy(env))
                self.copies += 1
            traj.append((tok, obs, h))
            if obs is None or (stop is not None and stop(obs)):
                break
        return traj, Handle(obs, depth, path, env=env)

    def adopt(self, handle: Handle) -> None:
        """Make this handle's state the backend's persistent state (used to
        chain to the next level after an unlock)."""
        self.env = handle.env


class ResetReplayBackend:
    """Live cost model: one real env; reaching a node at depth d costs
    1 reset + d replay actions; testing an action costs 1 more.
    Rollouts amortize: ONE reset per rollout, momentum action-repeat
    (~(1 + d/k) actions per action tried — NOTES.md live-lane model)."""

    name = "reset_replay"

    def __init__(self, env):
        self.env = env
        self.actions_spent = 0   # every engine call: resets AND steps
        self.resets = 0
        # MEASURED 2026-08-27 and deliberately NOT optimised. This backend
        # spends one reset per NODE (tu93: 3,739 resets for 3,727 nodes), so
        # ~94% of every action is replay rather than new information. A path
        # cache that skipped the walk when the env already stood on the
        # requested state was built and measured: IDENTICAL search results,
        # 9 actions saved out of 58,522 (0.015%) — the frontier order means the
        # env is essentially never where the next request wants it. It was
        # reverted because it bought nothing and made this class understate the
        # live cost it exists to model (test_reset_replay_cost_model pins the
        # 1 reset + d replay + 1 action contract). The overhead is structural:
        # star-expansion on one undo-less env costs k*(1 + d) per parent.
        # Beating it needs a different search SHAPE, or no search at all.

    def _reset(self):
        obs = self.env.reset()
        self.resets += 1
        self.actions_spent += 1
        return obs

    def _replay(self, path: list[tuple]):
        from arcengine import GameState

        obs = self._reset()
        if obs is None:
            return None
        for tok in path:
            obs = _apply(self.env, tok)
            self.actions_spent += 1
            if obs is None or obs.state == GameState.GAME_OVER:
                return None
        return obs

    def root(self) -> Handle:
        obs = self._reset()
        return Handle(obs, 0, [])

    def children(self, handle: Handle, tokens: list[tuple], consume: bool = False):
        for tok in tokens:
            obs = self._replay(handle.path)
            if obs is None:
                yield tok, None
                continue
            obs = _apply(self.env, tok)
            self.actions_spent += 1
            if obs is None:
                yield tok, None
                continue
            yield tok, Handle(obs, handle.depth + 1, handle.path + [tok])

    def rollout(self, handle: Handle, choose: Callable[[Any, tuple | None], tuple | None],
                k: int, momentum: float = 0.90, rng=None,
                stop: Callable[[Any], bool] | None = None,
                snap: Callable[[Any, int], bool] | None = None,
                ) -> tuple[list[tuple], Handle]:
        rng = rng or random.Random(0)
        obs = self._replay(handle.path)     # the ONE reset of this rollout
        if obs is None:
            return [], Handle(None, handle.depth, list(handle.path))
        depth, path = handle.depth, list(handle.path)
        traj: list[tuple] = []
        prev_tok = None
        for _ in range(k):
            tok = prev_tok if (prev_tok is not None and rng.random() < momentum) \
                else choose(obs, prev_tok)
            if tok is None:
                break
            prev_tok = tok
            obs = _apply(self.env, tok)
            self.actions_spent += 1
            depth += 1
            path.append(tok)
            h = None
            if obs is not None and snap is not None and snap(obs, depth):
                h = Handle(obs, depth, list(path))   # path IS the handle live
            traj.append((tok, obs, h))
            if obs is None or (stop is not None and stop(obs)):
                break
        return traj, Handle(obs, depth, path)

    def adopt(self, handle: Handle) -> None:
        # children() leaves the real env AT the yielded child's state; the
        # winning child is always the last executed, so the env is already
        # there. Future paths are relative to the new level's start.
        pass


# --------------------------------------------------------------------------
# SearchCore: mask learning + nbfs + tier escalation
# --------------------------------------------------------------------------

class SearchCore:
    """One game, one backend, one frozen mask, one learned click memory."""

    MACRO_CAP = 12            # run-length macro: max repeats of one action
    IGNITION_PAIR_CAP = 4000  # composite pairs probed at an inert root

    def __init__(self, env, backend: str = "snapshot", *,
                 warmup_rounds: int = 6, warmup_clicks: int = 3,
                 warmup_min_transitions: int = 40,
                 max_states: int = 20000, dead_click_k: int = 3,
                 use_macros: bool = False, max_depth: int | None = None):
        # max_depth: hard search-depth bound for the CURRENT level. 10 of the
        # 25 public games enforce a per-level move budget B
        # (`StepCounter` -> wa30.py:1252 `elif not current_steps: self.lose()`,
        # checked AFTER the win test, so a solve landing on the last move still
        # counts). Past B the level is GAME_OVER, so no node deeper than B can
        # ever be part of a scored solution. Under reset-replay those doomed
        # branches are the most expensive in the tree — reaching depth d costs
        # 1 reset + d replay actions — so the bound is a direct cost lever, not
        # just a node-count one. None = unbounded = byte-identical to the
        # pre-2026-08-27 core. See step_budgets.py.
        self.max_depth = max_depth
        self.backend = SnapshotBackend(env) if backend == "snapshot" \
            else ResetReplayBackend(env)
        self.mask = VolatilityMask()
        # None = warmup validation not run yet (fall back to the raw mask);
        # a list = the validated post-warmup cells (may be empty)
        self.active_mask_cells: list[tuple[int, int]] | None = None
        self._mask_np: np.ndarray | None = None
        self.dead = DeadClickMemory(k=dead_click_k)
        self.clicks = ClickGenerator(self.dead)
        self.change = ChangeMemory()
        self.max_states = int(max_states)
        self.warmup_rounds = int(warmup_rounds)
        self.warmup_clicks = int(warmup_clicks)
        self.warmup_min_transitions = int(warmup_min_transitions)
        self.use_macros = bool(use_macros)
        self.graph = FrontierGraph()   # crc32 transposition machinery (live seam)
        self.stats = {"deferred_expanded": 0, "noop_clicks_recorded": 0,
                      "macro_nodes": 0, "ignition_pairs": 0,
                      "ignition_seeds": 0}

    # ---- mask -------------------------------------------------------------

    def warmup_and_freeze(self) -> int:
        """Learn the VolatilityMask on throwaway transitions, then FREEZE it
        before any search uses its cells for signatures (doc §A law).

        Warmup mixes basic actions with repeated centroid clicks so that
        click-only games (vc33-class, 99.5% masked-noop) still produce the
        interior-unchanged transitions the slow-tick band rule needs."""
        from arcengine import GameState

        env = self.backend.env
        obs = env.reset()
        if obs is None:
            self.mask.freeze()
            return 0
        self.mask.update(settled(obs).tolist())
        transitions = 0
        # low-action games see too few transitions in a fixed round count for
        # the slow-tick band rule (2 interior-quiet events/row) — measured on
        # ls20: 18 transitions learned 25 of ~128 HUD cells, inflating L1 from
        # 96 to 1505 states. Extend rounds until a minimum transition count.
        rnd = 0
        while rnd < self.warmup_rounds or transitions < self.warmup_min_transitions:
            if rnd >= self.warmup_rounds * 8:   # safety: never warm up forever
                break
            avail = set(int(a) for a in (obs.available_actions or []))
            toks = [("S", a) for a in sorted(avail) if a not in (0, 6)]
            if 6 in avail:
                cents = self.clicks.tier1(settled(obs).tolist())
                lo = (rnd * self.warmup_clicks) % max(1, len(cents)) if cents else 0
                toks += [("C", x, y) for x, y in cents[lo:lo + self.warmup_clicks]]
            for tok in toks:
                nxt = _apply(env, tok)
                self.backend.actions_spent += 1
                if nxt is None:
                    continue
                obs = nxt
                self.mask.update(settled(obs).tolist())
                transitions += 1
                if obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                    self.backend.actions_spent += 1
                    if obs is None:
                        break
                    self.mask.update(settled(obs).tolist())
            if obs is None:
                break
            rnd += 1
        self.mask.freeze()
        cells = self.mask.mask_cells()
        if cells and obs is not None:
            cells = self._validate_mask_cells(cells)
        self.active_mask_cells = list(cells or [])
        self._mask_np = None
        if cells:
            g = settled(obs) if obs is not None else np.zeros((64, 64), dtype=np.int8)
            self._mask_np = mask_to_bool(cells, g.shape)
        return transitions

    def _validate_mask_cells(self, cells: list[tuple[int, int]]
                             ) -> list[tuple[int, int]]:
        """Drop ACTION-DEPENDENT cells from the learned mask.

        REPAIRED 2026-08-25 (full2700 falsifier regression): the slow-tick
        band rule masks any border row/col that changes while the interior
        stays quiet — but on su15 the bottom row IS the game's primary click
        feedback: 215/224 effective raster clicks change ONLY row 63, so the
        learned mask turned nearly every click into a masked no-op and L1
        collapsed to a 33-state exhaustion (the probe, which masked nothing
        on su15, solved L1 at depth 7). Discriminator: a true status band
        (timer / action counter) advances IDENTICALLY whichever action is
        taken from the same state; a reactive band's content depends on the
        action. One single-action fan-out from the root: any masked cell
        whose value differs across the twins is reactive game state and is
        unmasked. (r11l: the learned mask covered all four edge cols = 256
        cells while the real HUD is col 0 — only true-status cells survive.)"""
        from arcengine import GameState

        root = self.backend.root()
        if root.obs is None:
            return cells
        g0 = settled(root.obs)
        h, w = g0.shape
        avail = set(int(a) for a in (root.obs.available_actions or []))
        toks: list[tuple] = [("S", a) for a in sorted(avail) if a not in (0, 6)]
        if 6 in avail:
            clicks = self.clicks.tier1(g0.tolist())[:4]
            spread = [(w // 4, h // 4), (3 * w // 4, h // 4),
                      (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4),
                      (w // 2, h // 2)]
            # probe INSIDE the masked region too: a click whose effect lands
            # in the masked band (r11l paints cells under its learned
            # edge-col mask) can only prove those cells reactive if some
            # twin actually clicks there — one raster-center probe per 4-px
            # chunk of the mask, so every masked cell gets local evidence
            probes: list[tuple[int, int]] = []
            for y, x in cells:
                xy = (min(w - 1, x - x % 4 + 2), min(h - 1, y - y % 4 + 2))
                if xy not in probes:
                    probes.append(xy)
                if len(probes) >= 16:
                    break
            for xy in spread + probes:
                if xy not in clicks:
                    clicks.append(xy)
            toks += [("C", x, y) for x, y in clicks]
        toks = toks[:28]
        grids: list[np.ndarray] = []
        for _, child in self.backend.children(root, toks):
            if child is None or child.obs is None \
                    or child.obs.state == GameState.GAME_OVER:
                continue
            g = settled(child.obs)
            if g.shape == (h, w):
                grids.append(g)
        if len(grids) < 2:
            return cells
        stack = np.stack(grids)
        kept = [(y, x) for y, x in cells
                if not (0 <= y < h and 0 <= x < w
                        and np.unique(stack[:, y, x]).size > 1)]
        return kept

    def mask_bool(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self._mask_np is not None and self._mask_np.shape != shape:
            cells = self.active_mask_cells if self.active_mask_cells is not None \
                else self.mask.mask_cells()
            return mask_to_bool(cells, shape)
        return self._mask_np

    # ---- per-node action generation --------------------------------------

    def actions_for(self, obs, tier: int) -> list[tuple]:
        avail = set(int(a) for a in (obs.available_actions or []))
        if not avail:
            avail = {1, 2, 3, 4, 5, 6}
        toks = [("S", a) for a in sorted(avail) if a not in (0, 6)]
        if 6 in avail:
            g = masked_grid(settled(obs), self.mask_bool(settled(obs).shape))
            toks += [("C", x, y)
                     for x, y in self.clicks.targets(g, tier, self.change)]
        return toks

    # ---- nbfs -------------------------------------------------------------

    def _observe(self, grid_masked: np.ndarray) -> None:
        self.change.observe(grid_masked)

    def nbfs_level(self, target: int, budget_s: float, tier: int,
                   seeds: list[tuple[Handle, np.ndarray]] | None = None) -> dict:
        """Novelty-preferred complete best-first from the level start.

        With use_macros: every basic-action child that changed the frame is
        run-length-extended (repeat while the masked frame keeps changing,
        cap MACRO_CAP), and EVERY intermediate state is emitted as a node
        (the probe's mnbfs swallowed intermediates — ls20-L2 regression).

        `seeds` (from an ignition probe) are extra start handles enqueued
        beside the root.

        Returns dict(solved, handle?, depth?, states, nodes, wall, reason)
        with reason in {solved, exhausted, timeout, state_cap}."""
        t0 = time.time()
        root = self.backend.root()
        if root.obs is None:
            return dict(solved=False, states=0, nodes=0, wall=0.0,
                        reason="reset_failed")
        if lv(root.obs) >= target:
            return dict(solved=True, handle=root, depth=0, states=1, nodes=0,
                        wall=0.0, reason="solved")
        from arcengine import GameState

        g0 = settled(root.obs)
        mb = self.mask_bool(g0.shape)
        gm0 = masked_grid(g0, mb)
        self._observe(gm0)
        seen = {masked_bytes(lv(root.obs), g0, mb)}
        h, w = gm0.shape
        atoms = np.zeros((h, w, 16), dtype=bool)
        yy, xx = np.arange(h)[:, None], np.arange(w)[None, :]
        atoms[yy, xx, np.clip(gm0, 0, 15)] = True
        queue: deque[tuple[Handle, np.ndarray]] = deque([(root, gm0)])
        slow: deque[tuple[Handle, np.ndarray]] = deque()
        nodes = 0

        def emit(child: Handle, gm: np.ndarray) -> bool:
            """Dedup + novelty-route one discovered state. True if new."""
            k = masked_bytes(lv(child.obs), settled(child.obs), mb)
            if k in seen:
                return False
            seen.add(k)
            novel = ~atoms[yy, xx, np.clip(gm, 0, 15)]
            if novel.any():
                atoms[yy, xx, np.clip(gm, 0, 15)] = True
                queue.append((child, gm))
            else:
                slow.append((child, gm))   # DEFERRED, never dropped
            return True

        for sh, sgm in seeds or []:
            self._observe(sgm)
            emit(sh, sgm)
        had_clicks = False
        reason = "exhausted"
        while queue or slow:
            if time.time() - t0 > budget_s:
                reason = "timeout"
                break
            if len(seen) > self.max_states:
                reason = "state_cap"
                break
            if queue:
                handle, pgm = queue.popleft()
            else:
                handle, pgm = slow.popleft()
                self.stats["deferred_expanded"] += 1
            # Level move budget: a child of this node sits at depth+1, and any
            # node past the budget is already GAME_OVER, so expanding here can
            # only buy doomed branches (and, on reset-replay, their full replay
            # cost). Bound is inclusive: a solve landing ON the last move wins,
            # because the engine tests the win before the budget.
            if self.max_depth is not None and handle.depth >= self.max_depth:
                self.stats["depth_pruned"] = self.stats.get("depth_pruned", 0) + 1
                continue
            toks = self.actions_for(handle.obs, tier)
            if not had_clicks and any(t[0] == "C" for t in toks):
                had_clicks = True
            if not had_clicks and 6 in set(
                    int(a) for a in (handle.obs.available_actions or [])):
                had_clicks = True   # clicks available even if tier found no target
            if not toks:
                continue
            parent_crc = zlib.crc32(np.ascontiguousarray(pgm, dtype=np.uint8).tobytes())
            for tok, child in self.backend.children(handle, toks, consume=True):
                if time.time() - t0 > budget_s:
                    reason = "timeout"
                    break
                nodes += 1
                if child is None or child.obs is None:
                    continue
                if lv(child.obs) >= target:
                    return dict(solved=True, handle=child, depth=child.depth,
                                states=len(seen), nodes=nodes,
                                wall=round(time.time() - t0, 1), reason="solved")
                if child.obs.state == GameState.GAME_OVER:
                    continue
                g = settled(child.obs)
                gm = masked_grid(g, mb)
                changed = not np.array_equal(gm, pgm)
                if tok[0] == "C":
                    self.dead.record((tok[1], tok[2]), parent_crc, changed)
                    if not changed:
                        self.stats["noop_clicks_recorded"] += 1
                if not changed:
                    continue
                self._observe(gm)
                is_new = emit(child, gm)
                # ---- run-length macro: repeat while the frame keeps changing,
                # EMITTING every intermediate state as a node ----------------
                if is_new and self.use_macros and tok[0] == "S":
                    cur, cur_gm = child, gm
                    for _ in range(self.MACRO_CAP - 1):
                        if time.time() - t0 > budget_s:
                            break
                        if (self.max_depth is not None
                                and cur.depth >= self.max_depth):
                            break     # same budget bound as the main loop
                        (_, nxt), = self.backend.children(cur, [tok])
                        nodes += 1
                        if nxt is None or nxt.obs is None:
                            break
                        if lv(nxt.obs) >= target:
                            return dict(solved=True, handle=nxt,
                                        depth=nxt.depth, states=len(seen),
                                        nodes=nodes,
                                        wall=round(time.time() - t0, 1),
                                        reason="solved")
                        if nxt.obs.state == GameState.GAME_OVER:
                            break
                        gm_n = masked_grid(settled(nxt.obs), mb)
                        if np.array_equal(gm_n, cur_gm):
                            break               # frame stopped changing
                        self._observe(gm_n)
                        if not emit(nxt, gm_n):
                            break               # already known: its own
                        self.stats["macro_nodes"] += 1  # expansion covers it
                        cur, cur_gm = nxt, gm_n
            if reason == "timeout":
                break
        return dict(solved=False, states=len(seen), nodes=nodes,
                    wall=round(time.time() - t0, 1), reason=reason,
                    had_clicks=had_clicks)

    # ---- composite ignition probes (inert roots, sc25 class) --------------

    def ignition_probe(self, target: int, budget_s: float, tier: int) -> tuple:
        """Root is inert: no single action changes the masked frame. Probe
        click+click and click+move pairs from the root; return
        (seeds, solved_result_or_None). ~b^2 cost, run ONLY at inert roots."""
        from arcengine import GameState

        t0 = time.time()
        root = self.backend.root()
        if root.obs is None:
            return [], None
        g0 = settled(root.obs)
        mb = self.mask_bool(g0.shape)
        gm0 = masked_grid(g0, mb)
        toks = self.actions_for(root.obs, tier)
        clicks = [t for t in toks if t[0] == "C"]
        singles = [t for t in toks if t[0] == "S"]
        local_seen = {masked_bytes(lv(root.obs), g0, mb)}
        seeds: list[tuple[Handle, np.ndarray]] = []
        pairs = 0
        for c1 in clicks:
            if pairs >= self.IGNITION_PAIR_CAP or time.time() - t0 > budget_s:
                break
            (_, h1), = self.backend.children(root, [c1])
            if h1 is None or h1.obs is None \
                    or h1.obs.state == GameState.GAME_OVER:
                continue
            for t2 in clicks + singles:
                if pairs >= self.IGNITION_PAIR_CAP \
                        or time.time() - t0 > budget_s:
                    break
                pairs += 1
                self.stats["ignition_pairs"] += 1
                (_, h2), = self.backend.children(h1, [t2])
                if h2 is None or h2.obs is None:
                    continue
                if lv(h2.obs) >= target:
                    return [], dict(solved=True, handle=h2, depth=h2.depth,
                                    states=len(local_seen), nodes=pairs,
                                    wall=round(time.time() - t0, 1),
                                    reason="solved", ignition=True)
                if h2.obs.state == GameState.GAME_OVER:
                    continue
                g2 = settled(h2.obs)
                gm2 = masked_grid(g2, mb)
                if np.array_equal(gm2, gm0):
                    continue                    # pair did not ignite
                k2 = masked_bytes(lv(h2.obs), g2, mb)
                if k2 in local_seen:
                    continue
                local_seen.add(k2)
                seeds.append((h2, gm2))
        self.stats["ignition_seeds"] += len(seeds)
        return seeds, None

    def solve_level(self, target: int, budget_s: float, max_tier: int = 4) -> dict:
        """nbfs with click-tier escalation on frontier exhaustion OR state-cap,
        then composite ignition probes if the root turned out fully inert.

        REPAIRED 2026-08-25: state_cap now ESCALATES instead of terminating —
        r11l's tier-2 interior lattice blew the 20k cap at 322 s and the old
        break meant tier 3 (whose raster solves r11l L1 at depth 3 in ~100 s)
        never ran. A wider click set can reach the unlock SHALLOWER, before
        the cap binds; timeout remains terminal (no new information)."""
        t0 = time.time()
        tier = 1
        last = None
        while tier <= max_tier:
            remain = budget_s - (time.time() - t0)
            if remain <= 0:
                break
            res = self.nbfs_level(target, remain, tier)
            res["tier"] = tier
            last = res
            if res["solved"] or res["reason"] in ("timeout", "reset_failed"):
                break
            if not res.get("had_clicks"):
                break     # no ACTION6 anywhere: wider click tiers change nothing
            tier += 1     # exhausted or state-capped -> widen the click set
        if last is None:
            last = dict(solved=False, states=0, nodes=0, wall=0.0,
                        reason="no_budget", tier=tier)
        # inert root: NO single action ever changed the frame at any tier
        if (not last.get("solved") and last.get("reason") == "exhausted"
                and last.get("states", 0) <= 1 and last.get("had_clicks")):
            remain = budget_s - (time.time() - t0)
            if remain > 1:
                seeds, solved = self.ignition_probe(target, remain, max_tier)
                if solved is not None:
                    solved["tier"] = max_tier
                    last = solved
                elif seeds:
                    remain = budget_s - (time.time() - t0)
                    if remain > 0:
                        res = self.nbfs_level(target, remain, max_tier,
                                              seeds=seeds)
                        res["tier"] = max_tier
                        res["ignition"] = True
                        last = res
        last["wall"] = round(time.time() - t0, 1)
        return last


# --------------------------------------------------------------------------
# Go-Explore scheduler (stage 4a)
# --------------------------------------------------------------------------

def _size_bucket(size: int) -> int:
    return int(size).bit_length()          # log2 buckets: 1,2-3,4-7,8-15,...


def coarse_cell_key(level: int, gm: np.ndarray) -> tuple:
    """Coarse archive cell: multiset of (color, size-bucket, bbox//4) over
    the masked grid's non-background components. Groups near-identical
    states (sub-4px jitter, animation residue) once the exact-cell archive
    outgrows the coarse switch."""
    rows = gm.tolist()
    bg = _background_color(rows)
    comps = [(c["color"], _size_bucket(c["size"]),
              (c["bbox"][0] // 4, c["bbox"][1] // 4,
               c["bbox"][2] // 4, c["bbox"][3] // 4))
             for c in _components(rows) if c["color"] != bg]
    return (int(level), tuple(sorted(comps)))


class GoExplorer:
    """Archive scheduler over a SearchCore's backend/mask/click machinery.

    Cells selected with weight 1/sqrt(visits+1); return-then-explore
    momentum rollouts; SHORTER-trajectory replacement (a rediscovered cell
    keeps its visit count but adopts the shorter path); exact masked-hash
    primary cells with automatic switch to the coarse component-multiset
    tier past `coarse_switch` cells."""

    def __init__(self, core: "SearchCore", *, tier: int = 3,
                 k_rollout: int = 30, momentum: float = 0.92,
                 max_cells: int = 12000, coarse_switch: int = 10000,
                 basic_weight: int = 5, rng_seed: int = 0):
        self.core = core
        self.tier = int(tier)
        self.k_rollout = int(k_rollout)
        self.momentum = float(momentum)
        self.max_cells = int(max_cells)
        self.coarse_switch = int(coarse_switch)
        self.basic_weight = int(basic_weight)
        self.rng = random.Random(rng_seed)

    # archive entry: [handle, gm, visits, depth]
    @staticmethod
    def archive_insert(archive: dict, key, handle: Handle,
                       gm: np.ndarray) -> bool:
        """Insert or shorter-trajectory replace. Returns True if the cell is
        NEW. Replacement keeps the visit count (Go-Explore rule)."""
        hit = archive.get(key)
        if hit is None:
            archive[key] = [handle, gm, 0, handle.depth]
            return True
        if handle.depth < hit[3]:
            hit[0], hit[1], hit[3] = handle, gm, handle.depth
        return False

    def _key(self, level: int, g: np.ndarray, gm: np.ndarray,
             coarse: bool, mb) -> tuple:
        if coarse:
            return coarse_cell_key(level, gm)
        return masked_bytes(level, g, mb)

    def _coarsen(self, archive: dict) -> dict:
        """Rebuild the exact-cell archive under coarse keys: min depth wins,
        visit counts merge."""
        out: dict = {}
        for handle, gm, visits, depth in archive.values():
            ck = coarse_cell_key(lv(handle.obs), gm)
            hit = out.get(ck)
            if hit is None:
                out[ck] = [handle, gm, visits, depth]
            else:
                hit[2] += visits
                if depth < hit[3]:
                    hit[0], hit[1], hit[3] = handle, gm, depth
        return out

    def explore_level(self, target: int, budget_s: float) -> dict:
        from arcengine import GameState

        core = self.core
        t0 = time.time()
        root = core.backend.root()
        if root.obs is None:
            return dict(solved=False, states=0, nodes=0, wall=0.0,
                        reason="reset_failed", algo="goexplore")
        if lv(root.obs) >= target:
            return dict(solved=True, handle=root, depth=0, states=1, nodes=0,
                        wall=0.0, reason="solved", algo="goexplore")
        g0 = settled(root.obs)
        mb = core.mask_bool(g0.shape)
        gm0 = masked_grid(g0, mb)
        core._observe(gm0)
        coarse = False
        archive: dict = {}
        self.archive_insert(archive, self._key(lv(root.obs), g0, gm0,
                                               coarse, mb), root, gm0)
        steps = 0
        reason = "timeout"

        def choose(obs, prev):
            toks = core.actions_for(obs, self.tier)
            if not toks:
                return None
            weights = [self.basic_weight if t[0] == "S" else 1 for t in toks]
            return self.rng.choices(toks, weights=weights)[0]

        def stop(obs):
            return (lv(obs) >= target
                    or obs.state == GameState.GAME_OVER)

        def snap(obs, depth):
            if lv(obs) >= target:
                return True
            if obs.state == GameState.GAME_OVER:
                return False
            g = settled(obs)
            gm = masked_grid(g, mb)
            k = self._key(lv(obs), g, gm, coarse, mb)
            hit = archive.get(k)
            if hit is not None:
                return depth < hit[3]              # shorter-path replacement
            return len(archive) < self.max_cells   # bounded memory

        while True:
            if time.time() - t0 > budget_s:
                reason = "timeout"
                break
            if len(archive) == 1 and steps > 400:
                reason = "inert"       # nothing ever changed: hand back
                break
            if not coarse and len(archive) > self.coarse_switch:
                archive = self._coarsen(archive)
                coarse = True
            cells = list(archive.values())
            ws = [1.0 / math.sqrt(c[2] + 1) for c in cells]
            cell = self.rng.choices(cells, weights=ws)[0]
            cell[2] += 1
            traj, _ = core.backend.rollout(
                cell[0], choose, k=self.k_rollout, momentum=self.momentum,
                rng=self.rng, stop=stop, snap=snap)
            steps += len(traj)
            for tok, obs, h in traj:
                if obs is None or h is None:
                    continue
                if lv(obs) >= target:
                    return dict(solved=True, handle=h, depth=h.depth,
                                states=len(archive), nodes=steps,
                                wall=round(time.time() - t0, 1),
                                reason="solved", coarse=coarse,
                                algo="goexplore")
                if obs.state == GameState.GAME_OVER:
                    continue
                g = settled(obs)
                gm = masked_grid(g, mb)
                core._observe(gm)
                self.archive_insert(
                    archive, self._key(lv(obs), g, gm, coarse, mb), h, gm)
        return dict(solved=False, states=len(archive), nodes=steps,
                    wall=round(time.time() - t0, 1), reason=reason,
                    coarse=coarse, algo="goexplore")


# --------------------------------------------------------------------------
# portfolio racer (stage 4c): successive halving + frame-0 archetype dispatch
# --------------------------------------------------------------------------

def archetype_frame0(avail: Iterable[int]) -> str:
    """Playbook dispatch rule: {6}-only => CLICK; {1,2,3,4}(+5, no 6) =>
    AVATAR; else/unknown => MIXED."""
    a = {int(x) for x in (avail or []) if 1 <= int(x) <= 6}
    if a == {6}:
        return "CLICK"
    if a and 6 not in a and a <= {1, 2, 3, 4, 5}:
        return "AVATAR"
    return "MIXED"


# macros only matter where basic actions exist; CLICK games skip that lane
DISPATCH_ORDER = {
    "AVATAR": ["nbfs_macros", "nbfs", "goexplore"],
    "MIXED": ["nbfs_macros", "goexplore", "nbfs"],
    "CLICK": ["nbfs", "goexplore"],
}


def sh_promote(scores: dict[str, float], order: list[str],
               keep: int) -> list[str]:
    """Successive-halving promotion: keep the `keep` best by score, ties
    broken by dispatch order. Returned in dispatch order."""
    ranked = sorted(scores, key=lambda n: (-scores[n], order.index(n)))
    kept = set(ranked[:keep])
    return [n for n in order if n in kept]


def solve_with(core: SearchCore, algo: str, target: int, budget_s: float,
               max_tier: int, go: GoExplorer) -> dict:
    if algo == "goexplore":
        return go.explore_level(target, budget_s)
    core.use_macros = (algo == "nbfs_macros")
    res = core.solve_level(target, budget_s, max_tier=max_tier)
    res["algo"] = algo
    return res


def portfolio_race(core: SearchCore, go: GoExplorer, target: int,
                   order: list[str], t1: float, deadline: float,
                   max_tier: int) -> tuple[str, list[dict], dict | None]:
    """Successive halving on novel-states-per-100-actions. Any probe that
    SOLVES the level ends the race immediately (progress is banked).
    Returns (winner, race_log, solved_result_or_None)."""
    cands = list(order)
    t = t1
    log: list[dict] = []
    while len(cands) > 1:
        scores: dict[str, float] = {}
        for name in cands:
            remain = deadline - time.time()
            if remain <= 1:
                return cands[0], log, None
            a0 = core.backend.actions_spent
            res = solve_with(core, name, target, min(t, remain), max_tier, go)
            da = max(1, core.backend.actions_spent - a0)
            score = 100.0 * res.get("states", 0) / da
            scores[name] = score
            log.append(dict(round_t=round(t, 1), algo=name,
                            score=round(score, 3),
                            states=res.get("states"), actions=da,
                            reason=res.get("reason")))
            if res.get("solved"):
                return name, log, res
        cands = sh_promote(scores, order, max(1, math.ceil(len(cands) / 2)))
        t *= 2
    return cands[0], log, None


# --------------------------------------------------------------------------
# game driver (chained levels, probe-compatible semantics)
# --------------------------------------------------------------------------

def discover_games(environments_dir: str) -> dict[str, str]:
    games = {}
    for stem in sorted(os.listdir(environments_dir)):
        p = os.path.join(environments_dir, stem)
        if not os.path.isdir(p):
            continue
        ver = [v for v in os.listdir(p)
               if not v.startswith("_") and not v.startswith(".")]
        if ver:
            games[stem] = f"{stem}-{ver[0]}"
    return games


# deterministic lanes: identical re-runs of an exhausted search cannot help
DETERMINISTIC_LANES = {"nbfs", "nbfs_macros"}
SLICE_MIN_S = 300.0        # smallest per-lane time slice on a level


def run_game(stem: str, budget_s: float, *, backend: str = "snapshot",
             algo: str = "nbfs", environments_dir: str | None = None,
             max_states: int = 20000, max_tier: int = 4,
             warmup_rounds: int = 6, dead_click_k: int = 3,
             race_t1: float | None = None) -> dict:
    """Chained level-by-level solve of one game. ONLY_RESET_LEVELS must be
    'true' in the environment (run_falsifier sets it).

    algo: nbfs | nbfs_macros | goexplore | portfolio.

    Portfolio scheduling — REPAIRED 2026-08-25: the racer used to COMMIT to
    the level-1 winner, and a lane timeout was terminal for the level — on
    ls20 the race's lucky goexplore L1 solve locked goexplore in, which then
    burned the remaining 2557 s failing L2 while nbfs (3 ls20 levels in the
    probe) never ran. Now every level is scheduled by LANE ROTATION with
    growing time slices: each lane in the ranking gets min(slice, remain);
    on failure (timeout included) the next lane gets its own slice; when all
    lanes have failed the slice doubles and rotation repeats. A lane that
    solves a level moves to the front of the ranking (move-to-front). A
    DETERMINISTIC lane (nbfs/nbfs_macros) that truly exhausted a level is
    closed for that level — identical re-runs cannot help — and when every
    lane is closed the game is over. The race now only sets the OPENING
    ranking (and banks a level solved mid-race)."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameState

    environments_dir = environments_dir or os.path.join(ROOT, "environment_files")
    games = discover_games(environments_dir)
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=environments_dir)
    env = client.make(games[stem])
    obs0 = env.reset()
    avail0 = list(obs0.available_actions or []) if obs0 is not None else []
    archetype = archetype_frame0(avail0)
    core = SearchCore(env, backend=backend, max_states=max_states,
                      warmup_rounds=warmup_rounds, dead_click_k=dead_click_k)
    go = GoExplorer(core)
    t_all = time.time()
    deadline = t_all + budget_s
    warm_transitions = core.warmup_and_freeze()
    levels: list[dict] = []
    race_log: list[dict] = []
    won = 0
    cracked = False

    def bank(res: dict) -> bool:
        """Record a solved level, adopt its state. True if the GAME is won."""
        nonlocal won, cracked
        entry = {k: v for k, v in res.items() if k != "handle"}
        entry["level"] = won + 1
        levels.append(entry)
        won += 1
        handle = res["handle"]
        core.backend.adopt(handle)
        if handle.obs is not None and handle.obs.state == GameState.WIN:
            cracked = True
            levels.append(dict(game_won=True))
            return True
        return False

    # stage 6: frame-only specialist detection (bounded snapshot probes).
    # On a match the specialist lane OPENS the ranking (and the race is
    # skipped — the specialist either solves the level outright or fails
    # once and closes, after which lane rotation runs the generic lanes).
    specialist = None
    if algo == "portfolio":
        try:
            specialist = specialists.detect(core)
        except Exception:  # noqa: BLE001 — fail-open by contract
            specialist = None

    if algo == "portfolio":
        order = DISPATCH_ORDER[archetype]
        if specialist is not None:
            ranking = ["specialist"] + list(order)
        else:
            t1 = race_t1 if race_t1 is not None else min(60.0, budget_s / 20)
            winner, race_log, solved = portfolio_race(
                core, go, 1, order, t1, deadline, max_tier)
            ranking = [winner] + [n for n in order if n != winner]
            if solved is not None and bank(solved):
                pass   # full game cracked during the race
    else:
        ranking = [algo]

    slice0 = max(SLICE_MIN_S, budget_s / 6.0)
    while not cracked and time.time() < deadline - 2:
        target = won + 1
        solved_res = None
        # lane -> ever_changed count at its last true exhaustion; the lane
        # stays closed while the change memory has not grown since (an
        # identical deterministic re-run cannot help; new ever-changed cells
        # reopen it because T4 then offers new targets)
        closed: dict[str, int] = {}

        def _ever_count() -> int:
            ec = core.change.ever_changed
            return int(ec.sum()) if ec is not None else 0

        slice_s = slice0
        while solved_res is None and time.time() < deadline - 2:
            open_lanes = [n for n in ranking
                          if n not in closed or _ever_count() > closed[n]]
            if not open_lanes:
                break                  # every lane closed: level unreachable
            for name in open_lanes:
                remain = deadline - time.time()
                if remain <= 2:
                    break
                # the front (last-successful) lane gets a DOUBLE slice: a
                # deterministic lane redoes all prior work after a timeout,
                # so near-miss thrash (vc33 L5 = 616 s fresh) is costlier
                # than a generous first slice
                lane_slice = slice_s * 2 if name == ranking[0] else slice_s
                if name == "specialist":
                    a0 = core.backend.actions_spent
                    res = specialists.solve_level(
                        core, specialist, target, min(lane_slice, remain))
                    # engine steps the specialist actually spent (probes +
                    # verification chains) — honest live-cost accounting
                    res["nodes"] = core.backend.actions_spent - a0
                else:
                    res = solve_with(core, name, target,
                                     min(lane_slice, remain), max_tier, go)
                entry = {k: v for k, v in res.items() if k != "handle"}
                entry["level"] = target
                entry["algo"] = res.get("algo", name)
                entry["slice_s"] = round(min(lane_slice, remain), 1)
                if res.get("solved"):
                    solved_res = res
                    ranking.remove(name)
                    ranking.insert(0, name)   # move-to-front
                    break
                levels.append(entry)
                if name == "specialist":
                    # deterministic one-shot: a failed specialist attempt
                    # closes the lane for this level permanently
                    closed[name] = float("inf")
                elif name in DETERMINISTIC_LANES \
                        and res.get("reason") == "exhausted":
                    closed[name] = _ever_count()
                elif name in closed:
                    del closed[name]   # lane ran again: clear stale closure
            slice_s *= 2
        if solved_res is None:
            break
        if bank(solved_res):
            break
    return dict(game=stem, algo=algo, backend=backend,
                archetype=archetype, specialist=specialist,
                winner=(ranking[0] if ranking else algo),
                levels_won=won, cracked=cracked, budget_s=budget_s,
                wall=round(time.time() - t_all, 1),
                warmup_transitions=warm_transitions,
                mask_cells=len(core.active_mask_cells or []),
                mask_cells_learned=len(core.mask.mask_cells()),
                actions_spent=core.backend.actions_spent,
                deferred_expanded=core.stats["deferred_expanded"],
                macro_nodes=core.stats["macro_nodes"],
                ignition_pairs=core.stats["ignition_pairs"],
                ignition_seeds=core.stats["ignition_seeds"],
                race=race_log, levels=levels)
