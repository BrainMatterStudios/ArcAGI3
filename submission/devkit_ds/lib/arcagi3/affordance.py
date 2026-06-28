"""C3 — Affordance model (observe-only, non-load-bearing).

Per object COLOR (and optional per-instance object key), learn the avatar's interaction
effect on contact: BLOCK / PASS / PUSH / COLLECT / TOGGLE / GOAL / HARM. The 8-label
ontology (incl. NONE) is a complete-by-construction closure of single-object-on-contact
outcomes on a 64x64 / 16-color grid: on stepping into a cell the object either stops you
(BLOCK), lets you through unchanged (PASS), translates rigidly (PUSH), disappears (COLLECT),
changes something elsewhere (TOGGLE), ends the level with reward (GOAL), or kills you (HARM).

Learning is a pure, training-free, color-keyed online accumulator. Each real step is fed to
``observe_step(before, after, action, mm, bg, reward, terminal, ...)`` which converts the
transition into per-color affordance votes, accumulated as a Dirichlet posterior so a
half-learned model is provably inert (``Verdict.reliable`` gates on support>=2 & conf>=0.66;
terminal GOAL/HARM signals are reliable at support>=1 via a single-shot confidence boost).

This module is **observe-only**: it writes ONLY to its own ``AffordanceModel`` state and is
read by NOTHING in the decision path in this PR. It is wired into ``policy.py`` behind an
``enable_affordance`` flag and wrapped in try/except, so a C3 bug degrades to "no learning",
never a policy crash, and the reactive action stream is byte-identical with the flag on or
off (proven by the golden action-trace test). Push/sokoban is solved by the GraphStrategy
BFS fallback, NOT navigation, so touching ``_next_target``/``_nav_step`` is deliberately out
of scope here; consumption is deferred to C4 (forward model) and C6 (planner), which read the
fixed query contract: ``affordance``/``blocks``/``is_harm``/``pushables``/``collectibles``/
``goal_colors``/``harm_colors``/``navigation_cost``.

The ``ContactDetector`` seam (the arrow/click contact + classification front-end) and the
``_remote_change`` C2-stand-in are the adapter points where real C1 (object tracking) / C2
(causal events) can replace the heuristic internals later without touching the learner or any
policy hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import movement as MV


class Effect(str, Enum):
    """Complete-by-construction closure of single-object-on-contact outcomes."""

    BLOCK = "BLOCK"      # avatar didn't translate -> solid wall
    PASS = "PASS"        # avatar entered, object unchanged -> passable floor/decor
    PUSH = "PUSH"        # object rigidly translated by the same delta -> pushable
    COLLECT = "COLLECT"  # object vanished where the avatar arrived -> item
    TOGGLE = "TOGGLE"    # contact caused a remote change -> switch/lever
    GOAL = "GOAL"        # contact yielded reward (level up) -> goal cell
    HARM = "HARM"        # contact ended the level (terminal) -> hazard
    NONE = "NONE"        # no signal


# Tie-break priority: terminal signals dominate; among physics, interactive effects
# (COLLECT/PUSH/TOGGLE) beat structural ones (BLOCK/PASS) so an interactive object is never
# mislabeled a wall on a tie. BLOCK does NOT outrank GOAL (P1's bug, fixed here).
_PRIORITY: dict[Effect, int] = {
    Effect.GOAL: 7,
    Effect.HARM: 7,
    Effect.COLLECT: 5,
    Effect.PUSH: 4,
    Effect.TOGGLE: 3,
    Effect.BLOCK: 2,
    Effect.PASS: 1,
    Effect.NONE: 0,
}

_ALPHA = 0.5  # Jeffreys prior for the Dirichlet posterior
_N_LABELS = 8  # |Effect| including NONE (denominator normaliser)


@dataclass
class Verdict:
    """A queried affordance: the winning effect, its confidence, and contact support."""

    effect: Effect
    confidence: float  # 0..1 (Dirichlet posterior of the winner)
    support: int       # number of contacts observed for this color

    @property
    def reliable(self) -> bool:
        """A half-learned verdict is inert; consumers act only on reliable ones.

        Reliable iff support>=2 and confidence>=0.66, EXCEPT terminal GOAL/HARM signals
        which are reliable at support>=1 (their single observation already gets a conf>=0.9
        boost, since stepping on a hazard / goal once is enough to trust it).
        """
        if self.effect in (Effect.GOAL, Effect.HARM):
            return self.support >= 1 and self.confidence >= 0.66
        return self.support >= 2 and self.confidence >= 0.66


@dataclass
class ColorStat:
    """Per-color accumulator of contact effects -> a majority verdict with confidence."""

    color: int
    counts: dict[Effect, int] = field(default_factory=dict)
    total: int = 0
    last_step: int = 0

    def observe(self, e: Effect, step: int) -> None:
        if e == Effect.NONE:
            return
        self.counts[e] = self.counts.get(e, 0) + 1
        self.total += 1
        self.last_step = step

    def verdict(self) -> tuple[Effect, float, int]:
        """Return (winning_effect, Dirichlet confidence, support)."""
        if not self.counts:
            return (Effect.NONE, 0.0, 0)
        # argmax over counts, ties broken by _PRIORITY (terminal signals dominate)
        winner = max(self.counts, key=lambda e: (self.counts[e], _PRIORITY[e]))
        conf = (self.counts[winner] + _ALPHA) / (self.total + _N_LABELS * _ALPHA)
        # single-shot high-confidence override for irreversible terminal signals
        if winner in (Effect.GOAL, Effect.HARM) and self.counts[winner] >= 1:
            conf = max(conf, 0.9)
        return (winner, conf, self.total)


# --------------------------------------------------------------------------------------
# Module-level pure helpers (directly unit-testable; no frame/model state)
# --------------------------------------------------------------------------------------

def _target_cells(a_before: np.ndarray, dr: int, dc: int,
                  shape: tuple[int, int]) -> set[tuple[int, int]]:
    """Cells the avatar TRIED to step into: footprint+delta, in-bounds, minus own footprint.

    ``a_before`` is an (N,2) array of avatar (row,col) cells on the BEFORE grid.
    """
    h, w = shape
    own = {tuple(p) for p in a_before.tolist()}
    out: set[tuple[int, int]] = set()
    for (r, c) in own:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in own:
            out.add((nr, nc))
    return out


def _colors_at(grid: np.ndarray, cells: set[tuple[int, int]], bg: int,
               avatar_colors: frozenset[int],
               distractor_colors: frozenset[int] = frozenset()) -> set[int]:
    """Distinct non-bg, non-avatar, non-distractor colors at ``cells`` on ``grid``."""
    out: set[int] = set()
    for (r, c) in cells:
        v = int(grid[r, c])
        if v != bg and v not in avatar_colors and v not in distractor_colors:
            out.add(v)
    return out


def _color_translated_by(before: np.ndarray, after: np.ndarray, col: int,
                         dr: int, dc: int, bg: int) -> bool:
    """True iff ``col`` rigidly translated by exactly (dr,dc). Wraps infer_all_translations.

    Does NOT reimplement rigid-translation logic — reuses movement.infer_all_translations
    verbatim (the same pass the probe path uses).
    """
    trans = MV.infer_all_translations(before, after, bg)
    return trans.get(col) == (dr, dc)


def _object_consumed(before: np.ndarray, after: np.ndarray, col: int, bg: int,
                     target_cells: set[tuple[int, int]]) -> bool:
    """True iff ``col`` shrank AND a contacted cell that held it is no longer ``col``.

    Heuristic for COLLECT: the avatar arrived on the item's cell and the item vanished
    (fewer ``col`` cells overall, and at least one contacted ``col`` cell is now non-col).
    """
    cb = int(np.count_nonzero(before == col))
    ca = int(np.count_nonzero(after == col))
    if ca >= cb:
        return False
    for (r, c) in target_cells:
        if int(before[r, c]) == col and int(after[r, c]) != col:
            return True
    return False


def _remote_change(before: np.ndarray, after: np.ndarray, bg: int,
                   avatar_colors: frozenset[int], exclude: frozenset[int],
                   footprint: set[tuple[int, int]]) -> bool:
    """True iff something changed OUTSIDE the avatar footprint and the contacted color.

    C2-stand-in seam: a TOGGLE is a contact whose effect is a change elsewhere on the grid
    (a remote door opening when a switch is touched). We subtract the avatar's before/after
    footprint and the contacted color ``exclude`` from the diff; any residual non-incidental
    change is the remote effect. Real C2 replaces this without touching the learner.
    """
    diff = before != after
    if not diff.any():
        return False
    for (r, c) in np.argwhere(diff).tolist():
        if (r, c) in footprint:
            continue
        b = int(before[r, c])
        a = int(after[r, c])
        if b in exclude or a in exclude:
            continue
        # incidental iff both endpoints are avatar/background (avatar trail over floor)
        if b in avatar_colors or a in avatar_colors:
            continue
        if b == bg and a == bg:
            continue
        return True
    return False


def _component_color_at(grid: np.ndarray, x: int, y: int, bg: int) -> int | None:
    """Color under a click at (x=col, y=row); None if out of bounds or background."""
    h, w = grid.shape
    r, c = int(y), int(x)
    if not (0 <= r < h and 0 <= c < w):
        return None
    v = int(grid[r, c])
    return None if v == bg else v


# --------------------------------------------------------------------------------------
# The learner
# --------------------------------------------------------------------------------------

class AffordanceModel:
    """Online, color-keyed affordance accumulator. Observe-only; never alters caller state.

    Primary keying is by COLOR (generalizes across instances of the same object). An optional
    per-instance ``obj_key`` records a parallel ColorStat used as an override when a confident
    per-object verdict exists (P2's two-level keying); default consumers pass color only.
    """

    def __init__(self) -> None:
        self.stats: dict[int, ColorStat] = {}            # color -> stat (primary)
        self.by_obj: dict[tuple, ColorStat] = {}         # obj_key -> stat (optional override)
        self.harm: set[int] = set()                      # fast O(1) per-nav-step lookup
        self.goal: set[int] = set()

    # -- learning -------------------------------------------------------------------
    def observe_step(self, before: np.ndarray, after: np.ndarray, action: tuple,
                     mm, bg: int, reward: float, terminal: bool, step: int = 0,
                     distractor_colors: frozenset[int] = frozenset(),
                     obj_key: tuple | None = None) -> dict[int, Effect]:
        """Convert one real step into per-color affordance votes. Returns {color: Effect}.

        Pure and observe-only: mutates only ``self.*``; never touches ``before``/``after``/
        ``mm``. Returns an empty dict on any no-signal step (so callers can ignore it).
        """
        if before is None or after is None or before.shape != after.shape:
            return {}
        if not action:
            return {}
        distractor_colors = frozenset(distractor_colors)

        # ----- click path (generality for click / mixed games) -----
        if action[0] == "C":
            return self._observe_click(before, after, action, bg, reward, terminal, step)

        # ----- arrow / simple-move path -----
        if mm is None or not mm.ok or action[0] != "S":
            return {}
        delta = mm.deltas.get(action[1])
        if delta is None or delta == (0, 0):
            return {}
        dr, dc = delta
        avatar_colors = frozenset(mm.avatar_colors) or (
            frozenset({mm.avatar_color}) if mm.avatar_color is not None else frozenset())

        a_before = mm.avatar_cells(before)
        if len(a_before) == 0:
            return {}
        target_cells = _target_cells(a_before, dr, dc, before.shape)
        if not target_cells:
            return {}
        touched = _colors_at(before, target_cells, bg, avatar_colors, distractor_colors)
        if not touched:
            return {}  # walked onto open floor -> no signal

        # did the avatar actually translate by (dr,dc)?
        before_set = {tuple(p) for p in a_before.tolist()}
        after_set = {tuple(p) for p in mm.avatar_cells(after).tolist()}
        expected = {(r + dr, c + dc) for (r, c) in before_set}
        moved = expected == after_set
        footprint = before_set | after_set

        results: dict[int, Effect] = {}
        for col in touched:
            if terminal:
                e = Effect.HARM
            elif reward > 0:
                e = Effect.GOAL
            elif not moved:
                e = Effect.BLOCK
            elif _object_consumed(before, after, col, bg, target_cells):
                e = Effect.COLLECT
            elif _color_translated_by(before, after, col, dr, dc, bg):
                e = Effect.PUSH
            elif _remote_change(before, after, bg, avatar_colors,
                                exclude=frozenset({col}), footprint=footprint):
                e = Effect.TOGGLE
            else:
                e = Effect.PASS
            results[col] = e
            self._record(col, e, step, obj_key)
        return results

    def _observe_click(self, before: np.ndarray, after: np.ndarray, action: tuple,
                       bg: int, reward: float, terminal: bool, step: int) -> dict[int, Effect]:
        if len(action) < 3:
            return {}
        col = _component_color_at(before, action[1], action[2], bg)
        if col is None or col == bg:
            return {}
        if terminal:
            e = Effect.HARM
        elif reward > 0:
            e = Effect.GOAL
        elif int(np.count_nonzero(after == col)) < int(np.count_nonzero(before == col)):
            e = Effect.COLLECT
        elif _remote_change(before, after, bg, frozenset(),
                            exclude=frozenset({col}), footprint=set()):
            e = Effect.TOGGLE
        else:
            e = Effect.NONE
        if e == Effect.NONE:
            return {}
        self._record(col, e, step, obj_key=None)
        return {col: e}

    def _record(self, col: int, e: Effect, step: int, obj_key: tuple | None) -> None:
        if e == Effect.NONE:
            return
        cs = self.stats.setdefault(col, ColorStat(col))
        cs.observe(e, step)
        if obj_key is not None:
            ocs = self.by_obj.setdefault(obj_key, ColorStat(col))
            ocs.observe(e, step)
        if e == Effect.HARM:
            self.harm.add(col)
        if e == Effect.GOAL:
            self.goal.add(col)

    # -- query API (the C4 / C6 contract) -------------------------------------------
    def affordance(self, color: int, obj_key: tuple | None = None) -> Verdict:
        """Best verdict for a color, preferring a confident per-instance override."""
        if obj_key is not None and obj_key in self.by_obj:
            e, conf, n = self.by_obj[obj_key].verdict()
            ov = Verdict(e, conf, n)
            if ov.reliable:
                return ov
        cs = self.stats.get(color)
        if cs is None:
            return Verdict(Effect.NONE, 0.0, 0)
        e, conf, n = cs.verdict()
        return Verdict(e, conf, n)

    def blocks(self, color: int) -> bool:
        v = self.affordance(color)
        return v.effect == Effect.BLOCK and v.reliable

    def is_harm(self, color: int) -> bool:
        return color in self.harm

    def _reliable_colors(self, effect: Effect) -> list[int]:
        out: list[int] = []
        for c, s in self.stats.items():
            v = Verdict(*s.verdict())
            if v.effect == effect and v.reliable:
                out.append(c)
        return sorted(out)

    def pushables(self) -> list[int]:
        return self._reliable_colors(Effect.PUSH)

    def collectibles(self) -> list[int]:
        return self._reliable_colors(Effect.COLLECT)

    def goal_colors(self) -> list[int]:
        return sorted(self.goal)

    def harm_colors(self) -> list[int]:
        return sorted(self.harm)

    def navigation_cost(self, color: int) -> float:
        """A* edge cost for stepping onto ``color`` (for C6). Unreliable -> neutral 1.0."""
        if color in self.harm:
            return float("inf")
        v = self.affordance(color)
        if not v.reliable:
            return 1.0
        if v.effect == Effect.BLOCK:
            return float("inf")
        if v.effect == Effect.PUSH:
            return 3.0  # pushing is costlier than walking; planner may still choose it
        if v.effect in (Effect.GOAL, Effect.COLLECT):
            return 0.0
        return 1.0

    def reset_level(self) -> None:
        """On a new level keep per-COLOR stats (cross-level priors as votes; contradictions
        outvoted over time) and the harm/goal sets; clear only per-OBJECT stats.

        One-line tunable: replace the body with a x0.5 decay of every count if a real game
        shows harmful carry across re-skinned levels.
        """
        self.by_obj = {}
