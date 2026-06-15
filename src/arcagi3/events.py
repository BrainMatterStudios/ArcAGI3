"""C2 — Causal event extraction (read-only, default-OFF).

Given the settled before/after grid pair around a single action (plus the existing
MotionModel), this module subtracts the avatar's actual footprint — the cells it vacated or
newly covers — and classifies the remaining geometric *residual* into a typed event stream
(VANISH, APPEAR, OBJECT_MOVE, RECOLOR, COUNTER, AVATAR_BLOCKED, LEVEL_COMPLETED, ...).

The point: a pure avatar move over floor/maze leaves zero residual -> empty event list ->
``has_real_event == False``. That is the direct antidote to the "98% of states look like
progress" failure that defeated the naive goal-inference. Collect is captured by promoting
"avatar stepped onto a non-bg, non-avatar cell" to a contact VANISH; the switchdoor remote
door is a non-contact VANISH on the same step as the switch contact; push is OBJECT_MOVE via
``movement.infer_all_translations`` restricted to non-avatar colors.

This module is **read-only** and **stateless** (except a tiny per-color change-history ring
buffer used only to confirm COUNTER colors). It is wired into ``policy.py`` behind an
``emit_events=False`` flag and is consulted by NOTHING in the default decision path. When
the flag is off the entire block is skipped; when on it only *writes* the log. Consumers
(C3/C5/C7) read ``policy.last_step_events`` / ``policy.events``; they must never enable
``emit_events`` in the submission path nor route C2 output into the state key / distractor
set until C7's selector gates a new policy.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from . import movement as MV
from . import perception as P


class EventType(Enum):
    LEVEL_COMPLETED = "level_completed"
    OBJECT_VANISHED = "object_vanished"
    OBJECT_APPEARED = "object_appeared"
    OBJECT_MOVED = "object_moved"
    OBJECT_RECOLORED = "object_recolored"
    COUNTER_CHANGED = "counter_changed"
    AVATAR_BLOCKED = "avatar_blocked"
    REGION_CHANGED = "region_changed"


# Salience order (lower index = more salient). Used to sort the per-step event tuple.
_SALIENCE = {
    EventType.LEVEL_COMPLETED: 0,
    EventType.OBJECT_VANISHED: 1,
    EventType.OBJECT_RECOLORED: 2,
    EventType.OBJECT_APPEARED: 3,
    EventType.OBJECT_MOVED: 4,
    EventType.REGION_CHANGED: 5,
    EventType.COUNTER_CHANGED: 6,
    EventType.AVATAR_BLOCKED: 7,
}

# Event types that do NOT by themselves count as a "real" game event.
_INERT = {EventType.AVATAR_BLOCKED, EventType.COUNTER_CHANGED}


@dataclass(frozen=True)
class Event:
    """A single typed change attributed to one action (frozen / immutable)."""

    type: EventType
    color: Optional[int] = None
    cells: tuple[tuple[int, int], ...] = ()
    bbox: Optional[tuple[int, int, int, int]] = None
    size: int = 0
    delta: Optional[tuple[int, int]] = None  # (dr, dc) for OBJECT_MOVED
    action: Optional[tuple] = None
    avatar_cell: Optional[tuple[int, int]] = None
    contact: bool = False  # change is adjacent to where the avatar acted
    local: bool = False  # change is near the avatar centroid
    low_confidence: bool = False
    extra: tuple = ()  # frozen key/value pairs, e.g. (("from",2),("to",5))


@dataclass(frozen=True)
class StepEvents:
    """All events extracted for one step, plus cheap step-level summaries."""

    events: tuple[Event, ...] = ()
    action: Optional[tuple] = None
    reward: float = 0.0
    avatar_moved: bool = False
    avatar_blocked: bool = False

    @property
    def has_real_event(self) -> bool:
        """True iff something happened beyond plain navigation/counters/blocks."""
        if self.reward > 0:
            return True
        return any(e.type not in _INERT for e in self.events)

    @property
    def salient(self) -> tuple[Event, ...]:
        """Events excluding inert (BLOCKED/COUNTER) signals."""
        return tuple(e for e in self.events if e.type not in _INERT)


@dataclass
class EventLog:
    """Per-level accumulator of StepEvents (used by C5 goal-inference)."""

    steps: list[StepEvents] = field(default_factory=list)

    def append(self, se: StepEvents) -> None:
        self.steps.append(se)

    def clear(self) -> None:
        self.steps = []

    def __len__(self) -> int:
        return len(self.steps)

    def last_real(self, n: int = 1) -> list[StepEvents]:
        """The most recent ``n`` steps that carried a real event (most recent first)."""
        out: list[StepEvents] = []
        for se in reversed(self.steps):
            if se.has_real_event:
                out.append(se)
                if len(out) >= n:
                    break
        return out


def _components_of_mask(mask: np.ndarray, grid: np.ndarray) -> list[P.Obj]:
    """Connected components (4-conn) restricted to ``mask`` cells, colored by ``grid``.

    Reuses perception.connected_components by painting a temp grid where non-mask cells are
    a sentinel background. Memoized inside connected_components.
    """
    if not mask.any():
        return []
    sentinel = -1
    tmp = np.where(mask, grid, np.int8(sentinel))
    return P.connected_components(tmp, background=sentinel)


class EventExtractor:
    """Stateless event extractor (except a tiny per-color change-history ring buffer).

    The history buffer only confirms COUNTER colors over time; it never affects masking or
    the decision path.
    """

    def __init__(self, history: int = 6) -> None:
        self._hist_len = history
        # color -> ring buffer of recent "this color changed this step" booleans
        self._change_hist: dict[int, deque[bool]] = {}
        self.mm: MV.MotionModel | None = None

    def update_model(self, mm: MV.MotionModel | None) -> None:
        self.mm = mm

    def reset(self) -> None:
        self._change_hist = {}

    # -- main entry -----------------------------------------------------------------
    def extract(
        self,
        before: np.ndarray,
        after: np.ndarray,
        action: Optional[tuple],
        reward: float,
        mm: MV.MotionModel | None = None,
        bg: int | None = None,
        distractor_colors: set[int] | None = None,
        click_xy: tuple[int, int] | None = None,
    ) -> StepEvents:
        mm = mm if mm is not None else self.mm
        distractor_colors = distractor_colors or set()
        if bg is None:
            bg = P.detect_background(after)

        completed = reward > 0

        # ----- Step 0: FAST PATHS -----
        if before is None or before.shape != after.shape:
            evs: tuple[Event, ...] = ()
            if completed:
                evs = (Event(EventType.LEVEL_COMPLETED, action=action),)
            return StepEvents(events=evs, action=action, reward=reward)

        if np.array_equal(before, after):
            evs = (Event(EventType.LEVEL_COMPLETED, action=action),) if completed else ()
            return StepEvents(events=evs, action=action, reward=reward)

        # ----- Step 1: CHANGED MASK + REDRAW GUARD -----
        diff = before != after
        n = int(diff.sum())
        head: list[Event] = []
        if completed:
            head.append(Event(EventType.LEVEL_COMPLETED, action=action))
        if n > 0.40 * before.size:
            head.append(Event(EventType.REGION_CHANGED, size=n, low_confidence=True))
            return self._finalize(head, action, reward, after, mm, click_xy,
                                  avatar_moved=False, avatar_blocked=False)

        avatar_cols = self._avatar_colors(mm)

        # ----- Step 2: AVATAR FOOTPRINT -----
        av_b = set()
        av_a = set()
        avatar_moved = False
        if mm is not None and mm.ok:
            av_b = {tuple(p) for p in mm.avatar_cells(before).tolist()}
            av_a = {tuple(p) for p in mm.avatar_cells(after).tolist()}
            if av_b and av_a:
                cb = np.mean(np.array(list(av_b)), axis=0)
                ca = np.mean(np.array(list(av_a)), axis=0)
                avatar_moved = float(np.hypot(*(ca - cb))) >= 0.5
        footprint = av_b | av_a

        diff_cells = {tuple(p) for p in np.argwhere(diff).tolist()}

        # ----- Step 3: COLLECT PROMOTION (reveal-under-footprint) -----
        collect_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
        newly_covered = av_a - av_b
        for (r, c) in newly_covered:
            under = int(before[r, c])
            if under != bg and under not in avatar_cols:
                collect_cells[under].append((r, c))

        collect_events: list[Event] = []
        consumed: set[tuple[int, int]] = set()
        for color, cells in collect_cells.items():
            for grp in self._group_cells(cells):
                collect_events.append(
                    Event(
                        EventType.OBJECT_VANISHED,
                        color=color,
                        cells=tuple(sorted(grp)),
                        bbox=_bbox(grp),
                        size=len(grp),
                        contact=True,
                    )
                )
                consumed |= set(grp)

        # ----- Step 4: RESIDUAL = diff - footprint - collect (and incidental drop) -----
        residual: set[tuple[int, int]] = set()
        for (r, c) in diff_cells:
            if (r, c) in footprint or (r, c) in consumed:
                continue
            b = int(before[r, c])
            a = int(after[r, c])
            # incidental iff both endpoints are avatar/background (avatar passing over floor)
            if {b, a} <= (avatar_cols | {bg}):
                continue
            residual.add((r, c))

        body: list[Event] = list(collect_events)

        # update COUNTER change-history for all changed non-bg colors (read-only)
        self._update_history(before, after, bg)

        if not residual:
            return self._finalize(head + body, action, reward, after, mm, click_xy,
                                  avatar_moved=avatar_moved, avatar_blocked=False)

        # ----- Step 5: CLASSIFY RESIDUAL (order matters) -----
        body += self._classify_residual(before, after, bg, avatar_cols,
                                        distractor_colors, residual)

        # ----- Step 6: AVATAR_BLOCKED -----
        avatar_blocked = False
        if (
            mm is not None and mm.ok and action is not None
            and len(action) == 2 and action[0] == "S"
        ):
            aid = action[1]
            d = mm.deltas.get(aid)
            if d is not None and d != (0, 0) and not avatar_moved:
                moved_into = any(
                    e.type == EventType.OBJECT_MOVED for e in body
                )
                if not moved_into:
                    avatar_blocked = True
                    body.append(Event(EventType.AVATAR_BLOCKED, action=action,
                                      delta=d, contact=True))

        return self._finalize(head + body, action, reward, after, mm, click_xy,
                              avatar_moved=avatar_moved, avatar_blocked=avatar_blocked)

    # -- helpers --------------------------------------------------------------------
    def _avatar_colors(self, mm: MV.MotionModel | None) -> set[int]:
        if mm is None or not mm.ok:
            return set()
        cols = set(mm.avatar_colors)
        if not cols and mm.avatar_color is not None:
            cols = {mm.avatar_color}
        return cols

    @staticmethod
    def _group_cells(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """4-connectivity grouping of a list of (r,c) cells."""
        cellset = set(cells)
        seen: set[tuple[int, int]] = set()
        groups: list[list[tuple[int, int]]] = []
        for start in cells:
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            grp = []
            while q:
                r, c = q.popleft()
                grp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in cellset and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            groups.append(grp)
        return groups

    def _classify_residual(
        self,
        before: np.ndarray,
        after: np.ndarray,
        bg: int,
        avatar_cols: set[int],
        distractor_colors: set[int],
        residual: set[tuple[int, int]],
    ) -> list[Event]:
        out: list[Event] = []
        remaining = set(residual)

        # 5a. OBJECT_MOVED: rigid translations of non-avatar colors overlapping residual
        trans = MV.infer_all_translations(before, after, bg)
        for color, (dr, dc) in trans.items():
            # avatar colors are footprint, distractor colors are reported as COUNTER (via
            # the vanish/appear passes below) -- never as a salient OBJECT_MOVED.
            if color in avatar_cols or color in distractor_colors or (dr, dc) == (0, 0):
                continue
            after_cells = {tuple(p) for p in np.argwhere(after == color).tolist()}
            before_cells = {tuple(p) for p in np.argwhere(before == color).tolist()}
            moved_cells = (after_cells | before_cells) & remaining
            if not moved_cells:
                continue
            out.append(
                Event(
                    EventType.OBJECT_MOVED,
                    color=color,
                    cells=tuple(sorted(after_cells)),
                    bbox=_bbox(list(after_cells)) if after_cells else None,
                    size=len(after_cells),
                    delta=(dr, dc),
                )
            )
            remaining -= moved_cells

        # 5b. RECOLOR: cells where before!=bg and after!=bg, both non-avatar (in-place flip)
        recolor_cells = [
            (r, c) for (r, c) in remaining
            if int(before[r, c]) != bg and int(after[r, c]) != bg
            and int(before[r, c]) not in avatar_cols and int(after[r, c]) not in avatar_cols
        ]
        # group by (from,to) then by connectivity
        by_pair: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for (r, c) in recolor_cells:
            by_pair[(int(before[r, c]), int(after[r, c]))].append((r, c))
        for (frm, to), cells in by_pair.items():
            for grp in self._group_cells(cells):
                out.append(
                    Event(
                        EventType.OBJECT_RECOLORED,
                        color=to,
                        cells=tuple(sorted(grp)),
                        bbox=_bbox(grp),
                        size=len(grp),
                        extra=(("from", frm), ("to", to)),
                    )
                )
                remaining -= set(grp)

        # 5c. VANISH: before non-bg/non-avatar, after == bg, in residual
        vanish_cells = [
            (r, c) for (r, c) in remaining
            if int(after[r, c]) == bg
            and int(before[r, c]) != bg and int(before[r, c]) not in avatar_cols
        ]
        by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (r, c) in vanish_cells:
            by_color[int(before[r, c])].append((r, c))
        for color, cells in by_color.items():
            etype = (EventType.COUNTER_CHANGED if color in distractor_colors
                     else EventType.OBJECT_VANISHED)
            for grp in self._group_cells(cells):
                out.append(
                    Event(
                        etype,
                        color=color,
                        cells=tuple(sorted(grp)),
                        bbox=_bbox(grp),
                        size=len(grp),
                    )
                )
                remaining -= set(grp)

        # 5d. APPEAR: after non-bg/non-avatar, before == bg, in residual
        appear_cells = [
            (r, c) for (r, c) in remaining
            if int(before[r, c]) == bg
            and int(after[r, c]) != bg and int(after[r, c]) not in avatar_cols
        ]
        by_color = defaultdict(list)
        for (r, c) in appear_cells:
            by_color[int(after[r, c])].append((r, c))
        for color, cells in by_color.items():
            etype = (EventType.COUNTER_CHANGED if color in distractor_colors
                     else EventType.OBJECT_APPEARED)
            for grp in self._group_cells(cells):
                out.append(
                    Event(
                        etype,
                        color=color,
                        cells=tuple(sorted(grp)),
                        bbox=_bbox(grp),
                        size=len(grp),
                    )
                )
                remaining -= set(grp)

        # 5e. leftover residual -> conservative unclassified APPEAR (never miss a goal)
        if remaining:
            grp = sorted(remaining)
            out.append(
                Event(
                    EventType.OBJECT_APPEARED,
                    color=int(after[grp[0][0], grp[0][1]]),
                    cells=tuple(grp),
                    bbox=_bbox(grp),
                    size=len(grp),
                    extra=(("unclassified", True),),
                )
            )
        return out

    def _update_history(self, before: np.ndarray, after: np.ndarray, bg: int) -> None:
        colors = set(np.unique(before).tolist()) | set(np.unique(after).tolist())
        for c in colors:
            c = int(c)
            if c == bg:
                continue
            changed = not np.array_equal(before == c, after == c)
            buf = self._change_hist.setdefault(c, deque(maxlen=self._hist_len))
            buf.append(changed)

    def _finalize(
        self,
        events: list[Event],
        action: Optional[tuple],
        reward: float,
        after: np.ndarray,
        mm: MV.MotionModel | None,
        click_xy: tuple[int, int] | None,
        avatar_moved: bool,
        avatar_blocked: bool,
    ) -> StepEvents:
        # ----- Step 7: CONTACT / LOCAL flags -----
        ac = None
        if mm is not None and mm.ok:
            ac = mm.avatar_centroid(after)
        contacted = None
        radius = 1.5
        if mm is not None and mm.ok and mm.deltas:
            max_step = max((abs(dr) + abs(dc) for dr, dc in mm.deltas.values()),
                           default=1)
            radius = max(1.5, 1.5 * max_step)
            if ac is not None and action is not None and len(action) == 2 and action[0] == "S":
                d = mm.deltas.get(action[1], (0, 0))
                contacted = (ac[0] + d[0], ac[1] + d[1])
        if contacted is None and click_xy is not None:
            # click games: contacted cell is (row=y, col=x)
            contacted = (click_xy[1], click_xy[0])

        flagged: list[Event] = []
        for e in events:
            if not e.cells or e.type == EventType.LEVEL_COMPLETED:
                flagged.append(e)
                continue
            contact = e.contact
            local = e.local
            if contacted is not None:
                cd = min(abs(r - contacted[0]) + abs(c - contacted[1]) for (r, c) in e.cells)
                contact = contact or (cd <= radius)
            if ac is not None:
                ld = min(abs(r - ac[0]) + abs(c - ac[1]) for (r, c) in e.cells)
                local = local or (ld <= radius)
            avc = (int(round(ac[0])), int(round(ac[1]))) if ac is not None else None
            flagged.append(
                Event(
                    type=e.type, color=e.color, cells=e.cells, bbox=e.bbox, size=e.size,
                    delta=e.delta, action=e.action or action, avatar_cell=avc,
                    contact=contact, local=local, low_confidence=e.low_confidence,
                    extra=e.extra,
                )
            )

        # ----- Step 8: sort by salience -----
        flagged.sort(key=lambda e: _SALIENCE.get(e.type, 99))
        return StepEvents(
            events=tuple(flagged),
            action=action,
            reward=reward,
            avatar_moved=avatar_moved,
            avatar_blocked=avatar_blocked,
        )


def _bbox(cells: list[tuple[int, int]]) -> Optional[tuple[int, int, int, int]]:
    if not cells:
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))
