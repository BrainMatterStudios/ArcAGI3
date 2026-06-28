"""Object tracking / persistence — C1 of the rebuild plan.

Consumes the already-memoized ``perception.connected_components`` output (never re-segments,
never re-implements geometry), assigns stable per-instance integer ids across frames, and
emits a typed event stream {APPEARED, VANISHED, MOVED, CHANGED, RECOLORED, SPLIT, MERGED}.

DORMANT LANDING: this module is NOT imported anywhere in policy.py / agent.py.
No call site exists in the live path, so the submission agent is byte-identical to before.
The optional gated wiring (``track_objects`` flag) is deferred to the C7 integration PR.

Design decisions (from the judged C1 blueprint):
- Matching is per-color, motion-aware greedy nearest-neighbour with a distance gate and a
  translation/velocity prior reusing ``movement.infer_all_translations``.
- Short occlusion grace with velocity coasting prevents avatar-over-floor flicker.
- ``match()`` is PURE (no mutation of inputs, no I/O).
- ``_next_id`` is monotonic and is NOT reset across levels so ids are never ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import perception as P
from . import movement as MV

# ---------------------------------------------------------------------------
# Event-kind constants
# ---------------------------------------------------------------------------

APPEARED = "APPEARED"
VANISHED = "VANISHED"
MOVED = "MOVED"
CHANGED = "CHANGED"
RECOLORED = "RECOLORED"
SPLIT = "SPLIT"
MERGED = "MERGED"

# ---------------------------------------------------------------------------
# Tunables (conservative, hand-set, NOT tuned on the local 8-game suite)
# ---------------------------------------------------------------------------

_MAX_MATCH_DIST: float = 8.0   # manhattan gate on centroid-to-centroid
_OCCLUSION_GRACE: int = 2      # frames an object may be missing before VANISHED
_OVERLAP_THRESH: float = 0.5   # fraction of own cells that must overlap for SPLIT/MERGE
_VELOCITY_EMA: float = 0.5     # exponential moving-average weight for velocity
_W_SIZE: float = 0.25          # tie-break cost weight for size difference


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackedObj:
    """A connected component with stable id and temporal state."""

    id: int
    color: int
    cells: tuple[tuple[int, int], ...]     # (row, col) pairs
    bbox: tuple[int, int, int, int]        # (r0, c0, r1, c1) inclusive
    size: int
    centroid: tuple[float, float]
    age: int = 0                            # frames alive
    missed: int = 0                         # consecutive frames not matched
    first_seen: int = 0                     # frame index when first tracked
    last_seen: int = 0                      # frame index when last matched
    velocity: tuple[float, float] = (0.0, 0.0)   # EMA of centroid delta (dr, dc)
    color_history: tuple[int, ...] = ()     # ordered history of observed colors

    @classmethod
    def from_obj(cls, oid: int, obj: P.Obj, frame: int) -> "TrackedObj":
        """Construct from a perception.Obj (stateless, used to populate cur list)."""
        return cls(
            id=oid,
            color=obj.color,
            cells=obj.cells,
            bbox=obj.bbox,
            size=obj.size,
            centroid=obj.centroid,
            age=0,
            missed=0,
            first_seen=frame,
            last_seen=frame,
            velocity=(0.0, 0.0),
            color_history=(obj.color,),
        )


@dataclass
class Event:
    """A typed change event emitted by ``match()``."""

    kind: str                                    # one of the constants above
    id: int                                      # stable object id (-1 for APPEARED placeholders)
    color: int
    delta: tuple[float, float] = (0.0, 0.0)     # centroid displacement (dr, dc)
    size_delta: int | None = None                # after.size - before.size, or None
    before: tuple | None = None                  # (bbox, size) snapshot before change
    after: tuple | None = None                   # (bbox, size) snapshot after change
    extra: dict = field(default_factory=dict)    # lineage: split into=[ids] / merge from=[ids]


@dataclass
class TrackResult:
    """Result of one ``match()`` / ``update()`` call."""

    objects: list[TrackedObj]                        # current TrackedObj list (ids assigned)
    events: list[Event]                              # all events this frame
    correspondences: dict[int, int]                  # cur_idx -> prev_id (matched pairs)
    by_id: dict[int, TrackedObj]                     # id -> TrackedObj (all live tracks)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def _is_rigid_shift(
    prev_cells: tuple[tuple[int, int], ...],
    cur_cells: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> bool:
    """Return True iff shifting prev_cells by (dr,dc) yields exactly cur_cells.

    Inlined from movement.py:91-92 logic; movement.py is NOT modified.
    """
    shifted = {(r + dr, c + dc) for r, c in prev_cells}
    return shifted == set(cur_cells)


# ---------------------------------------------------------------------------
# ObjectTracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """Stateful tracker: call ``update(grid)`` each frame to get a TrackResult.

    ``match()`` is kept PURE for testing and reuse; ``update()`` is the only
    side-effecting entry point.

    State that persists across levels (on purpose):
    - ``_next_id``: monotonic counter so ids are globally unique, never reused.

    State that is cleared on ``reset()``:
    - ``_tracks``, ``_frame``, ``_prev_grid``, ``bg``.
    """

    def __init__(
        self,
        bg: int | None = None,
        ignore_colors: set[int] | None = None,
        detect_split_merge: bool = True,
    ) -> None:
        self.bg = bg
        self.ignore_colors: set[int] = ignore_colors or set()
        self.detect_split_merge = detect_split_merge

        # Monotonic id counter — intentionally NOT reset across levels.
        self._next_id: int = 0
        # Mutable tracker state — cleared on reset().
        self._tracks: dict[int, TrackedObj] = {}
        self._frame: int = 0
        self._prev_grid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-level state; keep ``_next_id`` monotonic."""
        self._tracks = {}
        self._frame = 0
        self._prev_grid = None
        # bg / ignore_colors remain: caller can override them per call.

    def objects(self, grid: np.ndarray) -> list[TrackedObj]:
        """Segment ``grid`` into TrackedObj instances with id=-1 (stateless).

        Uses the already-memoized ``perception.connected_components``.
        """
        bg = self.bg if self.bg is not None else P.detect_background(grid)
        raw = P.connected_components(grid, background=bg)
        return [
            TrackedObj.from_obj(-1, o, self._frame)
            for o in raw
            if o.color not in self.ignore_colors
        ]

    def update(
        self,
        grid: np.ndarray,
        background: int | None = None,
        ignore_colors: set[int] | None = None,
    ) -> TrackResult:
        """Stateful convenience: segment ``grid``, match against previous tracks,
        rebuild ``_tracks``, advance ``_frame``, and return a ``TrackResult``.

        Args:
            grid: the settled 64x64 frame (pass through ``perception.to_grid`` first).
            background: override background color (else uses ``self.bg`` or auto-detect).
            ignore_colors: override ignore set for this call only.
        """
        # Resolve bg / ignore for this call.
        bg = background if background is not None else (
            self.bg if self.bg is not None else P.detect_background(grid)
        )
        ignore = ignore_colors if ignore_colors is not None else self.ignore_colors

        # Segment current frame.
        raw_cur = P.connected_components(grid, background=bg)
        cur: list[TrackedObj] = [
            TrackedObj.from_obj(-1, o, self._frame)
            for o in raw_cur
            if o.color not in ignore
        ]

        # Translation prior from movement module.
        translations: dict[int, tuple[int, int]] | None = None
        if self._prev_grid is not None:
            translations = MV.infer_all_translations(self._prev_grid, grid, bg)

        # Pure match.
        prev_list = list(self._tracks.values())
        result = self.match(prev_list, cur, translations=translations, ignore_colors=ignore)

        # Rebuild _tracks applying correspondences.
        new_tracks: dict[int, TrackedObj] = {}

        # Matched cur objects: carry ids from previous, update state.
        for cur_idx, prev_id in result.correspondences.items():
            cobj = result.objects[cur_idx]
            prev_track = self._tracks.get(prev_id)
            if prev_track is None:
                continue
            dr = cobj.centroid[0] - prev_track.centroid[0]
            dc = cobj.centroid[1] - prev_track.centroid[1]
            new_vel = (
                _VELOCITY_EMA * dr + (1 - _VELOCITY_EMA) * prev_track.velocity[0],
                _VELOCITY_EMA * dc + (1 - _VELOCITY_EMA) * prev_track.velocity[1],
            )
            updated = TrackedObj(
                id=prev_id,
                color=cobj.color,
                cells=cobj.cells,
                bbox=cobj.bbox,
                size=cobj.size,
                centroid=cobj.centroid,
                age=prev_track.age + 1,
                missed=0,
                first_seen=prev_track.first_seen,
                last_seen=self._frame,
                velocity=new_vel,
                color_history=(
                    prev_track.color_history + (cobj.color,)
                    if (not prev_track.color_history or cobj.color != prev_track.color_history[-1])
                    else prev_track.color_history
                ),
            )
            new_tracks[prev_id] = updated

        # APPEARED: allocate real ids.
        appeared_ids: dict[int, int] = {}  # cur_idx -> new_id
        matched_cur_idxs = set(result.correspondences.keys())
        for cur_idx, cobj in enumerate(result.objects):
            if cur_idx not in matched_cur_idxs:
                new_id = self._next_id
                self._next_id += 1
                appeared_ids[cur_idx] = new_id
                new_obj = TrackedObj(
                    id=new_id,
                    color=cobj.color,
                    cells=cobj.cells,
                    bbox=cobj.bbox,
                    size=cobj.size,
                    centroid=cobj.centroid,
                    age=0,
                    missed=0,
                    first_seen=self._frame,
                    last_seen=self._frame,
                    velocity=(0.0, 0.0),
                    color_history=(cobj.color,),
                )
                new_tracks[new_id] = new_obj

        # Fix up APPEARED events with real ids.
        for ev in result.events:
            if ev.kind == APPEARED and ev.id == -1:
                # find which cur_idx this event corresponds to — matched by centroid
                for cur_idx, new_id in appeared_ids.items():
                    if result.objects[cur_idx].centroid == (ev.color, ev.delta):
                        # Not the right comparison — we'll match by object reference below.
                        pass

        # Simpler: replace events list with fixed APPEARED ids.
        fixed_events: list[Event] = []
        for ev in result.events:
            if ev.kind == APPEARED and ev.id == -1:
                # find cur_idx for this appeared event by (color, centroid stored in before)
                matched_new_id = -1
                for cur_idx, new_id in appeared_ids.items():
                    o = result.objects[cur_idx]
                    if o.color == ev.color and o.centroid == ev.before:
                        matched_new_id = new_id
                        break
                fixed_events.append(Event(
                    kind=APPEARED,
                    id=matched_new_id,
                    color=ev.color,
                    after=ev.after,
                ))
            else:
                fixed_events.append(ev)

        # Unmatched prev objects: increment missed, coast by velocity, or emit VANISHED.
        matched_prev_ids = set(result.correspondences.values())
        for prev_id, prev_track in self._tracks.items():
            if prev_id in matched_prev_ids:
                continue
            # Check if it was handled by split/merge (present in correspondences values).
            missed = prev_track.missed + 1
            if missed > _OCCLUSION_GRACE:
                # VANISHED event already in result.events from match()
                pass  # don't carry forward
            else:
                # Coast by velocity.
                coasted = TrackedObj(
                    id=prev_id,
                    color=prev_track.color,
                    cells=prev_track.cells,
                    bbox=prev_track.bbox,
                    size=prev_track.size,
                    centroid=(
                        prev_track.centroid[0] + prev_track.velocity[0],
                        prev_track.centroid[1] + prev_track.velocity[1],
                    ),
                    age=prev_track.age + 1,
                    missed=missed,
                    first_seen=prev_track.first_seen,
                    last_seen=prev_track.last_seen,
                    velocity=prev_track.velocity,
                    color_history=prev_track.color_history,
                )
                new_tracks[prev_id] = coasted

        self._tracks = new_tracks
        self._prev_grid = grid.copy()
        self._frame += 1

        # Build final result with real ids in objects list.
        final_objects: list[TrackedObj] = []
        for cur_idx, cobj in enumerate(result.objects):
            if cur_idx in result.correspondences:
                prev_id = result.correspondences[cur_idx]
                final_objects.append(new_tracks[prev_id])
            elif cur_idx in appeared_ids:
                final_objects.append(new_tracks[appeared_ids[cur_idx]])
            else:
                final_objects.append(cobj)

        return TrackResult(
            objects=final_objects,
            events=fixed_events,
            correspondences=result.correspondences,
            by_id=dict(self._tracks),
        )

    # ------------------------------------------------------------------
    # Pure match  (no mutation of inputs, no I/O)
    # ------------------------------------------------------------------

    def match(
        self,
        prev: list[TrackedObj],
        cur: list[TrackedObj],
        translations: dict[int, tuple[int, int]] | None = None,
        ignore_colors: set[int] | None = None,
    ) -> TrackResult:
        """Pure: given prev tracks and cur observations, produce correspondences + events.

        Object ids in the returned ``objects`` list will be -1 for APPEARED entries;
        ``update()`` assigns real ids after calling this.
        """
        ignore = ignore_colors if ignore_colors is not None else self.ignore_colors
        events: list[Event] = []
        correspondences: dict[int, int] = {}  # cur_idx -> prev_id

        # Index by color.
        prev_by_color: dict[int, list[TrackedObj]] = {}
        for p in prev:
            prev_by_color.setdefault(p.color, []).append(p)

        cur_by_color: dict[int, list[tuple[int, TrackedObj]]] = {}
        for ci, c in enumerate(cur):
            cur_by_color.setdefault(c.color, []).append((ci, c))

        # Track which prev/cur indices are already consumed.
        used_prev_ids: set[int] = set()
        used_cur_idxs: set[int] = set()

        # ------------------------------------------------------------------
        # Step 2a: SPLIT / MERGE pass (per color, exact cell-set overlap)
        # ------------------------------------------------------------------
        if self.detect_split_merge:
            for color in set(prev_by_color) | set(cur_by_color):
                if color in ignore:
                    continue
                p_list = prev_by_color.get(color, [])
                c_list = cur_by_color.get(color, [])
                if not p_list or not c_list:
                    continue

                # Pre-compute frozensets for overlap checks.
                p_sets = {p.id: frozenset(p.cells) for p in p_list}
                c_sets = {ci: frozenset(c.cells) for ci, c in c_list}

                # MERGE: >=2 prev each overlap SAME cur by >= overlap_thresh of THEIR cells.
                for ci, cobj in c_list:
                    if ci in used_cur_idxs:
                        continue
                    c_set = c_sets[ci]
                    overlapping_prevs = []
                    for p in p_list:
                        if p.id in used_prev_ids:
                            continue
                        p_set = p_sets[p.id]
                        if len(p_set) == 0:
                            continue
                        overlap = len(p_set & c_set) / len(p_set)
                        if overlap >= _OVERLAP_THRESH:
                            overlapping_prevs.append(p)
                    if len(overlapping_prevs) >= 2:
                        # Oldest (smallest id) inherits.
                        inheritor = min(overlapping_prevs, key=lambda p: p.id)
                        others = [p for p in overlapping_prevs if p.id != inheritor.id]
                        correspondences[ci] = inheritor.id
                        used_cur_idxs.add(ci)
                        used_prev_ids.add(inheritor.id)
                        for p in others:
                            used_prev_ids.add(p.id)
                        events.append(Event(
                            kind=MERGED,
                            id=inheritor.id,
                            color=color,
                            extra={"into": inheritor.id, "from": [p.id for p in others]},
                        ))

                # SPLIT: one prev overlaps >=2 cur each by >= overlap_thresh of CUR cells.
                for p in p_list:
                    if p.id in used_prev_ids:
                        continue
                    p_set = p_sets[p.id]
                    overlapping_curs = []
                    for ci, cobj in c_list:
                        if ci in used_cur_idxs:
                            continue
                        c_set = c_sets[ci]
                        if len(c_set) == 0:
                            continue
                        overlap = len(p_set & c_set) / len(c_set)
                        if overlap >= _OVERLAP_THRESH:
                            overlapping_curs.append((ci, cobj))
                    if len(overlapping_curs) >= 2:
                        # Largest fragment keeps prev id.
                        largest = max(overlapping_curs, key=lambda x: x[1].size)
                        others = [(ci, c) for ci, c in overlapping_curs if ci != largest[0]]
                        correspondences[largest[0]] = p.id
                        used_cur_idxs.add(largest[0])
                        used_prev_ids.add(p.id)
                        for ci, _ in others:
                            used_cur_idxs.add(ci)  # will get new ids from update()
                        child_ids = [-1] * len(others)  # placeholders; update() allocates
                        events.append(Event(
                            kind=SPLIT,
                            id=p.id,
                            color=color,
                            extra={"id": p.id, "into": [p.id] + child_ids},
                        ))

        # ------------------------------------------------------------------
        # Step 2b: RECOLORED — same exact footprint, different color
        # ------------------------------------------------------------------
        remaining_prev = [p for p in prev if p.id not in used_prev_ids]
        remaining_cur = [(ci, c) for ci, c in enumerate(cur) if ci not in used_cur_idxs]

        for p in list(remaining_prev):
            p_set = frozenset(p.cells)
            for ci, cobj in list(remaining_cur):
                if cobj.color == p.color:
                    continue  # same color -> handled in normal matching
                c_set = frozenset(cobj.cells)
                if p_set == c_set:
                    # Exact footprint match: recolored.
                    correspondences[ci] = p.id
                    used_prev_ids.add(p.id)
                    used_cur_idxs.add(ci)
                    events.append(Event(
                        kind=RECOLORED,
                        id=p.id,
                        color=cobj.color,
                        extra={"old_color": p.color, "new_color": cobj.color},
                    ))
                    break

        # ------------------------------------------------------------------
        # Steps 1 + 3 + 4: translation/velocity prediction -> greedy 1-1 match
        # ------------------------------------------------------------------
        # Refresh remaining pools after 2a/2b.
        remaining_prev_by_color: dict[int, list[TrackedObj]] = {}
        for p in prev:
            if p.id not in used_prev_ids:
                remaining_prev_by_color.setdefault(p.color, []).append(p)

        remaining_cur_by_color: dict[int, list[tuple[int, TrackedObj]]] = {}
        for ci, c in enumerate(cur):
            if ci not in used_cur_idxs:
                remaining_cur_by_color.setdefault(c.color, []).append((ci, c))

        for color in set(remaining_prev_by_color) & set(remaining_cur_by_color):
            p_list = remaining_prev_by_color[color]
            c_list = remaining_cur_by_color[color]

            # Build cost triples (cost, prev_id, cur_idx).
            triples: list[tuple[float, int, int]] = []
            for p in p_list:
                # Predict centroid using translation prior or velocity.
                if translations is not None and color in translations:
                    tdr, tdc = translations[color]
                    pred_r = p.centroid[0] + tdr
                    pred_c = p.centroid[1] + tdc
                else:
                    pred_r = p.centroid[0] + p.velocity[0]
                    pred_c = p.centroid[1] + p.velocity[1]

                for ci, cobj in c_list:
                    raw_dist = abs(cobj.centroid[0] - p.centroid[0]) + abs(cobj.centroid[1] - p.centroid[1])
                    if raw_dist > _MAX_MATCH_DIST:
                        continue
                    cost = (
                        abs(cobj.centroid[0] - pred_r) + abs(cobj.centroid[1] - pred_c)
                        + _W_SIZE * (p.size != cobj.size)
                    )
                    triples.append((cost, p.id, ci))

            # Greedy assignment: sort by (cost, prev_id, cur_idx) for determinism.
            triples.sort(key=lambda t: (t[0], t[1], t[2]))
            local_used_prev: set[int] = set()
            local_used_cur: set[int] = set()
            for cost, prev_id, ci in triples:
                if prev_id in local_used_prev or ci in local_used_cur:
                    continue
                local_used_prev.add(prev_id)
                local_used_cur.add(ci)
                correspondences[ci] = prev_id
                used_prev_ids.add(prev_id)
                used_cur_idxs.add(ci)

        # ------------------------------------------------------------------
        # Step 5: emit MOVED / CHANGED for matched pairs
        # ------------------------------------------------------------------
        for ci, prev_id in correspondences.items():
            p = next((x for x in prev if x.id == prev_id), None)
            cobj = cur[ci]
            if p is None:
                continue
            ddr = cobj.centroid[0] - p.centroid[0]
            ddc = cobj.centroid[1] - p.centroid[1]

            # Only emit MOVED/CHANGED for non-MERGED/SPLIT/RECOLORED pairs.
            already_special = any(
                ev.id == prev_id and ev.kind in (MERGED, SPLIT, RECOLORED)
                for ev in events
            )
            if already_special:
                continue

            moved = round(ddr) != 0 or round(ddc) != 0
            if moved:
                events.append(Event(
                    kind=MOVED,
                    id=prev_id,
                    color=cobj.color,
                    delta=(ddr, ddc),
                    before=(p.bbox, p.size),
                    after=(cobj.bbox, cobj.size),
                ))

            rigid = _is_rigid_shift(p.cells, cobj.cells, round(ddr), round(ddc))
            size_changed = cobj.size != p.size
            if not rigid or size_changed:
                events.append(Event(
                    kind=CHANGED,
                    id=prev_id,
                    color=cobj.color,
                    delta=(ddr, ddc),
                    size_delta=cobj.size - p.size,
                    before=(p.bbox, p.size),
                    after=(cobj.bbox, cobj.size),
                ))

        # ------------------------------------------------------------------
        # APPEARED events (id=-1 placeholder; update() assigns real ids)
        # ------------------------------------------------------------------
        for ci, cobj in enumerate(cur):
            if ci not in used_cur_idxs:
                events.append(Event(
                    kind=APPEARED,
                    id=-1,
                    color=cobj.color,
                    before=cobj.centroid,   # stored here so update() can correlate
                    after=(cobj.bbox, cobj.size),
                ))

        # ------------------------------------------------------------------
        # VANISHED events for unmatched prev (past occlusion grace)
        # ------------------------------------------------------------------
        for p in prev:
            if p.id not in used_prev_ids:
                new_missed = p.missed + 1
                if new_missed > _OCCLUSION_GRACE:
                    events.append(Event(
                        kind=VANISHED,
                        id=p.id,
                        color=p.color,
                        before=(p.bbox, p.size),
                    ))

        # Build by_id from prev for external reference (update() will rebuild with new).
        by_id = {p.id: p for p in prev}

        return TrackResult(
            objects=list(cur),
            events=events,
            correspondences=correspondences,
            by_id=by_id,
        )
