"""C5 — Goal inference: form/refine a typed goal hypothesis from rewarding transitions.

The naive version of this (diff the frame on which reward arrived) is defeated by the
level-change MASKING: when a level is completed the engine immediately rebuilds the next
level, so the post-reward frame is the *new* board, not the rewarding event. The fix here
is structural — never diff across that swap. Instead we:

  * buffer the per-step events of recent transitions (a small ring), so the rewarding
    event is read from the PRE-swap record captured one step earlier;
  * guard every transition with `_is_board_swap` so a level rebuild contributes ZERO
    events (learn nothing, never the wrong thing);
  * derive events ourselves (`_derive_events`) from the same memoized perception primitives
    the policy already calls, with the proven avatar-footprint/distractor masking, so this
    module is fully self-contained and needs no upstream C1/C2/C3 (C2 events are accepted
    as an optional enrichment param and used verbatim when supplied).

On each level-up we credit candidate hypotheses with a three-tier signature
(role -> color -> shape_sig), accumulate Laplace-smoothed confirm/contradict evidence, and
expose the best hypothesis via `current_goal()` plus an instantiation helper
`goal_target_cells(grid)`. This is MILESTONE 1: pure observation. Nothing here touches the
action path — the policy only *calls* observe_step/on_level_start and *reads* current_goal;
the durable deliverable is the typed GoalHypothesis that C6 (the planner) consumes.

Pure CPU, no policy import. Reuses perception.connected_components (lru_cached),
movement.infer_all_translations, MotionModel.avatar_colors/avatar_centroid,
perception.detect_background.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import movement as MV
from . import perception as P

# --- event kinds (string consts; cheap to compare and log) ----------------------------
EV_VANISH = "VANISH"        # a color's cells went from >0 to 0
EV_APPEAR = "APPEAR"        # a color's cells went from 0 to >0
EV_MOVE_OBJ = "MOVE_OBJ"    # a (non-avatar) color rigidly translated (push / sokoban)
EV_COUNT_UP = "COUNT_UP"    # a color's cell count increased (but stayed >0)
EV_COUNT_DN = "COUNT_DN"    # a color's cell count decreased (but stayed >0)
EV_RECOLOR = "RECOLOR"      # same count, cells differ, no clean translation (toggle/door)
EV_AVATAR_ON = "AVATAR_ON"  # avatar centroid sits on a non-ignored color this frame
EV_CLICK = "CLICK"          # the rewarding action was a click on a color

# --- goal taxonomy --------------------------------------------------------------------
GOAL_KINDS = (
    "VANISH_ALL", "PUSH_TO", "REACH", "RECOLOR_ALL",
    "CLICK_TARGET", "TOGGLE_THEN_REACH", "APPEAR",
)


@dataclass(frozen=True)
class Event:
    kind: str
    color: int
    where: tuple[int, int] | None = None  # (row, col) when meaningful (click / landing)
    size: int = 0


@dataclass
class StepRecord:
    action: tuple | None
    events: tuple[Event, ...]
    avatar_xy: tuple[float, float] | None
    moved: bool


@dataclass
class GoalHypothesis:
    kind: str
    color: int
    aux_color: int | None = None
    shape_sig: tuple | None = None  # (h, w, size) — color-invariant fallback
    role: str | None = None         # stays None until C3 affordance roles are wired
    support: int = 1
    contra: int = 0
    last_level: int = -1

    @property
    def confidence(self) -> float:
        # Laplace-smoothed: a single confirm gives 2/3, two give 3/4, etc.
        return (self.support + 1.0) / (self.support + self.contra + 2.0)


@dataclass
class GoalModel:
    hyps: dict[tuple, GoalHypothesis] = field(default_factory=dict)  # key=(kind,color,aux)
    levelups_seen: int = 0

    def best(self, min_conf: float = 0.6, min_support: int = 1) -> GoalHypothesis | None:
        """Highest-confidence hypothesis above the gate, deterministic tie-break.

        Sorted by (confidence, support) then a stable key so the offline suite stays
        reproducible. A single noisy level-up (conf 2/3 with support 1) only passes when
        min_support is 1; callers can raise min_support to require >=2 confirmations.
        """
        cands = [
            h for h in self.hyps.values()
            if h.confidence >= min_conf and h.support >= min_support
        ]
        if not cands:
            return None
        cands.sort(key=lambda h: (h.confidence, h.support, h.kind, h.color,
                                  h.aux_color if h.aux_color is not None else -1),
                   reverse=True)
        return cands[0]


def _is_board_swap(prev: np.ndarray, cur: np.ndarray, bg: int | None) -> bool:
    """True if cur looks like a freshly-rebuilt board rather than a within-level move.

    Conservative by design: any of shape change, >=60% of cells changed, changed cells
    >= 1.5x the non-background cell count, or a background-color change. On a detected
    swap we emit zero events (worst case: a missed lesson, never a corrupted hypothesis).
    """
    if prev is None or cur is None:
        return True
    if prev.shape != cur.shape:
        return True
    changed = int(np.count_nonzero(prev != cur))
    if changed == 0:
        return False
    size = prev.size
    if changed >= 0.6 * size:
        return True
    # A normal object move changes up to ~2x its own cells (old + new positions), so the
    # "changed vs non-bg" ratio test must require a much larger, structural repaint to avoid
    # flagging ordinary motion as a swap. Gate it on a meaningful absolute floor (sparse
    # scenes never trip it) AND a high multiple, so only a near-total rebuild of a populated
    # board (most non-bg content replaced) qualifies.
    nonbg = int(np.count_nonzero(prev != bg)) if bg is not None else size
    if nonbg >= 32 and changed >= 1.5 * nonbg:
        return True
    if bg is not None and P.detect_background(prev) != P.detect_background(cur):
        return True
    return False


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _shape_sig_for_color(grid: np.ndarray, color: int, bg: int | None) -> tuple | None:
    """(height, width, size) of the largest connected component of `color`, or None."""
    best = None
    for o in P.connected_components(grid, background=bg):
        if o.color != color:
            continue
        if best is None or o.size > best.size:
            best = o
    if best is None:
        return None
    return (best.height, best.width, best.size)


class GoalInference:
    """One per game; persists hypotheses across levels, resets per game.

    M1 contract: this object is OBSERVE-ONLY. The policy calls observe_step/on_level_start
    and reads current_goal()/goal_target_cells(); nothing here selects actions.
    """

    def __init__(self, K: int = 4) -> None:
        self.K = K
        self.reset_all()

    # --- lifecycle ---------------------------------------------------------------------
    def reset_all(self) -> None:
        """New game: drop all learned hypotheses and per-level state."""
        self.model = GoalModel()
        self._buf: deque[StepRecord] = deque(maxlen=self.K)
        self._level_census: dict[int, int] = {}
        self._level = -1

    def on_level_start(self, grid: np.ndarray, bg: int | None, level: int) -> None:
        """Snapshot the per-color census and clear the per-level ring; model persists."""
        self._level = level
        self._buf.clear()
        if grid is None:
            self._level_census = {}
            return
        census = _color_counts(grid)
        if bg is not None:
            census.pop(bg, None)
        self._level_census = census

    # --- observation -------------------------------------------------------------------
    def observe_step(self, *, prev_grid: np.ndarray | None, cur_grid: np.ndarray,
                     prev_action: tuple | None, reward: float, bg: int | None,
                     distractor_colors, avatar, events=None) -> None:
        """Record one transition; on reward, credit hypotheses from buffered pre-swap recs.

        Guards (return early, attribute nothing): missing prev_grid/prev_action. The policy
        is responsible for not calling us across a reset (it nulls prev_action there).
        """
        if prev_grid is None or prev_action is None:
            return

        swap = _is_board_swap(prev_grid, cur_grid, bg)

        # events for THIS transition
        if events is not None:
            ev = tuple(self._coerce_external_events(events))
        elif swap:
            ev = ()  # NEVER diff across a swap — this is the whole fix
        else:
            ev = self._derive_events(prev_grid, cur_grid, bg, distractor_colors, avatar)

        # A click acts on the PRE-transition board (prev_grid), even across a level swap, so
        # the clicked color is always recoverable here. Token convention: ("C", x=col, y=row)
        # per agent.py:44. Append a CLICK event so _hyps_from_record can form CLICK_TARGET
        # even when the rewarding frame is a swap (which carries no derived events).
        if prev_action is not None and len(prev_action) == 3 and prev_action[0] == "C":
            x, y = int(prev_action[1]), int(prev_action[2])
            if 0 <= y < prev_grid.shape[0] and 0 <= x < prev_grid.shape[1]:
                col = int(prev_grid[y, x])
                if col != bg:
                    ev = ev + (Event(EV_CLICK, col, where=(y, x)),)

        avatar_xy = None
        moved = False
        if avatar is not None and getattr(avatar, "ok", False):
            avatar_xy = avatar.avatar_centroid(prev_grid)
            try:
                trans = MV.infer_all_translations(prev_grid, cur_grid, bg)
                moved = any(c in (avatar.avatar_colors or set()) for c in trans)
            except Exception:
                moved = False

        # Single-step level-up (R5): when the rewarding transition is ITSELF the level swap
        # (e.g. collect-the-last-item that completes the level in one action) there is no
        # separate pre-swap step to buffer, so `ev` is empty. Recover the contacted target
        # from the avatar's INTENDED destination on the OLD board: avatar centroid + action
        # delta, then read prev_grid there. This is the safe pre-swap board, never the swap.
        if reward > 0 and swap and not ev and avatar is not None and getattr(avatar, "ok", False):
            dest_color = self._intended_dest_color(prev_grid, prev_action, bg,
                                                   distractor_colors, avatar)
            if dest_color is not None:
                ev = (Event(EV_AVATAR_ON, dest_color),)

        rec = StepRecord(action=prev_action, events=ev, avatar_xy=avatar_xy, moved=moved)

        if reward > 0:
            self._credit_levelup(rec)
            self._buf.clear()
        else:
            self._buf.append(rec)

    def _intended_dest_color(self, prev: np.ndarray, action, bg: int | None,
                             distractor_colors, avatar) -> int | None:
        """Color the avatar would have moved onto on the OLD board (for single-step level-up).

        Uses the learned per-action delta to project the avatar centroid; reads prev_grid at
        the destination. Returns the non-ignored color there, or None.
        """
        if action is None or not action or action[0] != "S":
            return None
        deltas = getattr(avatar, "deltas", None)
        if not deltas or action[1] not in deltas:
            return None
        ac = avatar.avatar_centroid(prev)
        if ac is None:
            return None
        dr, dc = deltas[action[1]]
        r = int(round(ac[0] + dr))
        c = int(round(ac[1] + dc))
        if not (0 <= r < prev.shape[0] and 0 <= c < prev.shape[1]):
            return None
        ignore = set(distractor_colors or set())
        if bg is not None:
            ignore.add(int(bg))
        ignore |= set(getattr(avatar, "avatar_colors", set()) or set())
        col = int(prev[r, c])
        return None if col in ignore else col

    def _coerce_external_events(self, events) -> list[Event]:
        """Accept C2 StepEvents/iterables; keep only what maps cleanly to our Event kinds.

        C2 is optional enrichment; if its shape is unfamiliar we just take any objects that
        already look like our Event (duck-typed kind/color), else fall back to empty.
        """
        out: list[Event] = []
        seq = getattr(events, "events", events)
        try:
            iterator = list(seq)
        except TypeError:
            return out
        for e in iterator:
            kind = getattr(e, "kind", None)
            color = getattr(e, "color", None)
            if isinstance(kind, str) and isinstance(color, int):
                out.append(Event(kind=kind, color=int(color),
                                 where=getattr(e, "where", None),
                                 size=int(getattr(e, "size", 0) or 0)))
        return out

    def _derive_events(self, prev: np.ndarray, cur: np.ndarray, bg: int | None,
                       distractor_colors, avatar) -> tuple[Event, ...]:
        """Self-contained (C2-less) event derivation with avatar-footprint masking."""
        ignore: set[int] = set(distractor_colors or set())
        if bg is not None:
            ignore.add(int(bg))
        if avatar is not None:
            ignore |= set(getattr(avatar, "avatar_colors", set()) or set())

        trans = MV.infer_all_translations(prev, cur, bg)
        pc = _color_counts(prev)
        cc = _color_counts(cur)
        colors = (set(pc) | set(cc)) - ignore

        # Avatar-footprint masking (the proven distractor-masking fix applied to goals):
        # the cells the avatar now occupies were freshly entered this step, so whatever
        # color sat there on `prev` was OCCLUDED by the avatar, not removed by a game event.
        # That contacted color is the REACH target; its count-drop attributable to the
        # footprint is suppressed so "reach goal" isn't misread as "collect/vanish".
        contacted: int | None = None
        footprint_loss: dict[int, int] = {}
        if avatar is not None and getattr(avatar, "ok", False) and prev.shape == cur.shape:
            try:
                cells = avatar.avatar_cells(cur)
            except Exception:
                cells = None
            if cells is not None and len(cells):
                for (r, cc_) in cells:
                    pcol = int(prev[r, cc_])
                    if pcol not in ignore:
                        footprint_loss[pcol] = footprint_loss.get(pcol, 0) + 1
                if footprint_loss:
                    # the dominant occluded color is the contacted object
                    contacted = max(footprint_loss, key=footprint_loss.get)

        out: list[Event] = []
        for c in sorted(colors):
            pmask = prev == c
            amask = cur == c
            if np.array_equal(pmask, amask):
                continue
            pn = int(pc.get(c, 0))
            an = int(cc.get(c, 0))
            if c in trans and trans[c] != (0, 0):
                a_cells = np.argwhere(amask)
                where = None
                if len(a_cells):
                    ctr = a_cells.mean(axis=0)
                    where = (int(round(ctr[0])), int(round(ctr[1])))
                out.append(Event(EV_MOVE_OBJ, c, where=where, size=an))
            elif c == contacted and pn - an <= footprint_loss.get(c, 0):
                # the entire drop is explained by the avatar footprint occluding this color
                # -> a REACH contact, NOT a game-event change. EV_AVATAR_ON emitted below.
                continue
            elif pn > 0 and an == 0:
                out.append(Event(EV_VANISH, c, size=pn))
            elif pn == 0 and an > 0:
                out.append(Event(EV_APPEAR, c, size=an))
            elif an < pn:
                out.append(Event(EV_COUNT_DN, c, size=an))
            elif an > pn:
                out.append(Event(EV_COUNT_UP, c, size=an))
            else:  # same count, cells moved, not a clean translation -> toggle/recolor
                out.append(Event(EV_RECOLOR, c, size=an))

        if contacted is not None:
            out.append(Event(EV_AVATAR_ON, contacted))
        return tuple(out)

    # --- crediting ---------------------------------------------------------------------
    def _credit_levelup(self, reward_rec: StepRecord) -> None:
        self.model.levelups_seen += 1
        # the rewarding step is the most recent buffered record with events, else the
        # current rewarding record itself (e.g. collect: discard + next_level same action).
        rec = reward_rec
        if not rec.events:
            for r in reversed(self._buf):
                if r.events:
                    rec = r
                    break

        cands = self._hyps_from_record(rec)

        applicable_keys = set(cands.keys())
        # confirm / insert
        for key, hyp in cands.items():
            existing = self.model.hyps.get(key)
            if existing is None:
                hyp.last_level = self._level
                self.model.hyps[key] = hyp
            else:
                existing.support += 1
                existing.last_level = self._level
                if existing.shape_sig is None:
                    existing.shape_sig = hyp.shape_sig

        # fair contradiction: only penalise a hypothesis whose precondition (its target
        # color present at this level's start) WAS applicable this level yet wasn't credited.
        for key, hyp in self.model.hyps.items():
            if key in applicable_keys:
                continue
            if self._level_census.get(hyp.color, 0) > 0:
                hyp.contra += 1

        self._prune()

    def _hyps_from_record(self, rec: StepRecord) -> dict[tuple, GoalHypothesis]:
        cands: dict[tuple, GoalHypothesis] = {}

        def add(kind, color, aux=None, shape=None):
            key = (kind, int(color), aux if aux is None else int(aux))
            cands[key] = GoalHypothesis(kind=kind, color=int(color),
                                        aux_color=None if aux is None else int(aux),
                                        shape_sig=shape, support=1, contra=0)

        for e in rec.events:
            if e.kind in (EV_VANISH, EV_COUNT_DN) and self._level_census.get(e.color, 0) >= 1:
                add("VANISH_ALL", e.color)
            elif e.kind == EV_MOVE_OBJ:
                aux = None
                if e.where is not None:
                    aux = self._census_color_at_start(e.where)
                add("PUSH_TO", e.color, aux=aux)
            elif e.kind == EV_AVATAR_ON:
                add("REACH", e.color)
            elif e.kind == EV_RECOLOR:
                add("RECOLOR_ALL", e.color)
            elif e.kind == EV_APPEAR:
                add("APPEAR", e.color)

        # click target: token convention is ("C", x=col, y=row) per agent.py:44
        act = rec.action
        if act is not None and len(act) == 3 and act[0] == "C":
            x, y = int(act[1]), int(act[2])
            col = self._click_color(rec, y, x)
            if col is not None:
                add("CLICK_TARGET", col, shape=None)

        return cands

    def _census_color_at_start(self, where: tuple[int, int]) -> int | None:
        # We only retain the level-start census as counts, not positions; PUSH_TO's aux is
        # best-effort and refined once C3/C6 supply target markers. Return None here so the
        # hypothesis keys on the block color (still a usable PUSH_TO signal for C6).
        return None

    def _click_color(self, rec: StepRecord, row: int, col: int) -> int | None:
        for e in rec.events:
            if e.kind == EV_CLICK:
                return e.color
        return None

    def _prune(self) -> None:
        if self.model.levelups_seen < 2:
            return
        dead = [
            k for k, h in self.model.hyps.items()
            if h.support == 0 or (h.confidence < 0.3 and h.support == 0)
        ]
        for k in dead:
            del self.model.hyps[k]

    # --- public read API consumed by C6/C7 --------------------------------------------
    def current_goal(self, min_conf: float = 0.6, min_support: int = 1) -> GoalHypothesis | None:
        return self.model.best(min_conf=min_conf, min_support=min_support)

    def goal_target_cells(self, grid: np.ndarray, bg: int | None = None,
                          min_conf: float = 0.6, min_support: int = 1) -> list[tuple[int, int]]:
        """Instantiate the best hypothesis on the CURRENT board -> target (row, col) cells.

        Matching is color-first with a shape_sig fallback (color-invariant transfer). For
        C6: VANISH_ALL/REACH/CLICK_TARGET -> matching-color object centroids; PUSH_TO ->
        block centroid(s). Returns [] if there is no confident goal.
        """
        g = self.current_goal(min_conf=min_conf, min_support=min_support)
        if g is None or grid is None:
            return []
        if bg is None:
            bg = P.detect_background(grid)
        objs = P.connected_components(grid, background=bg)

        def centroids_for_color(color: int) -> list[tuple[int, int]]:
            out = []
            for o in objs:
                if o.color == color:
                    out.append((int(round(o.centroid[0])), int(round(o.centroid[1]))))
            return out

        cells = centroids_for_color(g.color)
        if not cells and g.shape_sig is not None:
            h, w, sz = g.shape_sig
            for o in objs:
                if o.height == h and o.width == w and o.size == sz:
                    cells.append((int(round(o.centroid[0])), int(round(o.centroid[1]))))
        return cells
