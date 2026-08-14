"""Perception layer: learned HUD/housekeeping masking + canonical masked-state.

Why a *learned* mask: 18+ of 25 public games paint a step-budget bar into the
frame and tick it BEFORE the legality check (memory
`arcagi3-hud-breaks-frame-identity`), so raw frame identity is broken — but the
hand-built row/col lists in scratchpad/ideas/hud_mask.py do not transfer to
hidden games. This module learns the mask per game from the wiggle battery's
recorded per-step diffs (design doc §2 Layer 1).

The HUD effect is REGIONAL, not per-pixel: as the bar decrements, different
pixels within the line change at different times, so a per-pixel "always
changes" test finds nothing. We therefore classify whole candidate LINES
(rows/cols):

  1. A line is a candidate if it changed on >= max(min_count, min_rate*T) of
     the battery's usable steps (scene-transition steps — diffs covering a
     large frame fraction — are excluded), AND its changes ROVE along the
     line (a bar ticks different pixels over time; a static board object that
     blinks in place changes the same pixels every step and is NOT HUD), AND
     its pixel values never REVISIT earlier values (a budget bar depletes
     monotonically; real board content on a line — paddles, avatars,
     selections — oscillates: measured revisit fractions ~0.5-0.7 on
     bp35/sc25 board rows vs exactly 0 on every true HUD bar).
  2. Backward elimination against an "explained steps" criterion: a step is
     explained if ALL its changed pixels lie inside the union of the kept
     lines (i.e. it is a true no-op once the HUD is removed). A candidate that
     removes no explanatory power is dropped. This is set-based, not greedy
     per-line, because some games (m0r0: rows 0 AND 63) tick two lines in the
     same step — neither line alone explains anything.
  3. If the surviving set explains zero steps there is no housekeeping
     evidence at all -> empty primary mask (re86-class games).
  4. Secondary channel for bars that NEVER tick alone (wa30: row 63 ticks
     every ~5 actions, always during avatar movement, so exclusivity is
     unobservable in a 16-action battery): a line that changed >= min_count
     times, roves, is monotone, AND ticked under >= 3 distinct action ids
     (action-agnostic — requires >= 3 ids probed) is accepted without
     exclusivity evidence.

Everything that compares frames must go through Perception: state hashing,
board_changed, novelty, dead-action memory, graph node identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ("row" | "col", index)
HudLine = tuple[str, int]

FRAME_SHAPE: tuple[int, int] = (64, 64)


# ---------------------------------------------------------------------------
# HUD learning
# ---------------------------------------------------------------------------

def learn_hud_lines(
    diffs: Sequence[np.ndarray],
    shape: tuple[int, int] = FRAME_SHAPE,
    *,
    min_rate: float = 0.25,
    min_count: int = 3,
    big_diff_frac: float = 0.30,
    max_lines: int = 8,
    max_revisit_frac: float = 0.2,
    action_ids: Sequence[int] | None = None,
) -> list[HudLine]:
    """Classify HUD/housekeeping lines from per-step raw change masks.

    diffs: one int array per battery step, either (n, 4) rows of
    [y, x, old_value, new_value] (preferred — enables the revisit filter) or
    (n, 2) rows of [y, x]. RESET steps must already be excluded by the
    caller — a reset redraws the board and is not evidence about
    housekeeping. action_ids: the action id of each step, aligned with
    diffs — enables the action-agnostic secondary channel.
    """
    h, w = shape
    npix = h * w
    if action_ids is not None and len(action_ids) != len(diffs):
        raise ValueError("action_ids must align 1:1 with diffs")
    keep = [i for i, d in enumerate(diffs) if 0 < len(d) <= big_diff_frac * npix]
    usable = [np.asarray(diffs[i]) for i in keep]
    step_ids = [action_ids[i] for i in keep] if action_ids is not None else None
    t_total = len(usable)
    if t_total == 0:
        return []
    has_values = all(d.shape[1] >= 4 for d in usable)

    # pixel value histories (in step order) for the revisit test
    pixel_hist: dict[tuple[int, int], list[int]] = {}
    if has_values:
        for d in usable:
            for y, x, old, new in d[:, :4].tolist():
                hist = pixel_hist.setdefault((y, x), [int(old)])
                hist.append(int(new))

    row_sets = [np.unique(d[:, 0]) for d in usable]
    col_sets = [np.unique(d[:, 1]) for d in usable]
    row_count = np.zeros(h, dtype=int)
    col_count = np.zeros(w, dtype=int)
    for rs, cs in zip(row_sets, col_sets):
        row_count[rs] += 1
        col_count[cs] += 1

    def roves(kind: str, idx: int) -> bool:
        """True if the line's changes spread to fresh pixels over time.

        distinct changed positions must exceed one step's footprint — a
        static blinker has distinct == per-step count; a ticking bar
        accumulates fresh pixels every change."""
        axis, other = (0, 1) if kind == "row" else (1, 0)
        per_step: list[int] = []
        positions: set[int] = set()
        for d in usable:
            on_line = d[d[:, axis] == idx]
            if len(on_line):
                per_step.append(len(on_line))
                positions.update(int(v) for v in on_line[:, other])
        if not per_step:
            return False
        mean_px = sum(per_step) / len(per_step)
        return len(positions) >= max(mean_px * 1.5, mean_px + 2)

    def monotone(kind: str, idx: int) -> bool:
        """True if the line's pixels never return to an earlier value.

        A step-budget bar depletes monotonically (measured revisit fraction 0
        on every hand-masked HUD line); board content on a line — paddles,
        selections, animations — oscillates (~0.5-0.7 on bp35/sc25)."""
        if not has_values:
            return True
        transitions = revisits = 0
        for (y, x), hist in pixel_hist.items():
            if (y if kind == "row" else x) != idx:
                continue
            for i in range(1, len(hist)):
                transitions += 1
                if hist[i] in hist[:i]:
                    revisits += 1
        if transitions == 0:
            return False
        return revisits / transitions <= max_revisit_frac

    thresh = max(min_count, math.ceil(min_rate * t_total))
    cands: list[HudLine] = [
        ("row", int(r)) for r in np.nonzero(row_count >= thresh)[0]
        if roves("row", int(r)) and monotone("row", int(r))
    ]
    cands += [
        ("col", int(c)) for c in np.nonzero(col_count >= thresh)[0]
        if roves("col", int(c)) and monotone("col", int(c))
    ]

    def count_of(line: HudLine) -> int:
        kind, idx = line
        return int(row_count[idx] if kind == "row" else col_count[idx])

    def explained(lines: Iterable[HudLine]) -> int:
        rows = {i for k, i in lines if k == "row"}
        cols = {i for k, i in lines if k == "col"}
        n = 0
        for d in usable:
            covered = np.isin(d[:, 0], list(rows)) | np.isin(d[:, 1], list(cols))
            n += int(covered.all())
        return n

    kept = list(cands)
    changed = True
    while changed:
        changed = False
        base = explained(kept)
        # weakest evidence first, so strong lines are pruned last
        for line in sorted(kept, key=count_of):
            trial = [x for x in kept if x != line]
            if explained(trial) == base:
                kept = trial
                changed = True
                break

    primary = kept if kept and explained(kept) > 0 else []

    # secondary channel: action-agnostic monotone tickers with no
    # exclusivity evidence (see module docstring, criterion 4)
    secondary: list[HudLine] = []
    if step_ids is not None and len(set(step_ids)) >= 3:
        def changed_ids(kind: str, idx: int) -> set[int]:
            sets = row_sets if kind == "row" else col_sets
            return {step_ids[t] for t, s in enumerate(sets) if idx in s}

        pool = [("row", int(r)) for r in np.nonzero(row_count >= min_count)[0]]
        pool += [("col", int(c)) for c in np.nonzero(col_count >= min_count)[0]]
        for kind, idx in pool:
            line = (kind, idx)
            if line in primary or not (roves(kind, idx) and monotone(kind, idx)):
                continue
            if len(changed_ids(kind, idx)) >= 3:
                secondary.append(line)

    result = primary + secondary
    if not result:
        return []
    result.sort(key=count_of, reverse=True)
    return sorted(result[:max_lines])


def lines_to_mask(lines: Iterable[HudLine], shape: tuple[int, int] = FRAME_SHAPE) -> np.ndarray:
    """Boolean mask, True where the frame carries HUD rather than board."""
    m = np.zeros(shape, dtype=bool)
    for kind, idx in lines:
        if kind == "row":
            m[idx, :] = True
        elif kind == "col":
            m[:, idx] = True
        else:  # pragma: no cover - guarded by the HudLine type
            raise ValueError(f"unknown line kind {kind!r}")
    return m


# ---------------------------------------------------------------------------
# Frames and canonical state
# ---------------------------------------------------------------------------

def settled_frame(obs: object) -> np.ndarray:
    """Last layer of the returned frame stack (animation runs to 61 layers on
    g50t; consuming anything but the settled layer is the duck's verified
    frame[-1] defect done right)."""
    a = np.asarray(obs.frame)  # type: ignore[attr-defined]
    return a[-1] if a.ndim == 3 else a


@dataclass(frozen=True)
class MaskedState:
    """Canonical hashable board identity: the masked frame's bytes.

    Built with `.tobytes()` on a real contiguous array — never `str()` on a
    numpy frame, which truncates with '...' and silently collides every hash.
    """

    stem: str
    key: bytes

    def __hash__(self) -> int:
        return hash((self.stem, self.key))


class Perception:
    """Frame -> masked-state for one game, given learned HUD lines."""

    def __init__(
        self,
        stem: str,
        hud_lines: Sequence[HudLine],
        shape: tuple[int, int] = FRAME_SHAPE,
        fill: int = 0,
    ) -> None:
        self.stem = stem
        self.hud_lines: list[HudLine] = sorted(hud_lines)
        self.shape = shape
        self.fill = fill
        self._mask = lines_to_mask(self.hud_lines, shape)
        self._board = ~self._mask

    @property
    def hud_mask(self) -> np.ndarray:
        return self._mask.copy()

    @property
    def board_mask(self) -> np.ndarray:
        return self._board.copy()

    def mask_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame with HUD pixels flattened to a constant, so hashes ignore the
        counter."""
        if not self._mask.any():
            return frame
        out = frame.copy()
        out[self._mask] = self.fill
        return out

    def frames_equal(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Board-equality — the comparison `board_changed` should have been
        doing."""
        if not self._mask.any():
            return bool(np.array_equal(a, b))
        return bool(np.array_equal(a[self._board], b[self._board]))

    def state_key(self, frame: np.ndarray) -> bytes:
        return np.ascontiguousarray(self.mask_frame(frame), dtype=np.int8).tobytes()

    def masked_state(self, frame: np.ndarray) -> MaskedState:
        return MaskedState(stem=self.stem, key=self.state_key(frame))


if __name__ == "__main__":  # standalone self-check on synthetic diffs
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # 12 steps of [y, x, old, new] diffs: row 63 ticks every step (2 px,
    # roving, monotone -> HUD); a static 2-px object blinks in place on half
    # the steps (fails rove); a paddle oscillates along row 30 on most steps
    # (roves, but revisits values -> fails monotone); an avatar blob moves in
    # 2-D on 4 steps (never reaches candidate rate on any single line).
    diffs = []
    for t in range(12):
        pts = [[63, (5 * t) % 64, 3, 0], [63, (5 * t + 1) % 64, 3, 0]]  # HUD
        if t % 2 == 0:
            v = 7 if (t // 2) % 2 == 0 else 2
            pts += [[40, 8, 9 - v, v], [40, 9, 9 - v, v]]  # static blinker
        if t % 3 == 1:
            p = 10 + 2 * (t % 5)
            pts += [[30, p, 5, 0], [30, p + 2, 0, 5]]  # oscillating paddle
        if t % 3 == 0:
            r, c = 20 + t % 4, 28 + (t // 3) % 4
            pts += [[r, c, 0, 2], [r + 1, c, 0, 2], [r, c + 1, 2, 0]]  # avatar
        diffs.append(np.array(pts, dtype=int))
    lines = learn_hud_lines(diffs)
    assert lines == [("row", 63)], lines
    p = Perception("test", lines)
    f1 = np.zeros(FRAME_SHAPE, dtype=np.int8)
    f2 = f1.copy()
    f2[63, 10] = 7  # HUD tick only
    assert p.frames_equal(f1, f2) and p.masked_state(f1) == p.masked_state(f2)
    f3 = f1.copy()
    f3[10, 10] = 3  # board change
    assert not p.frames_equal(f1, f3)
    print("perception self-check OK:", lines)
