"""T1/T2 effect models with accuracy gates (design doc §2 Layer 3, Stage 2a).

The Stage-1b verdict: T0's exact graph wins levels but pays a blind frontier
sweep — per-state re-discovery of what a rule could predict (ft09: 3189/3561
tried edges were self-loops). This module compresses observed transitions into
per-action rules so the planner can (a) stop probing pairs whose outcome is
already known, and (b) route through predicted, never-visited states.

Rule families
-------------
T1 (per non-click action id, scopes A1-A5/A7):
  * TranslationRule — the moving blob (a learned color set) translates by a
    fixed (dy, dx); vacated cells take a learned fill color. Predicts only
    when the target cells are clear (conservative: blocked moves decline
    rather than guess).
  * BlobRelocateRule — the SELF blob (colors C) relocates by a fixed (dy, dx)
    and ends in a CANONICAL post-action layout. Strictly more expressive than
    translation: it covers move-with-reorientation (measured on tu93 — the
    avatar's facing pixel relocates inside the blob on direction changes, so
    104/120 real moves are inexpressible as pure translations).
  * ColorMapRule — a global value mapping (toggle/recolor patterns): dst is
    exactly lut(src) for a learned per-color lut.
  * ConstantDiffRule — a fixed positional write: the same (cells, new values)
    every time the action fires.
T2 (per clicked color — the generalization over ACTION6 coordinates):
  * ClickRecolorRule — clicking any cell of color c recolors the clicked
    connected component to a learned c'. Learned at some coordinates, applies
    at all coordinates of that color.
  * ClickNullRule — clicking color c never changes the board. Feeds the menu
    (predicted-dead clicks go last), not the edge set.

Stage 2b additions:
  * MoveBlockedRule (T1) — per-direction blocked-POSITION predicate for the
    tu93/re86/tr87/wa30 class where legality is memory, not layout (invisible
    fences: the frame shows nothing at the blocked cell). Learns the mover
    blob's colors from successful relocations, then keys null outcomes by the
    blob's bbox anchor: a position observed blocked >= min_fit times (and
    never moved-from) predicts null FROM ANY BOARD STATE with the blob there
    — exactly the generalization T0 cannot make (T0 re-probes the fence at
    every new state).
  * WinRule (game-level, scope per action id) — cross-level win-condition
    predicate learned from observed win transitions (the graph is per-level;
    this is the only cross-level channel besides T1/T2). Signature = the
    action id, plus the clicked masked color for ACTION6; precondition = the
    color-presence envelope of observed win sources (colors present at every
    win must be present; colors never seen at any win source must be absent).
    Serves candidate (state, action) pairs so commit-mode can target
    PREDICTED wins on levels with no observed win edge; every attempt is
    verified on execution and a hard failure cap kills a wrong predicate.

Held-out accuracy: prequential (test-THEN-train)
------------------------------------------------
Every incoming transition in a rule's scope is first used to score the rule's
current prediction (the rule has never seen this transition — a true held-out
test), and only then to train it. Gating is over a sliding window of the most
recent outcomes:

    gated  :=  window has >= min_evals outcomes  AND  window accuracy >= 90%

so a rule is re-validated continuously as new transitions arrive and degrades
gracefully: accuracy decay un-gates it without human intervention.

Demotion: a LIVE misprediction (the planner executed a predicted edge and the
env disagreed) clears the rule's window immediately — it must re-earn the gate
from fresh evidence — and flags the source state as SUSPECT: no rule ever
predicts from a suspect state again (that state is handed back to the exact
Markov machinery, which now holds the true observed edge anyway).

Fitting is by majority evidence, not first-pair-wins: each family keeps a
tally of candidate hypotheses across observed pairs and predicts with the
majority one (>= min_fit supporters). Pairs no hypothesis explains are simply
counted against the family's accuracy by the prequential window — real games
mix rule-following transitions with exceptions (item pickups, trails), and a
first measurement showed a reset-on-contradiction policy killing every T1
family on tu93 within 50 transitions. The window is the sole gatekeeper; a
family below 90% simply never feeds the planner.

Scope note (deliberate deviation, reported): the design doc resets model +
buffer per level (goose's W1 escape, aimed at CNNs). Rule tiers instead
persist across levels within a game — the prequential window plus live
demotion already provide per-level re-validation, and a hard reset would
discard exactly the cross-level regularity (movement deltas, click mappings)
that rules exist to capture. The GRAPH remains strictly per-level.

Everything operates on HUD-masked frames; nothing here ever touches an env.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.graph import ActionKey
from engineered.perception import Perception

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

_LUT_SIZE = 256  # frame values are small non-negative ints (ARC palette)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EffectConfig:
    window: int = 40          # sliding prequential window length
    min_evals: int = 10       # outcomes required before the gate can open
    gate_acc: float = 0.90    # the design's accuracy gate
    min_fit: int = 3          # supporting pairs before a hypothesis predicts
    # live failures before a rule dies for the game. Prequential accuracy is
    # measured on EXECUTED transitions; the planner spends predictions on
    # UNTRIED pairs — a rule can be right about the executed distribution and
    # systematically wrong exactly where it is used (measured on sb26: 52
    # live failures at 0.88 prequential). The cap ends that cycle.
    max_live_mispred: int = 5
    edge_penalty: float = 0.25   # predicted-edge cost = 1 + this
    max_pred_depth: int = 8      # predicted-chain hops beyond the visited set
    max_pred_nodes: int = 1500   # predicted expansions per plan call
    max_shift: int = 12          # translation search radius
    max_blob_ratio: float = 3.0  # decline translation if blob grew this much
    # Stage 2b — the ft09 fix: every audit_every-th frontier plan MUST probe
    # the cheapest rule-predicted untried pair. Without this the effect-guided
    # sweep is a de-facto PRUNE: on a game whose unpredicted frontier grows
    # forever (ft09 L1: c=9 clicks change the board 97% of the time, spawning
    # fresh states each step), goal 2c is never reached and a win behind a
    # gated click_null color is starved for the whole budget (measured:
    # s2a ft09 L1 = 3732 actions, 0 wins; T0-control found it).
    audit_every: int = 8
    # Stage 2b — win predicate
    win_min_obs: int = 2         # observed wins before the predicate gates
    max_win_failures: int = 5    # live failed attempts before it dies


# ---------------------------------------------------------------------------
# Rule base: prequential stats + gate
# ---------------------------------------------------------------------------

class Rule:
    """One hypothesis for one scope, with prequential gate bookkeeping."""

    family: str = "?"
    # prediction precedence among gated rules (higher first, before window
    # accuracy). Position-specific MEMORY must trump generic dynamics: a
    # gated TranslationRule happily predicts a move THROUGH a learned fence
    # (the frame shows clear cells), while MoveBlockedRule knows better.
    priority: int = 0

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        self.scope = scope
        self.cfg = cfg
        self.window: deque[int] = deque(maxlen=cfg.window)
        self.n_scored = 0         # lifetime held-out outcomes
        self.n_correct = 0        # lifetime held-out correct
        self.transitions_seen = 0  # scope transitions consumed (train + test)
        self.gated_at: int | None = None  # transitions_seen when first gated
        self.live_mispredictions = 0
        self.dead = False

    # -- gate --------------------------------------------------------------

    @property
    def gated(self) -> bool:
        if self.dead or len(self.window) < self.cfg.min_evals:
            return False
        return sum(self.window) / len(self.window) >= self.cfg.gate_acc

    @property
    def accuracy(self) -> float | None:
        return (self.n_correct / self.n_scored) if self.n_scored else None

    @property
    def window_accuracy(self) -> float | None:
        return (sum(self.window) / len(self.window)) if self.window else None

    def note_outcome(self, ok: bool) -> None:
        self.window.append(1 if ok else 0)
        self.n_scored += 1
        self.n_correct += int(ok)
        if self.gated and self.gated_at is None:
            self.gated_at = self.transitions_seen

    def demote(self) -> None:
        """Live misprediction: the gate closes NOW; re-earn from scratch.
        A rule that keeps failing live despite re-earning is systematically
        wrong where the planner uses it — it dies at the cap."""
        self.live_mispredictions += 1
        self.window.clear()
        if self.live_mispredictions >= self.cfg.max_live_mispred:
            self.dead = True

    # -- family interface --------------------------------------------------

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        """Predicted successor frame, or None to decline. An identity return
        (equal to `frame`) is a deliberate null prediction."""
        raise NotImplementedError  # pragma: no cover - overridden

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        raise NotImplementedError  # pragma: no cover - overridden

    def stats(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "scope": self.scope,
            "transitions": self.transitions_seen,
            "scored": self.n_scored,
            "accuracy": round(self.accuracy, 4) if self.n_scored else None,
            "window_accuracy": (round(self.window_accuracy, 4)
                                if self.window else None),
            "gated": self.gated,
            "gated_at": self.gated_at,
            "live_mispredictions": self.live_mispredictions,
            "dead": self.dead,
        }


# ---------------------------------------------------------------------------
# T1: translation
# ---------------------------------------------------------------------------

def _fit_translation(src: np.ndarray, dst: np.ndarray,
                     max_shift: int) -> tuple[int, int, frozenset[int], int] | None:
    """Explain dst as src with one pixel set translated by (dy, dx).

    Returns (dy, dx, moving_colors, fill) or None. For each candidate shift
    and fill value, the moving set is derived from the evidence only:
    vacated cells (changed, now showing fill) plus the pre-images of occupied
    cells (changed, now showing object) — never from unchanged pixels, which
    a looser heuristic would wrongly drag along (measured on tu93: a static
    structure one shift behind the avatar broke every fit). The derived set
    must reproduce dst exactly.
    """
    changed = src != dst
    if not changed.any():
        return None
    ys, xs = np.nonzero(changed)
    span_y = int(ys.max() - ys.min())
    span_x = int(xs.max() - xs.min())
    h, w = src.shape
    bg = int(np.bincount(src.ravel()).argmax())  # board-majority color
    fills = [int(v) for v in np.unique(dst[changed])]
    for dy in range(-min(span_y, max_shift), min(span_y, max_shift) + 1):
        for dx in range(-min(span_x, max_shift), min(span_x, max_shift) + 1):
            if dy == 0 and dx == 0:
                continue
            for fill in fills:
                vac = changed & (dst == fill)
                occ = changed & (dst != fill)
                if not vac.any() or not occ.any():
                    continue
                moving = vac.copy()
                oy, ox = np.nonzero(occ)
                py, px = oy - dy, ox - dx
                if ((py < 0) | (py >= h) | (px < 0) | (px >= w)).any():
                    continue
                moving[py, px] = True
                if (src[moving] == fill).any():
                    continue  # a "moving" pixel showing fill is no object
                out = _apply_translation(src, moving, dy, dx, fill)
                if out is not None and np.array_equal(out, dst):
                    colors = frozenset(int(c) for c in np.unique(src[moving]))
                    if colors == {bg}:
                        # degenerate dual: "a patch of background moved the
                        # other way" also reproduces dst but generalizes as
                        # nonsense — the object is never the majority color
                        continue
                    return dy, dx, colors, fill
    return None


def _apply_translation(src: np.ndarray, moving: np.ndarray, dy: int, dx: int,
                       fill: int) -> np.ndarray | None:
    ys, xs = np.nonzero(moving)
    ny, nx = ys + dy, xs + dx
    h, w = src.shape
    if (ny < 0).any() or (ny >= h).any() or (nx < 0).any() or (nx >= w).any():
        return None
    vals = src[ys, xs]
    out = src.copy()
    out[ys, xs] = fill
    out[ny, nx] = vals
    return out


class TranslationRule(Rule):
    family = "translation"

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        super().__init__(scope, cfg)
        # (dy, dx, fill) -> {"n": supporters, "colors": set, "max_px": int}
        self._cand: dict[tuple[int, int, int], dict] = {}
        self._best: tuple[int, int, int] | None = None

    def _refresh_best(self) -> None:
        best = None
        for key, v in self._cand.items():
            if v["n"] < self.cfg.min_fit:
                continue
            if best is None or (v["n"], key) > (self._cand[best]["n"], best):
                best = key
        self._best = best

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        if self._best is None:
            return None
        dy, dx, fill = self._best
        hyp = self._cand[self._best]
        moving = np.isin(frame, list(hyp["colors"]))
        n = int(moving.sum())
        if n == 0 or n > hyp["max_px"] * self.cfg.max_blob_ratio:
            return None  # blob missing or implausibly grown (color reused)
        ys, xs = np.nonzero(moving)
        ny, nx = ys + dy, xs + dx
        h, w = frame.shape
        if (ny < 0).any() or (ny >= h).any() or (nx < 0).any() or (nx >= w).any():
            return None
        # conservative: only predict when the leading cells are clear fill
        lead = np.zeros_like(moving)
        lead[ny, nx] = True
        lead &= ~moving
        if (frame[lead] != fill).any():
            return None
        out = _apply_translation(frame, moving, dy, dx, fill)
        if out is None or np.array_equal(out, frame):
            return None
        return out

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead or np.array_equal(frame, dst):
            return  # null transitions carry no translation evidence
        fit = _fit_translation(frame, dst, self.cfg.max_shift)
        if fit is None:
            return  # unexplained pair: the prequential window judges it
        dy, dx, colors, fill = fit
        c = self._cand.setdefault((dy, dx, fill),
                                  {"n": 0, "colors": set(), "max_px": 0})
        c["n"] += 1
        c["colors"] |= set(colors)
        c["max_px"] = max(c["max_px"],
                          int(np.isin(frame, list(c["colors"])).sum()))
        self._refresh_best()


# ---------------------------------------------------------------------------
# T1: blob relocation with canonical post-action layout
# ---------------------------------------------------------------------------

def _moving_component(field: np.ndarray, changed: np.ndarray) -> np.ndarray | None:
    """The single connected component of `field` intersecting `changed`;
    None when zero or several do (static same-color structures elsewhere on
    the board must stay out of the blob — measured on tu93, folding them in
    collapsed every fit to delta (0,0) with never-repeating layouts)."""
    lab, n = ndimage.label(field, structure=_S4)
    ids = np.unique(lab[changed & field])
    ids = ids[ids != 0]
    if len(ids) != 1:
        return None
    return lab == ids[0]


def _fit_blob_relocate(src: np.ndarray, dst: np.ndarray) -> tuple | None:
    """Explain dst as: ONE connected blob of colors C vanishes to the board
    background and reappears with a fixed layout at a bbox shifted by
    (dy, dx). Returns (C, delta, bg, layout, mask, blob_px) or None."""
    changed = src != dst
    if not changed.any():
        return None
    # the erase color is whatever the vacated cells EXPOSE — not the global
    # board mode (tu93's majority color is not its corridor color)
    for bg in (int(v) for v in np.unique(dst[changed])):
        cs = set(src[changed].tolist()) | set(dst[changed].tolist())
        cs.discard(bg)
        if not cs:
            continue
        colors = sorted(int(c) for c in cs)
        b_src = _moving_component(np.isin(src, colors), changed)
        b_dst = _moving_component(np.isin(dst, colors), changed)
        if b_src is None or b_dst is None:
            continue
        out = src.copy()
        out[b_src] = bg
        out[b_dst] = dst[b_dst]
        if not np.array_equal(out, dst):
            continue  # something beyond the blob changed
        sy, sx = np.nonzero(b_src)
        ty, tx = np.nonzero(b_dst)
        delta = (int(ty.min()) - int(sy.min()), int(tx.min()) - int(sx.min()))
        y0, x0 = int(ty.min()), int(tx.min())
        h = int(ty.max()) - y0 + 1
        w = int(tx.max()) - x0 + 1
        layout = dst[y0:y0 + h, x0:x0 + w].copy()
        mask = b_dst[y0:y0 + h, x0:x0 + w].copy()
        # colors this successful move swept over (blob erased, pre-stamp):
        # evidence of TRAVERSABLE terrain for the clearance test
        erased = src.copy()
        erased[b_src] = bg
        rect = erased[min(int(sy.min()), y0):max(int(sy.max()) + 1, y0 + h),
                      min(int(sx.min()), x0):max(int(sx.max()) + 1, x0 + w)]
        travers = frozenset(int(v) for v in np.unique(rect))
        return (frozenset(colors), delta, bg, layout, mask,
                (int(b_src.sum()), int(b_dst.sum())), travers)
    return None


class BlobRelocateRule(Rule):
    family = "blob_move"

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        super().__init__(scope, cfg)
        # vote key -> {"n", "colors", "delta", "bg", "layout", "mask", "max_px"}
        self._cand: dict[tuple, dict] = {}
        self._best: tuple | None = None

    def _refresh_best(self) -> None:
        best = None
        for key, v in self._cand.items():
            if v["n"] < self.cfg.min_fit:
                continue
            if best is None or (v["n"], key) > (self._cand[best]["n"], best):
                best = key
        self._best = best

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        if self._best is None:
            return None
        hyp = self._cand[self._best]
        lo, hi = hyp["px_range"]
        lab, n = ndimage.label(np.isin(frame, sorted(hyp["colors"])),
                               structure=_S4)
        cand_ids = [i for i in range(1, n + 1)
                    if lo <= int((lab == i).sum()) <= hi]
        if len(cand_ids) != 1:
            return None  # zero or ambiguous SELF candidates: decline
        blob = lab == cand_ids[0]
        dy, dx = hyp["delta"]
        ys, xs = np.nonzero(blob)
        y0, x0 = int(ys.min()) + dy, int(xs.min()) + dx
        layout, mask, bg = hyp["layout"], hyp["mask"], hyp["bg"]
        h, w = layout.shape
        if y0 < 0 or x0 < 0 or y0 + h > frame.shape[0] or x0 + w > frame.shape[1]:
            return None
        out = frame.copy()
        out[blob] = bg
        # swept-corridor clearance: the rectangle spanning source AND target
        # bboxes may only contain colors that supporting (successful) moves
        # are KNOWN to sweep over. A landing-zone-only test predicts moves
        # through walls (measured on tu93: every wrong prediction was
        # predicted-move-but-null with a wall in the 6-px hop); a bg-only
        # test declines every legal move (tu93 corridors carry color-2
        # marker strips). Traversability is learned, not assumed.
        ry0 = min(int(ys.min()), y0)
        rx0 = min(int(xs.min()), x0)
        ry1 = max(int(ys.max()) + 1, y0 + h)
        rx1 = max(int(xs.max()) + 1, x0 + w)
        rect_colors = {int(v) for v in np.unique(out[ry0:ry1, rx0:rx1])}
        if not rect_colors <= hyp["traversable"]:
            return None  # unknown terrain in the path: decline, not guess
        region = out[y0:y0 + h, x0:x0 + w]
        if (region[mask] != bg).any():
            return None  # landing cells must expose the erase color
        region[mask] = layout[mask]
        if np.array_equal(out, frame):
            return None
        return out

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead or np.array_equal(frame, dst):
            return
        fit = _fit_blob_relocate(frame, dst)
        if fit is None:
            return  # unexplained pair: the prequential window judges it
        colors, delta, bg, layout, mask, (px_s, px_d), travers = fit
        key = (delta, bg, layout.shape,
               layout.tobytes(), mask.tobytes(), tuple(sorted(colors)))
        c = self._cand.setdefault(key, {
            "n": 0, "colors": set(colors), "delta": delta, "bg": bg,
            "layout": layout, "mask": mask,
            "px_range": (min(px_s, px_d), max(px_s, px_d)),
            "traversable": set(),
        })
        c["n"] += 1
        lo, hi = c["px_range"]
        c["px_range"] = (min(lo, px_s, px_d), max(hi, px_s, px_d))
        c["traversable"] |= set(travers)
        self._refresh_best()


# ---------------------------------------------------------------------------
# T1: global color map (toggle / recolor patterns)
# ---------------------------------------------------------------------------

class ColorMapRule(Rule):
    family = "colormap"

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        super().__init__(scope, cfg)
        # old color -> Counter of observed new colors (majority wins)
        self._votes: dict[int, "Counter[int]"] = {}
        self.mapping: dict[int, int] = {}

    def _rebuild(self) -> None:
        self.mapping = {}
        for old, ctr in self._votes.items():
            new, n = max(ctr.items(), key=lambda kv: (kv[1], kv[0]))
            if n >= self.cfg.min_fit and new != old:
                self.mapping[old] = new

    def _lut(self) -> np.ndarray:
        lut = np.arange(_LUT_SIZE, dtype=np.int16)
        for old, new in self.mapping.items():
            lut[old] = new
        return lut

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        if not self.mapping:
            return None
        out = self._lut()[frame]
        if np.array_equal(out, frame):
            return None  # no domain color present -> nothing to say
        return out

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead:
            return
        changed = frame != dst
        if not changed.any():
            return  # null pair: the prequential window judges the rule
        olds = frame[changed]
        news = dst[changed]
        pair_map: dict[int, int] = {}
        for o, n in zip(olds.tolist(), news.tolist()):
            if pair_map.setdefault(o, n) != n:
                return  # not a color-map pair; no vote
        # the pair map must explain this pair GLOBALLY (all instances of a
        # domain color changed, everything else untouched)
        lut = np.arange(_LUT_SIZE, dtype=np.int16)
        for old, new in pair_map.items():
            lut[old] = new
        if not np.array_equal(lut[frame], dst):
            return
        for old, new in pair_map.items():
            self._votes.setdefault(old, Counter())[new] += 1
        self._rebuild()


# ---------------------------------------------------------------------------
# T1: constant positional write
# ---------------------------------------------------------------------------

class ConstantDiffRule(Rule):
    family = "constdiff"

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        super().__init__(scope, cfg)
        # (cells, values) signature -> supporters (majority wins)
        self._votes: "Counter[tuple]" = Counter()
        self._best: tuple | None = None

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        if self._best is None:
            return None
        cells, values = self._best
        out = frame.copy()
        for (y, x), v in zip(cells, values):
            out[y, x] = v
        if np.array_equal(out, frame):
            return None
        return out

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead:
            return
        changed = np.argwhere(frame != dst)
        if len(changed) == 0:
            return
        cells = tuple(sorted((int(y), int(x)) for y, x in changed))
        values = tuple(int(dst[y, x]) for y, x in cells)
        self._votes[(cells, values)] += 1
        sig, n = max(self._votes.items(), key=lambda kv: (kv[1], repr(kv[0])))
        self._best = sig if n >= self.cfg.min_fit else None


# ---------------------------------------------------------------------------
# T1: blocked-move position predicate (Stage 2b)
# ---------------------------------------------------------------------------

class MoveBlockedRule(Rule):
    """Per-direction blocked (position) predicate learned from failed moves.

    tu93/re86/tr87/wa30 pattern: a move action returns a null transition at
    specific avatar positions regardless of what the rest of the board shows
    (invisible fences — legality is memory, not layout). The predicate:

        blocked(p, A)  :=  action A observed null >= min_fit times with the
                           mover blob anchored at p, and never observed to
                           move from p

    predicts identity (null) from ANY board state where the mover sits at p.
    The mover is identified from successful `_fit_blob_relocate` fits (colors
    + pixel-size range, majority vote); with zero or several candidate blobs
    on a frame the rule declines. A position later observed to move (a door
    opened) stops predicting immediately, and the prequential window de-gates
    the rule if blockedness turns out to be state- rather than
    position-driven.
    """

    family = "move_blocked"
    priority = 1  # fence memory outranks generic movement dynamics

    def __init__(self, scope: str, cfg: EffectConfig) -> None:
        super().__init__(scope, cfg)
        # frozenset(colors) -> {"n": supporters, "lo": px, "hi": px}
        self._color_votes: dict[frozenset[int], dict] = {}
        self._mover: frozenset[int] | None = None
        self.blocked: Counter = Counter()   # (y0, x0) -> null observations
        self.moved: Counter = Counter()     # (y0, x0) -> successful moves

    def _refresh_mover(self) -> None:
        best = None
        for colors, v in self._color_votes.items():
            if v["n"] < self.cfg.min_fit:
                continue
            if best is None or (v["n"], sorted(colors)) > \
                    (self._color_votes[best]["n"], sorted(best)):
                best = colors
        self._mover = best

    def _anchor(self, frame: np.ndarray) -> tuple[int, int] | None:
        """bbox min-corner of the UNIQUE mover-colored component within the
        learned pixel-size range; None when zero or ambiguous."""
        if self._mover is None:
            return None
        v = self._color_votes[self._mover]
        lab, n = ndimage.label(np.isin(frame, sorted(self._mover)),
                               structure=_S4)
        hits = []
        for i in range(1, n + 1):
            px = int((lab == i).sum())
            if v["lo"] <= px <= v["hi"]:
                hits.append(i)
                if len(hits) > 1:
                    return None
        if len(hits) != 1:
            return None
        ys, xs = np.nonzero(lab == hits[0])
        return (int(ys.min()), int(xs.min()))

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        p = self._anchor(frame)
        if p is None:
            return None
        if self.blocked[p] >= self.cfg.min_fit and self.moved[p] == 0:
            return frame  # identity = deliberate null prediction
        return None

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead:
            return
        if np.array_equal(frame, dst):
            p = self._anchor(frame)
            if p is not None:
                self.blocked[p] += 1
            return
        fit = _fit_blob_relocate(frame, dst)
        if fit is None:
            return  # unexplained non-null pair: the window judges it
        colors, _delta, _bg, _layout, _mask, (px_s, px_d), _trav = fit
        v = self._color_votes.setdefault(
            frozenset(colors), {"n": 0, "lo": min(px_s, px_d),
                                "hi": max(px_s, px_d)})
        v["n"] += 1
        v["lo"] = min(v["lo"], px_s, px_d)
        v["hi"] = max(v["hi"], px_s, px_d)
        self._refresh_mover()
        p = self._anchor(frame)
        if p is not None:
            self.moved[p] += 1


# ---------------------------------------------------------------------------
# Game-level: win-condition predicate (Stage 2b)
# ---------------------------------------------------------------------------

class WinRule(Rule):
    """Cross-level win-signature predicate, scope per action id.

    Learned from observed win (level-up) transitions — the one regularity the
    per-level graph cannot carry across levels. Gates once >= win_min_obs
    wins share the signature (same action id; for ACTION6 also the same
    clicked masked color). The precondition is the color-presence envelope of
    the observed win sources; a level state with a color no win source ever
    showed (or missing a color every win source had) is declined —
    conservative on novel levels.

    This rule never predicts frames. It serves candidate (state, action)
    pairs via `candidates()`; the planner targets them as predicted wins and
    the agent verifies on execution: `attempt_result(False)` counts toward a
    hard failure cap (a predicate that keeps not winning is wrong — dead), a
    success feeds the prequential window like any verified prediction.
    """

    family = "win_sig"

    def __init__(self, scope: str, cfg: EffectConfig, aid: int) -> None:
        super().__init__(scope, cfg)
        self.aid = aid
        self.win_colors: list[frozenset[int]] = []
        self.click_colors: Counter = Counter()
        self.attempt_failures = 0
        self.attempt_successes = 0

    def observe_win(self, frame: np.ndarray, akey: ActionKey) -> None:
        """One observed level-up transition sourced at masked `frame`."""
        self.win_colors.append(
            frozenset(int(v) for v in np.unique(frame)))
        if self.aid == 6:
            _, x, y = akey
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                self.click_colors[int(frame[y, x])] += 1

    @property
    def signature_color(self) -> int | None:
        """For ACTION6: the one clicked color every observed win shares."""
        if len(self.click_colors) == 1:
            return next(iter(self.click_colors))
        return None

    @property
    def gated(self) -> bool:  # override: gate = consistent win evidence
        if self.dead or len(self.win_colors) < self.cfg.win_min_obs:
            return False
        if self.aid == 6 and self.signature_color is None:
            return False
        return True

    def precondition(self, frame: np.ndarray) -> bool:
        """Colors present at EVERY observed win must be present. The stricter
        no-novel-colors test applies only to non-click predicates: measured
        on vc33, every level introduces a fresh color (L0:11, L1:14, L2:15),
        so requiring colors <= union(win sources) made the ACTION6 predicate
        permanently inert on exactly the new levels it exists for. Click
        predicates are already localized by the signature color and bounded
        by the failure cap; bare-action predicates have no such filter and
        keep the conservative envelope."""
        colors = {int(v) for v in np.unique(frame)}
        always = set(self.win_colors[0])
        union: set[int] = set()
        for cs in self.win_colors:
            always &= cs
            union |= cs
        if not always <= colors:
            return False
        return self.aid == 6 or colors <= union

    def candidates(self, frame: np.ndarray, avail: tuple[int, ...],
                   menu: list[ActionKey]) -> list[ActionKey]:
        """Predicted-win action keys at a masked frame, menu order."""
        if not self.gated or self.aid not in avail:
            return []
        if not self.precondition(frame):
            return []
        if self.aid != 6:
            return [(self.aid, -1, -1)]
        c = self.signature_color
        h, w = frame.shape
        return [a for a in menu
                if a[0] == 6 and 0 <= a[2] < h and 0 <= a[1] < w
                and int(frame[a[2], a[1]]) == c]

    def attempt_result(self, success: bool) -> None:
        if success:
            self.attempt_successes += 1
            self.note_outcome(True)
        else:
            self.attempt_failures += 1
            self.note_outcome(False)
            if self.attempt_failures >= self.cfg.max_win_failures:
                self.dead = True

    # frame-prediction interface unused — this rule serves candidates instead
    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        return None

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        out = super().stats()
        out["win_obs"] = len(self.win_colors)
        out["attempts"] = self.attempt_successes + self.attempt_failures
        out["attempt_successes"] = self.attempt_successes
        return out


# ---------------------------------------------------------------------------
# T2: click rules (generalization over ACTION6 coordinates)
# ---------------------------------------------------------------------------

class ClickRecolorRule(Rule):
    """Click any cell of color c -> the clicked component recolors to c'."""

    family = "click_recolor"

    def __init__(self, scope: str, cfg: EffectConfig, color: int) -> None:
        self.color = color
        self._lab_cache: dict[bytes, np.ndarray] = {}
        super().__init__(scope, cfg)
        self._votes: "Counter[int]" = Counter()  # observed to-colors
        self.to_color: int | None = None

    def _component(self, frame: np.ndarray, y: int, x: int) -> np.ndarray | None:
        """Connected component of self.color containing (y, x). Labeling is
        cached per frame (the planner classifies many clicks per node)."""
        kb = frame.tobytes()
        lab = self._lab_cache.get(kb)
        if lab is None:
            lab, _ = ndimage.label(frame == self.color, structure=_S4)
            if len(self._lab_cache) >= 256:
                self._lab_cache.clear()
            self._lab_cache[kb] = lab
        cid = lab[y, x]
        if cid == 0:
            return None
        return lab == cid

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        if self.to_color is None:
            return None
        _, x, y = akey
        if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
            return None
        if frame[y, x] != self.color:
            return None
        comp = self._component(frame, y, x)
        if comp is None:
            return None
        out = frame.copy()
        out[comp] = self.to_color
        return out

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        if self.dead:
            return
        _, x, y = akey
        if frame[y, x] != self.color:
            return
        changed = frame != dst
        if not changed.any():
            return  # null click; the prequential score already counted it
        comp = self._component(frame, y, x)
        if comp is None or (changed & ~comp).any():
            return  # non-local effect: this family cannot express the pair
        news = np.unique(dst[changed])
        if len(news) != 1 or (dst[comp] != news[0]).any():
            return  # not a uniform component recolor
        self._votes[int(news[0])] += 1
        new, n = max(self._votes.items(), key=lambda kv: (kv[1], kv[0]))
        self.to_color = new if n >= self.cfg.min_fit else None


class ClickNullRule(Rule):
    """Click on color c never changes the board. Gates on prequential
    evidence alone; feeds MENU demotion, not the edge set."""

    family = "click_null"

    def __init__(self, scope: str, cfg: EffectConfig, color: int) -> None:
        self.color = color
        super().__init__(scope, cfg)

    def try_predict(self, frame: np.ndarray,
                    akey: ActionKey) -> np.ndarray | None:
        _, x, y = akey
        if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
            return None
        if frame[y, x] != self.color:
            return None
        return frame  # identity = deliberate null prediction

    def train(self, frame: np.ndarray, akey: ActionKey,
              dst: np.ndarray) -> None:
        pass  # the prequential score is the entire learning signal


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

T1_ACTION_IDS = (1, 2, 3, 4, 5, 7)


@dataclass
class Prediction:
    """One gated-rule prediction offered to the planner."""

    kind: str  # "edge" | "null"
    key: bytes | None = None            # predicted masked-state key (edge)
    frame: np.ndarray | None = None     # predicted masked frame (edge)
    penalty: float = 0.0
    rule: Rule | None = None


class EffectEngine:
    """Owns all rules for one game; observes transitions, serves predictions.

    Prediction identity: predicted keys are Perception.state_key bytes of the
    predicted masked frame — the same raw keys the graph uses, so a predicted
    node that already exists in the graph merges with it naturally.
    """

    def __init__(self, perception: Perception,
                 cfg: EffectConfig | None = None) -> None:
        self.cfg = cfg or EffectConfig()
        self.perception = perception
        self.rules: dict[tuple[str, str], Rule] = {}
        self.suspect: set[Hashable] = set()
        self.version = 0  # bumped on any gate-set change (cache invalidation)
        self._issued: dict[tuple[Hashable, ActionKey], Rule] = {}
        self._cache: dict[tuple[Hashable, ActionKey],
                          tuple[int, "Prediction | None"]] = {}

    # -- rule access -------------------------------------------------------

    def _t1_rules(self, aid: int) -> list[Rule]:
        scope = f"A{aid}"
        out: list[Rule] = []
        for family in (TranslationRule, BlobRelocateRule, ColorMapRule,
                       ConstantDiffRule, MoveBlockedRule):
            k = (family.family, scope)
            if k not in self.rules:
                self.rules[k] = family(scope, self.cfg)
            out.append(self.rules[k])
        return out

    def _win_rule(self, aid: int) -> WinRule:
        k = ("win_sig", f"A{aid}")
        rule = self.rules.get(k)
        if not isinstance(rule, WinRule):
            rule = WinRule(f"A{aid}", self.cfg, aid)
            self.rules[k] = rule
        return rule

    def _t2_rules(self, color: int) -> list[Rule]:
        scope = f"click:c={color}"
        out: list[Rule] = []
        for family in (ClickRecolorRule, ClickNullRule):
            k = (family.family, scope)
            if k not in self.rules:
                self.rules[k] = family(scope, self.cfg, color)
            out.append(self.rules[k])
        return out

    def _rules_for(self, frame: np.ndarray, akey: ActionKey) -> list[Rule]:
        aid = akey[0]
        if aid in T1_ACTION_IDS:
            return self._t1_rules(aid)
        if aid == 6:
            _, x, y = akey
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                return self._t2_rules(int(frame[y, x]))
        return []

    # -- observation: prequential test-then-train --------------------------

    def observe(self, src_frame: np.ndarray, akey: ActionKey,
                dst_frame: np.ndarray) -> None:
        """One executed within-level, non-terminal transition (RAW frames;
        masking happens here). Terminal transitions (WIN / GAME_OVER) must
        NOT be fed: rules model board dynamics, not level boundaries."""
        src = np.asarray(self.perception.mask_frame(
            np.asarray(src_frame, dtype=np.int16)))
        dst = np.asarray(self.perception.mask_frame(
            np.asarray(dst_frame, dtype=np.int16)))
        gates_before = self._gate_signature()
        for rule in self._rules_for(src, akey):
            if rule.dead:
                continue
            rule.transitions_seen += 1
            pred = rule.try_predict(src, akey)
            if pred is not None:
                rule.note_outcome(np.array_equal(pred, dst))
            rule.train(src, akey, dst)
        if self._gate_signature() != gates_before:
            self.version += 1
            self._cache.clear()

    def _gate_signature(self) -> tuple[bool, ...]:
        return tuple(r.gated for r in self.rules.values())

    # -- prediction service -------------------------------------------------

    def predict(self, state_key: Hashable, frame: np.ndarray,
                akey: ActionKey) -> Prediction | None:
        """Best gated rule's prediction for (state, action); None if no gated
        rule speaks, or the state is suspect. Cached per engine version."""
        if state_key in self.suspect:
            return None
        ck = (state_key, akey)
        hit = self._cache.get(ck)
        if hit is not None and hit[0] == self.version:
            return hit[1]
        result = self._predict_uncached(state_key, frame, akey)
        self._cache[ck] = (self.version, result)
        return result

    def _predict_uncached(self, state_key: Hashable, frame: np.ndarray,
                          akey: ActionKey) -> Prediction | None:
        frame = np.asarray(self.perception.mask_frame(
            np.asarray(frame, dtype=np.int16)))
        candidates = [r for r in self._rules_for(frame, akey) if r.gated]
        candidates.sort(key=lambda r: (-r.priority,
                                       -(r.window_accuracy or 0.0)))
        for rule in candidates:
            pred = rule.try_predict(frame, akey)
            if pred is None:
                continue
            if np.array_equal(pred, frame):
                return Prediction(kind="null", rule=rule)
            self._issued[(state_key, akey)] = rule
            return Prediction(
                kind="edge",
                key=self.perception.state_key(pred),
                frame=pred,
                penalty=self.cfg.edge_penalty,
                rule=rule,
            )
        return None

    def chain_successors(
        self, state_key: Hashable, frame: np.ndarray,
        avail: tuple[int, ...],
    ) -> list[tuple[ActionKey, bytes, np.ndarray, float]]:
        """T1-only successors of a PREDICTED frame — used to extend predicted
        chains beyond the visited set. T2 needs per-frame component analysis
        and is deliberately depth-1 only (cost control, reported)."""
        out = []
        for aid in T1_ACTION_IDS:
            if aid not in avail:
                continue
            akey: ActionKey = (aid, -1, -1)
            p = self.predict(state_key, frame, akey)
            if p is not None and p.kind == "edge":
                assert p.key is not None and p.frame is not None
                out.append((akey, p.key, p.frame, p.penalty))
        return out

    # -- win predicate (Stage 2b) -------------------------------------------

    def observe_win(self, src_frame: np.ndarray, akey: ActionKey) -> None:
        """One observed level-up transition (RAW source frame; masked here).
        Terminal transitions stay OUT of `observe`; this is their channel."""
        src = np.asarray(self.perception.mask_frame(
            np.asarray(src_frame, dtype=np.int16)))
        rule = self._win_rule(akey[0])
        rule.transitions_seen += 1
        rule.observe_win(src, akey)
        self.version += 1
        self._cache.clear()

    def win_rules(self) -> list[WinRule]:
        return [r for r in self.rules.values()
                if isinstance(r, WinRule) and r.gated]

    @property
    def has_win_rules(self) -> bool:
        return bool(self.win_rules())

    def win_candidates(self, frame: np.ndarray, avail: tuple[int, ...],
                       menu: list[ActionKey]) -> list[ActionKey]:
        """Predicted-win action keys at a RAW frame (masked here)."""
        f = np.asarray(self.perception.mask_frame(
            np.asarray(frame, dtype=np.int16)))
        out: list[ActionKey] = []
        for rule in self.win_rules():
            out.extend(rule.candidates(f, avail, menu))
        return out

    def win_attempt_result(self, akey: ActionKey, success: bool) -> None:
        """The agent executed a predicted-win action; the env has spoken."""
        rule = self.rules.get(("win_sig", f"A{akey[0]}"))
        if isinstance(rule, WinRule):
            rule.attempt_result(success)
            self.version += 1
            self._cache.clear()

    # -- level boundary -----------------------------------------------------

    def on_level_up(self) -> None:
        """Per-level probation (the design's per-level reset, scoped to the
        GATE rather than the hypothesis): every rule keeps what it learned
        but must re-earn its gate from the new level's own transitions.
        Measured motivation: ft09's L0-learned click-null rules stayed gated
        through L1 on a stale window and demoted exactly the clicks L1
        needed. Suspect states are level-local keys — cleared too."""
        for rule in self.rules.values():
            rule.window.clear()
        self.suspect.clear()
        self.version += 1
        self._cache.clear()

    # -- live demotion ------------------------------------------------------

    def live_mispredict(self, state_key: Hashable, akey: ActionKey) -> None:
        """The planner executed a predicted edge and the env disagreed."""
        rule = self._issued.pop((state_key, akey), None)
        if rule is not None:
            rule.demote()
        self.suspect.add(state_key)
        self.version += 1
        self._cache.clear()

    # -- reporting ----------------------------------------------------------

    def stats_rows(self) -> list[dict[str, Any]]:
        rows = [r.stats() for r in self.rules.values()]
        rows.sort(key=lambda r: (r["family"], str(r["scope"])))
        return rows

    def gated_rules(self) -> list[Rule]:
        return [r for r in self.rules.values() if r.gated]


if __name__ == "__main__":  # standalone self-check on synthetic transitions
    from engineered.graph import click_key, simple_key

    cfg = EffectConfig()
    p = Perception("test", [])
    eng = EffectEngine(p, cfg)

    # translation: a 2x2 color-5 blob steps right on ACTION3
    def frame_at(x0: int) -> np.ndarray:
        f = np.zeros((64, 64), dtype=np.int16)
        f[10:12, x0:x0 + 2] = 5
        return f

    a3 = simple_key(3)
    for x in range(2, 40):
        eng.observe(frame_at(x), a3, frame_at(x + 1))
    rule = eng.rules[("translation", "A3")]
    assert rule.gated and rule.accuracy is not None and rule.accuracy >= 0.9
    pr = eng.predict("sK", frame_at(50), a3)
    assert pr is not None and pr.kind == "edge"
    assert pr.frame is not None and np.array_equal(pr.frame, frame_at(51))
    # click null: clicks on color 0 never do anything
    f = frame_at(5)
    for i in range(15):
        eng.observe(f, click_key(30 + i, 30), f)
    null_rule = eng.rules[("click_null", "click:c=0")]
    assert null_rule.gated
    prn = eng.predict("sK2", f, click_key(60, 60))
    assert prn is not None and prn.kind == "null"
    # live demotion
    eng.live_mispredict("sK", a3)
    assert not rule.gated and "sK" in eng.suspect
    assert eng.predict("sK", frame_at(50), a3) is None
    print("effects self-check OK:", len(eng.rules), "rules")
