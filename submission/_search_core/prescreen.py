"""ZERO-ACTION frame-0 pre-screen — the cheap probe.

WHY THIS EXISTS (measured; docs/ENVELOPE-2026-08-26-v8.md Addendum 3 §C4).
The v8.2 crack-or-nothing config's expected value per game is

    EV(p) = p * GAIN - (1 - p) * C_probe

with ``GAIN = +85.7`` (ft09 14.29 -> a banked 100.0, measured on
``CompetitionArcadeServer``) and ``C_probe = 2.7 - 3.8`` points — the cost of
the ~43-action warmup plus ~45-action detect on a game that never cracks,
because those engine actions are billed into the SAME play the LLM keeps
using (a failed engagement cannot open a fresh play: only a post-WIN reset
escapes the competition guard, ``api.py:316-334``). At the only estimate of
``p`` we have — ft09_gf2 fires on 1 of 25 dev games = 4 % — EV is
``0.04 * 85.7 - 0.96 * 2.7 ~= 0 +- 1``: indistinguishable from zero.

The probe cost is the dominant term and it is paid on the ~96 % of games that
never crack. It cannot be isolated into its own play (that is exactly the
reset the guard swallows), so the only lever is to SPEND FEWER ACTIONS. This
module spends **none**: it decides from frame 0 — the observation the harness
already holds at game start — whether a game is even plausibly of a class
worth probing. A declined game never reaches warmup or detection and pays 0.

THE SIGNAL (frame 0 only, no engine call).
``detect_ft09``'s own preconditions are already mostly frame-only: CLICK
available, >= 4 uniform 6x6 tiles, and an aligned patterned clue block
adjacent to the tile lattice. What costs 40 clicks is only the live-toggle
confirmation. Two frame-0 geometric facts separate the class outright on the
dev corpus:

  A. RING-ISOLATED TILES. A GF2 toggle board is drawn as 3x3 sprites at
     camera scale 2 (6x6 px) on an 8 px lattice, so every tile is surrounded
     by a 1 px gutter containing none of its own colour. Big flat regions in
     other games also yield "uniform 6x6 blocks" under a stride-2 scan, but
     those blocks sit INSIDE the region, so the ring repeats their colour.
     Measured on the 25 dev fixtures: ft09 has 32 ring-isolated tiles; the
     other 24 games have **zero** — separation with no threshold at all.

  B. LATTICE DOMINANCE. Independently, ft09's tiles concentrate on 3 lattice
     phases with 16 of 32 on one of them (fraction 0.50); every other fixture
     spreads its blocks over 11-16 phases with a dominant fraction <= 0.21
     (tn36 reads 0.33 but has only 3 blocks in total). Kept as a second,
     independent branch so a hidden variant whose tiles TOUCH (no gutter,
     branch A blind) can still pass.

A game is admitted if EITHER branch fires. On the dev corpus the union is
exactly {ft09}: 1 true positive, 0 false negatives, 0 false positives.

CONTRACT.
  * Pure function of (grid, available_actions). No engine, no I/O, no state.
  * Cost: 0 engine actions, ~10 ms of CPU on a 64x64 frame.
  * FAIL-OPEN: any malformed input returns "admit" (``screen`` raises
     nothing; callers that catch are belt-and-braces). Being wrong in the
     admit direction costs the probe we pay today; being wrong in the decline
     direction destroys the whole value case.
"""

from __future__ import annotations

import numpy as np

# --- geometry (mirrors specialists.py; asserted equal in test_prescreen) ---
TILE = 6          # displayed tile size (3x3 sprite at camera scale 2)
LAT = 8           # tile lattice pitch in display px
NB8 = [(-LAT, -LAT), (0, -LAT), (LAT, -LAT),
       (-LAT, 0), (LAT, 0),
       (-LAT, LAT), (0, LAT), (LAT, LAT)]

# --- thresholds, with the measured margin each one carries -----------------
#
# CORRECTED 2026-08-28 after the first out-of-sample test (holdout_screen.py).
# The original branch-A bars (both 4) were read off ft09's OWN geometry — 32
# ring-isolated tiles, 16 on the dominant phase — and that is in-sample fitting.
# The holdout corpus contains one further genuine ft09_gf2 game, cx01, whose
# board is far sparser: 3 ring-isolated tiles, 2 on the dominant phase. The
# old bars DECLINED it, i.e. a false negative on the only unseen same-class
# game we have — the failure direction the module contract calls fatal.
#
# The SIGNAL was never the problem. Across all 38 games in both corpora,
# ``n_strict`` is categorical: ft09 32, cx01 3, and **every one of the 36
# negatives is exactly 0**. So branch A's bars are set to the lowest values
# that admit the known positives, and the margin they retain is 3-vs-0 —
# categorical, not a numeric hair. Raising them again re-introduces the false
# negative; lowering them further buys nothing (0 is already excluded).
MIN_TILES = 4              # detect_ft09's own floor, on the raw block count
MIN_STRICT = 3             # branch A: ring-isolated tiles (cx01 3, ft09 32)
MIN_STRICT_LATTICE = 2     # branch A: strict tiles on the dominant phase
FRAC_MIN_TILES = 8         # branch B needs a real tile field, not 3 blocks
FRAC_MIN_LATTICE = 6       # ... with a substantial dominant phase
FRAC_MIN = 0.35            # ft09 0.50 vs next-highest CLICK game 0.21

# Retained for compatibility; branch A no longer reads it.
MIN_LATTICE = 4


# --------------------------------------------------------------------------
# frame-0 extraction (no engine call)
# --------------------------------------------------------------------------

def grid_of(obs) -> np.ndarray | None:
    """The settled (last) frame of anything frame-shaped, with no engine call:
    an arcengine observation (``.frame`` = list of 2-D frames), a taaf
    ``GameState`` (``.frame`` -> ``Frame.data``), a taaf ``Frame``, or a raw
    list/array. Returns None when no 2-D grid can be recovered."""
    cur = obs
    for _ in range(4):                       # observation -> frame -> data
        if cur is None:
            return None
        if isinstance(cur, (np.ndarray, list, tuple)):
            break
        nxt = getattr(cur, "frame", None)
        if nxt is None:
            nxt = getattr(cur, "data", None)
        if nxt is None:
            raw = getattr(cur, "raw", None)
            nxt = getattr(raw, "frame", None) if raw is not None else None
        if nxt is None:
            return None
        cur = nxt
    try:
        a = np.asarray(cur)
    except Exception:  # noqa: BLE001 — fail-open
        return None
    while a.ndim > 2:
        a = a[-1]
    if a.ndim != 2 or a.size == 0 or not np.issubdtype(a.dtype, np.number):
        return None
    return a


def avail_of(obs) -> set[int]:
    raw = getattr(obs, "available_actions", None)
    if raw is None and isinstance(obs, (list, tuple, set)):
        raw = obs
    out = set()
    for a in (raw or []):
        try:
            out.add(int(a))
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------
# frame-0 primitives
# --------------------------------------------------------------------------

def _background_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _uniform_blocks(g: np.ndarray) -> list[tuple[int, int, int]]:
    """(x, y, colour) of every TILExTILE uniform non-background square at
    even coordinates. Same scan as ``specialists._uniform_blocks``."""
    bg = _background_color(g)
    h, w = g.shape
    out = []
    for y in range(0, h - TILE + 1, 2):
        for x in range(0, w - TILE + 1, 2):
            b = g[y:y + TILE, x:x + TILE]
            c = int(b[0, 0])
            if c != bg and bool((b == b[0, 0]).all()):
                out.append((x, y, c))
    return out


def _ring_isolated(g: np.ndarray, x: int, y: int, c: int) -> bool:
    """True when the 1 px ring around the block contains NO cell of the
    block's own colour — a free-standing sprite tile rather than the interior
    of a larger uniform region."""
    h, w = g.shape
    y0, y1 = max(0, y - 1), min(h, y + TILE + 1)
    x0, x1 = max(0, x - 1), min(w, x + TILE + 1)
    return int((g[y0:y1, x0:x1] == c).sum()) == TILE * TILE


def _patterned_blocks(g: np.ndarray) -> list[tuple[int, int]]:
    """(x, y) of every TILExTILE block that decomposes into 3x3 uniform 2x2
    sub-blocks with >= 2 colours — clue sprites. Same scan as
    ``specialists._patterned_blocks``, positions only."""
    h, w = g.shape
    out = []
    for y in range(0, h - TILE + 1, 2):
        for x in range(0, w - TILE + 1, 2):
            b = g[y:y + TILE, x:x + TILE]
            q = b.reshape(3, 2, 3, 2)
            if not bool((q == q[:, :1, :, :1]).all()):
                continue
            P = q[:, 0, :, 0]
            if np.unique(P).size >= 2:
                out.append((x, y))
    return out


def _phase_table(tiles: set[tuple[int, int]]) -> dict[tuple[int, int], list]:
    phases: dict[tuple[int, int], list] = {}
    for (x, y) in tiles:
        phases.setdefault((x % LAT, y % LAT), []).append((x, y))
    return phases


def _dominant(tiles: set[tuple[int, int]], pats: list[tuple[int, int]]) -> dict:
    """The lattice phase carrying the most tiles, with its aligned clue count."""
    best = {"phase": None, "n": 0, "clues": 0, "rows": 0, "cols": 0}
    for phase, members in _phase_table(tiles).items():
        ms = set(members)
        clues = sum(1 for (px, py) in pats
                    if (px % LAT, py % LAT) == phase
                    and any((px + dx, py + dy) in ms for dx, dy in NB8))
        if (len(members), clues) > (best["n"], best["clues"]):
            best = {"phase": phase, "n": len(members), "clues": clues,
                    "rows": len({y for _, y in members}),
                    "cols": len({x for x, _ in members})}
    return best


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------

def ft09_gf2_features(grid, avail=None) -> dict:
    """Every number the ft09_gf2 screen decides on. Zero engine actions."""
    g = grid_of(grid)
    if g is None:
        return {"usable": False}
    acts = avail_of(avail if avail is not None else grid)
    blocks = _uniform_blocks(g)
    strict = {(x, y) for (x, y, c) in blocks if _ring_isolated(g, x, y, c)}
    allpos = {(x, y) for (x, y, _) in blocks}
    pats = _patterned_blocks(g)
    block_lat = _dominant(allpos, pats)
    return {
        "usable": True,
        "shape": tuple(int(v) for v in g.shape),
        "click": 6 in acts,
        "n_blocks": len(allpos),
        "n_strict": len(strict),
        "n_patterned": len(pats),
        "strict_lattice": _dominant(strict, pats),
        "block_lattice": block_lat,
        "block_frac": (block_lat["n"] / len(allpos)) if allpos else 0.0,
    }


def prescreen_ft09_gf2(grid, avail=None) -> tuple[bool, str]:
    """(admit, reason). Admit = "plausibly ft09_gf2-class, worth the probe"."""
    f = ft09_gf2_features(grid, avail)
    if not f["usable"]:
        return True, "no_frame"                     # fail-open
    if not f["click"]:
        return False, "no_click_action"
    if f["n_blocks"] < MIN_TILES:
        return False, f"tiles<{MIN_TILES}({f['n_blocks']})"
    # branch A — ring-isolated sprite tiles on one lattice phase
    sl = f["strict_lattice"]
    if f["n_strict"] >= MIN_STRICT and sl["n"] >= MIN_STRICT_LATTICE:
        return True, (f"strict_lattice(strict={f['n_strict']},"
                      f"n={sl['n']},clues={sl['clues']})")
    # branch B — a dominant lattice phase among touching tiles, with clues
    bl = f["block_lattice"]
    if (f["n_blocks"] >= FRAC_MIN_TILES and bl["n"] >= FRAC_MIN_LATTICE
            and f["block_frac"] >= FRAC_MIN and bl["clues"] >= 1):
        return True, f"lattice_frac({f['block_frac']:.2f})"
    return False, (f"no_lattice(strict={f['n_strict']},"
                   f"frac={f['block_frac']:.2f},clues={bl['clues']})")


SCREENS = {
    "ft09_gf2": prescreen_ft09_gf2,
}


def screen(grid, avail=None, classes=("ft09_gf2",)) -> tuple[bool, str]:
    """Admit the expensive probe if ANY requested class is plausible.

    A requested class with no registered screen (or the ``*`` wildcard) means
    "we cannot pre-screen this" -> admit. Any internal failure -> admit.
    Returns ``(admit, note)``; never raises."""
    try:
        wanted = {c.strip() for c in classes if str(c).strip()} \
            if not isinstance(classes, str) \
            else {c.strip() for c in classes.split(",") if c.strip()}
        if not wanted or "*" in wanted:
            return True, "unscreenable(*)"
        unknown = sorted(wanted - set(SCREENS))
        if unknown:
            return True, f"unscreenable({','.join(unknown)})"
        notes = []
        for name in sorted(wanted):
            ok, why = SCREENS[name](grid, avail)
            notes.append(f"{name}:{why}")
            if ok:
                return True, "|".join(notes)
        return False, "|".join(notes)
    except Exception as exc:  # noqa: BLE001 — fail-open, always
        return True, f"screen_error:{type(exc).__name__}"
