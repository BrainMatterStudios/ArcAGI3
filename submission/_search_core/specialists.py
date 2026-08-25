"""Specialist tier — stage 6 of the SearchCore plan
(docs/RESEARCH-2026-08-23-searchcore-and-multirole.md §A item 6 + Addendum 3).

Four mechanic-class solvers ported from the dev-tuned in-tree originals
(scripts/research_2026_07_01/{ht_ft09,wa30_planner,sc25_solve}.py and the
Track-2 engineered lane) behind FRAME-ONLY DETECTORS, so they can fire on
hidden games of the same class without knowing the game id:

  ft09_gf2      toggle-tile constraint puzzles: clicks cycle tile colors
                through a per-level palette (identity or neighborhood masks
                = linear system over Z_k); patterned clue blocks encode
                equal/differ constraints on their 8 neighbors. Solver:
                probe-learn the effect matrix + palette cycle from
                snapshots, read clues from the frame, per-tile CSP, solve
                the linear system mod k, snapshot-verify, execute.
                (Engine-verified on ft09: L1-L4 clue-solved end-to-end.)

  tn36_program  program-register machines: a row of slots, each a bitmask
                of toggle cells (click = toggle), plus a RUN button whose
                click animates a program execution (multi-layer frame) and
                resets on failure. Solver: probe-learn toggle cells + run
                buttons, enumerate uniform slot patterns first (the L1/L2
                solutions are uniform), then mixed patterns from the
                movement vocabulary, snapshot-run each config.
                (Engine-verified: tn36 L1 = [3,3,3,3,3]+run, 12 clicks.)

  sc25_glyph    glyph-cast games: root is inert for basic actions; two
                3x3-cell panels (a TARGET pattern panel and a clickable
                SLOT panel); drawing the target pattern into the slots
                casts a spell (avatar shrinks), after which basic-action
                BFS solves the maze. (Engine-verified: sc25 L1 cracked
                frame-only: 4 slot clicks + 12 moves.)

  wa30_grabdrag grab-drag block puzzles: ACTION5 grabs an adjacent block,
                moves drag it rigidly; win = every block on a goal pad.
                Solver: learn avatar color from move probes, perceive
                blocks (ring comps with interior marker), pads (large
                marker-colored comps), walls (all other non-bg cells,
                plus latent walls inferred under blocks), then per-block
                A* legs over facings x orders, snapshot-verified
                shortest-first. (Engine-verified: wa30 L1+L2; L3's optimal
                plan (169 acts) exceeds the level's step budget (~100) —
                measured wall, specialist fails open there.)

All solvers operate exclusively through the SearchCore backend Handle API
(backend.root / backend.children), so every "solved" result is
engine-verified by construction (the returned handle observed the level-up)
and both snapshot and reset-replay backends work. Detectors run bounded
snapshot probes; a detector that fires wrongly burns its lane budget, so
each is measured against all 25 fixtures (test_specialists.py matrix).
Fail-open: any failure returns solved=False and the generic lanes proceed.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any  # noqa: F401

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPLORER_DIR = os.path.join(os.path.dirname(_HERE), "_explorer_floor")
if _EXPLORER_DIR not in sys.path:
    sys.path.insert(0, _EXPLORER_DIR)

from graft_explorer import _background_color, _components  # noqa: E402


# --------------------------------------------------------------------------
# small helpers (Handle-level; no dependency on search_core to avoid cycles)
# --------------------------------------------------------------------------

def settled(obs) -> np.ndarray:
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


def n_layers(obs) -> int:
    a = np.asarray(obs.frame)
    return a.shape[0] if a.ndim == 3 else 1


def lv(obs) -> int:
    return int(obs.levels_completed or 0)


def C(x: int, y: int) -> tuple:
    return ("C", int(x), int(y))


def S(a: int) -> tuple:
    return ("S", int(a))


def step1(backend, handle, tok):
    """One child via the backend (deepcopy branch). None on failure."""
    for _, child in backend.children(handle, [tok]):
        return child
    return None


def chain(backend, handle, toks, stop_on_level=None):
    """Apply toks sequentially (linear chain of snapshot branches).
    Returns (final_handle, leveled_handle_or_None). Stops early on GAME_OVER
    (returns (None, None)) or when levels_completed reaches stop_on_level."""
    from arcengine import GameState

    cur = handle
    for tok in toks:
        cur = step1(backend, cur, tok)
        if cur is None or cur.obs is None:
            return None, None
        if stop_on_level is not None and (
                lv(cur.obs) >= stop_on_level
                or cur.obs.state == GameState.WIN):
            return cur, cur
        if cur.obs.state == GameState.GAME_OVER:
            return None, None
    return cur, None


def avail_of(obs) -> set:
    return {int(a) for a in (obs.available_actions or [])}


def masked_eq(a: np.ndarray, b: np.ndarray, mb: np.ndarray | None) -> bool:
    if a.shape != b.shape:
        return False
    d = a != b
    if mb is not None and mb.shape == d.shape:
        d = d & ~mb
    return not d.any()


def masked_diff(a: np.ndarray, b: np.ndarray, mb: np.ndarray | None) -> np.ndarray:
    d = a != b
    if mb is not None and mb.shape == d.shape:
        d = d & ~mb
    return d


def _result(solved: bool, *, handle=None, depth=0, states=0, nodes=0,
            wall=0.0, reason="spec_failed", **extra) -> dict:
    r = dict(solved=solved, states=states, nodes=nodes,
             wall=round(wall, 1), reason=reason, **extra)
    if solved:
        r["handle"] = handle
        r["depth"] = depth
    return r


# --------------------------------------------------------------------------
# ft09_gf2 — toggle-tile constraint puzzles
# --------------------------------------------------------------------------

TILE = 6          # displayed tile size (3x3 sprite at camera scale 2)
LAT = 8           # tile lattice pitch in display px
NB8 = [(-LAT, -LAT, 0, 0), (0, -LAT, 0, 1), (LAT, -LAT, 0, 2),
       (-LAT, 0, 1, 0), (LAT, 0, 1, 2),
       (-LAT, LAT, 2, 0), (0, LAT, 2, 1), (LAT, LAT, 2, 2)]


def _uniform_blocks(g: np.ndarray) -> list[tuple[int, int, int]]:
    """(x, y, color) of every 6x6 uniform non-bg square at even coords."""
    bg = _background_color(g.tolist())
    h, w = g.shape
    out = []
    for y in range(0, h - TILE + 1, 2):
        for x in range(0, w - TILE + 1, 2):
            b = g[y:y + TILE, x:x + TILE]
            if b[0, 0] != bg and (b == b[0, 0]).all():
                out.append((x, y, int(b[0, 0])))
    return out


def _patterned_blocks(g: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """(x, y, P) for 6x6 blocks decomposing into 3x3 uniform 2x2 sub-blocks
    with >= 2 colors (clue sprites and NTi-style patterned tiles)."""
    h, w = g.shape
    out = []
    for y in range(0, h - TILE + 1, 2):
        for x in range(0, w - TILE + 1, 2):
            b = g[y:y + TILE, x:x + TILE]
            P = np.zeros((3, 3), dtype=int)
            ok = True
            for j in range(3):
                for i in range(3):
                    sub = b[2 * j:2 * j + 2, 2 * i:2 * i + 2]
                    if sub[0, 0] != sub[0, 1] or sub[0, 0] != sub[1, 0] \
                            or sub[0, 0] != sub[1, 1]:
                        ok = False
                        break
                    P[j, i] = sub[0, 0]
                if not ok:
                    break
            if ok and np.unique(P).size >= 2:
                out.append((x, y, P))
    return out


def _center_color(g: np.ndarray, x: int, y: int) -> int:
    return int(g[y + 2, x + 2])


def _ft09_probe_tiles(backend, root, g0, cands, budget_deadline,
                      max_probes=90, by_regions=False):
    """Click each candidate once (snapshot); effect column = the tile
    positions whose content changed.

    by_regions=False (uniform-lattice boards, proven on ft09 L1-L5):
    columns are candidate positions whose CENTER color changed.

    by_regions=True (all-patterned boards, ft09 L6): candidate windows
    alias the same physical tile at several offsets, so columns are read
    from the CHANGED-REGION components instead (each tile-sized changed
    comp's bbox top-left is the physical anchor), and candidates whose
    click point lands in an already-probed lattice cell are skipped."""
    effects: dict[tuple[int, int], list[tuple[int, int]]] = {}
    if not by_regions:
        for (x, y) in list(cands)[:max_probes]:
            if time.time() > budget_deadline:
                break
            child = step1(backend, root, C(x + 2, y + 2))
            if child is None or child.obs is None:
                continue
            g1 = settled(child.obs)
            if g1.shape != g0.shape:
                continue
            col = [(cx, cy) for (cx, cy) in cands
                   if _center_color(g1, cx, cy) != _center_color(g0, cx, cy)]
            if col:
                effects[(x, y)] = col
        return effects

    off = None                       # lattice offset, learned from anchors
    probed_cells: set[tuple[int, int]] = set()
    probes = 0
    for (x, y) in list(cands):
        if probes >= max_probes or time.time() > budget_deadline:
            break
        px, py = x + 2, y + 2
        if off is not None:
            cell = ((px - off[0]) // LAT, (py - off[1]) // LAT)
            if cell in probed_cells:
                continue
        child = step1(backend, root, C(px, py))
        probes += 1
        if child is None or child.obs is None:
            continue
        g1 = settled(child.obs)
        if g1.shape != g0.shape:
            continue
        d = (g1 != g0)
        d[60:, :] = False            # HUD band is not board content
        if not d.any():
            if off is not None:
                probed_cells.add(((px - off[0]) // LAT,
                                  (py - off[1]) // LAT))
            continue
        anchors = []
        for comp in _components(d.tolist()):
            if comp["color"] != 1 or comp["size"] < 3:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if r1 - r0 + 1 > TILE or c1 - c0 + 1 > TILE:
                continue
            anchors.append((c0, r0))
        if not anchors:
            continue
        if off is None:
            off = (anchors[0][0] % LAT, anchors[0][1] % LAT)
        # keep only lattice-consistent anchors (hint flashes etc. drop out)
        anchors = [(ax, ay) for (ax, ay) in anchors
                   if (ax % LAT, ay % LAT) == off]
        if not anchors:
            continue
        effects[(x, y)] = sorted(anchors)
        probed_cells.add(((px - off[0]) // LAT, (py - off[1]) // LAT))
    return effects


def _solve_mod_k(A_cols: dict, pinned: dict, k: int,
                 targets: list) -> dict | None:
    """Solve sum_c x_c * A[:,c] = d (mod k) for the pinned tile rows.
    A_cols: click target -> list of tile positions it advances (+1).
    pinned: tile position -> required cycle delta.
    Returns {target: clicks mod k} or None. k must be prime (2, 3, 5)."""
    tiles = sorted(pinned)
    t_index = {t: i for i, t in enumerate(tiles)}
    n, m = len(tiles), len(targets)
    if n == 0:
        return {}
    A = np.zeros((n, m), dtype=np.int64)
    for j, tgt in enumerate(targets):
        for t in A_cols.get(tgt, []):
            if t in t_index:
                A[t_index[t], j] = (A[t_index[t], j] + 1) % k
    d = np.array([pinned[t] % k for t in tiles], dtype=np.int64)
    # Gaussian elimination mod prime k
    A = A.copy() % k
    d = d.copy() % k
    inv = {a: pow(a, k - 2, k) if k > 2 else a for a in range(1, k)}
    row = 0
    piv_of_col: dict[int, int] = {}
    for col in range(m):
        p = next((r for r in range(row, n) if A[r, col] % k), None)
        if p is None:
            continue
        A[[row, p]] = A[[p, row]]
        d[[row, p]] = d[[p, row]]
        f = inv[int(A[row, col]) % k]
        A[row] = (A[row] * f) % k
        d[row] = (d[row] * f) % k
        for r in range(n):
            if r != row and A[r, col] % k:
                f2 = int(A[r, col]) % k
                A[r] = (A[r] - f2 * A[row]) % k
                d[r] = (d[r] - f2 * d[row]) % k
        piv_of_col[col] = row
        row += 1
        if row == n:
            break
    # consistency: zero rows must have zero rhs
    for r in range(row, n):
        if not (A[r] % k).any() and d[r] % k:
            return None
    x = np.zeros(m, dtype=np.int64)
    for col, r in piv_of_col.items():
        x[col] = d[r] % k
    # verify (defensive; free-column choices are all zero)
    chk = np.zeros(n, dtype=np.int64)
    for j, tgt in enumerate(targets):
        if x[j]:
            for t in A_cols.get(tgt, []):
                if t in t_index:
                    chk[t_index[t]] = (chk[t_index[t]] + x[j]) % k
    if any(int(chk[t_index[t]]) % k != pinned[t] % k for t in tiles):
        return None
    return {targets[j]: int(x[j]) for j in range(m) if x[j]}


def detect_ft09(core) -> bool:
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return False
    if 6 not in avail_of(root.obs):
        return False
    g0 = settled(root.obs)
    uni = _uniform_blocks(g0)
    if len(uni) < 4:
        return False
    # probe up to 40 (decorative decoy tile-grids exist — the ft09 L1 board
    # has 24 decoy tiles before the 8 live ones): need 3 self-only toggles
    live = 0
    for (x, y, col) in uni[:40]:
        child = step1(backend, root, C(x + 2, y + 2))
        if child is None or child.obs is None:
            continue
        g1 = settled(child.obs)
        if g1.shape != g0.shape:
            continue
        d = g1 != g0
        ys, xs = np.nonzero(d)
        if len(ys) == 0:
            continue
        inside = (ys >= y) & (ys < y + TILE) & (xs >= x) & (xs < x + TILE)
        if inside.sum() >= 3 and (~inside).sum() <= 4 \
                and _center_color(g1, x, y) != col:
            live += 1
        if live >= 3:
            break
    if live < 3:
        return False
    # at least one aligned patterned clue adjacent to a live-ish tile lattice
    tiles = {(x, y) for (x, y, _) in uni}
    offs = {(x % LAT, y % LAT) for (x, y) in tiles}
    for (cx, cy, P) in _patterned_blocks(g0):
        if (cx % LAT, cy % LAT) not in offs:
            continue
        if any((cx + dx, cy + dy) in tiles for dx, dy, _, _ in NB8):
            return True
    return False


def solve_ft09(core, target: int, budget_s: float) -> dict:
    from arcengine import GameState  # noqa: F401

    t0 = time.time()
    deadline = t0 + budget_s
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return _result(False, wall=time.time() - t0, reason="reset_failed")
    if lv(root.obs) >= target:
        return _result(True, handle=root, depth=0, states=1,
                       wall=time.time() - t0, reason="solved")
    g0 = settled(root.obs)
    uni = _uniform_blocks(g0)
    pat = _patterned_blocks(g0)
    # patterned candidates only on the uniform-tile lattice (an NTi-style
    # tile shares the lattice; the hundreds of misaligned pattern hits on a
    # decorated board are junk that would eat the probe budget). On an
    # all-patterned board (ft09 L6: every tile is NTi-class) there is no
    # uniform lattice — fall back to the DOMINANT patterned-block offset
    # class instead.
    uoffs = {(x % LAT, y % LAT) for (x, y, _) in uni}
    pat_cands = [(x, y) for (x, y, _) in pat if (x % LAT, y % LAT) in uoffs]
    if len(uni) < 4 and pat:
        # all-patterned board (ft09 L6: every tile is NTi-class): the real
        # tiles repeat ONE identical pattern; junk overlaps vary. Take the
        # positions of repeated pattern-signature groups, largest first.
        from collections import defaultdict

        groups: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
        for (x, y, P) in pat:
            groups[P.tobytes()].append((x, y))
        for sig in sorted(groups, key=lambda s: -len(groups[s])):
            if len(groups[sig]) < 4 or len(pat_cands) >= 80:
                break
            pat_cands += groups[sig]
    by_regions = len(uni) < 4
    cands = [(x, y) for (x, y, _) in uni] + pat_cands[:160]
    if len(cands) < 2:
        return _result(False, wall=time.time() - t0)
    effects = _ft09_probe_tiles(backend, root, g0, cands, deadline,
                                by_regions=by_regions)
    if not effects:
        return _result(False, wall=time.time() - t0)
    # tiles = positions whose center is ever affected
    tiles: set[tuple[int, int]] = set()
    for col in effects.values():
        tiles.update(col)
    if not tiles:
        return _result(False, wall=time.time() - t0)
    # palette cycle: chain clicks on one live target, watch an affected tile
    tgt0 = next(iter(effects))
    watch = effects[tgt0][0]
    cyc = [_center_color(g0, *watch)]
    cur = root
    for _ in range(6):
        cur = step1(backend, cur, C(tgt0[0] + 2, tgt0[1] + 2))
        if cur is None or cur.obs is None:
            break
        c1 = _center_color(settled(cur.obs), *watch)
        if c1 == cyc[0]:
            break
        cyc.append(c1)
    k = len(cyc)
    if k < 2 or k > 5 or k == 4:      # need prime modulus (2, 3, 5)
        return _result(False, wall=time.time() - t0, reason="bad_cycle")
    palette = set(cyc)
    # clue blocks: patterned, aligned, static (not a live target and its own
    # center unaffected), center color in palette, adjacent to >= 1 tile
    offs = {(x % LAT, y % LAT) for (x, y) in tiles}
    req_eq: dict[tuple[int, int], set[int]] = {}
    req_ne: dict[tuple[int, int], set[int]] = {}
    n_clues = 0
    for (cx, cy, P) in pat:
        if (cx, cy) in effects or (cx, cy) in tiles:
            continue
        if (cx % LAT, cy % LAT) not in offs:
            continue
        if int(P[1, 1]) not in palette:
            continue
        nb = [(cx + dx, cy + dy, j, i) for dx, dy, j, i in NB8
              if (cx + dx, cy + dy) in tiles]
        if not nb:
            continue
        n_clues += 1
        nRq = int(P[1, 1])
        for (tx, ty, j, i) in nb:
            if int(P[j, i]) == 0:
                req_eq.setdefault((tx, ty), set()).add(nRq)
            else:
                req_ne.setdefault((tx, ty), set()).add(nRq)
    if n_clues == 0:
        return _result(False, wall=time.time() - t0, reason="no_clues")
    # per-tile CSP -> pinned cycle deltas (ne tiles: alternatives kept)
    idx = {c: i for i, c in enumerate(cyc)}
    pinned: dict[tuple[int, int], int] = {}
    ne_alt: list[tuple[tuple[int, int], list[int]]] = []
    for t in sorted(set(req_eq) | set(req_ne)):
        curc = _center_color(g0, *t)
        if curc not in idx:
            return _result(False, wall=time.time() - t0, reason="tile_color")
        eq = req_eq.get(t, set())
        ne = req_ne.get(t, set())
        if len(eq) > 1 or (eq and next(iter(eq)) in ne):
            return _result(False, wall=time.time() - t0, reason="infeasible")
        if eq:
            want = next(iter(eq))
            if want not in idx:
                return _result(False, wall=time.time() - t0, reason="want")
            pinned[t] = (idx[want] - idx[curc]) % k
        else:
            opts = [c for c in cyc if c not in ne]
            if not opts:
                return _result(False, wall=time.time() - t0, reason="ne_opts")
            deltas = [(idx[o] - idx[curc]) % k for o in opts]
            pinned[t] = deltas[0]
            if len(deltas) > 1:
                ne_alt.append((t, deltas[1:]))
    targets = sorted(effects)
    alt_queue: list[dict] = [dict(pinned)]
    for (t, deltas) in ne_alt[:3]:          # bounded alternative expansion
        extra = []
        for q in alt_queue:
            for d in deltas:
                q2 = dict(q)
                q2[t] = d
                extra.append(q2)
        alt_queue += extra
        if len(alt_queue) > 9:
            alt_queue = alt_queue[:9]
    states = 0
    for pin in alt_queue:
        if time.time() > deadline:
            break
        sol = _solve_mod_k(effects, pin, k, targets)
        if sol is None:
            continue
        toks = []
        for (x, y), n in sorted(sol.items()):
            toks += [C(x + 2, y + 2)] * n
        if not toks:
            continue
        final, won = chain(backend, root, toks, stop_on_level=target)
        states += 1
        if won is not None:
            return _result(True, handle=won, depth=len(won.path),
                           states=states, wall=time.time() - t0,
                           reason="solved", clues=n_clues, cycle=cyc)
    return _result(False, states=states, wall=time.time() - t0,
                   reason="spec_failed")


# --------------------------------------------------------------------------
# tn36_program — program-register machines
# --------------------------------------------------------------------------

def _click_targets(g: np.ndarray, cap: int = 150) -> list[tuple[int, int]]:
    rows = g.tolist()
    bg = _background_color(rows)
    out = []
    for c in _components(rows):
        if c["color"] == bg or c["size"] > 400:
            continue
        x, y = c["centroid"]
        out.append((int(x), int(y)))
        if len(out) >= cap:
            break
    return out


def _tn36_probe(core, root, g0, deadline):
    """Probe click targets: classify into period-2 toggle cells and animated
    run buttons. Returns (toggles, runs) as lists of (x, y)."""
    backend = core.backend
    mb = core.mask_bool(g0.shape)
    toggles: list[tuple[int, int]] = []
    runs: list[tuple[int, int]] = []
    for (x, y) in _click_targets(g0):
        if time.time() > deadline:
            break
        c1 = step1(backend, root, C(x, y))
        if c1 is None or c1.obs is None:
            continue
        if n_layers(c1.obs) >= 3:
            runs.append((x, y))
            continue
        g1 = settled(c1.obs)
        if g1.shape != g0.shape:
            continue
        d1 = masked_diff(g0, g1, mb)
        nd = int(d1.sum())
        if not (1 <= nd <= 16):
            continue
        c2 = step1(backend, c1, C(x, y))
        if c2 is None or c2.obs is None:
            continue
        if masked_eq(settled(c2.obs), g0, mb):
            toggles.append((x, y))
    return toggles, runs


class _H:
    """Minimal Handle-compatible carrier (obs/depth/path/env)."""

    __slots__ = ("obs", "depth", "path", "env")

    def __init__(self, obs, depth, path, env):
        self.obs = obs
        self.depth = depth
        self.path = path
        self.env = env


def _apply_direct(env, tok):
    from arcengine import GameAction

    if tok[0] == "C":
        return env.step(GameAction.ACTION6,
                        data={"x": int(tok[1]), "y": int(tok[2])})
    return env.step(GameAction.from_id(int(tok[1])))


def _replay_fresh(backend, toks, target):
    """Reset the backend's PERSISTENT env and replay toks with DIRECT steps
    (no deepcopy). tn36's opcode table is a dict of lambdas closing over
    `self`; copy.deepcopy leaves those lambdas bound to the ORIGINAL machine,
    so program runs on snapshot copies mutate the wrong object and never win
    (measured 2026-08-25: identical frames through the toggle clicks, then
    the run click levels on the raw env and no-ops on the copy — this is
    also why the generic lanes could not crack tn36's 1024-state register).
    Returns (won_handle | None, final_obs | None)."""
    from arcengine import GameState

    env = backend.env
    obs = env.reset()
    backend.resets += 1
    backend.actions_spent += 1
    if obs is None:
        return None, None
    path = []
    for tok in toks:
        obs = _apply_direct(env, tok)
        backend.actions_spent += 1
        if obs is None:
            return None, None
        path.append(tok)
        if lv(obs) >= target or obs.state == GameState.WIN:
            return _H(obs, len(path), path, env), obs
        if obs.state == GameState.GAME_OVER:
            return None, None
    return None, obs


def _tn36_slots(toggles: list[tuple[int, int]]):
    """Group toggle cells into slots by x-cluster (cells within 4 px)."""
    if not toggles:
        return []
    xs = sorted(set(x for (x, y) in toggles))
    groups: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= 3:
            groups[-1].append(x)
        else:
            groups.append([x])
    slots = []
    for grp in groups:
        cells = sorted([(x, y) for (x, y) in toggles if x in grp],
                       key=lambda p: (p[1], p[0]))
        slots.append(cells)
    return slots


def detect_tn36(core) -> bool:
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return False
    av = avail_of(root.obs)
    if 6 not in av or (av - {0, 6}):
        return False                      # click-only games
    g0 = settled(root.obs)
    # must NOT look like an ft09 tile board (ordering guard, cheap)
    if len(_uniform_blocks(g0)) >= 4:
        return False
    deadline = time.time() + 30
    toggles, runs = _tn36_probe(core, root, g0, deadline)
    if len(toggles) < 4 or not runs:
        return False
    slots = _tn36_slots(toggles)
    # a register: >= 3 slots in a horizontal band
    if len(slots) < 3:
        return False
    ys = [min(y for (_, y) in s) for s in slots]
    return max(ys) - min(ys) <= 6


def solve_tn36(core, target: int, budget_s: float) -> dict:
    t0 = time.time()
    deadline = t0 + budget_s
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return _result(False, wall=time.time() - t0, reason="reset_failed")
    if lv(root.obs) >= target:
        return _result(True, handle=root, depth=0, states=1,
                       wall=time.time() - t0, reason="solved")
    g0 = settled(root.obs)
    toggles, runs = _tn36_probe(core, root, g0, min(deadline, t0 + 60))
    slots = _tn36_slots(toggles)
    slots = [s for s in slots if len(s) <= 6]
    if len(slots) < 2 or not runs:
        return _result(False, wall=time.time() - t0, reason="no_register")

    # ACTIVE cells can be background-colored (tn36 L1: the '5' swatches of
    # value-3 slots are invisible components), so whole slots go missing
    # from the centroid probe. Extrapolate the slot row by its x-pitch and
    # period-2-probe the candidate cell positions.
    mb = core.mask_bool(g0.shape)
    xs = sorted(min(x for (x, _) in s) for s in slots)
    pitch = min((b - a) for a, b in zip(xs, xs[1:])) if len(xs) > 1 else 0
    y_rows = sorted({y for s in slots for (_, y) in s})
    if pitch >= 3:
        cand_xs = set()
        for base_x in range(xs[0] - 3 * pitch, xs[-1] + 3 * pitch + 1, pitch):
            if not any(abs(base_x - x) <= 2 for x in xs):
                cand_xs.add(base_x)
        for cx in sorted(cand_xs):
            if not (0 <= cx < g0.shape[1]):
                continue
            found = []
            for cy in y_rows:
                c1 = step1(backend, root, C(cx, cy))
                if c1 is None or c1.obs is None or n_layers(c1.obs) >= 3:
                    continue
                g1 = settled(c1.obs)
                if g1.shape != g0.shape:
                    continue
                nd = int(masked_diff(g0, g1, mb).sum())
                if not (1 <= nd <= 16):
                    continue
                c2 = step1(backend, c1, C(cx, cy))
                if c2 is not None and c2.obs is not None \
                        and masked_eq(settled(c2.obs), g0, mb):
                    found.append((cx, cy))
            if len(found) == len(y_rows):
                slots.append(sorted(found, key=lambda p: (p[1], p[0])))
        slots.sort(key=lambda s: s[0][0])
    ncell = len(slots[0])
    if any(len(s) != ncell for s in slots) or ncell > 6:
        return _result(False, wall=time.time() - t0, reason="ragged_slots")

    # visual class of each cell: pixel signature of a 3x3 window; the class
    # bit of slot s, row i is "does it look like slot 0's row-i cell". A
    # REGISTER pattern p (bitmask over rows) is realized by clicking, in
    # each slot, the row-i cells whose class differs from bit i of p — this
    # reaches uniform register states from ANY initial register (the L1
    # start [3,0,3,0,0] is non-uniform; uniform CLICK patterns cannot).
    def cell_sig(x, y):
        return g0[max(0, y - 1):y + 2, max(0, x - 1):x + 2].tobytes()

    ref = [cell_sig(*slots[0][i]) for i in range(ncell)]
    cls = [[1 if cell_sig(*s[i]) == ref[i] else 0 for i in range(ncell)]
           for s in slots]

    def toks_for(patterns: list[int]) -> list[tuple]:
        toks = []
        for s_i, (s, patt) in enumerate(zip(slots, patterns)):
            for i in range(ncell):
                if cls[s_i][i] != (patt >> i & 1):
                    toks.append(C(*s[i]))
        return toks

    states = 0
    moving: set[int] = set()

    def try_config(patterns: list[int]):
        """Config test by DIRECT replay on the persistent env (deepcopy
        breaks tn36's lambda-captured opcode table — see _replay_fresh)."""
        nonlocal states
        toggle_toks = toks_for(patterns)
        for (rx, ry) in runs[:3]:
            won, final_obs = _replay_fresh(
                backend, toggle_toks + [C(rx, ry)], target)
            states += 1
            if won is not None:
                return won
            if final_obs is not None and len(set(patterns)) == 1 \
                    and n_layers(final_obs) >= 3:
                a = np.asarray(final_obs.frame)
                if (a[0] != a[a.shape[0] // 2]).any():
                    moving.add(patterns[0])   # this opcode animates: vocab
        return None

    # phase 1: uniform register patterns (2^ncell <= 64); covers the
    # engine-verified tn36 L1 ([3,3,3,3,3]) and L2 ([33,33,33,33]) solutions
    for patt in range(1 << ncell):
        if time.time() > deadline:
            return _result(False, states=states, wall=time.time() - t0,
                           reason="timeout")
        won = try_config([patt] * len(slots))
        if won is not None:
            return _result(True, handle=won, depth=len(won.path),
                           states=states, wall=time.time() - t0,
                           reason="solved", phase="uniform")
    # phase 2: mixed programs over the movement vocabulary (+ class-0)
    vocab = sorted(moving | {0})
    if 1 < len(vocab) <= 6 and len(slots) <= 6 \
            and len(vocab) ** len(slots) <= 4096:
        import itertools

        combos = sorted(itertools.product(vocab, repeat=len(slots)),
                        key=lambda cmb: sum(bin(p).count("1") for p in cmb))
        for cmb in combos:
            if time.time() > deadline:
                return _result(False, states=states,
                               wall=time.time() - t0, reason="timeout")
            if len(set(cmb)) == 1:
                continue              # phase 1 covered
            won = try_config(list(cmb))
            if won is not None:
                return _result(True, handle=won, depth=len(won.path),
                               states=states, wall=time.time() - t0,
                               reason="solved", phase="mixed")
    return _result(False, states=states, wall=time.time() - t0,
                   reason="spec_failed")


# --------------------------------------------------------------------------
# sc25_glyph — glyph-cast + maze
# --------------------------------------------------------------------------

def _ring_panels(g: np.ndarray) -> list[dict]:
    """Candidate glyph panels: ring (hollow border) OR lattice (grid of
    cell borders) components framing a 3x3 cell area — the sc25 target
    panel is a ring, the slot panel a connected 3x3 grid frame."""
    rows = g.tolist()
    bg = _background_color(rows)
    out = []
    for c in _components(rows):
        if c["color"] == bg or c["size"] < 24:
            continue
        r0, c0, r1, c1 = c["bbox"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 9 or w < 9 or h > 30 or w > 30:
            continue
        area = h * w
        if c["size"] > 0.88 * area:
            continue                   # too filled to frame anything
        panel = dict(color=c["color"], bbox=(r0, c0, r1, c1),
                     size=c["size"], area=area)
        sub = (g[r0:r1 + 1, c0:c1 + 1] != c["color"]).tolist()
        holes = [h for h in _components(sub)
                 if h["color"] == 1 and h["size"] >= 2]
        if len(holes) == 9:
            out.append(panel)              # 3x3 grid frame (slot panel)
        elif 1 <= len(holes) <= 4 and c["size"] <= 2 * 2 * (h + w):
            out.append(panel)              # hollow ring (target panel)
    return out


def _panel_cells(panel: dict, g: np.ndarray | None = None
                 ) -> list[tuple[int, int, int, int]]:
    """(i, j, cx, cy) centers of the panel's 3x3 cells.

    With `g` given, prefer the actual HOLES of the frame color (grid panels:
    exactly 9 non-frame regions -> exact cell centroids, robust to pitch);
    otherwise (or for ring panels with one open interior) fall back to an
    arithmetic 3x3 subdivision of the interior."""
    r0, c0, r1, c1 = panel["bbox"]
    if g is not None:
        sub = (g[r0:r1 + 1, c0:c1 + 1] != panel["color"]).tolist()
        holes = [c for c in _components(sub) if c["color"] == 1]
        holes = [h for h in holes if h["size"] >= 2]
        if len(holes) == 9:
            pts = sorted(((h["centroid"][1] + r0, h["centroid"][0] + c0)
                          for h in holes))
            rows_sorted = sorted(pts, key=lambda p: p[0])
            cells = []
            for j in range(3):
                band = sorted(rows_sorted[3 * j:3 * j + 3],
                              key=lambda p: p[1])
                for i, (cy, cx) in enumerate(band):
                    cells.append((i, j, int(cx), int(cy)))
            return cells
    ih, iw = (r1 - 1) - (r0 + 2) + 1, (c1 - 1) - (c0 + 2) + 1
    cells = []
    for j in range(3):
        for i in range(3):
            cy = r0 + 2 + (2 * j + 1) * ih // 6
            cx = c0 + 2 + (2 * i + 1) * iw // 6
            cells.append((i, j, cx, cy))
    return cells


def _cell_sig(g: np.ndarray, cx: int, cy: int, rad: int = 2) -> tuple:
    sub = g[max(0, cy - rad):cy + rad + 1, max(0, cx - rad):cx + rad + 1]
    vals, counts = np.unique(sub, return_counts=True)
    return tuple(sorted(zip(vals.tolist(), counts.tolist())))


def _marked_cells(g: np.ndarray, panel: dict) -> set[tuple[int, int]]:
    """Marked/lit cells of a panel.

    Grid panels (9 frame holes): a cell is lit when its hole's pixel
    signature differs from the majority hole. Ring panels (one open
    interior): the mark color is the interior color distinct from the
    dominant fill; a 3x3-thirds cell is marked when it holds >= 2 mark
    pixels."""
    from collections import Counter

    r0, c0, r1, c1 = panel["bbox"]
    sub = (g[r0:r1 + 1, c0:c1 + 1] != panel["color"]).tolist()
    holes = [h for h in _components(sub)
             if h["color"] == 1 and h["size"] >= 2]
    if len(holes) == 9:
        cells = _panel_cells(panel, g)
        sigs = {(i, j): _cell_sig(g, cx, cy) for (i, j, cx, cy) in cells}
        common = Counter(sigs.values()).most_common(1)[0][0]
        return {ij for ij, s in sigs.items() if s != common}
    inner = g[r0 + 1:r1, c0 + 1:c1]
    vals = inner[inner != panel["color"]]
    if vals.size == 0:
        return set()
    fill = int(np.bincount(vals).argmax())
    marks = (inner != fill) & (inner != panel["color"])
    if not marks.any():
        return set()
    ih, iw = inner.shape
    out = set()
    for j in range(3):
        for i in range(3):
            band = marks[j * ih // 3:(j + 1) * ih // 3,
                         i * iw // 3:(i + 1) * iw // 3]
            if int(band.sum()) >= 2:
                out.add((i, j))
    return out


def detect_sc25(core) -> bool:
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return False
    av = avail_of(root.obs)
    basics = sorted(a for a in av if a in (1, 2, 3, 4, 5))
    if 6 not in av or len(basics) < 2:
        return False
    g0 = settled(root.obs)
    # every basic action must be a no-op at the root
    for a in basics:
        child = step1(backend, root, S(a))
        if child is None or child.obs is None:
            return False
        g1 = settled(child.obs)
        if g1.shape != g0.shape or (g1 != g0).any():
            return False
    panels = _ring_panels(g0)
    if len(panels) < 2:
        return False
    # one panel marked (target), and clicking a cell of the other changes
    # the frame (slot panel responds)
    panels = sorted(panels, key=lambda p: p["area"])
    tgt, slot = panels[0], panels[-1]
    if not _marked_cells(g0, tgt):
        tgt, slot = slot, tgt
        if not _marked_cells(g0, tgt):
            return False
    # a PRIME click (on the target/spell icon) is consumed before slot
    # clicks register (measured on sc25: slot clicks are no-ops unprimed)
    r0, c0, r1, c1 = tgt["bbox"]
    prime = C((c0 + c1) // 2, (r0 + r1) // 2)
    primed = step1(backend, root, prime)
    if primed is None or primed.obs is None:
        return False
    (_, _, cx, cy) = _panel_cells(slot, g0)[0]
    child = step1(backend, primed, C(cx, cy))
    if child is None or child.obs is None:
        return False
    return (settled(child.obs) != settled(primed.obs)).any()


def solve_sc25(core, target: int, budget_s: float) -> dict:
    from arcengine import GameState

    t0 = time.time()
    deadline = t0 + budget_s
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return _result(False, wall=time.time() - t0, reason="reset_failed")
    if lv(root.obs) >= target:
        return _result(True, handle=root, depth=0, states=1,
                       wall=time.time() - t0, reason="solved")
    g0 = settled(root.obs)
    panels = sorted(_ring_panels(g0), key=lambda p: p["area"])
    if len(panels) < 2:
        return _result(False, wall=time.time() - t0, reason="no_panels")
    basics = sorted(a for a in avail_of(root.obs) if a in (1, 2, 3, 4))

    def cast_and_bfs(tgt, slot):
        marked = _marked_cells(g0, tgt)
        if not marked or len(marked) > 8:
            return None
        cells = {(i, j): (cx, cy) for (i, j, cx, cy) in _panel_cells(slot, g0)}
        if any(ij not in cells for ij in marked):
            return None
        # PRIME on the target icon (consumed), then draw the target pattern
        r0, c0, r1, c1 = tgt["bbox"]
        toks = [C((c0 + c1) // 2, (r0 + r1) // 2)]
        toks += [C(*cells[ij]) for ij in sorted(marked)]
        base, won = chain(backend, root, toks, stop_on_level=target)
        if won is not None:
            return won
        if base is None:
            return None
        # BFS over basic actions from the cast state
        from collections import deque

        mb = core.mask_bool(g0.shape)
        seen = set()
        gb = settled(base.obs)
        seen.add(gb.tobytes())
        q = deque([base])
        nodes = 0
        while q and time.time() < deadline and nodes < 12000:
            h = q.popleft()
            for a in basics:
                child = step1(backend, h, S(a))
                nodes += 1
                if child is None or child.obs is None:
                    continue
                if lv(child.obs) >= target \
                        or child.obs.state == GameState.WIN:
                    return child
                if child.obs.state == GameState.GAME_OVER:
                    continue
                gc = settled(child.obs)
                key = gc.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                q.append(child)
        return None

    order = [(panels[0], panels[-1]), (panels[-1], panels[0])]
    states = 0
    for tgt, slot in order:
        if time.time() > deadline:
            break
        states += 1
        won = cast_and_bfs(tgt, slot)
        if won is not None:
            return _result(True, handle=won, depth=len(won.path),
                           states=states, wall=time.time() - t0,
                           reason="solved")
    return _result(False, states=states, wall=time.time() - t0,
                   reason="spec_failed")


# --------------------------------------------------------------------------
# wa30_grabdrag — grab-drag block puzzles (A* planner port)
# --------------------------------------------------------------------------

CELL = 4
DELTAS = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}


def _facing_of(dx, dy):
    if dy < 0:
        return 0
    if dx > 0:
        return 90
    if dy > 0:
        return 180
    return 270


def _faced_cell(ax, ay, facing):
    if facing == 0:
        return (ax, ay - CELL)
    if facing == 180:
        return (ax, ay + CELL)
    if facing == 90:
        return (ax + CELL, ay)
    return (ax - CELL, ay)


class _DragModel:
    def __init__(self, walls: set):
        self.walls = walls

    def step(self, state, action):
        ax, ay, bx, by, facing, held = state
        if action == 5:
            if held:
                return (ax, ay, bx, by, facing, False)
            if _faced_cell(ax, ay, facing) == (bx, by):
                return (ax, ay, bx, by, facing, True)
            return state
        dx, dy = DELTAS[action]
        if not held:
            nf = _facing_of(dx, dy)
            tgt = (ax + dx, ay + dy)
            if tgt not in self.walls and tgt != (bx, by):
                return (tgt[0], tgt[1], bx, by, nf, held)
            return (ax, ay, bx, by, nf, held)
        nav = (ax + dx, ay + dy)
        nbl = (bx + dx, by + dy)
        ok = ((nav not in self.walls or nav == (bx, by)) and
              (nbl not in self.walls or nbl == (ax, ay)))
        if ok:
            return (nav[0], nav[1], nbl[0], nbl[1], facing, held)
        return state


def _astar_leg(model, start, goal_block_xy, node_cap=200000):
    import heapq

    gx, gy = goal_block_xy

    def h(s):
        return (abs(s[2] - gx) + abs(s[3] - gy)) // CELL

    def is_win(s):
        return (s[2], s[3]) == goal_block_xy and not s[5]

    if is_win(start):
        return []
    counter = 0
    pq = [(h(start), 0, counter, start, [])]
    best = {start: 0}
    nodes = 0
    while pq and nodes < node_cap:
        f, g, _, s, path = heapq.heappop(pq)
        if g > best.get(s, 1 << 30):
            continue
        nodes += 1
        for a in (1, 2, 3, 4, 5):
            ns = model.step(s, a)
            ng = g + 1
            if ng >= best.get(ns, 1 << 30):
                continue
            if is_win(ns):
                return path + [a]
            best[ns] = ng
            counter += 1
            heapq.heappush(pq, (ng + h(ns), ng, counter, ns, path + [a]))
    return None


def _wa30_perceive(g: np.ndarray, av_color: int):
    from collections import Counter

    rows = g.tolist()
    bg = _background_color(rows)
    comps = [c for c in _components(rows)
             if c["color"] != bg and c["bbox"][0] < 60]
    avs = [c for c in comps if c["color"] == av_color and 6 <= c["size"] <= 40]
    if not avs:
        return None
    r0, c0, r1, c1 = avs[0]["bbox"]
    avatar = ((c0 // CELL) * CELL, (r0 // CELL) * CELL)
    raw_blocks = []
    for c in comps:
        if c["color"] == av_color or not (6 <= c["size"] <= 40):
            continue
        r0, c0_, r1, c1_ = c["bbox"]
        if r1 - r0 != 3 or c1_ - c0_ != 3:
            continue
        inner = g[r0 + 1:r0 + 3, c0_ + 1:c0_ + 3]
        if (inner != c["color"]).any():
            mc = int(inner[inner != c["color"]][0])
            raw_blocks.append(((c0_ // CELL) * CELL, (r0 // CELL) * CELL, mc))
    if not raw_blocks:
        return None
    marker = Counter(m for (_, _, m) in raw_blocks).most_common(1)[0][0]
    blocks = sorted({(x, y) for (x, y, m) in raw_blocks if m == marker})
    pads = []
    for c in comps:
        if c["color"] == marker and c["size"] >= 16:
            r0, c0_, r1, c1_ = c["bbox"]
            for x in range((c0_ // CELL) * CELL, c1_ + 1, CELL):
                for y in range((r0 // CELL) * CELL, r1 + 1, CELL):
                    pads.append((x, y))
    pads = sorted(set(pads))
    if not pads or len(blocks) > len(pads):
        return None
    known = {avatar} | set(blocks) | set(pads)
    walls = set()
    for yy in range(0, 60, CELL):
        for xx in range(0, 64, CELL):
            cellpix = g[yy:yy + CELL, xx:xx + CELL]
            if (cellpix != bg).any() and (xx, yy) not in known:
                walls.add((xx, yy))
    for i in range(0, 64, CELL):
        walls |= {(-CELL, i), (64, i), (i, -CELL), (i, 64)}
    latent = {(x, y) for (x, y) in blocks
              if ((x - CELL, y) in walls and (x + CELL, y) in walls)
              or ((x, y - CELL) in walls and (x, y + CELL) in walls)}
    return avatar, blocks, pads, walls, latent


def _learn_avatar_color(backend, root, g0) -> int | None:
    bg = _background_color(g0.tolist())
    for a in (1, 2, 3, 4):
        child = step1(backend, root, S(a))
        if child is None or child.obs is None:
            continue
        g1 = settled(child.obs)
        if g1.shape != g0.shape:
            continue
        d = g1 != g0
        n = int(d.sum())
        if 0 < n <= 64:
            ys, xs = np.nonzero(d)
            vals = g1[ys, xs]
            vals = vals[vals != bg]
            if len(vals):
                return int(np.bincount(vals).argmax())
    return None


def detect_wa30(core) -> bool:
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return False
    av = avail_of(root.obs)
    if 6 in av or not {1, 2, 3, 4, 5} <= av:
        return False
    g0 = settled(root.obs)
    # ACTION5 must be a no-op at the root (grab with nothing adjacent)
    c5 = step1(backend, root, S(5))
    if c5 is None or c5.obs is None:
        return False
    if (settled(c5.obs) != g0).any():
        return False
    av_color = _learn_avatar_color(backend, root, g0)
    if av_color is None:
        return False
    per = _wa30_perceive(g0, av_color)
    if per is None:
        return False
    avatar, blocks, pads, walls, latent = per
    # decisive drag probe: walk adjacent to a block, grab, move away, and
    # require the block to follow (pull, not push)
    from collections import deque

    model = _DragModel(walls | set(blocks) | latent)
    b = blocks[0]
    targets = {}
    for a_id, (dx, dy) in DELTAS.items():
        adj = (b[0] - dx, b[1] - dy)     # stand at adj, face (dx,dy) to b
        away = (adj[0] - dx, adj[1] - dy)
        if adj not in model.walls and away not in model.walls \
                and away != b:
            targets[adj] = (a_id, away)
    if not targets:
        return False
    seen = {avatar}
    q = deque([(avatar, [])])
    plan = None
    while q and plan is None:
        (pos, path) = q.popleft()
        if len(seen) > 400:
            break
        for a_id, (dx, dy) in DELTAS.items():
            np_ = (pos[0] + dx, pos[1] + dy)
            if np_ in targets:
                a_to, (a_face, away) = np_, targets[np_]
                # face the block, grab, then step in the opposite direction
                face_a = next(a for a, d in DELTAS.items()
                              if (np_[0] + d[0], np_[1] + d[1]) == b)
                back_a = next(a for a, d in DELTAS.items()
                              if (d[0], d[1]) == (-DELTAS[face_a][0],
                                                  -DELTAS[face_a][1]))
                plan = path + [a_id, face_a, 5, back_a]
                break
            if np_ in seen or np_ in model.walls or np_ == b:
                continue
            seen.add(np_)
            q.append((np_, path + [a_id]))
    if plan is None:
        return False
    toks = [S(a) for a in plan]
    final, _ = chain(core.backend, root, toks)
    if final is None or final.obs is None:
        return False
    g1 = settled(final.obs)
    per1 = _wa30_perceive(g1, av_color)
    if per1 is None:
        return False
    blocks1 = per1[1]
    if b in blocks1:
        return False                       # block did not move at all
    face_a = plan[-3]
    fdx, fdy = DELTAS[face_a]
    pushed = (b[0] + fdx, b[1] + fdy)
    if pushed in blocks1 and len(blocks1) == len(blocks):
        return False                       # push mechanics, not grab-drag
    # dragged toward the avatar (count preserved) or held-and-hidden
    dragged = (b[0] - fdx, b[1] - fdy)
    return (dragged in blocks1 and len(blocks1) == len(blocks)) \
        or len(blocks1) == len(blocks) - 1


def solve_wa30(core, target: int, budget_s: float) -> dict:
    from itertools import permutations

    t0 = time.time()
    deadline = t0 + budget_s
    backend = core.backend
    root = backend.root()
    if root is None or root.obs is None:
        return _result(False, wall=time.time() - t0, reason="reset_failed")
    if lv(root.obs) >= target:
        return _result(True, handle=root, depth=0, states=1,
                       wall=time.time() - t0, reason="solved")
    g0 = settled(root.obs)
    av_color = _learn_avatar_color(backend, root, g0)
    if av_color is None:
        # a boxed-in start (all four moves blocked) hides the avatar from
        # the move probe; the color is game-stable, so reuse last level's
        av_color = getattr(core, "_wa30_avcolor", None)
    if av_color is None:
        return _result(False, wall=time.time() - t0, reason="no_avatar")
    core._wa30_avcolor = av_color
    per = _wa30_perceive(g0, av_color)
    if per is None:
        return _result(False, wall=time.time() - t0, reason="no_percept")
    avatar, blocks, pads, walls, latent = per

    def greedy_assign():
        pairs = sorted(((abs(b[0] - p[0]) + abs(b[1] - p[1]), bi, pi)
                        for bi, b in enumerate(blocks)
                        for pi, p in enumerate(pads)))
        assign = {}
        used = set()
        for _, bi, pi in pairs:
            if bi in assign or pi in used:
                continue
            assign[bi] = pads[pi]
            used.add(pi)
        return ([assign[i] for i in range(len(blocks))]
                if len(assign) == len(blocks) else None)

    assign = greedy_assign()
    if assign is None:
        return _result(False, wall=time.time() - t0, reason="no_assign")

    def plan_sequence(facing0, order):
        ax, ay, facing = avatar[0], avatar[1], facing0
        placed, full = [], []
        for idx, i in enumerate(order):
            b = blocks[i]
            others = set(blocks[j] for j in order[idx + 1:])
            lat = latent - {b}
            model = _DragModel(walls | others | lat | set(placed))
            path = _astar_leg(model, (ax, ay, b[0], b[1], facing, False),
                              assign[i])
            if path is None:
                return None
            s = (ax, ay, b[0], b[1], facing, False)
            for a in path:
                s = model.step(s, a)
            ax, ay, facing = s[0], s[1], s[4]
            placed.append(assign[i])
            full.extend(path)
        return full

    orders = (list(permutations(range(len(blocks))))
              if len(blocks) <= 5 else [tuple(range(len(blocks)))])
    cand = []
    for facing0 in (0, 90, 180, 270):
        for order in orders:
            if time.time() > deadline:
                break
            plan = plan_sequence(facing0, order)
            if plan is not None:
                cand.append((len(plan), plan))
    cand.sort(key=lambda t: t[0])
    states = 0
    for _, plan in cand[:40]:
        if time.time() > deadline:
            break
        states += 1
        toks = [S(a) for a in plan]
        final, won = chain(backend, root, toks, stop_on_level=target)
        if won is not None:
            return _result(True, handle=won, depth=len(won.path),
                           states=states, wall=time.time() - t0,
                           reason="solved", plans=len(cand))
    return _result(False, states=states, wall=time.time() - t0,
                   reason="spec_failed", plans=len(cand))


# --------------------------------------------------------------------------
# registry / dispatch
# --------------------------------------------------------------------------

# detection order matters: ft09 boards would also pass the tn36 probe
# (uniform-tile toggles + the L0 hint animation), so ft09 runs first and
# tn36 explicitly rejects tile boards.
DETECTORS = [
    ("ft09_gf2", detect_ft09),
    ("tn36_program", detect_tn36),
    ("sc25_glyph", detect_sc25),
    ("wa30_grabdrag", detect_wa30),
]

SOLVERS = {
    "ft09_gf2": solve_ft09,
    "tn36_program": solve_tn36,
    "sc25_glyph": solve_sc25,
    "wa30_grabdrag": solve_wa30,
}


def detect(core) -> str | None:
    """First matching specialist class for this game (frame-only probes on
    the backend; call after warmup_and_freeze so the mask is usable)."""
    for name, fn in DETECTORS:
        try:
            if fn(core):
                return name
        except Exception:
            continue
    return None


def detect_matrix(core) -> dict[str, bool]:
    """Every detector's verdict (for the 25x4 false-positive matrix)."""
    out = {}
    for name, fn in DETECTORS:
        try:
            out[name] = bool(fn(core))
        except Exception:
            out[name] = False
    return out


def solve_level(core, name: str, target: int, budget_s: float) -> dict:
    fn = SOLVERS.get(name)
    if fn is None:
        return _result(False, reason="unknown_specialist")
    try:
        res = fn(core, target, budget_s)
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        res = _result(False, reason=f"spec_error:{type(e).__name__}")
    res["algo"] = f"spec:{name}"
    return res
