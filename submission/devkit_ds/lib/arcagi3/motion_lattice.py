"""The agent's motion lattice: the discrete grid its moves actually reach.

Frames are 64x64 PIXELS but the agent moves in fixed pixel jumps (e.g. ls20 = 5 px/step), so the
cells it can occupy form a lattice anchored at its start. Planning targets must lie on this lattice
to be reachable; `snap` projects an arbitrary pixel position to the nearest lattice cell.
"""
from __future__ import annotations


def lattice_pitch(deltas: dict) -> tuple[int, int]:
    """Per-axis pixel pitch = the smallest nonzero |displacement| on each axis. 0 = no motion on
    that axis (caller must treat pitch 0 as 'do not snap that axis')."""
    prs = [abs(dr) for dr, dc in deltas.values() if dr != 0]
    pcs = [abs(dc) for dr, dc in deltas.values() if dc != 0]
    return (min(prs) if prs else 0, min(pcs) if pcs else 0)


def snap(pos, origin, pitch, bounds) -> tuple[int, int]:
    """Project `pos` (r,c) to the nearest lattice cell anchored at `origin` with `pitch`, clamped
    to `bounds` (H,W). A 0 pitch on an axis keeps that axis at the origin's coordinate."""
    r, c = pos
    orr, oc = origin
    pr, pc = pitch
    H, W = bounds
    nr = orr if pr == 0 else orr + round((r - orr) / pr) * pr
    nc = oc if pc == 0 else oc + round((c - oc) / pc) * pc
    nr = max(0, min(H - 1, int(nr)))
    nc = max(0, min(W - 1, int(nc)))
    return (nr, nc)
