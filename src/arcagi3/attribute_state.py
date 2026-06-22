"""The agent's OWN mutable attributes (color, shape, orientation) as a hashable vector.

This is the piece the rigid-object stack lacks: in transform-match games the avatar's color/
shape/orientation change when it steps on transformer tiles. We capture them generically from
the agent's cell-set, normalising position out so only intrinsic attributes remain.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AttrVec:
    color: int
    shape_sig: tuple[tuple[int, int], ...]   # rotation-/translation-sensitive signature of the cell pattern


def agent_attributes(grid, agent_cells) -> AttrVec:
    """Extract the agent's intrinsic attributes (color + orientation-sensitive shape signature)
    from its cell-set. Position is normalised out (offsets from the min row/col), so the signature
    captures shape+orientation but not location. Empty cell-set -> AttrVec(-1, ())."""
    cells = sorted((int(r), int(c)) for r, c in agent_cells)
    if not cells:
        return AttrVec(color=-1, shape_sig=())
    color = int(grid[cells[0][0], cells[0][1]])
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    # translation-normalised but NOT rotation-normalised -> orientation is encoded in the sig
    shape_sig = tuple((r - r0, c - c0) for r, c in cells)
    return AttrVec(color=color, shape_sig=shape_sig)
