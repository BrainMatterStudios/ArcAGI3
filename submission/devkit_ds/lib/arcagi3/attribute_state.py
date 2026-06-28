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
    captures shape+orientation but not location. Empty cell-set -> AttrVec(-1, ()).

    The signature is COLOR-AWARE: each cell contributes (dr, dc, color). This is what makes
    ORIENTATION observable for a multi-color rigid avatar whose silhouette is rotation-invariant
    (e.g. a solid 5x5 block whose top rows are a distinct "head" color): a 90-degree rotation leaves
    the silhouette identical but moves the head color, so a position-only signature would miss the
    orientation change while a color-aware one captures it. For a single-color blob this reduces to
    the silhouette (all colors equal), so existing single-color behaviour is preserved.

    `color` is the agent's dominant (most frequent) cell color — stable for multi-color sprites,
    where picking an arbitrary cell would flicker between the head and body colors frame to frame."""
    cells = sorted((int(r), int(c)) for r, c in agent_cells)
    if not cells:
        return AttrVec(color=-1, shape_sig=())
    from collections import Counter
    colors = [int(grid[r, c]) for r, c in cells]
    # dominant color (ties -> smallest value, for determinism)
    color = min(Counter(colors).most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    # translation-normalised but NOT rotation-normalised; color-aware so orientation is encoded.
    shape_sig = tuple((r - r0, c - c0, int(grid[r, c])) for r, c in cells)
    return AttrVec(color=color, shape_sig=shape_sig)
