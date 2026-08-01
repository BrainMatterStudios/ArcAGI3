"""E1 — HUD masking. The precondition for every frame-identity test.

18 of 25 games paint a step-budget bar into the frame and tick it BEFORE the legality
check, so an exact frame comparison misses real no-ops. Measured over 1,200 random
actions per game, the corpus no-op rate is 0.208 raw and 0.455 masked. Seven games
where the raw test reports ZERO no-ops are actually 27-99.6%.

The effect is REGIONAL, not per-pixel: as the bar decrements, different pixels within
the row change at different times, so the row differs on every action while no single
pixel does. A per-pixel "always changes" test finds nothing — which is why an earlier
check in this session wrongly refuted the phenomenon.

Locations come from 340 human sessions and were confirmed here by the delta each mask
produces in the measured no-op rate.

Everything that compares frames must go through `mask_frame` or `frames_equal`:
board_changed, state hashing, novelty, dead-action memory, graph-search node identity.
"""
from __future__ import annotations

import numpy as np

# stem -> spec. ("row", r) | ("col", c) | ("rows", (r1, r2))
HUD_REGIONS: dict[str, tuple] = {
    "cd82": ("row", 63), "dc22": ("row", 63), "ka59": ("row", 63), "re86": ("row", 63),
    "s5i5": ("row", 63), "tu93": ("row", 63), "wa30": ("row", 63), "bp35": ("row", 63),
    "tr87": ("row", 63),
    "cn04": ("row", 0), "vc33": ("row", 0), "sp80": ("row", 0), "lf52": ("row", 0),
    "sk48": ("row", 53), "sb26": ("row", 53),
    "r11l": ("col", 0), "lp85": ("col", 0),
    "ls20": ("rows", (61, 62)),
}

# Measured no-op rate before -> after masking, 1,200 random actions, seed 0.
# Kept in-tree so a future change to the mask can be checked against a known result
# rather than re-derived from scratch.
MEASURED_NOOP = {
    "lf52": (0.000, 0.996), "vc33": (0.000, 0.995), "s5i5": (0.000, 0.949),
    "sb26": (0.625, 0.961), "tu93": (0.000, 0.601), "cd82": (0.225, 0.620),
    "r11l": (0.000, 0.458), "dc22": (0.227, 0.449), "bp35": (0.000, 0.369),
    "ka59": (0.119, 0.348), "cn04": (0.186, 0.319), "sk48": (0.253, 0.298),
    "ls20": (0.004, 0.274), "wa30": (0.133, 0.198), "sp80": (0.000, 0.138),
    "re86": (0.003, 0.008),
}
CORPUS_NOOP_RAW, CORPUS_NOOP_MASKED = 0.208, 0.455


def game_stem(game_id: str) -> str:
    """'tu93-0768757b' -> 'tu93'. Accepts a bare stem unchanged."""
    return game_id.split("-")[0][:4]


def hud_mask(game_id: str, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Boolean mask, True where the frame carries HUD rather than board."""
    m = np.zeros(shape, dtype=bool)
    spec = HUD_REGIONS.get(game_stem(game_id))
    if spec is None:
        return m
    kind, val = spec
    if kind == "row":
        m[val, :] = True
    elif kind == "col":
        m[:, val] = True
    elif kind == "rows":
        for r in val:
            m[r, :] = True
    return m


def mask_frame(frame: np.ndarray, game_id: str, fill: int = 0) -> np.ndarray:
    """Frame with HUD pixels flattened to a constant, so hashes ignore the counter."""
    m = hud_mask(game_id, frame.shape)
    if not m.any():
        return frame
    out = frame.copy()
    out[m] = fill
    return out


def frames_equal(a: np.ndarray, b: np.ndarray, game_id: str) -> bool:
    """Board-equality: the comparison `board_changed` should have been doing."""
    m = hud_mask(game_id, a.shape)
    return np.array_equal(a[~m], b[~m]) if m.any() else np.array_equal(a, b)


def frame_key(frame: np.ndarray, game_id: str) -> bytes:
    """Hashable board identity, HUD removed.

    Use `.tobytes()` on a real array — never `str()` on a numpy frame, which
    truncates with '...' and silently collides every hash.
    """
    return np.ascontiguousarray(mask_frame(frame, game_id), dtype=np.int8).tobytes()


def settled_frame(obs) -> np.ndarray:
    """Last layer of the returned stack. Animation runs to 61 layers on g50t."""
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a
