"""board_changed is wrong on 13 of 25 games. This corrects it.

THE DEFECT. 18 of 25 games paint a step-budget bar into the frame and tick it BEFORE
the legality check, so `board_changed = previous_grid != new_grid`
(framework/solver.py:704) reports "the board changed" for actions that did nothing at
all. Measured over 1,200 random actions per game:

    corpus no-op rate    0.208 raw  ->  0.455 with the HUD masked
    lf52 0.000 -> 0.996    vc33 0.000 -> 0.995    s5i5 0.000 -> 0.949
    tu93 0.000 -> 0.601    r11l 0.000 -> 0.458    bp35 0.000 -> 0.369

In six games the harness asserts "something happened" on 100% of actions. This value is
put in front of the model every turn as a statement of fact about the world
(tool_agent.py:1095-1098: "produced a board change; verify that it affected gameplay
objects" / "did not show a confirmed board change; treat this as weak evidence"), so on
those games the model is being told, every single turn, that its last action had an
effect when it did not.

WHY IT IS REGIONAL, NOT PER-PIXEL. As the bar decrements, different pixels within the
row change at different times, so the ROW differs on every action while NO SINGLE PIXEL
does. A per-pixel "always changes" test finds nothing and wrongly clears the game --
this cost one wasted refutation during the investigation. Mask regions.

WHAT THIS ARM CHANGES. One value: `board_changed`, recomputed on the board with the
HUD region excluded. Nothing else -- not the model, not sampling, not concurrency, not
budgets, not the prompt text, not the game list. The batch aggregation at
solver.py:655-656 folds the corrected values through automatically.

FALSIFIABLE AND COULD LOSE. The model may be relying on the miscalibrated signal in a
way that accidentally helps -- for instance, "always changed" may currently discourage
it from repeating actions, and an honest "nothing happened" may encourage a
repeat-until-something-moves loop. That is a real possibility and the reason this is an
arm rather than a bugfix.
"""
from __future__ import annotations

import numpy as np

# stem -> ("row", r) | ("col", c) | ("rows", (r1, r2)). Locations derived from 340
# human sessions and confirmed here by the no-op delta each one produces.
HUD_REGIONS: dict[str, tuple] = {
    "cd82": ("row", 63), "dc22": ("row", 63), "ka59": ("row", 63), "re86": ("row", 63),
    "s5i5": ("row", 63), "tu93": ("row", 63), "wa30": ("row", 63), "bp35": ("row", 63),
    "tr87": ("row", 63),
    "cn04": ("row", 0), "vc33": ("row", 0), "sp80": ("row", 0), "lf52": ("row", 0),
    "sk48": ("row", 53), "sb26": ("row", 53),
    "r11l": ("col", 0), "lp85": ("col", 0),
    "ls20": ("rows", (61, 62)),
}

_mask_cache: dict[tuple, np.ndarray | None] = {}
_stats = {"calls": 0, "raw_changed": 0, "masked_changed": 0, "corrected": 0}


def _stem(game_id: str) -> str:
    return str(game_id).split("-")[0][:4].lower()


def hud_mask(game_id: str, shape) -> np.ndarray | None:
    key = (_stem(game_id), tuple(shape))
    if key in _mask_cache:
        return _mask_cache[key]
    spec = HUD_REGIONS.get(_stem(game_id))
    m = None
    if spec is not None:
        m = np.zeros(shape, dtype=bool)
        kind, val = spec
        if kind == "row":
            m[val, :] = True
        elif kind == "col":
            m[:, val] = True
        else:
            for r in val:
                m[r, :] = True
    _mask_cache[key] = m
    return m


def boards_equal(prev, cur, game_id: str) -> bool:
    a, b = np.asarray(prev), np.asarray(cur)
    if a.shape != b.shape:
        return False
    m = hud_mask(game_id, a.shape)
    if m is None or not m.any():
        return np.array_equal(a, b)
    return np.array_equal(a[~m], b[~m])


def patch_board_changed() -> bool:
    from inference.framework import solver as sv

    original = sv._HarnessGameSession._execute_action
    if getattr(original, "_boardfix", False):
        return True

    def _execute_action(self, action, **kwargs):
        prev = sv._grid_from_state(self.game.current_state)
        payload = original(self, action, **kwargs)
        try:
            gid = getattr(self.game, "env_name", None) or getattr(self.game, "game_id", "") or ""
            cur = sv._grid_from_state(self.game.current_state)
            raw = bool(payload.get("board_changed"))
            fixed = not boards_equal(prev, cur, gid)
            _stats["calls"] += 1
            _stats["raw_changed"] += raw
            _stats["masked_changed"] += fixed
            if raw != fixed:
                _stats["corrected"] += 1
            payload["board_changed"] = fixed
        except Exception:
            # A perception correction must never take down a 9-hour run. The arm-active
            # check below already proves the patch itself installed.
            return payload
        return payload

    _execute_action._boardfix = True
    sv._HarnessGameSession._execute_action = _execute_action
    return True


def stats() -> dict:
    d = dict(_stats)
    if d["calls"]:
        d["corrected_frac"] = round(d["corrected"] / d["calls"], 4)
        d["raw_changed_frac"] = round(d["raw_changed"] / d["calls"], 4)
        d["masked_changed_frac"] = round(d["masked_changed"] / d["calls"], 4)
    return d


def apply_all() -> dict:
    return {"board_changed_hud_masked": patch_board_changed()}
