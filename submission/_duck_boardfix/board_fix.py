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
_stats = {"calls": 0, "raw_changed": 0, "masked_changed": 0, "corrected": 0,
          "unresolved": 0, "masked_games": set(), "unmasked_games": set()}


def _stem(game_id: str) -> str:
    return str(game_id).split("-")[0][:4].lower()


_stem_cache: dict[int, str] = {}
_CLONE_RE = __import__("re").compile(r"^k\d{3}$")


def _resolve_stem(game) -> str:
    """Map a running game to its official stem, and make failure countable.

    THE TRAP THIS EXISTS FOR. Under the competition arcade every game is a CLONE with
    id 'k000'..'k109'; `env_name` is that clone id, not 'tu93'. A naive lookup finds
    nothing in HUD_REGIONS, the mask is empty, and `boards_equal` degrades to plain
    array equality -- the arm becomes byte-identical to baseline and the experiment
    measures nothing while every install check still passes.

    The clone's `private_tags` (carrying `taaf_source_game:`) are stripped client-side,
    but `title` survives and equals the stem uppercased -- verified against a live
    competition arcade, 28/28 clones, zero mismatches.

    Resolution is attempted for EVERY clone id, not only ones whose title happens to be
    in HUD_REGIONS. Otherwise "this game has no HUD" and "stem resolution is broken"
    would be indistinguishable -- both yield no mask -- and a total resolution failure
    would look exactly like a corpus of HUD-free games. `_stats["unresolved"]` counts
    the difference so `assert_fired` can refuse to report on it.
    """
    key = id(game)
    hit = _stem_cache.get(key)
    if hit is not None:
        return hit
    cand = str(getattr(game, "env_name", "") or getattr(game, "game_id", "") or "")
    if _CLONE_RE.match(cand):
        resolved = ""
        for path in ("env", "_env"):
            env = getattr(game, path, None)
            info = getattr(env, "environment_info", None) if env is not None else None
            title = str(getattr(info, "title", "") or "")
            if title:
                resolved = title
                break
        if resolved:
            cand = resolved
        else:
            _stats["unresolved"] += 1
    _stem_cache[key] = cand
    return cand


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
            gid = _resolve_stem(self.game)
            cur = sv._grid_from_state(self.game.current_state)
            raw = bool(payload.get("board_changed"))
            fixed = not boards_equal(prev, cur, gid)
            (_stats["masked_games"] if hud_mask(gid, np.asarray(prev).shape) is not None
             else _stats["unmasked_games"]).add(_stem(gid))
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
    d["masked_games"] = sorted(d["masked_games"])
    d["unmasked_games"] = sorted(d["unmasked_games"])
    if d["calls"]:
        d["corrected_frac"] = round(d["corrected"] / d["calls"], 4)
        d["raw_changed_frac"] = round(d["raw_changed"] / d["calls"], 4)
        d["masked_changed_frac"] = round(d["masked_changed"] / d["calls"], 4)
    return d


def assert_fired(min_corrections: int = 1) -> None:
    """Fail loudly if the arm installed but never actually changed anything.

    Checking that the monkeypatch installed proves nothing -- this campaign has already
    shipped an arm whose toggle was read by no code, and this arm's own stem-resolution
    bug made it a no-op while every install check passed. The only honest evidence is a
    non-zero correction count.
    """
    st = stats()
    if st.get("unresolved", 0):
        raise RuntimeError(
            f"[boardfix] failed to resolve the official stem for {st['unresolved']} clone "
            "games — the HUD table cannot be applied and the arm would silently be "
            "baseline. Refusing to proceed.")
    if st.get("calls", 0) == 0:
        raise RuntimeError("[boardfix] patch never ran — no actions were executed through it")
    if st.get("corrected", 0) < min_corrections:
        raise RuntimeError(
            f"[boardfix] patch ran {st['calls']} times and corrected NOTHING ({st}). "
            "The arm is behaviourally identical to baseline — refusing to report an A/A "
            "run as an A/B."
        )


def apply_all() -> dict:
    return {"board_changed_hud_masked": patch_board_changed()}
