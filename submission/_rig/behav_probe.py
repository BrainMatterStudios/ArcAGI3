"""Per-action behavioural instrumentation. Observes only; changes nothing.

WHY THIS EXISTS. Scoring a rig run is low-power: 28 games, near-binary outcomes, and
17 of 28 flip between identical repeats (measured). A 3-repeat score A/B sits at the
noise floor. But the *mechanism* an arm claims to affect is a PER-ACTION quantity with
~1,600 samples per run — roughly 50x the statistical power on the same GPU minutes.

So we measure whether the agent's behaviour changed in the intended direction, and
leave score to the one instrument that actually measures the hidden set: a submission.

WHAT IT MEASURES, with the human benchmarks it is judged against:
  dead_reissue     re-issuing a (board, action) pair already observed to do nothing,
                   state-conditioned. Humans 3.4%; our agent 27.1% unconditioned.
                   This is the quantity the board_changed fix should move.
  noop_raw         action left the frame byte-identical (what the harness believes)
  noop_masked      action left the BOARD identical, HUD excluded (the truth)
  immediate_repeat same action twice in a row. Humans 68% overall, 52% keys, 10.6% clicks.
  distinct_states  exploration efficiency: unique masked boards per action

IT MUST BE IDENTICAL IN BOTH ARMS. It carries its own HUD table and stem resolution
rather than importing the arm under test, so the baseline arm is instrumented exactly
like the treatment arm and the instrument cannot become the difference between them.

It wraps OUTSIDE any arm patch (applied last), so it observes the final executed
action either way. It never mutates the payload.
"""
from __future__ import annotations

import re

import numpy as np

HUD_REGIONS: dict[str, tuple] = {
    "cd82": ("row", 63), "dc22": ("row", 63), "ka59": ("row", 63), "re86": ("row", 63),
    "s5i5": ("row", 63), "tu93": ("row", 63), "wa30": ("row", 63), "bp35": ("row", 63),
    "tr87": ("row", 63),
    "cn04": ("row", 0), "vc33": ("row", 0), "sp80": ("row", 0), "lf52": ("row", 0),
    "sk48": ("row", 53), "sb26": ("row", 53),
    "r11l": ("col", 0), "lp85": ("col", 0),
    "ls20": ("rows", (61, 62)),
}
_CLONE = re.compile(r"^k\d{3}$")
_mask_cache: dict[tuple, np.ndarray | None] = {}
_stem_cache: dict[int, str] = {}

# per game stem -> counters
_G: dict[str, dict] = {}
_unresolved = {"n": 0}


def _stem(g: str) -> str:
    return str(g).split("-")[0][:4].lower()


def _resolve(game) -> str:
    key = id(game)
    hit = _stem_cache.get(key)
    if hit is not None:
        return hit
    cand = str(getattr(game, "env_name", "") or getattr(game, "game_id", "") or "")
    if _CLONE.match(cand):
        got = ""
        for path in ("env", "_env"):
            env = getattr(game, path, None)
            info = getattr(env, "environment_info", None) if env is not None else None
            t = str(getattr(info, "title", "") or "")
            if t:
                got = t
                break
        if got:
            cand = got
        else:
            _unresolved["n"] += 1
    out = _stem(cand)
    _stem_cache[key] = out
    return out


def _mask(stem: str, shape):
    key = (stem, tuple(shape))
    if key in _mask_cache:
        return _mask_cache[key]
    spec = HUD_REGIONS.get(stem)
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


def _key(grid, stem) -> bytes:
    a = np.asarray(grid)
    m = _mask(stem, a.shape)
    if m is not None and m.any():
        a = np.where(m, 0, a)
    return np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _slot(stem: str) -> dict:
    return _G.setdefault(stem, {
        "actions": 0, "noop_raw": 0, "noop_masked": 0,
        "dead_reissue": 0, "reissue_opportunities": 0,
        "immediate_repeat": 0, "key_actions": 0, "key_repeat": 0,
        "click_actions": 0, "click_repeat": 0,
        "_seen": {}, "_states": set(), "_last": None,
    })


def install() -> bool:
    from inference.framework import solver as sv

    original = sv._HarnessGameSession._execute_action
    if getattr(original, "_behav", False):
        return True

    def _execute_action(self, action, **kwargs):
        stem = _resolve(self.game)
        prev = sv._grid_from_state(self.game.current_state)
        payload = original(self, action, **kwargs)
        try:
            s = _slot(stem)
            cur = sv._grid_from_state(self.game.current_state)
            pa, ca = np.asarray(prev), np.asarray(cur)
            pk, ck = _key(pa, stem), _key(ca, stem)

            name = getattr(getattr(action, "id", None), "name", "?")
            data = dict(getattr(action, "data", {}) or {})
            akey = (name, data.get("x"), data.get("y")) if name == "ACTION6" else (name,)

            s["actions"] += 1
            s["noop_raw"] += int(np.array_equal(pa, ca))
            changed = pk != ck
            s["noop_masked"] += int(not changed)
            s["_states"].add(ck)

            if name == "ACTION6":
                s["click_actions"] += 1
                s["click_repeat"] += int(s["_last"] == akey)
            else:
                s["key_actions"] += 1
                s["key_repeat"] += int(s["_last"] == akey)
            s["immediate_repeat"] += int(s["_last"] == akey)
            s["_last"] = akey

            # State-conditioned dead-action memory: have we been at THIS board and
            # taken THIS action before, and did it do nothing?
            memo = s["_seen"]
            mk = (pk, akey)
            if mk in memo:
                s["reissue_opportunities"] += 1
                if memo[mk] is False:
                    s["dead_reissue"] += 1
            memo[mk] = changed
        except Exception:
            pass          # observation must never affect the run
        return payload

    _execute_action._behav = True
    sv._HarnessGameSession._execute_action = _execute_action
    return True


def report() -> dict:
    per = {}
    tot = {k: 0 for k in ("actions", "noop_raw", "noop_masked", "dead_reissue",
                          "reissue_opportunities", "immediate_repeat",
                          "key_actions", "key_repeat", "click_actions", "click_repeat")}
    states = 0
    for stem, s in _G.items():
        n = max(s["actions"], 1)
        per[stem] = {
            "actions": s["actions"],
            "noop_raw": round(s["noop_raw"] / n, 4),
            "noop_masked": round(s["noop_masked"] / n, 4),
            "dead_reissue_of_actions": round(s["dead_reissue"] / n, 4),
            "dead_reissue_of_opportunities": round(
                s["dead_reissue"] / max(s["reissue_opportunities"], 1), 4),
            "immediate_repeat": round(s["immediate_repeat"] / n, 4),
            "distinct_states_per_action": round(len(s["_states"]) / n, 4),
        }
        for k in tot:
            tot[k] += s[k]
        states += len(s["_states"])
    n = max(tot["actions"], 1)
    return {
        "unresolved_games": _unresolved["n"],
        "n_games": len(_G),
        "corpus": {
            "actions": tot["actions"],
            "noop_raw": round(tot["noop_raw"] / n, 4),
            "noop_masked": round(tot["noop_masked"] / n, 4),
            # The headline. Humans 3.4%; our agent 27.1% unconditioned.
            "dead_reissue_of_actions": round(tot["dead_reissue"] / n, 4),
            "dead_reissue_of_opportunities": round(
                tot["dead_reissue"] / max(tot["reissue_opportunities"], 1), 4),
            "immediate_repeat": round(tot["immediate_repeat"] / n, 4),
            "key_repeat": round(tot["key_repeat"] / max(tot["key_actions"], 1), 4),
            "click_repeat": round(tot["click_repeat"] / max(tot["click_actions"], 1), 4),
            "distinct_states_per_action": round(states / n, 4),
        },
        "per_game": per,
    }


def assert_observed(min_actions: int = 50) -> None:
    r = report()
    if r["unresolved_games"]:
        raise RuntimeError(f"[behav] {r['unresolved_games']} games had unresolvable stems — "
                           "HUD masking in the probe is unreliable, refusing to report")
    if r["corpus"]["actions"] < min_actions:
        raise RuntimeError(f"[behav] only {r['corpus']['actions']} actions observed — "
                           "not enough to compare arms")
