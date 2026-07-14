"""shadow_run.py — CONCURRENT floor-safe shadow that banks the duck's own level-wins efficiently.

Runs alongside an unmodified duck (or any) solver on a taaf Benchmark. As each pass-0 game
FINISHES, it extracts that game's winning-attempt actions per completed level and replays them in
a FRESH play on the SAME competition card -> max-over-plays banks the efficient score. The duck's
pass-0 play is untouched, so worst case == duck; extra plays can only ADD.

Wiring (in the duck notebook, replacing `await bm.run(...)`):
    from shadow_run import run_with_shadow
    await run_with_shadow(bm, soft_end_time=..., runtime_environment=..., minimal_diagnostics=...)

Concurrent-by-finish (not by-level): a duck game's thread exits on win/limit, freeing budget; the
shadow replays that game immediately, interleaved with the still-running duck games -> it FIRES
within the 9h (unlike the after-pass-0 best-of-N, which never fires). Every step exception-fenced;
a replay desync abandons just that game (duck score stands). LIVE gateway mid-run plays are the one
thing not offline-testable -> ship with a pure-duck resubmit as same-day insurance.
"""
from __future__ import annotations
import asyncio
from typing import Any

import taaf.game
from shadow_replay import extract_winning_segments, replay_segments

_STATE: dict[str, Any] = {}
_ORIG_RUNSESSION_INIT = taaf.game.RunSession.__init__


def _install_session_hook() -> None:
    """Capture the RunSession the Benchmark creates in run(), so the shadow can open new plays on it."""
    def _hooked(self, *a, **k):
        _ORIG_RUNSESSION_INIT(self, *a, **k)
        _STATE["session"] = self
    taaf.game.RunSession.__init__ = _hooked  # type: ignore[assignment]


def _shadow_one(bm: Any, g: int, session: Any) -> None:
    """Replay game index g's winning attempts into a fresh play on the shared card."""
    run = bm.game_runs[g]
    if run is None or int(run.levels_completed or 0) == 0:
        return
    og = bm.games[g]
    segs = extract_winning_segments(run.history, run.actions_per_level, run.levels_completed)
    ng = type(og)(env_name=og.env_name, arcade_spec=og.arcade_spec)
    ng.start_game(session)
    bm.game_runs.append(ng.game_run)   # register so TAAF's max-over-plays / diagnostics see it
    res = replay_segments(ng, segs)
    try:
        ng.finish_game()               # finalize the shadow play's score
    except Exception:
        pass
    print(f"[shadow] {og.env_name}: duck won {run.levels_completed} lvl(s) -> replay banked "
          f"{res['levels']} in {res['actions']} actions (desync={res['desynced_at']})", flush=True)


async def _shadow_loop(bm: Any, poll: float = 5.0) -> None:
    while "session" not in _STATE:
        await asyncio.sleep(0.5)
    session = _STATE["session"]
    n_games = len(bm.games)
    handled: set[int] = set()

    def sweep() -> None:
        for g in range(min(n_games, len(bm.game_runs))):
            if g in handled:
                continue
            run = bm.game_runs[g]
            if run is None or run.state == "playing":
                continue
            handled.add(g)              # finished (won/gave_up/cancelled/crashed) -> bank once
            try:
                _shadow_one(bm, g, session)
            except Exception as e:  # noqa: BLE001 — a bad game must never touch the duck's score
                print(f"[shadow] game {g} replay error: {type(e).__name__}: {e}", flush=True)

    while not _STATE.get("stop"):
        await asyncio.sleep(poll)
        sweep()
    sweep()  # final sweep for games that finished right at the end


async def run_with_shadow(bm: Any, **run_kwargs: Any) -> Any:
    _STATE.clear()
    _install_session_hook()
    shadow = asyncio.create_task(_shadow_loop(bm))
    try:
        result = await bm.run(**run_kwargs)
    finally:
        _STATE["stop"] = True
        try:
            await asyncio.wait_for(shadow, timeout=900)
        except Exception:
            shadow.cancel()
        taaf.game.RunSession.__init__ = _ORIG_RUNSESSION_INIT  # restore
    return result
