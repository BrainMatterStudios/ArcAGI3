#!/usr/bin/env python3
"""Contention test: does one grind starve concurrent games on the shared
competition gateway? (The suspected mechanism behind the 1.33 live draw.)

Setup mirrors the scored geometry in miniature: the local CompetitionArcadeServer
(the SAME arc_agi REST app the Kaggle gateway runs) serves N cloned games; W
worker threads play them concurrently with a fixed think-time per action
(simulating LLM pacing); one extra thread runs a full-speed grind burst.
Measured: per-action latency of the NORMAL players with the grind OFF vs ON.

Run:  .venv/bin/python submission/_explorer_floor/test_contention.py
"""
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ONLY_RESET_LEVELS"] = "true"
sys.path.insert(0, str(REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"))

import arc_agi  # noqa: E402
import arcengine  # noqa: E402
from taaf.competition_arcade import CompetitionArcadeServer  # noqa: E402

PLAYERS = 8            # concurrent normal games (scored run: up to 28)
THINK_S = 0.5          # pacing between a normal player's actions
PHASE_S = 25           # measurement window per phase
GRIND_GAME_INDEX = 0   # the grinder's own game


def player_loop(env, latencies, stop, think_s):
    val2name = {a.value: a.name for a in arcengine.GameAction}
    env.reset()
    moves = [arcengine.GameAction.from_name(n) for n in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
    i = 0
    while not stop.is_set():
        t0 = time.monotonic()
        env.step(moves[i % len(moves)])
        latencies.append(time.monotonic() - t0)
        i += 1
        time.sleep(think_s)


def grind_loop(env, counter, stop):
    env.reset()
    moves = [arcengine.GameAction.from_name(n) for n in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
    i = 0
    while not stop.is_set():
        env.step(moves[i % len(moves)])
        if i % 40 == 0:
            env.step(arcengine.GameAction.from_name("RESET"))
        counter[0] += 1
        i += 1


def measure(with_grind: bool) -> tuple[list[float], int]:
    server = CompetitionArcadeServer(
        game_ids=["vc33-5430563c"],
        total_runs=PLAYERS + 1,
        environments_dir=str(REPO / "environment_files"),
    ).start()
    try:
        client = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                                arc_base_url=server.base_url, arc_api_key=server.api_key)
        card_id = client.open_scorecard()
        games = server.exposed_game_ids
        envs = [client.make(g, scorecard_id=card_id) for g in games]
        stop = threading.Event()
        latencies: list[float] = []
        grind_count = [0]
        threads = [threading.Thread(target=player_loop,
                                    args=(envs[i + 1], latencies, stop, THINK_S), daemon=True)
                   for i in range(PLAYERS)]
        if with_grind:
            threads.append(threading.Thread(target=grind_loop,
                                            args=(envs[GRIND_GAME_INDEX], grind_count, stop),
                                            daemon=True))
        for t in threads:
            t.start()
        time.sleep(PHASE_S)
        stop.set()
        for t in threads:
            t.join(timeout=5)
        return latencies, grind_count[0]
    finally:
        server.stop()


def main() -> None:
    base_lat, _ = measure(with_grind=False)
    grind_lat, grind_actions = measure(with_grind=True)

    def stats(xs):
        xs = sorted(xs)
        return (statistics.mean(xs), statistics.median(xs),
                xs[int(0.95 * len(xs)) - 1] if xs else 0.0)

    bm, bmed, bp95 = stats(base_lat)
    gm, gmed, gp95 = stats(grind_lat)
    print(f"baseline  (no grind): n={len(base_lat)} mean={bm*1000:.1f}ms "
          f"median={bmed*1000:.1f}ms p95={bp95*1000:.1f}ms")
    print(f"with grind ({grind_actions} grind actions in {PHASE_S}s = "
          f"{grind_actions/PHASE_S:.0f}/s): n={len(grind_lat)} mean={gm*1000:.1f}ms "
          f"median={gmed*1000:.1f}ms p95={gp95*1000:.1f}ms")
    slow = gm / bm if bm else float("inf")
    print(f"VERDICT: normal players' mean action latency {slow:.1f}x under one "
          f"full-speed grind ({'CONTENTION REPRODUCED' if slow > 2 else 'no material contention'})")


if __name__ == "__main__":
    main()
