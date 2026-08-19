#!/usr/bin/env python3
"""End-to-end v5 validation: grind-to-win-and-bank on tu93 through the
competition-parity server, scored by the OFFICIAL aggregate scorer.

Claim under test: a game the agent scores 0 on becomes a high-scoring game via
(grind full win at any cost) + (banked minimal replay on a fresh play).

Run:  .venv/bin/python submission/_explorer_floor/validate_v5_tu93.py
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ONLY_RESET_LEVELS"] = "true"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"))

import arc_agi  # noqa: E402
import arcengine  # noqa: E402
from arc_agi.scorecard import EnvironmentScorecard  # noqa: E402
from taaf.competition_arcade import CompetitionArcadeServer  # noqa: E402

from graft_explorer import FrontierGraph, VolatilityMask, compact_click_candidates  # noqa: E402

MAX_DEPTH = 30
P0_BUDGET = 60000
BUDGET = 400000


def main() -> None:
    server = CompetitionArcadeServer(game_ids=["tu93-0768757b"],
                                     environments_dir=str(REPO / "environment_files")).start()
    try:
        client = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                                arc_base_url=server.base_url, arc_api_key=server.api_key)
        card_id = client.open_scorecard()
        env = client.make("tu93-0768757b", scorecard_id=card_id)
        graph = FrontierGraph()
        mask = VolatilityMask()
        val2name = {a.value: a.name for a in arcengine.GameAction}
        executed = 0
        resp = None

        def grid_of(r):
            data = r.frame[-1]
            rows = data.tolist() if hasattr(data, "tolist") else data
            return [[int(c) for c in row] for row in rows]

        def step(plan):
            nonlocal executed, resp
            aid = arcengine.GameAction.from_name(plan[0])
            data = ({"x": int(plan[1]), "y": int(plan[2])}
                    if plan[0] == "ACTION6" and len(plan) == 3 else {})
            r = env.step(aid, data=data)
            executed += 1
            mask.update(grid_of(r))
            resp = r
            return r

        def cands(phase):
            names = [val2name[v] for v in (resp.available_actions or []) if v in val2name]
            plans = [(n,) for n in names if n not in ("RESET", "ACTION6")]
            if phase >= 1 and "ACTION6" in names:
                plans.extend(("ACTION6", x, y) for x, y in compact_click_candidates(
                    graph.masked_rows(grid_of(resp), mask.mask_cells()), limit=20))
            return plans

        def bfs_one_level(base_levels):
            def sig():
                return graph.node_key(base_levels, grid_of(resp), mask.mask_cells())

            def unlocked():
                return int(resp.levels_completed) != base_levels

            for phase in (0, 1):
                p0_exec = executed
                seen = set()
                queue = deque([[]])
                step(("RESET",))
                if unlocked():
                    return "level_unlocked", [("RESET",)]
                seen.add(sig())
                while queue:
                    if executed >= BUDGET:
                        return "budget", None
                    if phase == 0 and executed - p0_exec >= P0_BUDGET:
                        break
                    seq = queue.popleft()
                    if len(seq) >= MAX_DEPTH:
                        continue
                    step(("RESET",))
                    dead = False
                    for plan in seq:
                        step(plan)
                        if resp.state == arcengine.GameState.GAME_OVER:
                            dead = True
                            break
                        if unlocked():
                            return "level_unlocked", seq
                    if dead:
                        continue
                    for plan in cands(phase):
                        if executed >= BUDGET:
                            return "budget", None
                        step(("RESET",))
                        dead = False
                        for prev in seq:
                            step(prev)
                            if resp.state == arcengine.GameState.GAME_OVER:
                                dead = True
                                break
                        if dead:
                            continue
                        step(plan)
                        if unlocked():
                            return "level_unlocked", seq + [plan]
                        if resp.state == arcengine.GameState.GAME_OVER:
                            continue
                        k = sig()
                        if k not in seen:
                            seen.add(k)
                            queue.append(seq + [plan])
            return "frontier_exhausted", None

        t0 = time.monotonic()
        step(("RESET",))
        # warmup
        warm = [val2name[v] for v in (resp.available_actions or [])
                if v in val2name and val2name[v] not in ("RESET", "ACTION6")]
        for _ in range(6):
            for wn in warm:
                step((wn,))
                if resp.state == arcengine.GameState.GAME_OVER:
                    step(("RESET",))

        level_seqs = []
        while True:
            base = int(resp.levels_completed)
            reason, seq = bfs_one_level(base)
            if reason != "level_unlocked":
                print(f"stuck at level {base + 1}: {reason} after {executed} actions")
                break
            level_seqs.append(seq)
            print(f"level {base + 1} unlocked ({len(seq)} minimal actions, "
                  f"{executed} total spent, {time.monotonic()-t0:.0f}s)")
            if resp.state == arcengine.GameState.WIN:
                print(f"FULL WIN via grinder in {executed} actions")
                break

        assert resp.state == arcengine.GameState.WIN, "expected a full grinder win on tu93"

        # BANK: fresh play, minimal replay
        r = step(("RESET",))
        assert int(r.levels_completed) == 0 and r.state != arcengine.GameState.WIN, \
            "post-WIN RESET must open a fresh play"
        replay_total = 0
        for seq in level_seqs:
            for plan in seq:
                step(plan)
                replay_total += 1
                assert resp.state != arcengine.GameState.GAME_OVER, "replay died"
        print(f"bank replay: {replay_total} actions, final state {resp.state.name}")
        assert resp.state == arcengine.GameState.WIN

        sc = server._arcade.scorecard_manager.scorecards[card_id]
        agg = EnvironmentScorecard.from_scorecard(sc, list(server._arcade.available_environments))
        (g,) = list(agg.environments)
        per_play = [round(r_.score, 2) for r_ in g.runs]
        print(f"official per-play scores: {per_play} -> game score {g.score:.2f} (max)")
        assert g.score > 50, f"expected banked score > 50, got {g.score}"
        print(f"V5 END-TO-END PASSED: zero-game -> {g.score:.2f} points, "
              f"total wall {time.monotonic()-t0:.0f}s, {executed} engine actions")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
