"""Does level-1 knowledge reduce the cost of clearing later levels?

THE DESIGN. Two arms play the same game with the same seed and the same explorer.
They are bit-identical through level 1, because knowledge evolves identically there.
They diverge at exactly one point -- the level boundary:

    COLD  wipes Knowledge and recomputes the HUD mask, as both published
          implementations do (they discard the graph, effect table AND mask).
    WARM  carries Knowledge across the boundary.

So the measured difference on a(2), a(3), ... is attributable to the carry and to
nothing else. a(l) is per-level and independent (measured 2026-08-01), so level-1
cost is identical in both arms and cancels out.

FAIRNESS NOTE. COLD is allowed to recompute the HUD mask at each new level, at its
true cost in actions. Denying it that would handicap the control and manufacture the
result we are testing for.

REPORTED. Both the relative reduction (does transfer happen at all?) and the absolute
a(l) against the human baseline h(l) (is it economically viable?). Those are different
questions: a 10x reduction that still lands at 50x baseline scores ~0 either way.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
logging.disable(logging.CRITICAL)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from explorer import GraphExplorer, Knowledge, frame_np, learn_hud_mask  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

PER_LEVEL_CAP = int(os.environ.get("PER_LEVEL_CAP", 4000))
TOTAL_CAP = int(os.environ.get("TOTAL_CAP", 20000))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
GAMES = [g for g in os.environ.get("GAMES", "").split(",") if g]
OUT = Path(__file__).parent / os.environ.get("OUT", "transfer_results.json")


def baselines(env_meta_dir: Path) -> list[int]:
    for meta in env_meta_dir.rglob("metadata.json"):
        try:
            d = json.loads(meta.read_text())
            b = d.get("baseline_actions")
            if b:
                return list(b)
        except Exception:
            pass
    return []


def official_score(level_actions: dict[int, int], cleared: set[int], n_levels: int,
                   base: list[int]) -> float:
    """min(weighted mean of level scores, completed-share cap); weight = level index."""
    if n_levels <= 0:
        return 0.0
    total_w = sum(range(1, n_levels + 1))
    num = 0.0
    done_w = 0
    for lv in range(1, n_levels + 1):
        w = lv
        if lv in cleared:
            done_w += w
            h = base[lv - 1] if lv - 1 < len(base) else None
            a = level_actions.get(lv, 0)
            s = min(115.0, 100.0 * (h / a) ** 2) if (h and a) else 0.0
            num += w * s
    return min(num / total_w, 100.0 * done_w / total_w)


def play(game_id: str, arcade: Arcade, seed: int, carry: bool) -> dict:
    env = arcade.make(game_id=game_id, scorecard_id=f"tx-{'w' if carry else 'c'}-{seed}-{game_id[:8]}")
    obs = env.reset()
    n_levels = int(obs.win_levels or 0)

    def step_simple(aid):
        return env.step(GameAction.from_id(aid))

    knowledge = Knowledge()
    ex = GraphExplorer(knowledge, seed=seed)

    actions = 0
    level = 1
    level_actions: dict[int, int] = {}
    cleared: set[int] = set()
    spent_this_level = 0

    mask, obs, used = learn_hud_mask(step_simple, obs)
    knowledge.hud_mask = mask
    actions += used
    spent_this_level += used

    prev_completed = int(obs.levels_completed)
    stop = None

    while actions < TOTAL_CAP:
        if obs.state == GameState.WIN:
            stop = "win"
            break
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            actions += 1
            spent_this_level += 1
            ex.reset_level_state()
            continue
        if spent_this_level >= PER_LEVEL_CAP:
            stop = "level_cap"
            break

        grid = frame_np(obs)
        nid = ex.node_id(grid)
        if ex.level_first_hash is None:
            ex.level_first_hash = nid

        key, mode = ex.choose(nid, grid, obs.available_actions or [])
        if key is None:
            stop = "exhausted"
            break

        if key[0] == "S":
            obs = env.step(GameAction.from_id(key[1]))
        else:
            _, _colour, y, x = key
            obs = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        actions += 1
        spent_this_level += 1

        after = frame_np(obs)
        new_nid = ex.node_id(after)
        changed = new_nid != nid
        ex.record(nid, key, new_nid, changed, after)
        ex.learn(key, changed)

        completed = int(obs.levels_completed)
        if completed > prev_completed:
            level_actions[level] = spent_this_level
            cleared.add(level)
            prev_completed = completed
            level += 1
            spent_this_level = 0
            ex.reset_level_state()
            if not carry:
                # What both published implementations do: discard everything.
                ex.k = Knowledge()
                m, obs, used = learn_hud_mask(step_simple, obs)
                ex.k.hud_mask = m
                actions += used
                spent_this_level += used
    else:
        stop = "total_cap"

    if level not in level_actions and spent_this_level:
        level_actions[level] = spent_this_level   # unfinished level, for the record

    return {
        "levels_cleared": sorted(cleared),
        "level_actions": level_actions,
        "total_actions": actions,
        "n_levels": n_levels,
        "stop": stop or "total_cap",
        "knowledge": ex.k.summary(),
    }


def main() -> None:
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
    if GAMES:
        envs = [e for e in envs if e.game_id.split("-")[0][:4] in GAMES]

    root = Path("environment_files")
    results = []
    t0 = time.time()
    print(f"per_level_cap={PER_LEVEL_CAP} total_cap={TOTAL_CAP} seeds={SEEDS} games={len(envs)}", flush=True)
    print(f"{'game':6} {'seed':>4} {'coldLv':>7} {'warmLv':>7} {'cold_a2':>8} {'warm_a2':>8} {'coldScore':>10} {'warmScore':>10}", flush=True)

    for e in envs:
        stem = e.game_id.split("-")[0][:4]
        base = baselines(root / stem)
        for seed in SEEDS:
            cold = play(e.game_id, arcade, seed, carry=False)
            warm = play(e.game_id, arcade, seed, carry=True)
            cs = official_score(cold["level_actions"], set(cold["levels_cleared"]), cold["n_levels"], base)
            ws = official_score(warm["level_actions"], set(warm["levels_cleared"]), warm["n_levels"], base)
            rec = {"game": stem, "seed": seed, "baselines": base,
                   "cold": cold, "warm": warm, "cold_score": cs, "warm_score": ws}
            results.append(rec)
            print(f"{stem:6} {seed:>4} {len(cold['levels_cleared']):>7} {len(warm['levels_cleared']):>7} "
                  f"{str(cold['level_actions'].get(2,'-')):>8} {str(warm['level_actions'].get(2,'-')):>8} "
                  f"{cs:>10.2f} {ws:>10.2f}", flush=True)
            json.dump(results, open(OUT, "w"), indent=1)

    n = len(results)
    if n:
        cl = sum(len(r["cold"]["levels_cleared"]) for r in results)
        wl = sum(len(r["warm"]["levels_cleared"]) for r in results)
        cs = sum(r["cold_score"] for r in results) / n
        ws = sum(r["warm_score"] for r in results) / n
        wins = sum(1 for r in results if r["warm_score"] > r["cold_score"] + 1e-9)
        loss = sum(1 for r in results if r["cold_score"] > r["warm_score"] + 1e-9)
        print(f"\nruns {n}   levels cold {cl} / warm {wl}")
        print(f"mean official score  cold {cs:.3f}   warm {ws:.3f}")
        print(f"per-run: warm better {wins}, cold better {loss}, tied {n - wins - loss}")
    print(f"elapsed {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
