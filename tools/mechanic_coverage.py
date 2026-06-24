"""Phase 1 — Mechanic Coverage Benchmark (winning/mechanic-model-search).

Before building new solvers, MEASURE whether mechanics are reusable: for each game, run a fixed
probe budget, induce compact transition primitives, and score how well they PREDICT held-out
transitions and enable planning. The decisive kill gate (mission Phase 1):

    If fewer than ~20-25% of games get a useful transition model, STOP. Ship transfer-dense.

The key metric is HELD-OUT transition accuracy (train primitives on 70% of the probe's transitions,
test prediction on the last 30%) — a real generalization signal, not memorized counts. Reuses the
existing induction (DiscoveryExplorer fits movement + paint + collect + attr-cycle online); the
benchmark only instruments and scores it. Source/live env used as a black box.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python tools/mechanic_coverage.py [probe] [games...]
Outputs: coverage.jsonl + mechanic_coverage_report.md
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv; load_dotenv()

from arcagi3 import perception as P
from arcagi3.discovery_explorer import DiscoveryExplorer
from arcengine import GameAction, GameState

logging.basicConfig(level=logging.ERROR)
ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]
DEFAULT_GAMES = ["collect"] + HOLDOUT          # collect = known-positive control
USEFUL_PRED_ACC = 0.70                          # a "useful transition model" threshold


def make_game(game: str):
    from arc_agi import Arcade, OperationMode
    if game == "collect":
        c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="src/arcagi3/games",
                   logger=logging.getLogger("mc"))
        return c.make(game_id="collect", scorecard_id="sc-collect")
    c = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("mc"))
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    return c.make(game_id=gid, scorecard_id=c.open_scorecard(tags=["coverage"]))


def n_objects(grid, bg):
    """Cheap connected-component count of non-bg cells (4-conn)."""
    seen = np.zeros_like(grid, dtype=bool)
    H, W = grid.shape
    n = 0
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or seen[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == grid[r, c]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


def centroid(grid, colors):
    mask = np.isin(grid, list(colors))
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (float(ys.mean()), float(xs.mean()))


def push_evidence(prev, grid, agent_colors, bg):
    """Cheap detector: a non-agent, non-bg connected blob that rigidly translates by a small vector
    (same multiset of cells, shifted). Signals a PUSH/PULL mechanic our primitives don't model yet."""
    diff = (prev != grid)
    if not diff.any():
        return False
    # candidate moved color = a color whose cell-count is preserved but positions shifted
    for col in set(np.unique(prev)).intersection(np.unique(grid)) - set(agent_colors) - {bg}:
        a = np.argwhere(prev == col); b = np.argwhere(grid == col)
        if len(a) == len(b) and len(a) >= 3:
            shift = b.min(0) - a.min(0)
            if 0 < abs(shift).sum() <= 3 and np.array_equal(np.sort(a - a.min(0), 0), np.sort(b - b.min(0), 0)):
                return True
    return False


def world_deltas(prev, grid, agent_colors):
    """Non-agent cell changes as (from_color, to_color)."""
    out = []
    diff = np.argwhere(prev != grid)
    for r, c in diff:
        f, t = int(prev[r, c]), int(grid[r, c])
        if f in agent_colors or t in agent_colors:
            continue
        out.append((f, t))
    return out


def run_game(game: str, probe: int) -> dict:
    eng = DiscoveryExplorer(seed=0, planner_backend="painted_set")
    eng.reset_all()
    try:
        env = make_game(game)
        obs = env.reset()
    except Exception as e:  # noqa: BLE001
        return {"game": game, "error": str(e)[:80]}
    level = int(obs.levels_completed or 0)
    transitions = []          # (prev_grid, action_id, grid)
    prev_grid = prev_aid = None
    plan_ever = False
    levelups = 0
    push_seen = False
    n = 0
    while n < probe:
        if obs is None or obs.state == GameState.WIN or np.asarray(obs.frame).size == 0:
            break
        grid = P.to_grid(obs.frame)
        if prev_grid is not None and prev_aid is not None:
            transitions.append((prev_grid, prev_aid, grid))
        token = eng.decide(grid=grid, gstate_terminal=(obs.state == GameState.GAME_OVER),
                           gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                           levels=level, available=list(obs.available_actions or []))
        plan_ever = plan_ever or bool(getattr(eng, "_plan", None))
        prev_grid = grid
        prev_aid = token[1] if token[0] == "S" else None   # movement metrics: directional only
        try:
            if token[0] == "reset":
                obs = env.reset()
            elif token[0] == "S":
                obs = env.step(GameAction.from_id(token[1])); n += 1
            else:
                obs = env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]}); n += 1
        except Exception:
            break
        if obs is None:
            break
        nl = int(obs.levels_completed or 0)
        if nl > level:
            levelups += 1; level = nl

    # --- induce on 70%, test held-out 30% (real generalization signal) ---
    agent_colors = set(getattr(eng, "_agent_colors", set()))
    if getattr(eng, "_agent_color", None) is not None:
        agent_colors.add(int(eng._agent_color))
    bg = getattr(eng, "_bg", 0)
    bg = 0 if bg is None else int(bg)
    agent_found = bool(agent_colors)

    split = int(len(transitions) * 0.7)
    train, test = transitions[:split], transitions[split:]

    move_train = defaultdict(list); recolor_train = Counter(); collect_train = set()
    for prev, aid, grid in train:
        if aid is not None and agent_found:
            cp, cg = centroid(prev, agent_colors), centroid(grid, agent_colors)
            if cp and cg:
                move_train[aid].append((round(cg[0] - cp[0]), round(cg[1] - cp[1])))
        for f, t in world_deltas(prev, grid, agent_colors):
            (collect_train.add(f) if t == bg else recolor_train.update([(f, t)]))
    deltas = {a: Counter(v).most_common(1)[0][0] for a, v in move_train.items() if v}
    recolors = {ft for ft, c in recolor_train.items() if c >= 1}

    move_hit = move_tot = world_hit = world_tot = 0
    for prev, aid, grid in test:
        if aid is not None and aid in deltas and agent_found:
            cp, cg = centroid(prev, agent_colors), centroid(grid, agent_colors)
            if cp and cg:
                actual = (round(cg[0] - cp[0]), round(cg[1] - cp[1]))
                if actual == (0, 0):
                    continue   # agent didn't move (blocked/no-op) — not a movement-model test
                move_tot += 1
                if actual == deltas[aid]:
                    move_hit += 1
        for f, t in world_deltas(prev, grid, agent_colors):
            world_tot += 1
            if (t == bg and f in collect_train) or ((f, t) in recolors):
                world_hit += 1
        if not push_seen and push_evidence(prev, grid, agent_colors, bg):
            push_seen = True

    n_move = move_tot or 1; n_world = world_tot or 0
    move_acc = move_hit / n_move if move_tot else 0.0
    world_acc = world_hit / n_world if world_tot else (1.0 if move_tot else 0.0)
    # combined held-out transition accuracy (weighted by event counts)
    denom = move_tot + world_tot
    pred_acc = ((move_hit + world_hit) / denom) if denom else 0.0

    n_world_obs = len(getattr(eng, "_world_obs", []))
    n_rules = len(deltas) + len(recolors) + len(collect_train)
    objs = n_objects(transitions[-1][2], bg) if transitions else 0

    prim = {
        "move": round(move_acc, 2),
        "paint_on_move": round(min(1.0, len(recolors) / max(1, world_tot)) if world_tot else 0.0, 2),
        "collect_on_contact": round(len(collect_train) / max(1, len(recolors) + len(collect_train)), 2)
                              if (recolors or collect_train) else 0.0,
        "push": 1.0 if push_seen else 0.0,   # detected EVIDENCE; primitive not yet implemented
        "toggle": round(len(getattr(eng, "_cycles", [])) / 4.0, 2) if getattr(eng, "_cycles", None) else 0.0,
    }
    reward_explained = levelups > 0 and plan_ever
    # mission scoring formula
    agent_action_id = 0.5 * agent_found + 0.5 * (len(deltas) > 0)
    compression = max(0.0, 1.0 - n_rules / 10.0)
    coverage = (0.2 * agent_action_id + 0.3 * pred_acc + 0.2 * compression
                + 0.2 * (1.0 if plan_ever else 0.0) + 0.1 * (1.0 if reward_explained else 0.0))
    useful = (pred_acc >= USEFUL_PRED_ACC) and agent_found

    return {
        "game": game, "agent_found": agent_found, "agent_colors": sorted(agent_colors),
        "actions_verified": sorted(deltas), "movement_model": {str(k): v for k, v in deltas.items()},
        "objects_found": objs, "world_deltas": n_world_obs, "transitions": len(transitions),
        "primitive_fits": prim, "next_state_accuracy": round(pred_acc, 3),
        "move_acc": round(move_acc, 3), "world_acc": round(world_acc, 3),
        "reward_event_explained": reward_explained, "plan_found": plan_ever,
        "levels_cleared": level, "n_rules": n_rules, "mechanic_coverage": round(coverage, 3),
        "useful": useful,
    }


def main():
    probe = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    games = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_GAMES
    print(f"=== Mechanic Coverage Benchmark — probe={probe}/game, {len(games)} games ===\n", flush=True)
    rows = []
    for g in games:
        print(f"[{g}] probing...", flush=True)
        r = run_game(g, probe)
        rows.append(r)
        if "error" in r:
            print(f"   ERROR {r['error']}", flush=True)
        else:
            print(f"   agent={r['agent_found']} acts={r['actions_verified']} pred_acc={r['next_state_accuracy']} "
                  f"plan={r['plan_found']} cleared={r['levels_cleared']} cov={r['mechanic_coverage']} "
                  f"useful={r['useful']}", flush=True)

    (ROOT / "coverage.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = [r for r in rows if "error" not in r]
    n_useful = sum(r["useful"] for r in ok)
    frac = n_useful / len(ok) if ok else 0.0

    lines = ["# Mechanic Coverage Report", "",
             f"Probe budget: {probe} actions/game · {len(ok)} games · useful threshold: held-out pred_acc ≥ {USEFUL_PRED_ACC}", "",
             "| game | agent | acts | pred_acc | move | world | best primitives | plan? | cleared | cov | useful |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        best = ",".join(k for k, v in sorted(r["primitive_fits"].items(), key=lambda x: -x[1]) if v >= 0.5) or "—"
        lines.append(f"| {r['game']} | {'Y' if r['agent_found'] else 'n'} | {r['actions_verified']} | "
                     f"{r['next_state_accuracy']} | {r['move_acc']} | {r['world_acc']} | {best} | "
                     f"{'Y' if r['plan_found'] else 'n'} | {r['levels_cleared']} | {r['mechanic_coverage']} | "
                     f"{'**YES**' if r['useful'] else 'no'} |")
    gate_pass = frac >= 0.20
    lines += ["",
              f"**Useful transition models: {n_useful}/{len(ok)} = {frac:.0%}**",
              f"**KILL GATE (≥20-25%): {'PASS — proceed to Phase 2 (primitive library + model beam)' if gate_pass else 'FAIL — stop; ship transfer-dense'}**"]
    (ROOT / "mechanic_coverage_report.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[-3:]), flush=True)
    print(f"\nwrote coverage.jsonl + mechanic_coverage_report.md", flush=True)


if __name__ == "__main__":
    main()
