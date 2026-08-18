"""Explorer-floor fixture derivation (2026-08-18).

Re-derives the July mechanical wins for the six explorer-winnable games against the
LOCAL offline engine and persists them as replayable fixtures. Pure CPU, frame-only:
BFS over the action space via reset+replay with HUD-masked frame dedup (the dc22
technique generalized — scripts/research_2026_07_01/general_search.py), then
delete-and-replay minimization (minimize_solution.py), then double fresh-env
verification with per-step frame equality.

Usage: ONLY_RESET_LEVELS=true PYTHONPATH=src .venv/bin/python \
           submission/_explorer_floor/solve_floor.py <stem> [levels]
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque

import numpy as np

logging.disable(logging.CRITICAL)

ROOT = "/Users/ahmed/Documents/ArcAGI3"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts/research_2026_07_01"))
sys.path.insert(0, os.path.join(ROOT, "scratchpad/ideas"))

from arc_agi import Arcade, OperationMode              # noqa: E402
from arcengine import GameAction, GameState            # noqa: E402
from arcagi3 import perception as P                    # noqa: E402
from click_affordance_probe import salient_targets     # noqa: E402
from hud_mask import frame_key                         # noqa: E402

GAMES = {
    "dc22": "dc22-fdcac232",
    "ka59": "ka59-38d34dbb",
    "m0r0": "m0r0-492f87ba",
    "sk48": "sk48-d8078629",
    "wa30": "wa30-ee6fef47",
    "tu93": "tu93-0768757b",
}

# Per-game action-space config, from the July solvers + mechanics compendium.
#   simple: GameAction ids tried as-is (intersected with available_actions)
#   clicks: None | "panel" (dc22-style, right of play area) | "salient"
#   fallback_clicks: added only if the click-free graph exhausts without a win
CONFIG = {
    "dc22": dict(simple=[1, 2, 3, 4], clicks="panel", fallback_clicks=None),
    # ka59: blocks must be click-SELECTED at their CURRENT position, then arrows
    # push/launch them (ht_ka59_solve.py) — targets recomputed per node.
    "ka59": dict(simple=[1, 2, 3, 4], clicks="dynamic", fallback_clicks=None),
    "m0r0": dict(simple=[1, 2, 3, 4, 5], clicks="salient", fallback_clicks="dynamic"),
    "sk48": dict(simple=[1, 2, 3, 4, 5, 7], clicks=None, fallback_clicks="salient"),
    "wa30": dict(simple=[1, 2, 3, 4, 5], clicks=None, fallback_clicks="salient"),
    "tu93": dict(simple=[1, 2, 3, 4], clicks=None, fallback_clicks=None),
}

FIXDIR = os.path.join(ROOT, "submission/_explorer_floor/fixtures")


def settled(obs):
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


def lv(obs):
    return int(obs.levels_completed or 0)


def make_env(stem):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(ROOT, "environment_files"))
    return client.make(GAMES[stem])


def panel_targets(grid):
    """dc22-style: compact colored regions right of the play area (dc22_solve.py)."""
    bg = P.detect_background(grid)
    out = []
    for o in P.connected_components(grid, background=bg):
        if o.color == bg or o.size < 6:
            continue
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        if 0 <= cy < 64 and 0 <= cx < 64 and grid[cy, cx] == o.color and cx >= 30:
            out.append((cx, cy))
    return sorted(set(out))


def dynamic_targets(grid, max_n=12):
    """Per-node click targets: centroid of every compact non-bg component, INCLUDING
    multi-color units whose centroid pixel differs from the body color (ka59's c14
    units carry a 1px selection marker at center, which salient_targets excludes)."""
    bg = P.detect_background(grid)
    comps = [o for o in P.connected_components(grid, background=bg)
             if o.color != bg and 2 <= o.size <= 80]
    comps.sort(key=lambda o: o.size)
    out = []
    for o in comps:
        r0, c0, r1, c1 = o.bbox
        if r0 >= 60 or (r1 - r0) > 20 or (c1 - c0) > 20:   # HUD / huge regions
            continue
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        cy, cx = min(max(cy, r0), r1), min(max(cx, c0), c1)
        t = (cx, cy)
        if all(abs(t[0] - u[0]) + abs(t[1] - u[1]) > 2 for u in out):
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def click_mode(stem, use_fallback):
    cfg = CONFIG[stem]
    return cfg["clicks"] or (cfg["fallback_clicks"] if use_fallback else None)


def build_actions(stem, obs, use_fallback=False):
    """Action set at THIS observation. mode 'dynamic' recomputes salient click
    targets from the current frame (movable objects must be clicked where they
    currently are — ka59 block-select)."""
    cfg = CONFIG[stem]
    avail = set(obs.available_actions or [])
    acts = [("S", a) for a in cfg["simple"] if a in avail]
    mode = click_mode(stem, use_fallback)
    if mode and 6 in avail:
        grid = settled(obs)
        if mode == "panel":
            tgts = panel_targets(grid)
        elif mode == "dynamic":
            tgts = dynamic_targets(grid)
        else:  # "salient": static start-frame targets
            tgts = salient_targets(grid, max_n=12)
        acts += [("C", int(x), int(y)) for (x, y) in tgts]
    return acts


def apply_tok(env, tok):
    if tok[0] == "C":
        return env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
    return env.step(GameAction.from_id(int(tok[1])))


def bfs_level(env, stem, target, budget_s, use_fallback=False, max_depth=200):
    """BFS from env.reset() (level start under ONLY_RESET_LEVELS) to levels>=target.

    Returns (winning token seq or None, stats). On success the env has just won the
    level (advanced), so the caller must not reuse it for this level again.
    """
    obs = env.reset()
    static_actions = build_actions(stem, obs, use_fallback)
    dynamic = click_mode(stem, use_fallback) == "dynamic"
    t0 = time.time()
    seen = {frame_key(settled(obs), stem)}
    q = deque([[]])
    nodes = 0
    exhausted = False
    branching = len(static_actions)
    while q:
        if time.time() - t0 > budget_s:
            break
        seq = q.popleft()
        if len(seq) >= max_depth:
            continue
        if dynamic:
            obs = env.reset()
            for tok in seq:
                obs = apply_tok(env, tok)
            actions = build_actions(stem, obs, use_fallback)
            branching = max(branching, len(actions))
        else:
            actions = static_actions
        for a in actions:
            obs = env.reset()
            dead = False
            for tok in seq:
                obs = apply_tok(env, tok)
            obs = apply_tok(env, tok=a)
            nodes += 1
            if lv(obs) >= target:
                st = dict(nodes=nodes, states=len(seen), elapsed_s=round(time.time() - t0, 1),
                          branching=branching, fallback_clicks=use_fallback)
                return seq + [a], st
            if obs.state == GameState.GAME_OVER:
                continue
            k = frame_key(settled(obs), stem)
            if k not in seen:
                seen.add(k)
                q.append(seq + [a])
    else:
        exhausted = True
    st = dict(nodes=nodes, states=len(seen), elapsed_s=round(time.time() - t0, 1),
              branching=branching, fallback_clicks=use_fallback, exhausted=exhausted)
    return None, st


class LevelStart:
    """Fresh env positioned at the start of the level reached by `prefix`.

    Reused across failing trials (reset returns to level start); rebuilt after any
    trial that wins the level, since the env then advances irreversibly.
    """

    def __init__(self, stem, prefix, target):
        self.stem, self.prefix, self.target = stem, prefix, target
        self._rebuild()

    def _rebuild(self):
        self.env = make_env(self.stem)
        obs = self.env.reset()
        for tok in self.prefix:
            obs = apply_tok(self.env, tok)
        assert lv(obs) == self.target - 1, \
            f"prefix reaches level {lv(obs)}, expected {self.target - 1}"

    def wins(self, tokens):
        obs = self.env.reset()
        won = False
        for tok in tokens:
            obs = apply_tok(self.env, tok)
            if lv(obs) >= self.target:
                won = True
                break
            if obs.state == GameState.GAME_OVER:
                break
        if won:
            self._rebuild()
        return won


def minimize_segment(stem, prefix, segment, target):
    """delete-and-replay (minimize_solution.py): shortest winning prefix, then ddmin g=1."""
    ls = LevelStart(stem, prefix, target)
    assert ls.wins(segment), "seed segment must win"
    seq = list(segment)
    lo, hi = 1, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if ls.wins(seq[:mid]):
            hi = mid
        else:
            lo = mid + 1
    seq = seq[:lo]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(seq):
            cand = seq[:i] + seq[i + 1:]
            if cand and ls.wins(cand):
                seq = cand
                changed = True
            else:
                i += 1
    return seq


def verify(stem, tokens, target):
    """Replay on TWO fresh envs; assert level reached and per-step frame equality."""
    runs = []
    for _ in range(2):
        env = make_env(stem)
        obs = env.reset()
        frames = [settled(obs).copy()]
        levels = [lv(obs)]
        for tok in tokens:
            obs = apply_tok(env, tok)
            frames.append(settled(obs).copy())
            levels.append(lv(obs))
        runs.append((frames, levels))
    (fa, la), (fb, lb) = runs
    assert la == lb, "levels_completed traces differ between replays"
    assert la[-1] >= target, f"replay reached level {la[-1]}, expected >= {target}"
    assert la[0] == 0 and max(la) == la[-1]
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert np.array_equal(x, y), f"frame mismatch at step {i}"
    return len(fa), la[-1]


def tok_json(tok):
    if tok[0] == "C":
        return {"name": "ACTION6", "x": tok[1], "y": tok[2]}
    return {"name": f"ACTION{tok[1]}"}


def wa30_planner_segment(env, target):
    """wa30: frame perception + A* over the validated grab-drag forward model
    (scripts/research_2026_07_01/wa30_planner.py). Blind BFS cannot reach wa30's
    ~71-action L1 depth; this is the July frame-only technique."""
    import wa30_planner as WP
    t0 = time.time()
    obs = env.reset()
    avatar, blocks, pads = WP.perceive(settled(obs))
    if avatar is None or not blocks or not pads:
        return None, dict(planner="wa30", perceived=False)
    plan = WP.plan_all(avatar, blocks, pads)
    if not plan:
        return None, dict(planner="wa30", perceived=True, plan=None)
    seg = []
    for a in plan:
        obs = apply_tok(env, ("S", int(a)))
        seg.append(("S", int(a)))
        if lv(obs) >= target:
            return seg, dict(planner="wa30-grab-drag A*", plan_len=len(plan),
                             elapsed_s=round(time.time() - t0, 1))
        if obs.state == GameState.GAME_OVER:
            return None, dict(planner="wa30", died_at=len(seg))
    return None, dict(planner="wa30", executed=len(seg), no_levelup=True)


def solve_game(stem, levels=1, budget_s=600):
    print(f"=== {stem} ({GAMES[stem]}) target levels={levels} budget={budget_s}s ===", flush=True)
    t_start = time.time()
    env = make_env(stem)
    env.reset()
    full = []            # minimized concat
    seg_stats = []
    won_levels = 0
    for k in range(1, levels + 1):
        remain = budget_s - (time.time() - t_start)
        if remain < 20:
            print(f"  L{k}: out of budget", flush=True)
            break
        if stem == "wa30":
            seg, st = wa30_planner_segment(env, target=k)
            if seg is None:
                print(f"  L{k}: planner failed ({st}); falling back to BFS", flush=True)
                seg, st = bfs_level(env, stem, target=k, budget_s=remain)
        else:
            seg, st = bfs_level(env, stem, target=k, budget_s=remain)
        if seg is None and st.get("exhausted") and CONFIG[stem]["fallback_clicks"]:
            print(f"  L{k}: graph exhausted without win ({st}); retrying with clicks", flush=True)
            remain = budget_s - (time.time() - t_start)
            if remain > 20:
                seg, st = bfs_level(env, stem, target=k, budget_s=remain, use_fallback=True)
        if seg is None:
            print(f"  L{k}: NOT solved — {st}", flush=True)
            seg_stats.append(dict(level=k, solved=False, **st))
            break
        print(f"  L{k}: solved raw len={len(seg)} {st}", flush=True)
        mseg = minimize_segment(stem, full, seg, target=k)
        print(f"  L{k}: minimized {len(seg)} -> {len(mseg)}", flush=True)
        seg_stats.append(dict(level=k, solved=True, raw_len=len(seg), min_len=len(mseg), **st))
        full += mseg
        won_levels = k
        # re-sync the search env: it advanced via the RAW segment; level starts are
        # canonical under ONLY_RESET_LEVELS, so its reset() is already at level k+1.
    if won_levels == 0:
        out = dict(game=stem, versioned_id=GAMES[stem], level=0, solved=False,
                   search_stats=seg_stats, derived="2026-08-18")
        path = os.path.join(FIXDIR, f"{stem}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  FAILED — recorded {path}", flush=True)
        return out
    frames_checked, reached = verify(stem, full, won_levels)
    total_s = round(time.time() - t_start, 1)
    out = dict(game=stem, versioned_id=GAMES[stem], level=reached,
               actions=[tok_json(t) for t in full], length=len(full), verified=True,
               frames_checked=frames_checked,
               search_stats=dict(levels=seg_stats, total_wall_s=total_s,
                                 method="reset-replay BFS, HUD-masked frame dedup, "
                                        "delete-and-replay minimization"),
               derived="2026-08-18")
    path = os.path.join(FIXDIR, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  VERIFIED level={reached} len={len(full)} frames_checked={frames_checked} "
          f"wall={total_s}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    stem = sys.argv[1]
    levels = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    solve_game(stem, levels, budget)
