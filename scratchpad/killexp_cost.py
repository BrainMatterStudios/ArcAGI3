"""DEPLOYABLE consecutive-depth solver — NO copy.deepcopy anywhere (only env.reset()/env.step()).

Measures the REAL env-step cost of finding consecutive levels via prefix-replay, the way the online
REST eval forces (the env cannot be deepcopy-snapshotted).

Reset semantics (verified empirically, see reset_probe.py, and arcengine/base_game.handle_reset):
  * completing level k -> set_level(k+1) -> action_count := 0
  * env.reset() with action_count==0  -> FULL reset  (back to level 0)
  * env.reset() with action_count>0   -> LEVEL reset (restart current level, score preserved)
  => DOUBLE-reset always reaches level 0; a single reset after >=1 action restarts the current level.

Two deployable FIND positioning modes (both use ONLY reset()+step()):
  mode="prefix"     : before every candidate, double-reset to level 0 then REPLAY the full solved_prefix
                      (L0..L(k-1) winning actions). This is the literal "prefix-replay" the task asks to cost.
  mode="levelreset" : reach level k ONCE via prefix-replay, then retry level k in place with a single
                      level_reset per candidate (no prefix replay). The cheaper deployable path.

Every env.step and every env.reset counts as one real action.
"""
from __future__ import annotations
import sys, time
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

import os
MAXCLICK = 16          # salient click targets per level (bounds click branching)
CAP = int(os.getenv("KILLEXP_CAP", "200000"))     # hard per-game env-step cap
WALL = float(os.getenv("KILLEXP_WALL", "600"))    # hard per-(game,mode) wall-clock cap (s)
CAND_CAP = 40_000      # max candidates expanded per level (safety)
DEPTH_CAP = 60         # max macro-sequence length per level


def mk(game):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    e = next(x for x in c.get_environments() if x.game_id.startswith(game))
    return c.make(game_id=e.game_id, scorecard_id="prefixcost")


def lv(o):
    return int(getattr(o, "levels_completed", 0) or 0)


def is_over(o):
    return o.state == GameState.GAME_OVER


def grid_of(o, last):
    if o.frame is not None and len(o.frame):
        return P.to_grid(o.frame)
    return last


class Cnt:
    def __init__(self):
        self.actions = 0        # total real env interactions (steps + resets)
        self.repos = 0          # resets + solved-prefix replay (re-reaching level-k start)
        self.rewalk = 0         # re-treading a known candidate prefix inside the level
        self.novel = 0          # the genuinely-new frontier macro
        self.t0 = time.time()
        self.bucket = "repos"   # which bucket _apply/_reset charge to

    def out_of_budget(self):
        return self.actions >= CAP or (time.time() - self.t0) > WALL


def _charge(c, n=1):
    c.actions += n
    setattr(c, c.bucket, getattr(c, c.bucket) + n)


def _reset(env, c):
    c.repos += 1
    c.actions += 1
    return env.reset()


def _apply(env, tok, c, last):
    _charge(c)
    if tok[0] == "S":
        o = env.step(GameAction.from_id(tok[1]))
    else:  # ("C", x, y)
        o = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
    return o, grid_of(o, last)


def macros_for(grid, available):
    ms = [("S", a) for a in (1, 2, 3, 4, 5) if a in available]
    if 6 in available:
        for x, y, _p in P.salient_click_targets(grid, max_targets=MAXCLICK, coarse_grid_step=8):
            ms.append(("C", int(x), int(y)))
    return ms or [("S", a) for a in (available or [1])]


def goto_level0(env, c):
    """Guaranteed level-0 start via double reset (works from any state)."""
    o = _reset(env, c)          # 1st: level_reset if mid-level, else full
    o = _reset(env, c)          # 2nd: action_count==0 now -> FULL reset -> level 0
    return o


def replay(env, seq, c, last, split=False):
    """split=True: charge all but the final macro as `rewalk`, the final macro as `novel`."""
    o = None
    g = last
    prev = c.bucket
    for i, tok in enumerate(seq):
        if split:
            c.bucket = "novel" if i == len(seq) - 1 else "rewalk"
        o, g = _apply(env, tok, c, g)
        if is_over(o):
            c.bucket = prev
            return o, g, True
    c.bucket = prev
    return o, g, False


def goto_level_start(env, prefix, k, c, last):
    """Double-reset to level 0, replay solved_prefix -> land at start of level k."""
    o = goto_level0(env, c)
    g = grid_of(o, last)
    if prefix:
        o, g, over = replay(env, prefix, c, g)
    return o, g


def search_level(env, k, prefix, root_grid, root_avail, c, mode, log):
    """BFS-with-object-dedup over macro sequences for level k. env is AT level-k start.
    Returns the winning suffix (list of macros) or None if exhausted / capped."""
    macros = macros_for(root_grid, set(root_avail))
    queue = deque([[m] for m in macros])
    seen = set()
    seen.add(P.object_state_key(root_grid))
    cands = 0
    first = True
    last = root_grid
    start_actions = c.actions
    s_repos, s_rewalk, s_novel = c.repos, c.rewalk, c.novel

    while queue and not c.out_of_budget() and cands < CAND_CAP:
        seq = queue.popleft()
        if len(seq) > DEPTH_CAP:
            continue
        # ---- position at level-k start ----
        if first:
            first = False           # already there (just replayed prefix / prior solve)
        elif mode == "prefix":
            o, last = goto_level_start(env, prefix, k, c, last)
            if lv(o) != k:
                return None          # prefix desynced (should not happen) -> bail
        else:  # levelreset: single reset restarts level k (action_count>0 from prev candidate)
            o = _reset(env, c)
            last = grid_of(o, last)
            if lv(o) != k:           # unexpected full reset -> re-anchor via prefix
                o, last = goto_level_start(env, prefix, k, c, last)
        # ---- replay the candidate sequence ----
        o, g, over = replay(env, seq, c, last, split=True)
        last = g
        cands += 1
        if o is not None and lv(o) > k:
            log(f"    L{k} SOLVED: suffix_len={len(seq)} cands={cands} "
                f"level_actions={c.actions-start_actions} cum={c.actions} "
                f"repos={c.repos-s_repos} rewalk={c.rewalk-s_rewalk} novel={c.novel-s_novel}")
            return seq
        if over:
            continue
        # ---- expand frontier on genuinely-new object state ----
        key = P.object_state_key(g)
        if key not in seen:
            seen.add(key)
            for m in macros:
                queue.append(seq + [m])
    log(f"    L{k} UNSOLVED: cands={cands} queue={len(queue)} "
        f"level_actions={c.actions-start_actions} cum={c.actions} "
        f"repos={c.repos-s_repos} rewalk={c.rewalk-s_rewalk} novel={c.novel-s_novel} "
        f"reason={'cap' if c.actions>=CAP else ('candcap' if cands>=CAND_CAP else ('wall' if c.out_of_budget() else 'exhausted'))}")
    return None


def solve_game(game, mode, log):
    env = mk(game)
    c = Cnt()
    o = goto_level0(env, c)
    g = grid_of(o, np.zeros((64, 64), np.int8))
    prefix = []                     # concatenated winning suffixes L0..L(k-1)
    rows = []                       # (level_reached, cumulative_actions)
    k = 0
    in_place = False                # levelreset: env already sits at start of level k
    while not c.out_of_budget():
        # anchor at level-k start and capture root grid/avail
        if in_place:
            o, g = env.observation_space, g
            g = grid_of(o, g)
        else:
            o, g = goto_level_start(env, prefix, k, c, g)
        if lv(o) != k:
            log(f"    ! anchor failed at L{k}: levels={lv(o)} (expected {k})")
            break
        root_avail = list(o.available_actions or [])
        suffix = search_level(env, k, prefix, g, root_avail, c, mode, log)
        if suffix is None:
            break
        prefix += suffix
        k += 1
        rows.append((k, c.actions, c.repos, c.rewalk, c.novel))
        if mode == "levelreset":
            # we are already standing at the start of level k -> no prefix replay, no verify
            in_place = True
            continue
        # verify the extended prefix truly replays to level k on a fresh double-reset
        vo, vg = goto_level_start(env, prefix, k, c, g)
        g = vg
        log(f"    VERIFY prefix -> levels_completed={lv(vo)} (want {k}), cum={c.actions}")
        if lv(vo) != k:
            log(f"    ! prefix verify FAILED at L{k}")
            break
    return rows, c.actions, (c.repos, c.rewalk, c.novel)


def main():
    games = sys.argv[1:] or ["tu93", "cd82", "lp85"]
    modes = ["levelreset", "prefix"]
    results = {}
    for game in games:
        for mode in modes:
            t0 = time.time()

            def log(s, _g=game, _m=mode):
                print(f"[{_g}/{_m}] {s}", flush=True)

            log(f"=== start (CAP={CAP}) ===")
            try:
                rows, total, buckets = solve_game(game, mode, log)
            except Exception as ex:
                import traceback; traceback.print_exc()
                rows, total, buckets = [], -1, (0, 0, 0)
            dt = time.time() - t0
            results[(game, mode)] = (rows, total, dt, buckets)
            log(f"=== done: depth={len(rows)} total_actions={total} wall={dt:.0f}s ===")

    # ---- summary tables ----
    print("\n================ COST TABLE (depth reached vs cumulative real actions) ================")
    for (game, mode), (rows, total, dt, bk) in results.items():
        rp, rw, nv = bk
        tot = max(1, rp + rw + nv)
        print(f"\n{game} / {mode}  (total_actions={total}, wall={dt:.0f}s) "
              f"repos={rp} ({100*rp/tot:.1f}%) rewalk={rw} ({100*rw/tot:.1f}%) novel={nv} ({100*nv/tot:.1f}%)")
        print(f"  {'level':>6} | {'cum_actions':>12} | {'cum_repos':>10} {'cum_rewalk':>11} {'cum_novel':>10}")
        for r in rows:
            print(f"  {r[0]:>6} | {r[1]:>12} | {r[2]:>10} {r[3]:>11} {r[4]:>10}")
        if not rows:
            print("   (no level solved)")

    print("\n================ JSON ================")
    import json as _json
    print("KILLEXP_JSON " + _json.dumps({f"{g}|{m}": {"rows": r, "total": t, "wall": round(d,1),
          "repos": b[0], "rewalk": b[1], "novel": b[2]} for (g, m), (r, t, d, b) in results.items()}))

    print("\n================ DEEPEST CONSECUTIVE LEVEL PER BUDGET BRACKET ================")
    print(f"  {'game':>6} {'mode':>11} | {'<=10k':>6} {'<=50k':>6} {'<=150k':>7}")
    for (game, mode), (rows, total, dt, bk) in results.items():
        def deepest(budget):
            d = 0
            for (levreached, cum, *_x) in rows:
                if cum <= budget:
                    d = levreached
            return d
        print(f"  {game:>6} {mode:>11} | {deepest(10_000):>6} {deepest(50_000):>6} {deepest(150_000):>7}")


if __name__ == "__main__":
    main()
