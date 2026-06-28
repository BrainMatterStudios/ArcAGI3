"""GO-EXPLORE MULTI-PLAY PoC (2026-06-28) — test whether a PHYSICALLY-FAITHFUL replay can capture the
max-over-plays scoring lever, defeating the masked-graph desync.

Mechanism under test (verified facts in docs/.../2026-06-28-engine-scoring-findings.md):
- per-game score = MAX over full-reset PLAYS; agent memory persists across reset.
- the masked-key graph geodesic DESYNCS (non-stationary key aliases states).
Fix: use a STATIONARY EXACT-frame key (hash of raw grid) so a stored ACTION TRAJECTORY replays
faithfully (vc33 is a click game with no hidden state -> exact frame == exact state). Go-Explore:
archive the shortest trajectory-from-reset to each cell; revisit promising cells (full-reset + replay,
which is faithful) and explore onward, progressively SHORTENING the path to each level-up. The final
scored play replays the shortest solution found.

Question: does the faithful replay (a) reproduce level-ups and (b) reach each level in FEWER actions
than a single SalienceExplorer play (the 0.33-regime baseline)?

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/goexplore_multiplay_poc.py [budget] [game] [explore_steps]
"""
from __future__ import annotations

import hashlib
import logging
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from arcagi3 import perception as P  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
REF = 200.0


def s_level(a):
    return min(1.15, (REF / max(a, 1)) ** 2)


def eff(marks):
    return round(sum(s_level(a) for a in marks), 3)


def ekey(frame_grid):
    return hashlib.blake2b(np.ascontiguousarray(frame_grid, dtype=np.int16).tobytes(), digest_size=16).digest()


def candidates(grid, available, rng):
    """Salience-prioritised candidate actions (same perception prior as SalienceExplorer)."""
    cands, weights = [], []
    for aid in (1, 2, 3, 4, 5):
        if aid in available:
            cands.append(("S", aid)); weights.append(1.0)
    if 6 in available:
        for x, y, prio in P.salient_click_targets(grid, max_targets=256, coarse_grid_step=4):
            cands.append(("C", int(x), int(y))); weights.append(1.0 / (int(prio) + 1.0))
    return cands, np.array(weights, dtype=float)


def _do(env, tok):
    if tok == ("reset",):
        return env.reset()
    if tok[0] == "S":
        return env.step(GameAction.from_id(tok[1]))
    return env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})


def salience_faithful_record(client, gid, budget):
    """Explore with the STRONG SalienceExplorer (reaches deep levels) while recording, per EXACT frame
    key, the shortest FULL action-stream from game-start (resets included as tokens) that reaches it.
    Returns reach_level_traj: level -> shortest faithful trajectory that reaches that level count."""
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["sfr"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)
    obs = env.reset()
    n = 0
    prev = 0
    cur = []                                   # full token stream from game-start (incl. resets)
    archive = {ekey(P.to_grid(obs.frame)): []}
    reach = {0: []}
    while n < budget:
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame), obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        obs = _do(env, tok)
        n += 1
        cur = cur + [tok]
        lv = int(obs.levels_completed or 0)
        if obs.state in (GameState.GAME_OVER,):
            continue
        nk = ekey(P.to_grid(obs.frame))
        if nk not in archive or len(cur) < len(archive[nk]):
            archive[nk] = list(cur)
        if lv not in reach or len(cur) < len(reach[lv]):
            reach[lv] = list(cur)
        prev = max(prev, lv)
    return reach, len(archive), n


def full_reset(env):
    """Force a full reset (levels_completed -> 0); a single reset may be a level-reset."""
    for _ in range(4):
        obs = env.reset()
        if int(obs.levels_completed or 0) == 0:
            return obs
    return obs


def replay(env, traj):
    obs = full_reset(env)
    for tok in traj:
        obs = _do(env, tok)
    return obs


def single_play_baseline(client, gid, budget):
    """SalienceExplorer single play — the 0.33-regime baseline marks."""
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["base"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)
    obs = env.reset(); n = 0; prev = 0; last = 0; marks = []
    while n < budget:
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame), obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        if tok == ("reset",):
            obs = env.reset(); continue
        obs = _do(env, tok); n += 1
        lv = int(obs.levels_completed or 0)
        if lv > prev:
            marks.append(n - last); last = n; prev = lv
    return prev, marks


def go_explore(client, gid, budget, explore_steps, seed=0):
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["goex"]))
    rng = np.random.default_rng(seed)
    obs = full_reset(env)
    root = ekey(P.to_grid(obs.frame))
    archive = {root: []}                 # exact key -> shortest trajectory from reset
    key_level = {root: 0}                # exact key -> levels_completed observed there
    reach_level_traj = {0: []}           # level -> shortest trajectory that reaches that level count
    n = 0
    while n < budget:
        # ---- select a cell to revisit: bias toward deepest level, then shortest trajectory ----
        keys = list(archive.keys())
        lvls = np.array([key_level[k] for k in keys], dtype=float)
        lens = np.array([len(archive[k]) for k in keys], dtype=float)
        w = (lvls + 1.0) ** 3 / (lens + 1.0)        # prefer deep + short
        w = w / w.sum()
        k = keys[int(rng.choice(len(keys), p=w))]
        traj = list(archive[k])
        obs = replay(env, traj)
        n += len(traj)
        cur_key = k
        # ---- explore onward ----
        for _ in range(explore_steps):
            if n >= budget or obs.state == GameState.WIN:
                break
            grid = P.to_grid(obs.frame)
            cands, ws = candidates(grid, list(obs.available_actions or []), rng)
            if not cands:
                break
            a = cands[int(rng.choice(len(cands), p=ws / ws.sum()))]
            obs = _do(env, a); n += 1
            traj = traj + [a]
            if obs.state == GameState.GAME_OVER:
                break
            lv = int(obs.levels_completed or 0)
            nk = ekey(P.to_grid(obs.frame))
            if nk not in archive or len(traj) < len(archive[nk]):
                archive[nk] = list(traj)
                key_level[nk] = lv
            if lv not in reach_level_traj or len(traj) < len(reach_level_traj[lv]):
                reach_level_traj[lv] = list(traj)
            cur_key = nk
    return archive, reach_level_traj, n


def marks_from_traj(client, gid, traj):
    """Replay a trajectory in a fresh play and record faithful actions-to-each-level."""
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["score"]))
    obs = full_reset(env)
    prev = 0; last = 0; marks = []
    for i, tok in enumerate(traj, 1):
        obs = _do(env, tok)
        lv = int(obs.levels_completed or 0)
        if lv > prev:
            marks.append(i - last); last = i; prev = lv
    return prev, marks


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    game = sys.argv[2] if len(sys.argv) > 2 else "vc33"
    explore_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("poc"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    print(f"GO-EXPLORE MULTI-PLAY PoC | game={game} budget={budget} explore_steps={explore_steps}\n")

    b_levels, b_marks = single_play_baseline(client, gid, budget)
    print(f"  single-play SalienceExplorer: levels={b_levels}  marks={b_marks}  eff={eff(b_marks):.3f}")

    # --- Method A: from-scratch Go-Explore (weak explorer, faithful replay) ---
    _arc, reach_ge, used_ge = go_explore(client, gid, budget, explore_steps)
    sol_ge = reach_ge[max(reach_ge)]
    ge_levels, ge_marks = marks_from_traj(client, gid, sol_ge)
    print(f"  [A] go-explore replay:        levels={ge_levels}  marks={ge_marks}  eff={eff(ge_marks):.3f}")

    # --- Method B: SalienceExplorer explore + exact-key faithful replay (the real design) ---
    reach_sf, ncells, used_sf = salience_faithful_record(client, gid, budget)
    sol_sf = reach_sf[max(reach_sf)]
    sf_levels, sf_marks = marks_from_traj(client, gid, sol_sf)
    print(f"  [B] salience+faithful replay: levels={sf_levels}  marks={sf_marks}  eff={eff(sf_marks):.3f}  "
          f"(archived {ncells} exact cells)")
    print()
    best = max(eff(b_marks), eff(ge_marks), eff(sf_marks))
    print(f"  RESULT: single-play eff={eff(b_marks):.3f}  |  best multi-play(replay) eff={best:.3f}  "
          f"-> lift={best - eff(b_marks):+.3f}")
    if sf_levels == 0 and ge_levels == 0:
        print("  VERDICT: replay FAILED to reproduce level-ups (faithfulness broken — hidden state?).")
    elif best > eff(b_marks) + 0.05:
        print("  VERDICT: LEVER WORKS — a faithful replay play beats single-play efficiency (max-over-plays).")
    else:
        print("  VERDICT: faithful replay reproduces levels but NO net efficiency gain yet.")


if __name__ == "__main__":
    main()
