"""DECISIVE live-replay test for the efficiency post-pass hybrid.

NO offline path-length arithmetic. We build a geodesic-compressed action
sequence from the duck's recorded exact-frame graph, then ACTUALLY REPLAY it
on a fresh engine and read LIVE obs.levels_completed. A replay that does not
reach the duck's completed-level depth has DESYNCED.

Scorer (reverse-engineered & verified to reproduce the duck's 1.781 mean exactly):
    final_score = sum_{completed level i} (i+1) * min((baseline_i/actions_i)^2 * 100, 100)
                  / sum(1..total_levels)
i.e. SQUARED efficiency, depth-weighted (weight = 1-based level), all-levels denominator.
"""
import json
import glob
import os
import re
from collections import deque, defaultdict

import numpy as np

from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from arcagi3.runner import P_to_grid

ART = "/tmp/repdone/artifacts"
ENVDIR = "environment_files"

MOUSE_RE = re.compile(r"row=(\d+),\s*col=(\d+)")


def load_actions(gid_full):
    p = f"{ART}/{gid_full}_p0_events.jsonl"
    out = []
    with open(p) as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "action":
                out.append(e)
    return out


def board_hash(board):
    return tuple(tuple(int(c) for c in row) for row in board)


def grid_hash(grid):
    a = np.asarray(grid).astype(int)
    return tuple(tuple(int(c) for c in row) for row in a)


def action_token(e):
    name = e.get("action_name")
    if name == "RESET":
        return ("R",)
    if name == "ACTION6":
        m = MOUSE_RE.search(e.get("action_display") or "")
        if not m:
            return None
        row, col = int(m.group(1)), int(m.group(2))
        return ("C", col, row)  # x=col, y=row  (verified convention)
    if name and name.startswith("ACTION"):
        return ("S", int(name[6:]))
    return None


def bfs(graph, src, dst):
    """Shortest action-label path from src board-hash to dst board-hash."""
    if src == dst:
        return []
    prev = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for tok, nxt in graph.get(cur, ()):  # (action_token, next_hash)
            if nxt not in prev:
                prev[nxt] = (cur, tok)
                if nxt == dst:
                    # reconstruct
                    seq = []
                    node = nxt
                    while prev[node] is not None:
                        pcur, ptok = prev[node]
                        seq.append(ptok)
                        node = pcur
                    return list(reversed(seq))
                q.append(nxt)
    return None  # unreachable


def build_graph_and_milestones(events, s0_hash):
    """Return (graph, milestones). graph: hash -> set of (token, next_hash)."""
    graph = defaultdict(set)
    milestones = []
    before = s0_hash
    for e in events:
        after = board_hash(e["board"])
        tok = action_token(e)
        if tok is not None:
            graph[before].add((tok, after))
        if e.get("level_completed"):
            milestones.append(after)
        before = after
    return graph, milestones


def replay(env, seq, total_levels, action_cap):
    """Execute seq on fresh engine. Return (live_completed, per_level_actions, trace)."""
    obs = env.reset()
    live_lc = int(obs.levels_completed or 0)
    per_level = [0] * (total_levels + 2)
    trace = []
    steps = 0
    for tok in seq:
        if steps >= action_cap:
            break
        if tok[0] == "R":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:  # click
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        steps += 1
        cur = min(live_lc, total_levels + 1)
        per_level[cur] += 1
        new_lc = int(obs.levels_completed or 0)
        if new_lc > live_lc:
            trace.append((steps, live_lc, new_lc))
            live_lc = new_lc
    return live_lc, per_level, trace, steps


def level_score(baseline_i, actions_i):
    if actions_i <= 0:
        return 0.0
    return min((baseline_i / actions_i) ** 2 * 100.0, 100.0)


def play_score(completed, per_level_actions, baseline, total_levels):
    """Depth-weighted squared score over `completed` levels."""
    num = 0.0
    for j in range(completed):
        b = baseline[j] if j < len(baseline) else baseline[-1]
        num += (j + 1) * level_score(b, per_level_actions[j])
    den = sum(range(1, total_levels + 1))
    return num / den if den else 0.0


def main():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    gid_map = {}
    for p in sorted(glob.glob(f"{ART}/*_p0_events.jsonl")):
        full = os.path.basename(p).split("_p0_")[0]
        gid_map[full.split("-")[0]] = full

    viewer = {}
    for p in sorted(glob.glob(f"{ART}/*_p0_viewer_data.json")):
        d = json.load(open(p))
        viewer[os.path.basename(p).split("-")[0]] = d

    rows = []
    for gid in sorted(gid_map):
        full = gid_map[gid]
        events = load_actions(full)
        meta = json.load(open(glob.glob(f"{ENVDIR}/{gid}/*/metadata.json")[0]))
        baseline = meta["baseline_actions"]
        total_levels = len(baseline)
        vd = viewer[gid]
        duck_score = vd["final_score"]
        duck_completed = vd["levels_completed"]

        # classify move vs click by duck action mix
        n6 = sum(1 for e in events if e.get("action_name") == "ACTION6")
        nmove = sum(1 for e in events if (e.get("action_name") or "").startswith("ACTION") and e.get("action_name") != "ACTION6")
        cls = "click" if n6 >= nmove else "move"

        env = arc.make(gid)
        obs = env.reset()
        s0 = grid_hash(P_to_grid(obs.frame))

        graph, milestones = build_graph_and_milestones(events, s0)
        K = len(milestones)

        if K == 0:
            rows.append(dict(gid=gid, cls=cls, duck_lvls=duck_completed, duck_score=duck_score,
                             replay_lvls=0, faithful=False, replay_score=0.0,
                             hybrid=duck_score, gain=0.0, seq_len=0, note="duck completed 0"))
            print(f"{gid:5s} {cls:5s} K=0  duck completed 0 levels -> skip replay")
            continue

        # geodesic compress: S0 -> m1 -> ... -> mK
        seq = []
        cur = s0
        unreachable = False
        for m in milestones:
            part = bfs(graph, cur, m)
            if part is None:
                unreachable = True
                break
            seq.extend(part)
            cur = m

        if unreachable:
            rows.append(dict(gid=gid, cls=cls, duck_lvls=duck_completed, duck_score=duck_score,
                             replay_lvls=0, faithful=False, replay_score=0.0,
                             hybrid=duck_score, gain=0.0, seq_len=0, note="graph unreachable"))
            print(f"{gid:5s} {cls:5s} milestone UNREACHABLE in exact-frame graph -> no replay")
            continue

        action_cap = len(seq) + 5
        env2 = arc.make(gid)
        live_lc, per_level, trace, steps = replay(env2, seq, total_levels, action_cap)

        faithful = live_lc >= K
        replay_score = play_score(min(live_lc, K), per_level, baseline, total_levels)
        # honest: only credit up to duck's completed depth K
        hybrid = max(duck_score, replay_score)
        gain = hybrid - duck_score

        rows.append(dict(gid=gid, cls=cls, duck_lvls=duck_completed, duck_score=duck_score,
                         replay_lvls=live_lc, faithful=faithful, replay_score=replay_score,
                         hybrid=hybrid, gain=gain, seq_len=len(seq), note=""))
        tr = ",".join(f"@{s}:{a}->{b}" for s, a, b in trace) or "none"
        print(f"{gid:5s} {cls:5s} K={K} seq={len(seq):4d} (duck acts {len(events)}) "
              f"LIVE completions[{tr}] reached_lvl={live_lc} faithful={faithful} "
              f"replay_pl={per_level[:K]} duck={duck_score:.4f} replay={replay_score:.4f} hybrid={hybrid:.4f}")

    # ---- report ----
    print("\n=== PER-GAME TABLE ===")
    hdr = f"{'game':5s} {'class':5s} {'duckL':5s} {'duckScore':10s} {'replayL':7s} {'faith':6s} {'replayScore':12s} {'hybrid':10s} {'gain':10s}"
    print(hdr)
    for r in rows:
        print(f"{r['gid']:5s} {r['cls']:5s} {r['duck_lvls']:<5d} {r['duck_score']:<10.4f} "
              f"{r['replay_lvls']:<7d} {str(r['faithful']):6s} {r['replay_score']:<12.4f} "
              f"{r['hybrid']:<10.4f} {r['gain']:<+10.4f} {r['note']}")

    duck_mean = sum(r["duck_score"] for r in rows) / len(rows)
    hybrid_mean = sum(r["hybrid"] for r in rows) / len(rows)
    n_faithful = sum(1 for r in rows if r["faithful"])
    n_gain = sum(1 for r in rows if r["gain"] > 1e-9)

    def submean(cls, key):
        sub = [r for r in rows if r["cls"] == cls]
        return (sum(r[key] for r in sub) / len(sub) if sub else 0.0), len(sub)

    dmv, nmv = submean("move", "duck_score")
    hmv, _ = submean("move", "hybrid")
    dcl, ncl = submean("click", "duck_score")
    hcl, _ = submean("click", "hybrid")

    print("\n=== VERDICT NUMBERS ===")
    print(f"duck mean (25 games)   = {duck_mean:.4f}   (reference 1.781)")
    print(f"HYBRID mean (25 games) = {hybrid_mean:.4f}")
    print(f"absolute gain          = {hybrid_mean - duck_mean:+.4f}   ({100*(hybrid_mean-duck_mean)/duck_mean:+.1f}%)")
    print(f"games where replay faithfully reached duck depth = {n_faithful}/25")
    print(f"games with any positive gain                     = {n_gain}/25")
    print(f"MOVE  games (n={nmv}): duck {dmv:.4f} -> hybrid {hmv:.4f}  (gain {hmv-dmv:+.4f})")
    print(f"CLICK games (n={ncl}): duck {dcl:.4f} -> hybrid {hcl:.4f}  (gain {hcl-dcl:+.4f})")


if __name__ == "__main__":
    main()
