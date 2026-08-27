"""FAMILY E+F cheap-prior probes (KILL TEST).

(E) STRUCTURAL-ANOMALY first-touch: greedy exploration that always tries the
    most structurally-anomalous interaction first (odd-one-out color, unique-size
    object, nearest-grid-center), interleaving the primitive move/A5 actions.
(F1) L0-MINIMAL exhaustive: bounded exhaustive single-action probing on L0 only
    (every primitive + every salient click target, fresh reset each trial),
    then a bounded depth-2 sweep over the top click targets.

Reports MEASURED numbers: reached_first_reward, actions_to_first_reward (or cap).
"""
import sys, time
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from arcagi3 import perception as P

GAMES = ["bp35", "g50t", "re86", "sb26", "sc25", "cd82", "tu93", "lp85"]

def mk(game):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    e = next(x for x in c.get_environments() if x.game_id.startswith(game))
    return c.make(game_id=e.game_id, scorecard_id="probe")

def grid_of(obs, last):
    if obs.frame is not None and len(obs.frame):
        return P.to_grid(obs.frame)
    return last

def reached(obs):
    try:
        return int(obs.levels_completed or 0) >= 1
    except Exception:
        return False

def avail(obs):
    aa = getattr(obs, "available_actions", None) or [1, 2, 3, 4, 5, 6]
    aa = [int(a) for a in aa]
    clicks = [a for a in aa if a in (6, 7)]
    moves = [a for a in aa if a in (1, 2, 3, 4, 5)]
    return clicks, moves

def do_click(env, a, x, y):
    return env.step(GameAction.from_id(a), data={"x": int(x), "y": int(y)})

# ---------------- anomaly scoring ----------------
def anomaly_targets(grid):
    """Return list of (x, y, score) click targets ranked by structural anomaly (higher=first)."""
    bg = P.detect_background(grid)
    objs = P.connected_components(grid, background=bg)
    if not objs:
        return []
    h, w = grid.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    color_counts = {}
    size_counts = {}
    for o in objs:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
        size_counts[o.size] = size_counts.get(o.size, 0) + 1
    maxdist = (cy ** 2 + cx ** 2) ** 0.5 + 1e-9
    out = []
    for o in objs:
        r, cc = o.centroid
        # odd-one-out color: rarer color -> higher
        color_anom = 1.0 / color_counts[o.color]
        # unique-size object: unique size -> higher
        size_anom = 1.0 / size_counts[o.size]
        # symmetric-center: closer to grid center -> higher
        dist = ((r - cy) ** 2 + (cc - cx) ** 2) ** 0.5
        center_anom = 1.0 - dist / maxdist
        # small objects more likely interactive
        small = 1.0 / (1.0 + o.size / 4.0)
        # de-prioritise status-bar-like flat wide blobs
        is_bar = (o.height <= 2 or o.width <= 2) and (o.width >= w * 0.6 or o.height >= h * 0.6)
        score = color_anom + size_anom + 0.5 * center_anom + 0.5 * small
        if is_bar:
            score *= 0.05
        out.append((int(round(cc)), int(round(r)), score))
    out.sort(key=lambda t: -t[2])
    return out

# ---------------- Probe E: anomaly first-touch greedy ----------------
def probe_E(game, cap=700):
    env = mk(game)
    obs = env.reset()
    last = grid_of(obs, np.zeros((64, 64), dtype=np.int8))
    if reached(obs):
        return True, 0
    clicks, moves = avail(obs)
    if not moves:
        moves = [1, 2, 3, 4]
    acts = 0
    tried = set()          # (statekey, click_action, x, y) already tried
    stagnate = 0
    mi = 0
    while acts < cap:
        g = grid_of(obs, last); last = g
        skey = P.object_state_key(g)
        tgts = anomaly_targets(g) if clicks else []
        did = False
        for x, y, sc in tgts:
            for ca in clicks:
                key = (skey, ca, x, y)
                if key in tried:
                    continue
                tried.add(key)
                prev = P.object_state_key(g)
                obs = do_click(env, ca, x, y); acts += 1; did = True
                ng = grid_of(obs, last)
                if reached(obs):
                    return True, acts
                stagnate = 0 if P.object_state_key(ng) != prev else stagnate + 1
                break
            if did:
                break
        if not did:
            a = moves[mi % len(moves)]; mi += 1
            obs = env.step(GameAction.from_id(a)); acts += 1
            if reached(obs):
                return True, acts
            stagnate += 1
        if stagnate and stagnate % 6 == 0:
            a = moves[mi % len(moves)]; mi += 1
            obs = env.step(GameAction.from_id(a)); acts += 1
            if reached(obs):
                return True, acts
        if stagnate > 80:
            break
    return False, acts

# ---------------- Probe F1: L0-minimal bounded exhaustive ----------------
def probe_F1(game, click_cap=300, depth2_top=12):
    acts = 0
    env = mk(game); obs0 = env.reset()
    clicks, moves = avail(obs0)
    g0 = grid_of(obs0, np.zeros((64, 64), dtype=np.int8))
    # 1) each available move replayed as a short sequence (games needing repeats)
    for a in moves:
        env = mk(game); obs = env.reset()
        if reached(obs):
            return True, acts
        for _ in range(25):
            obs = env.step(GameAction.from_id(a)); acts += 1
            if reached(obs):
                return True, acts
    # 1b) also try repeated same-spot clicks (some games count clicks, e.g. lp85)
    if clicks:
        for (x, y, prio) in P.salient_click_targets(g0, max_targets=24):
            for ca in clicks:
                env = mk(game); obs = env.reset()
                for _ in range(8):
                    obs = do_click(env, ca, x, y); acts += 1
                    if reached(obs):
                        return True, acts
    # 2) exhaustive single click on salient + coarse targets, fresh reset each
    hit_after_click = []
    if clicks:
        tgts = P.salient_click_targets(g0, coarse_grid_step=8, max_targets=click_cap)
        for (x, y, prio) in tgts:
            for ca in clicks:
                env = mk(game); obs = env.reset()
                prev = P.object_state_key(grid_of(obs, g0))
                obs = do_click(env, ca, x, y); acts += 1
                if reached(obs):
                    return True, acts
                ng = grid_of(obs, g0)
                if P.object_state_key(ng) != prev:
                    hit_after_click.append((ca, x, y))
            if acts > click_cap + 300:
                break
    # 3) bounded depth-2 from state-changing clicks: follow with each move + a second click
    for (ca, x, y) in hit_after_click[:depth2_top]:
        for a in moves:
            e2 = mk(game); o2 = e2.reset()
            o2 = do_click(e2, ca, x, y); acts += 1
            o2 = e2.step(GameAction.from_id(a)); acts += 1
            if reached(o2):
                return True, acts
        e2 = mk(game); o2 = e2.reset(); o2 = do_click(e2, ca, x, y)
        g1 = grid_of(o2, g0)
        for (x2, y2, _p) in P.salient_click_targets(g1, max_targets=20):
            for ca2 in clicks:
                e2 = mk(game); o2 = e2.reset()
                o2 = do_click(e2, ca, x, y); acts += 1
                o2 = do_click(e2, ca2, x2, y2); acts += 1
                if reached(o2):
                    return True, acts
    return False, acts

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    games = [which] if which in GAMES else GAMES
    print(f"{'game':6} | {'E_reach':7} {'E_acts':7} {'E_t':6} | {'F1_reach':8} {'F1_acts':7} {'F1_t':6}")
    for game in games:
        t0 = time.time()
        try:
            eR, eA = probe_E(game)
        except Exception as ex:
            eR, eA = f"ERR:{ex}", -1
        te = time.time() - t0
        t1 = time.time()
        try:
            fR, fA = probe_F1(game)
        except Exception as ex:
            fR, fA = f"ERR:{ex}", -1
        tf = time.time() - t1
        print(f"{game:6} | {str(eR):7} {eA:7} {te:6.1f} | {str(fR):8} {fA:7} {tf:6.1f}", flush=True)
