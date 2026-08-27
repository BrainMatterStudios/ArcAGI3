"""DEPTH-EXTENSION PoC (v2, faithful reset semantics).

From the duck's recorded trajectory, replay its exact action stream VERBATIM (deployable: single full
reset from a fresh env, then the duck's tokens incl. its own RESETs -- the stream self-recovers through
transient GAME_OVERs) to reach its furthest completed level L. Then run OUR blind search-replay
(affordance-pruned BFS, the machinery that caps ~0.33 on novel mechanics) to try to complete L, L+1, ...

Deployable-style: only env.reset()/env.step(); NO deepcopy. LIVE-verifies every extension by a fresh
full replay (phantom guard). Separates PREFIX-REPLAY cost from SEARCH cost.
"""
from __future__ import annotations
import sys, os, json, time, glob, re
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from prefix_replay_cost import mk, lv, is_over, grid_of, Cnt, macros_for
from arcagi3 import perception as P
from arcengine import GameAction

ART = "/tmp/repdone/artifacts"
MOUSE_RE = re.compile(r"row=(\d+),\s*col=(\d+)")
Z = np.zeros((64, 64), np.int8)


def parse_tokens(events_path):
    tokens, L = [], 0
    for line in open(events_path):
        d = json.loads(line)
        if d.get("type") != "action":
            continue
        nm = d["action_name"]
        sc = int(d.get("score") or 0)
        L = max(L, sc)
        if nm == "RESET":
            tok = ("R",)
        elif nm == "ACTION6":
            m = MOUSE_RE.search(d.get("action_display", ""))
            tok = ("C", int(m.group(2)), int(m.group(1))) if m else ("R",)
        else:
            tok = ("S", int(nm[-1]))
        tokens.append((sc, tok))
    return tokens, L


def prefix_to_level(tokens, L):
    if L <= 0:
        return []
    for i, (sc, tok) in enumerate(tokens):
        if sc >= L:
            return [t for _, t in tokens[: i + 1]]
    return [t for _, t in tokens]


def step_tok(env, tok, c):
    c.actions += 1
    if tok[0] == "R":
        return env.reset()
    if tok[0] == "S":
        return env.step(GameAction.from_id(tok[1]))
    return env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})


def replay_verbatim(env, toks, c, last, last_o=None, stop_on_over=False):
    """Replay a token stream verbatim. Does NOT bail on transient game-over unless asked."""
    o, g = last_o, last
    for tok in toks:
        o = step_tok(env, tok, c)
        g = grid_of(o, g)
        if stop_on_over and is_over(o):
            return o, g, True
    return o, g, False


def position_at_L(game, duck_prefix, c):
    """Fresh env + single full reset + verbatim duck-prefix -> at level L. Returns (env,o,g)."""
    env = mk(game)
    c.actions += 1
    o = env.reset()                      # fresh env, action_count==0 -> FULL reset to level 0
    g = grid_of(o, Z)
    o, g, _ = replay_verbatim(env, duck_prefix, c, g, last_o=o)
    return env, o, g


def faithful_search(env, k, duck_prefix, root_grid, root_avail, c, budget, game, log):
    """Blind affordance-pruned BFS on level k. env is AT level-k start (action_count==0).
    Between candidates: level-reset (single reset, valid once action_count>0). On desync/over that
    a level-reset can't fix: reposition via fresh env + duck-prefix. Returns winning suffix or None."""
    macros = macros_for(root_grid, set(root_avail))
    queue = deque([[m] for m in macros])
    seen = {P.object_state_key(root_grid)}
    first = True
    last = root_grid
    start = c.actions
    cands = 0
    env_ref = {"env": env}

    def reposition():
        e2, o2, g2 = position_at_L(game, duck_prefix, c)
        env_ref["env"] = e2
        return o2, g2

    while queue and (c.actions - start) < budget:
        seq = queue.popleft()
        if len(seq) > 60:
            continue
        e = env_ref["env"]
        if first:
            first = False
        else:
            c.actions += 1
            o = e.reset()                       # level-reset (action_count>0 from prior candidate)
            last = grid_of(o, last)
            if lv(o) != k:                      # full-reset trap or over -> reposition faithfully
                o, last = reposition()
                e = env_ref["env"]
                if lv(o) != k:
                    log(f"    L{k}: reposition failed (lv={lv(o)}); abort")
                    return None
        # apply candidate
        o, g, over = replay_verbatim(e, seq, c, last, stop_on_over=True)
        last = g
        cands += 1
        if o is not None and lv(o) > k:
            log(f"    L{k} SOLVED: suffix_len={len(seq)} cands={cands} "
                f"search_actions={c.actions-start}")
            return seq
        if over:
            continue
        key = P.object_state_key(g)
        if key not in seen:
            seen.add(key)
            for m in macros:
                queue.append(seq + [m])
    log(f"    L{k} UNSOLVED: cands={cands} queue={len(queue)} search_actions={c.actions-start} "
        f"reason={'budget' if (c.actions-start)>=budget else 'exhausted'}")
    return None


def run_game(game, events_path, budget, max_extra, log):
    tokens, L = parse_tokens(events_path)
    duck_prefix = prefix_to_level(tokens, L)

    c = Cnt()
    env, o, g = position_at_L(game, duck_prefix, c)
    prefix_cost = c.actions
    reached = lv(o) if o is not None else 0
    log(f"prefix-replay: duck_L={L} tokens={len(duck_prefix)} -> LIVE levels_completed={reached} "
        f"(cost={prefix_cost})")
    if reached != L:
        log(f"  ! prefix desync (reached {reached} != {L}); skip")
        return dict(game=game, duck_L=L, total=None, prefix_ok=False, prefix_cost=prefix_cost,
                    reached=reached, extensions=[], search_actions=0)

    ext_prefix = list(duck_prefix)
    extensions = []
    search_start = c.actions
    k = L
    while len(extensions) < max_extra:
        root_avail = list(o.available_actions or [])
        suffix = faithful_search(env, k, ext_prefix, g, root_avail, c, budget, game, log)
        if suffix is None:
            log(f"  L{k}: NOT extended")
            break
        ext_prefix = ext_prefix + suffix
        # LIVE phantom-guard: fresh env + full replay of extended prefix
        c.actions += 1
        venv = mk(game)
        vo = venv.reset(); vc = Cnt()
        vg = grid_of(vo, Z)
        vo, vg, _ = replay_verbatim(venv, ext_prefix, vc, vg)
        live = lv(vo) if vo is not None else -1
        ok = (live == k + 1)
        extensions.append(dict(level_index=k, suffix_len=len(suffix),
                               live_levels_completed=live, verified=ok))
        log(f"  L{k}->L{k+1}: EXTENDED suffix_len={len(suffix)} "
            f"LIVE_VERIFY levels_completed={live} verified={ok}")
        if not ok:
            log(f"  ! phantom at L{k}; stop")
            break
        # continue from the verified env
        env, o, g = venv, vo, vg
        k += 1

    return dict(game=game, duck_L=L, prefix_ok=True, prefix_cost=prefix_cost, reached=reached,
                extensions=extensions, search_actions=c.actions - search_start,
                total_actions=c.actions)


def main():
    budget = int(os.environ.get("BUDGET", "6000"))
    max_extra = int(os.environ.get("MAXEXTRA", "3"))
    only = set(sys.argv[1:])
    files = sorted(glob.glob(f"{ART}/*_events.jsonl"))
    results = []
    for f in files:
        game = os.path.basename(f)[:4]
        if only and game not in only:
            continue

        def log(s, _g=game):
            print(f"[{_g}] {s}", flush=True)

        t0 = time.time()
        try:
            r = run_game(game, f, budget, max_extra, log)
        except Exception as ex:
            import traceback; traceback.print_exc()
            r = dict(game=game, error=str(ex))
        r["wall"] = time.time() - t0
        results.append(r)
        log(f"=== done wall={r['wall']:.0f}s ===\n")

    print("\n================ DEPTH-EXTENSION SUMMARY ================")
    print(f"budget/level={budget} max_extra={max_extra}")
    print(f"{'game':>5} {'duckL':>5} {'pfxcost':>7} {'extra':>5} {'search_act':>10}  verified_live")
    n_ext = 0
    for r in results:
        if r.get("error"):
            print(f"{r['game']:>5}  ERROR {r['error']}")
            continue
        exts = r.get("extensions", [])
        got = sum(1 for e in exts if e["verified"])
        n_ext += 1 if got > 0 else 0
        lives = ",".join(str(e["live_levels_completed"]) for e in exts) or "-"
        print(f"{r['game']:>5} {r['duck_L']:>5} {r['prefix_cost']:>7} {got:>5} "
              f"{r.get('search_actions',0):>10}  {lives}")
    print(f"\nGAMES EXTENDED (>=+1 verified level): {n_ext}/{len(results)}")
    with open(os.path.join(os.path.dirname(__file__), "depth_extension_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
