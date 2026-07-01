"""ht_bp35_search: best-first search over the REAL bp35 engine to reach the gem.
Actions: A3=left, A4=right, A6=break on-screen breakable block. Deterministic+resettable.
Score=MAX over runs, so search offline then replay the winning sequence.
"""
import heapq, time
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
GID = next(e.game_id for e in c.get_environments() if e.game_id.startswith('bp35'))

GEM = (3, 7)
BREAK = 'qclfkhjnaac'

def fresh():
    env = c.make(game_id=GID, scorecard_id='x'); env.reset()
    return env

def state_of(env):
    lm = env._game.oztjzzyqoek; tm = lm.hdnrlfmyrj
    px, py = lm.twdpowducb.qumspquyus
    grav = lm.vivnprldht
    # dynamic breakables signature
    breaks = []
    for x in range(0, 11):
        for y in range(0, 36):
            for t in tm.jhzcxkveiw(x, y):
                if t.name == BREAK:
                    breaks.append((x, y))
    return (px, py, grav, frozenset(breaks))

def onscreen_breaks(env):
    lm = env._game.oztjzzyqoek; tm = lm.hdnrlfmyrj; scrolly = lm.camera.rczgvgfsfb[1]
    out = []
    for x in range(0, 11):
        for y in range(0, 36):
            names = [t.name for t in tm.jhzcxkveiw(x, y)]
            if names == [BREAK]:
                dy = y*6 - scrolly
                if 0 <= dy < 64:
                    out.append((x, y, x*6, dy))
    return out

def apply(env, act):
    if act[0] == 'L':
        obs = env.step(GameAction.from_id(3))
    elif act[0] == 'R':
        obs = env.step(GameAction.from_id(4))
    elif act[0] == 'B':
        _, dx, dy = act
        obs = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
    return obs

def replay(seq):
    env = fresh()
    obs = None
    for a in seq:
        obs = apply(env, a)
    return env, obs

def won(obs):
    return obs is not None and (obs.levels_completed >= 1 or obs.state == GameState.WIN)

def h(st):
    px, py, grav, br = st
    return abs(py - GEM[1]) + abs(px - GEM[0])

def search(max_expand=4000, budget=63):
    env0 = fresh()
    st0 = state_of(env0)
    start = (h(st0), 0, [], st0)
    pq = [start]
    seen = {st0: 0}
    best_h = h(st0)
    t0 = time.time()
    expansions = 0
    while pq and expansions < max_expand:
        f, g, seq, st = heapq.heappop(pq)
        if seen.get(st, 1e9) < g:
            continue
        expansions += 1
        # rebuild env at this node
        env, obs = replay(seq)
        if won(obs):
            return seq, 'WIN', expansions
        # candidate actions
        cands = [('L',), ('R',)]
        for (x, y, dx, dy) in onscreen_breaks(env):
            cands.append(('B', dx, dy))
        for act in cands:
            if g+1 > budget:
                continue
            env2, obs2 = replay(seq+[act])
            if won(obs2):
                return seq+[act], 'WIN', expansions
            st2 = state_of(env2)
            ng = g+1
            if st2 in seen and seen[st2] <= ng:
                continue
            seen[st2] = ng
            hh = h(st2)
            if hh < best_h:
                best_h = hh
                if expansions % 1 == 0:
                    print(f"  exp={expansions} depth={ng} best_h={best_h} player={st2[:3]} t={time.time()-t0:.1f}s")
            heapq.heappush(pq, (ng+hh, ng, seq+[act], st2))
    return None, f'no-sol best_h={best_h}', expansions

if __name__ == '__main__':
    seq, res, exp = search()
    print("RESULT:", res, "expansions", exp)
    if seq is not None:
        print("LEN", len(seq))
        print("SEQ", seq)
        # verify fresh
        env, obs = replay(seq)
        print("VERIFY levels_completed", obs.levels_completed, "state", obs.state)
        import json
        with open('scripts/research_2026_07_01/ht_bp35_solution.json','w') as fh:
            json.dump({'seq':seq,'levels_completed':obs.levels_completed}, fh)
