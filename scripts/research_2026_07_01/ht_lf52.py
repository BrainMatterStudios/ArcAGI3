"""Frames-only solver for ARC-AGI-3 game lf52, level 0.

MECHANIC (reverse-engineered from environment_files/lf52/271a04aa/lf52.py):
    lf52 is PEG SOLITAIRE played entirely with ACTION6 (click).
      - The pegs are the color-14 (MAROON/E) 4x4 blocks on the board.
      - Click a peg  -> selects it and reveals a direction marker at every cell
        that is a LEGAL jump (engine: dghsidbuet -> xpcuvjyrgu -> qikmikecdf,
        which requires an adjacent peg to jump over and an empty floor cell to
        land on, exactly 2 cells away).
      - Click the landing cell (2 cells / 12px past the peg) -> executes the jump:
        the selected peg moves 2 cells and the jumped-over peg is REMOVED
        (engine: cfilhtifcb removes qcerbdpdcl, ddaguepwkt -= 1).
      - WIN is set (engine: cfilhtifcb -> tdcblgbfxw -> win() -> iajuzrgttrv=True,
        then next_level) the moment the jump leaves exactly ONE peg (ddaguepwkt==1).
    Board->screen mapping on L0: cell = 6px, peg center = (12.5+6*gx, 7.5+6*gy);
    a jump landing = peg_center + dir*12px, dir in {up,right,down,left}.

    Budget: L0 loses at asqvqzpfdi >= 64 clicks; our win uses 8. RESET zeroes the
    budget, games are deterministic, and scoring is MAX over plays, so a
    search-then-replay attack on the real env is valid and frames-only.

ATTACK (this file, frames+feedback ONLY -- no engine introspection):
    Replay-based DFS on the real (deterministic, resettable) env. A candidate
    "jump" = click(peg_center) then click(peg_center + dir*12). A jump is legal
    iff the color-14 peg count drops by exactly one and the game is not over.
    Recurse until obs.levels_completed>=1 (WIN). Then do one clean play applying
    the discovered click sequence.

Run:
    ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 \
        .venv/bin/python scripts/research_2026_07_01/ht_lf52.py
"""
from __future__ import annotations

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

PEG_COLOR = 14
CELL = 6
DIRS = [(0, -2 * CELL), (2 * CELL, 0), (0, 2 * CELL), (-2 * CELL, 0)]  # up,right,down,left landings


def _arcade():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("lf52"))
    return c, gid


def peg_centers(grid):
    """Frames-only: peg = connected component of PEG_COLOR. Return sorted (x,y) centers."""
    objs = [o for o in P.connected_components(grid, background=0) if o.color == PEG_COLOR]
    return sorted((int(round(o.centroid[1])), int(round(o.centroid[0]))) for o in objs)


def replay(c, gid, clicks):
    """Apply a list of (x,y) ACTION6 clicks on a fresh play. Return (obs, grid)."""
    env = c.make(game_id=gid, scorecard_id="x")
    obs = env.reset()
    for (x, y) in clicks:
        obs = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        if obs.state in (GameState.WIN, GameState.GAME_OVER) or obs.levels_completed >= 1:
            break
    return obs, P.to_grid(obs.frame)


def search(c, gid):
    """DFS for the winning click sequence using only frames + obs feedback."""
    _, g0 = replay(c, gid, [])
    seen: set = set()
    best: list = []

    def dfs(clicks, pegs) -> bool:
        key = frozenset(pegs)
        if key in seen:
            return False
        seen.add(key)
        for (px, py) in pegs:
            for (dx, dy) in DIRS:
                lx, ly = px + dx, py + dy
                if not (0 <= lx < 64 and 0 <= ly < 64):
                    continue
                trial = clicks + [(px, py), (lx, ly)]
                obs, grid = replay(c, gid, trial)
                if obs.levels_completed >= 1 or obs.state == GameState.WIN:
                    best[:] = trial
                    return True
                if obs.state == GameState.GAME_OVER:
                    continue
                np2 = peg_centers(grid)
                if len(np2) == len(pegs) - 1:  # exactly one peg jumped/removed => legal
                    if dfs(trial, np2):
                        return True
        return False

    dfs([], peg_centers(g0))
    return best


def main():
    c, gid = _arcade()
    seq = search(c, gid)
    if not seq:
        print("lf52 L0: NO WIN FOUND")
        return
    # Final clean play.
    env = c.make(game_id=gid, scorecard_id="x")
    obs = env.reset()
    for i, (x, y) in enumerate(seq, 1):
        obs = env.step(GameAction.ACTION6, data={"x": x, "y": y})
    print("lf52 L0 SOLVED")
    print("  clicks:", len(seq))
    print("  action sequence (ACTION6 x,y):", seq)
    print("  final state:", obs.state, "levels_completed:", obs.levels_completed)
    assert obs.levels_completed >= 1, "level 0 not completed"


if __name__ == "__main__":
    main()
