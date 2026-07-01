"""
Frames-ONLY solver for ARC-AGI-3 game tr87, level 0.

MECHANIC (decoded from environment_files/tr87/cd924810/tr87.py):
  - Move-only game, actions 1..4 (no clicks). A5/A6 unused.
  - Screen has three glyph regions, all drawn as 5x5 color-5 patterns inside
    colored boxes:
      * TOP: six REWRITE RULES.  Each rule = [LHS glyph in a color-10 box]
        "=" [RHS glyph in a color-7 box].  (LHS glyphs are family A, RHS
        family B.)  Rule i means: A-glyph_i  ->  B-glyph_i.
      * BOTTOM color-10 strip: the TARGET row (fixed) = a sequence of A glyphs.
      * BOTTOM color-7 strip: the WORKING row (player-controlled) = a sequence
        of B glyphs, initially scrambled.
  - WIN condition (bsqsshqpox): segment the TARGET row greedily by rule LHS
    (for L0 every LHS is length 1, so it's cell-by-cell) and require the
    WORKING row to equal the concatenation of the corresponding rule RHS
    glyphs -- i.e. working[i] must be rule(target[i]).  Match is by glyph
    IDENTITY (the 5x5 pattern's digit), NOT by rotation: sprites are randomly
    rotated at level start purely as a visual distractor, and the 7 patterns in
    each family are all rotation-distinct (verified), so identity == the
    rotation-invariant pattern.
  - Controls (each costs 1 energy; start 128; hit 0 => lose):
      ACTION3 / ACTION4 : move the selection cursor left/right among working cells.
      ACTION1 / ACTION2 : cycle the SELECTED working cell's glyph -/+ one step
                          (within its 7-member family, wrapping).
  - Deterministic + resettable; scoring = MAX over runs; so read-the-frame +
    drive the real env is a valid, robust attack.

FRAMES-ONLY ATTACK:
  1. Parse the frame into boxes (connected comps of {boxcolor,5}).  Small 7x7
     comps in the top are rule glyphs; the wide comps in the bottom are the
     target row (color 10) and working row (color 7).  Each box splits into
     5x5 glyph cells on a pitch of 7.
  2. Build rule map canon(LHS_pattern) -> RHS_pattern, where canon() is the
     min-over-4-rotations byte key (rotation invariant).
  3. required[i] = rule[ canon(target[i]) ]  (a B pattern).
  4. For each working cell i: move cursor to i, press ACTION2 until the cell's
     canon matches required[i]'s canon (<=6 presses).  Win fires on the final
     correct press.
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

BOX_LHS = 10  # color of LHS / target boxes  (family A shown here)
BOX_RHS = 7   # color of RHS / working boxes (family B shown here)
GLYPH = 5


def _components(mask):
    """4-connected components of a boolean mask -> list of (rows,cols) bbox."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                pix = []
                while stack:
                    r, c = stack.pop()
                    pix.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                rs = [p[0] for p in pix]
                cs = [p[1] for p in pix]
                comps.append((min(rs), max(rs), min(cs), max(cs)))
    return comps


def _canon(patt):
    """Rotation-invariant key: min over 4 rotations of the 5x5 bool pattern."""
    keys = []
    p = patt
    for _ in range(4):
        keys.append(p.astype(np.uint8).tobytes())
        p = np.rot90(p)
    return min(keys)


def _split_box(grid, r0, r1, c0, c1):
    """A box bbox -> list of 5x5 (==5) glyph masks, left to right (pitch 7)."""
    w = c1 - c0 + 1
    n = max(1, round(w / 7))
    glyphs = []
    for i in range(n):
        gc0 = c0 + 1 + 7 * i
        gr0 = r0 + 1
        sub = grid[gr0:gr0 + 5, gc0:gc0 + 5]
        glyphs.append(((sub == GLYPH), (gr0, gc0)))
    return glyphs


def parse_frame(grid):
    """Return dict with rule pairs, target row glyphs, working row glyphs, cursor."""
    H, W = grid.shape
    # split top vs bottom by the color-3 padding band (rows ~34-39 are solid 3)
    row_is_pad = np.all(grid == 3, axis=1)
    pad_rows = np.where(row_is_pad)[0]
    split = int(pad_rows[0]) if len(pad_rows) else H // 2

    def boxes_of(color):
        mask = (grid == color) | (grid == GLYPH)
        out = []
        for (r0, r1, c0, c1) in _components(mask):
            # a real box has this color forming its border ring; require >=5 tall
            if r1 - r0 >= 4 and c1 - c0 >= 4:
                # verify it actually contains this box color (not a foreign 5-blob)
                if np.any(grid[r0:r1 + 1, c0:c1 + 1] == color):
                    out.append((r0, r1, c0, c1))
        return out

    lhs_boxes = boxes_of(BOX_LHS)
    rhs_boxes = boxes_of(BOX_RHS)

    top_lhs = sorted([b for b in lhs_boxes if b[1] < split], key=lambda b: (b[0], b[2]))
    top_rhs = sorted([b for b in rhs_boxes if b[1] < split], key=lambda b: (b[0], b[2]))
    bot_lhs = [b for b in lhs_boxes if b[0] >= split]
    bot_rhs = [b for b in rhs_boxes if b[0] >= split]

    rules = []
    for lb, rb in zip(top_lhs, top_rhs):
        lg = _split_box(grid, *lb)[0][0]
        rg = _split_box(grid, *rb)[0][0]
        rules.append((lg, rg))

    target = []
    if bot_lhs:
        target = [m for (m, _p) in _split_box(grid, *bot_lhs[0])]
    working = []
    if bot_rhs:
        working = _split_box(grid, *bot_rhs[0])  # list of (mask,pos)

    # cursor: color-0 marker; its column identifies the selected working cell
    cur_col = None
    zeros = np.argwhere(grid == 0)
    if len(zeros):
        cur_col = int(np.mean(zeros[:, 1]))
    return {"rules": rules, "target": target, "working": working, "cursor_col": cur_col, "split": split}


def solve():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("tr87"))
    env = c.make(game_id=gid, scorecard_id="x")
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    info = parse_frame(grid)

    # build rule map: canon(LHS) -> RHS pattern
    rmap = {_canon(lg): rg for (lg, rg) in info["rules"]}
    print(f"[parse] {len(info['rules'])} rules, target cells={len(info['target'])}, working cells={len(info['working'])}")

    # required RHS pattern for each target cell
    required = []
    for t in info["target"]:
        key = _canon(t)
        if key not in rmap:
            raise RuntimeError("target glyph has no matching rule")
        required.append(_canon(rmap[key]))
    print(f"[plan] required (canon) computed for {len(required)} cells")

    ncell = len(info["working"])
    actions = []  # (name, id)

    def do(gid_action, name):
        nonlocal obs
        obs = env.step(gid_action)
        actions.append(name)
        return P.to_grid(obs.frame)

    # current cursor cell index: read from frame via cursor column vs working positions
    def cursor_index(grid):
        info2 = parse_frame(grid)
        if info2["cursor_col"] is None or not info2["working"]:
            return 0
        cols = [pos[1] for (_m, pos) in info2["working"]]
        return int(np.argmin([abs(info2["cursor_col"] - cc) for cc in cols]))

    cur = cursor_index(grid)

    for i in range(ncell):
        # navigate cursor to cell i (mod ncell); pick shorter direction
        while cur != i:
            fwd = (i - cur) % ncell
            bwd = (cur - i) % ncell
            if fwd <= bwd:
                grid = do(GameAction.ACTION4, "A4>")  # cursor +1
                cur = (cur + 1) % ncell
            else:
                grid = do(GameAction.ACTION3, "A3<")  # cursor -1
                cur = (cur - 1) % ncell
        # cycle this cell with ACTION2 until it matches required[i]
        for attempt in range(8):
            info2 = parse_frame(grid)
            if obs.state == GameState.WIN or obs.levels_completed >= 1:
                break
            curmask = info2["working"][i][0]
            if _canon(curmask) == required[i]:
                break
            grid = do(GameAction.ACTION2, "A2+")  # cycle glyph +1
        if obs.state == GameState.WIN or obs.levels_completed >= 1:
            break

    print(f"[result] state={obs.state} levels_completed={obs.levels_completed} energy_used={len(actions)}")
    print(f"[sequence] {len(actions)} actions: {' '.join(actions)}")
    won = obs.state == GameState.WIN or obs.levels_completed >= 1
    print("WIN" if won else "NOT WON")
    return won, actions


if __name__ == "__main__":
    solve()
