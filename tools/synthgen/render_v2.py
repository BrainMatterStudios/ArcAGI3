"""Render v2 GameSpecs into self-contained ARC-AGI-3-style game files.

Same contract as render_game.py (single ARCBaseGame subclass, arcengine-only
imports, loadable via arc_agi local_wrapper). v1 families delegate to the v1
renderer; this module adds bodies for replay/carry/mirror/rules whose runtime
semantics mirror the families_v2 simulators EXACTLY (validate replays prove
it).
"""

from __future__ import annotations

import json
import pprint
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from render_game import _HEADER, class_name_for
from render_game import write_env_dir as write_env_dir_v1

_REPLAY_BODY = '''

_CELL = PARAMS["cell_px"]
_ROWS = PARAMS["rows"]
_COLS = PARAMS["cols"]
_COL = PARAMS["colors"]

_DIRS = {{1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}}


def _cells_sprite(cells, color, name, layer):
    px = np.full((_ROWS * _CELL, _COLS * _CELL), -1, dtype=np.int8)
    for r, c in cells:
        px[r * _CELL : (r + 1) * _CELL, c * _CELL : (c + 1) * _CELL] = color
    return Sprite(pixels=px, name=name, layer=layer, tags=[])


def _cell_sprite(color, name, layer, r, c, mask=None):
    if mask is None:
        px = [[color] * _CELL for _ in range(_CELL)]
    else:
        px = [[color if v else -1 for v in row] for row in mask]
    return Sprite(pixels=px, name=name, layer=layer, x=c * _CELL, y=r * _CELL, tags=[])


def _cells(grid, chars):
    return [
        (r, c) for r in range(_ROWS) for c in range(_COLS) if grid[r][c] in chars
    ]


def _build_level(idx, spec_level):
    grid = spec_level["map"]
    sprites = [_cells_sprite(_cells(grid, "#"), _COL["wall"], "walls", 0)]
    (pr, pc) = _cells(grid, "P")[0]
    (lr, lc) = _cells(grid, "L")[0]
    (gr, gc) = _cells(grid, "G")[0]
    sprites.append(_cell_sprite(_COL["plate"], "plate", 1, lr, lc))
    sprites.append(_cell_sprite(_COL["goal"], "goal", 1, gr, gc))
    for dr, dc in _cells(grid, "D"):
        sprites.append(_cell_sprite(_COL["door"], "door", 2, dr, dc))
    ghost = _cell_sprite(_COL["ghost"], "ghost", 4, pr, pc, mask=PARAMS["player_shape"])
    ghost.set_visible(False)
    sprites.append(ghost)
    sprites.append(_cell_sprite(_COL["player"], "player", 5, pr, pc, mask=PARAMS["player_shape"]))
    return Level(sprites=sprites, grid_size=(_COLS * _CELL, _ROWS * _CELL), name="L%d" % (idx + 1))


_LEVELS = [_build_level(i, lv) for i, lv in enumerate(PARAMS["levels"])]


class {cls}(ARCBaseGame):
    def __init__(self) -> None:
        self._hud = _Hud(_COL["hud_fill"], _COL["hud_empty"]) if PARAMS["hud"] else None
        camera = Camera(
            background=_COL["bg"],
            letter_box=_COL["letterbox"],
            interfaces=[self._hud] if self._hud else [],
        )
        super().__init__(
            game_id="{gid}",
            levels=_LEVELS,
            camera=camera,
            available_actions=PARAMS["available_actions"],
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        grid = PARAMS["levels"][idx]["map"]
        self._grid = grid
        self._start = _cells(grid, "P")[0]
        self._plate = _cells(grid, "L")[0]
        self._goal = _cells(grid, "G")[0]
        self._doors = set(_cells(grid, "D"))
        self._pos = self._start
        self._recorded = []
        self._banked = False
        self._ghost_path = []
        self._ghost_pos = None
        self._ghost_idx = 0
        self._budget = PARAMS["budget"][idx]
        if self._hud:
            self._hud.reset(self._budget)

    def _wall(self, r, c):
        if not (0 <= r < _ROWS and 0 <= c < _COLS):
            return True
        return self._grid[r][c] == "#"

    def _plate_occupied(self):
        return self._pos == self._plate or (self._banked and self._ghost_pos == self._plate)

    def _move_sprite(self, name, r, c):
        for s in self.current_level.get_sprites_by_name(name):
            s.set_position(c * _CELL, r * _CELL)

    def _sync_door(self):
        occupied = self._plate_occupied()
        for s in self.current_level.get_sprites_by_name("door"):
            s.set_visible(not occupied)

    def step(self) -> None:
        if self._hud:
            self._hud.update(self._budget - self._action_count)
        if self._action_count > self._budget:
            self.lose()
            self.complete_action()
            return
        a = self.action.id.value
        if a == 5:
            if not self._banked and self._recorded:
                self._banked = True
                self._ghost_path = list(self._recorded)
                self._ghost_pos = self._start
                self._ghost_idx = 0
                self._pos = self._start
                self._move_sprite("player", *self._pos)
                self._move_sprite("ghost", *self._ghost_pos)
                for s in self.current_level.get_sprites_by_name("ghost"):
                    s.set_visible(True)
            self._sync_door()
            self.complete_action()
            return
        d = _DIRS.get(a)
        if d is None:
            self.complete_action()
            return
        if self._banked and self._ghost_idx < len(self._ghost_path):
            gd = self._ghost_path[self._ghost_idx]
            self._ghost_pos = (self._ghost_pos[0] + gd[0], self._ghost_pos[1] + gd[1])
            self._ghost_idx += 1
            self._move_sprite("ghost", *self._ghost_pos)
        nr, nc = self._pos[0] + d[0], self._pos[1] + d[1]
        blocked = self._wall(nr, nc) or (
            (nr, nc) in self._doors and not self._plate_occupied()
        )
        if not blocked:
            self._pos = (nr, nc)
            self._move_sprite("player", nr, nc)
            if not self._banked:
                self._recorded.append(d)
            if self._pos == self._goal:
                self.next_level()
                self._sync_door()
                self.complete_action()
                return
        self._sync_door()
        self.complete_action()
'''

_CARRY_BODY = '''

_CELL = PARAMS["cell_px"]
_ROWS = PARAMS["rows"]
_COLS = PARAMS["cols"]
_COL = PARAMS["colors"]

_DIRS = {{1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}}


def _cells_sprite(cells, color, name, layer):
    px = np.full((_ROWS * _CELL, _COLS * _CELL), -1, dtype=np.int8)
    for r, c in cells:
        px[r * _CELL : (r + 1) * _CELL, c * _CELL : (c + 1) * _CELL] = color
    return Sprite(pixels=px, name=name, layer=layer, tags=[])


def _cell_sprite(color, name, layer, r, c):
    px = [[color] * _CELL for _ in range(_CELL)]
    return Sprite(pixels=px, name=name, layer=layer, x=c * _CELL, y=r * _CELL, tags=[])


def _cells(grid, chars):
    return [
        (r, c) for r in range(_ROWS) for c in range(_COLS) if grid[r][c] in chars
    ]


def _build_level(idx, spec_level):
    grid = spec_level["map"]
    sprites = [_cells_sprite(_cells(grid, "#"), _COL["wall"], "walls", 0)]
    sprites.append(_cells_sprite(_cells(grid, "Z"), _COL["zone"], "zone", 1))
    for i, (r, c) in enumerate(_cells(grid, "B")):
        sprites.append(_cell_sprite(_COL["block"], "block_%d" % i, 3, r, c))
    (pr, pc) = _cells(grid, "P")[0]
    sprites.append(_cell_sprite(_COL["player"], "player", 5, pr, pc))
    nose = Sprite(pixels=[[_COL["nose"]]], name="nose", layer=6, x=pc * _CELL, y=pr * _CELL, tags=[])
    nose.set_visible(False)
    sprites.append(nose)
    return Level(sprites=sprites, grid_size=(_COLS * _CELL, _ROWS * _CELL), name="L%d" % (idx + 1))


_LEVELS = [_build_level(i, lv) for i, lv in enumerate(PARAMS["levels"])]


class {cls}(ARCBaseGame):
    def __init__(self) -> None:
        self._hud = _Hud(_COL["hud_fill"], _COL["hud_empty"]) if PARAMS["hud"] else None
        camera = Camera(
            background=_COL["bg"],
            letter_box=_COL["letterbox"],
            interfaces=[self._hud] if self._hud else [],
        )
        super().__init__(
            game_id="{gid}",
            levels=_LEVELS,
            camera=camera,
            available_actions=PARAMS["available_actions"],
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        grid = PARAMS["levels"][idx]["map"]
        self._grid = grid
        self._pos = _cells(grid, "P")[0]
        self._blocks = _cells(grid, "B")
        self._zone = set(_cells(grid, "Z"))
        self._facing = None
        self._carrying = None
        self._budget = PARAMS["budget"][idx]
        if self._hud:
            self._hud.reset(self._budget)

    def _wall(self, r, c):
        if not (0 <= r < _ROWS and 0 <= c < _COLS):
            return True
        return self._grid[r][c] == "#"

    def _block_at(self, cell):
        for i, b in enumerate(self._blocks):
            if i != self._carrying and b == cell:
                return i
        return None

    def _move_sprite(self, name, r, c):
        for s in self.current_level.get_sprites_by_name(name):
            s.set_position(c * _CELL, r * _CELL)

    def _sync_avatar(self):
        (r, c) = self._pos
        self._move_sprite("player", r, c)
        for s in self.current_level.get_sprites_by_name("nose"):
            if self._facing is None:
                s.set_visible(False)
            else:
                s.set_visible(True)
                dr, dc = self._facing
                nx = c * _CELL + (_CELL // 2 if dc == 0 else (_CELL - 1 if dc > 0 else 0))
                ny = r * _CELL + (_CELL // 2 if dr == 0 else (_CELL - 1 if dr > 0 else 0))
                s.set_position(nx, ny)

    def step(self) -> None:
        if self._hud:
            self._hud.update(self._budget - self._action_count)
        if self._action_count > self._budget:
            self.lose()
            self.complete_action()
            return
        a = self.action.id.value
        d = _DIRS.get(a)
        if d is not None:
            self._facing = d
            nr, nc = self._pos[0] + d[0], self._pos[1] + d[1]
            if not self._wall(nr, nc) and self._block_at((nr, nc)) is None:
                self._pos = (nr, nc)
            self._sync_avatar()
            self.complete_action()
            return
        if a == 5 and self._facing is not None:
            t = (self._pos[0] + self._facing[0], self._pos[1] + self._facing[1])
            if self._carrying is None:
                b = self._block_at(t)
                if b is not None:
                    self._carrying = b
                    for s in self.current_level.get_sprites_by_name("block_%d" % b):
                        s.set_visible(False)
            else:
                if not self._wall(t[0], t[1]) and self._block_at(t) is None:
                    b = self._carrying
                    self._blocks[b] = t
                    self._carrying = None
                    self._move_sprite("block_%d" % b, t[0], t[1])
                    for s in self.current_level.get_sprites_by_name("block_%d" % b):
                        s.set_visible(True)
                    if all(blk in self._zone for blk in self._blocks):
                        self.next_level()
        self.complete_action()
'''

_MIRROR_BODY = '''

_CELL = PARAMS["cell_px"]
_ROWS = PARAMS["rows"]
_COLS = PARAMS["cols"]
_COL = PARAMS["colors"]
_AXIS = PARAMS["axis"]

_DIRS = {{1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}}


def _cells_sprite(cells, color, name, layer):
    px = np.full((_ROWS * _CELL, _COLS * _CELL), -1, dtype=np.int8)
    for r, c in cells:
        px[r * _CELL : (r + 1) * _CELL, c * _CELL : (c + 1) * _CELL] = color
    return Sprite(pixels=px, name=name, layer=layer, tags=[])


def _cell_sprite(color, name, layer, r, c):
    px = [[color] * _CELL for _ in range(_CELL)]
    return Sprite(pixels=px, name=name, layer=layer, x=c * _CELL, y=r * _CELL, tags=[])


def _cells(grid, chars):
    return [
        (r, c) for r in range(_ROWS) for c in range(_COLS) if grid[r][c] in chars
    ]


def _build_level(idx, spec_level):
    grid = spec_level["map"]
    sprites = [_cells_sprite(_cells(grid, "#"), _COL["wall"], "walls", 0)]
    (ar, ac) = _cells(grid, "A")[0]
    (br, bc) = _cells(grid, "B")[0]
    sprites.append(_cell_sprite(_COL["dot"], "dot_a", 4, ar, ac))
    sprites.append(_cell_sprite(_COL["dot"], "dot_b", 4, br, bc))
    return Level(sprites=sprites, grid_size=(_COLS * _CELL, _ROWS * _CELL), name="L%d" % (idx + 1))


_LEVELS = [_build_level(i, lv) for i, lv in enumerate(PARAMS["levels"])]


class {cls}(ARCBaseGame):
    def __init__(self) -> None:
        self._hud = _Hud(_COL["hud_fill"], _COL["hud_empty"]) if PARAMS["hud"] else None
        camera = Camera(
            background=_COL["bg"],
            letter_box=_COL["letterbox"],
            interfaces=[self._hud] if self._hud else [],
        )
        super().__init__(
            game_id="{gid}",
            levels=_LEVELS,
            camera=camera,
            available_actions=PARAMS["available_actions"],
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        grid = PARAMS["levels"][idx]["map"]
        self._grid = grid
        self._a = _cells(grid, "A")[0]
        self._b = _cells(grid, "B")[0]
        self._budget = PARAMS["budget"][idx]
        if self._hud:
            self._hud.reset(self._budget)

    def _wall(self, r, c):
        if not (0 <= r < _ROWS and 0 <= c < _COLS):
            return True
        return self._grid[r][c] == "#"

    def _move_sprite(self, name, r, c):
        for s in self.current_level.get_sprites_by_name(name):
            s.set_position(c * _CELL, r * _CELL)

    def step(self) -> None:
        if self._hud:
            self._hud.update(self._budget - self._action_count)
        if self._action_count > self._budget:
            self.lose()
            self.complete_action()
            return
        d = _DIRS.get(self.action.id.value)
        if d is None:
            self.complete_action()
            return
        if _AXIS == "x":
            db = (d[0], -d[1])
        else:
            db = (-d[0], d[1])
        ta = (self._a[0] + d[0], self._a[1] + d[1])
        tb = (self._b[0] + db[0], self._b[1] + db[1])
        if not self._wall(*ta):
            self._a = ta
            self._move_sprite("dot_a", *ta)
        if not self._wall(*tb):
            self._b = tb
            self._move_sprite("dot_b", *tb)
        if self._a == self._b:
            self.next_level()
        self.complete_action()
'''

_RULES_BODY = '''

_COL = PARAMS["colors"]
_ALPHA = PARAMS["alphabet"]
_RULES = PARAMS["rules"]
_MASKS = PARAMS["glyph_masks"]

_RULE_X, _RULE_Y0, _RULE_DY = 2, 2, 7
_SEP_X, _RHS_X = 9, 13
_ROW_X0, _ROW_DX = 2, 12
_TGT_Y, _WRK_Y, _MARK_Y = 38, 50, 61


def _glyph_px(name, color, scale):
    m = np.array(_MASKS[name], dtype=np.int8)
    m = np.kron(m, np.ones((scale, scale), dtype=np.int8))
    return np.where(m > 0, np.int8(color), np.int8(-1))


def _sprite(px, name, layer, x, y):
    return Sprite(pixels=px, name=name, layer=layer, x=x, y=y, tags=[])


def _build_level(idx, spec_level):
    sprites = []
    for i, (lhs, rhs) in enumerate(sorted(_RULES.items(), key=lambda kv: _ALPHA.index(kv[0]))):
        y = _RULE_Y0 + i * _RULE_DY
        sprites.append(_sprite(_glyph_px(lhs, _COL["target_glyph"], 1), "rule_lhs_%d" % i, 2, _RULE_X, y))
        sprites.append(_sprite([[_COL["separator"]] * 2], "rule_sep_%d" % i, 2, _SEP_X, y + 2))
        sprites.append(_sprite(_glyph_px(rhs, _COL["working_glyph"], 1), "rule_rhs_%d" % i, 2, _RHS_X, y))
    target = spec_level["target"]
    working = spec_level["working"]
    n = len(target)
    for i in range(n):
        x = _ROW_X0 + i * _ROW_DX
        sprites.append(
            _sprite(_glyph_px(target[i], _COL["target_glyph"], 2), "tgt_%d" % i, 2, x, _TGT_Y)
        )
        for g in _ALPHA:
            s = _sprite(_glyph_px(g, _COL["working_glyph"], 2), "w_%d_%s" % (i, g), 3, x, _WRK_Y)
            s.set_visible(g == working[i])
            sprites.append(s)
        mark = _sprite([[_COL["mark"]] * 10], "mark_%d" % i, 2, x, _MARK_Y)
        mark.set_visible(working[i] == _RULES[target[i]])
        sprites.append(mark)
    ring = np.full((12, 12), -1, dtype=np.int8)
    ring[0, :] = ring[-1, :] = _COL["cursor"]
    ring[:, 0] = ring[:, -1] = _COL["cursor"]
    sprites.append(_sprite(ring, "cursor", 4, _ROW_X0 - 1, _WRK_Y - 1))
    return Level(sprites=sprites, grid_size=(64, 64), name="L%d" % (idx + 1))


_LEVELS = [_build_level(i, lv) for i, lv in enumerate(PARAMS["levels"])]


class {cls}(ARCBaseGame):
    def __init__(self) -> None:
        camera = Camera(background=_COL["bg"], letter_box=_COL["letterbox"], interfaces=[])
        super().__init__(
            game_id="{gid}",
            levels=_LEVELS,
            camera=camera,
            available_actions=PARAMS["available_actions"],
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        self._target = list(PARAMS["levels"][idx]["target"])
        self._working = list(PARAMS["levels"][idx]["working"])
        self._cursor = 0
        self._n = len(self._target)
        self._budget = PARAMS["budget"][idx]

    def _sync(self, i):
        for g in _ALPHA:
            for s in self.current_level.get_sprites_by_name("w_%d_%s" % (i, g)):
                s.set_visible(g == self._working[i])
        for s in self.current_level.get_sprites_by_name("mark_%d" % i):
            s.set_visible(self._working[i] == _RULES[self._target[i]])

    def step(self) -> None:
        if self._action_count > self._budget:
            self.lose()
            self.complete_action()
            return
        a = self.action.id.value
        if a == 3:
            self._cursor = max(0, self._cursor - 1)
        elif a == 4:
            self._cursor = min(self._n - 1, self._cursor + 1)
        elif a in (1, 2):
            i = _ALPHA.index(self._working[self._cursor])
            i = (i + 1) % 7 if a == 1 else (i - 1) % 7
            self._working[self._cursor] = _ALPHA[i]
            self._sync(self._cursor)
        else:
            self.complete_action()
            return
        for s in self.current_level.get_sprites_by_name("cursor"):
            s.set_position(_ROW_X0 - 1 + self._cursor * _ROW_DX, _WRK_Y - 1)
        if all(self._working[j] == _RULES[self._target[j]] for j in range(self._n)):
            self.next_level()
        self.complete_action()
'''

_BODIES_V2 = {
    "replay": _REPLAY_BODY,
    "carry": _CARRY_BODY,
    "mirror": _MIRROR_BODY,
    "rules": _RULES_BODY,
}


def render_game_py_v2(spec: dict[str, Any]) -> str:
    params = {k: v for k, v in spec.items() if k not in ("solution_levels",)}
    body = _BODIES_V2[spec["family"]]
    header = _HEADER.format(
        family=spec["family"],
        game_id=spec["game_id"],
        seed=spec["seed"],
        params=pprint.pformat(params, width=100, sort_dicts=False),
    )
    return header + body.format(cls=class_name_for(spec["game_id"]), gid=spec["game_id"])


def write_env_dir_v2(spec: dict[str, Any], out_root: Path) -> Path:
    """environment_files-style dir; v1 families delegate to the v1 writer."""
    if spec["family"] not in _BODIES_V2:
        return write_env_dir_v1(spec, out_root)
    base, version = spec["game_id"].split("-")
    game_dir = Path(out_root) / base / version
    game_dir.mkdir(parents=True, exist_ok=True)
    (game_dir / f"{base}.py").write_text(render_game_py_v2(spec), encoding="utf-8")
    metadata = {
        "game_id": spec["game_id"],
        "title": base.upper(),
        "default_fps": 8,
        "tags": ["synthgen", spec["family"]],
        "local_dir": str(game_dir),
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
    }
    (game_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (game_dir / "spec.json").write_text(
        json.dumps({k: v for k, v in spec.items() if k != "solution_levels"}, indent=1),
        encoding="utf-8",
    )
    (game_dir / "solution.json").write_text(
        json.dumps(
            {
                "game_id": spec["game_id"],
                "family": spec["family"],
                "per_level": spec["solution_levels"],
                "trace": [a for lvl in spec["solution_levels"] for a in lvl],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return game_dir
