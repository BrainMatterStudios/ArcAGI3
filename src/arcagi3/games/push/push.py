"""Local dev game: push a block onto a target (sokoban-lite).

Archetype: move the agent (ACTION1-4); pushing into the block shifts the block one cell
if the destination is in-bounds. Get the block onto the target cell to advance. Tests
multi-step planning with object interaction. Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
AGENT, BLOCK, TARGET = 14, 6, 4

# (board, agent_xy, block_xy, target_xy)
LEVEL_SPECS = [
    (8, (1, 4), (3, 4), (6, 4)),
    (10, (1, 1), (4, 4), (8, 8)),
    (12, (6, 1), (6, 5), (6, 10)),
]


def _levels():
    return [Level(sprites=[], grid_size=(n, n), name=f"L{i+1}") for i, (n, *_ ) in enumerate(LEVEL_SPECS)]


class Push(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="push",
            levels=_levels(),
            camera=Camera(background=BG, letter_box=BG),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVEL_SPECS),
        )

    def on_set_level(self, level: Level) -> None:
        try:
            idx = int(level.name[1:]) - 1
        except Exception:
            idx = 0
        self._size, (ax, ay), (bx, by), (tx, ty) = LEVEL_SPECS[idx]
        self._ax, self._ay, self._bx, self._by = ax, ay, bx, by
        self._tx, self._ty = tx, ty
        level.remove_all_sprites()
        self._target = Sprite(pixels=[[TARGET]], name="target", x=tx, y=ty, collidable=False)
        self._block = Sprite(pixels=[[BLOCK]], name="block", x=bx, y=by)
        self._agent = Sprite(pixels=[[AGENT]], name="agent", x=ax, y=ay)
        level.add_sprite(self._target)
        level.add_sprite(self._block)
        level.add_sprite(self._agent)

    def step(self) -> None:
        dx = dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1
        nx, ny = self._ax + dx, self._ay + dy
        if not (0 <= nx < self._size and 0 <= ny < self._size):
            self.complete_action()
            return
        if (nx, ny) == (self._bx, self._by):
            # try to push block
            pbx, pby = self._bx + dx, self._by + dy
            if 0 <= pbx < self._size and 0 <= pby < self._size:
                self._bx, self._by = pbx, pby
                self._ax, self._ay = nx, ny
            # else blocked: no move
        else:
            self._ax, self._ay = nx, ny
        self._agent.set_position(self._ax, self._ay)
        self._block.set_position(self._bx, self._by)
        if (self._bx, self._by) == (self._tx, self._ty):
            self.next_level()
        self.complete_action()
