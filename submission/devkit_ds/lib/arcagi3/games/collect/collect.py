"""Local dev game: collect all items (multi-subgoal navigation).

Stress test: the avatar must touch EVERY item before the level clears — there is no single
goal, and intermediate pickups give no level reward. Tests whether single-target
navigation generalizes to sequential subgoals (navigate to each remaining item in turn).
Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
AGENT, ITEM = 14, 6
SIZE = 14

# (agent_xy, [item_xy...])
LEVEL_SPECS = [
    ((0, 0), [(13, 13), (0, 13), (13, 0)]),
    ((7, 7), [(0, 0), (13, 13), (0, 13), (13, 0)]),
    ((7, 0), [(2, 2), (11, 11), (2, 11), (11, 2), (7, 7)]),
]


def _levels():
    return [Level(sprites=[], grid_size=(SIZE, SIZE), name=f"L{i+1}") for i in range(len(LEVEL_SPECS))]


class Collect(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="collect",
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
        (ax, ay), items = LEVEL_SPECS[idx]
        self._ax, self._ay = ax, ay
        self._items = {tuple(p) for p in items}
        level.remove_all_sprites()
        self._item_sprites = {}
        for (x, y) in self._items:
            s = Sprite(pixels=[[ITEM]], name=f"item_{x}_{y}", x=x, y=y)
            self._item_sprites[(x, y)] = s
            level.add_sprite(s)
        self._agent = Sprite(pixels=[[AGENT]], name="agent", x=ax, y=ay)
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
        nx = min(max(self._ax + dx, 0), SIZE - 1)
        ny = min(max(self._ay + dy, 0), SIZE - 1)
        self._ax, self._ay = nx, ny
        self._agent.set_position(nx, ny)
        if (nx, ny) in self._items:
            self._items.discard((nx, ny))
            self.current_level.remove_sprite(self._item_sprites[(nx, ny)])
        if not self._items:
            self.next_level()
        self.complete_action()
