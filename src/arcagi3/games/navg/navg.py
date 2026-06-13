"""Local dev game: arrow-navigation to a goal.

Archetype: move a 1-pixel agent (ACTION1-4) across an empty board to reach the goal
pixel. Each level grows the board and moves the goal, so the agent must re-discover the
path. Tests navigation / shortest-path behaviour. Not used at eval — dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

AGENT_COLOR = 14
GOAL_COLOR = 4
BG = 0

# (board_size, agent_xy, goal_xy)
LEVEL_SPECS = [
    (8, (0, 0), (7, 7)),
    (12, (0, 0), (11, 5)),
    (16, (8, 8), (0, 0)),
    (20, (0, 19), (19, 0)),
    (24, (12, 0), (0, 23)),
]


def _levels():
    return [Level(sprites=[], grid_size=(n, n), name=f"L{i+1}") for i, (n, *_ ) in enumerate(LEVEL_SPECS)]


class Navg(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        camera = Camera(background=BG, letter_box=BG)
        super().__init__(
            game_id="navg",
            levels=_levels(),
            camera=camera,
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVEL_SPECS),
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.current_level_index if hasattr(self, "current_level_index") else 0
        # fall back: find index by matching name
        try:
            idx = int(level.name[1:]) - 1
        except Exception:
            idx = 0
        _, (ax, ay), (gx, gy) = LEVEL_SPECS[idx]
        self._ax, self._ay = ax, ay
        self._gx, self._gy = gx, gy
        self._size = LEVEL_SPECS[idx][0]
        level.remove_all_sprites()
        self._agent = Sprite(pixels=[[AGENT_COLOR]], name="agent", x=ax, y=ay)
        self._goal = Sprite(pixels=[[GOAL_COLOR]], name="goal", x=gx, y=gy, collidable=True)
        level.add_sprite(self._goal)
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
        nx = min(max(self._ax + dx, 0), self._size - 1)
        ny = min(max(self._ay + dy, 0), self._size - 1)
        self._ax, self._ay = nx, ny
        self._agent.set_position(nx, ny)
        if (nx, ny) == (self._gx, self._gy):
            self.next_level()
        self.complete_action()
