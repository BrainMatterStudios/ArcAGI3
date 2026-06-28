"""Local dev game: navigate a walled maze to the goal.

Stress test for generalization: unlike navg (open board), the avatar must route around
wall obstacles. Greedy coordinate navigation will get stuck at walls and must fall back
to graph exploration / smarter planning. Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
AGENT, GOAL, WALL = 14, 4, 8

# '#'=wall, 'A'=agent start, 'G'=goal, '.'=open
MAZES = [
    [
        "A.#..",
        ".##.#",
        "...#.",
        "#.#..",
        "#...G",
    ],
    [
        "A....#",
        "####.#",
        "...#.#",
        ".#.#..",
        ".#..#G",
        "....#.",
    ],
    [
        "A.....#",
        "#####.#",
        "....#.#",
        ".##.#.#",
        ".#..#.#",
        ".#.##..",
        "...#..G",
    ],
]


def _levels():
    return [Level(sprites=[], grid_size=(len(m), len(m[0])), name=f"L{i+1}")
            for i, m in enumerate(MAZES)]


class Maze(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="maze",
            levels=_levels(),
            camera=Camera(background=BG, letter_box=BG),
            available_actions=[1, 2, 3, 4],
            win_score=len(MAZES),
        )

    def on_set_level(self, level: Level) -> None:
        try:
            idx = int(level.name[1:]) - 1
        except Exception:
            idx = 0
        m = MAZES[idx]
        self._walls = set()
        self._h, self._w = len(m), len(m[0])
        level.remove_all_sprites()
        for y, row in enumerate(m):
            for x, ch in enumerate(row):
                if ch == "#":
                    self._walls.add((x, y))
                    level.add_sprite(Sprite(pixels=[[WALL]], name="wall", x=x, y=y))
                elif ch == "A":
                    self._ax, self._ay = x, y
                elif ch == "G":
                    self._gx, self._gy = x, y
        level.add_sprite(Sprite(pixels=[[GOAL]], name="goal", x=self._gx, y=self._gy))
        self._agent = Sprite(pixels=[[AGENT]], name="agent", x=self._ax, y=self._ay)
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
        if 0 <= nx < self._w and 0 <= ny < self._h and (nx, ny) not in self._walls:
            self._ax, self._ay = nx, ny
            self._agent.set_position(nx, ny)
            if (nx, ny) == (self._gx, self._gy):
                self.next_level()
        self.complete_action()
