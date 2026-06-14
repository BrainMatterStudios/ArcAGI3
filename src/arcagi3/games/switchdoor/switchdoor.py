"""Local dev game: press a switch to open a door, then reach the goal.

Stress test for CAUSAL interaction: the goal is walled off by a door; stepping on the
switch removes the door. The agent must learn that acting in one place (switch) enables
progress elsewhere (path to goal) — a delayed, non-local effect with no intermediate
reward. Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
AGENT, GOAL, WALL, SWITCH, DOOR = 14, 4, 8, 2, 6

# Each level: walls, switch, door (a removable wall), goal, agent start. Hand-designed so
# the goal is only reachable after the door opens.
# layout chars: '#'=wall, 'D'=door, 'S'=switch, 'G'=goal, 'A'=agent, '.'=open
LAYOUTS = [
    [
        "A.S..",
        "####D",
        "....G",
    ],
    [
        "A...S",
        "####D",
        "....G",
        "#####",
        ".....",
    ],
    [
        "A....S",
        ".####D",
        ".#...G",
        ".#.###",
        ".#....",
        ".#####",
    ],
]


def _levels():
    return [Level(sprites=[], grid_size=(len(m), len(m[0])), name=f"L{i+1}")
            for i, m in enumerate(LAYOUTS)]


class Switchdoor(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="switchdoor",
            levels=_levels(),
            camera=Camera(background=BG, letter_box=BG),
            available_actions=[1, 2, 3, 4],
            win_score=len(LAYOUTS),
        )

    def on_set_level(self, level: Level) -> None:
        try:
            idx = int(level.name[1:]) - 1
        except Exception:
            idx = 0
        m = LAYOUTS[idx]
        self._h, self._w = len(m), len(m[0])
        self._walls, self._doors, self._switches = set(), {}, set()
        level.remove_all_sprites()
        for y, row in enumerate(m):
            for x, ch in enumerate(row):
                if ch == "#":
                    self._walls.add((x, y))
                    level.add_sprite(Sprite(pixels=[[WALL]], name="wall", x=x, y=y))
                elif ch == "D":
                    s = Sprite(pixels=[[DOOR]], name="door", x=x, y=y)
                    self._doors[(x, y)] = s
                    level.add_sprite(s)
                elif ch == "S":
                    self._switches.add((x, y))
                    level.add_sprite(Sprite(pixels=[[SWITCH]], name="switch", x=x, y=y))
                elif ch == "A":
                    self._ax, self._ay = x, y
                elif ch == "G":
                    self._gx, self._gy = x, y
        level.add_sprite(Sprite(pixels=[[GOAL]], name="goal", x=self._gx, y=self._gy))
        self._agent = Sprite(pixels=[[AGENT]], name="agent", x=self._ax, y=self._ay)
        level.add_sprite(self._agent)
        self._doors_open = False

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
        blocked = (nx, ny) in self._walls or (
            (nx, ny) in self._doors and not self._doors_open
        )
        if 0 <= nx < self._w and 0 <= ny < self._h and not blocked:
            self._ax, self._ay = nx, ny
            self._agent.set_position(nx, ny)
            if (nx, ny) in self._switches and not self._doors_open:
                self._doors_open = True
                for pos, sprite in self._doors.items():
                    self.current_level.remove_sprite(sprite)
            if (nx, ny) == (self._gx, self._gy):
                self.next_level()
        self.complete_action()
