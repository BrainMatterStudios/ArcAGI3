"""Local dev game: navigation with an independent moving counter (status animation).

Stress test: a counter sprite sweeps the top row every step, independent of the action —
mimicking the step-counter / status-bar animations confirmed present in real ARC-AGI-3
games. This should confuse naive avatar detection (two things move each step). The fix:
the avatar is the object whose motion CORRELATES with the action (distinct deltas per
action), unlike the counter (same motion regardless). Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
AGENT, GOAL, COUNTER = 14, 4, 9
SIZE = 12

# (agent_xy, goal_xy)
LEVEL_SPECS = [
    ((5, 5), (11, 11)),
    ((2, 8), (10, 2)),
    ((8, 10), (1, 3)),
]


def _levels():
    return [Level(sprites=[], grid_size=(SIZE, SIZE), name=f"L{i+1}") for i in range(len(LEVEL_SPECS))]


class Navgc(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="navgc",
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
        (ax, ay), (gx, gy) = LEVEL_SPECS[idx]
        self._ax, self._ay, self._gx, self._gy = ax, ay, gx, gy
        self._t = 0
        level.remove_all_sprites()
        level.add_sprite(Sprite(pixels=[[GOAL]], name="goal", x=gx, y=gy))
        self._counter = Sprite(pixels=[[COUNTER]], name="counter", x=0, y=0)
        level.add_sprite(self._counter)
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
        ny = min(max(self._ay + dy, 1), SIZE - 1)  # keep avatar out of the counter row 0
        self._ax, self._ay = nx, ny
        self._agent.set_position(nx, ny)
        # counter sweeps the top row every step, independent of the action
        self._t += 1
        self._counter.set_position(self._t % SIZE, 0)
        if (nx, ny) == (self._gx, self._gy):
            self.next_level()
        self.complete_action()
