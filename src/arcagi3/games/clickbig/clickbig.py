"""Local dev game: click the odd-one-out among many objects, at native 64x64.

Stress test for real games (ft09-style): full-resolution board (no upscaling), many
small distractor objects sharing a color, and one unique-colored target. Click the target
(the color that appears exactly once) to advance. Each level adds distractors. Tests
object-centric click discrimination at scale. Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
SIZE = 64
TARGET = 7
DISTRACTOR = 5

# (target_xy, [distractor_xy...]) at native 64x64 coords (top-left of 2x2 sprites)
LEVEL_SPECS = [
    ((30, 30), [(8, 8), (50, 12), (12, 48), (44, 44)]),
    ((10, 40), [(8, 8), (50, 12), (12, 48), (44, 44), (30, 30), (58, 58), (2, 30)]),
    ((52, 6), [(8, 8), (50, 12), (12, 48), (44, 44), (30, 30), (58, 58),
               (2, 30), (24, 6), (6, 24), (40, 20), (20, 58)]),
]


def _levels():
    return [Level(sprites=[], grid_size=(SIZE, SIZE), name=f"L{i+1}") for i in range(len(LEVEL_SPECS))]


class Clickbig(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="clickbig",
            levels=_levels(),
            camera=Camera(background=BG, letter_box=BG),
            available_actions=[6],
            win_score=len(LEVEL_SPECS),
        )

    def on_set_level(self, level: Level) -> None:
        try:
            idx = int(level.name[1:]) - 1
        except Exception:
            idx = 0
        (tx, ty), distractors = LEVEL_SPECS[idx]
        self._tx, self._ty = tx, ty
        level.remove_all_sprites()
        for (dx, dy) in distractors:
            level.add_sprite(Sprite(pixels=[[DISTRACTOR, DISTRACTOR], [DISTRACTOR, DISTRACTOR]], name="d", x=dx, y=dy))
        level.add_sprite(Sprite(pixels=[[TARGET, TARGET], [TARGET, TARGET]], name="t", x=tx, y=ty))

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            cx = self.action.data.get("x")
            cy = self.action.data.get("y")
            if cx is not None and cy is not None:
                # target occupies a 2x2 block at (tx,ty)
                if self._tx <= cx <= self._tx + 1 and self._ty <= cy <= self._ty + 1:
                    self.next_level()
        self.complete_action()
