"""Local dev game: click the correct button.

Archetype: several colored buttons on the board; clicking (ACTION6) the one matching a
hidden rule advances the level. Wrong clicks are harmless (no penalty). The rule: the
correct button is the SMALLEST-color-value button present. Each level adds buttons /
changes positions. Tests object-centric click-target discovery. Dev only.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BG = 0
BOARD = 16

# Each level: list of (color, x, y) buttons. Correct = lowest color value.
LEVEL_SPECS = [
    [(3, 2, 2), (7, 12, 3)],
    [(5, 2, 12), (3, 10, 10), (8, 6, 4)],
    [(9, 1, 1), (4, 14, 1), (6, 7, 8), (2, 3, 13)],
    [(7, 0, 0), (5, 15, 15), (3, 0, 15), (8, 15, 0), (6, 7, 7)],
]


def _levels():
    return [Level(sprites=[], grid_size=(BOARD, BOARD), name=f"L{i+1}") for i in range(len(LEVEL_SPECS))]


class Btnc(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            game_id="btnc",
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
        self._buttons = LEVEL_SPECS[idx]
        self._correct_color = min(c for c, _x, _y in self._buttons)
        level.remove_all_sprites()
        for color, x, y in self._buttons:
            level.add_sprite(Sprite(pixels=[[color]], name=f"btn{color}", x=x, y=y))

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            cx = self.action.data.get("x")
            cy = self.action.data.get("y")
            # render is 64x64; map click back to logical board coords
            scale = 64 // BOARD
            lx = (cx // scale) if cx is not None else -1
            ly = (cy // scale) if cy is not None else -1
            for color, x, y in self._buttons:
                if (x, y) == (lx, ly) and color == self._correct_color:
                    self.next_level()
                    break
        self.complete_action()
