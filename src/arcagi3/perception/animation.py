"""Raw Multi-Frame Animation Exposer.

Extracts intermediate sub-frame sequences from ARC-AGI-3 environment step results
and exposes them as sandbox variables (`last_animation`), allowing LLMs to observe
multi-frame physical dynamics without blowing up standard prompt token budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AnimationSequence:
    """Represents a multi-frame animation sequence resulting from an action."""

    frames: list[np.ndarray] = field(default_factory=list)
    action_id: str | None = None
    step_index: int = 0

    @classmethod
    from_raw_frame(cls, raw_frame_data, action_id: str | None = None, step_index: int = 0) -> AnimationSequence:
        """Parse raw FrameData or multi-subframe array into AnimationSequence."""
        if hasattr(raw_frame_data, "frame"):
            raw_frame_data = raw_frame_data.frame
            
        arr = np.asarray(raw_frame_data, dtype=np.int8)
        if arr.ndim == 2:
            frames = [arr]
        elif arr.ndim == 3:
            frames = [arr[i] for i in range(arr.shape[0])]
        else:
            frames = []
            
        return cls(frames=frames, action_id=action_id, step_index=step_index)

    @property
    def frame_count(self) -> int:
        """Number of intermediate animation sub-frames."""
        return len(self.frames)

    @property
    def is_animated(self) -> bool:
        """True if the action triggered multi-frame movement/animation (> 1 sub-frame)."""
        return len(self.frames) > 1

    def get_diff_summary(self) -> dict[str, int]:
        """Compute pixel change counts across intermediate frames."""
        if not self.is_animated:
            return {"num_subframes": 1, "changed_pixels": 0}

        total_diff = 0
        for i in range(1, len(self.frames)):
            total_diff += int(np.sum(self.frames[i] != self.frames[i - 1]))

        return {
            "num_subframes": len(self.frames),
            "changed_pixels": total_diff,
        }

    def to_sandbox_dict(self) -> dict:
        """Format animation metadata for lightweight sandbox query channel."""
        summary = self.get_diff_summary()
        return {
            "action": self.action_id,
            "step": self.step_index,
            "num_subframes": summary["num_subframes"],
            "is_animated": self.is_animated,
            "total_changed_pixels": summary["changed_pixels"],
        }
