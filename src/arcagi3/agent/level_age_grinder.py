"""Level-Age Retriggered State-Grinder & Interaction Narration Synthesizer.

Triggers non-LLM state-space BFS exploration when the LLM spends >= 120 actions on an uncompleted level.
Because actions on uncompleted levels cost 0 points (score = 0 until Level 1 is cleared),
grinding extracts physical interaction mechanics at zero score penalty and synthesizes a concise
natural-language narration for LLM context injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LevelAgeGrinder:
    """Retriggers non-LLM exploration on stalled levels and synthesizes interaction narrations."""

    action_threshold: int = 120
    max_grind_steps: int = 500
    level_actions: dict[int, int] = field(default_factory=dict)
    grinder_engagements: int = 0
    narration_history: list[str] = field(default_factory=list)

    def record_action(self, level_index: int) -> int:
        """Record an action on the given level and return current level age."""
        self.level_actions[level_index] = self.level_actions.get(level_index, 0) + 1
        return self.level_actions[level_index]

    def should_trigger_grinder(self, level_index: int, level_completed: bool) -> bool:
        """Check if current level age exceeds threshold on an uncompleted level."""
        if level_completed:
            return False

        age = self.level_actions.get(level_index, 0)
        if age >= self.action_threshold:
            logger.info(f"Level {level_index} age ({age} actions) >= threshold ({self.action_threshold}). Triggering Grinder!")
            return True

        return False

    def synthesize_narration(self, level_index: int, state_changes: list[dict]) -> str:
        """Synthesize natural-language interaction narration from discovered state transitions."""
        if not state_changes:
            narration = f"Grinder probed Level {level_index}: No significant object movements or state transitions detected."
        else:
            num_transitions = len(state_changes)
            sample = state_changes[0]
            narration = (
                f"Grinder discovered {num_transitions} state transitions on Level {level_index}! "
                f"Key interaction: Action '{sample.get('action')}' shifted object color {sample.get('color')} "
                f"from {sample.get('from')} to {sample.get('to')}."
            )

        self.narration_history.append(narration)
        self.grinder_engagements += 1
        return narration

    def reset_level_age(self, level_index: int) -> None:
        """Reset action counter when a level is completed or reset."""
        if level_index in self.level_actions:
            self.level_actions[level_index] = 0
