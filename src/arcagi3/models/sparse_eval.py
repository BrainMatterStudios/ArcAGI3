"""Track B: Sparse Model Capability & Memory Persistence Module.

Evaluates high-capacity sparse LLMs (e.g. Qwen3.6-35B-A3B / AgentWorld) with
cross_level_notes persistence across level deaths and transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SparseModelCapabilityEval:
    """Track B evaluator managing sparse model serving and cross-level memory persistence."""

    model_name: str = "Qwen/Qwen3.6-35B-A3B-Instruct"
    max_notes_chars: int = 600
    cross_level_notes: str = ""
    level_history: list[dict] = field(default_factory=list)

    def record_level_transition(self, from_level: int, to_level: int, notes_summary: str) -> str:
        """Promote key learned rules into cross_level_notes across level transitions."""
        prefix = f"[Level {from_level}->{to_level} Notes]: "
        new_note = prefix + notes_summary.strip()
        
        if self.cross_level_notes:
            combined = self.cross_level_notes + "\n" + new_note
        else:
            combined = new_note

        # Cap notes length to prevent prompt inflation
        if len(combined) > self.max_notes_chars:
            combined = combined[-self.max_notes_chars:]

        self.cross_level_notes = combined
        logger.info(f"Updated cross_level_notes ({len(self.cross_level_notes)} chars): {self.cross_level_notes}")
        return self.cross_level_notes

    def handle_death_restart(self, current_level: int, fatal_action: dict) -> str:
        """Preserve cross_level_notes on level death auto-resets while clearing stale plan queues."""
        death_note = f"[Level {current_level} Death Warning]: Action {fatal_action} caused GAME_OVER."
        logger.info(f"Death restart on Level {current_level}. Retaining cross_level_notes: {death_note}")
        return death_note

    def inject_prompt_notes(self, base_prompt: str) -> str:
        """Inject persistent memory notes into model prompt."""
        if not self.cross_level_notes:
            return base_prompt

        return f"{base_prompt}\n\n### PERSISTENT MEMORY & CROSS-LEVEL NOTES:\n{self.cross_level_notes}"
