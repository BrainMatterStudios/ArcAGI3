"""Prompt Token Estimator & Context Window Controller.

Rectifies prompt token over-estimation from len(json)//3 to len(json)//4 (~1.08x real prompt tokens)
and expands analyzer context window from 32,768 to 49,152 tokens against the served 65,536 budget.
Prevents premature context eviction while retaining ~21k tokens of generation headroom for Qwen-27B thinking traces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_ESTIMATOR_RATIO = 4.0
DEFAULT_ANALYZER_WINDOW = 49152
SERVED_MAX_MODEL_LEN = 65536
REPLY_RESERVE_TOKENS = 512


@dataclass
class ContextTokenManager:
    """Manages prompt token estimation and history trimming for LLM context windows."""

    chars_per_token: float = DEFAULT_ESTIMATOR_RATIO
    max_context_window: int = DEFAULT_ANALYZER_WINDOW
    reply_reserve: int = REPLY_RESERVE_TOKENS

    def estimate_tokens(self, obj: list | dict | str) -> int:
        """Estimate token count of a JSON object or string using rectified //4 ratio."""
        if isinstance(obj, str):
            text_len = len(obj)
        else:
            text_len = len(json.dumps(obj, separators=(",", ":")))

        estimated = int(text_len // self.chars_per_token)
        return max(1, estimated)

    def trim_history(self, messages: list[dict], system_prompt_tokens: int = 2048) -> list[dict]:
        """Trim oldest assistant/user turn blocks to fit within max_context_window."""
        if not messages:
            return []

        effective_limit = self.max_context_window - system_prompt_tokens - self.reply_reserve
        current_tokens = sum(self.estimate_tokens(msg) for msg in messages)

        if current_tokens <= effective_limit:
            return messages

        logger.info(
            f"Trimming history: estimated tokens ({current_tokens}) exceeds effective ceiling ({effective_limit})."
        )

        trimmed = list(messages)
        while trimmed and sum(self.estimate_tokens(msg) for msg in trimmed) > effective_limit:
            if len(trimmed) > 1:
                trimmed.pop(0)
            else:
                break

        return trimmed
