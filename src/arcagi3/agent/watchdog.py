"""Stale-Close Watchdog Heartbeat.

Monitors agent activity and issues a server-side `RESET` heartbeat if inactive for > 600s (10 min).
Races Kaggle gateway's force-close threshold (DEFAULT_STALE_MINUTES = 15) to prevent silent score zeroing
during vLLM or model generation freezes.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_WATCHDOG_STALL_SECONDS = 600.0  # 10 minutes (heartbeat threshold)


@dataclass
class StaleCloseWatchdog:
    """Watchdog timer monitoring per-game stall duration and issuing RESET heartbeats."""

    stall_threshold_seconds: float = DEFAULT_WATCHDOG_STALL_SECONDS
    last_action_timestamp: float = 0.0
    stalls_recovered: int = 0

    def __post_init__(self) -> None:
        self.reset_timer()

    def reset_timer(self) -> None:
        """Reset activity timer on every successful action emission or state change."""
        self.last_action_timestamp = time.time()

    def get_elapsed_seconds(self) -> float:
        """Return elapsed seconds since last recorded action activity."""
        return time.time() - self.last_action_timestamp

    def is_stalled(self) -> bool:
        """Check if elapsed time exceeds the 600s stall threshold."""
        return self.get_elapsed_seconds() >= self.stall_threshold_seconds

    def handle_stall_recovery(self) -> dict:
        """Synthesize RESET heartbeat payload to recover from stall and ping server."""
        self.stalls_recovered += 1
        elapsed = self.get_elapsed_seconds()
        logger.warning(
            f"StaleCloseWatchdog: Stall detected ({elapsed:.1f}s >= {self.stall_threshold_seconds}s). "
            f"Issuing RESET heartbeat (recovery #{self.stalls_recovered})."
        )
        self.reset_timer()
        return {"action": "RESET", "reason": "watchdog_heartbeat_stall_recovery"}
