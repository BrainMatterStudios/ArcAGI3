"""Unit tests for Stale-Close Watchdog Heartbeat."""

import time
import pytest
from arcagi3.watchdog import StaleCloseWatchdog


def test_watchdog_stall_detection():
    watchdog = StaleCloseWatchdog(stall_threshold_seconds=0.1)
    
    assert watchdog.is_stalled() == False
    time.sleep(0.15)
    assert watchdog.is_stalled() == True


def test_watchdog_recovery_payload():
    watchdog = StaleCloseWatchdog(stall_threshold_seconds=0.1)
    time.sleep(0.15)
    
    payload = watchdog.handle_stall_recovery()
    assert payload["action"] == "RESET"
    assert watchdog.stalls_recovered == 1
    assert watchdog.is_stalled() == False
