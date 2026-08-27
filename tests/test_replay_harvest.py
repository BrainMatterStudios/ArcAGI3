"""Unit tests for Post-WIN Clean-Replay Harvest Loop."""

import pytest
from arcagi3.replay_harvest import ReplayHarvestLoop


def test_replay_harvest_trigger_on_win():
    harvest = ReplayHarvestLoop()
    
    assert harvest.should_trigger_reset_harvest(state="RUNNING", level_completed=False, game_won=False) == False
    assert harvest.should_trigger_reset_harvest(state="WIN", level_completed=True, game_won=True) == True


def test_replay_harvest_recording_and_playback():
    harvest = ReplayHarvestLoop()
    
    harvest.record_step(game_id="game_1", level_index=0, action={"type": "CLICK", "x": 10, "y": 20}, reward_delta=1.0)
    harvest.record_step(game_id="game_1", level_index=0, action={"type": "MOVE", "dir": "UP"}, reward_delta=0.0)
    
    assert len(harvest.recorded_win_traces[0]) == 2
    
    harvest.enter_replay_mode(game_id="game_1")
    assert harvest.in_replay_mode == True
    
    a1 = harvest.get_next_replay_action(current_level=0)
    assert a1 == {"type": "CLICK", "x": 10, "y": 20}
    
    a2 = harvest.get_next_replay_action(current_level=0)
    assert a2 == {"type": "MOVE", "dir": "UP"}
    
    a3 = harvest.get_next_replay_action(current_level=0)
    assert a3 is None, "End of recorded trace should return None"
