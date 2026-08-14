"""Engineered agent (DESIGN-2026-08-14) — Stage 1a: perception layer.

Modules:
    envs        — offline Arcade access helpers (repo-root anchored).
    perception  — learned HUD/housekeeping masking + canonical masked-state.
    battery     — the ≤16-action wiggle probe battery emitting a PerceptionProfile.
    evaluate_masks — milestone (a): learned-mask vs hand-mask agreement on 25 games.

No module here imports from scratchpad/ or src/arcagi3/ — the hand-mask
reference is loaded only by the evaluation script, by explicit path, as the
frozen comparison asset.
"""
from engineered.battery import BatteryConfig, BatteryState, PerceptionProfile, ProbeBattery
from engineered.perception import MaskedState, Perception, learn_hud_lines

__all__ = [
    "BatteryConfig",
    "BatteryState",
    "MaskedState",
    "Perception",
    "PerceptionProfile",
    "ProbeBattery",
    "learn_hud_lines",
]
