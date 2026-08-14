"""Engineered agent (DESIGN-2026-08-14) — Stages 1a-2a.

Modules:
    envs        — offline Arcade access helpers (repo-root anchored).
    perception  — learned HUD/housekeeping masking + canonical masked-state.
    battery     — the ≤16-action wiggle probe battery emitting a PerceptionProfile.
    graph       — T0 exact per-level transition graph (Markov escalation, fatal memory).
    effects     — T1/T2 effect rules with prequential accuracy gates (Stage 2a).
    planner     — Dijkstra over observed + gated-rule-predicted edges; commit mode.
    agent       — battery -> graph -> effects -> plan -> act loop.
    evaluate_masks  — Stage-1a milestone: learned-mask vs hand-mask on 25 games.
    evaluate_stage1 — Stage-1b milestones (b)+(c) vs duck banked / human medians.
    evaluate_stage2 — Stage-2a milestone: rule-accuracy table + effects re-run.

No module here imports from scratchpad/ or src/arcagi3/ — the hand-mask
reference is loaded only by the evaluation script, by explicit path, as the
frozen comparison asset.
"""
from engineered.battery import BatteryConfig, BatteryState, PerceptionProfile, ProbeBattery
from engineered.effects import EffectConfig, EffectEngine
from engineered.perception import MaskedState, Perception, learn_hud_lines

__all__ = [
    "BatteryConfig",
    "BatteryState",
    "EffectConfig",
    "EffectEngine",
    "MaskedState",
    "Perception",
    "PerceptionProfile",
    "ProbeBattery",
    "learn_hud_lines",
]
