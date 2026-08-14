"""Stage 2a milestone run: T1/T2 accuracy gates + effects/commit re-run.

Produces the two S2a deliverables (task + design doc §4 Stage 2 milestone a):

  1. Held-out accuracy table — per game, per rule family: transitions
     consumed, prequential (held-out) accuracy, gated in/out, and whether the
     gate was reached within <= 300 own-scope transitions. Milestone verdict:
     >= 8 of the 13 frame-Markov games with a rule at >= 90% within <= 300.
  2. Efficiency re-run — the exact Stage-1b protocol (same games, budget
     4000, wall cap) with the effect model + commit mode ON, side by side
     with the frozen Stage-1b numbers (scratchpad/engineered_stage1/
     milestone_bc.json). The question: how many games move under 3x human
     median actions per completed level.

Writes scratchpad/engineered_stage2/milestone_s2a.json + .md.
Run: .venv/bin/python src/engineered/evaluate_stage2.py [stems...]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.agent import AgentConfig, EngineeredAgent
from engineered.envs import open_arcade, repo_root, resolve_game_ids
from engineered.evaluate_stage1 import (
    BUDGET,
    FRAME_MARKOV,
    WALL_S,
    duck_banked,
    human_medians,
)

GATE_TRANSITION_CAP = 300  # design: gate reached within <= 300 own transitions
OUT_DIR = "engineered_stage2"


def rule_meets_gate(x: dict[str, Any]) -> bool:
    """Strict reading of the design gate: the rule reached >=90% within
    <=300 own transitions AND the accuracy held up — a rule whose window
    touched 90% at transition 22 and then collapsed (tu93 blob_move,
    lifetime 0.28-0.55) did NOT durably model the mechanic and must not
    count. 'Held up' = still gated at end of run, or lifetime held-out
    accuracy >= 90%."""
    if x["gated_at"] is None or x["gated_at"] > GATE_TRANSITION_CAP:
        return False
    return bool(x["gated"]) or (x["accuracy"] is not None
                                and x["accuracy"] >= 0.9)


def evaluate(stems: list[str], effects_on: bool = True) -> dict[str, Any]:
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    duck = duck_banked()
    human = human_medians()
    stage1 = json.loads((repo_root() / "scratchpad" / "engineered_stage1"
                         / "milestone_bc.json").read_text())
    s1_rows = {r["stem"]: r for r in stage1["rows"]}

    games: list[dict[str, Any]] = []
    for stem in stems:
        t0 = time.time()
        env = arcade.make(game_id=gid_of[stem], scorecard_id=f"eng2a-{stem}")
        agent = EngineeredAgent(AgentConfig(budget=BUDGET, wall_s=WALL_S,
                                            effects_enabled=effects_on))
        r = agent.play(env, gid_of[stem])
        apl = r.apl_mean
        ratio = (apl / human[stem]) if apl is not None else None
        rules = r.effect_stats
        gated_in_cap = [x for x in rules if rule_meets_gate(x)]
        games.append({
            "stem": stem,
            "levels": r.levels_completed,
            "win_levels": r.win_levels,
            "won": r.won,
            "duck_levels": duck.get(stem, 0),
            "parity": r.levels_completed >= duck.get(stem, 0),
            "actions_total": r.actions_total,
            "per_level_actions": r.per_level_actions,
            "apl_mean": round(apl, 1) if apl is not None else None,
            "human_apl": human[stem],
            "apl_ratio": round(ratio, 2) if ratio is not None else None,
            "battery_actions": r.battery_actions,
            "plan_mismatches": r.plan_mismatches,
            "rule_mispredictions": r.rule_mispredictions,
            "predicted_steps_taken": r.predicted_steps_taken,
            "violation_events": r.violation_events,
            "archetype": r.archetype,
            "end_reason": r.end_reason,
            "wall_s": round(time.time() - t0, 1),
            "rules": rules,
            "milestone_gate_met": bool(gated_in_cap),
            "stage1": {k: s1_rows[stem][k] for k in
                       ("levels", "apl_mean", "apl_ratio")}
            if stem in s1_rows else None,
        })
        g = games[-1]
        print(f"{stem}: levels={g['levels']}/{g['win_levels']} "
              f"(s1 {g['stage1']['levels'] if g['stage1'] else '?'}) "
              f"apl={g['apl_mean']} x{g['apl_ratio']} "
              f"(s1 x{g['stage1']['apl_ratio'] if g['stage1'] else '?'}) "
              f"gate_met={g['milestone_gate_met']} "
              f"pred_steps={g['predicted_steps_taken']} "
              f"mispred={g['rule_mispredictions']} end={g['end_reason']} "
              f"wall={g['wall_s']}s", flush=True)

    n_gate = sum(g["milestone_gate_met"] for g in games)
    completed = [g for g in games if g["apl_ratio"] is not None]
    n_3x = sum(1 for g in completed if g["apl_ratio"] <= 3)
    s1_completed = [g for g in games if g["stage1"]
                    and g["stage1"]["apl_ratio"] is not None]
    s1_3x = sum(1 for g in s1_completed if g["stage1"]["apl_ratio"] <= 3)
    return {
        "protocol": {
            "games": stems, "budget": BUDGET, "wall_s": WALL_S,
            "gate": f">=90% prequential within <={GATE_TRANSITION_CAP} "
                    "own-scope transitions (>=10 held-out outcomes), and the "
                    "accuracy HELD (still gated at end of run, or lifetime "
                    ">=90%) — once-gated-then-collapsed does not count",
            "milestone": "PASS if >=8/13 games have a rule meeting the gate",
            "stage1_reference": "scratchpad/engineered_stage1/milestone_bc.json",
        },
        "milestone_s2a": {
            "games_with_gate_met": n_gate, "of": len(games),
            "verdict": "PASS" if n_gate >= 8 else "FAIL",
        },
        "efficiency": {
            "completed_games": len(completed),
            "within_3x_now": n_3x,
            "within_3x_stage1": s1_3x,
        },
        "games": games,
    }


def _headline_rules(g: dict[str, Any], max_rows: int = 3) -> list[dict[str, Any]]:
    """The rules worth printing: gated-within-cap first, then best scored."""
    rules = [x for x in g["rules"] if x["scored"]]
    rules.sort(key=lambda x: (
        not rule_meets_gate(x),
        -(x["accuracy"] or 0.0),
        -x["scored"],
    ))
    return rules[:max_rows]


def to_markdown(payload: dict[str, Any]) -> str:
    m = payload["milestone_s2a"]
    e = payload["efficiency"]
    lines = [
        "# Stage-2a milestones — T1/T2 accuracy gates + effects/commit re-run",
        "",
        f"Protocol: budget {payload['protocol']['budget']} actions/game, "
        "13 frame-Markov games, effect model + commit mode ON. "
        "Held-out = prequential (test-then-train). "
        f"Gate = {payload['protocol']['gate']}.",
        "",
        "## Table 1 — held-out rule accuracy (top rules per game)",
        "",
        "| game | rule family | scope | transitions | held-out acc | "
        "gated_at | meets gate | gated now |",
        "|------|-------------|-------|-------------|--------------|"
        "----------|------------|-----------|",
    ]
    for g in payload["games"]:
        rows = _headline_rules(g)
        if not rows:
            lines.append(f"| {g['stem']} | — | — | — | — | — | N | — |")
            continue
        for i, x in enumerate(rows):
            name = g["stem"] if i == 0 else ""
            cap = "Y" if rule_meets_gate(x) else "N"
            lines.append(
                f"| {name} | {x['family']} | {x['scope']} | "
                f"{x['transitions']} | {x['accuracy']} | {x['gated_at']} | "
                f"{cap} | {'Y' if x['gated'] else 'N'} |")
    lines += [
        "",
        f"**Milestone S2a:** {m['games_with_gate_met']}/{m['of']} games with "
        f"a rule at >=90% held-out within <=300 own transitions -> "
        f"**{m['verdict']}** (pre-registered: PASS >=8/13)",
        "",
        "## Table 2 — Stage-1b (T0) vs Stage-2a (T0+T1/T2+commit)",
        "",
        "| game | levels s1 | levels s2a | duck | apl s1 | apl s2a | "
        "x-human s1 | x-human s2a | pred steps | mispred | end |",
        "|------|-----------|------------|------|--------|---------|"
        "------------|-------------|------------|---------|-----|",
    ]
    for g in payload["games"]:
        s1 = g["stage1"] or {}
        lines.append(
            f"| {g['stem']} | {s1.get('levels', '?')} | {g['levels']} | "
            f"{g['duck_levels']} | {s1.get('apl_mean')} | {g['apl_mean']} | "
            f"{s1.get('apl_ratio')} | {g['apl_ratio']} | "
            f"{g['predicted_steps_taken']} | {g['rule_mispredictions']} | "
            f"{g['end_reason']} |")
    lines += [
        "",
        f"**Efficiency movement:** {e['within_3x_now']}/{e['completed_games']} "
        f"games within 3x human median (Stage-1b: {e['within_3x_stage1']}).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    args = sys.argv[1:]
    t0_control = "--t0" in args  # same-code control run with effects OFF
    args = [a for a in args if a != "--t0"]
    stems = args or FRAME_MARKOV
    payload = evaluate(stems, effects_on=not t0_control)
    out_dir = repo_root() / "scratchpad" / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = "milestone_s2a_t0control" if t0_control else "milestone_s2a"
    (out_dir / f"{stem_name}.json").write_text(json.dumps(payload, indent=1))
    md = to_markdown(payload)
    (out_dir / f"{stem_name}.md").write_text(md)
    print()
    print(md)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
