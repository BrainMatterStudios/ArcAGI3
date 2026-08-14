"""Consolidate the Stage-2a runs into the final milestone artifact.

Inputs (scratchpad/engineered_stage2/):
  milestone_s2a.json            — effects+commit ON (per-level probation)
  milestone_s2a_t0control.json  — same code, effects OFF (the fair T0
                                  control: Stage-1b numbers predate the
                                  determinism + commit-precedence fixes)
Plus scratchpad/engineered_stage1/milestone_bc.json (frozen Stage-1b).

Recomputes the strict gate metric over the stored rule rows (the eval
processes may have run with an older in-memory metric) and writes
milestone_s2a_final.json + milestone_s2a_final.md.
"""
from __future__ import annotations

import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.envs import repo_root
from engineered.evaluate_stage2 import (
    GATE_TRANSITION_CAP,
    _headline_rules,
    rule_meets_gate,
)


def main() -> int:
    out = repo_root() / "scratchpad" / "engineered_stage2"
    s2a = json.loads((out / "milestone_s2a.json").read_text())
    t0 = json.loads((out / "milestone_s2a_t0control.json").read_text())
    t0_rows = {g["stem"]: g for g in t0["games"]}

    for g in s2a["games"]:
        g["milestone_gate_met"] = any(rule_meets_gate(x) for x in g["rules"])
        c = t0_rows.get(g["stem"])
        g["t0_control"] = ({"levels": c["levels"], "apl_mean": c["apl_mean"],
                            "apl_ratio": c["apl_ratio"]} if c else None)
    n_gate = sum(g["milestone_gate_met"] for g in s2a["games"])
    s2a["milestone_s2a"] = {
        "games_with_gate_met": n_gate, "of": len(s2a["games"]),
        "verdict": "PASS" if n_gate >= 8 else "FAIL",
        "gate": f">=90% prequential within <={GATE_TRANSITION_CAP} own "
                "transitions AND accuracy held (still gated or lifetime >=90%)",
    }
    completed = [g for g in s2a["games"] if g["apl_ratio"] is not None]
    t0_completed = [g for g in t0["games"] if g["apl_ratio"] is not None]
    s1 = [g["stage1"] for g in s2a["games"]
          if g["stage1"] and g["stage1"]["apl_ratio"] is not None]
    s2a["efficiency"] = {
        "within_3x_s2a": sum(1 for g in completed if g["apl_ratio"] <= 3),
        "completed_s2a": len(completed),
        "within_3x_t0control": sum(1 for g in t0_completed
                                   if g["apl_ratio"] <= 3),
        "completed_t0control": len(t0_completed),
        "within_3x_stage1b": sum(1 for r in s1 if r["apl_ratio"] <= 3),
        "completed_stage1b": len(s1),
    }

    lines = [
        "# Stage-2a FINAL — T1/T2 accuracy gates + effects/commit re-run",
        "",
        "Protocol: 13 frame-Markov games, budget 4000 actions/game, wall cap "
        "900 s (identical to Stage-1b). Three arms:",
        "  * **s2a** — T0 + T1/T2 effect rules + predicted-edge planning + "
        "commit mode + per-level gate probation.",
        "  * **t0c** — same code, effects OFF (fair control: Stage-1b "
        "numbers predate the determinism/commit fixes).",
        "  * **s1b** — the frozen Stage-1b table.",
        "Held-out = prequential (test-then-train). Gate (strict) = "
        f"{s2a['milestone_s2a']['gate']}.",
        "",
        "## Table 1 — held-out rule accuracy (top rules per game, s2a arm)",
        "",
        "| game | rule family | scope | transitions | held-out acc | "
        "gated_at | meets gate |",
        "|------|-------------|-------|-------------|--------------|"
        "----------|------------|",
    ]
    for g in s2a["games"]:
        rows = _headline_rules(g)
        if not rows:
            lines.append(f"| {g['stem']} | — | — | — | — | — | N |")
            continue
        for i, x in enumerate(rows):
            lines.append(
                f"| {g['stem'] if i == 0 else ''} | {x['family']} | "
                f"{x['scope']} | {x['transitions']} | {x['accuracy']} | "
                f"{x['gated_at']} | {'Y' if rule_meets_gate(x) else 'N'} |")
    m = s2a["milestone_s2a"]
    e = s2a["efficiency"]
    lines += [
        "",
        f"**Milestone S2a (strict): {m['games_with_gate_met']}/{m['of']} "
        f"games -> {m['verdict']}** (pre-registered: PASS >= 8/13).",
        "",
        "## Table 2 — levels + actions-per-completed-level, three arms",
        "",
        "| game | lv s1b | lv t0c | lv s2a | duck | apl s1b | apl t0c | "
        "apl s2a | x-h s1b | x-h t0c | x-h s2a | pred | mispred |",
        "|------|--------|--------|--------|------|---------|---------|"
        "---------|---------|---------|---------|------|---------|",
    ]
    for g in s2a["games"]:
        s1r = g["stage1"] or {}
        c = g["t0_control"] or {}
        lines.append(
            f"| {g['stem']} | {s1r.get('levels', '?')} | "
            f"{c.get('levels', '?')} | {g['levels']} | {g['duck_levels']} | "
            f"{s1r.get('apl_mean')} | {c.get('apl_mean')} | {g['apl_mean']} | "
            f"{s1r.get('apl_ratio')} | {c.get('apl_ratio')} | "
            f"{g['apl_ratio']} | {g['predicted_steps_taken']} | "
            f"{g['rule_mispredictions']} |")
    lines += [
        "",
        f"**Efficiency movement (games <= 3x human median):** "
        f"s1b {e['within_3x_stage1b']}/{e['completed_stage1b']} -> "
        f"t0c {e['within_3x_t0control']}/{e['completed_t0control']} -> "
        f"s2a {e['within_3x_s2a']}/{e['completed_s2a']}.",
        "",
    ]
    (out / "milestone_s2a_final.json").write_text(json.dumps(s2a, indent=1))
    (out / "milestone_s2a_final.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
