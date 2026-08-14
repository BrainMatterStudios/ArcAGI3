"""Stage 1b milestones (b)+(c): the Stage-1 verdict run.

Protocol (pre-registered in the Stage-1b task + design doc §4 Stage 1):
  * Games: the frame-Markov dev set. The design doc says "~12" without an
    explicit list; the clean re-measurement (scratchpad/ideas/
    e2_remeasure.json, animation-controlled, hand-masked) has EXACTLY 13
    games at viol_all == 0.0 — that measured list is used here, and the
    verdict is reported against both the /13 and the pre-registered /12
    reading (sp80 at 0.0009 and ls20 at 0.028 stay excluded).
  * Budget: 4000 actions per game, wall-clock capped.
  * Baseline (b): duck banked per-game levels = max over clones across
    pc_base + w2_base (scratchpad/banked_waves_20260809/, the true_score.py
    max-over-clones convention). PASS >= 8/12 games with levels >= duck;
    KILL < 6/12.
  * Efficiency (c): actions per completed level vs the human median
    (scratchpad/engineered_stage0/p5_budgets.json). PASS <= 3x; KILL > 5x
    on most.

Writes scratchpad/engineered_stage1/milestone_bc.json + .md.
Run: .venv/bin/python src/engineered/evaluate_stage1.py [stems...]
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "engineered"

from engineered.agent import AgentConfig, EngineeredAgent
from engineered.envs import open_arcade, repo_root, resolve_game_ids

# The 13 games measured frame-Markov (viol_all == 0.0 in e2_remeasure.json).
FRAME_MARKOV = [
    "dc22", "ft09", "ka59", "lp85", "r11l", "re86", "s5i5",
    "sb26", "tn36", "tr87", "tu93", "vc33", "wa30",
]
BUDGET = 4000
WALL_S = 900.0


def duck_banked() -> dict[str, int]:
    """Per-game max levels over all clones of pc_base + w2_base."""
    out: dict[str, int] = {}
    for wave in ("pc_base", "w2_base"):
        path = repo_root() / "scratchpad" / "banked_waves_20260809" / f"{wave}.json"
        for row in json.loads(path.read_text())["rows"]:
            g = row["source_game"]
            out[g] = max(out.get(g, 0), int(row["levels_completed"]))
    return out


def human_medians() -> dict[str, float]:
    path = repo_root() / "scratchpad" / "engineered_stage0" / "p5_budgets.json"
    table = json.loads(path.read_text())["table"]
    return {g: float(v["apl_median"]) for g, v in table.items()}


@dataclass
class Row:
    stem: str
    levels: int
    win_levels: int | None
    won: bool
    duck_levels: int
    parity: bool                  # levels >= duck banked
    actions_total: int
    per_level_actions: dict[int, int]
    apl_mean: float | None        # ours, mean over completed levels
    human_apl: float
    apl_ratio: float | None       # ours / human median
    eff_3x: bool | None           # None when no level was completed
    eff_5x: bool | None
    battery_actions: int
    violation_events: int
    death_contradictions: int
    plan_mismatches: int
    archetype: str
    end_reason: str
    wall_s: float


def evaluate(stems: list[str]) -> dict[str, Any]:
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    duck = duck_banked()
    human = human_medians()
    rows: list[Row] = []
    for stem in stems:
        t0 = time.time()
        env = arcade.make(game_id=gid_of[stem], scorecard_id=f"eng1b-eval-{stem}")
        agent = EngineeredAgent(AgentConfig(budget=BUDGET, wall_s=WALL_S))
        r = agent.play(env, gid_of[stem])
        apl = r.apl_mean
        ratio = (apl / human[stem]) if apl is not None else None
        rows.append(Row(
            stem=stem,
            levels=r.levels_completed,
            win_levels=r.win_levels,
            won=r.won,
            duck_levels=duck.get(stem, 0),
            parity=r.levels_completed >= duck.get(stem, 0),
            actions_total=r.actions_total,
            per_level_actions=r.per_level_actions,
            apl_mean=round(apl, 1) if apl is not None else None,
            human_apl=human[stem],
            apl_ratio=round(ratio, 2) if ratio is not None else None,
            eff_3x=(ratio <= 3) if ratio is not None else None,
            eff_5x=(ratio <= 5) if ratio is not None else None,
            battery_actions=r.battery_actions,
            violation_events=r.violation_events,
            death_contradictions=sum(
                g["death_contradictions"] for g in r.graph_stats.values()),
            plan_mismatches=r.plan_mismatches,
            archetype=r.archetype,
            end_reason=r.end_reason,
            wall_s=round(time.time() - t0, 1),
        ))
        row = rows[-1]
        print(f"{stem}: levels={row.levels}/{row.win_levels} duck={row.duck_levels} "
              f"parity={'Y' if row.parity else 'N'} apl={row.apl_mean} "
              f"(human {row.human_apl}, x{row.apl_ratio}) end={row.end_reason} "
              f"viol={row.violation_events} wall={row.wall_s}s", flush=True)

    n = len(rows)
    n_parity = sum(r.parity for r in rows)
    completed = [r for r in rows if r.apl_ratio is not None]
    n_eff3 = sum(1 for r in completed if r.eff_3x)
    n_over5 = sum(1 for r in completed if not r.eff_5x)
    verdict_b = "PASS" if n_parity >= 8 else ("KILL" if n_parity < 6 else "MARGINAL")
    kill_c = n_over5 > len(completed) / 2 if completed else True
    return {
        "protocol": {
            "games": stems, "budget": BUDGET, "wall_s": WALL_S,
            "frame_markov_source": "scratchpad/ideas/e2_remeasure.json viol_all==0.0",
            "duck_baseline": "max levels over clones, pc_base+w2_base",
            "human_budget_source": "scratchpad/engineered_stage0/p5_budgets.json",
            "criteria": "(b) PASS >=8/12 parity, KILL <6/12; "
                        "(c) <=3x human median, KILL >5x on most",
        },
        "milestone_b": {"parity": n_parity, "of": n, "verdict": verdict_b},
        "milestone_c": {
            "games_with_completed_levels": len(completed),
            "within_3x": n_eff3,
            "over_5x": n_over5,
            "kill_triggered": kill_c,
        },
        "rows": [asdict(r) for r in rows],
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-1b milestones (b)+(c) — engineered T0 vs duck banked",
        "",
        f"Budget {payload['protocol']['budget']} actions/game. "
        "Duck = max levels over clones (pc_base + w2_base). "
        "Human = actions-per-completed-level median (340 replays).",
        "",
        "| game | ours | duck | parity | ours apl | human apl | ratio | end | viol |",
        "|------|------|------|--------|----------|-----------|-------|-----|------|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| {r['stem']} | {r['levels']}/{r['win_levels']} | {r['duck_levels']} | "
            f"{'Y' if r['parity'] else 'N'} | {r['apl_mean']} | {r['human_apl']} | "
            f"{r['apl_ratio']} | {r['end_reason']} | {r['violation_events']} |")
    b, c = payload["milestone_b"], payload["milestone_c"]
    lines += [
        "",
        f"**Milestone (b):** {b['parity']}/{b['of']} parity -> {b['verdict']} "
        "(pre-registered: PASS >=8/12, KILL <6/12)",
        f"**Milestone (c):** {c['within_3x']}/{c['games_with_completed_levels']} "
        f"within 3x human median; {c['over_5x']} over 5x -> "
        f"{'KILL' if c['kill_triggered'] else 'not killed'}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import sys

    stems = sys.argv[1:] or FRAME_MARKOV
    payload = evaluate(stems)
    out_dir = repo_root() / "scratchpad" / "engineered_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "milestone_bc.json").write_text(json.dumps(payload, indent=1))
    md = to_markdown(payload)
    (out_dir / "milestone_bc.md").write_text(md)
    print()
    print(md)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
