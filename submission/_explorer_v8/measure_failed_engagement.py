#!/usr/bin/env python3
"""measure_failed_engagement.py — what does a NON-cracking engagement cost?

Measured on ``taaf.competition_arcade.CompetitionArcadeServer`` (the real
``arc_agi.server.create_app(competition_mode=True)`` REST app), because only a
post-WIN reset opens a fresh play: an engagement that does NOT crack leaves its
actions inside the play the LLM keeps using, charged to whatever level was in
progress when they were spent.

Two shapes, because the level matters more than the action count:
  FAIL-ON-L1  the engagement burns N actions without unlocking anything, so
              they land on level 1 (weight 1).
  FAIL-ON-L3  the engagement unlocks levels 1-2 and then burns N actions on
              level 3 (weight 3) before giving up — the case the W-bound does
              NOT protect, because the damaged level's weight is 3, not 1.

In both arms the LLM afterwards completes levels 1-3 with its own minimal
plans, and the score is read from the environment row of the closed scorecard
(max over plays).

Usage: ONLY_RESET_LEVELS=true .venv/bin/python \
           submission/_explorer_v8/measure_failed_engagement.py [N ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(ROOT, "submission/_search_core"),
           os.path.join(ROOT, "submission/_explorer_floor"),
           os.path.join(ROOT, "submission/_duck38_v12_bank"),
           os.path.join(ROOT, "submission/_adopt/taaf-src/src/"
                              "tufa-arc-agi-framework/src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import arcengine  # noqa: E402
import graft_explorer as v7  # noqa: E402
import graft_explorer_v8 as v8  # noqa: E402
import search_core as sc  # noqa: E402
import specialists  # noqa: E402
import taaf.game  # noqa: E402
import taaf.game_api  # noqa: E402
from taaf.competition_arcade import CompetitionArcadeServer  # noqa: E402
from test_competition_scoring import ENV_DIR, GrinderSession, engine_score  # noqa: E402

BASELINES = [43, 12, 23, 28, 65, 37]     # ft09, engine-read
WEIGHTS = sum(range(1, 7))               # 21


def ft09_plans(n_levels: int = 3):
    """Minimal token plans for ft09 levels 1..n, derived offline."""
    import types

    from arc_agi import Arcade, OperationMode

    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    env = client.make(sc.discover_games(ENV_DIR)["ft09"])
    session = GrinderSession(types.SimpleNamespace(
        env=env, game_run=types.SimpleNamespace(game_id="ft09", state="playing")))
    xs = v7._session_state(session)
    genv = v8.GuardedEnv(env, session, xs, t0=time.monotonic(),
                         action_cap=10 ** 6, time_cap_s=1e9, owned_time_cap_s=1e9)
    _, core, _, spec, _ = v8._warm_and_detect(xs, genv, "ft09", sc, specialists)
    plans = []
    for level in range(1, n_levels + 1):
        res = specialists.solve_level(core, spec, level, 60.0)
        assert res.get("solved"), (level, res)
        plans.append(list(res["handle"].path))
        core.backend.adopt(res["handle"])
    return plans


def apply(game, plan):
    for tok in plan:
        if tok[0] == "C":
            action = arcengine.ActionInput(id=arcengine.GameAction.ACTION6,
                                           data={"x": int(tok[1]), "y": int(tok[2])})
        else:
            action = arcengine.ActionInput(
                id=arcengine.GameAction.from_id(int(tok[1])), data={})
        game.execute_action(action)


def burn(game, n):
    """N engine actions the way a failing engagement spends them: raw env
    steps that never reach a win, so no fresh play is ever opened."""
    for _ in range(n):
        game.env.step(arcengine.GameAction.RESET, data={})


def arm(plans, burn_n, burn_after_levels):
    """Play ft09 levels 1-3 with `burn_n` wasted actions injected after
    `burn_after_levels` levels are already done."""
    with CompetitionArcadeServer(game_ids=["ft09"], environments_dir=ENV_DIR) as server:
        gid = server.exposed_game_ids[0]
        game = taaf.game_api.GameAPI(env_name=gid, arcade_spec=server.arcade_spec)
        game.start_game(taaf.game.RunSession(record_intermediate_states=False))
        t0 = time.time()
        for i, plan in enumerate(plans):
            if i == burn_after_levels and burn_n:
                burn(game, burn_n)
            apply(game, plan)
        if burn_after_levels >= len(plans) and burn_n:
            burn(game, burn_n)
        env, _ = engine_score(game)
        return {
            "burn_n": burn_n, "burn_after_levels": burn_after_levels,
            "score": round(env.score, 4), "levels": env.levels_completed,
            "actions": env.actions, "plays": len(env.runs),
            "wall_s": round(time.time() - t0, 1),
        }


def model(burn_n, burn_level_idx, plan_lens, levels_completed=3):
    """Closed form: the harness/engine formula with `burn_n` charged to the
    level at `burn_level_idx` (0-based). Used to extrapolate past what is
    affordable to run over HTTP — validated against the measured points."""
    total, max_w = 0.0, 0
    for idx in range(6):
        weight = idx + 1
        acts = (plan_lens[idx] if idx < len(plan_lens) else 0) + \
               (burn_n if idx == burn_level_idx else 0)
        score = (min(115.0, (BASELINES[idx] / acts) ** 2 * 100)
                 if (idx < levels_completed and acts) else 0.0)
        if score > 0:
            max_w += weight
        total += score * weight
    return round(min(total / WEIGHTS, max_w / WEIGHTS * 100), 4)


def main() -> None:
    ns = [int(a) for a in sys.argv[1:]] or [2000, 10000]
    plans = ft09_plans(3)
    plan_lens = [len(p) for p in plans]
    print(f"ft09 minimal plans: {plan_lens} (baselines {BASELINES[:3]})")

    rows = [arm(plans, 0, 0)]
    print(f"CONTROL: score={rows[0]['score']} levels={rows[0]['levels']} "
          f"actions={rows[0]['actions']} plays={rows[0]['plays']}")
    control = rows[0]["score"]

    for n in ns:
        for after, label in ((0, "FAIL-ON-L1"), (2, "FAIL-ON-L3")):
            r = arm(plans, n, after)
            r["label"] = label
            r["delta"] = round(r["score"] - control, 4)
            r["model"] = model(n, after, plan_lens)
            rows.append(r)
            print(f"{label} N={n}: score={r['score']} (delta {r['delta']:+.4f}) "
                  f"levels={r['levels']} plays={r['plays']} "
                  f"wall={r['wall_s']}s | closed form {r['model']}")

    print("\nEXTRAPOLATION (closed form, validated above):")
    for n in (2000, 10000, 50000, 292500):
        l1 = model(n, 0, plan_lens)
        l3 = model(n, 2, plan_lens)
        print(f"  N={n:>7}: FAIL-ON-L1 {l1:>7} (delta {l1 - control:+.2f})   "
              f"FAIL-ON-L3 {l3:>7} (delta {l3 - control:+.2f})")

    out = {"control": control, "plan_lens": plan_lens, "rows": rows,
           "extrapolation": {n: {"fail_l1": model(n, 0, plan_lens),
                                 "fail_l3": model(n, 2, plan_lens)}
                             for n in (2000, 10000, 50000, 292500)}}
    dest = Path(ROOT) / "submission/_v8_smoke/results/failed_engagement_cost.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
