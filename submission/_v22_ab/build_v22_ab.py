#!/usr/bin/env python3
"""build_v22_ab.py — single-variable A/B smoke on the 27B V22 serving stack.

Usage: python3 build_v22_ab.py tp9|tp10

Two phases in one GPU session, one boot: STOCK duck (all graft flags 0), then
the ARM (exactly one graft enabled). Base = the byte-copy of the public 2.66
notebook (proven boot: svid 346312727 COMPLETE, V22-fallback serving args
confirmed in the server log), cells 2/6/8/10/12 verbatim; we replace only the
run cell with a two-phase loop and insert one graft-install cell.

Arms (pre-registered, one variable each):
  tp9  — TP9_ENABLE=1 (turn-pipeline repair: resume, livelock breaker, retry).
         Read: acting-turn share up, livelock games (r11l class) progress past
         their stall; levels/zero-level as the outcome.
  tp10 — TP10_ENABLE=1 (memory spine: journal, intent echo, honest accounting).
         Read: retry-replay collapse (sp80 class), zero-level count down.

READING: mechanism metrics first (per-game levels, zero-level count, actions
per level), local mean second; single 25-vs-25 draw, CV ~0.17-0.20 applies.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
SRC_NB = SUB / "_v31_copy" / "arc3-v31-copy.ipynb"
GRAFT_DIR = SUB / "_throughput_v1"
GRAFT_FILES = [
    "graft_throughput.py", "graft_control.py", "frontier_explorer.py",
    "graft_explore.py", "graft_emission.py", "graft_economy.py",
    "graft_deaths.py", "graft_durable.py", "graft_pipeline.py",
    "graft_memoryspine.py",
]
ALL_FLAGS = ["TP_ENABLE", "TP2_ENABLE", "TP4_ENABLE", "TP5_ENABLE", "TP6_ENABLE",
             "TP7_ENABLE", "TP8_ENABLE", "TP9_ENABLE", "TP10_ENABLE"]
ARMS = {"tp9": {"TP9_ENABLE": "1"}, "tp10": {"TP10_ENABLE": "1"}}
# v3: install ONLY the arm's graft — the keithtyser bundle is pure June stock and
# several other grafts wrap anim-fork-only seams (v2 died on graft_control's
# _aggregate_action_batch_result). Both arm grafts verified to install on june_stock.
ARM_FILES = {"tp9": ["graft_pipeline.py"], "tp10": ["graft_memoryspine.py"]}

GRAFT_CELL_HEAD = '''# ==== graft install (A/B machinery; every flag phase-controlled) ====
# All grafts install ONCE; behaviour is env-gated per phase. Defaults ALL OFF —
# the stock phase must be byte-equivalent stock behaviour.
import importlib as _il

for _flag in {flags!r}:
    os.environ[_flag] = "0"

_G_DIR = WORKING_DIR / "graft_bundle"
_G_DIR.mkdir(parents=True, exist_ok=True)
_GRAFT_SOURCES = {sources}
for _name, _src in _GRAFT_SOURCES.items():
    (_G_DIR / _name).write_text(_src, encoding="utf-8")
if str(_G_DIR) not in sys.path:
    sys.path.insert(0, str(_G_DIR))
_installed = {{}}
for _mod_name in {modules!r}:
    _m = _il.import_module(_mod_name)
    _st = _m.install()
    _installed[_mod_name] = _st
    assert _st.endswith(": OK"), _st
print("[ab] grafts installed:", _installed, flush=True)
'''

AB_RUN_CELL = '''# ==== two-phase A/B run: stock then {arm} (one boot, same server) ====
def _offline_games(env_dir: str):
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {{env_dir}}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


print((BUNDLE_DIR / "preamble.txt").read_text())
(WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))
assert not TRUE_SUBMISSION, "A/B smoke kernel must never run as a submission"

_ENV_DIR = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
PHASES = [
    ("stock", {{f: "0" for f in {flags!r}}}),
    ("{arm}", dict({{f: "0" for f in {flags!r}}}, **{arm_env!r})),
]
AB_RESULTS = {{}}
_PHASE_BUDGET_S = 4.0 * 3600

_v31_start_watchdog(bm.solver)
try:
    for _phase_name, _phase_env in PHASES:
        for _k, _v in _phase_env.items():
            os.environ[_k] = _v
        bm.game_runs = []
        bm.games = _offline_games(_ENV_DIR)
        bm.n_passes = 1
        bm.game_weights = None
        bm.label = "v22-ab-" + _phase_name
        _soft_end = datetime.now() + timedelta(seconds=_PHASE_BUDGET_S)
        print(f"=== PHASE {{_phase_name}}: {{len(bm.games)}} games env={{_phase_env}} "
              f"soft_end={{_soft_end}} ===", flush=True)
        try:
            await bm.run(soft_end_time=_soft_end, runtime_environment=target,
                         minimal_diagnostics=False)
        except Exception as _exc:  # noqa: BLE001
            import traceback

            print(f"PHASE {{_phase_name}} RAISED {{type(_exc).__name__}}: {{_exc}}", flush=True)
            traceback.print_exc()
        games = []
        for game_run in list(bm.game_runs):
            apl = list(game_run.actions_per_level or [])
            games.append({{
                "game_id": game_run.game_id,
                "state": str(game_run.state),
                "levels_completed": game_run.levels_completed,
                "number_of_levels": game_run.number_of_levels,
                "final_score": game_run.final_score,
                "actions": sum(apl) if apl else len(game_run.history),
                "actions_per_level": apl,
                "wallclock_s": game_run.final_wallclock_seconds,
            }})
        n = max(1, len(games))
        AB_RESULTS[_phase_name] = {{
            "games": games,
            "n": len(games),
            "mean_score": round(sum(g["final_score"] for g in games) / n, 3),
            "mean_levels": round(sum(g["levels_completed"] for g in games) / n, 3),
            "zero_level": sum(1 for g in games if not g["levels_completed"]),
            "total_actions": sum(g["actions"] for g in games),
        }}
        (WORKING_DIR / "ab_results.json").write_text(json.dumps(AB_RESULTS, indent=1))
        print(f"=== PHASE READ {{_phase_name}}: mean_score={{AB_RESULTS[_phase_name]['mean_score']}} "
              f"mean_levels={{AB_RESULTS[_phase_name]['mean_levels']}} "
              f"zero={{AB_RESULTS[_phase_name]['zero_level']}}/{{n}} "
              f"actions={{AB_RESULTS[_phase_name]['total_actions']}} ===", flush=True)
finally:
    _v31_stop_watchdog()
    for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
        print(f"taaf.kaggle: teardown command: {{command}}", flush=True)
        subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())

print("\\n==== A/B SUMMARY ({arm} vs stock) ====")
for _p, _r in AB_RESULTS.items():
    print(f"{{_p:8s}} mean_score={{_r['mean_score']}} mean_levels={{_r['mean_levels']}} "
          f"zero={{_r['zero_level']}}/{{_r['n']}} actions={{_r['total_actions']}}")
'''


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else ""
    assert arm in ARMS, f"usage: build_v22_ab.py {'|'.join(ARMS)}"
    slug = f"arc3-v22-ab-{arm}"

    src = json.loads(SRC_NB.read_text())
    cells = src["cells"]
    keep = {2: "WORKING_DIR =", 4: "arc_agi_3_wheels", 6: "_find_bundle_dir", 8: "setup_commands.json",
            10: "benchmark_initial.pkl", 12: "save_request_logs"}
    boot = []
    for idx, marker in keep.items():
        cell_src = "".join(cells[idx]["source"])
        assert marker in cell_src, f"cell {idx} missing marker {marker!r}"
        boot.append(cell_src)

    sources = {name: (GRAFT_DIR / name).read_text() for name in ARM_FILES[arm]}
    modules = [n[:-3] for n in ARM_FILES[arm]]
    graft_cell = GRAFT_CELL_HEAD.format(flags=ALL_FLAGS, sources=repr(sources),
                                        modules=modules)
    run_cell = AB_RUN_CELL.format(arm=arm, arm_env=ARMS[arm], flags=ALL_FLAGS)

    def code_cell(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": source}

    nb = {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": f"# {slug} — single-variable A/B: stock vs {arm.upper()} "
                       "on the 27B V22 serving stack (one boot, two 25-game phases)"},
            *[code_cell(s) for s in boot],
            code_cell(graft_cell),
            code_cell(run_cell),
        ],
    }
    (HERE / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    (HERE / f"kernel-metadata.{arm}.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                            "keithtyser/taaf-duck-qwen38-serving-v1"],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                        "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000",
    }, indent=1))
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    print("built", slug, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
