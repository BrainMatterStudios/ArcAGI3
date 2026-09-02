#!/usr/bin/env python3
"""build_v22_ab.py — single-variable A/B smoke on the 27B V22 serving stack.

Usage: python3 build_v22_ab.py tp9|tp10|aa|ba-tp9

Designs (09-02, instrument control — see docs/EXPERIMENT-2026-09-02-ab-instrument-control.md):
  aa     — STOCK then STOCK (graft installed, flags 0 in BOTH phases; byte-identical
           machinery to the tp9 kernel's stock phase in both positions). Measures the
           instrument's own phase-order drift with nothing else changed.
  ba-tp9 — TP9 then STOCK: the tp9 kernel with its phase order REVERSED. With the
           original stock->tp9 run this completes a crossover: order effect and
           lever effect become separable.
Every phase also snapshots the vLLM /metrics counters (prefix-cache hits/queries,
prompt/generation tokens, requests) so an order effect has a diagnosable cause.

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
_OFF = {f: "0" for f in ALL_FLAGS}
# design -> (slug, graft files to install, ordered phases [(name, env)], title)
DESIGNS = {
    "tp9": ("arc3-v22-ab-tp9", ["graft_pipeline.py"],
            [("stock", dict(_OFF)), ("tp9", dict(_OFF, **ARMS["tp9"]))],
            "single-variable A/B: stock vs TP9"),
    "tp10": ("arc3-v22-ab-tp10", ["graft_memoryspine.py"],
             [("stock", dict(_OFF)), ("tp10", dict(_OFF, **ARMS["tp10"]))],
             "single-variable A/B: stock vs TP10"),
    "aa": ("arc3-v22-aa", ["graft_pipeline.py"],
           [("stock_a", dict(_OFF)), ("stock_b", dict(_OFF))],
           "INSTRUMENT CONTROL A/A: stock vs stock (phase-order drift)"),
    "ba-tp9": ("arc3-v22-ba-tp9", ["graft_pipeline.py"],
               [("tp9", dict(_OFF, **ARMS["tp9"])), ("stock", dict(_OFF))],
               "INSTRUMENT CONTROL B/A: TP9 first, then stock (reversed order)"),
}
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

AB_RUN_CELL = '''# ==== two-phase run: {phase_names} (one boot, same server) ====
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
PHASES = {phases!r}
AB_RESULTS = {{}}
_PHASE_BUDGET_S = 4.0 * 3600
_METRICS_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
_METRICS_URL = _METRICS_URL[:-3] + "/metrics" if _METRICS_URL.endswith("/v1") else _METRICS_URL + "/metrics"
_METRIC_KEYS = ("prefix_cache_hits", "prefix_cache_queries", "prompt_tokens", "generation_tokens",
                "request_success", "num_preemptions", "request_prefill_time_seconds_sum",
                "request_decode_time_seconds_sum", "e2e_request_latency_seconds_sum",
                "e2e_request_latency_seconds_count")


def _vllm_counters() -> dict:
    """Best-effort snapshot of vLLM's Prometheus counters (never raises)."""
    out = {{}}
    try:
        import urllib.request

        with urllib.request.urlopen(_METRICS_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
        for line in text.splitlines():
            if line.startswith("#") or not line.startswith("vllm:"):
                continue
            name, _, val = line.partition(" ")
            base = name.split("{{", 1)[0][len("vllm:"):]
            for key in _METRIC_KEYS:
                if base == key or base == key + "_total":
                    try:
                        out[key] = out.get(key, 0.0) + float(val)
                    except ValueError:
                        pass
    except Exception as _exc:  # noqa: BLE001
        out["error"] = repr(_exc)[:200]
    return out


def _counter_delta(before: dict, after: dict) -> dict:
    d = {{k: round(after[k] - before[k], 3) for k in after if k in before and k != "error"}}
    if d.get("prefix_cache_queries"):
        d["prefix_hit_rate"] = round(d.get("prefix_cache_hits", 0.0) / d["prefix_cache_queries"], 4)
    for src in (before, after):
        if "error" in src:
            d["error"] = src["error"]
    return d


def _graft_counters() -> dict:
    """TP9 telemetry counters (attests the arm fired / stayed off). Never raises."""
    try:
        import sys as _sys

        st = _sys.modules["graft_pipeline"].status()
        return {{k: st.get(k) for k in ("enabled", "resumes_injected",
                                       "perturbations_injected", "retries_used")}}
    except Exception as _exc:  # noqa: BLE001
        return {{"error": repr(_exc)[:200]}}


def _server_pid() -> str:
    try:
        return (WORKING_DIR / "vllm-openai-server.pid").read_text(encoding="utf-8").strip()
    except Exception as _exc:  # noqa: BLE001
        return "unreadable: " + repr(_exc)[:120]

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
        _m_before = _vllm_counters()
        _g_before = _graft_counters()
        _pid_before = _server_pid()
        _t_start = datetime.utcnow().isoformat()
        print(f"=== PHASE {{_phase_name}}: {{len(bm.games)}} games env={{_phase_env}} "
              f"soft_end={{_soft_end}} vllm_before={{_m_before}} graft_before={{_g_before}} "
              f"server_pid={{_pid_before}} ===", flush=True)
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
        _m_after = _vllm_counters()
        _g_after = _graft_counters()
        _graft_delta = {{k: (_g_after.get(k, 0) or 0) - (_g_before.get(k, 0) or 0)
                        for k in ("resumes_injected", "perturbations_injected", "retries_used")
                        if isinstance(_g_after.get(k), int) and isinstance(_g_before.get(k), int)}}
        _graft_delta["enabled_at_start"] = _g_before.get("enabled")
        _graft_delta["enabled_at_end"] = _g_after.get("enabled")
        AB_RESULTS[_phase_name] = {{
            "graft": _graft_delta,
            "server_pid_start": _pid_before,
            "server_pid_end": _server_pid(),
            "games": games,
            "n": len(games),
            "phase_index": len(AB_RESULTS),
            "env": dict(_phase_env),
            "started_utc": _t_start,
            "ended_utc": datetime.utcnow().isoformat(),
            "vllm": _counter_delta(_m_before, _m_after),
            "mean_score": round(sum(g["final_score"] for g in games) / n, 3),
            "mean_levels": round(sum(g["levels_completed"] for g in games) / n, 3),
            "zero_level": sum(1 for g in games if not g["levels_completed"]),
            "total_actions": sum(g["actions"] for g in games),
        }}
        (WORKING_DIR / "ab_results.json").write_text(json.dumps(AB_RESULTS, indent=1))
        print(f"=== PHASE READ {{_phase_name}}: mean_score={{AB_RESULTS[_phase_name]['mean_score']}} "
              f"mean_levels={{AB_RESULTS[_phase_name]['mean_levels']}} "
              f"zero={{AB_RESULTS[_phase_name]['zero_level']}}/{{n}} "
              f"actions={{AB_RESULTS[_phase_name]['total_actions']}} "
              f"vllm={{AB_RESULTS[_phase_name]['vllm']}} graft={{_graft_delta}} "
              f"pid={{_pid_before}}->{{AB_RESULTS[_phase_name]['server_pid_end']}} ===", flush=True)
finally:
    _v31_stop_watchdog()
    for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
        print(f"taaf.kaggle: teardown command: {{command}}", flush=True)
        subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())

print("\\n==== SUMMARY ({phase_names}) ====")
for _p, _r in AB_RESULTS.items():
    print(f"{{_p:8s}} mean_score={{_r['mean_score']}} mean_levels={{_r['mean_levels']}} "
          f"zero={{_r['zero_level']}}/{{_r['n']}} actions={{_r['total_actions']}} vllm={{_r['vllm']}} "
          f"graft={{_r['graft']}} pid={{_r['server_pid_start']}}->{{_r['server_pid_end']}}")
'''


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else ""
    assert arm in DESIGNS, f"usage: build_v22_ab.py {'|'.join(DESIGNS)}"
    slug, arm_files, phases, title = DESIGNS[arm]
    phase_names = " -> ".join(name for name, _ in phases)

    src = json.loads(SRC_NB.read_text())
    cells = src["cells"]
    keep = {2: "WORKING_DIR =", 4: "arc_agi_3_wheels", 6: "_find_bundle_dir", 8: "setup_commands.json",
            10: "benchmark_initial.pkl", 12: "save_request_logs"}
    boot = []
    for idx, marker in keep.items():
        cell_src = "".join(cells[idx]["source"])
        assert marker in cell_src, f"cell {idx} missing marker {marker!r}"
        boot.append(cell_src)

    sources = {name: (GRAFT_DIR / name).read_text() for name in arm_files}
    modules = [n[:-3] for n in arm_files]
    graft_cell = GRAFT_CELL_HEAD.format(flags=ALL_FLAGS, sources=repr(sources),
                                        modules=modules)
    run_cell = AB_RUN_CELL.format(phases=phases, phase_names=phase_names)

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
             "source": f"# {slug} — {title} on the 27B V22 serving stack "
                       f"(one boot, two 25-game phases: {phase_names})"},
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
    # Per-design push directory: `kaggle kernels push -p submission/_v22_ab/push_<design>`
    push_dir = HERE / f"push_{arm}"
    push_dir.mkdir(exist_ok=True)
    (push_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text((HERE / f"kernel-metadata.{arm}.json").read_text())
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    print("built", slug, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
