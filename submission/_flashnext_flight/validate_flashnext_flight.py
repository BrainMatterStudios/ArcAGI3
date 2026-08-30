#!/usr/bin/env python3
"""validate_flashnext_flight.py — offline structural + syntax validation of the
arc3-flashnext-flight notebook and its kernel metadata. Never touches Kaggle.

Checks:
  1. structural invariants (3-game commit smoke @ 2,400 s, scored branch
     intact with the 9 h time guard, STOCK harness — no graft installs,
     watchdog wired, attestation hard, serve chain pins)
  2. per-cell syntax via compile(); top-level `await` is wrapped in an async
     function first (Kaggle's IPython allows it, plain compile does not)
  3. kernel-metadata.json contract (id, GPU shape, six datasets, internet
     off, competition source, NO model_sources)
  4. freshness: the on-disk notebook matches a fresh in-memory build
  5. optional: pyflakes over the concatenated code cells (undefined names)

Usage:  .venv/bin/python submission/_flashnext_flight/validate_flashnext_flight.py
"""
import ast
import hashlib
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "arc3-flashnext-flight.ipynb"
META_PATH = HERE / "kernel-metadata.json"
BUILD_PATH = HERE / "build_flashnext_flight.py"

SMOKE_GAMES = ("vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99")


def main() -> int:
    nb = json.loads(NB_PATH.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(cells)

    # 1. structural invariants -------------------------------------------------
    graft_ok = True
    for src in cells:
        for line in src.splitlines():
            if "graft_" in line and not ("#" in line and line.index("#") < line.index("graft_")):
                graft_ok = False
                print("FAIL graft reference outside a comment:", line.strip()[:120])

    checks = {
        "3 smoke games, once each": all(joined.count(g) == 1 for g in SMOKE_GAMES),
        "smoke cap 2400": '("smoke3", SMOKE_GAMES, 2400)' in joined,
        "smoke soft end 7200": "timedelta(seconds=7200)" in joined,
        "no 25-game phase": "GAMES_25" not in joined,
        "scored branch: gateway games": "bm.games = _competition_games()" in joined,
        "scored branch: rerun flag": "KAGGLE_IS_COMPETITION_RERUN" in joined,
        "scored branch: gateway wait": "Kaggle gateway did not become ready" in joined,
        "scored branch: dummy parquet kept": "submission.to_parquet(WORKING_DIR" in joined,
        "time guard line": "max(6600.0, (32400 - elapsed - 300) / 4)" in joined,
        "time guard printed": "flight time guard: elapsed=" in joined,
        "time guard after gateway games": (
            "max(6600.0, (32400 - elapsed - 300) / 4)" in joined
            and joined.index("bm.games = _competition_games()")
            < joined.index("max(6600.0, (32400 - elapsed - 300) / 4)")),
        "stock: no graft installs": graft_ok and "TP_ENABLE" not in joined
                                    and "install()" not in joined,
        "27B gone: no model mount": ("foysalemonshanto" not in joined
                                     and "Qwen3_5ForConditionalGeneration" not in joined
                                     and "vrfai" not in joined),
        "27B gone: setup_commands not run": '_run_shell_commands("setup' not in joined,
        "flashnext serve chain": ('"VLLM_PLE_CPU_OFFLOAD": "1"' in joined
                                  and '"--max-num-seqs", "22"' in joined
                                  and '"--max-model-len", "32768"' in joined
                                  and '"--tool-call-parser", "qwen3_xml"' in joined),
        "runtime pins": ("c06a78d59a74ac278dc2278d26dde6c70c48a4e28bb91fd4fbbefff4484e10f3" in joined
                         and "0.1.dev20073+g8e685d198 2.13.0+cu130 5.15.1 13.0" in joined),
        "flashnext attest hard": ("Qwen4ExpForConditionalGeneration" in joined
                                  and "qwen4_exp" in joined and "attest: OK" in joined
                                  and "206" in joined),
        "analyzer wiring": ('"LOCAL_ANALYZER_PROVIDER": "vllm"' in joined
                            and '"LOCAL_ANALYZER_TEMPERATURE": "0.6"' in joined
                            and '"24576", "4096"' in joined),
        "env before deploy artifacts": (joined.index('"LOCAL_ANALYZER_TEMPERATURE": "0.6"')
                                        < joined.index("benchmark_initial")),
        "attest before games": (joined.index("GPU misbind") < joined.index("attest: OK")
                                < joined.index("SMOKE_PHASES = [")),
        "reset-levels law": 'os.environ["ONLY_RESET_LEVELS"] = "true"' in joined,
        "watchdog wired": ("flashnext-watchdog.log" in joined
                           and "_WD_STOP.wait(60)" in joined
                           and "server_alive(timeout=20)" in joined
                           and "start_server(_flags, tag=_tag)" in joined),
        "watchdog stopped at teardown": "_wd.set()" in joined,
        "telemetry gated off scored": ("if not run_as_submission:\n                _fn_phase_begin"
                                       in joined
                                       and "(not run_as_submission) and not getattr(" in joined),
        "teardown shim": "flashnext stop_server" in joined,
        "renames complete": ("flashnext-smoke" not in joined and "flashnext_smoke" not in joined
                             and "arc3-flashnext-flight" in joined),
        "report cell": "FLASHNEXT FLIGHT COMMIT SMOKE" in joined,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    ok = ok and graft_ok

    # 2. per-cell syntax (top-level await is legal in notebooks; wrap it) ------
    for i, src in enumerate(cells):
        try:
            if "await " in src:
                ast.parse("async def _cell():\n"
                          + "".join("    " + line for line in src.splitlines(True)))
            else:
                ast.parse(src)
        except SyntaxError as exc:
            print("FAIL syntax in code cell", i, exc)
            ok = False

    # 3. kernel metadata -------------------------------------------------------
    meta = json.loads(META_PATH.read_text())
    meta_checks = {
        "id": meta.get("id") == "ahmedmobasher86/arc3-flashnext-flight",
        "code file": meta.get("code_file") == "arc3-flashnext-flight.ipynb",
        "gpu shape": (meta.get("machine_shape") == "NvidiaRtxPro6000"
                      and meta.get("enable_gpu") is True),
        "internet off": meta.get("enable_internet") is False,
        "competition source": meta.get("competition_sources") == ["arc-prize-2026-arc-agi-3"],
        "six datasets": (len(meta.get("dataset_sources", [])) == 6
                         and "jcole75/arc3-qwen36-runtime-wheels" in meta["dataset_sources"]
                         and "sonphamorg/arc3-flashnext-serving-part-a-v1" in meta["dataset_sources"]
                         and "sonphamorg/arc3-flashnext-serving-part-b-v1" in meta["dataset_sources"]
                         and "sonphamorg/arc3-flashnext-serving-part-c-v1" in meta["dataset_sources"]
                         and "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1" in meta["dataset_sources"]
                         and "jakobbrggen/taaf-kaggle-source-anim-20260807-anim" in meta["dataset_sources"]),
        "no model sources": meta.get("model_sources") == [],
        "private": meta.get("is_private") is True,
    }
    for name, passed in meta_checks.items():
        print(("OK  " if passed else "FAIL"), "meta:", name)
        ok = ok and passed

    # 4. freshness: rebuild in memory and compare code-cell sha ----------------
    builder = runpy.run_path(str(BUILD_PATH))
    fresh_nb = builder["build_notebook"]()
    fresh = "\n".join("".join(c["source"]) for c in fresh_nb["cells"] if c["cell_type"] == "code")
    sha_disk = hashlib.sha256(joined.encode()).hexdigest()
    sha_fresh = hashlib.sha256(fresh.encode()).hexdigest()
    if sha_disk != sha_fresh:
        print("FAIL freshness: disk", sha_disk[:12], "!= fresh", sha_fresh[:12],
              "— rerun the builder")
        ok = False
    else:
        print("OK   freshness (code cells sha", sha_disk[:12] + ")")

    # 5. pyflakes (optional) ---------------------------------------------------
    try:
        from pyflakes import api as pyflakes_api
        from pyflakes.reporter import Reporter
        import io

        wrapped = []
        for src in cells:
            if "await " in src:
                wrapped.append("async def _cell_%d():\n" % len(wrapped)
                               + "".join("    " + line for line in src.splitlines(True)))
            else:
                wrapped.append(src)
        buf_out, buf_err = io.StringIO(), io.StringIO()
        pyflakes_api.check("\n\n".join(wrapped), "notebook-cells",
                           Reporter(buf_out, buf_err))
        undefined = [ln for ln in buf_out.getvalue().splitlines() if "undefined name" in ln]
        for ln in undefined:
            print("FAIL pyflakes:", ln)
            ok = False
        other = [ln for ln in buf_out.getvalue().splitlines() if "undefined name" not in ln]
        if other:
            print("pyflakes warnings (non-fatal):", len(other))
            for ln in other[:12]:
                print("  ", ln)
        print("OK   pyflakes ran (undefined-name failures:", len(undefined), ")")
    except ImportError:
        print("SKIP pyflakes (not installed)")

    print("VALIDATE", "OK" if ok else "FAIL", "| cells:", len(nb["cells"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
