#!/usr/bin/env python3
"""build_flashnext_flight.py — THE FLIGHT KERNEL for the Flash-Next model swap
(2026-08-30). Competition-submittable.

Derived from submission/_flashnext_smoke/build_flashnext_smoke.py (v4 — its
kernel PASSED the 25-game stock read on Kaggle: levels/game 1.16, zero-level
4/25, boot 950 s on rung gcp_exact; results_v4/). This builder runs the smoke
build via runpy — so gate ASSEMBLE, BOOT ladder, duck-env exports (incl. the
BUNDLE_DIR re-pin + sonpham path pruning) and the hard Flash-Next attestation
are inherited VERBATIM — then applies the flight deltas:

  1. SMOKE HOOK -> the v12-style small commit smoke: 3 games
     [vc33, sb26, tn36], per-game cap 2,400 s, soft_end =
     NOTEBOOK_START_EPOCH + 7,200 s (3,600 s smoke + boot allowance).
     The scored-rerun branch is UNTOUCHED (gateway games, n_passes 1).
  2. 9 h TIME GUARD on the scored path only: after boot, at scored-rerun
     time, elapsed = time.time() - NOTEBOOK_START_EPOCH and
     bm.solver.max_runtime_s_per_game = max(6600.0, (32400 - elapsed - 300) / 4)
     — 4 waves at concurrency 28 over ~110 games. Same arithmetic as
     graft_throughput.time_guard_per_game_s, inlined: this kernel installs
     NO grafts (STOCK harness).
  3. SUBMISSION OUTPUT preserved: the v12 run-cell's dummy-parquet branch
     (non-submission) and the true-submission gateway path are kept; the
     smoke phase telemetry stays for the commit path but is gated OFF the
     scored branch (phase begin/end + play seam behind run_as_submission).
  4. SERVER WATCHDOG: background thread polls GET {base}/v1/models every
     60 s; on 3 consecutive failures relaunches the server with the SAME
     argv/env via start_server (their serving_env(); ARCH_OVERRIDE still
     holds the booted rung). Logs /kaggle/working/flashnext-watchdog.log.
     The teardown shim sets the watchdog stop event before stop_server so a
     clean shutdown is never "rescued".
  5. Attestation stays hard (Qwen4ExpForConditionalGeneration / qwen4_exp /
     206 shards > 186 GB) — a scored run that silently served nothing dies
     before spending the slot.

Usage:
  .venv/bin/python submission/_flashnext_flight/build_flashnext_flight.py
  .venv/bin/python submission/_flashnext_flight/validate_flashnext_flight.py
  # push (ONLY on Ahmed's go — a submittable kernel):
  cd submission/_flashnext_flight && python3 -m kaggle kernels push -p .
"""
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
SMOKE_BUILD = SUB / "_flashnext_smoke" / "build_flashnext_smoke.py"
KERNEL_SLUG = "arc3-flashnext-flight"

SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]
SMOKE_PER_GAME_S = 2400
SMOKE_SOFT_END_S = 7200      # NOTEBOOK_START_EPOCH + 7,200 s (smoke 3,600 + boot allowance)

# ---- markers into the BUILT smoke notebook ---------------------------------
MARK_SMOKE = "# Smoke/eval hook:"
MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"
MARK_REPORT = "FLASHNEXT SMOKE RESULTS"

MD_HEADER = """\
# arc3-flashnext-flight — FLIGHT: STOCK duck × Flash-Next NVFP4 (submittable)

The competition-submittable kernel for the model-swap arm. STOCK duck harness
(anim-20260807 bundle, the bytes that flew 1.55) served by sonpham's
**Qwen3.8-Flash-Next NVFP4** on ONE RTX Pro 6000 — no grafts, no 27B mount.
Assemble/boot/attestation inherited verbatim from arc3-flashnext-smoke, whose
v4 PASSED the 25-game stock read on Kaggle (levels 1.16, zero-level 4/25).

* **Commit path**: v12-style 3-game smoke (vc33 / sb26 / tn36), per-game cap
  2,400 s, soft end at NOTEBOOK_START_EPOCH + 7,200 s.
* **Scored rerun** (`KAGGLE_IS_COMPETITION_RERUN`): full gateway games,
  n_passes 1, per-game cap from the 9 h time guard —
  `max(6600, (32400 - elapsed_after_boot - 300) / 4)` (4 waves @ conc 28).
* **Watchdog**: /v1/models polled every 60 s; 3 consecutive failures →
  relaunch with the same argv/env (their vLLM-dev has none of its own).
* Attestation is HARD: Qwen4ExpForConditionalGeneration / qwen4_exp /
  206 shards > 186 GB, live decode through the analyzer endpoint — a run that
  silently served nothing dies before spending the slot.
"""

CELL_SMOKE = r'''# Smoke/eval hook: v12-style SMALL COMMIT SMOKE. A NORMAL COMMIT plays 3
# public games against the Flash-Next server (proves assemble + boot +
# attestation + the agent loop on the scored GPU class) — per-game cap
# 2,400 s, soft_end = NOTEBOOK_START_EPOCH + 7,200 s (3,600 s of smoke +
# boot allowance; assemble+boot measured ~740-1,550 s on the gate/smoke
# Kaggle runs, plus the tarball extract).
# The scored rerun (KAGGLE_IS_COMPETITION_RERUN) never enters this branch —
# it plays the FULL competition games from the gateway, n_passes 1, with the
# 9 h time guard applied in the run cell.
SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]
SMOKE_PHASES = [
    ("smoke3", SMOKE_GAMES, 2400),
]
FN_PHASE_ERRORS = []
FN_ALL_RUNS = []

if not run_as_submission:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    def _resolve_env_dir():
        candidates = [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
        ]
        for cand in candidates:
            if cand.is_dir():
                return str(cand)
        for hit in Path("/kaggle/input").rglob("environment_files"):
            if hit.is_dir():
                return str(hit)
        raise RuntimeError("environment_files dir not found in /kaggle/input")

    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_PHASES[0][1]]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "flashnext-flight-" + SMOKE_PHASES[0][0]
    bm.solver.max_runtime_s_per_game = float(SMOKE_PHASES[0][2])
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=7200)
    print(f"smoke hook: {len(bm.games)} games env_dir={_env_dir} "
          f"concurrency={bm.solver.concurrency} "
          f"per_game_cap={bm.solver.max_runtime_s_per_game}s soft_end={soft_end}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''

CELL_WATCHDOG = r'''# ---- server watchdog: the scored run is ~9 h and their vLLM-dev has no
# watchdog of its own. A background thread polls GET {base}/v1/models every
# 60 s; on 3 CONSECUTIVE failures it relaunches the server with the SAME
# argv/env (start_server reuses their serving_env(); the module-level
# ARCH_OVERRIDE still holds the booted rung's value). Events append to
# /kaggle/working/flashnext-watchdog.log. The teardown shim sets _WD_STOP
# before stop_server, so a clean shutdown is never "rescued".
import threading as _wd_threading

_WD_LOG = WORKING_DIR / "flashnext-watchdog.log"
_WD_STOP = _wd_threading.Event()
_WD_STATE = {"restarts": 0}


def _wd_note(msg):
    try:
        with _WD_LOG.open("a", encoding="utf-8") as _handle:
            _handle.write(datetime.utcnow().isoformat() + "Z " + msg + "\n")
    except Exception:  # noqa: BLE001
        pass
    print("flashnext-flight: watchdog:", msg, flush=True)


def _wd_loop():
    failures = 0
    while not _WD_STOP.wait(60):
        if server_alive(timeout=20):
            failures = 0
            continue
        failures += 1
        _proc = CURRENT_SERVER.get("proc")
        _rc = _proc.poll() if _proc is not None else None
        _wd_note(f"health check FAILED ({failures}/3 consecutive; server proc rc={_rc})")
        if failures < 3:
            continue
        failures = 0
        _flags = list(CURRENT_SERVER.get("flags") or [])
        if not _flags:
            _wd_note("no recorded server argv — cannot relaunch")
            continue
        _WD_STATE["restarts"] += 1
        _tag = f"{BOOT_TAG}-wd{_WD_STATE['restarts']}"
        _wd_note(f"3 consecutive failures — relaunching [{_tag}] with the same argv/env")
        try:
            stop_server(f"watchdog relaunch #{_WD_STATE['restarts']}")
            start_server(_flags, tag=_tag)
            _wd_note(f"relaunch [{_tag}] OK — server answering again")
        except Exception as _exc:  # noqa: BLE001
            _wd_note(("relaunch [" + _tag + "] FAILED: " + repr(_exc))[:500])


if BOOT_TAG is not None:
    _wd_thread = _wd_threading.Thread(target=_wd_loop, daemon=True, name="flashnext-watchdog")
    _wd_thread.start()
    _wd_note(f"armed: poll {VLLM_API}/models every 60 s; relaunch after 3 consecutive failures")
'''

CELL_REPORT = r'''# ---- flashnext flight report (commit-smoke path only; grep FLASHNEXT FLIGHT) ----
if run_as_submission:
    print("scored rerun: flight report inert — the gateway scorecard is the record")
else:
    print("=" * 78)
    print("FLASHNEXT FLIGHT COMMIT SMOKE — 3 games, per-game cap 2400 s")
    print("boot:", RESULTS["verdicts"].get("boot"))
    by = {p["phase"]: p for p in FN_PHASES}
    p = by.get("smoke3")
    verdict = "SMOKE-FAIL"
    if not p or not p["n_games"]:
        print("PHASE smoke3: MISSING")
    else:
        print(f"PHASE smoke3: games={p['n_games']} mean_actions={p['mean_actions']} "
              f"mean_levels={p['mean_levels']} mean_score={p['mean_score']} "
              f"zero_level={p['zero_level_games']} mean_turns={p['mean_turns']} "
              f"gen_tok_s={p['gen_tok_s']} wall={p['wall_s']}s")
        for g in sorted(p["games"], key=lambda g: g["game_id"]):
            print(f"  {g['game_id']}: levels={g['levels_completed']}/{g['number_of_levels']} "
                  f"score={g['final_score']} actions={g['actions']} turns={g['turns']} "
                  f"state={g['state']}")
        if p["n_games"] == 3 and not FN_PHASE_ERRORS:
            verdict = "SMOKE-OK"
    print(f"FLASHNEXT FLIGHT SMOKE: {verdict} phase_errors={FN_PHASE_ERRORS} "
          f"watchdog_restarts={_WD_STATE['restarts']}")
    results = {
        "kernel": "arc3-flashnext-flight",
        "harness": "stock-duck",
        "boot": RESULTS["verdicts"].get("boot"),
        "phases": FN_PHASES,
        "phase_errors": FN_PHASE_ERRORS,
        "watchdog_restarts": _WD_STATE["restarts"],
        "verdict": verdict,
    }
    (WORKING_DIR / "flashnext_flight_smoke.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("wrote", WORKING_DIR / "flashnext_flight_smoke.json")
'''

# ---- surgical edits inside the run cell ------------------------------------
TIME_GUARD_ANCHOR = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None
"""
TIME_GUARD_BLOCK = TIME_GUARD_ANCHOR + """
        # 9 h TIME GUARD (critical): the scored box is 32,400 s; assemble+boot
        # measured ~740-1,550 s on the gate/smoke Kaggle runs plus the tarball
        # extract. Same arithmetic as graft_throughput.time_guard_per_game_s,
        # inlined — this kernel installs NO grafts (STOCK harness): ~110 games
        # at concurrency 28 = 4 waves, 300 s teardown margin, 6,600 s floor.
        elapsed = time.time() - NOTEBOOK_START_EPOCH
        bm.solver.max_runtime_s_per_game = max(6600.0, (32400 - elapsed - 300) / 4)
        print(f"flight time guard: elapsed={elapsed:.0f}s -> per_game_cap="
              f"{bm.solver.max_runtime_s_per_game:.1f}s "
              f"(4 waves @ concurrency {bm.solver.concurrency}, margin 300 s)",
              flush=True)
"""

PHASE_BEGIN_OLD = "            _fn_phase_begin(_phase_name)\n"
PHASE_BEGIN_NEW = ("            if not run_as_submission:\n"
                   "                _fn_phase_begin(_phase_name)\n")
PHASE_END_OLD = """            try:
                _fn_phase_end(_phase_name, list(bm.game_runs))
            except Exception:  # noqa: BLE001
                pass
"""
PHASE_END_NEW = """            if not run_as_submission:
                try:
                    _fn_phase_end(_phase_name, list(bm.game_runs))
                except Exception:  # noqa: BLE001
                    pass
"""

# telemetry cell: the play seam only installs on the commit path
TEL_INSTALL_OLD = 'if not getattr(_solver_mod._HarnessGameSession.play, "_fn_tel", False):'
TEL_INSTALL_NEW = ('if (not run_as_submission) and not getattr(\n'
                   '        _solver_mod._HarnessGameSession.play, "_fn_tel", False):')

# run-cell comment: the smoke kernel's phase-loop note is wrong for a
# submittable kernel — the scored path here IS the flight.
LOOP_NOTE_OLD = """        # flashnext smoke: ONE phase — the STOCK duck against the Flash-Next
        # server. The scored-rerun path takes exactly one pass with the
        # competition games (and would need the 27B kernel, not this one).
"""
LOOP_NOTE_NEW = """        # flashnext flight: commit path = ONE 3-game smoke phase; the scored
        # rerun takes exactly one pass with the FULL competition games under
        # the 9 h time guard set above.
"""

# teardown shim (CELL_DUCK_ENV): stop the watchdog before stopping the server
TEARDOWN_OLD = ('    print(f"taaf.kaggle: {label} ({filename}) -> flashnext stop_server",'
                ' flush=True)\n')
TEARDOWN_NEW = TEARDOWN_OLD + ('    _wd = globals().get("_WD_STOP")\n'
                               '    if _wd is not None:\n'
                               '        _wd.set()\n')

# kernel-wide renames, applied LAST to every code cell
RENAMES = [
    ("arc3-flashnext-smoke", "arc3-flashnext-flight"),
    ("flashnext-smoke", "flashnext-flight"),
    ("flashnext_smoke", "flashnext_flight"),
]


def build_notebook() -> dict:
    smoke = runpy.run_path(str(SMOKE_BUILD))
    nb = smoke["build_notebook"]()      # smoke invariants all assert here first

    def idx_of(marker: str) -> int:
        hits = [i for i, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": text.splitlines(keepends=True)}

    def replace_cell(marker: str, text: str) -> None:
        nb["cells"][idx_of(marker)]["source"] = text.splitlines(keepends=True)

    def edit_cell(marker: str, old: str, new: str) -> None:
        idx = idx_of(marker)
        src = "".join(nb["cells"][idx]["source"])
        assert src.count(old) == 1, (marker, old[:60], src.count(old))
        nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)

    # 0) markdown header
    assert nb["cells"][0]["cell_type"] == "markdown"
    nb["cells"][0]["source"] = MD_HEADER.splitlines(keepends=True)

    # 1) 25-game smoke hook -> v12-style 3-game commit smoke
    replace_cell(MARK_SMOKE, CELL_SMOKE)

    # 2) watchdog cell right after the attestation cell (server proven live)
    nb["cells"].insert(idx_of(MARK_ATTEST) + 1, code_cell(CELL_WATCHDOG))

    # 3) run cell: telemetry gated off the scored branch + the 9 h time guard
    edit_cell(MARK_RUN, PHASE_BEGIN_OLD, PHASE_BEGIN_NEW)
    edit_cell(MARK_RUN, PHASE_END_OLD, PHASE_END_NEW)
    edit_cell(MARK_RUN, TIME_GUARD_ANCHOR, TIME_GUARD_BLOCK)
    edit_cell(MARK_RUN, LOOP_NOTE_OLD, LOOP_NOTE_NEW)

    # 4) telemetry play seam only installs on the commit path
    edit_cell("[fn-tel] installed", TEL_INSTALL_OLD, TEL_INSTALL_NEW)

    # 5) teardown shim also stops the watchdog
    edit_cell("flashnext stop_server", TEARDOWN_OLD, TEARDOWN_NEW)

    # 6) 25-game report -> commit-smoke report
    replace_cell(MARK_REPORT, CELL_REPORT)

    # 7) kernel-wide renames (new cells already say flight; idempotent there)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in RENAMES:
            src = src.replace(old, new)
        cell["source"] = src.splitlines(keepends=True)

    # ---- invariants ----------------------------------------------------------
    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    code_joined = "\n".join(code_cells)
    # ordering: GPU gate -> assemble -> boot -> env -> attest -> watchdog -> games -> run
    assert joined.index("GPU misbind") < joined.index("Phase 1 — ASSEMBLE")
    assert joined.index("Phase 1 — ASSEMBLE") < joined.index("Phase 2 — BOOT")
    assert joined.index("Phase 2 — BOOT") < joined.index("duck analyzer env")
    assert joined.index("duck analyzer env") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("server watchdog")
    assert joined.index("server watchdog") < joined.index("SMOKE_PHASES = [")
    assert joined.index("[fn-tel] installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("FLASHNEXT FLIGHT COMMIT SMOKE")
    # the serve chain is theirs, verbatim (inherited from the smoke build)
    for needed in ('"VLLM_PLE_CPU_OFFLOAD": "1"', '"--tensor-parallel-size", "1"',
                   '"--max-model-len", "32768"', '"--max-num-seqs", "22"',
                   '"--tool-call-parser", "qwen3_xml"', '"--enable-prefix-caching"',
                   "Qwen4ExpForConditionalGeneration", "qwen4_exp", "attest: OK",
                   'os.environ["ONLY_RESET_LEVELS"] = "true"'):
        assert needed in joined, f"missing from kernel: {needed}"
    # flight smoke hook: 3 games, once each, cap 2400, soft end 7200
    assert '("smoke3", SMOKE_GAMES, 2400)' in joined
    assert "timedelta(seconds=7200)" in joined
    assert "GAMES_25" not in joined
    for name in SMOKE_GAMES:
        assert joined.count(name) == 1, name
    # scored branch: gateway games + n_passes 1 + parquet + the 9 h time guard
    assert "bm.games = _competition_games()" in joined
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined
    assert "Kaggle gateway did not become ready" in joined
    assert "submission.to_parquet(WORKING_DIR" in joined
    assert "max(6600.0, (32400 - elapsed - 300) / 4)" in joined
    assert (joined.index("bm.games = _competition_games()")
            < joined.index("max(6600.0, (32400 - elapsed - 300) / 4)"))
    # telemetry gated off the scored branch
    assert PHASE_BEGIN_NEW in code_joined and PHASE_END_NEW in code_joined
    assert "(not run_as_submission) and not getattr(" in code_joined
    # watchdog wired to the real server lifecycle
    for needed in ("flashnext-watchdog.log", "_WD_STOP.wait(60)", "server_alive(timeout=20)",
                   "start_server(_flags, tag=_tag)", "_wd.set()"):
        assert needed in code_joined, f"missing watchdog wiring: {needed}"
    # STOCK: no graft installs — 'graft_' only ever inside comments
    for cell_src in code_cells:
        for line in cell_src.splitlines():
            if "graft_" in line:
                assert "#" in line and line.index("#") < line.index("graft_"), (
                    f"graft reference outside a comment: {line!r}")
    for banned in ("TP_ENABLE", "TP2_ENABLE", "foysalemonshanto", "/kaggle/input/models",
                   "Qwen3_5ForConditionalGeneration", "vrfai", '_run_shell_commands("setup'):
        assert banned not in joined, f"banned token in flight kernel: {banned}"
    # renames complete
    assert "flashnext-smoke" not in code_joined and "flashnext_smoke" not in code_joined
    return nb


def main() -> None:
    nb = build_notebook()
    nb_path = HERE / f"{KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "jcole75/arc3-qwen36-runtime-wheels",
            "sonphamorg/arc3-flashnext-serving-part-a-v1",
            "sonphamorg/arc3-flashnext-serving-part-b-v1",
            "sonphamorg/arc3-flashnext-serving-part-c-v1",
            "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
        ],
        "kernel_sources": [],
        # THE RTX Pro 6000 GATE (08-22 push lesson): the competition source is
        # what admits the kernel to the scored GPU pool — and makes it
        # submittable.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # NO model_sources: the 27B mount is deliberately absent — the model
        # axis is the experiment; flashnext ships as the datasets above.
        "model_sources": [],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest()[:16])
    print("cells:", len(nb["cells"]), "| notebook:", nb_path, "|", nb_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
