#!/usr/bin/env python3
"""Build the GPU measurement rig — a commit kernel that scores arms without a slot.

WHY. Until now every arm cost a submission slot to evaluate, at one per day, against
a public-LB noise floor that needs ~60 draws to resolve +0.10. Thirty-four scored
submissions produced one usable distribution and zero measured improvements.

`make_benchmark_kaggle_official_110(competition_sim=True)` builds the submission-shaped
110-game benchmark from the 25 public games and starts a local CompetitionArcadeServer:
shared scorecard, hidden baselines, clone IDs, one make() per environment, level-resets
forced. A commit run therefore produces a real, submission-shaped score, and commit
output IS retrievable (`kaggle kernels output`) where scored-rerun output is not.

BUDGET — corrected by measurement, run #3. Shortening the per-game box does not
shrink the regime, it eliminates it. At 600s/game the agent took a MEAN OF 8.8
ACTIONS (median 6) and completed a level in only 5 of 110 games, because 28 games
share one GPU and a turn costs ~70s wall. The real 7920s box yields ~113
actions/game, consistent with the audit's measured median of 78 in real episodes.
Six actions is barely past the opening; there is nothing there to compare.

So buy per-game realism by cutting GAME COUNT, not the box. One concurrency wave
at the full box:
    28 games @ 7920s = 1 wave  ->  ~2.2h + load  ->  ~11 sweeps per 30h week
    28 games @ 3600s = 1 wave  ->  ~1.0h + load  ->  ~23 sweeps, ~51 actions/game
    110 games @ 600s = 4 waves ->  ~0.7h        ->  MEASURES ALMOST NOTHING
Statistical power comes from 28 realistic games rather than 110 truncated ones.

WHAT IS REWRITTEN relative to duck-base:
  cell 8   force the serve. duck-base skips setup_commands unless
           KAGGLE_IS_COMPETITION_RERUN, and the rig needs Qwen actually served.
  cell 12  optional pack injection (the arm under test), same hook cell every arm uses.
  cell 14  replaced wholesale. duck-base's submission branch polls a Kaggle gateway
           that does not exist in a commit run, so the rig substitutes its own run:
           competition_sim benchmark, budget override, per-game score dump.

Usage:
    python3 submission/_rig/build_rig.py                          # baseline, no pack
    RIG_PACK=submission/_duck_boardfix/board_fix.py \
    RIG_LABEL=boardfix RIG_BUDGET=7920 RIG_GAMES=28 python3 submission/_rig/build_rig.py
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
OUT_DIR = Path(__file__).parent

PACK = os.environ.get("RIG_PACK", "")
LABEL = os.environ.get("RIG_LABEL", "base")
BUDGET = float(os.environ.get("RIG_BUDGET", 7920))   # the real per-game box
NGAMES = int(os.environ.get("RIG_GAMES", 28))        # ONE concurrency wave
SLUG = f"arc-agi-3-rig-{LABEL}"
# One directory per label. Sharing a directory means every build overwrites the
# previous label's kernel-metadata.json, so `kaggle kernels push -p submission/_rig`
# silently pushes whichever variant was built last — which is exactly what happened
# on the first attempt.
ARM_DIR = OUT_DIR / LABEL
ARM_DIR.mkdir(parents=True, exist_ok=True)
OUT = ARM_DIR / f"rig-{LABEL}.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
SERVE_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it"
RUN_MARKER = "# Build the live competition game list from the gateway's available environments."


def hook_cell() -> str:
    if not PACK:
        return (
            "# rig: baseline arm, no pack injected.\n"
            f'print("[rig] pack: none (label={LABEL})", flush=True)\n'
        )
    body = Path(PACK).read_text()
    return (
        f"# rig: arm under test, inlined from {PACK}.\n"
        f"{body}\n"
        "\n"
        "_rig_pack = apply_all()\n"
        "for _n, _ok in _rig_pack.items():\n"
        "    print(f\"[rig] pack {_n}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
        "if not all(_rig_pack.values()):\n"
        '    raise RuntimeError(f"[rig] pack did not apply: {_rig_pack}")\n'
        f'print("[rig] pack: {LABEL} ACTIVE", flush=True)\n'
    )


def run_cell() -> str:
    return f'''# rig: competition-simulated run. Replaces duck-base's submission cell, whose
# gateway poll cannot succeed in a commit run.
#
# EVERYTHING is wrapped and a diagnostic file is written unconditionally: an ERRORed
# Kaggle kernel produces NO retrievable log (verified — a COMPLETE run yields a .log,
# an ERROR run does not), so a rig that dies silently teaches us nothing and costs an
# hour of quota. Files written to the working dir DO survive an error.
import json, time, traceback

_t0 = time.time()
_stage = "init"
_err = None
_rows = []
_rig = None

def _dump(extra=None):
    _o = {{"label": "{LABEL}", "pack": "{PACK}", "per_game_budget": {BUDGET},
          "stage": _stage, "elapsed_s": round(time.time() - _t0, 1),
          "error": _err, "n_games": len(_rows), "rows": _rows}}
    if extra:
        _o.update(extra)
    (WORKING_DIR / "rig_result.json").write_text(json.dumps(_o, indent=1, default=str))

try:
    _stage = "preamble"
    print((BUNDLE_DIR / "preamble.txt").read_text())
    os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

    # NOTE: taaf.standard_benchmarks.make_benchmark_kaggle_official_110 imports `re_arc`
    # to list the 25 official ids. That package is NOT in the shipped bundle and not in
    # the Kaggle image — it is what killed the first rig run. Build the same thing from
    # the offline arcade instead, which the competition dataset already provides.
    _stage = "official_ids"
    import arc_agi, taaf.game_api, taaf.competition_arcade as _ca, taaf.benchmark
    _env_dir = os.environ.get("ARC_ENVIRONMENTS_DIR", "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files")
    if not os.path.isdir(_env_dir):
        import glob as _g
        _cands = _g.glob("/kaggle/input/**/environment_files", recursive=True)
        if not _cands:
            raise RuntimeError("no environment_files directory found under /kaggle/input")
        _env_dir = _cands[0]
    _arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    _official = sorted(e.game_id for e in _arc.available_environments)
    print(f"[rig] env_dir={{_env_dir}} official={{len(_official)}}", flush=True)
    if len(_official) != 25:
        raise RuntimeError(f"expected 25 official environments, got {{len(_official)}}")

    # ArcadeSpec(competition_sim=True) is NOT usable here: it lazily calls
    # CompetitionArcadeServer.official_110(), which calls official_game_ids(), which
    # imports re_arc — the package that is absent from both the bundle and the Kaggle
    # image. That is what killed rig run #2 at stage=run. Construct the server
    # directly with explicit game_ids instead; verified locally to expose 110 unique
    # clones in COMPETITION mode with no re_arc anywhere on the path.
    _stage = "start_arcade_server"
    _srv = _ca.CompetitionArcadeServer(
        game_ids=tuple(_official), total_runs={NGAMES}, environments_dir=_env_dir,
    ).start()
    globals()["_rig_server"] = _srv          # keep it alive for the whole run
    _spec = _srv.arcade_spec
    _clones = _srv.exposed_game_ids
    print(f"[rig] arcade at {{_srv.base_url}} exposing {{len(_clones)}} clones", flush=True)
    if len(_clones) != {NGAMES}:
        raise RuntimeError(f"expected {NGAMES} exposed clones, got {{len(_clones)}}")

    _stage = "build_benchmark"
    _games = [taaf.game_api.GameAPI(env_name=_g2, arcade_spec=_spec) for _g2 in _clones]
    _rig = taaf.benchmark.Benchmark(label="rig_{LABEL}", games=_games, solver=bm.solver, n_passes=1)

    # Shorten the per-game box so a sweep fits the weekly quota. This changes the
    # regime, so scores compare BETWEEN ARMS, never to real submissions.
    for _attr in ("max_runtime_s_per_game", "max_runtime_s"):
        if hasattr(_rig.solver, _attr):
            setattr(_rig.solver, _attr, {BUDGET})
    print(f"[rig] games={{len(_rig.games)}} budget={BUDGET}s "
          f"concurrency={{getattr(_rig.solver, 'concurrency', '?')}}", flush=True)
    _dump()

    _stage = "run"
    await _rig.run(soft_end_time=None, runtime_environment=target, minimal_diagnostics=False)
    _stage = "collect"
except Exception:
    _err = traceback.format_exc()
    print(f"[rig] FAILED at stage={{_stage}}:\\n{{_err}}", flush=True)

# Collect whatever exists, whether or not the run completed.
try:
    for _gr in (getattr(_rig, "game_runs", None) or []):
        _rows.append({{"game_id": getattr(_gr, "game_id", None),
                      "score": getattr(_gr, "final_score", None),
                      "levels_total": getattr(_gr, "number_of_levels", None),
                      "actions_per_level": getattr(_gr, "actions_per_level", None),
                      "levels_completed": getattr(_gr, "levels_completed", None),
                      "actions": len(getattr(_gr, "history", []) or [])}})
except Exception as _e:
    _rows.append({{"collect_error": repr(_e)}})

# Shape probe: if .score is not where we expect, record what IS on the object so the
# next build can fix the extraction without spending another hour of quota.
_probe = None
try:
    _first = (getattr(_rig, "game_runs", None) or [None])[0]
    if _first is not None:
        _probe = [a for a in dir(_first) if not a.startswith("__")][:60]
except Exception:
    pass
_dump({{"game_run_attrs": _probe}})

_scored = [r.get("score") for r in _rows if isinstance(r.get("score"), (int, float))]
if _scored:
    print(f"[rig] RESULT label={LABEL} n={{len(_scored)}} "
          f"mean={{sum(_scored)/len(_scored):.4f}} max={{max(_scored):.4f}}", flush=True)
else:
    print(f"[rig] RESULT label={LABEL} — NO SCORES PARSED ({{len(_rows)}} rows, "
          f"stage={{_stage}}); see rig_result.json", flush=True)
print(f"[rig] DONE stage={{_stage}} in {{round(time.time()-_t0,1)}}s", flush=True)
'''


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"serve": False, "hook": False, "run": False}
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            if SERVE_GUARD in src:
                src = src.replace(SERVE_GUARD, "if True:  # rig: force the serve — a commit run must serve Qwen")
                seen["serve"] = True
            elif HOOK_MARKER in src:
                src = hook_cell()
                seen["hook"] = True
            elif RUN_MARKER in src:
                src = run_cell()
                seen["run"] = True
        out = dict(cell)
        out["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out["execution_count"] = None
            out["outputs"] = []
        cells.append(out)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"never found anchors: {missing}")
    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)}")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    (ARM_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{SLUG}",
        "title": SLUG,
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "ahmedmobasher86/taaf-src-hybrid",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    print(f"push with: kaggle kernels push -p {ARM_DIR}")
    print(f"wrote {OUT}  (label={LABEL} pack={PACK or 'none'} budget={BUDGET}s, anchors {sorted(seen)})")


if __name__ == "__main__":
    main()
