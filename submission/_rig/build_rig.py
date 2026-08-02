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
REPEATS = int(os.environ.get("RIG_REPEATS", 1))      # independent runs inside one kernel
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
    """Arm pack (optional) + behavioural probe (ALWAYS, identically in both arms).

    The probe is what makes a GPU run worth its minutes: scoring 28 near-binary games
    is at the noise floor (17 of 28 flip between identical repeats, measured), while
    the per-action mechanism has ~1,600 samples per run. It is injected in BOTH arms
    and observes only, so it cannot become the difference between them.
    """
    probe = (Path(__file__).parent / "behav_probe.py").read_text()
    parts = []
    if PACK:
        parts.append(f"# rig: arm under test, inlined from {PACK}.\n{Path(PACK).read_text()}\n")
        parts.append(
            "_rig_pack = apply_all()\n"
            "for _n, _ok in _rig_pack.items():\n"
            "    print(f\"[rig] pack {_n}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
            "if not all(_rig_pack.values()):\n"
            '    raise RuntimeError(f"[rig] pack did not apply: {_rig_pack}")\n'
            "board_fix_stats = stats\n"
            "board_fix_assert = assert_fired\n"
            f'print("[rig] pack: {LABEL} ACTIVE", flush=True)\n')
    else:
        parts.append(f'print("[rig] pack: none (label={LABEL})", flush=True)\n')
    parts.append("\n# --- behavioural probe: identical in every arm, observes only ---\n")
    parts.append(probe + "\n")
    parts.append(
        "if not install():\n"
        '    raise RuntimeError("[rig] behavioural probe failed to install")\n'
        "behav_report = report\n"
        "behav_assert = assert_observed\n"
        'print("[rig] behavioural probe ACTIVE", flush=True)\n')
    return "".join(parts)


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
_all_repeats = []

def _dump(extra=None):
    _o = {{"label": "{LABEL}", "pack": "{PACK}", "per_game_budget": {BUDGET},
          "config": globals().get("_cfg"), "clone_map": globals().get("_map"),
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
    # Proof of what actually ran. If sampling or context drifted, an arm comparison is
    # meaningless — record it rather than trusting the build.
    _stage = "config_snapshot"
    from inference.agent import tool_agent as _ta_cfg
    _cfg = {{
        "temperature": float(_ta_cfg._LOCAL_ANALYZER_TEMPERATURE),
        "top_k": int(_ta_cfg._LOCAL_ANALYZER_TOP_K),
        "top_p": float(_ta_cfg._LOCAL_ANALYZER_TOP_P),
        "seed": int(_ta_cfg._LOCAL_ANALYZER_SEED),
        "context_window": int(_ta_cfg._LOCAL_ANALYZER_CONTEXT_WINDOW),
        "concurrency": getattr(bm.solver, "concurrency", None),
        "per_game_budget": {BUDGET},
        "n_games": {NGAMES},
        "n_repeats": {REPEATS},
        "model": str(getattr(getattr(bm.solver, "analyzer_model", None), "model_id", "")) or None,
        "bundle_dir": str(BUNDLE_DIR),
        "estimator_chars_per_token": round(
            len(__import__("json").dumps({{"a": "x" * 300}})) / _ta_cfg._estimate_tokens({{"a": "x" * 300}}), 2),
    }}
    print(f"[rig] config {{_cfg}}", flush=True)

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
    # imports re_arc — absent from both the bundle and the Kaggle image. Construct the
    # server directly with explicit game_ids instead.
    #
    # REPEATS give independent runs of the SAME config inside one kernel. Each repeat
    # gets a fresh server and therefore a fresh scorecard, because competition mode
    # permits exactly one run per game id. This is what makes rho estimable: split the
    # games of each repeat into two disjoint halves and correlate the half-means ACROSS
    # repeats, which separates run-level noise (shared by both halves) from game-level
    # noise (independent) — exactly the decomposition that decides whether selecting on
    # the public half tells you anything about the private half.
    _all_repeats = []
    for _rep in range({REPEATS}):
        _stage = f"start_arcade_server[{{_rep}}]"
        _srv = _ca.CompetitionArcadeServer(
            game_ids=tuple(_official), total_runs={NGAMES}, environments_dir=_env_dir,
        ).start()
        globals()["_rig_server_%d" % _rep] = _srv
        _spec = _srv.arcade_spec
        _clones = _srv.exposed_game_ids
        if len(_clones) != {NGAMES}:
            raise RuntimeError(f"expected {NGAMES} clones, got {{len(_clones)}}")

        # AUTHORITATIVE clone->source mapping, read from the private tag the cloner
        # writes, and cross-checked against the modulo reconstruction. Getting this
        # join wrong would silently score every game against the wrong baselines.
        _map = {{}}
        for _ei in _srv._arcade.available_environments:
            _src = None
            for _t in (getattr(_ei, "private_tags", None) or []):
                if str(_t).startswith("taaf_source_game:"):
                    _src = str(_t).split(":", 1)[1]
            _map[_ei.game_id] = _src
        _recon = {{f"k{{i:03d}}": _official[i % len(_official)] for i in range(len(_clones))}}
        _mismatch = {{k: (v, _recon.get(k)) for k, v in _map.items() if v != _recon.get(k)}}
        if _mismatch:
            raise RuntimeError(f"clone->source mapping disagrees with reconstruction: {{_mismatch}}")
        if any(v is None for v in _map.values()):
            raise RuntimeError(f"clone->source mapping incomplete: {{_map}}")

        _stage = f"build_benchmark[{{_rep}}]"
        _games = [taaf.game_api.GameAPI(env_name=_g2, arcade_spec=_spec) for _g2 in _clones]
        _rig = taaf.benchmark.Benchmark(label=f"rig_{LABEL}_{{_rep}}", games=_games,
                                        solver=bm.solver, n_passes=1)
        for _attr in ("max_runtime_s_per_game", "max_runtime_s"):
            if hasattr(_rig.solver, _attr):
                setattr(_rig.solver, _attr, {BUDGET})
        print(f"[rig] repeat {{_rep}}: {{len(_rig.games)}} games @ {BUDGET}s "
              f"arcade={{_srv.base_url}}", flush=True)

        _stage = f"run[{{_rep}}]"
        await _rig.run(soft_end_time=None, runtime_environment=target, minimal_diagnostics=False)

        _stage = f"collect[{{_rep}}]"
        _rrows = []
        for _gr in (getattr(_rig, "game_runs", None) or []):
            # final_score is 0.0 BY DESIGN here: _compute_final_score returns 0.0 when
            # base_actions_per_level is None, and the competition arcade hides baselines
            # exactly as a real submission does. Record the RAW COMPONENTS and score
            # offline against environment_files/*/metadata.json, which we hold locally.
            _cid = getattr(_gr, "game_id", None)
            _apl = list(getattr(_gr, "actions_per_level", []) or [])
            _hist = len(getattr(_gr, "history", []) or [])
            _rrows.append({{
                "clone_id": _cid,
                "source_game": _map.get(_cid),
                "levels_completed": getattr(_gr, "levels_completed", None),
                "levels_total": getattr(_gr, "number_of_levels", None),
                "actions_per_level": _apl,
                "actions_total": _hist,
                # Invariant the harness documents: sum(actions_per_level)==len(history).
                # Logged rather than assumed so a violation is visible in the data.
                "apl_sum_matches_history": (sum(_apl) == _hist),
                "state": str(getattr(_gr, "state", None)),
                "wallclock_s": getattr(_gr, "final_wallclock_seconds", None),
                "gen_tokens": getattr(_gr, "final_generated_tokens", None),
                # Expected None in competition mode (baselines hidden). Logged to PROVE
                # that is why final_score is 0.0, rather than inferring it.
                "base_actions_per_level": getattr(_gr, "base_actions_per_level", None),
                "harness_final_score": getattr(_gr, "final_score", None),
            }})
        # EFFECT GATE. Verifying that a patch installed proves nothing — this arm's
        # stem-resolution bug made it a behavioural no-op while every install check
        # passed. Demand evidence it actually changed something.
        try:
            _bfstats = board_fix_stats()  # type: ignore[name-defined]
        except Exception:
            _bfstats = None
        try:
            _behav = behav_report()  # type: ignore[name-defined]
            print(f"[rig] behav after repeat {{_rep}}: {{_behav['corpus']}}", flush=True)
            globals()["_behav_last"] = _behav
            if _rep == 0:
                behav_assert()  # type: ignore[name-defined]
        except Exception as _e:
            print(f"[rig] behav report unavailable: {{_e!r}}", flush=True)
        if _bfstats is not None:
            print(f"[rig] boardfix stats after repeat {{_rep}}: {{_bfstats}}", flush=True)
            globals()["_bf_stats_last"] = _bfstats
            if _rep == 0:
                board_fix_assert()  # type: ignore[name-defined]
        _all_repeats.append(_rrows)
        _rows = list(_rrows)   # copy — sharing the object let the post-loop collector
                               # append into _all_repeats[-1] and report 56 games, not 28
        _sc = [r["score"] for r in _rrows if isinstance(r.get("score"), (int, float))]
        print(f"[rig] repeat {{_rep}} done: n={{len(_sc)}} "
              f"mean={{(sum(_sc)/len(_sc)) if _sc else float('nan'):.4f}}", flush=True)
        _dump({{"repeats": _all_repeats}})
        try:
            _srv.stop()
        except Exception:
            pass
    _stage = "collect"
except Exception:
    _err = traceback.format_exc()
    print(f"[rig] FAILED at stage={{_stage}}:\\n{{_err}}", flush=True)

# Shape probe only — the per-repeat collector above is authoritative. An earlier
# version re-collected here into the same list object and reported 56 games for a
# 28-game repeat, mixing two row schemas.
_probe = None
try:
    _first = (getattr(_rig, "game_runs", None) or [None])[0]
    if _first is not None:
        _probe = [a for a in dir(_first) if not a.startswith("__")][:60]
except Exception:
    pass
_dump({{"game_run_attrs": _probe, "repeats": _all_repeats,
       "boardfix_stats": globals().get("_bf_stats_last"),
       "behav": globals().get("_behav_last")}})

# NOTE: harness_final_score is 0.0 by design here (baselines hidden), so progress is
# reported as levels completed. Real scores are computed offline by score_rig.py.
_lvl = [r.get("levels_completed") or 0 for rr in _all_repeats for r in rr]
_act = [r.get("actions_total") or 0 for rr in _all_repeats for r in rr]
if _lvl:
    print(f"[rig] RESULT label={LABEL} repeats={{len(_all_repeats)}} games={{len(_lvl)}} "
          f"mean_levels={{sum(_lvl)/len(_lvl):.3f}} games_with_level={{sum(1 for x in _lvl if x>0)}} "
          f"mean_actions={{sum(_act)/max(len(_act),1):.1f}}", flush=True)
else:
    print(f"[rig] RESULT label={LABEL} — NO ROWS (stage={{_stage}})", flush=True)
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
