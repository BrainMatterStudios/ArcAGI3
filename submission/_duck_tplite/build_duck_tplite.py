#!/usr/bin/env python3
"""Build submission/_duck_tplite/duck-tplite.ipynb.

"Throughput-lite": the new best-guess config after sub 54887708 (full patch bundle)
scored 0.81 — below every clean base-duck draw (0.92/0.98/1.10/1.26). At n=1 that is
suggestive, not proof, but the plausible harm mechanism is the ANIMATION METADATA
(a per-action token tax on every animated action, and many games animate constantly),
not the ACTION7 fix (which only converts impossible error-turns into real probes on
the ~6/25 games that offer ACTION7). So this build decomposes the bundle:

  From duck-repro.ipynb (the accessible base bundle), apply:
    1. CPU-safe commit guards        (same as _duck_base / _duck_patched)
    2. ACTION7 round-trip fix ONLY   (animation metadata PARKED, not applied)
    3. vLLM KV retune                (max-model-len 65536 -> analyzer window + 8192;
                                      the KV cache supports ~10 concurrent max-len
                                      requests while the solver runs 28)
    4. adaptive per-game budget      (raise-only, derived from the served game count)

  Model, quantization, sampling, concurrency, n_passes, game list, submission path:
  unchanged.

Usage:  .venv/bin/python submission/_duck_tplite/build_duck_tplite.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_repro/duck-repro.ipynb"
PATCHES = REPO / "submission/_duck_patched/duck_patches.py"
OUT = Path(__file__).parent / "duck-tplite.ipynb"

HEADROOM = 8192

# ---- 1. CPU-safe commit guards (identical to _duck_base / _duck_patched) ----

CPU_SAFE_SETUP_OLD = (
    'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
    '    print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
    '    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n'
    "    # Re-read in case the command persisted new env keys.\n"
    "    env = _command_env()\n"
    "    os.environ.update(env)\n"
)
CPU_SAFE_SETUP_NEW = (
    "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it\n"
    '    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
    '        print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
    "        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n"
    "        env = _command_env()\n"
    "        os.environ.update(env)\n"
    "else:\n"
    '    print("[duck] commit: skipping setup_commands (no GPU serve)", flush=True)\n'
)

CPU_SAFE_RUN_OLD = (
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
    "    if not TRUE_SUBMISSION:\n"
    "        # An offline run isn't scored, but Kaggle still expects a submission.parquet output.\n"
    "        import pandas as pd\n"
    "\n"
    "        pd.DataFrame(\n"
    '            [["1_0", "1", True, 1]],\n'
    '            columns=["row_id", "game_id", "end_of_game", "score"],\n'
    "        ).to_parquet(WORKING_DIR / \"submission.parquet\", index=False)\n"
)
CPU_SAFE_RUN_NEW = (
    "    if TRUE_SUBMISSION:\n"
    "        await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
    "    else:\n"
    "        import pandas as pd\n"
    '        pd.DataFrame([["1_0", "1", True, 1]],\n'
    '                     columns=["row_id", "game_id", "end_of_game", "score"]\n'
    "                     ).to_parquet(WORKING_DIR / \"submission.parquet\", index=False)\n"
    '        print("[duck] commit: CPU-safe landing; scored rerun runs duck-tplite", flush=True)\n'
)

# ---- 2. patch cell: ACTION7 only ----

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"


def patch_cell_source() -> str:
    body = PATCHES.read_text()
    return (
        "# ============================================================================\n"
        "# duck-tplite patch cell. Inlines submission/_duck_patched/duck_patches.py but\n"
        "# applies ONLY the ACTION7 round-trip fix (+ the RESET assertion). The animation\n"
        "# metadata patches are PARKED: sub 54887708 (full bundle) scored 0.81, below\n"
        "# every clean base draw, and the per-action token tax of the animation fields is\n"
        "# the plausible harm mechanism. Decompose, don't guess.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "_patch_results = []\n"
        "for _fn in (patch_action7, verify_reset_already_handled):\n"
        "    try:\n"
        "        _patch_results.append(_fn())\n"
        "    except Exception as _exc:\n"
        "        _patch_results.append(f'{_fn.__name__}: FAIL ({_exc})')\n"
        "for _line in _patch_results:\n"
        "    print(f'[duck-patch] {_line}', flush=True)\n"
        "print('[duck-patch] animation metadata: PARKED (not applied in tplite)', flush=True)\n"
    )


# ---- 3. KV retune (identical logic to _duck_throughput) ----

SETUP_LOAD_OLD = 'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
SETUP_LOAD_NEW = "for command in _retuned_setup_commands():"
SETUP_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU"

RETUNE_HELPER = f'''
def _retuned_setup_commands():
    """Load the bundled setup commands, lowering vLLM's KV reservation to fit the agent.

    The bundle serves with --max-model-len 65536 while the analyzer caps prompt+completion
    at LOCAL_ANALYZER_CONTEXT_WINDOW (32768). The unused half of every sequence's KV
    reservation is what limits how many of the 28 concurrent games fit in cache
    (a real run logs "Maximum concurrency for 65,536 tokens per request: 10.12x").
    Rewrite the constant in place rather than hand-rolling the serve command.
    """
    import re

    raw = (BUNDLE_DIR / "setup_commands.json").read_text()
    commands = json.loads(raw)

    window = re.search(r"ANALYZER_CONTEXT_WINDOW\\s*=\\s*(\\d+)", raw)
    current = re.search(r"VLLM_MAX_MODEL_LEN\\s*=\\s*(\\d+)", raw)
    if not window or not current:
        print("[duck-tplite] WARNING: could not find the context constants; "
              "serving UNCHANGED", flush=True)
        return commands

    target = int(window.group(1)) + {HEADROOM}
    if target >= int(current.group(1)):
        print(f"[duck-tplite] target {{target}} not below current "
              f"{{current.group(1)}}; serving UNCHANGED", flush=True)
        return commands

    retuned, n = re.subn(
        r"VLLM_MAX_MODEL_LEN\\s*=\\s*\\d+",
        f"VLLM_MAX_MODEL_LEN = {{target}}",
        raw,
    )
    if n != 1:
        print(f"[duck-tplite] WARNING: expected 1 rewrite, made {{n}}; "
              "serving UNCHANGED", flush=True)
        return commands

    print(f"[duck-tplite] max-model-len {{current.group(1)}} -> {{target}} "
          f"(analyzer window {{window.group(1)}} + {HEADROOM} headroom)", flush=True)
    return json.loads(retuned)

'''

# ---- 4. adaptive per-game budget (identical logic to _duck_throughput) ----

GAMES_SET_OLD = "    bm.games = _competition_games()"
GAMES_SET_NEW = '''    bm.games = _competition_games()

    # --- adaptive per-game budget: raise-only, derived from the served game count ---
    # Shipped tuning (7920s x concurrency 28) exactly fills 9h IF the eval set is 110
    # games; the ARC technical report says the competition set is 55 (= 2 waves = 4.4h,
    # leaving ~4.6h idle). Never bet on either number: derive waves from what the
    # gateway serves and only ever RAISE the per-game budget, so a 110-game set is
    # byte-for-byte current behaviour.
    try:
        _n_games = len(bm.games)
        _conc = int(getattr(bm.solver, "concurrency", 0) or 0)
        _total = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
        _current = float(getattr(bm.solver, "max_runtime_s_per_game", 0.0) or 0.0)
        if _n_games > 0 and _conc > 0 and _total > 0 and _current > 0:
            _waves = -(-_n_games // _conc)  # ceil
            _elapsed = time.time() - NOTEBOOK_START_EPOCH  # setup + vLLM serve
            _per_game = (_total - _elapsed) * 0.9 / _waves
            if _per_game > _current:
                bm.solver.max_runtime_s_per_game = _per_game
                print(f"[duck-adaptive] {_n_games} games / concurrency {_conc} "
                      f"= {_waves} wave(s); per-game budget {_current:.0f}s -> "
                      f"{_per_game:.0f}s ({_elapsed:.0f}s spent on setup)", flush=True)
            else:
                print(f"[duck-adaptive] {_n_games} games / {_conc} = {_waves} wave(s); "
                      f"computed {_per_game:.0f}s not above shipped {_current:.0f}s "
                      "— budget UNCHANGED", flush=True)
        else:
            print(f"[duck-adaptive] missing a knob (games={_n_games}, conc={_conc}, "
                  f"total={_total}, current={_current}) — budget UNCHANGED", flush=True)
    except Exception as _exc:
        print(f"[duck-adaptive] WARNING: {_exc!r} — budget UNCHANGED", flush=True)
    # ------------------------------------------------------------------------------'''


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"setup": False, "run": False, "hook": False, "retune": False, "adaptive": False}
    cells = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] == "code" and "setup_commands.json" in src and "%%" not in src[:2]:
            if CPU_SAFE_SETUP_OLD not in src:
                raise SystemExit("setup_commands loop did not match — upstream changed")
            src = src.replace(CPU_SAFE_SETUP_OLD, CPU_SAFE_SETUP_NEW)
            seen["setup"] = True
            # retune: helper above the guard, loop swapped to the retuned loader
            if SETUP_GUARD not in src or src.count(SETUP_LOAD_OLD) != 1:
                raise SystemExit("retune anchors missing after CPU-safe transform")
            src = src.replace(SETUP_GUARD, RETUNE_HELPER + "\n" + SETUP_GUARD, 1)
            src = src.replace(SETUP_LOAD_OLD, SETUP_LOAD_NEW, 1)
            seen["retune"] = True

        elif cell["cell_type"] == "code" and "await bm.run(" in src:
            if CPU_SAFE_RUN_OLD not in src:
                raise SystemExit("run-cell block did not match — upstream changed")
            src = src.replace(CPU_SAFE_RUN_OLD, CPU_SAFE_RUN_NEW)
            seen["run"] = True
            if src.count(GAMES_SET_OLD) != 1:
                raise SystemExit("bm.games assignment anchor missing")
            src = src.replace(GAMES_SET_OLD, GAMES_SET_NEW, 1)
            seen["adaptive"] = True

        elif cell["cell_type"] == "code" and HOOK_MARKER in src:
            src = patch_cell_source()
            seen["hook"] = True

        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"anchors never found: {missing}")

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    attachments = sum(1 for c in cells if c.get("attachments"))
    print(f"wrote {OUT}  ({len(cells)} cells, {attachments} with attachments, "
          f"anchors: {sorted(k for k, v in seen.items() if v)})")


if __name__ == "__main__":
    main()
