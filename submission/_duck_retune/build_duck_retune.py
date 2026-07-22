#!/usr/bin/env python3
"""Build submission/_duck_retune/duck-retune.ipynb.

"Retune-only": isolates the throughput lever. After two consecutive sub-0.9 scored
draws whose COMMON component was the ACTION7 fix (54887708 = ACTION7+animation -> 0.81;
54897987 = ACTION7+retune+adaptive -> 0.85, vs clean base draws 0.92/0.98/1.10/1.26),
the ACTION7 fix is the prime suspect: our port fixed the action mapping but never
shipped the fork's guidance prompt, handing the model a newly-executable mystery action
(often "undo") that costs real actions against the squared-efficiency metric on every
probe. So this build drops ALL model-facing patches:

  From duck-repro.ipynb (the accessible base bundle), apply ONLY:
    1. CPU-safe commit guards       (same as _duck_base)
    2. vLLM KV retune               (max-model-len 65536 -> analyzer window + 8192)
    3. adaptive per-game budget     (raise-only, derived from the served game count)

  NO ACTION7 fix. NO animation metadata. The stock customization hook is untouched.
  Model, quantization, sampling, concurrency, n_passes, game list: unchanged.

Interpretation, pre-registered: >=~1.0 -> throughput lever live, ACTION7-as-shipped was
the harm. Still <0.9 -> retune/adaptive suspect or eval drift -> next slot = pure base
replicate to re-anchor.

Usage:  .venv/bin/python submission/_duck_retune/build_duck_retune.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_repro/duck-repro.ipynb"
OUT = Path(__file__).parent / "duck-retune.ipynb"

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
    '        print("[duck] commit: CPU-safe landing; scored rerun runs duck-retune", flush=True)\n'
)

# ---- 2. patch cell: ACTION7 only ----

# ---- 2. KV retune (identical logic to _duck_throughput) ----

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
        print("[duck-retune] WARNING: could not find the context constants; "
              "serving UNCHANGED", flush=True)
        return commands

    target = int(window.group(1)) + {HEADROOM}
    if target >= int(current.group(1)):
        print(f"[duck-retune] target {{target}} not below current "
              f"{{current.group(1)}}; serving UNCHANGED", flush=True)
        return commands

    retuned, n = re.subn(
        r"VLLM_MAX_MODEL_LEN\\s*=\\s*\\d+",
        f"VLLM_MAX_MODEL_LEN = {{target}}",
        raw,
    )
    if n != 1:
        print(f"[duck-retune] WARNING: expected 1 rewrite, made {{n}}; "
              "serving UNCHANGED", flush=True)
        return commands

    print(f"[duck-retune] max-model-len {{current.group(1)}} -> {{target}} "
          f"(analyzer window {{window.group(1)}} + {HEADROOM} headroom)", flush=True)
    return json.loads(retuned)

'''

# ---- 3. adaptive per-game budget (identical logic to _duck_throughput) ----

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
    seen = {"setup": False, "run": False, "retune": False, "adaptive": False}
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
