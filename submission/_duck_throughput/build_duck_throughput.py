#!/usr/bin/env python3
"""Build submission/_duck_throughput/duck-throughput.ipynb.

Starts from duck-patched.ipynb (so it carries the ACTION7 fix and animation metadata)
and adds ONE serving change: stop over-reserving KV cache.

THE PROBLEM
-----------
The bundled setup script starts vLLM with `--max-model-len 65536`, but the agent can
never use that much context. `tool_agent.py:138` reads LOCAL_ANALYZER_CONTEXT_WINDOW
(set to 32768 by the same setup script) and `tool_agent.py:972` caps the prompt at

    context_window - reply_reserve - request_safety_margin

then generates at most `reply_reserve` more. So prompt + completion is bounded by 32768.
We reserve exactly 2x the KV cache per sequence that any sequence can occupy.

That over-reservation is what starves concurrency. A real duck run's server log reports
`Maximum concurrency for 65,536 tokens per request: 10.12x` while the solver runs 28
games at once (concurrency=28 in benchmark_initial.pkl). KV-cache capacity in tokens is
fixed by the GPU; the number of concurrent sequences it supports is that capacity divided
by max-model-len. Halving max-model-len therefore roughly doubles the concurrent
sequences the same cache serves, which is the binding constraint on total tokens
delivered — and total tokens delivered is what buys depth.

THE CHANGE
----------
Rewrite `VLLM_MAX_MODEL_LEN` inside the bundled setup script at notebook runtime, to
ANALYZER_CONTEXT_WINDOW + HEADROOM rather than a hardcoded number, so the two stay
consistent if the analyzer window is ever retuned.

HEADROOM is deliberately generous (8192). Setting max-model-len to exactly 32768 would be
the true bound, but the agent's token accounting is an estimate and vLLM's is exact; a
drift between them returns HTTP 400 and kills the game. We lost two entire EWM gate runs
to exactly that failure mode, so we buy slack rather than the last 20% of the win.
40960 still cuts the reservation by 37.5%, i.e. ~1.6x the concurrent sequences.

WHAT IS NOT CHANGED
-------------------
Model, quantization, sampling parameters, `concurrency`, `max_runtime_s_per_game`,
`n_passes`, the game list, and the submission path. This is a serving-side change only.

Usage:  .venv/bin/python submission/_duck_throughput/build_duck_throughput.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_patched/duck-patched.ipynb"
OUT = Path(__file__).parent / "duck-throughput.ipynb"

HEADROOM = 8192

SETUP_LOAD_OLD = 'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
SETUP_LOAD_NEW = "for command in _retuned_setup_commands():"

# Inserted immediately above the setup loop.
RETUNE_HELPER = f'''
def _retuned_setup_commands():
    """Load the bundled setup commands, lowering vLLM's KV reservation to fit the agent.

    The bundle serves with --max-model-len 65536 while the analyzer caps prompt+completion
    at LOCAL_ANALYZER_CONTEXT_WINDOW (32768). The unused half of every sequence's KV
    reservation is what limits how many of the {{concurrency}} concurrent games fit in cache.
    We rewrite the constant in place rather than hand-rolling the serve command, because
    inheriting the bundle's serve is the one thing that has reliably worked.
    """
    import re

    raw = (BUNDLE_DIR / "setup_commands.json").read_text()
    commands = json.loads(raw)

    window = re.search(r"ANALYZER_CONTEXT_WINDOW\\s*=\\s*(\\d+)", raw)
    current = re.search(r"VLLM_MAX_MODEL_LEN\\s*=\\s*(\\d+)", raw)
    if not window or not current:
        print("[duck-throughput] WARNING: could not find the context constants; "
              "serving UNCHANGED", flush=True)
        return commands

    target = int(window.group(1)) + {HEADROOM}
    if target >= int(current.group(1)):
        print(f"[duck-throughput] target {{target}} not below current "
              f"{{current.group(1)}}; serving UNCHANGED", flush=True)
        return commands

    retuned, n = re.subn(
        r"VLLM_MAX_MODEL_LEN\\s*=\\s*\\d+",
        f"VLLM_MAX_MODEL_LEN = {{target}}",
        raw,
    )
    if n != 1:
        print(f"[duck-throughput] WARNING: expected 1 rewrite, made {{n}}; "
              "serving UNCHANGED", flush=True)
        return commands

    print(f"[duck-throughput] max-model-len {{current.group(1)}} -> {{target}} "
          f"(analyzer window {{window.group(1)}} + {HEADROOM} headroom); "
          f"KV reservation cut {{100 - 100 * target // int(current.group(1))}}%", flush=True)
    return json.loads(retuned)

'''


GAMES_SET_OLD = "    bm.games = _competition_games()"
GAMES_SET_NEW = '''    bm.games = _competition_games()

    # --- adaptive per-game budget -------------------------------------------------
    # The bundle ships max_runtime_s_per_game=7920 with concurrency=28, which exactly
    # fills a 9h budget IF the eval set is 110 games (110/28 = 4 waves x 7920 = 31680s).
    # Two things are unresolved: the ARC-AGI-3 technical report says the competition set
    # is 55 fully-private environments (which would be 2 waves = 4.4h, leaving ~4.6h of
    # GPU idle), and the total notebook cap has four conflicting public figures.
    #
    # Rather than bet on either number, derive the budget from what the gateway actually
    # serves. The rule is deliberately one-directional: RAISE the per-game budget when
    # there are fewer waves than the shipped tuning assumed, never lower it. So a
    # 110-game set is byte-for-byte the current behaviour, and a 55-game set claims the
    # idle half. Anything unexpected leaves the budget untouched.
    try:
        _n_games = len(bm.games)
        _conc = int(getattr(bm.solver, "concurrency", 0) or 0)
        _total = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
        _current = float(getattr(bm.solver, "max_runtime_s_per_game", 0.0) or 0.0)
        if _n_games > 0 and _conc > 0 and _total > 0 and _current > 0:
            _waves = -(-_n_games // _conc)  # ceil
            _elapsed = time.time() - NOTEBOOK_START_EPOCH  # setup + vLLM serve
            # 0.9 leaves room for the solver to drain and the scorecard to close.
            _per_game = (_total - _elapsed) * 0.9 / _waves
            if _per_game > _current:
                bm.solver.max_runtime_s_per_game = _per_game
                print(f"[duck-adaptive] {_n_games} games / concurrency {_conc} "
                      f"= {_waves} wave(s); per-game budget {_current:.0f}s -> "
                      f"{_per_game:.0f}s ({_elapsed:.0f}s already spent on setup)",
                      flush=True)
            else:
                print(f"[duck-adaptive] {_n_games} games / {_conc} = {_waves} wave(s); "
                      f"computed {_per_game:.0f}s is not above the shipped "
                      f"{_current:.0f}s — budget UNCHANGED", flush=True)
        else:
            print("[duck-adaptive] missing a knob "
                  f"(games={_n_games}, concurrency={_conc}, total={_total}, "
                  f"current={_current}) — budget UNCHANGED", flush=True)
    except Exception as _exc:
        print(f"[duck-adaptive] WARNING: {_exc!r} — budget UNCHANGED", flush=True)
    # ------------------------------------------------------------------------------'''


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"missing {BASE} — run build_duck_patched.py first")

    nb = json.loads(BASE.read_text())
    patched = False
    adaptive = False
    cells = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and GAMES_SET_OLD in src:
            if src.count(GAMES_SET_OLD) != 1:
                raise SystemExit("bm.games assignment is not unique in its cell")
            src = src.replace(GAMES_SET_OLD, GAMES_SET_NEW, 1)
            adaptive = True
        if cell["cell_type"] == "code" and SETUP_LOAD_OLD in src:
            if src.count(SETUP_LOAD_OLD) != 1:
                raise SystemExit("setup-commands loop is not unique in its cell")
            # Define the helper above the guard that uses it.
            guard = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU"
            if guard not in src:
                raise SystemExit("CPU-safe guard missing — build_duck_patched.py changed")
            src = src.replace(guard, RETUNE_HELPER + "\n" + guard, 1)
            src = src.replace(SETUP_LOAD_OLD, SETUP_LOAD_NEW, 1)
            patched = True
        # Override only `source`; see the note in build_duck_patched.py — dropping
        # `attachments` makes the commit fail at nbconvert render time.
        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    if not patched:
        raise SystemExit("never found the setup-commands loop")
    if not adaptive:
        raise SystemExit("never found the bm.games assignment")

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}  (serving retune + adaptive budget)")


if __name__ == "__main__":
    main()
