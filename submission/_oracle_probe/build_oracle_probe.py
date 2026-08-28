#!/usr/bin/env python3
"""build_oracle_probe.py — the ORACLE-MODEL PROBE on the SHIPPED 27B.

WHY THIS KERNEL EXISTS.

The local pilot (`oracle_probe.py`, qwen3:8b) ran three arms on wa30 L3 and
returned 0/5 on all of them, every run ending at exactly 3 blocks left:

    control  board + legal actions only
    oracle   + the COMPLETE verified mechanics + goal
    guided   + a per-turn plain-language instruction naming WHICH block to take
             and WHICH divider cell to push it into, with coordinates

Across 15 runs the LLM completed ZERO handoffs; everything delivered was the
autonomous carriers' free work while the model emitted [2,2,5] and
[2,2,5,2,2,2,5] verbatim for turns on end — the same limit-cycle pathology the
production sk48 transcript shows.

That pilot CANNOT deliver the verdict, for one reason: the shipped model is
Qwen3.8-27B-FP8 and its weights are not local (the HF cache holds metadata
only). A 27B is markedly better at spatial grid manipulation than an 8B, so a
negative at 8B is ambiguous. This kernel runs the identical probe against the
identical served model the flights use, on the same GPU class.

THE PRE-REGISTERED READING (fixed before the run, do not move it):

  oracle arm clears >= 50% of valid clones
    -> the model CAN act on a correct model it did not build. The missing
       function is CONSTRUCTION / PERSISTENCE, which trained weights (TTT) or
       structural memory can supply. The general-agent lane is alive.

  oracle < 50%, guided < 50%
    -> it cannot execute a correct model even when handed one, with nothing to
       discover and nothing to remember. Then notes stores, digests, latch
       repairs, compaction AND behaviour fine-tunes are dead by construction,
       because every one of them routes through "the model will use
       information it has". Only code-selects-actions remains, and the
       crack/specialist lane is the whole remaining game.

  oracle < 50% but guided >= 50%
    -> the deficit is comprehension/planning, not execution: it can follow an
       explicit instruction but cannot derive one from stated rules. Structural
       memory stays alive; free-form rule text does not.

WHY wa30 LEVEL 3. A negative is only meaningful if the task is doable, the
oracle correct and the budget sufficient. Only here are all three established:
the mechanics are engine-verified by a sprite-by-sprite trace,
`wa30_carriersim.py` reproduces the carrier drive EXACTLY in lockstep with the
engine on L1-L4, and an 82-move solution is engine-verified on a clean replay
against the level's own 100-move budget. "No plan fits" is excluded.

THIS IS NOT A SUBMISSION. It is a normal GPU commit on the smoke kernel class:
zero submission slots, no scored parquet, no leaderboard effect.

Usage:
  .venv/bin/python submission/_oracle_probe/build_oracle_probe.py
  cd submission/_oracle_probe && python3 -m kaggle kernels push -p . \
      --accelerator NvidiaRtxPro6000
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
KERNEL_SLUG = "arc3-oracle-probe"

# Everything the probe needs to reach wa30 L3 exactly as the flights do.
BUNDLE_FILES = {
    "graft_explorer.py": SUB / "_explorer_floor" / "graft_explorer.py",
    "search_core.py": SUB / "_search_core" / "search_core.py",
    "specialists.py": SUB / "_search_core" / "specialists.py",
    "step_budgets.py": SUB / "_search_core" / "step_budgets.py",
    "wa30_macro.py": SUB / "_search_core" / "wa30_macro.py",
    "oracle_probe.py": HERE / "oracle_probe.py",
}

MARK_RUN = "run_context = contextlib.nullcontext()"

RUN_BLOCK_OLD = """    try:
        await bm.run(
            soft_end_time=soft_end,
            runtime_environment=target,
            minimal_diagnostics=run_as_submission,
        )
"""

RUN_BLOCK_NEW = '''    try:
        if true_submission:
            # A scored rerun must never enter the probe. Fail loudly instead of
            # silently playing nothing: this kernel is not a submission arm.
            raise RuntimeError("oracle-probe kernel must not be run as a scored submission")
        _probe_main()
'''

PROBE_CELL = r'''# ================= ORACLE-MODEL PROBE (the whole point of this kernel) =======
# Fail-fast GPU assert FIRST: metadata machine_shape + --accelerator alone can
# still bind a P100 (three wasted pushes on the serving lab proved it). Die
# here, before any setup cost, rather than produce a probe result on the wrong
# hardware and read it as if it were the flight class.
import subprocess as _sp

_gpu = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"], capture_output=True, text=True)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
assert "RTX" in (_gpu.stdout or "").upper() and "6000" in (_gpu.stdout or ""), (
    f"GPU misbind — expected RTX Pro 6000, got {_gpu.stdout!r} {_gpu.stderr!r}")

# Runs the identical three-arm probe as the local pilot, against the SERVED
# Qwen3.8-27B-FP8 through the analyzer's own OpenAI-compatible endpoint, so the
# model under test is byte-identical to the one the flights use.
import json as _pj
import os as _po
import sys as _ps
import time as _pt
import urllib.request as _pu
from pathlib import Path as _PP

_PROBE_DIR = WORKING_DIR / "probe_bundle"
_PROBE_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in _PROBE_BUNDLE.items():
    (_PROBE_DIR / _name).write_text(_src, encoding="utf-8")
if str(_PROBE_DIR) not in _ps.path:
    _ps.path.insert(0, str(_PROBE_DIR))

_po.environ["ONLY_RESET_LEVELS"] = "true"   # LAW: before arcengine imports


def _probe_env_dir():
    for _c in (
        _PP("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
        _PP("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
    ):
        if _c.is_dir():
            return _c
    for _hit in _PP("/kaggle/input").rglob("environment_files"):
        if _hit.is_dir():
            return _hit
    raise RuntimeError("environment_files not found")


def _probe_chat(messages, temperature):
    """Call the SERVED model, the same endpoint the analyzer uses."""
    _base = (_po.environ.get("LOCAL_ANALYZER_BASE_URL")
             or "http://127.0.0.1:1234/v1").rstrip("/")
    _model = (_po.environ.get("LOCAL_ANALYZER_MODEL")
              or "Qwen/Qwen3.8-27B-FP8")
    _body = _pj.dumps({
        "model": _model, "messages": messages, "temperature": temperature,
        "max_tokens": 512, "stream": False,
    }).encode()
    _req = _pu.Request(
        _base + "/chat/completions", data=_body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + (_po.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY")})
    with _pu.urlopen(_req, timeout=600) as _r:
        _d = _pj.loads(_r.read())
    return (_d["choices"][0]["message"].get("content") or "")


def _probe_main():
    import oracle_probe as OP

    OP.ask = lambda messages, model, temperature: _probe_chat(messages, temperature)
    OP.ROOT = str(_probe_env_dir().parent)   # oracle_probe joins ROOT/environment_files

    clones = int(_po.environ.get("ORACLE_PROBE_CLONES", "10"))
    arms = (_po.environ.get("ORACLE_PROBE_ARMS", "oracle,control,guided")).split(",")
    print(f"=== ORACLE-MODEL PROBE on the SERVED 27B ===", flush=True)
    print(f"endpoint={_po.environ.get('LOCAL_ANALYZER_BASE_URL')} "
          f"clones={clones} arms={arms}", flush=True)
    results = []
    for _arm in arms:
        print(f"--- arm {_arm} ---", flush=True)
        for _c in range(clones):
            try:
                _r = OP.one_clone(_c, _arm.strip(), "served-27b", 100, verbose=False)
            except Exception as _exc:
                _r = {"clone": _c, "arm": _arm.strip(), "error": f"{type(_exc).__name__}: {_exc}"}
            results.append(_r)
            print(f"  {_arm} clone {_c}: won={_r.get('won')} "
                  f"moves={_r.get('moves')} left={_r.get('blocks_left')}"
                  + (f" ERR {_r['error'][:90]}" if _r.get("error") else ""), flush=True)
    print("\n" + "=" * 60, flush=True)
    _summary = {}
    for _arm in arms:
        _a = [r for r in results if r["arm"] == _arm.strip()]
        _ok = [r for r in _a if not r.get("error")]
        _w = sum(1 for r in _ok if r.get("won"))
        _summary[_arm.strip()] = {"valid": len(_ok), "won": _w,
                                  "errored": len(_a) - len(_ok)}
        print(f"{_arm.strip():8} cleared {_w}/{len(_ok)} valid "
              f"({len(_a) - len(_ok)} errored, excluded)", flush=True)
    _o = _summary.get("oracle", {})
    if not _o.get("valid"):
        print("\nNO VALID ORACLE CLONES — instrument failure, NO VERDICT.", flush=True)
    else:
        _rate = _o["won"] / _o["valid"]
        print(f"\nPRE-REGISTERED BAR: oracle >= 50%.  RESULT: "
              + ("PASS — the model CAN act on a model it did not build"
                 if _rate >= 0.5 else
                 "FAIL — it cannot execute a correct handed-to-it model"), flush=True)
    (WORKING_DIR / "oracle_probe_results.json").write_text(
        _pj.dumps({"summary": _summary, "results": results}, indent=1),
        encoding="utf-8")
    print("wrote oracle_probe_results.json", flush=True)
# ============================================================================
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> int:
    nb = json.loads(BASE_NB.read_text())

    bundle = {name: path.read_text() for name, path in BUNDLE_FILES.items()}
    for name, path in BUNDLE_FILES.items():
        assert path.is_file(), f"missing bundle file {path}"
    head = ("# Files the probe needs, embedded so the kernel stays offline.\n"
            "_PROBE_BUNDLE = " + repr(bundle) + "\n")

    def idx_of(mark: str) -> int:
        for i, c in enumerate(nb["cells"]):
            if mark in "".join(c["source"]):
                return i
        raise SystemExit(f"marker not found: {mark!r}")

    run_idx = idx_of(MARK_RUN)
    run_src = "".join(nb["cells"][run_idx]["source"])
    assert RUN_BLOCK_OLD in run_src, "run block not found — base notebook drifted"
    run_src = run_src.replace(RUN_BLOCK_OLD, RUN_BLOCK_NEW)
    nb["cells"][run_idx]["source"] = run_src.splitlines(keepends=True)
    nb["cells"].insert(run_idx, code_cell(head + PROBE_CELL))

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
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
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")
    print(f"wrote {KERNEL_SLUG}.ipynb  cells={len(nb['cells'])}  "
          f"bytes={len(json.dumps(nb))}")
    print("push:  cd submission/_oracle_probe && python3 -m kaggle kernels "
          "push -p . --accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
