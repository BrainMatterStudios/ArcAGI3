#!/usr/bin/env python3
"""build_oracle_probe_flashnext.py — the ORACLE-MODEL PROBE on the SERVED Flash-Next (2026-09-12).

Rebuilds the 08-29 probe kernel on the FLOWN base (submission/_keith_copy/push_bothmounts:
keith V14 byte copy, Qwen3.8-Flash-Next NVFP4, kv5-bf16-mtp3-c8-cg32) instead of the retired
27B duck38-v12 notebook, with the request path fixed:

  * the 08-29 run passed its short PREFLIGHT and still made 0 moves in 15/15 clones: it sent
    chat_template_kwargs {"preserve_thinking": False}, which only controls whether PAST
    reasoning is kept in the history. The flag that turns thinking OFF in the Flash-Next
    template is `enable_thinking: false` (chat_template.jinja:165-166). Game-length prompts
    produced long reasoning that exhausted max_tokens before any ACTIONS line.
  * this build sends {"enable_thinking": False, "preserve_thinking": False}, max_tokens 3000,
    preflights with a GAME-LENGTH prompt and asserts finish_reason == "stop", and the probe
    itself now aborts a clone as INVALID after three unparsable replies (oracle_probe.py).

Reading (pre-registered, unchanged): oracle >= 50 % of VALID clones = the brain can act on a
correct model it did not build; guided is the positive control and MUST win, else the
instrument is still broken and there is no read.

Usage:
  python3 submission/_oracle_probe/build_oracle_probe_flashnext.py
  kaggle kernels push -p submission/_oracle_probe/push_flashnext
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BASE_DIR = SUB / "_keith_copy" / "push_bothmounts"
BASE_NB = BASE_DIR / "arc3-keith-copy.ipynb"
OUT_DIR = HERE / "push_flashnext"
KERNEL_SLUG = "arc3-oracle-probe-fn"

BUNDLE_FILES = {
    "graft_explorer.py": SUB / "_explorer_floor" / "graft_explorer.py",
    "search_core.py": SUB / "_search_core" / "search_core.py",
    "specialists.py": SUB / "_search_core" / "specialists.py",
    "step_budgets.py": SUB / "_search_core" / "step_budgets.py",
    "wa30_macro.py": SUB / "_search_core" / "wa30_macro.py",
    "oracle_probe.py": HERE / "oracle_probe.py",
}

MARK_RUN = "PUBLIC_GAME_IDS = tuple(["
RUN_BLOCK_START = "try:\n    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)\n"
RUN_BLOCK_END = "\nfinally:\n    try:\n        vllm_watchdog.stop_background"
RUN_BLOCK_NEW = '''try:
    if TRUE_SUBMISSION:
        # A scored rerun must never enter the probe. Fail loudly instead of
        # silently playing nothing: this kernel is not a submission arm.
        raise RuntimeError("oracle-probe kernel must not be run as a scored submission")
    _probe_main()
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet(WORKING_DIR / "submission.parquet", index=False)'''

PROBE_CELL = r'''# ================= ORACLE-MODEL PROBE on the SERVED Flash-Next (2026-09-12) =====
import subprocess as _sp

_gpu = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"], capture_output=True, text=True)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
assert "RTX" in (_gpu.stdout or "").upper() and "6000" in (_gpu.stdout or ""), (
    f"GPU misbind — expected RTX Pro 6000, got {_gpu.stdout!r} {_gpu.stderr!r}")

import json as _pj
import os as _po
import sys as _ps
import urllib.request as _pu
from pathlib import Path as _PP

_PROBE_DIR = WORKING_DIR / "probe_bundle"
_PROBE_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in _PROBE_BUNDLE.items():
    (_PROBE_DIR / _name).write_text(_src, encoding="utf-8")
if str(_PROBE_DIR) not in _ps.path:
    _ps.path.insert(0, str(_PROBE_DIR))

_po.environ["ONLY_RESET_LEVELS"] = "true"   # LAW: before arcengine imports
_po.environ.setdefault("ORACLE_PROBE_CLONES", "5")


def _probe_env_dir():
    _root = _po.environ.get("ARC3_COMP_ROOT")
    if _root and (_PP(_root) / "environment_files").is_dir():
        return _PP(_root) / "environment_files"
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


def _setup_env():
    try:
        return _pj.loads(_PP(SETUP_ENV_PATH).read_text())
    except Exception:  # noqa: BLE001
        return {}


_SENV = _setup_env()
_BASE = (_SENV.get("LOCAL_ANALYZER_BASE_URL") or _po.environ.get("LOCAL_ANALYZER_BASE_URL")
         or "http://127.0.0.1:1234/v1").rstrip("/")
_MODEL = (_SENV.get("LOCAL_ANALYZER_MODEL_ID") or _po.environ.get("LOCAL_ANALYZER_MODEL_ID")
          or "Qwen/Qwen3.8-Flash-Next-NVFP4")
_KEY = (_SENV.get("LOCAL_ANALYZER_API_KEY") or _po.environ.get("LOCAL_ANALYZER_API_KEY")
        or _SENV.get("OPENAI_API_KEY") or "EMPTY")
_FINISH = {}


def _probe_chat(messages, temperature):
    """Call the SERVED model through the analyzer's own endpoint, thinking OFF.

    08-29 lesson: `preserve_thinking` only keeps or drops PAST reasoning; the flag that
    disables thinking in this template is `enable_thinking: false`. Both are sent, and
    max_tokens leaves headroom so the ACTIONS line lands even if thinking sneaks in.
    """
    _body = _pj.dumps({
        "model": _MODEL, "messages": messages, "temperature": temperature,
        "max_tokens": 3000, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }).encode()
    _req = _pu.Request(_BASE + "/chat/completions", data=_body,
                       headers={"Content-Type": "application/json", "Authorization": "Bearer " + _KEY})
    with _pu.urlopen(_req, timeout=900) as _r:
        _d = _pj.loads(_r.read())
    _ch = _d["choices"][0]
    _fr = str(_ch.get("finish_reason"))
    _FINISH[_fr] = _FINISH.get(_fr, 0) + 1
    _m = _ch["message"]
    return ((_m.get("content") or "") + "\n" + (_m.get("reasoning_content") or ""))


def _probe_main():
    import oracle_probe as OP

    OP.ask = lambda messages, model, temperature: _probe_chat(messages, temperature)
    OP.ROOT = str(_probe_env_dir().parent)   # oracle_probe joins ROOT/environment_files
    print(f"probe endpoint={_BASE} model={_MODEL}", flush=True)

    # PREFLIGHT 1 — short prompt (the 08-29 check; necessary, not sufficient).
    _pf = _probe_chat([
        {"role": "system", "content": "Reply with a short sentence, then a "
         "final line of the exact form:\nACTIONS: n, n, n\nusing numbers 1..5."},
        {"role": "user", "content": "Say hello and give three actions."}], 0.2)
    _acts = OP.parse_actions(_pf)
    print(f"PREFLIGHT-1 parsed={_acts} from {len(_pf)} chars finish={_FINISH}", flush=True)
    # PREFLIGHT 2 — a GAME-LENGTH prompt: the oracle mechanics as system, a 16x16 board as
    # user. This is the shape that silently drowned in reasoning on 08-29.
    _board = "\n".join(" ".join(str((r * 7 + c) % 4) for c in range(16)) for r in range(16))
    _pf2 = _probe_chat([
        {"role": "system", "content": OP.ORACLE_WA30 + "\n\nEach turn, reply with a short justification and then a final line "
         "of the exact form:\nACTIONS: n, n, n\nusing 1-8 action numbers from 1..5. Nothing after that line."},
        {"role": "user", "content": "BOARD (16x16, digits are colours):\n" + _board
         + "\n\nYou are at (2,3) facing RIGHT. Blocks at (5,3) and (9,10). Divider column x=8. "
           "Moves used: 0/100. Give your next actions."}], 0.2)
    _acts2 = OP.parse_actions(_pf2)
    print(f"PREFLIGHT-2 parsed={_acts2} from {len(_pf2)} chars finish={_FINISH}", flush=True)
    if not _acts or not _acts2 or _FINISH.get("length"):
        print("PREFLIGHT RAW REPLY 2 (first 1500 chars):\n" + _pf2[:1500], flush=True)
        raise RuntimeError(
            "PREFLIGHT FAILED: no parsable ACTIONS line on a game-length prompt, or a "
            "length-capped reply. Aborting BEFORE spending the GPU. Fix the request "
            "(chat_template_kwargs / max_tokens) and re-push.")

    clones = int(_po.environ.get("ORACLE_PROBE_CLONES", "5"))
    arms = (_po.environ.get("ORACLE_PROBE_ARMS", "guided,oracle,control")).split(",")
    print(f"=== ORACLE-MODEL PROBE on the SERVED {_MODEL} ===", flush=True)
    print(f"endpoint={_BASE} clones={clones} arms={arms}", flush=True)
    results = []
    for _arm in arms:
        print(f"--- arm {_arm} ---", flush=True)
        for _c in range(clones):
            try:
                _r = OP.one_clone(_c, _arm.strip(), _MODEL, 100, verbose=True)
            except Exception as _exc:
                _r = {"clone": _c, "arm": _arm.strip(), "error": f"{type(_exc).__name__}: {_exc}"}
            results.append(_r)
            print(f"  {_arm} clone {_c}: won={_r.get('won')} moves={_r.get('moves')} "
                  f"turns={_r.get('turns')} left={_r.get('blocks_left')} finish={dict(_FINISH)}"
                  + (f" ERR {_r['error'][:90]}" if _r.get("error") else ""), flush=True)
            if _r.get("unparsed_sample"):
                print("    unparsed sample: " + repr(_r["unparsed_sample"][:400]), flush=True)
    print("\n" + "=" * 60, flush=True)
    _summary = {}
    for _arm in arms:
        _a = [r for r in results if r["arm"] == _arm.strip()]
        _ok = [r for r in _a if OP.clone_is_valid(r)]
        _w = sum(1 for r in _ok if r.get("won"))
        _summary[_arm.strip()] = {"valid": len(_ok), "won": _w, "invalid": len(_a) - len(_ok)}
        print(f"{_arm.strip():8} cleared {_w}/{len(_ok)} valid ({len(_a) - len(_ok)} invalid, excluded)", flush=True)
    _g = _summary.get("guided", {})
    _o = _summary.get("oracle", {})
    if not _g.get("valid") or not _o.get("valid"):
        print("\nNO VALID CLONES in guided or oracle — instrument failure, NO VERDICT.", flush=True)
    elif _g["won"] / _g["valid"] < 0.5:
        print("\nGUIDED (positive control) < 50 % — the action channel is still not a read; NO VERDICT.", flush=True)
    else:
        _rate = _o["won"] / _o["valid"]
        print(f"\nPRE-REGISTERED BAR: oracle >= 50 %.  RESULT: "
              + ("PASS — the model CAN act on a model it did not build"
                 if _rate >= 0.5 else
                 "FAIL — it cannot execute a correct handed-to-it model"), flush=True)
    (WORKING_DIR / "oracle_probe_results.json").write_text(
        _pj.dumps({"summary": _summary, "finish_reasons": _FINISH, "results": results}, indent=1),
        encoding="utf-8")
    print("wrote oracle_probe_results.json", flush=True)
# ============================================================================
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> int:
    nb = json.loads(BASE_NB.read_text())
    for name, path in BUNDLE_FILES.items():
        assert path.is_file(), f"missing bundle file {path}"
    bundle = {name: path.read_text() for name, path in BUNDLE_FILES.items()}
    head = ("# Files the probe needs, embedded so the kernel stays offline.\n"
            "_PROBE_BUNDLE = " + repr(bundle) + "\n")

    def idx_of(mark: str) -> int:
        for i, c in enumerate(nb["cells"]):
            if mark in "".join(c["source"]):
                return i
        raise SystemExit(f"marker not found: {mark!r}")

    run_idx = idx_of(MARK_RUN)
    run_src = "".join(nb["cells"][run_idx]["source"])
    a, b = run_src.find(RUN_BLOCK_START), run_src.find(RUN_BLOCK_END)
    assert a >= 0 and b > a, "run block not found — base notebook drifted"
    run_src = run_src[:a] + RUN_BLOCK_NEW + run_src[b:]
    nb["cells"][run_idx]["source"] = run_src.splitlines(keepends=True)
    nb["cells"].insert(run_idx, code_cell(head + PROBE_CELL))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    meta = json.loads((BASE_DIR / "kernel-metadata.json").read_text())
    meta.update({"id": f"ahmedmobasher86/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": f"{KERNEL_SLUG}.ipynb"})
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT_DIR / (KERNEL_SLUG + '.ipynb')}  cells={len(nb['cells'])}  bytes={len(json.dumps(nb))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
