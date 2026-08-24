#!/usr/bin/env python3
"""build_judge_probe2.py — ROUND-2 serve probe for the bankruptcy-judge falsifier.

Clone of submission/_judge_probe/build_judge_probe.py (the proven serve+battery
kernel): boot the standard 27B serve chain (duck38-v12 cells up to and including
boot attestation — no harness run), then serve-replay the frozen ROUND-2 prompt
pack (submission/_bankruptcy_judge/falsifier/round2/round2_prompts.jsonl,
81 cases x 3 samples = 243 completions) with run_round2.py against localhost.

Arms + PRE-REGISTERED bars (see round2/build_round2.py header):
  V1 MENU_PICK        pass >= 60%, kill < 40%  (27 scored cases, chance 25%)
  V2 PRED_DIVERGENCE  pass >= 60%              (10 cases)
  V3 FLAG_RICH        revived >= 3/6, dead <= 1/6
  R1X (diagnostic)    FLAG >= 50% => round-1 kill was a truncation artifact

Instrument fix vs round 1: max_tokens 8192 (round 1's 2048 left 87/114 samples
EMPTY — reasoning consumed the budget before any verdict was emitted).

ANSWER-KEY HYGIENE: round2_key.jsonl rides the notebook but is consumed only by
run_round2.py's metric step after every completion has returned — no part of it
is ever placed in any model context.

GPU envelope: serve setup ~30-35 min + 243 completions at --parallel 8 at up to
8192 output tokens => ~40-70 min replay => ~1.5-2h total.

Usage:
  python3 submission/_judge_probe2/build_judge_probe2.py
  cd submission/_judge_probe2 && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_NB = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
ROUND2 = HERE.parent / "_bankruptcy_judge" / "falsifier" / "round2"
KERNEL_SLUG = "arc3-judge-probe2"

MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_ATTEST = "attest: OK"

GPU_ASSERT_CELL = r'''# Fail-fast GPU assert: metadata machine_shape + --accelerator alone can still
# bind P100 (3 wasted pushes on serving-lab proved it; the competition source
# attachment is the real RTX Pro 6000 gate). Die here, before any setup cost.
import subprocess as _sp

_gpu = _sp.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
_gpu_name = (_gpu.stdout or "").upper()
assert "RTX" in _gpu_name and "6000" in _gpu_name, (
    f"GPU misbind — expected RTX Pro 6000, got: {_gpu.stdout!r} {_gpu.stderr!r}"
)
'''

ROUND2_CELL_TEMPLATE = '''# Round-2 judge falsifier serve-replay (NO harness run in this kernel).
# 81 frozen cases (V1 27 menu + V2 10 prediction + V3 6 rich-evidence + R1X 38
# round-1 replay) x 3 samples, temp 1.0, reasoning_effort=medium via
# chat_template_kwargs, max_tokens 8192 (the round-1 truncation fix).
# The answer key is written to disk here but run_round2.py opens it ONLY after
# all completions return; it never enters any model context.
ROUND2_DIR = WORKING_DIR / "judge_round2"
ROUND2_DIR.mkdir(parents=True, exist_ok=True)

_EMBEDDED_FILES = {{
    "round2_prompts.jsonl": {round2_prompts!r},
    "round2_key.jsonl": {round2_key!r},
    "run_round2.py": {run_round2!r},
}}
for _name, _content in _EMBEDDED_FILES.items():
    (ROUND2_DIR / _name).write_text(_content, encoding="utf-8")
print("round2 files:", sorted(p.name for p in ROUND2_DIR.iterdir()))
_n_prompts = sum(1 for _ in open(ROUND2_DIR / "round2_prompts.jsonl"))
assert _n_prompts == 81, f"round-2 pack must carry 81 cases, got {{_n_prompts}}"

_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
if not _base.endswith("/v1"):
    _base += "/v1"
_env = dict(os.environ)
_env["OPENAI_API_KEY"] = os.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY"
_cmd = [
    sys.executable, str(ROUND2_DIR / "run_round2.py"),
    "--endpoint", _base,
    "--model", QWEN_SERVED_MODEL_NAME,
    "--samples", "3",
    "--temperature", "1.0",
    "--effort", "medium",
    "--effort-field", "chat_template_kwargs",
    "--max-tokens", "8192",
    "--parallel", "8",
]
print("round2 cmd:", " ".join(_cmd), flush=True)
_t0 = time.time()
_rc = subprocess.run(_cmd, env=_env, cwd=str(ROUND2_DIR)).returncode
print(f"round2 rc={{_rc}} wall={{time.time() - _t0:.0f}}s", flush=True)
assert _rc == 0, "run_round2.py failed — no metrics to read"
'''

VERDICT_CELL = r'''# ---- Round-2 verdict + disposition (grep for JUDGE ROUND-2 / DISPOSITION) ----
_res = json.loads((ROUND2_DIR / "round2_results.json").read_text())

_v1 = _res["V1_MENU_PICK"]["verdict"]
_v2 = _res["V2_PRED_DIVERGENCE"]["verdict"]
_v3 = _res["V3_FLAG_RICH"]["verdict"]
_r1x = _res["R1X"]["verdict"]

print("=" * 72)
print("JUDGE ROUND-2 DISPOSITION (persona track)")
print(f"  menu-based rebuild viable:        {_v1} (V1 MENU_PICK "
      f"{_res['V1_MENU_PICK']['num']}/{_res['V1_MENU_PICK']['den']})")
print(f"  prediction-based bankruptcy:      {_v2} (V2 PRED_DIVERGENCE "
      f"{_res['V2_PRED_DIVERGENCE']['num']}/{_res['V2_PRED_DIVERGENCE']['den']})")
print(f"  generative form w/ rich evidence: {_v3} (V3 FLAG_RICH "
      f"{_res['V3_FLAG_RICH']['flips']}/{_res['V3_FLAG_RICH']['den']})")
print(f"  round-1 kill attribution:         {_r1x} (R1X FLAG "
      f"{_res['R1X']['FLAG']['num']}/{_res['R1X']['FLAG']['den']}, "
      f"RESCUE {_res['R1X']['RESCUE']['num']}/{_res['R1X']['RESCUE']['den']}, "
      f"CTRL-FF {_res['R1X']['CONTROL_false_flag']['num']}/"
      f"{_res['R1X']['CONTROL_false_flag']['den']})")
_alive = [n for n, v in [("V1-menu", _v1), ("V2-prediction", _v2)]
          if v == "PASS"] + (["V3-generative-rich"] if _v3 == "REVIVED" else [])
print(f"  surviving variants: {_alive if _alive else 'NONE — track stays dead'}")

import shutil
shutil.copy2(ROUND2_DIR / "round2_raw.jsonl", WORKING_DIR / "round2_raw.jsonl")
shutil.copy2(ROUND2_DIR / "round2_results.json", WORKING_DIR / "round2_results.json")
print("wrote", WORKING_DIR / "round2_results.json")
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())

    def idx_of(marker: str) -> int:
        hits = [
            i
            for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        ]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    # 1. Keep only the serve chain: everything up to and including boot attest.
    attest_idx = idx_of(MARK_ATTEST)
    nb["cells"] = nb["cells"][: attest_idx + 1]

    # 2. GPU fail-fast right after the imports cell.
    nb["cells"].insert(idx_of(MARK_IMPORTS) + 1, code_cell(GPU_ASSERT_CELL))

    # 3. Round-2 replay + verdict/disposition after the attest cell.
    payloads = {
        "round2_prompts": (ROUND2 / "round2_prompts.jsonl").read_text(),
        "round2_key": (ROUND2 / "round2_key.jsonl").read_text(),
        "run_round2": (ROUND2 / "run_round2.py").read_text(),
    }
    assert payloads["round2_key"].startswith('{"_warning": "ROUND-2 ANSWER KEY')
    assert payloads["round2_prompts"].count("\n") == 81
    assert "--parallel" in payloads["run_round2"]
    cell_text = ROUND2_CELL_TEMPLATE.format(**payloads)
    # Embedded sources must ride byte-identically (repr round-trips by language
    # guarantee — same technique as the judge-probe / effort-smoke builders).
    for source in payloads.values():
        assert repr(source) in cell_text
    nb["cells"].append(code_cell(cell_text))
    nb["cells"].append(code_cell(VERDICT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("round2 cmd:")
    assert joined.index("round2 cmd:") < joined.index("JUDGE ROUND-2 DISPOSITION")
    assert "run_context" not in joined, "harness run cells must be gone"
    assert "bm.games" not in joined, "smoke hook must be gone"

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(
        json.dumps(
            {
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
                # LAW: the competition source is what gates the RTX Pro 6000
                # pool — machine_shape + --accelerator alone bind P100.
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "model_sources": [
                    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
                ],
            },
            indent=2,
        )
        + "\n"
    )

    import hashlib

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    print("built", KERNEL_SLUG, "cells", len(nb["cells"]),
          "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
