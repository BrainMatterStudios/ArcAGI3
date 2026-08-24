#!/usr/bin/env python3
"""build_judge_probe.py — Stage-3 serve probe for the bankruptcy-judge falsifier.

WHAT (task order 2026-08-24, from docs/RESEARCH-2026-08-23-searchcore-and-multirole.md
§B): boot the standard 27B serve chain (the duck38-v12 cells up to and including
boot attestation — no harness run), then serve-replay the frozen judge prompt
pack (submission/_bankruptcy_judge/falsifier/judge_prompts.jsonl, 38 cases x 3
samples, temperature 1.0, effort medium via chat_template_kwargs) with
run_stage3.py against localhost, score the pre-registered metrics against the
answer key (opened ONLY after all completions), run the judge graft's parser on
top for parse-rate, and print the verdict table.

ANSWER-KEY HYGIENE: answer_key.jsonl rides the notebook but is consumed only by
run_stage3.py's metric step after every completion has returned — no part of it
is ever placed in any model context.

PRE-REGISTERED KILL CRITERIA (task order):
  FLAG               PASS >= 70%   KILL < 50%
  RESCUE             PASS >= 40%   KILL < 25%
  CONTROL false-flag PASS <= 20%   KILL > 35%
  DIVERSITY          PASS >= 80%   (all-pairs Jaccard < 0.5 rate)

GPU envelope: serve setup ~30-35 min + 114 completions at --parallel 8 ~10-20
min => ~40-60 min total.

Usage:
  .venv/bin/python submission/_judge_probe/build_judge_probe.py
  cd submission/_judge_probe && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_NB = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
FALSIFIER = HERE.parent / "_bankruptcy_judge" / "falsifier"
GRAFT_PY = HERE.parent / "_bankruptcy_judge" / "graft_judge.py"
KERNEL_SLUG = "arc3-judge-probe"

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

STAGE3_CELL_TEMPLATE = '''# Stage-3 judge falsifier serve-replay (NO harness run in this kernel).
# 38 frozen cases x 3 samples, temp 1.0, reasoning_effort=medium via
# chat_template_kwargs — the pre-registered protocol. --parallel 8 is
# protocol-neutral (wall-clock only). The answer key is written to disk here
# but run_stage3.py opens it ONLY after all completions return; it never
# enters any model context.
FALSIFIER_DIR = WORKING_DIR / "judge_falsifier"
FALSIFIER_DIR.mkdir(parents=True, exist_ok=True)

_EMBEDDED_FILES = {{
    "judge_prompts.jsonl": {judge_prompts!r},
    "answer_key.jsonl": {answer_key!r},
    "run_stage3.py": {run_stage3!r},
    "graft_judge.py": {graft_judge!r},
}}
for _name, _content in _EMBEDDED_FILES.items():
    (FALSIFIER_DIR / _name).write_text(_content, encoding="utf-8")
print("falsifier files:", sorted(p.name for p in FALSIFIER_DIR.iterdir()))
_n_prompts = sum(1 for _ in open(FALSIFIER_DIR / "judge_prompts.jsonl"))
assert _n_prompts == 38, f"prompt pack must carry 38 cases, got {{_n_prompts}}"

_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
if not _base.endswith("/v1"):
    _base += "/v1"
_env = dict(os.environ)
_env["OPENAI_API_KEY"] = os.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY"
_cmd = [
    sys.executable, str(FALSIFIER_DIR / "run_stage3.py"),
    "--endpoint", _base,
    "--model", QWEN_SERVED_MODEL_NAME,
    "--samples", "3",
    "--temperature", "1.0",
    "--effort", "medium",
    "--effort-field", "chat_template_kwargs",
    "--max-tokens", "2048",
    "--parallel", "8",
]
print("stage3 cmd:", " ".join(_cmd), flush=True)
_t0 = time.time()
_rc = subprocess.run(_cmd, env=_env, cwd=str(FALSIFIER_DIR)).returncode
print(f"stage3 rc={{_rc}} wall={{time.time() - _t0:.0f}}s", flush=True)
assert _rc == 0, "run_stage3.py failed — no metrics to read"
'''

VERDICT_CELL = r'''# ---- Stage-3 verdict table (grep for JUDGE PROBE / METRIC / VERDICT) --------
sys.path.insert(0, str(FALSIFIER_DIR))
import graft_judge as _gj

_metrics = json.loads((FALSIFIER_DIR / "stage3_metrics.json").read_text())
_raw = [json.loads(l) for l in open(FALSIFIER_DIR / "stage3_raw.jsonl")]

# Judge-graft parser on top: the live graft only rebuilds on a parse that
# yields REJECT + a #N-cited contradiction + >=1 hypothesis. Score how much of
# the sampled behavior the graft's parser can actually harvest.
_parse = {
    "samples": len(_raw),
    "verdict_parsed": 0,
    "verdict_agrees_stage3": 0,
    "reject_samples": 0,
    "reject_cited": 0,
    "reject_cited_with_3_hyps": 0,
    "rebuild_eligible": 0,  # REJECT + cited + >=1 hypothesis (live rebuild rule)
}
for _row in _raw:
    _r = _gj.parse_judge_ruling(_row.get("text") or "")
    if _r["verdict"]:
        _parse["verdict_parsed"] += 1
        if _r["verdict"] == (_row.get("verdict") or ""):
            _parse["verdict_agrees_stage3"] += 1
    if _r["verdict"] == "REJECT":
        _parse["reject_samples"] += 1
        if _r["cited"]:
            _parse["reject_cited"] += 1
            if len(_r["hypotheses"]) >= 3:
                _parse["reject_cited_with_3_hyps"] += 1
        if _r["cited"] and _r["hypotheses"]:
            _parse["rebuild_eligible"] += 1
_parse["parse_rate"] = _parse["verdict_parsed"] / max(1, _parse["samples"])
_parse["rebuild_eligible_rate_of_rejects"] = (
    _parse["rebuild_eligible"] / _parse["reject_samples"] if _parse["reject_samples"] else None
)

def _bar(rate, pass_at, kill_at, *, higher_is_better=True):
    if rate is None:
        return "NO-DATA"
    if higher_is_better:
        if rate >= pass_at:
            return "PASS"
        if kill_at is not None and rate < kill_at:
            return "KILL"
        return "AMBER"
    if rate <= pass_at:
        return "PASS"
    if kill_at is not None and rate > kill_at:
        return "KILL"
    return "AMBER"

_flag = _metrics["FLAG"]["rate"]
_rescue = _metrics["RESCUE"]["rate"]
_control = _metrics["CONTROL_false_flag"]["rate"]
_diversity = _metrics["DIVERSITY"]["diverse_rate"]
_verdicts = {
    "FLAG": _bar(_flag, 0.70, 0.50),
    "RESCUE": _bar(_rescue, 0.40, 0.25),
    "CONTROL_false_flag": _bar(_control, 0.20, 0.35, higher_is_better=False),
    "DIVERSITY": _bar(_diversity, 0.80, None),
}
if any(v == "KILL" for v in _verdicts.values()):
    _overall = "KILL"
elif all(v == "PASS" for v in _verdicts.values()):
    _overall = "BUILD"
else:
    _overall = "ITERATE"

print("=" * 72)
print("JUDGE PROBE — STAGE-3 VERDICT TABLE (pre-registered bars)")
print(f"METRIC FLAG               {_flag if _flag is not None else 'n/a':<8} "
      f"({_metrics['FLAG']['num']}/{_metrics['FLAG']['den']})  bar >=0.70 kill <0.50  -> {_verdicts['FLAG']}")
print(f"METRIC RESCUE             {_rescue if _rescue is not None else 'n/a':<8} "
      f"({_metrics['RESCUE']['num']}/{_metrics['RESCUE']['den']})  bar >=0.40 kill <0.25  -> {_verdicts['RESCUE']}")
print(f"METRIC CONTROL_false_flag {_control if _control is not None else 'n/a':<8} "
      f"({_metrics['CONTROL_false_flag']['num']}/{_metrics['CONTROL_false_flag']['den']})  bar <=0.20 kill >0.35  -> {_verdicts['CONTROL_false_flag']}")
print(f"METRIC DIVERSITY          {_diversity if _diversity is not None else 'n/a':<8} "
      f"({_metrics['DIVERSITY']['all_pairs_below_0.5']}/{_metrics['DIVERSITY']['reject_samples']})  bar >=0.80             -> {_verdicts['DIVERSITY']}")
print(f"METRIC all_keep_false_flag {_metrics['all_keep_false_flag']['rate']} "
      f"({_metrics['all_keep_false_flag']['num']}/{_metrics['all_keep_false_flag']['den']})  [reported, no bar]")
print(f"METRIC graft_parse_rate   {_parse['parse_rate']:.3f} "
      f"({_parse['verdict_parsed']}/{_parse['samples']} samples; "
      f"agree with stage3 parser {_parse['verdict_agrees_stage3']})")
print(f"METRIC rebuild_eligible   {_parse['rebuild_eligible']}/{_parse['reject_samples']} REJECT samples "
      f"(cited + >=1 hypothesis; cited+3hyp {_parse['reject_cited_with_3_hyps']})")
print(f"missed deserving cases: {_metrics['missed_deserving']}")
print(f"control flagged: {_metrics['CONTROL_false_flag']['flagged']}")
print(f"VERDICT OVERALL: {_overall}")

_results = {
    "probe": "arc3-judge-probe",
    "protocol": {"samples": 3, "temperature": 1.0, "effort": "medium",
                 "effort_field": "chat_template_kwargs", "max_tokens": 2048,
                 "parallel": 8, "n_cases": _metrics["n_cases"]},
    "metrics": _metrics,
    "graft_parser": _parse,
    "bars": _verdicts,
    "overall": _overall,
}
(WORKING_DIR / "stage3_results.json").write_text(json.dumps(_results, indent=1), encoding="utf-8")
# keep the raw samples + metrics in the kernel output too
import shutil
shutil.copy2(FALSIFIER_DIR / "stage3_raw.jsonl", WORKING_DIR / "stage3_raw.jsonl")
shutil.copy2(FALSIFIER_DIR / "stage3_metrics.json", WORKING_DIR / "stage3_metrics.json")
print("wrote", WORKING_DIR / "stage3_results.json")
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

    # 3. Stage-3 replay + verdict table after the attest cell.
    payloads = {
        "judge_prompts": (FALSIFIER / "judge_prompts.jsonl").read_text(),
        "answer_key": (FALSIFIER / "answer_key.jsonl").read_text(),
        "run_stage3": (FALSIFIER / "run_stage3.py").read_text(),
        "graft_judge": GRAFT_PY.read_text(),
    }
    assert "--parallel" in payloads["run_stage3"], "need the parallel-capable run_stage3"
    assert "def parse_judge_ruling" in payloads["graft_judge"]
    assert payloads["answer_key"].startswith('{"_warning": "ANSWER KEY')
    stage3_text = STAGE3_CELL_TEMPLATE.format(**payloads)
    # Embedded sources must ride byte-identically (repr round-trips by language
    # guarantee — same technique as the effort-smoke builder).
    for source in payloads.values():
        assert repr(source) in stage3_text
    nb["cells"].append(code_cell(stage3_text))
    nb["cells"].append(code_cell(VERDICT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("stage3 cmd:")
    assert joined.index("stage3 cmd:") < joined.index("JUDGE PROBE — STAGE-3 VERDICT TABLE")
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
                "enable_internet": False,
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
