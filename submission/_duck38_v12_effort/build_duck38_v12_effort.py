#!/usr/bin/env python3
"""build_duck38_v12_effort.py — LADDER ARM #1 (2026-08-24 slot): v12 + effort_medium.

ARM: the duck38-v12 base (Aug-07 community anim bundle + Qwen3.8-27B-FP8
repacked, boot-attested, flown live at 1.55) plus exactly ONE validated
change: submission/_effort_medium/graft_effort.py installed with

    EFFORT_MEDIUM=1      reasoning_effort=medium rides chat_template_kwargs on
                         every vllm request (the official chat template
                         silently defaults reasoning_effort to xhigh;
                         docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-1 #2)
    EFFORT_DEAD_RETRY=0  dead-completion retry hardening OFF —
                         single-variable purity, matching the smoke exactly

SMOKE EVIDENCE (arc3-effort-smoke commit, COMPLETE, RTX Pro 6000, pulled and
read 2026-08-23): dead completions 0/212 requests (corpus baseline 3.5% =
122/~3,480); 217/217 payloads carried reasoning_effort=medium; levels held
(vc33 2, tn36 1 under the tighter 50-min per-game smoke cap). Offline: 13/13
graft tests green against the exact mounted bundle (tool_agent.py md5
6a0e3dfd963e59d14f0a57d121c578eb).

CONSTRUCTION: the tracked v12 notebook (submission/_duck38_v12/
arc3-duck38-v12.ipynb, code-cell sha256 dc2c36f8…, the bytes that flew 1.55)
is replicated EXACTLY — boot attestation cell, commit-smoke hook (offline
3-game 60-min box; the scored KAGGLE_IS_COMPETITION_RERUN path never enters
it), TRUE_SUBMISSION wiring — with ONE inserted cell immediately before the
run cell: the graft source embedded inline (xd cell-10 pattern) + install()
with a HARD assert. Rationale for the hard gate (vs xd's fail-open): a
scored run that silently plays stock v12 poisons the single-variable ladder
read, while an ERROR costs no slot (2026-08-01 audit law). Install is
deterministic on this bundle — the smoke commit executed the identical path
and passed the identical assert.

ENVELOPE: the graft cannot extend run duration (it only changes a request
template kwarg); RESULTS-2026-08-22 envelope law holds.

Usage:
  python3 submission/_duck38_v12_effort/build_duck38_v12_effort.py
  cd submission/_duck38_v12_effort && \
    python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_NB = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
GRAFT_PY = HERE.parent / "_effort_medium" / "graft_effort.py"
KERNEL_SLUG = "arc3-duck38-v12-effort"

MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"
MARK_SMOKE = 'SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]'
BASE_CODE_SHA256 = "dc2c36f897428123b2b9e95d85470b57cf6d954910599be3908b0a38a8199615"

GRAFT_CELL_TEMPLATE = '''# LADDER ARM cell (the ONE delta vs flown v12): effort_medium graft install.
# EFFORT_MEDIUM=1: reasoning_effort=medium is setdefault-ed into
# chat_template_kwargs on every vllm request — the official template's silent
# default is xhigh. EFFORT_DEAD_RETRY=0: the paired retry hardening stays OFF
# for single-variable purity (matches the validated smoke exactly).
# Smoke evidence (arc3-effort-smoke, RTX Pro 6000): 0/212 dead completions vs
# 3.5% corpus baseline; 217/217 payloads carried effort=medium; levels held.
# HARD GATE rationale: a scored run silently playing stock v12 poisons the
# ladder read; an ERROR costs no slot. Install is deterministic on this
# bundle — the smoke commit passed this exact assert.
os.environ["EFFORT_MEDIUM"] = "1"
os.environ["EFFORT_DEAD_RETRY"] = "0"

_GRAFT_SOURCE = {graft_source!r}

import importlib.util as _ilu

_graft_path = WORKING_DIR / "graft_effort.py"
_graft_path.write_text(_GRAFT_SOURCE, encoding="utf-8")
_graft_spec = _ilu.spec_from_file_location("graft_effort", _graft_path)
_graft = _ilu.module_from_spec(_graft_spec)
sys.modules["graft_effort"] = _graft
_graft_spec.loader.exec_module(_graft)
_graft_status = _graft.install()
print("[effort]", _graft_status)
assert _graft_status == "effort_medium: OK", (
    f"single-variable arm requires the graft live, got: {{_graft_status}}"
)
'''


def main() -> None:
    import hashlib

    nb = json.loads(BASE_NB.read_text())
    base_code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert hashlib.sha256(base_code.encode()).hexdigest() == BASE_CODE_SHA256, (
        "base v12 notebook drifted from the flown-1.55 bytes — re-attest before building"
    )

    graft_source = GRAFT_PY.read_text()
    assert "def install() -> str:" in graft_source

    run_hits = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and MARK_RUN in "".join(c["source"])
    ]
    assert len(run_hits) == 1, run_hits
    run_idx = run_hits[0]

    graft_cell_text = GRAFT_CELL_TEMPLATE.format(graft_source=graft_source)
    # The graft source must ride the notebook byte-identically: embedded via
    # !r, and Python str repr round-trips by language guarantee.
    assert repr(graft_source) in graft_cell_text
    nb["cells"].insert(
        run_idx,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": graft_cell_text.splitlines(keepends=True),
        },
    )

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined and MARK_SMOKE in joined
    assert joined.index(MARK_ATTEST) < joined.index(MARK_SMOKE)
    assert joined.index(MARK_SMOKE) < joined.index('os.environ["EFFORT_MEDIUM"]')
    assert joined.index('os.environ["EFFORT_MEDIUM"]') < joined.index(MARK_RUN)
    assert joined.count("EFFORT_DEAD_RETRY") >= 2  # cell flag set + graft source
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined  # TRUE_SUBMISSION path intact
    assert len(nb["cells"]) == 12  # 11 v12 cells + the one graft cell

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
                # LAW (serving-lab, 3 wasted pushes): the competition source is
                # what gates the RTX Pro 6000 pool — machine_shape +
                # --accelerator alone bind P100.
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "model_sources": [
                    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
                ],
                "enable_tpu": False,
                "keywords": ["gpu"],
            },
            indent=2,
        )
        + "\n"
    )

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
