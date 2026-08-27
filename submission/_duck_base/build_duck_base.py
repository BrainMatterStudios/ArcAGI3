#!/usr/bin/env python3
"""Build submission/_duck_base/duck-base.ipynb — the RELIABLE floor: plain base duck, CPU-safe commit.

Transforms submission/_repro/duck-repro.ipynb (accessible taaf-src-hybrid bundle) with ONLY the
CPU-safe-commit fix (guard the vLLM serve + make the run-cell else a dummy parquet) — NO shadow,
NO EWM. The scored rerun is the plain duck the user has scored before (~1.26). Purpose: guarantee a
real score today + confirm the base submission flow works on the scored infra (isolating the shadow
as the cause of sub 54802596's ERROR).
"""
import json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
BASE = REPO / "submission/_repro/duck-repro.ipynb"
OUT = REPO / "submission/_duck_base"

nb = json.loads(BASE.read_text())
new_cells = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if c["cell_type"] == "code" and "setup_commands.json" in src:
        old = (
            'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
            '    print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
            '    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n'
            '    # Re-read in case the command persisted new env keys.\n'
            '    env = _command_env()\n'
            '    os.environ.update(env)\n')
        new = (
            'if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it\n'
            '    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
            '        print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
            '        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n'
            '        env = _command_env()\n'
            '        os.environ.update(env)\n'
            'else:\n'
            '    print("[duck] commit: skipping setup_commands (no GPU serve)", flush=True)\n')
        if old not in src:
            raise SystemExit("setup_commands loop did not match")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.replace(old, new).splitlines(keepends=True)})
        continue
    if c["cell_type"] == "code" and "await bm.run(" in src:
        old = (
            '    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    if not TRUE_SUBMISSION:\n'
            '        # An offline run isn\'t scored, but Kaggle still expects a submission.parquet output.\n'
            '        import pandas as pd\n'
            '\n'
            '        pd.DataFrame(\n'
            '            [["1_0", "1", True, 1]],\n'
            '            columns=["row_id", "game_id", "end_of_game", "score"],\n'
            '        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n')
        new = (
            '    if TRUE_SUBMISSION:\n'
            '        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    else:\n'
            '        import pandas as pd\n'
            '        pd.DataFrame([["1_0", "1", True, 1]],\n'
            '                     columns=["row_id", "game_id", "end_of_game", "score"]\n'
            '                     ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n'
            '        print("[duck] commit: CPU-safe landing; scored rerun runs the plain duck", flush=True)\n')
        if old not in src:
            raise SystemExit("run-cell block did not match")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.replace(old, new).splitlines(keepends=True)})
        continue
    new_cells.append(c)
nb["cells"] = new_cells

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "duck-base.ipynb").write_text(json.dumps(nb, indent=1))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "ahmedmobasher86/arc-agi-3-duck-base",
    "title": "arc-agi-3-duck-base",
    "code_file": "duck-base.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "ahmedmobasher86/taaf-src-hybrid",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [], "model_sources": [],
}, indent=2))
print(f"wrote {OUT/'duck-base.ipynb'} ({len(nb['cells'])} cells)")
