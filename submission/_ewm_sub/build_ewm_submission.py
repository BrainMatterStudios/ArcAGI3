#!/usr/bin/env python3
"""Build submission/_ewm_sub/ewm-submission.ipynb — the EWM (verification-by-execution) probe.

Rebased on the SAME proven base as the (already-committed) duck-shadow: submission/_repro/duck-repro.ipynb
(uses the accessible ahmedmobasher86/taaf-src-hybrid bundle; the old _finetuned base used a now-404
jeroencottaar dataset). ONE functional change vs the duck: bm.solver = SolverEWM(...) so each hidden
game is played by the local Qwen driving the verification-by-execution loop. Commit is CPU-safe (P100);
the scored rerun serves Qwen on the eval RTX 6000. Embeds the OFFLINE-validated EWM scaffold.

Run: python submission/_ewm_sub/build_ewm_submission.py
"""
import base64, json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
BASE = REPO / "submission/_repro/duck-repro.ipynb"
SCAF = REPO / "scratchpad/ewm_pathb"
OUT = REPO / "submission/_ewm_sub"

# ---- embed the validated scaffold at the layout ewm_solver.py expects ----
files = {}
def add(rel, path): files[rel] = path.read_bytes()
add("ewm_solver.py", SCAF / "ewm_solver.py")
add("ewm_agent.py", SCAF / "ewm_agent.py")
add("score_run.py", SCAF / "score_run.py")
add("src/server/server_taaf.py", SCAF / "src/server/server_taaf.py")
for p in (SCAF / "src/agent/workspace_init").rglob("*"):
    if p.is_file() and "__pycache__" not in str(p):
        files["src/agent/workspace_init/" + str(p.relative_to(SCAF / "src/agent/workspace_init"))] = p.read_bytes()
BLOB = {k: base64.b64encode(v).decode() for k, v in files.items()}

EMBED = (
    "import base64, json, os, sys\n"
    "from pathlib import Path\n"
    "EWM = Path('/kaggle/working/ewm'); EWM.mkdir(parents=True, exist_ok=True)\n"
    "for rel, b64 in json.loads(%r).items():\n"
    "    p = EWM / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(b64))\n"
    "for entry in (str(EWM), str(EWM / 'src' / 'server')):\n"
    "    if entry not in sys.path: sys.path.insert(0, entry)\n"
    "print('[ewm] scaffold unpacked:', len(json.loads(%r)), 'files ->', EWM)\n"
) % (json.dumps(BLOB), json.dumps(BLOB))

nb = json.loads(BASE.read_text())
new_cells = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    # (a) guard the vLLM serve to the scored rerun (CPU-safe commit on P100)
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
            '    print("[ewm] commit: skipping setup_commands (no GPU serve)", flush=True)\n')
        if old not in src:
            raise SystemExit("setup_commands loop did not match")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.replace(old, new).splitlines(keepends=True)})
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": EMBED.splitlines(keepends=True)})
        continue
    # (b) swap the solver to SolverEWM + CPU-safe commit landing
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
            '        PER_GAME_SECONDS = float(os.environ.get("EWM_PER_GAME_SECONDS", "290"))  # ~110 games in 9h\n'
            '        from ewm_solver import SolverEWM\n'
            '        bm.solver = SolverEWM(base_url="http://127.0.0.1:1234/v1", model="vrfai/Qwen3.6-27B-FP8",\n'
            '                              per_game_seconds=PER_GAME_SECONDS, max_turns=160, max_ctx_chars=110000,\n'
            '                              obs_cap=4600, max_tokens=4096, work_root="/kaggle/working/ewm_runs")\n'
            '        print(f"[ewm] solver=SolverEWM per_game={PER_GAME_SECONDS}s", flush=True)\n'
            '        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    else:\n'
            '        import pandas as pd\n'
            '        pd.DataFrame([["1_0", "1", True, 1]],\n'
            '                     columns=["row_id", "game_id", "end_of_game", "score"]\n'
            '                     ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n'
            '        print("[ewm] commit: CPU-safe landing; scored rerun runs SolverEWM on hidden games", flush=True)\n')
        if old not in src:
            raise SystemExit("run-cell block did not match")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.replace(old, new).splitlines(keepends=True)})
        continue
    new_cells.append(c)
nb["cells"] = new_cells

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ewm-submission.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "ahmedmobasher86/arc-agi-3-ewm-rtx",
    "title": "arc-agi-3-ewm-rtx",
    "code_file": "ewm-submission.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "ahmedmobasher86/taaf-src-hybrid",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [], "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'ewm-submission.ipynb'} ({len(nb['cells'])} cells, {len(BLOB)} embedded files)")
