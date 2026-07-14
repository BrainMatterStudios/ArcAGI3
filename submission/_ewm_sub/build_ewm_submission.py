#!/usr/bin/env python3
"""Build submission/_ewm_sub/ewm-submission.ipynb — the EWM competition submission.

Strategy: TRANSFORM the PROVEN duck notebook (submission/_finetuned/duck-finetuned.ipynb) so all
the intricate, already-working plumbing (dataset mounts, setup_commands that serve Qwen3.6-27B-FP8
on 127.0.0.1:1234, COMPETITION gateway game list, benchmark load, teardown) is reused VERBATIM, and
only the solver changes: `bm.solver = SolverEWM(...)`. Embeds the OFFLINE-VALIDATED EWM scaffold
(server_taaf, ewm_solver, ewm_agent, score_run, workspace_init incl. search_lib) as base64.

Config knobs (top of the generated cell 13): PER_GAME_SECONDS, N_PASSES.
Run: python submission/_ewm_sub/build_ewm_submission.py
"""
import base64, json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
DUCK = REPO / "submission/_finetuned/duck-finetuned.ipynb"
SCAF = REPO / "scratchpad/ewm_pathb"
OUT = REPO / "submission/_ewm_sub"

# ---- collect + base64 the validated scaffold ----
files = {}
def add(rel, path): files[rel] = path.read_bytes()
# embed at the EXACT layout ewm_solver.py expects: _HERE=EWM; _SRC=EWM/src;
# workspace_init at EWM/src/agent/workspace_init; server_taaf importable from EWM/src/server.
add("ewm_solver.py", SCAF / "ewm_solver.py")
add("ewm_agent.py", SCAF / "ewm_agent.py")
add("score_run.py", SCAF / "score_run.py")
add("src/server/server_taaf.py", SCAF / "src/server/server_taaf.py")
for p in (SCAF / "src/agent/workspace_init").rglob("*"):
    if p.is_file() and "__pycache__" not in str(p):
        files["src/agent/workspace_init/" + str(p.relative_to(SCAF / "src/agent/workspace_init"))] = p.read_bytes()
BLOB = {k: base64.b64encode(v).decode() for k, v in files.items()}

# ---- embed cell: unpack scaffold + wire sys.path so SolverEWM imports resolve ----
EMBED = (
    "import base64, json, os, sys\n"
    "from pathlib import Path\n"
    "EWM = Path('/kaggle/working/ewm'); EWM.mkdir(parents=True, exist_ok=True)\n"
    "BLOB = json.loads(%r)\n"
    "for rel, b64 in BLOB.items():\n"
    "    p = EWM / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(b64))\n"
    "# ewm_solver + ewm_agent at EWM root; server_taaf under EWM/src/server (both on path)\n"
    "for entry in (str(EWM), str(EWM / 'src' / 'server')):\n"
    "    if entry not in sys.path: sys.path.insert(0, entry)\n"
    "print('[ewm] scaffold unpacked:', len(BLOB), 'files ->', EWM)\n"
) % json.dumps(BLOB)

# ---- cell 13 replacement: swap in SolverEWM ----
SOLVER_CELL = (
    "# ===== EWM SUBMISSION: replace the duck solver with verification-by-execution =====\n"
    "PER_GAME_SECONDS = float(os.environ.get('EWM_PER_GAME_SECONDS', '290'))  # ~110 games in 9h (probe: broad+shallow)\n"
    "N_PASSES = int(os.environ.get('EWM_N_PASSES', '1'))                      # 1 = probe; >=2 needs replay (todo)\n"
    "from ewm_solver import SolverEWM\n"
    "bm.solver = SolverEWM(\n"
    "    base_url='http://127.0.0.1:1234/v1',        # setup_commands serves Qwen3.6-27B-FP8 here\n"
    "    model='vrfai/Qwen3.6-27B-FP8',\n"
    "    per_game_seconds=PER_GAME_SECONDS,\n"
    "    max_turns=160, max_ctx_chars=110000, obs_cap=4600, max_tokens=4096,\n"
    "    work_root='/kaggle/working/ewm_runs',\n"
    ")\n"
    "print(f'[ewm] solver = SolverEWM  per_game={PER_GAME_SECONDS}s  n_passes={N_PASSES}', flush=True)\n"
)

# ---- transform the duck notebook ----
nb = json.loads(DUCK.read_text())
new_cells = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if c["cell_type"] == "code" and "SERVER_ENABLE_LORA" in src:
        continue  # drop the LoRA cell — EWM uses base Qwen
    if c["cell_type"] == "code" and "Make one-off changes to `bm`" in src:
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": SOLVER_CELL.splitlines(keepends=True)})
        continue
    if c["cell_type"] == "code" and "setup_commands.json" in src:
        new_cells.append(c)  # keep the serve-Qwen cell
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": EMBED.splitlines(keepends=True)})  # then unpack scaffold
        continue
    if c["cell_type"] == "code" and "_competition_games" in src:
        # drop finetuned-only HELDOUT filtering + prints (whole block, exact 4-space source); set n_passes from knob
        src = src.replace(
            'HELDOUT={"ft09","re86","sb26","sc25","tu93"}\n'
            'if not TRUE_SUBMISSION:\n'
            '    bm.games=[g for g in bm.games if getattr(g,"env_name","") in HELDOUT]\n'
            '    print("[finetuned] commit validates LoRA on held-out:", [g.env_name for g in bm.games], flush=True)\n',
            "")
        src = src.replace("bm.n_passes = 1", "bm.n_passes = N_PASSES")
        # ALWAYS compute a soft_end (incl. real submission) so SolverEWM self-paces + finishes
        # gracefully before the wall-clock kill (else an unbounded run risks losing the scorecard).
        src = src.replace(
            'soft_end = None\n'
            'if not TRUE_SUBMISSION:\n'
            '    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n'
            '    if budget > 0:\n'
            '        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))\n',
            'budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n'
            'soft_end = None\n'
            'if budget > 0:\n'
            '    margin = min(900.0, budget * 0.05) if TRUE_SUBMISSION else min(600.0, budget / 2)\n'
            '    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - margin)\n'
            'print(f"[ewm] budget={budget:.0f}s soft_end={soft_end}", flush=True)\n')
        src = src.replace(
            'print("[finetuned] commit: skipping game-play (CPU-safe landing); rerun runs base+LoRA on hidden games", flush=True)',
            'print("[ewm] commit: CPU-safe landing; the real rerun runs SolverEWM on hidden games", flush=True)')
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.splitlines(keepends=True)})
        continue
    new_cells.append(c)

nb["cells"] = new_cells
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ewm-submission.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "ahmedmobasher86/arc-agi-3-ewm-submission",
    "title": "arc-agi-3-ewm-submission",
    "code_file": "ewm-submission.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["jeroencottaar/taaf-kaggle-source-share",
                        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [], "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'ewm-submission.ipynb'} ({(OUT/'ewm-submission.ipynb').stat().st_size//1024} KB, "
      f"{len(nb['cells'])} cells, {len(BLOB)} embedded files)")
