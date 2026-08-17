#!/usr/bin/env python3
"""build_duck38_v12_bank.py — duck38-v12 + win-then-replay banking, attested.

CANDIDATE for the 2026-08-19 00:01Z slot, contingent on (a) tonight's v12
read and (b) a green smoke commit tomorrow. Single variable vs duck38-v12:
the banking graft (submission/_duck38_v12_bank/graft_bank.py), validated
2026-08-17 through the competition-parity server (commit 2460277).

Construction: the built duck38-v12 notebook (attestation + smoke hook, code
hash re-verified against the smoked v1) + ONE new cell inserted after the
benchmark-load/hook cells and before the run cell:

  - inlines graft_bank.py verbatim into the notebook (self-contained: no
    extra dataset, and the published notebook discloses the full mechanism),
    materialized at runtime as a real module via importlib;
  - verify_seam() — live _play_one blake2b must match the pinned copy,
    else NO install (stock behavior);
  - bm.solver = BankingHarnessSolver.from_solver(bm.solver) inside a
    fail-open try/except: any error prints [banking] install failed -> stock
    and leaves the stock solver untouched;
  - run-wide kill switch is inside the graft (fresh-play invariant).

Usage:
  .venv/bin/python submission/_duck38_v12_bank/build_duck38_v12_bank.py
  kaggle kernels push -p submission/_duck38_v12_bank --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
V12_NOTEBOOK = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
V12_HASH = "dc2c36f897428123b2b9e95d85470b57cf6d954910599be3908b0a38a8199615"
GRAFT = HERE / "graft_bank.py"
KERNEL_SLUG = "arc3-duck38-v12-bank"

RUN_MARKER = "await bm.run("
HOOK_MARKER = "SMOKE_GAMES"

INSTALL_TEMPLATE = '''# Win-then-replay banking graft (single variable vs duck38-v12).
# Full mechanism inlined below and disclosed; validated offline through the
# competition-parity server 2026-08-17 (post-WIN RESET opens play #2 through
# the guarded REST path; pruned replay zero-divergence; official scorer takes
# max over per-play scores). Fail-open: ANY error leaves the stock solver.
_GRAFT_SOURCE = {graft_source!r}

try:
    import importlib.util as _ilu

    _graft_path = WORKING_DIR / "graft_bank.py"
    _graft_path.write_text(_GRAFT_SOURCE, encoding="utf-8")
    _spec = _ilu.spec_from_file_location("graft_bank", _graft_path)
    _graft = _ilu.module_from_spec(_spec)
    sys.modules["graft_bank"] = _graft  # required for dataclass annotation resolution
    _spec.loader.exec_module(_graft)
    _graft.verify_seam()
    bm.solver = _graft.BankingHarnessSolver.from_solver(bm.solver)
    print("[banking] armed:", type(bm.solver).__name__,
          "| kill-switch ready | seam verified")
except Exception as _exc:  # noqa: BLE001 — fail open to stock
    print(f"[banking] install failed -> stock: {{type(_exc).__name__}}: {{_exc}}")
'''


def main() -> None:
    import hashlib

    nb = json.loads(V12_NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert hashlib.sha256(code.encode()).hexdigest() == V12_HASH, (
        "base v12 notebook drifted from the smoked v1 — rebuild/re-verify first")

    graft_source = GRAFT.read_text()
    assert "def verify_seam" in graft_source and "BankingHarnessSolver" in graft_source

    hook_idx = run_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if HOOK_MARKER in src:
            hook_idx = i
        if RUN_MARKER in src:
            run_idx = i
    assert hook_idx is not None and run_idx is not None and hook_idx < run_idx

    install_cell = {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [],
        "source": INSTALL_TEMPLATE.format(graft_source=graft_source).splitlines(keepends=True),
    }
    nb["cells"].insert(run_idx, install_cell)

    out = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert out.count("[banking] armed") == 1
    assert out.index("SMOKE_GAMES") < out.index("[banking] armed") < out.index(RUN_MARKER)

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")

    meta = json.loads((HERE.parent / "_duck38_v12" / "kernel-metadata.json").read_text())
    meta["id"] = f"ahmedmobasher86/{KERNEL_SLUG}"
    meta["title"] = KERNEL_SLUG
    meta["code_file"] = f"{KERNEL_SLUG}.ipynb"
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    new_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(new_code.encode()).hexdigest())


if __name__ == "__main__":
    main()
