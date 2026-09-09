#!/usr/bin/env python3
"""absorb_kernel.py — Track D: turn a released competitor kernel into a
byte-identical, push-ready copy bundle plus an attestation, in minutes.

WHY THIS EXISTS
---------------
On 09-02 -> 09-05 we absorbed keithtyser's V14 harness by hand: pull, eyeball,
retype the metadata, hash, push. It worked (live 3.25, our current base) but it
took a day and every step was a chance to silently change a byte. Track D of
PLAN-2026-09-08 asks for a 24-hour absorption checklist for the Oct-1 Milestone-2
releases. This is the mechanical half of that checklist.

WHAT IT GUARANTEES
------------------
* The copy's notebook is BYTE-IDENTICAL to the pulled source. We never edit the
  notebook here. A graft goes on afterwards, as a separate, attested change.
* The copy's kernel-metadata.json carries the source's dataset_sources,
  model_sources, docker_image and machine_shape UNCHANGED. Those four are the
  serving regime; changing one silently is how you get an unexplainable score.
* ATTEST.json records exactly what was pulled, when, and its two hashes, in the
  same format as submission/_keith_copy/ATTEST.json (verified to reproduce it).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never runs `kaggle kernels push`. Pushing is shared, irreversible state and
needs Ahmed's explicit go in-session. The tool prints the exact command instead.

HASHES (convention verified against submission/_keith_copy/ATTEST.json)
  file_sha256      = sha256 of the raw .ipynb bytes
  code_cell_sha256 = sha256 of "\n".join(source of each code cell), where a
                     cell's source list is concatenated with "" first.
The second is the identity that matters: it ignores outputs and execution
counts, so it is stable across a re-pull of the same code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Metadata keys that define the SERVING REGIME. They are copied verbatim and a
# mismatch is a hard error, never a warning: the whole reason the byte-copy
# scored 3.25 is that these were identical to keith's.
REGIME_KEYS = ("dataset_sources", "model_sources", "docker_image", "machine_shape")

# Keys that belong to the SOURCE kernel's identity and must not be carried over.
DROP_KEYS = ("id_no", "keywords")


# ---------------------------------------------------------------- attestation

def read_notebook(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    return raw, json.loads(raw)


def cell_source(cell: dict) -> str:
    src = cell.get("source")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def code_cell_sha256(nb: dict) -> str:
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    joined = "\n".join(cell_source(c) for c in cells)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def attest(path: Path) -> dict:
    raw, nb = read_notebook(path)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "code_cell_sha256": code_cell_sha256(nb),
        "n_cells": len(nb.get("cells", [])),
        "n_code_cells": len(cells),
        "bytes": len(raw),
    }


# ------------------------------------------------------------------ metadata

def copy_metadata(source_meta: dict, our_id: str, code_file: str) -> dict:
    """Build our kernel-metadata.json from the source's.

    Everything that defines the serving regime is carried verbatim. Only our
    own identity fields change, and the copy is forced private: a public copy
    of someone else's notebook is not ours to publish.
    """
    out = {k: v for k, v in source_meta.items() if k not in DROP_KEYS}
    out["id"] = our_id
    out["title"] = our_id.split("/", 1)[-1]
    out["code_file"] = code_file
    out["is_private"] = True
    for k in REGIME_KEYS:
        if k in source_meta:
            out[k] = source_meta[k]
    return out


def regime_diff(a: dict, b: dict) -> dict:
    """Regime keys that differ between two kernel metadatas. Empty == same rig."""
    return {k: (a.get(k), b.get(k)) for k in REGIME_KEYS if a.get(k) != b.get(k)}


# ----------------------------------------------------------------- kaggle I/O

def kaggle_pull(slug: str, dest: Path, *, runner=subprocess.run) -> Path:
    """`kaggle kernels pull -k <slug> -p <dest> -m`. Returns the .ipynb path.

    -m also writes kernel-metadata.json, which is where the regime lives.
    Note (measured, docs/SUBMISSION-LEDGER.md): version-specific pulls are
    refused with HTTP 403, so this always returns the LATEST version. Record
    the pull time; that is the only version anchor you get.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # The kernel slug is POSITIONAL in the kaggle CLI; there is no -k flag
    # (`-k` is silently ignored and you get "A kernel must be specified").
    r = runner([sys.executable, "-m", "kaggle", "kernels", "pull",
                slug, "-p", str(dest), "-m"],
               capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle kernels pull failed rc={r.returncode}\n"
                           f"{r.stdout}\n{r.stderr}")
    nbs = sorted(dest.glob("*.ipynb"))
    if len(nbs) != 1:
        raise RuntimeError(f"expected exactly one .ipynb in {dest}, found {[p.name for p in nbs]}")
    return nbs[0]


# ------------------------------------------------------------------- staging

def stage(slug: str, our_name: str, root: Path, *, runner=subprocess.run,
          now: str | None = None) -> dict:
    """Pull `slug` and lay down a push-ready copy bundle under `root`.

    Layout:
      <root>/source/<original>.ipynb + kernel-metadata.json   (pristine pull)
      <root>/push/<our_name>.ipynb   + kernel-metadata.json   (byte-identical copy)
      <root>/ATTEST.json
    """
    root = Path(root)
    src_dir, push_dir = root / "source", root / "push"
    nb_path = kaggle_pull(slug, src_dir, runner=runner)
    src_meta = json.loads((src_dir / "kernel-metadata.json").read_text())

    push_dir.mkdir(parents=True, exist_ok=True)
    our_nb = push_dir / f"{our_name}.ipynb"
    # copy2, not a re-serialised json.dump: re-serialising would change bytes.
    shutil.copy2(nb_path, our_nb)

    owner = os.environ.get("KAGGLE_USERNAME", "ahmedmobasher86")
    our_id = f"{owner}/{our_name}"
    meta = copy_metadata(src_meta, our_id, our_nb.name)
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")

    a = attest(nb_path)
    b = attest(our_nb)
    if a["file_sha256"] != b["file_sha256"]:
        raise RuntimeError("copy is not byte-identical to the pulled source")

    att = {
        "source_kernel": slug,
        "source_id_no": src_meta.get("id_no"),
        "pulled_utc": now or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "file_sha256": a["file_sha256"],
        "code_cell_sha256": a["code_cell_sha256"],
        "copy_kernel": our_id,
        "byte_identical_notebook": True,
        "sources": {
            "datasets": src_meta.get("dataset_sources", []),
            "models": src_meta.get("model_sources", []),
            "docker": src_meta.get("docker_image"),
        },
        "machine_shape": src_meta.get("machine_shape"),
        "n_code_cells": a["n_code_cells"],
        "bytes": a["bytes"],
    }
    (root / "ATTEST.json").write_text(json.dumps(att, indent=1) + "\n")
    return att


def verify(root: Path) -> dict:
    """Re-check a staged bundle against its own ATTEST.json. Run this after any
    graft, to prove exactly which hash changed and which did not."""
    root = Path(root)
    att = json.loads((root / "ATTEST.json").read_text())
    push_nb = next((root / "push").glob("*.ipynb"))
    now = attest(push_nb)
    return {
        "file_sha256_matches": now["file_sha256"] == att["file_sha256"],
        "code_cell_sha256_matches": now["code_cell_sha256"] == att["code_cell_sha256"],
        "attested": {k: att[k] for k in ("file_sha256", "code_cell_sha256")},
        "actual": {k: now[k] for k in ("file_sha256", "code_cell_sha256")},
    }


# ----------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("attest", help="hash a local .ipynb (offline)")
    pa.add_argument("notebook")

    ps = sub.add_parser("stage", help="pull a kernel and build a push-ready copy")
    ps.add_argument("slug", help="owner/kernel-name of the kernel to absorb")
    ps.add_argument("--as", dest="our_name", required=True, help="our kernel name")
    ps.add_argument("--root", default=None, help="staging dir (default submission/_absorb_<name>)")

    pv = sub.add_parser("verify", help="re-check a staged bundle against ATTEST.json")
    pv.add_argument("root")

    pd = sub.add_parser("diff", help="compare regime keys of two kernel-metadata.json files")
    pd.add_argument("a"); pd.add_argument("b")

    args = p.parse_args(argv)

    if args.cmd == "attest":
        print(json.dumps(attest(Path(args.notebook)), indent=1))
        return 0

    if args.cmd == "diff":
        d = regime_diff(json.loads(Path(args.a).read_text()),
                        json.loads(Path(args.b).read_text()))
        if not d:
            print("SAME REGIME — dataset_sources, model_sources, docker_image, machine_shape all match")
            return 0
        print("REGIME DIFFERS:")
        for k, (x, y) in d.items():
            print(f"  {k}\n    a: {x}\n    b: {y}")
        return 1

    if args.cmd == "verify":
        v = verify(Path(args.root))
        print(json.dumps(v, indent=1))
        return 0 if v["file_sha256_matches"] else 1

    root = Path(args.root) if args.root else Path("submission") / f"_absorb_{args.our_name}"
    att = stage(args.slug, args.our_name, root)
    print(json.dumps(att, indent=1))
    print(f"\nStaged at {root}")
    print("NOT PUSHED. Pushing is shared state and needs Ahmed's explicit go.")
    print(f"  Command when authorised:  python3 -m kaggle kernels push -p {root/'push'}")
    print("Before pushing, work the checklist: docs/TRACK-D-absorption-checklist.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
