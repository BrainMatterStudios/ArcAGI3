#!/usr/bin/env python3
"""transplant_serving.py — Track D step 3 (2026-09-14): put OUR serving under THEIR harness.

Every TAAF-lineage Kaggle kernel is a thin runner over a dataset bundle that holds the AGENT
(`src/`, `deploy_target.pkl`, `benchmark_initial.pkl`) and the SERVING (`serving_setup.py`,
launched by `setup_commands.json`). A release on a weaker model (e.g. the June milestone winner on
Qwen3.6-27B-FP8) can be re-served on the flown Flash-Next NVFP4 regime by mounting BOTH bundles:
theirs for the agent, keithtyser's smoke bundle for serving.

What this does to a staged copy (`absorb_kernel.py stage` output):
  metadata  dataset_sources = theirs (index 0 stays THEIR bundle) + keith's two datasets;
            model_sources   = keith's Flash-Next NVFP4 model; docker_image = keith's digest.
  notebook  1. `BUNDLE_DIR = _find_bundle_dir()` -> pinned to THEIR bundle by slug (two bundles
               now carry the marker file, rglob order is arbitrary);
            2. one inserted cell after the bundle-discovery cell: finds SERVING_BUNDLE_DIR (keith's
               smoke bundle) by slug, sets the serving profile env (kv10-bf16-mtp0-c8-cg32 by default),
               ONLY_RESET_LEVELS and the CUDA linker path exactly as the flown notebook does;
            3. two substitutions in the setup-commands cell so serving_setup.py is launched from
               SERVING_BUNDLE_DIR while every agent path still points at THEIR bundle.
  attest    TRANSPLANT.json: source hashes, the inserted cell's sha256, substitution counts.

Nothing else in their notebook changes; the harness bytes stay theirs.

Usage:
  python3 offkaggle/transplant_serving.py submission/_absorb_june --as arc3-transplant-june
  # then, with Ahmed's go:  kaggle kernels push -p submission/_absorb_june/push_transplant
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

KEITH_DATASETS = ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                  "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]
KEITH_MODEL = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
KEITH_DOCKER = "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
SERVING_SLUG = KEITH_DATASETS[0]

PROFILES = {
    # measured 09-12..09-14 on the Kaggle box (offkaggle/REGIME_WAVE_STATUS.md CLOSURE 2026-09-14)
    "kv10-bf16-mtp0-c8-cg32": {"TAAF_VLLM_ENABLE_PREFIX_CACHING": "0", "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
                               "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "10737418240", "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
                               "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192", "TAAF_VLLM_MAX_NUM_SEQS": "8",
                               "TAAF_VLLM_MTP_TOKENS": "0", "TAAF_VLLM_OMP_THREADS": "1"},
    "kv5-bf16-mtp3-c8-cg32": {"TAAF_VLLM_ENABLE_PREFIX_CACHING": "0", "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
                              "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120", "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
                              "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192", "TAAF_VLLM_MAX_NUM_SEQS": "8",
                              "TAAF_VLLM_MTP_TOKENS": "3", "TAAF_VLLM_OMP_THREADS": "1"},
}

FIND_OLD = "BUNDLE_DIR = _find_bundle_dir()"
FIND_NEW = '''# TRANSPLANT: two marker-bearing bundles are mounted now; pin the AGENT bundle to this notebook's own
# first dataset source instead of taking whichever marker rglob yields first.
def _find_bundle_dir_pinned() -> Path:
    owner, slug = DATASET_SOURCES[0].split("/", 1)
    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        if slug in str(marker.parent):
            return marker.parent
    raise RuntimeError(f"TAAF agent bundle {DATASET_SOURCES[0]} not found under /kaggle/input.")


BUNDLE_DIR = _find_bundle_dir_pinned()'''
SUB_ENV_OLD = 'env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)'
SUB_ENV_NEW = 'env["TAAF_KAGGLE_BUNDLE_DIR"] = str(SERVING_BUNDLE_DIR)  # TRANSPLANT: serving_setup.py resolves itself here'
SUB_CMD_OLD = 'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())'
SUB_CMD_NEW = 'json.loads((SERVING_BUNDLE_DIR / "setup_commands.json").read_text())'   # TRANSPLANT: OUR serving, THEIR agent

SERVING_CELL = '''# ===================== TRANSPLANT (offkaggle/transplant_serving.py, {stamp}) =====================
# THEIR agent bundle stays BUNDLE_DIR. OUR serving (keithtyser smoke bundle: serving_setup.py + the
# Flash-Next NVFP4 model + the vLLM runtime layers) is launched from SERVING_BUNDLE_DIR instead of
# theirs. Profile {profile} — measured on this box 2026-09-12..14 (offkaggle/REGIME_WAVE_STATUS.md).
SERVING_SLUG = "{serving_slug}"


def _find_serving_bundle_dir() -> Path:
    slug = SERVING_SLUG.split("/", 1)[1]
    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        if slug in str(marker.parent):
            return marker.parent
    raise RuntimeError(f"serving bundle {{SERVING_SLUG}} not found under /kaggle/input.")


SERVING_BUNDLE_DIR = _find_serving_bundle_dir()
if SERVING_BUNDLE_DIR == BUNDLE_DIR:
    raise RuntimeError("TRANSPLANT: serving bundle resolved to the agent bundle; refusing to run.")
print(f"taaf.kaggle: TRANSPLANT agent bundle = {{BUNDLE_DIR}}  serving bundle = {{SERVING_BUNDLE_DIR}}", flush=True)

PUBLIC25_VLLM_PROFILE_NAME = {profile!r}
PUBLIC25_VLLM_PROFILE_ENV = {profile_env}
for _k, _v in PUBLIC25_VLLM_PROFILE_ENV.items():
    os.environ[_k] = _v
print(f"PUBLIC25_VLLM_PROFILE name={{PUBLIC25_VLLM_PROFILE_NAME}} env={{json.dumps(PUBLIC25_VLLM_PROFILE_ENV, sort_keys=True)}}", flush=True)
os.environ["ONLY_RESET_LEVELS"] = "true"          # LAW: before any arcengine import
_cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    e for e in [_cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if e)
# ==================================================================================================
'''


def _code_cells(nb: dict) -> list[int]:
    return [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]


def _src(cell: dict) -> str:
    s = cell.get("source")
    return "".join(s) if isinstance(s, list) else (s or "")


def _set(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def transplant(root: Path, our_name: str, *, profile: str = "kv10-bf16-mtp0-c8-cg32",
               owner: str | None = None, now: str | None = None) -> dict:
    root = Path(root)
    push = root / "push"
    src_nb = next(p for p in sorted(push.iterdir()) if p.suffix == ".ipynb")
    meta = json.loads((push / "kernel-metadata.json").read_text())
    nb = json.loads(src_nb.read_text())
    stamp = now or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    # 1. pin the agent bundle
    find_idx = next(i for i in _code_cells(nb) if FIND_OLD in _src(nb["cells"][i]))
    text = _src(nb["cells"][find_idx])
    assert text.count(FIND_OLD) == 1
    _set(nb["cells"][find_idx], text.replace(FIND_OLD, FIND_NEW, 1))

    # 2. insert the serving cell right after the bundle-discovery cell
    cell_text = SERVING_CELL.format(stamp=stamp, profile=profile, serving_slug=SERVING_SLUG,
                                    profile_env=json.dumps(PROFILES[profile], indent=4, sort_keys=True))
    nb["cells"].insert(find_idx + 1, {"cell_type": "code", "execution_count": None, "metadata": {},
                                      "outputs": [], "source": cell_text.splitlines(keepends=True)})

    # 3. launch OUR setup commands
    cmd_idx = next(i for i in _code_cells(nb) if SUB_CMD_OLD in _src(nb["cells"][i]))
    text = _src(nb["cells"][cmd_idx])
    n_env, n_cmd = text.count(SUB_ENV_OLD), text.count(SUB_CMD_OLD)
    assert n_env == 1 and n_cmd == 1, (n_env, n_cmd)
    _set(nb["cells"][cmd_idx], text.replace(SUB_ENV_OLD, SUB_ENV_NEW, 1).replace(SUB_CMD_OLD, SUB_CMD_NEW, 1))

    # metadata: their bundle first, ours appended; our model + docker
    owner = owner or os.environ.get("KAGGLE_USERNAME", "ahmedmobasher86")
    out_meta = dict(meta)
    ds = list(meta.get("dataset_sources", []))
    out_meta["dataset_sources"] = ds + [d for d in KEITH_DATASETS if d not in ds]
    out_meta["model_sources"] = [KEITH_MODEL]
    out_meta["docker_image"] = KEITH_DOCKER
    out_meta["id"] = f"{owner}/{our_name}"
    out_meta["title"] = our_name
    out_meta["code_file"] = f"{our_name}.ipynb"
    out_meta["is_private"] = True

    out = root / "push_transplant"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{our_name}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (out / "kernel-metadata.json").write_text(json.dumps(out_meta, indent=1) + "\n")
    att = {
        "source_copy": src_nb.name,
        "source_file_sha256": hashlib.sha256(src_nb.read_bytes()).hexdigest(),
        "inserted_cell_sha256": hashlib.sha256(cell_text.encode()).hexdigest(),
        "substitutions": {"bundle_pin": 1, "setup_env_bundle_dir": n_env, "setup_commands_dir": n_cmd},
        "profile": profile, "profile_env": PROFILES[profile],
        "serving_bundle": SERVING_SLUG, "model": KEITH_MODEL, "docker_image": KEITH_DOCKER,
        "dataset_sources": out_meta["dataset_sources"], "stamp": stamp, "kernel": out_meta["id"],
    }
    (out / "TRANSPLANT.json").write_text(json.dumps(att, indent=1) + "\n")
    return att


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", help="staged absorb dir (holds push/<copy>.ipynb + kernel-metadata.json)")
    p.add_argument("--as", dest="our_name", required=True)
    p.add_argument("--profile", default="kv10-bf16-mtp0-c8-cg32", choices=sorted(PROFILES))
    a = p.parse_args(argv)
    att = transplant(Path(a.root), a.our_name, profile=a.profile)
    print(json.dumps(att, indent=1))
    print(f"\nNOT PUSHED. When authorised:  kaggle kernels push -p {Path(a.root) / 'push_transplant'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
