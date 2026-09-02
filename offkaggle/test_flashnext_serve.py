#!/usr/bin/env python3
"""Host-only tests for the Flash-Next (keithtyser V14) Modal replica. No
network, no GPU, no modal client required.

The fidelity anchor is the machine-readable block in offkaggle/KEITH_REGIME.md
(transcribed from the public notebook, its serving_setup.py, SOURCE_IDENTITY
and the commit-run vllm-server-identity.json). What is checked:

  1. modal_flashnext_serve constants == the regime pin (GPU, model, revision,
     config sha, served name, vLLM version, image digests, PLE hashes).
  2. vllm_cmd(<MODEL_DIR>) == Keith's argv, token for token.
  3. server_env() applies every exact key, removes PYTORCH_ALLOC_CONF, sets
     the seven cache keys, and matches the notebook profile (OMP 1).
  4. The PLE patch bundle in offkaggle/flashnext_patches/ hashes to the pinned
     values and the applicator's embedded constants agree with the pin.
  5. Cost guards + auth exemptions.

Run:  .venv/bin/python offkaggle/test_flashnext_serve.py     (or pytest)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import modal_flashnext_serve as mfs  # noqa: E402

REGIME_MD = HERE / "KEITH_REGIME.md"
PATCH_DIR = HERE / "flashnext_patches"


def _regime() -> dict:
    text = REGIME_MD.read_text(encoding="utf-8")
    m = re.search(r"<!-- REGIME-JSON-BEGIN -->\s*```json\s*(.*?)```\s*<!-- REGIME-JSON-END -->",
                  text, flags=re.S)
    assert m, "regime JSON block missing from KEITH_REGIME.md"
    return json.loads(m.group(1))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. identity constants --------------------------------------------------

def test_identity_constants_match_regime():
    r = _regime()
    assert mfs.GPU_KIND == r["gpu"] == "RTX-PRO-6000"
    assert mfs.N_GPU == 1
    assert mfs.HF_MODEL_REPO == r["hf_model_repo"]
    assert mfs.HF_MODEL_REVISION == r["hf_model_revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", mfs.HF_MODEL_REVISION)
    assert mfs.MODEL_CONFIG_SHA256 == r["model_config_sha256"]
    assert mfs.SERVED_MODEL_NAME == r["served_model_name"]
    assert mfs.VLLM_VERSION == r["vllm_version"]
    assert mfs.VLLM_IMAGE == r["vllm_image"]
    assert mfs.VLLM_IMAGE_INDEX_DIGEST == r["vllm_image_index_digest"]
    assert mfs.VLLM_IMAGE_AMD64_DIGEST == r["vllm_image_amd64_manifest_digest"]
    assert mfs.VLLM_IMAGE_REF == f"vllm/vllm-openai@{r['vllm_image_amd64_manifest_digest']}"
    assert mfs.SITE_PACKAGES == r["site_packages"]
    assert mfs.IMAGE_PYTHON == r["python"]
    assert mfs.PLE_PATCH_TARGET == r["ple_patch_target"]
    assert mfs.PLE_STOCK_SHA256 == r["ple_stock_sha256"]
    assert mfs.PLE_PATCHED_SHA256 == r["ple_patched_sha256"]
    assert mfs.PLE_APPLICATOR_SHA256 == r["ple_applicator_sha256"]
    assert mfs.PLE_PATCH_DIFF_SHA256 == r["ple_patch_diff_sha256"]
    assert mfs.VLLM_HOST == r["vllm_host"] and mfs.VLLM_PORT == r["vllm_port"]


# --- 2. the argv -------------------------------------------------------------

def test_vllm_cmd_matches_keith_argv_token_for_token():
    r = _regime()
    cmd = mfs.vllm_cmd("<MODEL_DIR>")
    assert cmd[:5] == [r["python"], "-m", "vllm.entrypoints.cli.main", "serve", "<MODEL_DIR>"], cmd[:5]
    assert cmd[5:] == r["argv_after_model_dir"], "\n".join(
        f"{i}: {a!r} vs {b!r}" for i, (a, b) in enumerate(zip(cmd[5:], r["argv_after_model_dir"]))
        if a != b) or f"length {len(cmd[5:])} vs {len(r['argv_after_model_dir'])}"
    # `vllm serve`, never the openai.api_server module our 27B rigs use
    assert "vllm.entrypoints.openai.api_server" not in cmd
    # the flags our 27B rigs pass that Keith does NOT
    assert "--generation-config" not in cmd
    assert "--default-chat-template-kwargs" not in cmd
    assert "--enable-prefix-caching" not in cmd and "--no-enable-prefix-caching" in cmd
    assert "--gpu-memory-utilization" not in cmd
    assert "--kv-cache-dtype" not in cmd
    assert "--moe-backend" not in cmd
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec == {"method": "mtp", "num_speculative_tokens": 3}
    assert cmd[cmd.index("--speculative-config") + 1] == '{"method":"mtp","num_speculative_tokens":3}'
    # the model dir is the only host-specific token
    real = mfs.vllm_cmd("/cache/huggingface/hub/x/snapshots/y")
    diff = [(a, b) for a, b in zip(cmd, real) if a != b]
    assert diff == [("<MODEL_DIR>", "/cache/huggingface/hub/x/snapshots/y"),
                    ("<MODEL_DIR>/chat_template.jinja",
                     "/cache/huggingface/hub/x/snapshots/y/chat_template.jinja")], diff


def test_profile_env_is_the_notebooks_cell3_and_drives_the_argv():
    r = _regime()
    assert mfs.PUBLIC25_VLLM_PROFILE_NAME == r["public25_vllm_profile_name"]
    assert mfs.PUBLIC25_VLLM_PROFILE_ENV == r["public25_vllm_profile_env"]
    p = mfs.PUBLIC25_VLLM_PROFILE_ENV
    cmd = mfs.vllm_cmd("<MODEL_DIR>")

    def arg(flag):
        return cmd[cmd.index(flag) + 1]

    assert arg("--max-num-seqs") == p["TAAF_VLLM_MAX_NUM_SEQS"] == "8"
    assert arg("--kv-cache-memory-bytes") == p["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"] == str(5 * 1024**3)
    assert arg("--max-cudagraph-capture-size") == p["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"] == "32"
    assert arg("--max-num-batched-tokens") == p["TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert p["TAAF_VLLM_MTP_TOKENS"] == "3" and p["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "0"
    assert p["TAAF_VLLM_KV_CACHE_DTYPE"] == "auto" and p["TAAF_VLLM_OMP_THREADS"] == "1"
    assert arg("--max-model-len") == "32768"


# --- 3. the process environment -------------------------------------------

def test_server_env_matches_regime(tmp_path=None):
    r = _regime()
    assert mfs.SERVER_ENV_EXACT == r["server_env_exact"]
    assert mfs.SERVER_ENV_REMOVED == r["server_env_removed"]
    assert sorted(mfs.SERVER_ENV_CACHE) == sorted(r["server_env_cache_keys"])
    base = {"PYTORCH_ALLOC_CONF": "expandable_segments:True", "PATH": "/usr/bin",
            "PYTHONPATH": "/x", "LD_LIBRARY_PATH": "/y", "HOME": "/root"}
    # server_env mkdirs its cache roots — point them somewhere writable for the test
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        saved = dict(mfs.SERVER_ENV_CACHE), mfs.TMP_ROOT
        try:
            for k in mfs.SERVER_ENV_CACHE:
                mfs.SERVER_ENV_CACHE[k] = f"{td}/{k}"
            mfs.TMP_ROOT = f"{td}/tmp"
            env = mfs.server_env(base)
        finally:
            mfs.SERVER_ENV_CACHE.clear(); mfs.SERVER_ENV_CACHE.update(saved[0]); mfs.TMP_ROOT = saved[1]
    assert "PYTORCH_ALLOC_CONF" not in env
    for k, v in r["server_env_exact"].items():
        assert env[k] == v, (k, env.get(k), v)
    assert env["HOME"] == "/root"                                      # parent env inherited
    assert env["PYTHONPATH"].split(":")[:2] == [
        f"{mfs.SITE_PACKAGES}/nvidia_cutlass_dsl/dsl_packages", mfs.SITE_PACKAGES]
    assert env["PYTHONPATH"].endswith(":/x")
    assert env["PATH"].startswith(f"{mfs.CUDA_HOME}/bin:/usr/local/bin:") and env["PATH"].endswith("/usr/bin")
    assert env["LD_LIBRARY_PATH"].endswith(":/y") or env["LD_LIBRARY_PATH"] == "/y"
    for k in r["server_env_cache_keys"]:
        assert env[k].startswith(td), (k, env[k])
    assert env["OMP_NUM_THREADS"] == mfs.PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_OMP_THREADS"]
    assert env["VLLM_PLE_OFFLOAD_READY_TIMEOUT"] == str(mfs.SERVER_READY_TIMEOUT) == "1500"
    assert env["VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256"] == mfs.MODEL_CONFIG_SHA256


def test_cache_layout_is_on_the_volume_with_keiths_relative_layout():
    c = mfs.SERVER_ENV_CACHE
    assert c["HF_HOME"] == f"{mfs.CACHE_DIR}/huggingface" == mfs.HF_HOME
    assert c["XDG_CACHE_HOME"] == mfs.CACHE_ROOT and c["TORCH_HOME"] == f"{mfs.CACHE_ROOT}/torch"
    for k, sub in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton"),
                   ("CUDA_CACHE_PATH", "cuda"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer")):
        assert c[k] == f"{mfs.COMPILE_CACHE_ROOT}/{sub}", (k, c[k])
    assert mfs.CACHE_ROOT.startswith(mfs.CACHE_DIR) and mfs.COMPILE_CACHE_ROOT.startswith(mfs.CACHE_DIR)
    assert mfs.TMP_ROOT == "/tmp/qwen38-flash-next-vllm-tmp"


# --- 4. the PLE patch bundle -------------------------------------------------

def test_patch_bundle_hashes_and_applicator_constants():
    r = _regime()
    ident = json.loads((PATCH_DIR / "PATCH_IDENTITY.json").read_text())
    assert _sha(PATCH_DIR / "PATCH_IDENTITY.json") == "f0bc8a948567e551d1bdf63f1163503ded897c3eba070d149d44c0c08fccbd35"
    assert _sha(PATCH_DIR / "README.md") == "18e73020660bfe46b66c0ec1dfc3e702e848b51bc3d4773f58f6ae06ef126f9d"
    assert _sha(PATCH_DIR / mfs.PLE_APPLICATOR) == r["ple_applicator_sha256"]
    assert _sha(PATCH_DIR / "radixark_nvfp4_ple_fp8.patch") == r["ple_patch_diff_sha256"] == ident["artifact_sha256"]
    assert ident["stock_target_sha256"] == r["ple_stock_sha256"]
    assert ident["patched_target_sha256"] == r["ple_patched_sha256"]
    assert ident["target"] == r["ple_patch_target"]
    assert ident["vllm_version"] == r["vllm_version"]
    assert ident["vllm_image_amd64_manifest"] == r["vllm_image_amd64_manifest_digest"]
    assert ident["vllm_image_index_digest"] == r["vllm_image_index_digest"]
    assert ident["exact_model_config_sha256"] == r["model_config_sha256"]
    src = (PATCH_DIR / mfs.PLE_APPLICATOR).read_text()
    assert f'EXPECTED_STOCK_SHA256 = (\n    "{r["ple_stock_sha256"]}"' in src
    assert f'EXPECTED_PATCHED_SHA256 = (\n    "{r["ple_patched_sha256"]}"' in src
    assert 'TARGET_RELATIVE_PATH = Path(\n    "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"\n)' in src
    assert f'"{r["model_config_sha256"]}"' in src   # the gate's config hash
    assert '"VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8"' in src
    assert mfs.PATCH_DIR_LOCAL == PATCH_DIR


def test_applicator_transforms_a_synthetic_stock_file_deterministically():
    """The applicator replaces three exact snippets; prove the replacement
    logic runs (hash gates aside) so a build-time failure is a hash mismatch,
    not a code path we never exercised."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ple_patch", PATCH_DIR / mfs.PLE_APPLICATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fake = (mod._STOCK_IMPORT + "\nX\n" + mod._STOCK_FUNCTION + "\nY\n" + mod._STOCK_CALL).encode()
    saved = mod.EXPECTED_STOCK_SHA256, mod.EXPECTED_PATCHED_SHA256
    try:
        mod.EXPECTED_STOCK_SHA256 = hashlib.sha256(fake).hexdigest()
        mod.EXPECTED_PATCHED_SHA256 = "__PATCHED_SHA256_PENDING__"
        out = mod.patched_bytes(fake).decode()
    finally:
        mod.EXPECTED_STOCK_SHA256, mod.EXPECTED_PATCHED_SHA256 = saved
    assert "import os" in out and "_is_exact_radixark_nvfp4_ple" in out
    assert 'f"{prefix}.ngram_embedding", config' in out
    assert mod.EXPECTED_STOCK_SHA256 == mfs.PLE_STOCK_SHA256
    assert mod.EXPECTED_PATCHED_SHA256 == mfs.PLE_PATCHED_SHA256


# --- 5. guards ---------------------------------------------------------------

def test_guards_and_auth_exemptions():
    r = _regime()["modal_guards"]
    assert mfs.IDLE_TIMEOUT_S == r["idle_timeout_s"] == 900
    assert 2 <= mfs.IDLE_TIMEOUT_S <= 20 * 60          # Modal's scaledown_window bounds
    assert mfs.MAX_LIFETIME_S == r["max_lifetime_s"] == 4 * 3600
    assert mfs.MAX_CONTAINERS == r["max_containers"] == 1
    assert mfs.MEMORY_MIB == r["memory_mib"]
    assert mfs.MEMORY_MIB * 1024**2 >= _regime()["min_host_available_bytes"] == mfs.MIN_HOST_AVAILABLE_BYTES
    assert mfs.AUTH_EXEMPT == {("GET", "/v1/models"), ("GET", "/health")}
    assert (("GET", mfs.IDENTITY_PATH) not in mfs.AUTH_EXEMPT) and mfs.IDENTITY_PATH.startswith("/arc3/")
    assert ("POST", "/v1/chat/completions") not in mfs.AUTH_EXEMPT
    assert mfs.PROXY_PORT != mfs.VLLM_PORT and mfs.VLLM_HOST == "127.0.0.1"
    assert mfs.APP_NAME == "arc3-flashnext" and mfs.SECRET_NAME == "arc3-vllm-token"
    assert any("Maximum concurrency" in p for p in mfs.IDENTITY_PATTERNS)
    assert "3.21x" in _regime()["expected_max_concurrency_line"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
