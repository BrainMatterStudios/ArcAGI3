#!/usr/bin/env python3
"""Host-only tests for the off-Kaggle certification rig. No network, no GPU,
no modal client required.

The fidelity anchor is the GIT BLOB of the scored serving config —
`git show HEAD:scratchpad/taaf_scored_ref/setup_commands.json` — parsed here
with ast, never re-typed. What is checked:

  1. modal_vllm_serve constants (versions, model id, served alias, max len,
     tensor parallel, the semantic serve-flag tail) == the HEAD blob.
  2. modal auth exemption for GET /v1/models exists AND is justified: the
     byte-frozen pc_driver really does probe /models without an auth header.
  3. run_wave.SETUP_ENV == the blob's setup_env dict, key for key, minus the
     documented exclusions (PYTHONPATH) and runtime overrides (base URLs).
  4. run_wave arm contracts == the rig's frozen configs (base -> ARM_ENV
     ["base"], shipped -> empty env + UNPATCHED sentinel).
  5. results layout: <out>/<ts>-<arm>/patch_closure_result.json is exactly
     what classify_shipped.py consumes — proven by feeding it a REAL banked
     base wave plus a synthesized shipped twin through the actual CLI main().

Run:  .venv/bin/python offkaggle/test_offkaggle.py     (or pytest)
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
AB = REPO / "submission/_ab_patch_closure"

for p in (HERE, AB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import modal_vllm_serve as mvs  # noqa: E402
import run_wave as rw  # noqa: E402
import patch_closure_config as pcc  # noqa: E402
import shipped_screen_config as ssc  # noqa: E402

BLOB_PATH = "scratchpad/taaf_scored_ref/setup_commands.json"


# --- HEAD blob parsing ------------------------------------------------------


def _head_setup_source() -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"HEAD:{BLOB_PATH}"])
    commands = json.loads(raw)
    assert isinstance(commands, list) and len(commands) == 1, commands
    lines = commands[0].split("\n")
    assert lines[0].endswith("<<'PYSETUP'"), lines[0]
    assert lines[-1] == "PYSETUP", lines[-1]
    return "\n".join(lines[1:-1])


def _module_consts(tree: ast.Module) -> dict:
    """Top-level `NAME = <python literal>` assignments in the blob source,
    extracted via ast.literal_eval (static parsing only — nothing from the
    blob is ever executed)."""
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass  # non-literal assignment (Path(...), f-string, ...)
    return out


def _resolve(node: ast.AST, consts: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return consts[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "str" and len(node.args) == 1):
        return str(_resolve(node.args[0], consts))
    raise ValueError(f"unresolvable node: {ast.dump(node)}")


def _find_assign(body: list, name: str) -> ast.AST:
    for node in body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            return node.value
    raise AssertionError(f"assignment {name!r} not found in blob")


def _parsed_blob():
    tree = ast.parse(_head_setup_source())
    consts = _module_consts(tree)
    return tree, consts


# --- 1. modal constants vs the HEAD blob ------------------------------------


def test_modal_versions_match_head_stamp():
    _, consts = _parsed_blob()
    stamp = consts["STAMP_TEXT"]
    versions = dict(re.findall(r"(\w+)==([\d.]+)", stamp))
    assert versions == {"vllm": mvs.VLLM_VERSION, "torch": mvs.TORCH_VERSION,
                        "flashinfer": mvs.FLASHINFER_VERSION}, (versions, stamp)


def test_modal_model_identity_matches_head():
    _, consts = _parsed_blob()
    assert mvs.SERVED_MODEL_NAME == consts["SERVED_MODEL_NAME"]
    # The scored kernel serves the Kaggle mirror of the HF snapshot under the
    # HF repo id as alias; off-Kaggle we pull the HF repo directly, so the
    # repo id must equal the served alias (verified live 2026-08-10).
    assert mvs.HF_MODEL_REPO == consts["SERVED_MODEL_NAME"]
    assert mvs.VLLM_MAX_MODEL_LEN == consts["VLLM_MAX_MODEL_LEN"] == 65536
    assert mvs.VLLM_TENSOR_PARALLEL_SIZE == consts["VLLM_TENSOR_PARALLEL_SIZE"] == 1


def test_modal_serve_flags_match_head():
    tree, consts = _parsed_blob()
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "start_vllm_server")
    cmd_node = _find_assign(fn.body, "cmd")
    assert isinstance(cmd_node, ast.List)
    resolved = []
    for el in cmd_node.elts:
        try:
            resolved.append(_resolve(el, consts))
        except (ValueError, KeyError):
            resolved.append(None)  # sys.executable / Path calls — plumbing
    anchor = resolved.index("--enable-auto-tool-choice")
    assert resolved[anchor:] == mvs.VLLM_EVAL_FLAGS, (
        f"HEAD flag tail {resolved[anchor:]} != modal {mvs.VLLM_EVAL_FLAGS}")
    # And the module entrypoint is the scored one, not `vllm serve`:
    assert resolved[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    local_cmd = mvs.vllm_cmd("/fake/snapshot")
    assert local_cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert local_cmd[local_cmd.index("--enable-auto-tool-choice"):] == mvs.VLLM_EVAL_FLAGS
    i = local_cmd.index("--served-model-name")
    assert local_cmd[i + 1] == mvs.SERVED_MODEL_NAME
    i = local_cmd.index("--tensor-parallel-size")
    assert local_cmd[i + 1] == str(mvs.VLLM_TENSOR_PARALLEL_SIZE)


def test_modal_guards_and_auth_exemption():
    import os

    assert mvs.SECRET_NAME == "arc3-vllm-token"
    assert mvs.GPU_KIND == "H100" and mvs.N_GPU == 1
    # Cost guards (defaults; env-overridable at deploy time):
    assert mvs.IDLE_TIMEOUT_S == int(os.environ.get("ARC3_IDLE_TIMEOUT_S", "600"))
    assert mvs.MAX_LIFETIME_S == int(
        os.environ.get("ARC3_MAX_LIFETIME_S", str(6 * 3600)))
    # GET /v1/models must be auth-exempt BECAUSE the byte-frozen driver probes
    # it with a bare urlopen (no Authorization header). Verify both sides:
    assert ("GET", "/v1/models") in mvs.AUTH_EXEMPT
    driver_src = (AB / "pc_driver.py").read_text()
    assert 'urlopen(base_url + "/models", timeout=30)' in driver_src, (
        "pc_driver's serving probe changed — re-audit the auth exemption")


# --- 3. run_wave env vs the HEAD blob ---------------------------------------


def test_setup_env_matches_head_blob():
    tree, consts = _parsed_blob()
    env_node = _find_assign(tree.body, "setup_env")
    assert isinstance(env_node, ast.Dict)
    blob_keys = []
    for k_node, v_node in zip(env_node.keys, env_node.values):
        key = k_node.value
        blob_keys.append(key)
        if key in rw.SETUP_ENV_EXCLUDED:
            continue  # PYTHONPATH: Kaggle wheelhouse plumbing, documented
        if key in rw.OVERRIDDEN_AT_RUNTIME:
            assert key not in rw.SETUP_ENV, (
                f"{key} must come from --base-url, not a baked constant")
            continue
        expected = _resolve(v_node, consts)
        assert key in rw.SETUP_ENV, f"run_wave.SETUP_ENV is missing {key!r}"
        assert rw.SETUP_ENV[key] == expected, (
            f"{key}: run_wave has {rw.SETUP_ENV[key]!r}, HEAD blob has {expected!r}")
    # No invented keys either: everything in SETUP_ENV must exist in the blob.
    extras = set(rw.SETUP_ENV) - set(blob_keys)
    assert not extras, f"run_wave.SETUP_ENV carries keys not in the blob: {extras}"
    # Sanity on the excluded/overridden bookkeeping:
    assert set(rw.SETUP_ENV_EXCLUDED) <= set(blob_keys)
    assert set(rw.OVERRIDDEN_AT_RUNTIME) <= set(blob_keys)
    assert rw.SERVED_MODEL_NAME == consts["SERVED_MODEL_NAME"]


# --- 4. run_wave arm contracts vs the frozen rig configs --------------------


def test_arm_contracts_match_rig_configs():
    base = rw.arm_contract("base")
    assert base["arm_env"] == pcc.ARM_ENV["base"] == pcc.BASE_ENV
    assert base["hypothesis"] == pcc.HYPOTHESIS
    assert base["reading"] is None
    assert base["patch_sha256"] is None  # filled with the real inlined hash

    shipped = rw.arm_contract("shipped")
    assert shipped["arm_env"] == {} == ssc.SHIPPED_ENV
    assert shipped["hypothesis"] == ssc.SHIPPED_HYPOTHESIS
    assert shipped["patch_sha256"] == ssc.UNPATCHED_SENTINEL == "UNPATCHED"
    assert shipped["reading"] == ssc.SHIPPED_READING

    try:
        rw.arm_contract("candidate")
        raise AssertionError("unknown arm must raise")
    except ValueError:
        pass


# --- 5. results layout == classify_shipped's input contract -----------------


def test_results_layout_feeds_classify_shipped_unchanged():
    # (a) the filename is pc_driver's own artifact name, not an invention:
    assert rw.RESULT_FILENAME == "patch_closure_result.json"
    driver_src = (AB / "pc_driver.py").read_text()
    assert '"patch_closure_result.json"' in driver_src
    d = rw.result_dir("/x", "shipped", "20260810-120000")
    assert str(d) == "/x/20260810-120000-shipped"

    # (b) end-to-end through the REAL CLI: a genuine banked base wave (a real
    # pc_driver artifact at the registered geometry) + a synthesized shipped
    # twin, laid out exactly as run_wave lays results out.
    banked = REPO / "scratchpad/banked_waves_20260809/pc_base.json"
    assert banked.is_file(), f"missing banked control {banked}"
    base_result = json.loads(banked.read_text())
    assert base_result["arm"] == "base" and base_result["stage"] == "done"

    shipped_result = copy.deepcopy(base_result)
    shipped_result["arm"] = "shipped"
    shipped_result["hypothesis"] = ssc.SHIPPED_HYPOTHESIS
    shipped_result["arm_env"] = {}
    shipped_result["patch_sha256"] = ssc.UNPATCHED_SENTINEL
    shipped_result["identity"] = dict(shipped_result["identity"])
    shipped_result["identity"]["patch_proof"] = {
        "shipped_no_patch_markers": True}
    pd = copy.deepcopy(shipped_result["patch_diagnostics"])
    pd["animation"] = {"payload_deliveries": 0, "frames_delivered": 0}
    pd["graph"] = {k: 0 for k in pd.get("graph", {})}
    pd["watchdog"] = {"stall_kills": 0, "wall_cap_kills": 0,
                      "recovery_resets": 0, "stall_s_observed": []}
    shipped_result["patch_diagnostics"] = pd

    import classify_shipped  # the untouched CLI under test

    with tempfile.TemporaryDirectory() as tmp:
        ts = "20260810-000000"
        shipped_dir = rw.result_dir(tmp, "shipped", ts)
        base_dir = rw.result_dir(tmp, "base", ts)
        shipped_dir.mkdir(parents=True)
        base_dir.mkdir(parents=True)
        (shipped_dir / rw.RESULT_FILENAME).write_text(json.dumps(shipped_result))
        (base_dir / rw.RESULT_FILENAME).write_text(json.dumps(base_result))

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = classify_shipped.main([
                str(shipped_dir / rw.RESULT_FILENAME),
                str(base_dir / rw.RESULT_FILENAME),
            ])
        verdict = json.loads(out.getvalue())
    assert rc == 0, f"classify_shipped rc={rc}: {verdict.get('reasons')}"
    assert verdict["state"] == "INDISTINGUISHABLE", verdict  # same rows twice
    assert verdict["metrics"]["delta_all_games_shipped_minus_base"] == 0.0


# --- runner -----------------------------------------------------------------


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
