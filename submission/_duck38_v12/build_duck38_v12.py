#!/usr/bin/env python3
"""build_duck38_v12.py — duck38-v12 arm: Qwen3.8 + Aug-07 community harness, attested.

ARM (2026-08-17, slot 2026-08-18 00:01Z, Ahmed-approved "v12 arm + fallback"):
Fork of the public field recipe the 1.9-2.1 pack runs — kernel
foysalemonshanto/arc3-duck-v12-with-qwen-3-8-27b (pulled + fully read 08-17) —
with two additions of ours:

  1. BOOT ATTESTATION cell (doctrine v2): asserts the mounted weights carry the
     official Qwen3.8-FP8 config signature (quant_method=fp8/e4m3, transformers
     5.8.0.dev0, Qwen3_5ForConditionalGeneration) — verified this session
     against the local official snapshot AND the vrfai 3.6 config
     (compressed-tensors config_groups, transformers 5.6.2), so a silently
     mounted 3.6 fails the assert. Logs config/index/first-shard hashes and a
     greedy decode fingerprint for cross-run comparison. Runs before any game.
  2. SMOKE HOOK: a normal commit plays a 3-game, 60-min-soft-capped offline
     smoke (validates serve + attestation + agent loop); the scored rerun path
     (KAGGLE_IS_COMPETITION_RERUN) is untouched — full competition games.

AUDIT TRAIL for the community code (all 2026-08-17, this repo's session logs):
  - anim bundle full-tree diff vs our verified June-12 bundle: only the
    documented behavioral files change (noop_guard.py, animation.py, prompts/
    tool_agent/python_tool_sandbox/solver/run hunks + de-vendoring);
    setup_commands.json and teardown_commands.json are md5-IDENTICAL to June
    (99e4b35d…, 3eca3c6f…) — the shell-executed surface is unchanged.
  - grep sweep of the new files: zero network/exec/exfil primitives.
  - Scored reruns are internet-disabled and carry no user secrets.
  This is the audited-diff gate satisfied on the attach-the-bundle form; the
  rebuilt-onto-our-base bundle remains the follow-up for the durable lane.

READING RULE (pre-registered): this arm is the FIELD RECIPE, multi-variable vs
our June lane (bundle generation + model mount + scaffold). A draw >= 1.9
confirms field parity (the pack's floor is real and ours to keep). 1.30-1.9 =
partial: recipe underperforms the pack for us; investigate serving/geometry
before crediting the bundle. Inside the old band (0.69-1.30) = the recipe does
NOT explain the pack; rethink. LB keeps max(1.74, this), so downside = slot.

Usage:
  .venv/bin/python submission/_duck38_v12/build_duck38_v12.py
  kaggle kernels push -p submission/_duck38_v12 --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD = HERE.parent / "_parity_ab" / "scaffold-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
KERNEL_SLUG = "arc3-duck38-v12"

HOOK_MARKER = "# Inline customization hook."
SETUP_MARKER = "TAAF/vLLM setup completed for Qwen3.8"

ATTEST_CELL = r'''# Boot attestation (doctrine v2, 2026-08-17): the mounted weights must be the
# OFFICIAL Qwen3.8-FP8. Discriminators verified offline against both the
# official HF snapshot and the vrfai 3.6 config: 3.8 = quant_method fp8 /
# fmt e4m3 / transformers 5.8.0.dev0; 3.6-vrfai = compressed-tensors
# config_groups / transformers 5.6.2. A wrong mount must DIE here, before any
# game action is spent. --served-model-name is a rename and proves nothing.
import hashlib as _hashlib
import urllib.request as _rq

_cfg_path = QWEN_MODEL_PATH / "config.json"
_cfg_raw = _cfg_path.read_bytes()
_cfg = json.loads(_cfg_raw)
_q = _cfg.get("quantization_config") or {}
assert _cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"], (
    f"attest FAIL: architectures {_cfg.get('architectures')}")
assert _q.get("quant_method") == "fp8" and _q.get("fmt") == "e4m3", (
    f"attest FAIL: quantization_config is not official fp8/e4m3: {_q}")
assert _cfg.get("transformers_version") == "5.8.0.dev0", (
    f"attest FAIL: transformers_version {_cfg.get('transformers_version')} "
    "(vrfai 3.6 stamps 5.6.2)")
print("attest: config sha256", _hashlib.sha256(_cfg_raw).hexdigest())

_idx_path = QWEN_MODEL_PATH / "model.safetensors.index.json"
if _idx_path.is_file():
    print("attest: index sha256", _hashlib.sha256(_idx_path.read_bytes()).hexdigest())
_shards = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
assert _shards, "attest FAIL: no safetensors shards at model path"
_total = sum(p.stat().st_size for p in _shards)
print(f"attest: {len(_shards)} shards, {_total} bytes total")
assert _total > 25_000_000_000, f"attest FAIL: total shard bytes {_total} too small for 27B FP8"
_h = _hashlib.sha256()
with open(_shards[0], "rb") as _f:
    _h.update(_f.read(1 << 20))
print("attest: first-shard-1MiB sha256", _h.hexdigest())

# Greedy decode fingerprint — logged (not asserted) for cross-run comparison.
_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
if not _base.endswith("/v1"):
    _base += "/v1"
_body = json.dumps({
    "model": QWEN_SERVED_MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly the sum of 17 and 25, then the word quack."}],
    "temperature": 0.0,
    "max_tokens": 48,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
_req = _rq.Request(_base + "/chat/completions", data=_body, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer " + (os.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY"),
})
with _rq.urlopen(_req, timeout=180) as _resp:
    _reply = json.loads(_resp.read())["choices"][0]["message"].get("content") or ""
print("attest: decode fingerprint", repr(_reply)[:160])
print("attest: decode sha256", _hashlib.sha256(_reply.encode()).hexdigest())
print("attest: OK — official Qwen3.8-FP8 signature verified before any game")
'''

SMOKE_CELL = r'''# Smoke/eval hook: a NORMAL COMMIT runs a 3-game, 60-min-soft-capped offline
# smoke (proves serve + attestation + agent loop on the scored GPU class).
# The scored rerun (KAGGLE_IS_COMPETITION_RERUN) never enters this branch —
# it plays the full competition games exactly as the upstream scaffold does.
SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]

if not run_as_submission:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    def _resolve_env_dir():
        candidates = [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
        ]
        for cand in candidates:
            if cand.is_dir():
                return str(cand)
        for hit in Path("/kaggle/input").rglob("environment_files"):
            if hit.is_dir():
                return str(hit)
        raise RuntimeError("environment_files dir not found in /kaggle/input")

    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_GAMES]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "duck38-v12-smoke"
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=3600)
    print(f"smoke hook: {len(bm.games)} games, env_dir={_env_dir}, soft_end={soft_end}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''


def main() -> None:
    nb = json.loads(SCAFFOLD.read_text())
    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert joined.count(HOOK_MARKER) == 1
    assert joined.count(SETUP_MARKER) >= 1

    hook_idx = setup_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if HOOK_MARKER in src:
            hook_idx = i
        if SETUP_MARKER in src and "print" in src:
            setup_idx = i
    assert hook_idx is not None and setup_idx is not None
    assert setup_idx < hook_idx, (setup_idx, hook_idx)

    nb["cells"][hook_idx]["source"] = SMOKE_CELL.splitlines(keepends=True)
    attest_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": ATTEST_CELL.splitlines(keepends=True)}
    nb["cells"].insert(setup_idx + 1, attest_cell)

    out = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "attest: OK" in out and "SMOKE_GAMES" in out
    assert out.index("attest: OK") < out.index("SMOKE_GAMES")

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(json.dumps({
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
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")

    import hashlib
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
