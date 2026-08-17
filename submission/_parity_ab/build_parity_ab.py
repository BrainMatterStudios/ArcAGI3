#!/usr/bin/env python3
"""build_parity_ab.py — harness-generation ("field parity") A/B on Kaggle commit-runs.

MOTIVATION (2026-08-17): duck-38 v2 (June-12 Tufa bundle + Qwen3.8) drew 1.29
then 1.74 live while the pack, forking the public "duck v12 + Qwen 3.8"
notebook built on the Aug-07 community bundle (hard_noop_guard +
animation_awareness), posts 2.0-2.76. Composite-jump doctrine approved by
Ahmed 2026-08-17: bundle generation is treated as ONE variable.

DESIGN — one variable, same everything else:
  Arm A (control):   FOYSAL scaffold + Qwen3.8 Kaggle Model + JUNE bundle
                     (jeroencottaar/taaf-kaggle-source-share) — the harness
                     our live 1.29/1.74 lane uses.
  Arm B (treatment): identical scaffold + same model + ANIM bundle
                     (jakobbrggen/taaf-kaggle-source-anim-20260807-anim).
  Both arms: identical customization hook pinning the SAME offline eval —
  the 25 public games, 1 pass, explicit environment_files dir — on the same
  RTX Pro 6000 the scored rerun uses. A normal Save&Run plays the games;
  neither kernel is a submission candidate as-is.

SCAFFOLD PROVENANCE: public kernel
  foysalemonshanto/arc3-duck-v12-with-qwen-3-8-27b (47 votes, run 08-16),
  pulled 2026-08-17. Its cell flow: wheelhouse install → bundle discovery →
  setup_commands patched to the Qwen3.8 model path (replacement counts
  asserted) → vLLM serve with the June-identical serve blob → post-setup
  assert INFERENCE_ANALYZER_MODEL == Qwen/Qwen3.8-27B-FP8 → bm.run().

READING RULE (pre-registered): per-game paired comparison over the 25 public
games (each arm yields one score per game). Ship Arm B's bundle to the live
lane only if (a) B's 25-game mean beats A's by >= 0.25 (the historical
one-wave bar 0.2712 rounded) AND (b) B wins >= 60% of decided (non-tied)
games. A mean delta inside +/-0.25 = no ship, fall back to single-lever
laddering. Quota budget: ~2.5-3.5h GPU per arm.

Usage:
  .venv/bin/python submission/_parity_ab/build_parity_ab.py
  kaggle kernels push -p submission/_parity_ab/arm_a --accelerator NvidiaRtxPro6000
  kaggle kernels push -p submission/_parity_ab/arm_b --accelerator NvidiaRtxPro6000
(The explicit accelerator flag is mandatory — a bare GPU push falls back to a
P100; that killed subs 55538970/55542148.)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD = HERE / "scaffold-arc3-duck-v12-with-qwen-3-8-27b.ipynb"

ANIM_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
JUNE_REF = "jeroencottaar/taaf-kaggle-source-share"
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
LEGACY_36_REF = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
MODEL_SOURCE = "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
COMPETITION = "arc-prize-2026-arc-agi-3"

HOOK_MARKER = "# Inline customization hook."

# The June-12 bundle's own benchmark game list (extracted from its
# benchmark_initial.pkl this session) — versioned public env IDs.
PARITY_GAMES = [
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
]

HOOK_TEMPLATE = '''# Parity A/B hook — pins BOTH arms to the identical offline eval.
# Arm identity is the ONLY intended difference between the two kernels
# besides the attached source bundle.
PARITY_ARM = "{arm}"
PARITY_GAMES = {games!r}

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


if not run_as_submission:
    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=_env_dir,
    )
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in PARITY_GAMES]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = f"parity-{{PARITY_ARM}}"
    assert len(bm.games) == 25, len(bm.games)
    print(f"parity hook: arm={{PARITY_ARM}} games=25 passes=1 env_dir={{_env_dir}}")
else:
    print("parity hook: run_as_submission — hook inert (these kernels are A/B-only)")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''


def build_arm(arm: str, bundle_ref: str, kernel_slug: str) -> None:
    nb = json.loads(SCAFFOLD.read_text())

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert joined.count(ANIM_REF) == 1, "scaffold bundle ref drifted"
    assert joined.count(HOOK_MARKER) == 1, "scaffold hook cell drifted"

    hook_replaced = swapped = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if HOOK_MARKER in src:
            cell["source"] = HOOK_TEMPLATE.format(arm=arm, games=PARITY_GAMES).splitlines(keepends=True)
            hook_replaced = True
        elif ANIM_REF in src and bundle_ref != ANIM_REF:
            cell["source"] = src.replace(ANIM_REF, bundle_ref).splitlines(keepends=True)
            swapped = True
    assert hook_replaced, "hook cell not found"
    assert swapped or bundle_ref == ANIM_REF, "bundle ref swap did not fire"

    out_joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert bundle_ref in out_joined
    assert (ANIM_REF not in out_joined) or bundle_ref == ANIM_REF
    assert f'PARITY_ARM = "{arm}"' in out_joined

    arm_dir = HERE / f"arm_{arm[0]}"
    arm_dir.mkdir(exist_ok=True)
    (arm_dir / f"{kernel_slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (arm_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{kernel_slug}",
        "title": kernel_slug,
        "code_file": f"{kernel_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [WHEELHOUSE_REF, bundle_ref, LEGACY_36_REF],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [MODEL_SOURCE],
    }, indent=2) + "\n")
    print(f"built arm_{arm[0]}: bundle={bundle_ref} kernel={kernel_slug}")


def main() -> None:
    build_arm("a-june-control", JUNE_REF, "arc3-parity-a-june38")
    build_arm("b-anim-treatment", ANIM_REF, "arc3-parity-b-anim38")


if __name__ == "__main__":
    main()
