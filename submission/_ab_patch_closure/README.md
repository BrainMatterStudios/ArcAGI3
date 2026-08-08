# Patch-closure gate — two-kernel eval-geometry A/B (2026-08-04 plan)

Compares byte-reconstructed **v7** behaviour (`base`: watchdog stall 900s,
animation OFF, graph OFF — pinned explicitly, never inferred from code
defaults) against the approved three-mechanism **candidate** delta
(`TAAF_WATCHDOG_STALL_S=600`, `TAAF_ANIMATION=1`, `TAAF_GRAPH=1`). Everything
else — including the f0ec605 refill-aware HUD unmask guard — is shared base,
byte-identical in both arms. NOT competition submissions: two private GPU
commit kernels on the `submission/_rig` mechanism (28 competition-sim clones,
7920s per game, concurrency 28), each writing a frozen
`patch_closure_result.json`. The pre-registered classifier
(`patch_closure_config.classify_result`) emits GO / NO_GO / INFRA_FAILURE /
INVALID; its thresholds are frozen in `test_patch_closure_config.py`.

Files:

- `patch_closure_config.py` — frozen arm envs, geometry, thresholds, classifier
- `classify.py` — JSON CLI over the classifier (exit 0 = GO/NO_GO, 2 otherwise)
- `pc_driver.py` — single-arm run driver (inlined into both kernels AND
  imported by the dry run)
- `build_patch_closure.py` — builds `base/` and `candidate/` kernels from
  duck-base + duck_patches.py; the arm env pin is the only delta
- `dry_run.py` — GPU-free end-to-end (mock brain, real harness, real
  competition server, full 28-clone geometry, both arms, NO_GO verdict)

## Preflight (read-only, run before any push)

```bash
PYTHONPATH=reference/arc-agi-toolkit .venv/bin/python -m pytest \
  submission/_ab_patch_closure submission/_duck_patched -q
.venv/bin/python submission/_ab_patch_closure/build_patch_closure.py --all
git diff --check
```

Note `PYTHONPATH=reference/arc-agi-toolkit`: the venv's `arc_agi` 0.9.1
predates `OperationMode.COMPETITION`, which the harness suites and the
competition-sim server need. `dry_run.py` inserts the toolkit path itself, so
the plain `.venv/bin/python submission/_ab_patch_closure/dry_run.py`
invocation also works standalone.

The build must report the SAME `base_sha256`/`patch_sha256` for both arms and
DISTINCT `arm_env_fp` fingerprints. Rebuild + byte-verify before pushing if
`duck_patches.py` or `duck-base.ipynb` changed since the last build.

## GPU execution (approval-gated — do NOT run without Ahmed's explicit go)

Push both arms (the `--accelerator` flag is REQUIRED; metadata
`machine_shape` is inert — July-proven on the sft kernels):

```bash
kaggle kernels push -p submission/_ab_patch_closure/base --accelerator NvidiaRtxPro6000
kaggle kernels push -p submission/_ab_patch_closure/candidate --accelerator NvidiaRtxPro6000
```

Each kernel is a ~2.2h wave plus model load. Check status:

```bash
kaggle kernels status ahmedmobasher86/arc-agi-3-patch-closure-base
kaggle kernels status ahmedmobasher86/arc-agi-3-patch-closure-candidate
```

## Pull outputs + classify

```bash
kaggle kernels output ahmedmobasher86/arc-agi-3-patch-closure-base -p scratchpad/patch_closure/base
kaggle kernels output ahmedmobasher86/arc-agi-3-patch-closure-candidate -p scratchpad/patch_closure/candidate
.venv/bin/python submission/_ab_patch_closure/classify.py \
  scratchpad/patch_closure/base/patch_closure_result.json \
  scratchpad/patch_closure/candidate/patch_closure_result.json
```

## Package screen (probe infrastructure, approved 2026-08-09)

Candidate-only screen on the same machinery: `package/` kernel
(`ahmedmobasher86/arc-agi-3-package-screen`), arm env = BASE_ENV + the four
package flags (`TAAF_DIFF_LINES/TAAF_WIGGLE/TAAF_RUN_PROBE/TAAF_DISPATCH=1`),
read against the banked closure base pair (11 / 12 levels excl-ft09) with
`classify_package.py` (ADVANCE bars pre-registered in
`package_screen_config.py`: >= 18 levels excl-ft09 OR >= 2 target first
unlocks). A post-benchmark PARALLEL LOAD PROBE cell (28- then 56-stream
phases against the still-live vLLM endpoint, 5-min hard cap, failure-proof)
writes `parallel_load.json` for the patch20 verifier premise.

```bash
.venv/bin/python submission/_ab_patch_closure/build_package_screen.py
.venv/bin/python submission/_ab_patch_closure/dry_run_package.py
kaggle kernels push -p submission/_ab_patch_closure/package --accelerator NvidiaRtxPro6000
kaggle kernels output ahmedmobasher86/arc-agi-3-package-screen -p scratchpad/package_screen
.venv/bin/python submission/_ab_patch_closure/classify_package.py \
  scratchpad/package_screen/patch_closure_result.json
```

Source pinning: the builder refuses to inline a `duck_patches.py` that
differs from HEAD (uncommitted in-flight patches are the silent-drift failure
class); pin explicitly with `PC_PATCHES_REF=<ref>` when needed.

## Struct screen (structural plan channel, final kernel of the 2026-08-09 sprint)

Candidate-only screen: `struct/` kernel
(`ahmedmobasher86/arc-agi-3-struct-screen`), arm env = BASE_ENV +
`{TAAF_DIFF_LINES,TAAF_WIGGLE,TAAF_DISPATCH,TAAF_STRUCT}=1`
(`TAAF_RUN_PROBE` unset — superseded by the plan channel; `TAAF_VERIFY`
unset — one variable at a time). Read against the banked base pair (11/12
excl-ft09) AND the package screen (10) with `classify_struct.py`; same
ADVANCE bars. The headline number is ADOPTION:
`result.adoption.plan_actions_per_llm_turn` (plan-actions per
ToolAgent.analyze call; banked base ~1.0, cfeb92a mock dry run 4.65). No
parallel-load probe cell (already measured: +41%/+36%).

```bash
.venv/bin/python submission/_ab_patch_closure/build_struct_screen.py
.venv/bin/python submission/_ab_patch_closure/dry_run_struct.py
kaggle kernels push -p submission/_ab_patch_closure/struct --accelerator NvidiaRtxPro6000
kaggle kernels output ahmedmobasher86/arc-agi-3-struct-screen -p scratchpad/struct_screen
.venv/bin/python submission/_ab_patch_closure/classify_struct.py \
  scratchpad/struct_screen/patch_closure_result.json
```

## Reading rules (pre-registered; do not move after the data lands)

- **GO** — >= 2 candidate-only FIRST unlocks on the never-unlocked targets
  (`wa30 m0r0 g50t dc22`) and no regression veto.
- **NO_GO** — anything less, or `su15`/`tu93` losing >= 2 levels.
- **INFRA_FAILURE** — a run died or under-delivered rows/actions: re-run,
  don't read.
- **INVALID** — contract violation (env drift, hash mismatch, base-arm
  mechanism leak): fix the build, never interpret.
- Levels are compared EXCLUDING `ft09` (its 8-level depth drowns the unlock
  signal). Score deltas at 1 wave/arm are noise (A/A floor RMS 0.707
  levels/game-run) — read unlock events and mechanism texture
  (`animation_uptake`, `grinder`, `stale_closes`), not means.
- One wave per arm resolves only large effects; a GO here still needs the
  submission-slot confirmation the campaign plan prescribes before anything
  ships.
