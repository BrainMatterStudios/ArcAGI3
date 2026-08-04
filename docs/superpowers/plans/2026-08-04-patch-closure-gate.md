# Patch Closure Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-kernel, eval-geometry A/B that compares byte-reconstructed v7 behavior against animation + level-age grinder/narration + 600-second heartbeat and emits a frozen GO/NO_GO result.

**Architecture:** Reuse the existing `submission/_rig` competition-sim runner and `submission/_duck_patched/duck_patches.py`. Generate one notebook per arm from the same source notebook; the arm environment mapping is the only behavioral delta. Analyze the two machine-readable results locally with pre-registered thresholds.

**Tech Stack:** Python 3.12, pytest, Jupyter notebook JSON, TAAF competition simulator, Kaggle CLI, vLLM.

## Global Constraints

- Do not modify or delete user-owned changes outside the listed files.
- Use 28 competition-sim clones, 7,920 seconds per game and concurrency 28.
- Base arm explicitly pins v7 defaults; do not infer them from current code defaults.
- Candidate delta is limited to animation, level-age grinder/narration and watchdog stall 600 seconds.
- Do not run `kaggle kernels push` or submit to the competition without Ahmed's explicit approval.
- Use `.venv/bin/python`; do not use the system Python for tests or builds.
- Commits use Conventional Commits and the required Codex co-author footer.

---

### Task 1: Freeze the arm contract

**Files:**
- Create: `submission/_ab_patch_closure/patch_closure_config.py`
- Create: `submission/_ab_patch_closure/classify.py`
- Test: `submission/_ab_patch_closure/test_patch_closure_config.py`

**Interfaces:**
- Produces: `BASE_ENV: dict[str, str]`, `CANDIDATE_ENV: dict[str, str]`, `ARM_ENV: dict[str, dict[str, str]]`, `GateThresholds`, `classify_result(base: dict, candidate: dict) -> dict`, and `classify.main(argv: Sequence[str] | None = None) -> int`.
- Consumes: result row schema from `submission/_rig/build_rig.py` and diagnostics exported by `submission/_duck_patched/duck_patches.py`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_candidate_delta_is_exactly_the_approved_three_mechanisms():
    changed = {k for k in BASE_ENV | CANDIDATE_ENV if BASE_ENV.get(k) != CANDIDATE_ENV.get(k)}
    assert changed == {"TAAF_WATCHDOG_STALL_S", "TAAF_ANIMATION", "TAAF_GRAPH"}

def test_base_reconstructs_v7_changed_defaults():
    assert BASE_ENV["TAAF_WATCHDOG_STALL_S"] == "900"
    assert BASE_ENV["TAAF_ANIMATION"] == "0"
    assert BASE_ENV["TAAF_GRAPH"] == "0"

def test_candidate_pins_grinder_parameters():
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_AGE_ACTIONS"] == "120"
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_AGE_TURNS"] == "10"
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_MAX_PER_LEVEL"] == "2"
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure/test_patch_closure_config.py -q`

Expected: FAIL because `patch_closure_config.py` does not exist.

- [ ] **Step 3: Implement the frozen mappings and classifier**

```python
BASE_ENV = {
    "TAAF_WATCHDOG": "1", "TAAF_WATCHDOG_STALL_S": "900",
    "TAAF_HUD_MASK": "1", "TAAF_WIN_REPLAY": "1",
    "TAAF_ANTIFREEZE": "1", "TAAF_ANIMATION": "0",
    "TAAF_GRAPH": "0", "TAAF_GRAPH_GRIND_AGE_ACTIONS": "120",
    "TAAF_GRAPH_GRIND_AGE_TURNS": "10",
    "TAAF_GRAPH_GRIND_MAX_PER_LEVEL": "2",
    "TAAF_COMPACT": "0", "TAAF_PLAYBOOK": "0",
    "TAAF_GRID_BURNER": "0",
}
CANDIDATE_ENV = {
    **BASE_ENV,
    "TAAF_WATCHDOG_STALL_S": "600",
    "TAAF_ANIMATION": "1",
    "TAAF_GRAPH": "1",
}
```

`classify_result` must return one of `GO`, `NO_GO`, `INFRA_FAILURE`, or `INVALID`; it computes levels excluding `ft09`, target-only first unlocks, `su15`/`tu93` regression, animation uptake, grinder engagement and stale-close count.

Implement `classify.py` as a thin JSON CLI:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    result = classify_result(
        json.loads(args.base.read_text()),
        json.loads(args.candidate.read_text()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {"GO", "NO_GO"} else 2
```

- [ ] **Step 4: Add decision-boundary tests**

```python
def test_go_on_two_new_target_unlocks(valid_pair):
    valid_pair.candidate["rows_by_source"]["dc22"]["levels"] = 1
    valid_pair.candidate["rows_by_source"]["m0r0"]["levels"] = 1
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "GO"

def test_no_go_when_su15_regresses_two_levels(valid_pair):
    valid_pair.base["rows_by_source"]["su15"]["levels"] = 3
    valid_pair.candidate["rows_by_source"]["su15"]["levels"] = 1
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "NO_GO"

def test_invalid_when_identity_proof_is_missing(valid_pair):
    del valid_pair.candidate["identity"]
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"
```

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure/test_patch_closure_config.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add submission/_ab_patch_closure/patch_closure_config.py submission/_ab_patch_closure/classify.py submission/_ab_patch_closure/test_patch_closure_config.py
git commit -q -m "test(patch-gate): freeze eval arm contract" -m "- Reconstruct v7 defaults explicitly
- Pre-register patch closure decision boundaries

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 2: Build one notebook per arm

**Files:**
- Create: `submission/_ab_patch_closure/build_patch_closure.py`
- Create: `submission/_ab_patch_closure/base/kernel-metadata.json`
- Create: `submission/_ab_patch_closure/candidate/kernel-metadata.json`
- Generate: `submission/_ab_patch_closure/base/patch-closure-base.ipynb`
- Generate: `submission/_ab_patch_closure/candidate/patch-closure-candidate.ipynb`
- Test: `submission/_ab_patch_closure/test_build_patch_closure.py`

**Interfaces:**
- Consumes: `ARM_ENV` from Task 1, `submission/_rig/build_rig.py`, `submission/_rig/behav_probe.py`, and `submission/_duck_patched/duck_patches.py`.
- Produces: `build_arm(arm: Literal["base", "candidate"], output_root: pathlib.Path | None = None) -> pathlib.Path` and two locally compilable notebooks.

- [ ] **Step 1: Write failing builder tests**

```python
def test_built_notebooks_share_base_hash_and_differ_only_in_arm_env(tmp_path):
    base = build_arm("base", output_root=tmp_path)
    candidate = build_arm("candidate", output_root=tmp_path)
    assert notebook_contract(base)["source_base_sha256"] == notebook_contract(candidate)["source_base_sha256"]
    assert notebook_contract(base)["arm_env"] == BASE_ENV
    assert notebook_contract(candidate)["arm_env"] == CANDIDATE_ENV

def test_metadata_is_private_commit_kernel():
    for arm in ("base", "candidate"):
        meta = json.loads(Path(f"submission/_ab_patch_closure/{arm}/kernel-metadata.json").read_text())
        assert meta["is_private"] is True
        assert meta["enable_internet"] is False
        assert meta["machine_shape"] == "NvidiaRtxPro6000"
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure/test_build_patch_closure.py -q`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement the builder**

Copy the notebook transformation pattern from `submission/_rig/build_rig.py`, but inline the complete patch module in both arms and call `apply_all()`. Insert the selected `ARM_ENV` immediately before `apply_all()`. Force `RIG_GAMES=28`, `RIG_BUDGET=7920`, `RIG_REPEATS=1`, include the behavioral probe, and write these immutable fields to `patch_closure_result.json`:

```python
{
    "schema_version": 1,
    "hypothesis": "patch-closure-2026-08-04",
    "arm": arm,
    "arm_env": ARM_ENV[arm],
    "source_base_sha256": source_hash,
    "patch_sha256": patch_hash,
    "geometry": {"clones": 28, "per_game_s": 7920, "concurrency": 28},
    "identity": identity_proof,
    "rows": rows,
    "behavior": behav_report(),
    "patch_diagnostics": collect_patch_diagnostics(),
}
```

- [ ] **Step 4: Generate and statically validate both notebooks**

Run: `.venv/bin/python submission/_ab_patch_closure/build_patch_closure.py --all`

Expected: two notebook paths, matching base hash, distinct arm environment fingerprints, and every code cell compiling.

- [ ] **Step 5: Run focused and existing patch tests**

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure submission/_duck_patched -q`

Expected: PASS with no failures.

- [ ] **Step 6: Commit**

```bash
git add submission/_ab_patch_closure
git commit -q -m "feat(patch-gate): build eval-geometry closure arms" -m "- Generate isolated v7 and candidate commit kernels
- Capture identity, token and mechanism diagnostics

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 3: Add a GPU-free end-to-end dry-run

**Files:**
- Create: `submission/_ab_patch_closure/dry_run.py`
- Modify: `submission/_ab_patch_closure/test_build_patch_closure.py`

**Interfaces:**
- Consumes: generated notebook run-cell contract.
- Produces: `dry_run_arm(arm: str, output_dir: Path) -> dict` with the same schema as a GPU result.

- [ ] **Step 1: Write a failing dry-run schema test**

```python
def test_dry_run_emits_classifiable_pair(tmp_path):
    base = dry_run_arm("base", tmp_path / "base")
    candidate = dry_run_arm("candidate", tmp_path / "candidate")
    assert base["schema_version"] == candidate["schema_version"] == 1
    assert classify_result(base, candidate)["state"] in {"GO", "NO_GO"}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure/test_build_patch_closure.py::test_dry_run_emits_classifiable_pair -q`

Expected: FAIL because `dry_run_arm` is absent.

- [ ] **Step 3: Implement deterministic fake sessions**

Reuse the stub-game pattern from `submission/_ab_wmr/dry_run.py`. Emit 28 rows, captured animation, one level-age grinder engagement, tool/token counters and a complete identity block. Do not encode a forced GO; use equal levels so the default dry-run verdict is `NO_GO`.

- [ ] **Step 4: Run the dry-run and full local gate suite**

Run: `.venv/bin/python submission/_ab_patch_closure/dry_run.py`

Expected: writes base and candidate JSON and prints `state=NO_GO`.

Run: `.venv/bin/python -m pytest submission/_ab_patch_closure submission/_duck_patched -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add submission/_ab_patch_closure
git commit -q -m "test(patch-gate): exercise closure result path" -m "- Emit production-shaped dry-run artifacts
- Verify frozen classifier end to end

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 4: Prepare the approval-gated GPU runbook

**Files:**
- Create: `submission/_ab_patch_closure/README.md`

**Interfaces:**
- Consumes: both generated kernel directories and classifier.
- Produces: exact read-only preflight, push, output-pull and classification commands.

- [ ] **Step 1: Document the preflight commands**

```bash
.venv/bin/python -m pytest submission/_ab_patch_closure submission/_duck_patched -q
.venv/bin/python submission/_ab_patch_closure/build_patch_closure.py --all
git diff --check
```

- [ ] **Step 2: Document the approval-gated commands without executing them**

```bash
kaggle kernels push -p submission/_ab_patch_closure/base --accelerator NvidiaRtxPro6000
kaggle kernels push -p submission/_ab_patch_closure/candidate --accelerator NvidiaRtxPro6000
kaggle kernels output ahmedmobasher86/arc-agi-3-patch-closure-base -p scratchpad/patch_closure/base
kaggle kernels output ahmedmobasher86/arc-agi-3-patch-closure-candidate -p scratchpad/patch_closure/candidate
.venv/bin/python submission/_ab_patch_closure/classify.py \
  scratchpad/patch_closure/base/patch_closure_result.json \
  scratchpad/patch_closure/candidate/patch_closure_result.json
```

- [ ] **Step 3: Verify every referenced path and command parser locally**

Run: `.venv/bin/python submission/_ab_patch_closure/classify.py --help`

Expected: exit 0 and two positional result arguments.

- [ ] **Step 4: Commit and stop at the shared-state gate**

```bash
git add submission/_ab_patch_closure/README.md
git commit -q -m "docs(patch-gate): add GPU execution runbook" -m "- Record exact preflight and artifact commands
- Preserve approval gate before Kaggle pushes

Co-Authored-By: Codex <noreply@openai.com>"
```

Do not execute either `kaggle kernels push` command until Ahmed explicitly approves the shared-state action.
