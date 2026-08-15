# Qwen3.8 Emergency Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely replace the 2026-08-16 Qwen3.6 duck-p3 leaderboard slot with the materially stronger Qwen3.8 duck arm, but only after independently proving dataset completeness, exact notebook identity, a successful Kaggle commit, and single-runner slot ownership.

**Architecture:** Keep the model upload and competition submission as separate lanes. The already-running Claude Code session owns the local Hugging Face download and private Kaggle dataset upload; this lane consumes the independently visible public snapshot as the primary source and treats Ahmed's upload as a fallback. New pure verification helpers establish dataset, notebook, kernel-version, and result identities. A one-shot runner loads a generated attestation rather than hard-coding a not-yet-known Kaggle version, refuses to race duck-p3, rechecks the daily slot, and delegates the actual submission to the existing guarded path.

**Tech Stack:** Python 3.14 stdlib, pytest, Kaggle CLI/API, JSON notebooks, shell process inspection, existing `scripts/submit_gated.py` submission gate.

**Spec:** `docs/superpowers/specs/2026-08-15-qwen38-emergency-promotion-design.md`

## Global Constraints

- Do not write into `/Users/ahmed/models/qwen3.8-27b-fp8` or `/Users/ahmed/models/qwen3.8-27b-fp8-hf-snapshot`; the concurrent Claude Code session owns the active download.
- Do not start a second dataset upload while that session is alive or while its Kaggle dataset is processing.
- Do not overwrite or stage the concurrent edits in `submission/_duck_38/` until their writer is finished and the resulting diff has been reviewed.
- Use `mustangliu/qwen38-27b-fp8-hf-snapshot` as the primary launch dataset only after the structural, manifest, and live-commit gates below pass. Treat `ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot` as a durability fallback.
- Never infer remote notebook identity from a kernel slug or version label. Pull the exact remote bytes, compute the canonical code-cell hash, and bind the result to Kaggle's `scriptVersionId`.
- Never have duck-p3 and duck-q38 armed simultaneously. Q38 may take the slot only after a COMPLETE attested commit and a positive check that the p3 runner is stopped.
- Preserve all unrelated deletions, notebook edits, logs, and scratch artifacts in the dirty worktree.
- Use `.venv/bin/python -m pytest -q --import-mode=importlib`; the default pytest import mode has a known duplicate-module collection conflict.
- Local commits are allowed. Do not push, merge, deploy, or take another shared irreversible action without Ahmed's approval. The user has already authorized the private Kaggle dataset/kernel work and the one competition submission governed by this plan.

---

## Task 1: Add a deterministic Qwen3.8 snapshot verifier

**Files:**

- Create: `scripts/qwen38_snapshot.py`
- Create: `tests/test_qwen38_snapshot.py`

- [ ] **Step 1: Write failing fixture tests for a valid miniature snapshot**

Create a temporary snapshot containing a small index, `layers-0.safetensors`, `layers-1.safetensors`, `mtp.safetensors`, `outside.safetensors`, model/tokenizer configuration, and a matching `crc32.txt`. Pass an explicit miniature shard set and `min_bytes=0` so unit tests do not need 25 GB.

```python
def test_verify_snapshot_accepts_complete_fixture(tmp_path: Path):
    files = {
        "layers-0.safetensors": b"layer zero",
        "layers-1.safetensors": b"layer one",
        "mtp.safetensors": b"mtp",
        "outside.safetensors": b"outside",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    write_snapshot(tmp_path, files)
    report = verify_snapshot(
        tmp_path,
        required_weight_files=(
            "layers-0.safetensors",
            "layers-1.safetensors",
            "mtp.safetensors",
            "outside.safetensors",
        ),
        min_bytes=0,
    )
    assert report.crc_checked == len(files)
    assert report.incomplete_files == ()
```

- [ ] **Step 2: Add failing tests for every fatal integrity class**

Cover an absent index target, missing `outside.safetensors`, any `.incomplete` file, a CRC mismatch, malformed manifest entry, absent tokenizer/config, and a byte total below the threshold. Each assertion must match the offending filename in the error.

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_index_target", "layers-1.safetensors"),
        ("missing_outside", "outside.safetensors"),
        ("incomplete", ".incomplete"),
        ("crc_mismatch", "CRC32 mismatch"),
        ("missing_tokenizer", "tokenizer.json"),
    ],
)
def test_verify_snapshot_rejects_integrity_failure(tmp_path, mutation, message):
    root = complete_fixture(tmp_path)
    mutate_fixture(root, mutation)
    with pytest.raises(SnapshotError, match=re.escape(message)):
        verify_snapshot(root, required_weight_files=TEST_WEIGHTS, min_bytes=0)
```

- [ ] **Step 3: Implement the verifier and JSON CLI**

Use these public interfaces:

```python
@dataclass(frozen=True)
class SnapshotReport:
    root: str
    file_count: int
    total_bytes: int
    required_weight_files: Sequence[str]
    crc_checked: int
    incomplete_files: Sequence[str]

class SnapshotError(ValueError):
    pass
```

`crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int`

`parse_crc32(path: Path) -> dict[str, int]`

`indexed_weight_files(root: Path) -> Sequence[str]`

```python
def official_weight_files() -> Sequence[str]:
    return tuple(f"layers-{i}.safetensors" for i in range(64)) + (
        "mtp.safetensors",
        "outside.safetensors",
    )
def verify_snapshot(
    root: Path,
    *,
    required_weight_files: Sequence[str] | None = None,
    min_bytes: int = 25_000_000_000,
) -> SnapshotReport:
    """Validate index coverage, required files, byte floor, and CRC entries."""
```

The CLI must print one JSON object and exit `0` on success or print a JSON error to stderr and exit `2` on integrity failure.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib tests/test_qwen38_snapshot.py`

Expected: all tests pass.

---

## Task 2: Make the Qwen3.8 evidence report objective-aligned and explicit

**Files:**

- Create: `scripts/qwen38_result_report.py`
- Create: `tests/test_qwen38_result_report.py`
- Read: `scratchpad/qwen38/wave1_shipped_result.json`
- Read: `scratchpad/qwen38/wave1_shipped_q36_result.json`
- Reuse: `submission/_ab_patch_closure/true_score.py`

- [ ] **Step 1: Write tests that lock the two different statistics**

The report must keep the 28-row mean for provenance while using the leaderboard-aligned per-source-game maximum as the decision statistic.

```python
def test_real_q38_report_uses_leaderboard_aggregation():
    report = build_comparison(Q38_RESULT, Q36_RESULT)
    assert report["q38"]["row_mean_28"] == pytest.approx(2.5291, abs=5e-4)
    assert report["q36"]["row_mean_28"] == pytest.approx(1.4872, abs=5e-4)
    assert report["q38"]["per_game_max_mean_25"] == pytest.approx(2.7215, abs=5e-4)
    assert report["q36"]["per_game_max_mean_25"] == pytest.approx(1.6638, abs=5e-4)
    assert report["leaderboard_aligned_delta"] == pytest.approx(1.0577, abs=5e-4)
```

Add tests asserting 25 unique source games, 28 rows, 11 q38 wins, 8 ties, 6 losses, and a warning that the artifact's inherited `pre_registered_reading` describes a different experiment.

- [ ] **Step 2: Implement a dependency-light report builder**

Use these interfaces:

`row_mean(result: dict[str, Any]) -> float`

`paired_outcomes(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, int]`

`stale_preregistration_warning(result: dict[str, Any]) -> str | None`

`summarize(result: dict[str, Any]) -> dict[str, Any]`

`build_comparison(candidate_path: Path, control_path: Path) -> dict[str, Any]`

Import `per_game_true_score` from `submission/_ab_patch_closure/true_score.py` and calculate all comparison fields from those maps. The CLI must emit stable, sorted, indented JSON.

- [ ] **Step 3: Generate the evidence artifact without rewriting the raw waves**

Run:

```sh
.venv/bin/python scripts/qwen38_result_report.py \
  scratchpad/qwen38/wave1_shipped_result.json \
  scratchpad/qwen38/wave1_shipped_q36_result.json \
  > scratchpad/qwen38/wave1_objective_report.json
```

The generated report may remain untracked if `scratchpad/` is ignored; the tests are the durable lock.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib tests/test_qwen38_result_report.py submission/_ab_patch_closure/test_true_score.py`

Expected: objective-aligned aggregation and the independent banked-wave checks pass.

---

## Task 3: Verify the public Kaggle dataset without duplicating the upload

**Files:**

- Create: `scripts/qwen38_dataset_gate.py`
- Create: `tests/test_qwen38_dataset_gate.py`
- Read-only remote: `mustangliu/qwen38-27b-fp8-hf-snapshot`
- Optional fallback remote: `ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot`

- [ ] **Step 1: Write tests for remote listing normalization and structural gates**

Keep subprocess and network calls outside the pure verifier. Tests feed normalized `{name, size}` rows.

```python
@dataclass(frozen=True)
class DatasetReport:
    ref: str
    file_count: int
    total_bytes: int
    weight_files: Sequence[str]
    manifest_present: bool

def verify_dataset_rows(
    ref: str,
    rows: list[dict[str, object]],
    *,
    min_bytes: int = 25_000_000_000,
) -> DatasetReport:
    """Validate remote file rows without making network calls."""
```

Assert all 64 layer shards, `mtp.safetensors`, `outside.safetensors`, `model.safetensors.index.json`, `crc32.txt`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`. Reject duplicate names, zero-length required weights, and total size below the gate.

- [ ] **Step 2: Implement read-only Kaggle listing collection**

Invoke the Kaggle Python API/CLI only to list file metadata; never invoke `datasets create` or `datasets version` from this script. Add `--ref`, `--json-output`, and `--min-bytes` arguments. Print a stable JSON report.

- [ ] **Step 3: Cross-check the manifest source**

Download only `crc32.txt`, `model.safetensors.index.json`, and `config.json` from the public dataset into a fresh `mktemp -d` directory. Compare their bytes with the corresponding completed official Hugging Face snapshot only if the external downloader has finished. If the local download is still active, record `manifest_local_match: null` and rely on the successful live Kaggle load gate in Task 6; do not touch the local files.

- [ ] **Step 4: Capture both dataset states**

Run the gate against the public dataset. Then query Ahmed's private dataset read-only:

```sh
.venv/bin/python scripts/qwen38_dataset_gate.py \
  --ref mustangliu/qwen38-27b-fp8-hf-snapshot \
  --json-output logs/qwen38_public_dataset_gate.json
kaggle datasets files ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot --page-size 200
```

If Ahmed's ref is absent or processing, continue with the verified public source. If it is complete and byte-identical, retain it only as fallback for a later kernel version; do not change the imminent attested version after it reaches COMPLETE.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib tests/test_qwen38_dataset_gate.py`

---

## Task 4: Lock the builder to a single-variable, reproducible artifact

**Files:**

- Modify after concurrent handoff: `submission/_duck_38/build_duck_38.py`
- Regenerate after concurrent handoff: `submission/_duck_38/duck-38.ipynb`
- Review after concurrent handoff: `submission/_duck_38/kernel-metadata.json`
- Create: `tests/test_duck_38_builder.py`

- [ ] **Step 1: Wait for a stable handoff before editing**

Confirm the writer process for the other Claude Code session is no longer modifying the three `_duck_38` files. Record hashes twice at least 30 seconds apart. Do not stage or rewrite them while the hashes change.

- [ ] **Step 2: Write builder contract tests before changing the builder**

The tests must build to a temporary output, not overwrite the tracked notebook. Refactor `main()` to accept `base: Path = BASE` and `out: Path = OUT`, or expose `build_notebook(base: Path) -> dict`.

```python
def test_q38_builder_is_exact_model_swap(tmp_path: Path):
    candidate = build_notebook(BASE)
    source = code_source(candidate)
    assert "mustangliu/qwen38-27b-fp8-hf-snapshot" in source
    assert "Qwen/Qwen3.8-27B-FP8" in source
    assert "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot" not in source
    assert "_38_missed" in source
    assert "_d38_wheel_dirs" in source
    assert "_d38_env_candidates" in source
```

Add a structural diff whitelist: relative to `duck-base.ipynb`, only the dataset source, the three asserted setup-command swaps, and the two commit-only mount-probe cells may differ. Assert every executable cell has cleared outputs and `execution_count is None`.

- [ ] **Step 3: Make the builder's primary/fallback ownership explicit**

Define:

```python
PRIMARY_SNAPSHOT_REF = "mustangliu/qwen38-27b-fp8-hf-snapshot"
FALLBACK_SNAPSHOT_REF = "ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot"
NEW_SNAPSHOT_REF = PRIMARY_SNAPSHOT_REF
```

Update the module docstring so it no longer claims Ahmed's dataset is the current kernel source. Preserve the asserted three-swap mechanism and fail-loud behavior.

- [ ] **Step 4: Regenerate and prove determinism**

Run the builder twice and compare the canonical code-cell hash and the full notebook SHA-256 across runs. Both must be identical. Verify `kernel-metadata.json` and notebook `DATASET_SOURCES` name the same public ref.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib tests/test_duck_38_builder.py`

---

## Task 5: Add reusable Kaggle kernel attestation

**Files:**

- Create: `scripts/qwen38_attest.py`
- Create: `tests/test_qwen38_attest.py`

- [ ] **Step 1: Write tests using injected API/status functions**

Tests must never call Kaggle. Cover a passing attestation, remote/local hash mismatch, wrong dataset ref, missing Qwen3.8 markers, non-COMPLETE status, missing `scriptVersionId`, and a saved-attestation recheck whose remote bytes changed.

```python
@dataclass(frozen=True)
class KernelAttestation:
    kernel: str
    version: int
    script_version_id: str
    code_sha256: str
    notebook_sha256: str
    dataset_ref: str
    status: str
    attested_at_utc: str
```

`canonical_code_hash(nb: dict[str, Any]) -> str`

`code_source(nb: dict[str, Any]) -> str`

```python
def attest_latest(
    *,
    kernel: str,
    local_notebook: Path,
    expected_dataset_ref: str,
    api_get: Callable[[str], dict | list],
    status_get: Callable[[str], str | None],
) -> KernelAttestation:
    """Bind local bytes to the exact latest COMPLETE Kaggle version."""

def reattest_saved(
    attestation: KernelAttestation,
    *,
    local_notebook: Path,
    api_get: Callable[[str], dict | list],
    status_get: Callable[[str], str | None],
) -> None:
    """Repeat identity and status checks for a saved immutable attestation."""
```

- [ ] **Step 2: Implement exact remote binding**

`attest_latest` must:

1. Positively obtain the latest version number.
2. Pull that exact `version_label=vN` notebook.
3. Compare its canonical code-cell hash and full notebook SHA-256 with the local notebook.
4. Require the public dataset ref, `Qwen/Qwen3.8-27B-FP8`, `_38_missed`, `_d38_wheel_dirs`, and `_d38_env_candidates` in code.
5. Require the old Qwen3.6 dataset ref to be absent from executable code.
6. Read the output listing for that exact version and extract exactly one `/kf/<scriptVersionId>/`.
7. Require positive `COMPLETE` kernel status.

The CLI writes the dataclass as JSON only after all checks pass. `reattest_saved` repeats every identity check against the saved version and SVID.

- [ ] **Step 3: Run focused tests**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib tests/test_qwen38_attest.py`

---

## Task 6: Push, execute, and attest the Qwen3.8 Kaggle kernel

**Files:**

- Read: `submission/_duck_38/kernel-metadata.json`
- Read: `submission/_duck_38/duck-38.ipynb`
- Generate: `logs/qwen38_attestation.json`
- Generate: `logs/qwen38_commit_status.log`

- [ ] **Step 1: Check whether the concurrent session already pushed the kernel**

Query `ahmedmobasher86/arc-agi-3-duck-38` read-only. If a current version exists, pull and compare it first. Do not push a duplicate version when the remote already matches the deterministic local notebook and dataset ref.

- [ ] **Step 2: Push exactly once if no matching version exists**

Run from `submission/_duck_38`:

```sh
python3 -m kaggle kernels push -p .
```

The metadata already requests `NvidiaRtxPro6000`, internet off, and the ARC competition source. A push is allowed only after Tasks 3-5 pass.

- [ ] **Step 3: Poll to a positive terminal state**

Poll at one-minute intervals. Transport noise is unreadable, never a kernel verdict. Abort the promotion on positive ERROR/CANCEL. On COMPLETE, download the commit log and require evidence that:

- the dataset mounted;
- the setup swap fired;
- the served name is `Qwen/Qwen3.8-27B-FP8`;
- model loading completed;
- the output parquet was produced by the commit path.

A COMPLETE status without Qwen3.8 identity evidence proves wiring only and is insufficient to take the slot.

- [ ] **Step 4: Generate the immutable attestation**

Run:

```sh
.venv/bin/python scripts/qwen38_attest.py \
  --kernel ahmedmobasher86/arc-agi-3-duck-38 \
  --notebook submission/_duck_38/duck-38.ipynb \
  --dataset-ref mustangliu/qwen38-27b-fp8-hf-snapshot \
  --output logs/qwen38_attestation.json
```

Read back and display the exact kernel version, scriptVersionId, code hash, full notebook hash, dataset ref, and status.

---

## Task 7: Add a one-shot Qwen3.8 runner with explicit p3 arbitration

**Files:**

- Create: `scripts/submit_q38_20260816.py`
- Create: `tests/test_submit_q38_20260816.py`
- Reuse: `scripts/submit_gated.py`
- Read: `logs/qwen38_attestation.json`

- [ ] **Step 1: Write pure decision and process-guard tests**

Use these interfaces:

```python
TARGET_UTC = datetime(2026, 8, 16, 0, 1, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 16, 2, 0, tzinfo=timezone.utc)
Q38_MARKER = REPO / "logs/duck_q38_20260816.marker"
P3_MARKER = REPO / "logs/duck_p3_20260816.marker"
```

`p3_runner_pids(process_rows: Sequence[str]) -> Sequence[int]`

`validate_single_runner(p3_pids: Sequence[int], self_pid: int) -> None`

`slot_already_used(rows: list[dict[str, Any]], day_start: datetime) -> bool`

`build_submit_command(attestation: KernelAttestation, mock: bool) -> list[str]`

Test that any live `submit_p3_20260816.py` PID is fatal, unrelated Python processes are ignored, an existing p3 or q38 marker is fatal, a submission on the UTC day is fatal, and the generated command uses the attested version rather than a source constant.

- [ ] **Step 2: Implement the one-shot runner**

Copy the proven retry, time-window, marker-release, and post-submit semantics from `submit_p3_20260816.py`, but load `KernelAttestation` from JSON and call `reattest_saved` immediately before claiming the slot. The message must state the measured evidence honestly:

```python
MESSAGE = (
    "Qwen3.8 single-variable brain swap: shipped duck-base behavior with only "
    "the served model/dataset changed to Qwen/Qwen3.8-27B-FP8. Local 25-game "
    "screen: leaderboard-aligned per-game-max mean 2.7215 vs Qwen3.6 1.6638 "
    "(+1.0577); 11 wins, 8 ties, 6 losses. Directional screen only: paired "
    "bootstrap includes zero, so this submission is the live falsification."
)
```

Before sleeping toward the target, positively scan the process list and refuse to arm if p3 is alive. Repeat that check immediately before slot claim. The runner itself must never send a signal to another process; stopping p3 is an explicit operator step after Q38 attestation.

- [ ] **Step 3: Preserve the submission failure lesson**

If `submit_gated.py` returns non-zero, retain the Q38 marker when the submissions API proves a submission landed that UTC day. Release it only when no submission exists. Never launch p3 after Q38 ERROR unless Kaggle positively proves the daily slot is still unused.

- [ ] **Step 4: Run mock and focused tests**

Run:

```sh
.venv/bin/python -m pytest -q --import-mode=importlib tests/test_submit_q38_20260816.py
.venv/bin/python scripts/submit_q38_20260816.py --mock \
  --attestation logs/qwen38_attestation.json
```

Expected: all local guards execute, the remote attestation passes, and `submit_gated.py --dry-run` reaches its pre-submit success without submitting.

---

## Task 8: Full verification and local commit

**Files:**

- Modify only files named in Tasks 1-7 and the already-owned Q38 builder trio after handoff.

- [ ] **Step 1: Run the targeted Q38 suite**

Run:

```sh
.venv/bin/python -m pytest -q --import-mode=importlib \
  tests/test_qwen38_snapshot.py \
  tests/test_qwen38_result_report.py \
  tests/test_qwen38_dataset_gate.py \
  tests/test_duck_38_builder.py \
  tests/test_qwen38_attest.py \
  tests/test_submit_q38_20260816.py \
  submission/_ab_patch_closure/test_true_score.py
```

- [ ] **Step 2: Run the complete repository suite**

Run: `.venv/bin/python -m pytest -q --import-mode=importlib`

Baseline evidence is `589 passed, 7 deselected, 1 xfailed`; investigate any regression or changed count before proceeding.

- [ ] **Step 3: Review only the intended diff**

Use explicit paths. Confirm no model files, credentials, Kaggle logs, unrelated notebooks, scratch deletions, or environment files are staged.

- [ ] **Step 4: Commit the implementation locally**

Commit message:

```text
feat(qwen38): gate emergency leaderboard promotion

- verify public snapshot structure and objective-aligned evidence
- attest exact Kaggle kernel identity before slot ownership
- arbitrate the Qwen3.8 runner against the Qwen3.6 fallback

Co-Authored-By: Codex <noreply@openai.com>
```

---

## Task 9: Transfer the 2026-08-16 slot and launch

**Files:**

- Read: `logs/qwen38_attestation.json`
- Create at runtime: `logs/duck_q38_20260816.marker`
- Existing fallback: `scripts/submit_p3_20260816.py`

- [ ] **Step 1: Apply the 2026-08-15 23:30 UTC cutoff**

At or before 23:30 UTC, Q38 is eligible only if all of the following are positive:

- public or fallback dataset gate passed;
- remote Q38 kernel is COMPLETE;
- commit logs prove Qwen3.8 identity and output production;
- saved attestation rechecks exactly;
- mock runner passed;
- full local test suite passed;
- Kaggle reports the 2026-08-16 daily slot unused;
- the other download/upload session is not writing shared Q38 files.

If any item is unknown or negative, retain duck-p3 and stop Q38 work for this slot.

- [ ] **Step 2: Stop duck-p3 only after Q38 is eligible**

Resolve exact p3 runner PIDs using full command lines. Send `TERM` only to PIDs whose command contains the repository's exact `scripts/submit_p3_20260816.py` path. Re-read the process table and require zero matches. Do not use a broad process-name kill.

- [ ] **Step 3: Arm one Q38 runner**

Launch exactly one process:

```sh
.venv/bin/python scripts/submit_q38_20260816.py \
  --attestation logs/qwen38_attestation.json
```

Capture its PID and log path. Re-read the process table and require one Q38 runner and zero p3 runners.

- [ ] **Step 4: Let the guarded path submit after the target**

The runner targets 00:01 UTC, reattests, checks the daily race, claims the marker, then invokes `submit_gated.py`. The existing gate requires COMPLETE plus a ten-minute settle, so a healthy competition submission should land around 00:11 UTC, inside the declared 02:00 cutoff.

---

## Task 10: Monitor the real run and update the evidence ledger

**Files:**

- Modify after a real submission appears: `docs/submission-ledger.json`
- Create/update runtime logs under: `logs/`

- [ ] **Step 1: Watch the first 90 minutes**

Record submission ID, time, status, total bytes, description, kernel version, SVID, and hashes. Alarm on a parquet at or below 3500 bytes, ERROR, or implausibly fast COMPLETE under one hour.

- [ ] **Step 2: Follow through the expected full runtime**

A genuine run historically takes roughly nine hours. Continue checking until COMPLETE or ERROR. Do not submit another arm while this draw is in flight.

- [ ] **Step 3: Apply the pre-registered reading**

- `score > 1.30`: promote Qwen3.8 as the new base and use future slots for targeted high-upside behavioral improvements.
- `0.69 <= score <= 1.30`: one live draw is inconclusive; retain Qwen3.8's local advantage but seek another controlled live draw before a permanent base change.
- `0 < score < 0.69`: material negative evidence; inspect game coverage and infrastructure before another Q38 submission.
- `score == 0` or `ERROR`: treat as infrastructure/identity failure first, not a model-quality verdict.

- [ ] **Step 4: Commit only the final ledger update locally**

Use a conventional `docs:` commit with the submission ID and factual result. Stop before any git push or merge.

---

## Plan Self-Review

- [x] Every design-spec gate maps to a task: snapshot integrity (Tasks 1/3), objective evidence (Task 2), single-variable builder (Task 4), exact kernel identity (Tasks 5/6), single-runner arbitration (Task 7/9), guarded submission (Task 9), and monitoring/ledger (Task 10).
- [x] Concurrent Claude Code ownership is explicit; this plan never launches a duplicate model download or dataset upload.
- [x] The public dataset removes the upload from the critical path but does not bypass integrity or live-load evidence.
- [x] Unknown future Kaggle version/SVID values are runtime attestation data, not source placeholders.
- [x] All new interfaces have concrete signatures and failure tests.
- [x] No unresolved placeholder, secret, or production write is embedded in the implementation instructions.
- [x] The irreversible competition action remains bounded to one pre-authorized, gated daily submission.
