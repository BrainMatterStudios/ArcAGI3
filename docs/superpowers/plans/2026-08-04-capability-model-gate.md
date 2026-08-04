# Sparse Model Capability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether Qwen3.6-35B-A3B-FP8 or Qwen-AgentWorld-35B-A3B can reliably operate the existing duck agent and unlock at least two of four capability-wall games within production-equivalent time.

**Architecture:** Extract a model-agnostic six-game capability driver from the existing 35B notebook and EWM serving assertions. Generate one private commit kernel per model, with identical duck code, panel, sampling and budgets. A local classifier selects at most one model for a later structured-workspace A/B; this plan does not build an executable world model.

**Tech Stack:** Python 3.12, pytest, Jupyter notebook JSON, vLLM OpenAI server, Kaggle Models, TAAF/ARC offline environments.

## Global Constraints

- Panel is exactly `dc22,m0r0,sk48,tr87,ft09,su15`; the first four are targets and the last two are controls.
- Per-game production-equivalent GPU share is 283 seconds; model serve timeout is 3,600 seconds.
- Sampling is temperature 0.6, top-p 0.95 and top-k 20; context is 65,536.
- Candidate notebooks differ only in model source, resolved model selector and serve arguments required by the checkpoint format.
- The first stage is a brain-only swap in the existing duck harness. Do not add Tycho, EWM, new prompts or SFT.
- Do not run `kaggle kernels push`, create/update a Kaggle asset, or submit without Ahmed's explicit approval.
- Use `.venv/bin/python`; commits use Conventional Commits and the required Codex co-author footer.

---

### Task 1: Define the candidate and decision schema

**Files:**
- Create: `submission/_capability_gate/capability_config.py`
- Test: `submission/_capability_gate/test_capability_config.py`

**Interfaces:**
- Produces: `CandidateSpec`, `CANDIDATES`, `PANEL`, `TARGETS`, `CONTROLS`, `classify_candidate(result, baseline) -> dict`, and `select_candidate(results, baseline) -> dict`.
- Consumes: baseline rows reconstructed from existing dense-27B trace artifacts.

- [ ] **Step 1: Write failing schema tests**

```python
def test_panel_is_frozen():
    assert TARGETS == ("dc22", "m0r0", "sk48", "tr87")
    assert CONTROLS == ("ft09", "su15")
    assert PANEL == TARGETS + CONTROLS

def test_candidate_sources_are_exact():
    assert CANDIDATES["qwen36"].source == "nareshmeena07/qwen36-35b-a3b-fp8"
    assert CANDIDATES["agentworld"].source == "keras/qwen-agentworld/transformers/default/1"

def test_gate_constants():
    assert GAME_BUDGET_S == 283
    assert SERVE_TIMEOUT_S == 3600
    assert MIN_PROTOCOL_SUCCESS == pytest.approx(0.95)
    assert MIN_TARGET_UNLOCKS == 2
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_capability_config.py -q`

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement immutable dataclasses and classification**

```python
@dataclass(frozen=True)
class CandidateSpec:
    key: str
    source: str
    path_token: str
    served_name: str
    quantization: str | None

def protocol_success_rate(result: dict) -> float:
    attempts = int(result["protocol"]["assistant_turns"])
    successes = int(result["protocol"]["successful_tool_executions"])
    return successes / max(attempts, 1)
```

`classify_candidate` must reject CPU offload, server crashes, missing identity fields, under-95% protocol success, fewer than two target unlocks, control regression of two or more levels, or any over-budget game. `select_candidate` applies target unlocks, target levels, protocol success and tokens/second as ordered tie-breakers.

- [ ] **Step 4: Add boundary and tie-break tests**

```python
def test_exactly_two_target_unlocks_and_95_percent_protocol_pass(valid_result, baseline):
    valid_result["protocol"] = {
        "assistant_turns": 20,
        "parseable_tool_attempts": 19,
        "successful_tool_executions": 19,
    }
    assert classify_candidate(valid_result, baseline)["state"] == "GO"

def test_cpu_offload_is_infra_failure(valid_result, baseline):
    valid_result["serve"]["cpu_offload"] = True
    assert classify_candidate(valid_result, baseline)["state"] == "INFRA_FAILURE"

def test_tie_break_prefers_more_target_levels(two_valid_results, baseline):
    selected = select_candidate(two_valid_results, baseline)
    assert selected["selected"] == "agentworld"
```

- [ ] **Step 5: Run and commit**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_capability_config.py -q`

Expected: PASS.

```bash
git add submission/_capability_gate/capability_config.py submission/_capability_gate/test_capability_config.py
git commit -q -m "test(capability): freeze sparse-model gate" -m "- Pin model sources, panel and resource limits
- Encode capability and reliability thresholds

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 2: Extract model identity and serving assertions

**Files:**
- Create: `submission/_capability_gate/serving_probe.py`
- Test: `submission/_capability_gate/test_serving_probe.py`

**Interfaces:**
- Consumes: `CandidateSpec` and an OpenAI-compatible server at `http://127.0.0.1:1234/v1`.
- Produces: `resolve_model_path(spec: CandidateSpec, roots: Sequence[Path]) -> Path`, `collect_static_identity(path: Path) -> dict`, and `probe_server(base_url: str, spec: CandidateSpec) -> dict`.

- [ ] **Step 1: Write failing tests for ambiguous and incorrect model roots**

```python
def test_resolve_model_path_requires_one_matching_config(tmp_path, qwen36_spec):
    make_model(tmp_path / "a", name="qwen36")
    make_model(tmp_path / "b", name="qwen36")
    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_model_path(qwen36_spec, [tmp_path])

def test_static_identity_hashes_weight_index(model_dir):
    identity = collect_static_identity(model_dir)
    assert len(identity["weight_index_sha256"]) == 64
    assert identity["realpath"] == str(model_dir.resolve())
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_serving_probe.py -q`

Expected: FAIL because `serving_probe.py` is absent.

- [ ] **Step 3: Implement static and live proofs**

Reuse the path/cmdline/logprob checks from `submission/_serve_verify_k3/serving_assert.py` and `submission/_ewm_gate3/build_ewm_gate3.py`. The live probe must return:

```python
{
    "models_response": models_json,
    "server_cmdline": cmdline,
    "temperature_zero_fingerprint": fingerprint,
    "completion_tokens": completion_tokens,
    "elapsed_s": elapsed,
    "tokens_per_second": completion_tokens / elapsed,
    "cpu_offload": detect_cpu_offload(server_log),
}
```

- [ ] **Step 4: Test with a local fake OpenAI server**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_serving_probe.py -q`

Expected: PASS, including `/models`, chat-completions and incorrect-served-name cases.

- [ ] **Step 5: Commit**

```bash
git add submission/_capability_gate/serving_probe.py submission/_capability_gate/test_serving_probe.py
git commit -q -m "feat(capability): verify served model identity" -m "- Resolve checkpoints without first-match ambiguity
- Record logprob, throughput and offload proofs

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 3: Build the six-game capability driver

**Files:**
- Create: `submission/_capability_gate/capability_driver.py`
- Test: `submission/_capability_gate/test_capability_driver.py`

**Interfaces:**
- Consumes: a healthy served model alias, `PANEL`, normal duck benchmark objects and `probe_server` output.
- Produces: `run_capability_gate(candidate: CandidateSpec, working_dir: Path) -> dict` and `capability_result.json`.

- [ ] **Step 1: Write failing protocol-accounting tests**

```python
def test_protocol_rate_counts_parseable_attempts_not_assistant_turns(fake_sessions):
    result = summarize_sessions(fake_sessions)
    assert result["protocol"] == {
        "assistant_turns": 7,
        "parseable_tool_attempts": 5,
        "successful_tool_executions": 4,
        "rejected_actions": 1,
    }

def test_every_panel_game_has_a_row(fake_benchmark):
    result = run_capability_gate(CANDIDATES["qwen36"], fake_benchmark.working_dir)
    assert tuple(row["source_game"] for row in result["rows"]) == PANEL
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_capability_driver.py -q`

Expected: FAIL because the driver is absent.

- [ ] **Step 3: Implement the driver using the ordinary duck harness**

Construct six offline `GameAPI` objects in the frozen order, set concurrency to six, set both supported per-game runtime attributes to 283, and leave prompts/patches at the byte-reconstructed v7 pins. Wrap the existing analyzer and sandbox call sites only to observe:

```python
PROTOCOL_COUNTERS = {
    "assistant_turns": 0,
    "parseable_tool_attempts": 0,
    "successful_tool_executions": 0,
    "rejected_actions": 0,
    "timeouts": 0,
}
```

Write the result incrementally after serve probe and after every game so a partial infrastructure failure remains diagnosable.

- [ ] **Step 4: Run stub-brain end-to-end tests**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_capability_driver.py -q`

Expected: PASS for a valid action sequence, malformed Python, rejected action, timeout and partial-write case.

- [ ] **Step 5: Commit**

```bash
git add submission/_capability_gate/capability_driver.py submission/_capability_gate/test_capability_driver.py
git commit -q -m "feat(capability): add six-game behavior driver" -m "- Preserve production budgets and duck behavior
- Capture protocol, token and level outcomes

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 4: Generate one notebook per candidate

**Files:**
- Create: `submission/_capability_gate/build_capability_gate.py`
- Create: `submission/_capability_gate/qwen36/kernel-metadata.json`
- Create: `submission/_capability_gate/agentworld/kernel-metadata.json`
- Generate: `submission/_capability_gate/qwen36/capability-qwen36.ipynb`
- Generate: `submission/_capability_gate/agentworld/capability-agentworld.ipynb`
- Test: `submission/_capability_gate/test_build_capability_gate.py`

**Interfaces:**
- Consumes: the existing `submission/_model35b/duck-model35b.ipynb`, driver from Task 3, and serving probe from Task 2.
- Produces: `build_candidate(key: Literal["qwen36", "agentworld"], output_root: Path | None = None) -> Path`.

- [ ] **Step 1: Write failing notebook-contract tests**

```python
def test_candidates_differ_only_in_model_contract(tmp_path):
    q = inspect_contract(build_candidate("qwen36", output_root=tmp_path))
    a = inspect_contract(build_candidate("agentworld", output_root=tmp_path))
    assert q["driver_sha256"] == a["driver_sha256"]
    assert q["panel"] == a["panel"] == list(PANEL)
    assert q["sampling"] == a["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert q["model_source"] != a["model_source"]
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_build_capability_gate.py -q`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement candidate-specific metadata**

Qwen3.6 metadata keeps the existing wheelhouse, TAAF source and dataset:

```json
{
  "id": "ahmedmobasher86/arc-agi-3-capability-qwen36",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "machine_shape": "NvidiaRtxPro6000",
  "dataset_sources": [
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "ahmedmobasher86/taaf-src-model35b",
    "nareshmeena07/qwen36-35b-a3b-fp8"
  ],
  "competition_sources": ["arc-prize-2026-arc-agi-3"]
}
```

AgentWorld uses the same wheelhouse and TAAF source plus:

```json
"model_sources": ["keras/qwen-agentworld/transformers/default/1"]
```

Start AgentWorld with dynamic FP8 only if its config is not already quantized:

```python
serve_args = ["--quantization", "fp8"] if not config.get("quantization_config") else []
```

The serve log must prove that no CPU offload occurred. A failure to fit is `INFRA_FAILURE`, not a behavioral `NO_GO`.

- [ ] **Step 4: Build and statically validate both notebooks**

Run: `.venv/bin/python submission/_capability_gate/build_capability_gate.py --all`

Expected: both notebooks generated, all code cells compile, exact panel and model-source contracts printed.

- [ ] **Step 5: Run the local suite**

Run: `.venv/bin/python -m pytest submission/_capability_gate submission/_duck_patched -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add submission/_capability_gate
git commit -q -m "feat(capability): build sparse-model gate kernels" -m "- Generate Qwen3.6 and AgentWorld commit notebooks
- Hold agent, panel, budgets and sampling constant

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 5: Reconstruct the dense-27B control baseline and classifier CLI

**Files:**
- Create: `submission/_capability_gate/reconstruct_baseline.py`
- Create: `submission/_capability_gate/classify.py`
- Create: `submission/_capability_gate/baseline_27b.json`
- Test: `submission/_capability_gate/test_reconstruct_baseline.py`

**Interfaces:**
- Consumes: `docs/test-artifacts-2026-08-02/ab_wmr_result.json`, `ab_round2_result.json`, `ab_round3_result.json`, and any matching trace rows under `scratchpad/rl_gate/episodes`.
- Produces: source-attributed baseline medians for `ft09` and `su15`, plus a CLI accepting one or two candidate result files.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_baseline_rows_include_source_files(baseline):
    for game in CONTROLS:
        assert baseline["controls"][game]["n"] >= 2
        assert baseline["controls"][game]["source_files"]

def test_reconstruction_is_deterministic(tmp_path):
    a = reconstruct(output=tmp_path / "a.json")
    b = reconstruct(output=tmp_path / "b.json")
    assert a == b
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `.venv/bin/python -m pytest submission/_capability_gate/test_reconstruct_baseline.py -q`

Expected: FAIL because the reconstructor is absent.

- [ ] **Step 3: Implement source-attributed reconstruction**

Use only rows whose model identity is the dense Qwen3.6-27B base and whose patch pins match v7. Store median levels, sample count, every raw value and source file SHA-256. Abort rather than mixing adapter or 30B-coder rows.

- [ ] **Step 4: Generate and inspect the baseline artifact**

Run: `.venv/bin/python submission/_capability_gate/reconstruct_baseline.py --output submission/_capability_gate/baseline_27b.json`

Expected: exit 0, both controls have at least two qualifying rows, and the command prints each accepted source path.

- [ ] **Step 5: Exercise the classifier on synthetic valid results**

Run: `.venv/bin/python submission/_capability_gate/classify.py --self-test`

Expected: prints all four states (`GO`, `NO_GO`, `INFRA_FAILURE`, `INVALID`) and exits 0.

- [ ] **Step 6: Commit**

```bash
git add submission/_capability_gate/reconstruct_baseline.py submission/_capability_gate/classify.py submission/_capability_gate/baseline_27b.json submission/_capability_gate/test_reconstruct_baseline.py
git commit -q -m "feat(capability): anchor model gate to 27b controls" -m "- Reconstruct source-attributed control baselines
- Add frozen candidate selection CLI

Co-Authored-By: Codex <noreply@openai.com>"
```

### Task 6: Prepare the approval-gated runbook

**Files:**
- Create: `submission/_capability_gate/README.md`

**Interfaces:**
- Consumes: generated notebooks, baseline artifact and classifier.
- Produces: exact preflight, push, artifact-pull and decision commands.

- [ ] **Step 1: Record local preflight**

```bash
.venv/bin/python -m pytest submission/_capability_gate submission/_duck_patched -q
.venv/bin/python submission/_capability_gate/build_capability_gate.py --all
.venv/bin/python submission/_capability_gate/reconstruct_baseline.py --check
git diff --check
```

- [ ] **Step 2: Record approval-gated pushes without executing them**

```bash
kaggle kernels push -p submission/_capability_gate/qwen36 --accelerator NvidiaRtxPro6000
kaggle kernels push -p submission/_capability_gate/agentworld --accelerator NvidiaRtxPro6000
```

- [ ] **Step 3: Record output and classification commands**

```bash
kaggle kernels output ahmedmobasher86/arc-agi-3-capability-qwen36 -p scratchpad/capability/qwen36
kaggle kernels output ahmedmobasher86/arc-agi-3-capability-agentworld -p scratchpad/capability/agentworld
.venv/bin/python submission/_capability_gate/classify.py \
  scratchpad/capability/qwen36/capability_result.json \
  scratchpad/capability/agentworld/capability_result.json
```

- [ ] **Step 4: Commit and stop at the shared-state gate**

```bash
git add submission/_capability_gate/README.md
git commit -q -m "docs(capability): add sparse-model runbook" -m "- Record exact GPU gate commands
- Preserve approval before Kaggle mutation

Co-Authored-By: Codex <noreply@openai.com>"
```

Do not execute either `kaggle kernels push` command until Ahmed explicitly approves it.

### Task 7: Write the structured-workspace follow-on only after a GO

**Files:**
- Create after a GO: `docs/superpowers/specs/2026-08-05-structured-evidence-workspace-design.md`

**Interfaces:**
- Consumes: selected brain result and its target-game transcripts.
- Produces: a separately approved S0/S1 design; no implementation occurs in this task.

- [ ] **Step 1: Confirm the prerequisite mechanically**

Run: `.venv/bin/python submission/_capability_gate/classify.py scratchpad/capability/qwen36/capability_result.json scratchpad/capability/agentworld/capability_result.json`

Expected: exactly one `selected` candidate and its state is `GO`. If no candidate is selected, stop and do not create the workspace design.

- [ ] **Step 2: Draft the five-section evidence contract**

The design must define durable `observations`, `hypotheses`, `counterexamples`, `action_effects` and `open_questions`, plus the exact contradiction/revision trigger. It must not contain an executable world-model builder.

- [ ] **Step 3: Present the design for Ahmed's approval**

Do not implement the workspace until the design is explicitly approved.
