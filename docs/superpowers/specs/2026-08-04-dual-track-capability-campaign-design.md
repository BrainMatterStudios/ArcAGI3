# Dual-Track Capability Campaign Design

**Date:** 2026-08-04

**Status:** Approved by Ahmed on 2026-08-04

**Supersedes:** the patch-only sequencing in `docs/CAMPAIGN-PLAN-2026-08-04.md`; the endgame and ledger doctrines remain in force.

## 1. Decision

Run two bounded tracks in parallel at the campaign level:

1. **Patch closure:** measure the already-built animation, grinder/narration and 600-second heartbeat stack against the byte-reconstructed v7 configuration at the real 28-clone/7,920-second geometry.
2. **Capability gate:** test two sparse 35B-class local models on four never-unlocked targets and two controls, first as a brain-only swap in the existing duck harness. Only a passing brain earns a structured-workspace experiment; only a passing structured workspace earns an executable-world-model experiment.

The campaign remains optimized for configuration mean, not a lucky public draw. Public submissions are transfer probes and final duplicate banking, never small-n mean tests.

## 2. Evidence behind the change

- The verified base bytes have eight public draws with mean 0.929 and standard deviation 0.195. The 1.27 leaderboard score is the maximum of that distribution, not evidence of a 1.27-mean configuration.
- In 196 local game-runs, 70.4% completed zero levels and nine of 25 public games never unlocked. Completed levels were usually already at the scoring cap, so new unlocks and depth are the scarce currency.
- WMR moved the ten-game panel from 16 to 20 levels, but graph interventions never fired, compaction succeeded zero of 303 times, and the synthetic adapter reduced levels from 17 to 9.
- The animation, level-age grinder/narration and 600-second heartbeat changes were committed after the last A/B kernel. Unit coverage is green, but there is no performance result yet.
- The prior executable-world-model gate completed zero levels on `tu93`, `sb26` and `cd82` with Qwen3-Coder-30B-A3B despite about one hour per game. Repeating that architecture with another unproven local brain is not justified.
- Qwen3.6-35B-A3B already has a local notebook and attached FP8 Kaggle dataset in `submission/_model35b/`, but no kernel run exists. Qwen-AgentWorld-35B-A3B is available as the ready Kaggle model source `keras/qwen-agentworld/transformers/default/1` and has not been evaluated in this repository.

## 3. Scope

### In scope

- A reproducible eval-geometry A/B for the combined current patch candidate.
- A six-game sparse-model serve and behavior gate for Qwen3.6-35B-A3B-FP8 and Qwen-AgentWorld-35B-A3B.
- Machine-readable results, model identity proofs, throughput, tool/protocol reliability, levels, tokens and control regressions.
- A Tycho-inspired structured evidence workspace only after a model passes the brain-only gate.
- Updating the campaign plan and submission ledger discipline.

### Out of scope until a gate passes

- A full Tycho port.
- An executable `world_model.py` builder or delegated planner.
- New SFT corpus generation or training.
- More graph, compaction or playbook work.
- A competition submission, Kaggle kernel push, model upload, public release or production/shared-state mutation without Ahmed's explicit approval.

## 4. Track A — patch closure

### 4.1 Arms

The base arm reconstructs the scored v7 behavior explicitly rather than relying on changed defaults:

```python
BASE_ENV = {
    "TAAF_WATCHDOG": "1",
    "TAAF_WATCHDOG_STALL_S": "900",
    "TAAF_HUD_MASK": "1",
    "TAAF_WIN_REPLAY": "1",
    "TAAF_ANTIFREEZE": "1",
    "TAAF_ANIMATION": "0",
    "TAAF_GRAPH": "0",
    "TAAF_GRAPH_GRIND_AGE_ACTIONS": "120",
    "TAAF_GRAPH_GRIND_AGE_TURNS": "10",
    "TAAF_GRAPH_GRIND_MAX_PER_LEVEL": "2",
    "TAAF_COMPACT": "0",
    "TAAF_PLAYBOOK": "0",
    "TAAF_GRID_BURNER": "0",
}
```

The candidate changes only the three approved mechanisms:

```python
CANDIDATE_ENV = {
    **BASE_ENV,
    "TAAF_WATCHDOG_STALL_S": "600",
    "TAAF_ANIMATION": "1",
    "TAAF_GRAPH": "1",
    "TAAF_GRAPH_GRIND_AGE_ACTIONS": "120",
    "TAAF_GRAPH_GRIND_AGE_TURNS": "10",
    "TAAF_GRAPH_GRIND_MAX_PER_LEVEL": "2",
}
```

Graph recording, veto and grinder share `TAAF_GRAPH`. The gate must therefore report vetoes and grinder events separately; a combined win earns a later ablation before final freeze.

### 4.2 Geometry and order

- 28 competition-sim clones drawn from all 25 official public games.
- 7,920 seconds per game, concurrency 28.
- Two separate kernels, one arm per kernel. This avoids trying to fit two 2.2-hour waves plus repeated model starts into one fragile notebook.
- Run order is randomized before push and recorded in the result header. The second-run advantage observed in earlier fixed-order pairs must not be reintroduced.
- Both notebooks are built from the same base notebook and commit, with the environment dictionary as the only behavioral delta.

### 4.3 Patch decision

The combined candidate is **GO** only if both runs complete with identity proofs and either:

- at least two candidate-only first unlocks among `cn04`, `dc22`, `g50t`, `lf52`, `ls20`, `m0r0`, `sk48`, `tr87`, `wa30`; or
- at least +6 total levels, excluding `ft09`, with neither `su15` nor `tu93` regressing by two or more levels.

Additional mechanism requirements:

- animation query rate at least 20% on game-runs with captured multi-frame actions, unless the level criterion already passes;
- at least one level-age grinder engagement, otherwise grinder/narration remains untested and cannot be credited;
- zero scorecard stale-close incidents in the candidate run.

If the combined arm passes, run one attribution ablation before freeze: animation off versus grinder off on the same gate. If it fails, retain the byte-reconstructed v7 configuration and stop patch work.

## 5. Track B — sparse-model capability gate

### 5.1 Candidates

1. **Qwen3.6-35B-A3B-FP8** from the existing dataset source `nareshmeena07/qwen36-35b-a3b-fp8` and notebook `submission/_model35b/duck-model35b.ipynb`.
2. **Qwen-AgentWorld-35B-A3B** from `keras/qwen-agentworld/transformers/default/1`. It is a world-model pretraining candidate, not presumed to be a reliable action policy.

Each candidate runs in its own commit kernel. The current dense Qwen3.6-27B-FP8 is the recorded baseline from existing traces; no third GPU run is required unless the six-game baseline rows cannot be reconstructed.

### 5.2 Panel and budget

Targets: `dc22`, `m0r0`, `sk48`, `tr87`.

Controls: `ft09`, `su15`.

- 283 GPU-seconds per game is the production-equivalent share implied by 28-way contention at a 7,920-second box.
- The gate runs six games concurrently under the normal duck solver so contention, prompt construction and tool execution remain representative.
- Model loading and serving may consume at most 3,600 seconds. A candidate that cannot become healthy within that bound is a serve failure.
- Required context is 65,536 tokens. Sampling remains temperature 0.6, top-p 0.95 and top-k 20.

### 5.3 Identity and reliability proofs

Before play, each kernel records:

- resolved model path and real path;
- config `_name_or_path`, architecture and quantization fields;
- weight-index SHA-256;
- `/v1/models` response;
- server command line;
- deterministic temperature-zero logprob fingerprint;
- 400-token throughput probe.

During play it records assistant turns, parseable Python-tool calls, tool successes, rejected actions, timeouts, prompt tokens, completion tokens and levels.

`protocol_success_rate = successful_tool_executions / max(assistant_turns, 1)`.

This denominator makes malformed or non-tool responses count against reliability rather
than disappearing from the metric. `parseable_tool_attempts` remains a diagnostic.

### 5.4 Brain-only decision

A candidate is **GO** only if:

- it serves within 3,600 seconds and fits without CPU offload;
- protocol success rate is at least 95%;
- at least two of the four target games complete their first level;
- combined control levels do not regress by two or more versus the reconstructed 27B baseline;
- all six games finish within the production-equivalent budget without a server crash.

If both pass, choose the model with more target unlocks; ties break by total target levels, then protocol success, then tokens/second. If neither passes, stop the model route.

### 5.5 Structured-workspace stage

Only the selected brain receives a second A/B:

- **S0:** existing duck prompt/memory behavior.
- **S1:** durable sections for `observations`, `hypotheses`, `counterexamples`, `action_effects` and `open_questions`, with a forced evidence citation and explicit hypothesis revision after a contradicted prediction.

No executable model builder is present. S1 must produce at least one additional target unlock or +3 target levels without control regression. Only then may a separate executable-world-model design be proposed.

## 6. Resource and slot allocation

### GPU quota for the first seven days

| Work | Maximum GPU time |
|---|---:|
| Patch closure, two eval-geometry kernels | 5.5 h |
| Qwen3.6 capability kernel | 2.0 h |
| AgentWorld capability kernel | 2.5 h |
| One rerun for infrastructure-only failure | 2.5 h |
| Reserved for the winning track | 5.0 h |
| **Total ceiling** | **17.5 h** |

The remainder of the 30-hour weekly quota stays uncommitted until a gate produces a positive result.

### Submission slots

- No live submission for a gate failure or an infrastructure-only result.
- One live transfer probe for a gate-passed patch configuration.
- One live transfer probe for a gate-passed model/workspace configuration.
- On days without a ready experiment, byte-identical best-config draws are allowed as opportunistic rank banking but have zero evidentiary weight.
- Final 10–15 slots remain reserved for byte-identical duplicates of the frozen finalist.

## 7. Result states

Every gate ends in exactly one state:

- `GO`: all frozen criteria passed.
- `NO_GO`: run valid, criteria not passed.
- `INFRA_FAILURE`: result cannot answer the hypothesis; one corrected rerun is allowed.
- `INVALID`: identity, contamination or configuration proof failed; result is excluded and the cause is documented.

No threshold may be changed after results are visible. A changed threshold creates a new exploratory gate with a new result file.

## 8. Safety and approval boundaries

Local builders, tests, notebook generation and dry-runs are authorized by this design. The following remain approval-gated:

- `kaggle kernels push`;
- competition submission;
- Kaggle dataset/model creation or update;
- git push, PR merge, deployment, publication or production database writes.
