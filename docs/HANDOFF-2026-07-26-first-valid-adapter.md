# HANDOFF — 2026-07-26 ~14:30 UTC — First VALID adapter trained; corrupted-base saga closed

> **⚠️ 2026-07-26 EVENING AUDIT CORRECTIONS** (docs/REVIEW-2026-07-26-independent-audit.md, memory `arcagi3-audit-2026-07-26-corrections`): the §0 val-gate note is a conflation — the −12.5% gate ran **all 43 val rows** (the 2-row figure is the pre-flight base-health gate only), BUT the train/val split is per-turn with 100% episode overlap → it is fit evidence, not generalization. Also: RESET costs 1 scored action (not 0); final = separate 55-game private set (same-run scores, no rerun); per-game score capped at completed-level weight share; the §2.1 GATE_CKPT/glob edits are now DONE; the temp-0.3 defect never shipped and the local taaf-src tree has drifted from the scored dataset — audit against the dataset.

**Start here.** Campaign memory: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-ewm-verdict-REVERSED.md`
(the 07-26 sections at top are today's; read them). Deep-research validation of the plan:
memory `arcagi3-deepresearch-2026-07-25-plan-validation.md`. Supersedes HANDOFF-2026-07-25 on all training state.

---

## 0. THE HEADLINE

**SFT runs 1–5 were invalid — trained against a corrupted base** (the VL snapshot keeps 256
language Linears as f8e4m3 storage `w_true/scale`; our strip deleted the scales unapplied →
base per-token NLL 14.7 = worse than uniform). Found because the serving-verification gate's
merge crashed on a float8 `+=`. **Run 8 (kernel v8, COMPLETE 07-26 13:52 UTC) is the first
valid run**: pre-flight base NLL **0.911**, 15 optim steps at 79.5–81.8 GiB, in-kernel val gate
**PASS (target-loss base 0.7781 → tuned 0.6807, −12.5%)**. Working recipe =
`sft_common.fp8_scaled_linear_inplace` (f8 storage + correct scaled forward, bit-identical to
the merge path at bf16; full-bf16 dequant OOMs at +25GB). Commits: `6cf8f13` (root cause,
dequant, pre-flight gate), `46879b3` (scaled forward), `0b7b64f` (serve-verify kernel). All
unit-tested (`submission/_sft_k3/test_dequant.py`).

## 1. IN-FLIGHT / IMMEDIATE (check these FIRST)

### a. ckpts-dataset v2 — ✅ VERIFIED LANDED (2026-07-26 14:59 UTC; nothing to do)
File listing confirmed: `sft_adapter/*` complete (adapter 467,062,560 B + tokenizer/processor/
chat_template) and `sft_out/checkpoint-8/*` full incl. **optimizer.pt = 934,676,291 B**. The
valid run-8 artifacts are banked off-kernel; new kernel pushes can no longer shadow them.
⚠️ Dataset **v1 = the POISONED runs-1-5 ladder** (it ALSO contains a checkpoint-8 — the tell
for v2 is the sft_adapter/ dir + the 934MB optimizer.pt). **Never mount/serve/sweep v1 values.**
Local staging copy (if ever needed):
`/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/38ec9bae-690c-4713-aa14-c3245497ca9e/scratchpad/sft_v8_valid/`.

### b. Tonight's farm slot (Jul 27 00:00 UTC) — **NOT ARMED, by Ahmed's explicit instruction.**
Ahmed must decide/arm. Default candidate = 5th pinned duck-base v2 draw:
```sh
kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k ahmedmobasher86/arc-agi-3-duck-base -v 2 \
  -f submission.parquet -m "Pinned duck-base v2 farm draw (identical-bytes distribution: 0.92, 1.14, 0.82, 0.75)."
```
(1 scored sub/day; never mutate the pinned bytes; the auto-submit waiter may hit the permission
classifier — Ahmed can run it via `!` or approve.)

### c. GPU quota: ~0.5h left; pool is 30h shared (training+everything), resets weekly (next ≈ Sat Aug 1 00:00 UTC).
Verified from Ahmed's screenshot + burn accounting. All GPU work below waits for the reset.

## 2. NEXT STEPS (post-reset, in order)

1. **Serve-verify v2 on the run-8 adapter** (`submission/_serve_verify_k3/`, kernel
   `arc-agi-3-serve-verify-k3`; v1 = the run that exposed the corrupted base — its merge fix
   (dequant before strip) is already committed). **REQUIRED EDITS before pushing v2:**
   - `GATE_CKPT` default is `checkpoint-16` (no longer exists) → `checkpoint-8`, and add a
     second gate arm for `sft_adapter/` (the step-15 final; note it sits at `sft_adapter/`,
     NOT under `sft_out/checkpoint-*` — the CKPT glob needs that case).
   - Design: in-kernel merge (55GB bf16 in /tmp scratch — Kaggle's 20GB output cap makes
     publishing from a kernel impossible), serve MERGED then BASE with the duck's exact vLLM
     line, 3 greedy probes, weight-delta assert (delta == B@A×alpha/r). ~2-2.5h GPU.
2. **A1 checkpoint sweep** (the deep-research amendment): ckpt-8 vs step-15-adapter arms on the
   **game-level holdout** via the A/B driver (`scratchpad/rl_gate/run_rollout.py`), zero-game
   behavioral scorecard (`scratchpad/trace_forensics/`) as veto. Select by behavioral wins,
   NEVER by loss. Literature says OOD peaks at ~12-15% of steps — ckpt-8/15 may already be
   near-optimal; extend epochs ONLY if the sweep says deeper helps.
3. **Tuned debut** ONLY on a sweep GO, with pre-registered stop-losses (3 tuned draws mean <
   base−0.10 → revert; 5 draws < base+0.05 → revert). Selection note: final-2 defaults to the
   two highest PUBLIC draws (Ahmed verified) → never submit an unvalidated config to a slot.
4. **$0 corpus work** (anytime): SCoRe-style v2 (deep-research A2: student generates, K3
   corrects the earliest error — beats behavioral cloning by +3.1 SFT-only; K3 API cost is
   Ahmed-gated), flail-turn loss-masking, game-level-holdout retrain (exclude vc33/sc25/lp85).
5. **G6 prompt lines** offline A/B ≤ Aug 10 unconditionally (anti-HUD line = the 27B's #1
   verified killer; `scratchpad/mechanics_compendium.md`).

## 3. DEEP-RESEARCH VERDICT (2026-07-25, 104 agents, 3-vote verified — details in memory)
Plan survives; two amendments: **A1** early-checkpoint OOD peak (above), **A2** SCoRe
student-centered corpus. Leaderboard truth: **≥1.5 = ~rank 12; top-5 needs ~1.57+**; we sit in
a 10-team tie at exactly 1.26 (= unmodified public duck); 1.44-1.61 = 24-team plateau of inert
duck forks; nobody public has touched ACTION7-with-guidance or serving/concurrency. Kojima 1.86
= best-of-48 outlier, regresses privately. REFUTED (don't cite): Tufa "mean 1.6002",
"concurrency 16", the "SFT memorizes" headlines.

## 4. THE LAWS (additions from this session — the old five stand)
6. **Assert absolute sanity of the STARTING point, not just relative progress.** Runs 1-5 had
   smoothly declining loss — from 14.7, above uniform-random. The pre-flight base-NLL gate
   (<6.0 nats, in the trainer) now enforces this; keep it in every training kernel.
7. **The kernel imports `sft_common` FROM THE CORPUS DATASET**, not from the repo. Any
   sft_common change ⇒ `cp sft_common.py corpus/ && sh upload_data.sh` ⇒ poll
   `kaggle datasets files` for the exact new byte size ⇒ only then push the kernel (v6 died on this).
8. **`kernels_output` serves only the LATEST version** (no version param; running/failed newer
   versions shadow older outputs — per-version output needs the browser UI). Bank artifacts to
   a dataset BEFORE pushing a new version.
9. **Kaggle REJECTS self-referencing kernel_sources** ("not valid kernel sources") — the
   self-mount resume design is impossible. Resume = checkpoint dataset (see §1a).
10. **Long background jobs (up/downloads) may be killed** — verify completion by listing remote
   state, chunk foreground downloads with file_pattern, and re-run idempotently.

## 5. SUBMISSION LEDGER / SCOREBOARD (Jul 26)
Pinned duck-base v2 identical-bytes draws: **{0.92, 1.14, 0.82, 0.75}** (mean ≈0.91, spread
0.39 — Law 3 hardened). Best-ever 1.26 (sub 54554985) + 1.14 (54938402) = current select-2
pair; floor safe. Latest: 54986953 = 0.75 (Jul 26). 1 sub/day; low draws cost nothing.

## 6. ASSETS MAP (delta from 07-25 handoff)
| what | where |
|---|---|
| VALID run-8 artifacts | dataset `ahmedmobasher86/arc3-sft-k3-ckpts` **v2** (ckpt-8 full + sft_adapter step-15) — verify §1a |
| Trainer (fixed) | `submission/_sft_k3/` kernel v8; `fp8_scaled_linear_inplace` + pre-flight gate; EPOCHS default 0.30 |
| Dequant/scaled-forward tests | `submission/_sft_k3/test_dequant.py` (run with repo .venv) |
| Serving gate kernel | `submission/_serve_verify_k3/` (`arc-agi-3-serve-verify-k3`; v1 ran, found the bug; needs §2.1 edits) |
| vLLM offline recipe | wheelhouse dataset `driessmit1/arc3-vllm-h100-wheelhouse-v3` + duck setup_commands pattern (copied in gate kernel) |
| Corpus dataset | `ahmedmobasher86/arc3-sft-k3-corpus` (now carries the FIXED sft_common.py, 11,126 B) |
| Everything else | unchanged — see HANDOFF-2026-07-25 §5 |

## 7. GIT
Branch `winning/duck-patched`, HEAD `46879b3`, never pushed (no push authorization). Today's
commits: 6f9286e (v5 ladder), 0b7b64f (gate kernel), 6cf8f13 (root cause + dequant + pre-flight),
46879b3 (scaled forward). Working tree may have `corpus/sft_common.py` + `prep_stats` deltas —
commit them with the next change.

## 8. MONEY / PENDING-AHMED
Unchanged: $0 critical path; no spend authorized; NO training on Claude outputs until written
Anthropic approval arrives; Wave-2/SCoRe teacher-API spend is Ahmed-gated. Ahmed has verified
select-2 mechanics (= 2 highest public scores; whether manual override exists is still open —
matters in August) and the GPU quota panel (30h shared pool).
