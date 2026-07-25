# HANDOFF — 2026-07-25 ~11:30 UTC — Teacher-Distillation Campaign, mid-training

**Start here.** Strategic memory: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-ewm-verdict-REVERSED.md` (the full campaign log — read it).
Plan of record: **Master Plan v3 + judge amendments** → artifact `claude.ai/code/artifact/b7ade01e-39e2-47f2-86aa-66c78e999407`.
Teaching explainer (for Ahmed): artifact `claude.ai/code/artifact/7db48761-357d-45ad-a53c-c9d0645152ed`.

---

## 1. IN-FLIGHT RIGHT NOW (the previous session's background waiters ARE DEAD — re-arm what's needed)

### a. SFT training run 4 — RUNNING on Kaggle as of 11:27 UTC
- Kernel `ahmedmobasher86/arc-agi-3-sft-k3` **v4**, pushed ~02:15 UTC Jul 25 → **12h cap hits ~14:15 UTC**.
- LoRA r=16 (language layers only) on Qwen3.6-27B, corpus = 392 train/43 val win-turn samples
  (dataset `ahmedmobasher86/arc3-sft-k3-corpus`), 2-epoch target = 98 optim steps,
  **resumed from step 40** (run 3's checkpoints self-mounted via `kernel_sources` incl. its own slug).
- Throughput ~16 min/optim-step (accum 8); 58 steps remaining ⇒ it will likely **cap-kill again
  around step ~80-85**. If so: just `cd submission/_sft_k3 && kaggle kernels push -p . --accelerator
  NvidiaRtxPro6000` → v5 resumes from the newest checkpoints the same way. One more session finishes.
- Poll: `kaggle kernels status ahmedmobasher86/arc-agi-3-sft-k3`.
- **Fetch logs ONLY via** the API `file_pattern` trick (full output downloads die on GB checkpoints):
  ```python
  from kaggle.api.kaggle_api_extended import KaggleApi
  api=KaggleApi(); api.authenticate()
  api.kernels_output("ahmedmobasher86/arc-agi-3-sft-k3","/tmp/x",file_pattern=r".*\.log$")
  ```
- When the run COMPLETES, check in the log: truncation/drop count (SFT_MAX_LEN=24576 vs corpus
  prepped at 32768 — if >10% samples truncated, per RUNBOOK restore 32768 consideration), final
  loss, val loss, adapter-ON-vs-OFF assertion result.

### b. Tonight's slot (Jul 26 00:00 UTC) — **NO WAITER ARMED. Re-arm it:**
```bash
until [ "$(date -u +%Y-%m-%d)" = "2026-07-26" ]; do sleep 300; done
kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k ahmedmobasher86/arc-agi-3-duck-base -v 2 \
  -f submission.parquet -m "Pinned duck-base v2 farm draw (distribution on identical bytes: 0.92, 1.14, 0.82)."
```
(run_in_background; 1 scored sub/day confirmed; NEVER mutate the pinned bytes; never skip a day.)
Jul 25's slot already used: sub 54962014 = 0.82 (pinned draw #3).

### c. OpenRouter API key — **do not spend without Ahmed's explicit go** (he's flagged budget).
Old session stash (may persist): `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/5174edda-9963-4cdd-9306-18d2695d0fd0/scratchpad/.openrouter_key`.
If gone and needed, ask Ahmed. NEVER commit it or write it to memory files.

---

## 2. NEXT STEPS (the week, per Master Plan v3 §2 as amended by judges)

1. **Adapter completes** (today/tomorrow) → **merge + FP8 re-quant** on free Kaggle GPU per
   `submission/_sft_k3/RUNBOOK.md`.
2. **SERVING-VERIFICATION GATE** (non-negotiable; July's fine-tune v1 died here): served-weights
   checksum + logprob divergence from base + sb26-L1 canary (sb26 is memorized — tuned must
   reproduce teacher-style play there or the weights aren't serving).
3. **Amended offline A/B** (~36-48 free-GPU-hours as Kaggle commits, start 1 seed × 2 arms):
   - Panel: ft09, vc33, su15, sc25, re86, s5i5, lp85, tu93 (base scores 0 there → binary wins beat noise).
   - **Primary GO = depth-held-out wins** (levels beyond K3's corpus frontier: ft09 L5+, vc33 L4+,
     su15 L4+, tu93 L3+, L2+ on L1-only games). Memorization can't fake these.
   - **Zero-games (dc22, m0r0, tr87, tn36, sk48) = behavior instrument**: run the trace-forensics
     scorecard (`scratchpad/trace_forensics/metrics.json → scorecard_spec`) on both arms;
     **scorecard degradation there = NO-GO regardless of panel wins** (judge-forced).
   - Score the base arm with real depth-weighted RHAE → derives `duck_dev` (Unknown #2) for free.
   - Prefix-swap probe (student continues K3 prefixes vs from-scratch) when GPU-idle.
4. **Debut** per G1 ONLY on GO, with judge-forced stop-losses pre-registered: after 3 tuned draws
   mean < base−0.10 → revert; after 5 draws < base+0.05 → revert. No single-draw confirmation
   (base once drew 1.14). The "all-in on Δ≤+0.1" branch is DELETED.
5. **$0 corpus v1.5 work** (can start anytime): flail-turn loss-masking; add intermediate assistant
   turns of winning levels as targets; **game-level holdout retrain** (exclude vc33/sc25/lp85 wholly —
   a win on an excluded game becomes the primary evidence class). Then KTO on the 856 win/loss turns.
6. **G6 prompt lines** offline A/B ≤ Aug 10 unconditionally (anti-HUD line = the 27B's #1 verified
   killer; mechanics field guide draft in `scratchpad/mechanics_compendium.md`; ephemeral-REPL note).

## 3. MONEY / AHMED-PENDING
- **Ahmed: no more spend authorized.** The critical path is $0 (free GPU + slots + owned data).
  Wave-2 teacher corpus (~$150, deep boxes to fix the 62%-L1 corpus skew) is OPTIONAL, only after a
  v1 GO, only with his explicit go.
- **Anthropic authorization email SENT** (usersafety@ + support + sales) asking permission to train
  on Claude outputs. **Until written approval arrives: absolutely NO training on Claude outputs**
  (AUP + forced open-sourcing = published breach). If approved: Claude-teacher corpus via Claude Code
  headless on his subscription ≈ $0, supersedes Wave 2; capture pipeline is teacher-agnostic.
- Ahmed still owes (his browser, 15 min): Kaggle **My Submissions** page — verify select-2 final
  mechanics (Unknown #1; determines endgame selection strategy) + the **GPU quota panel** (weekly cap
  unknown; blocking gate on all A/B dates per judges).

## 4. THE LAWS (do not relearn these with slots)
1. **Depth-only scoring**: level 9 pays 9× level 1; efficiency on completed levels is capped at k/2 extra levels.
2. **Scaffolding hurts ≤96GB brains**: EWM closed (12 clean runs, 0 levels, same brain that wins in the duck).
3. **±0.3 measurement doctrine**: pinned identical bytes drew {0.82, 0.92, 1.14}. Slots = farming;
   all config truth from offline A/Bs. One play per game at eval (replay/banking/best-of-N dead).
4. **Final = private LB of ≤2 SELECTED subs** (MEDIUM confidence, Ahmed verifying): the target is
   TRUE MEAN ~1.5, not public 1.86 (Kojima's 1.86 = best-of-46 ⇒ his true mean ~1.45-1.55 and his
   private score regresses). Select final subs by best-EVIDENCED config, never best public draw.
5. **COMPLETE ≠ worked; ERROR ≠ your code failed** — always read the log. Kaggle output downloads
   are paginated (use file_pattern). Dataset version pushes need propagation checks before dependent
   kernel runs. Kernel notebooks: copy cells, never rebuild (attachments!). zsh doesn't word-split.

## 5. ASSETS MAP
| what | where |
|---|---|
| Campaign memory (read first) | `~/.claude/projects/.../memory/arcagi3-ewm-verdict-REVERSED.md` |
| SFT trainer + RUNBOOK + corpus prep | `submission/_sft_k3/` (kernel `arc-agi-3-sft-k3`, dataset `arc3-sft-k3-corpus`) |
| Teacher traces (4 brains, same games) | `scratchpad/rl_gate/episodes/*/trace.jsonl` (+ viewer events in workdirs) |
| Harvest + corpus | `scratchpad/rl_gate/harvest_sft.py`, `sft_data/all_wins.jsonl` (435 samples) |
| Rollout driver (doubles as A/B driver) | `scratchpad/rl_gate/run_rollout.py` (+ capture_proxy, sweep_driver) |
| Behavioral scorecard | `scratchpad/trace_forensics/` (`metrics.json → scorecard_spec`) |
| Mechanics compendium | `scratchpad/mechanics_compendium.md` |
| Click ground truth | `scratchpad/killexp_data/*.npz` |
| Research workflows (full reports) | journals under `~/.claude/projects/.../subagents/workflows/wf_fa2e7283-58b/` (#2), `wf_c1e8eb56-3f2/` (#3) |
| Layer-2 fixes (parked, un-A/B'd) | `submission/_duck_fixes/` |
| Duck source (patch targets) | `submission/_adopt/taaf-src/src/ARC3-Inference/` |

## 6. GIT
Branch `winning/duck-patched`, ~24 commits ahead, **never pushed** (Ahmed hasn't authorized push).
HEAD `13bee24`. Pre-existing dirty files (`.gitignore`, `_duck_shadow/*`, `_ewm_*`) predate this
campaign — leave them. Scored-submission floor: **1.26 = sub 54554985; never deselect it** until a
tuned sub has ≥2 draws of ≥1.26-class evidence.

## 7. SCOREBOARD SNAPSHOT (Jul 25)
Us 1.26 best (public), rank ~74/1850. #1 Kojima 1.86 public (true mean est. ~1.5). Plateau 1.44-1.61
= stock duck + luck. K3 teacher dev-mean ≈26/100 through our harness — the gap we're distilling.
Program spend: ~$110 teacher API + $0 GPU. Rival watch: ericmao shipped 7 LoRA adapters on our base
(Jul 21) — assume a 4-8 week window before someone else fields a tuned duck.
