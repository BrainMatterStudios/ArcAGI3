# HANDOFF 2026-08-09 — structural campaign: state, tonight's decision, four parallel tracks

Self-contained. Supersedes prior handoffs on operational state. Read with
`docs/RESULTS-2026-08-08-overnight-gates.md` (all week verdicts, appended chronologically).

## 1. Position (verified live 2026-08-09)

- **Public LB: 1.30, rank 102/2194** (new all-time high, banked by the 08-09 idle-day draw).
  21 teams ≥1.50; rank-113 bar is 1.29; Kojima #1 at 1.86 (best-of-61).
- **Base distribution (identical bytes, n=9): mean 0.970, sd 0.220, band 0.69–1.30**
  (0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69, 1.30). Supersedes "n=8 mean 0.929".
- GPU: ~27h of 30h weekly quota spent (resets Saturdays). Submissions: 1/day at 00:00 UTC;
  final = 2 selected subs' private scores, no rerun; ~85 slots remain to ~Oct 20 freeze.
- The pinned live config = **duck-base v2** (canonical code-cell hash `886dbc8a…`, method:
  `"\n".join(code cells)` — NO trailing newline in the ledger method; scriptVersionId 336252059;
  repo `submission/_duck_base/duck-base.ipynb` IS those bytes; remote v3 is an unscored
  gpu-probe variant — never submit v3).

## 2. Week verdicts (all replicated/instrumented; details in RESULTS doc)

- **Track B EXHAUSTED**: 35B gate NO_GO (0/4 targets, 91.7% protocol — capability, not
  integration); AgentWorld gate NO_GO (0/6 incl. controls, 86.5%). **The 27B is the brain.**
- **Track A closure NO_GO** (wave-1 +6 did not replicate: +1). Patch delta dead; live config = base bytes.
- **Package-as-advertised dead**: 27B ignored advertised tools (run_probe 1 call/28 games;
  believed it didn't exist — contract-placement failure). **Law: enforcement must be structural.**
- **Structural plan channel = the one replicated positive**: adoption 2.01/2.59/2.57
  plan-actions/deliberation (base 1.0) across 3 waves; coaching iteration moved nothing —
  **the ~2.5× plateau is the model's ceiling, not contract clarity**. Levels: struct
  {17,12,12} vs base {11,12} ≈ **+2/wave directional, uncertified** (needs 4–6 waves).
- Variance arm 0.85 IN_BAND (sampling not load-bearing either way at n=1).
- 08-09 "offline-week-challenge" doc: mechanics core VERIFIED (scoring chain: per-level
  `min(1.15,(baseline/actions)²)×(level+1)` weighted; actions accumulate across attempts —
  stuck-loop waste quadratically poisons eventual completions; knowledge-wipe defect
  `tool_agent.py:1113–1126`; frame[-1]-only; chars/3 token overcount). Its LB tail + strategy
  REFUTED (fabricated ranks). Independent review in session record.

## 3. Assets

- **Branch `winning/duck-patched` @ `f151409`** (+ kernel commits to `30fc4cf`): patches 1–22.
  New this week: 16 diff-lines, 17 wiggle/masks (+CURSOR/HERD/morph-avatar repairs), 18 run_probe,
  19 dispatch scaffolds, 20 verifier-at-commit (fail-open), 21 structural plan channel
  (mandatory 1–20 plans, lenient parser, yield-aware nudges, worked examples), 22 A-not-B brake +
  scout/commit phase gates. All env-gated OFF; struct set = `TAAF_STRUCT=1` (+DIFF/WIGGLE/DISPATCH).
  Suites ~420 green (`PYTHONPATH=reference/arc-agi-toolkit .venv/bin/python -m pytest`).
- **Screen kernels** `submission/_ab_patch_closure/{base,candidate,package,struct}` — closure-rig
  A/B machinery, source-pin guard, docker pin, dual-mount install, embedded reading contracts,
  classifiers (`classify.py`, `classify_package.py`, `classify_struct.py`).
- **Capability-gate infra** on worktree `.worktrees/dual-track-gates` (branch `winning/dual-track-gates`,
  ~10 commits ahead): frozen evidence contracts, serving probes (thinking-tolerant, structural
  zero-offload proof, wallclock slack 60s), protocol metric, qwen36 + agentworld kernels.
  Datasets: `arcagi3-bundle-35b` v3 (`--max-num-seqs 512`), `qwen-agentworld-35b-a3b-bf16` (private).
- **Human data**: 340 replays all 25 games (`scratchpad/human_replays/`), budget tables
  (session scratchpad `human_budgets/` — regenerate via extractor if missing; key numbers in
  memory `arcagi3-human-replay-dataset`). PRO-LONG cache (prompts/logs) in the 08-08 session
  scratchpad `prolong/`; key mechanics in memory `arcagi3-prime-agent-findings` + session records.
- **Submission tooling**: `scripts/submit_gated.py` (the ONLY submit path; auto-appends ledger);
  one-shot runner pattern `scripts/submit_{variance_20260808,base_20260809}.py` (one-shot UTC
  window, fire-time re-attest, race guard, `--mock` full-chain test, arm detached:
  `nohup caffeinate -i -s python3 … & disown`).

## 4. Tonight's decision (NOT ARMED — decide then arm)

**Proposed: first live probe of the struct config.**
1. Build duck-patched v9: rebuild notebook from `f151409` with `EXPERIMENT_ENV` pins
   `TAAF_DIFF_LINES/WIGGLE/DISPATCH/STRUCT=1` (see `build_duck_patched.py`; verify pins print in
   the commit log). Push (`kaggle kernels push -p submission/_duck_patched --accelerator
   NvidiaRtxPro6000`); commit run is CPU-safe (~free GPU).
2. Wait COMPLETE, settle >3h, clone the one-shot runner for the 00:01 UTC slot with fire-time
   re-attest (canonical hash of the NEW v9, version==9) and a message carrying the fingerprint
   `[struct-v9-<hash8>]` + hypothesis ("offline +2/wave, adoption 2.5x — transfer test") +
   reading rule ("only outside 0.69–1.30 individually actionable").
3. Alternative: repeat the base-v2 duplicate (rank lottery, ~10% of paying >1.30, zero info).
   The runner for base v2 needs only date/window edits to `submit_base_20260809.py`.

## 5. Four parallel tracks (no GPU, no slot; approved for exploration 08-09)

1. **Struct mechanism attribution** (highest value): from wave outputs (08-08 session scratchpad
   `struct/`, `struct2/`, `struct3/` — re-pull via `kaggle kernels output
   ahmedmobasher86/arc-agi-3-struct-screen` if gone): which games gained levels, did gains
   correlate with plan usage/lengths, where did g50t's w1 unlock come from? Decides whether the
   4–6-wave certification A/B (~12 GPU-h, next week) is justified.
2. **Rank-4b comparator**: machine-checked `predicted_diff` before each action + SURPRISE
   interrupt (sweep rank 4b/5; prerequisite for surprise-triggered parallel second opinions).
   Build as patch23, same discipline (env-gated, tests, dry-run).
3. **Hidden-state residual sweep**: fresh blind multi-agent ideation on the 13/25 games where
   frames are insufficient statistics — the explicitly-flagged frontier. Independence rules as
   in the 08-07 sweep (blind ideators, adversarial filters vs tried-list, cheapest-test bars).
4. **Budget-HUD patch**: surface per-level human-median action budgets (click ~20/level-1 …,
   avatar ~31; full tables in human budgets) in the prompt as a countdown; rationale = verified
   quadratic efficiency poisoning. Dev-set medians are scale priors only (hidden baselines differ).

## 6. Operational laws (hard-won this week; violate none)

- Canonical hash method varies by record: ledger = no trailing newline; session attestations =
  trailing newline. State the method whenever writing a hash down.
- Kernel `--accelerator NvidiaRtxPro6000` at push (metadata machine_shape is inert);
  ALWAYS pin `docker_image`; competition data mounts at `/kaggle/input/competitions/<slug>` OR
  `/kaggle/input/<slug>` — probe both; never discard pip stderr.
- Commit runs are CPU-safe landings — serving paths execute ONLY in scored/gate runs; a COMPLETE
  commit proves wiring, never serving.
- One armed automation at a time; `pgrep -fl submit_` before arming; detached via nohup+disown
  (session death killed a runner once — battery).
- Submissions: only via `submit_gated.py`; >3500 bytes = played; COMPLETE <1h = never-played alarm.
- Single-wave boundary results (closure +6, struct 17+g50t) have ~13% FP — replicate before believing.
- Contract placement: capabilities must appear in the tool schema + per-turn globals line;
  system-prompt-only advertisement reads as nonexistent to the model.
- Planless turns were 98% turn-clock exhaustion (inspection-only python), not defiance.

## 7. Key files

- Verdicts: `docs/RESULTS-2026-08-08-overnight-gates.md` · plans: `docs/CAMPAIGN-PLAN-2026-08-04.md`
  (+ 08-05 amendment; slot doctrine), `docs/PLAN-2026-08-07-variance-slot-and-offline-week.md`
- Research: `docs/RESEARCH-2026-08-07-human-play-idea-sweep.md` (Top-8 portfolio; ranks 5/6/8 =
  hypothesis ledger / forward model / executable WM remain unbuilt), 08-09 offline-week doc
  (mechanics only), Prime-Agent/PRO-LONG memory entries.
- Ledgers: `docs/submission-ledger.json` + `docs/SUBMISSION-LEDGER.md` (every conclusion cites them).
- Memory index: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/MEMORY.md`.

## 8. Open questions for Ahmed (blocking)

1. Tonight's arm: struct-v9 live probe vs base duplicate vs hold.
2. Next week's GPU allocation: struct certification A/B (4–6 waves, ~12h) vs hidden-state
   program vs hard-structural adoption forcing (protocol risk).
3. Sept 20 Milestone-2 go/forfeit and the ~Oct 20 config freeze remain on the calendar.
