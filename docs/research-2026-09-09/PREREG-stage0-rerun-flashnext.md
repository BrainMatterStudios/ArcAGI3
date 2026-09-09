# PRE-REGISTERED — Stage-0 rerun on Flash-Next (written 2026-09-09 before any data)

**Question (unchanged from 2026-09-02):** given the frontier's recorded level-0 transitions for a hard ARC-AGI-3 game
plus a backtest tool, can the deployed brain emit a backtest-green executable world model?

**Why it is not a re-run of a dead lever.** The lane was killed on Qwen3.8-27B-FP8, and the failure was specific:
on sk48 and cn04 the model never emitted a candidate file at all in either encoding arm, burning the whole 32-40k
output budget inside `<think>` (114 and 67 count/recount passages respectively). Flash-Next does not have that
failure mode: length-finish 0.4 %, median reasoning ~2.0k chars, and it won ft09 6/6 on the rig. The reopen audit
(`R-reopen-audit-0908.md` §4 row 1) ranks this the cheapest decisive test in the ledger. It is also the prerequisite
for A2 (persistent workspace / protocol-lite), which is otherwise 3-4 days of build on an untested premise —
especially now that A1 established that persistent *prose* knowledge changes nothing.

**Instrument: rebuilt and re-validated today.** The September scratchpad is gone; `transitions.py`, `backtest.py`,
`llm.py`, `stage0.py` survive in `docs/research-2026-09-02/stage0-artifacts/`. Traces re-fetched from HF
`schema-harness/arc-agi-3-schema-traces` (public, unchanged since 2026-07-16). Extraction reproduces the September
counts exactly: sk48 42 transitions (1 RESET), tn36 12 (12 clicks), cn04 18. **Mandatory validation gate re-run and
PASSED, reproducing September digit for digit:** sk48 `world_model_v5.py` 41/41 and `cleared_level_0.py` 40/41 (the
known missing terminal flag), `cleared_level_3.py` 41/41; tn36 both files 12/12; cn04 both files 18/18.

**Arm:** Arm A (hex encoding, the September pre-registered arm) on the live serving stack — Flash-Next NVFP4 on the
`arc3-flashnext` Modal endpoint, keith V14 profile (kv5, MTP-3, RTX PRO 6000), sampling T=0.6 / top_p=0.95 /
**top_k=20** (the deployed regime; September used no top_k), thinking on. `--max-calls 20 --budget-min 45
--encoding hex --seg-tokens 10000`. **2 draws per game** (September's stated caveat was n=1 per cell).

**KNOWN CONFOUND, recorded before data:** Flash-Next serves at `--max-model-len 32768`; the 27B ran at 65536. Output
budget is therefore SMALLER here (sk48 ~10.7k, tn36 ~12.5k, cn04 ~25.4k after the prompt), not larger. If Flash-Next
fails by exhausting its budget inside `<think>`, that result is confounded by the smaller window and must be reported
as such, NOT as a clean replication. If it emits files quickly (as its telemetry predicts), the confound does not bind.

**READS, locked:**
* **PRIMARY (the September kill rule, unchanged): green on >= 2 of the 3 games {sk48, tn36, cn04} within <= 20 calls
  per game.** With 2 draws a game counts green if green in >= 1 draw (best-of-2) — this is MORE generous than
  September's n=1 and must be stated whenever the result is quoted. >= 2/3 → the lane REOPENS: build A2 /
  protocol-lite with evidence. <= 1/3 → the lane stays dead, **A2 is dead by construction**, and Track A becomes
  A3/A4 (port NOOA / Polyphony as alternative loops) plus Track D (Oct-1 absorption).
* **SECONDARY, and the diagnostically important one: EMISSION.** Share of calls that emit a candidate file at all,
  and calls-to-first-file, per game. September's 27B emitted ZERO files on sk48 and cn04 across 4 game-runs / 12
  calls. "Emits files that are wrong" is a materially different and more hopeful failure than "never reaches the
  emit step", and would justify a follow-up; "never emits" on the same two games replicates the kill on a brain that
  terminates, which closes the lane properly rather than by budget artefact.
* Also logged: best matched/total per game, finish=length share, tokens per call, wall per game, cost.
* tn36 is the control: it was green in BOTH September arms, so a non-green tn36 here means the harness or endpoint
  is broken, not the brain — investigate before reading anything else.

## AMENDMENT — sk48 mitigation arm (written 2026-09-09 after draw 1, BEFORE the arm is run)

Draw 1 confirmed the confound this pre-registration anticipated, and only on sk48. Output budgets after the prompt:
sk48 10,632 tok, tn36 12,503, cn04 25,338. tn36 went GREEN (12/12, 3 calls, 187 s) so the instrument and endpoint are
healthy; cn04 burned its full 25.3k twice with no file, which is a FAIR test (25k ≈ the 27B's 40k and ~10x
Flash-Next's normal reasoning length) and replicates the September kill on a brain without the non-termination
pathology. sk48 at 10.6k is NOT a fair test and must not be scored as a kill.

Because a green sk48 would make the tally 2 of 3 and REOPEN the lane, that cell decides the outcome and is given a
fair test rather than being written off.

**Arm:** identical in every respect (same data, same contract, same backtest, same sampling, same stop rules) except
`--prompt-cap 8000`, which makes the hex encoding carry fewer full grids and more per-row diffs, moving budget from
prompt to output. Target: prompt ~8k, output ~24k — i.e. sk48 gets the same room cn04 already had and failed with.
2 draws, sk48 only. Everything logged as `runs/sk48*` under `mitigate1/`, `mitigate2/`.

**READS, locked before the arm runs:**
* If sk48 emits a candidate file at all (green or not), the "never reaches the emit step" failure is budget-bound on
  this game and the September kill does NOT cleanly transfer for sk48 — report it as an encoding/window artefact.
* If sk48 goes GREEN in >= 1 of 2 draws → tally 2/3 → **the lane REOPENS**: build A2 / protocol-lite with evidence.
* If sk48 still emits NO file across both draws at ~24k output → the kill is confirmed on two independent hard games
  (sk48 and cn04) at fair budgets → **tally 1/3, lane stays dead, A2 dead by construction**, Track A goes to A3/A4
  (port NOOA / Polyphony) + Track D.
* The tn36 green is NOT counted twice: it is the control, already green in both September arms and here.
