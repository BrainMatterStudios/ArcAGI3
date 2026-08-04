# Campaign plan (amended 2026-08-04, supersedes all prior plan docs)

Basis: docs/PLAN-REVIEW-2026-08-04.md (33-agent review + corrections). Approved by Ahmed 2026-08-04.

## Objective

**Raise the pinned config's MEAN score to ≥1.5** (bar derived from deflating rival
public maxima to implied means; Kojima 1.86/56 ≈ 1.43 mean). The final prize pays
the mean of 2 selected submissions (private, locked at run time, no rerun,
best-of-2 ≈ mean+0.56σ). A 1.5-mean config also yields a ≥1.8 public draw w.p.
~99% over the remaining slots. "Chase a high draw" is dead as an objective.

## Slot doctrine (~90 remaining)

- Every slot: a pre-registered discriminating hypothesis with an n=1-readable
  fingerprint in the sub message, OR an endgame duplicate. Never an LB mean-test
  at n<10 (MDE at n=3 is +0.28 — the LB is not an instrument).
- All arm selection happens OFFLINE at eval geometry (28 games @ 7920s — the
  permanent A/B standard; the old 10@3600 rig is ~1.27× token-richer than eval).
- ~10 discriminating draws, ~55-60 best-config draws (only A/B winners),
  ~10-15 endgame duplicates, 0 max-farming.

## Endgame playbook (committed now, per the 07-26 doctrine)

- **Sept 20:** Milestone-2 go/forfeit decision (default: forfeit; don't
  open-source the stack for an unreachable milestone).
- **~Oct 20:** config freeze. Last 6-10 slots = byte-identical duplicates of the
  frozen config via scripts/submit_gated.py (byte-check mandatory).
- **Selection rule (pre-registered):** select the 2 clean (non-zero,
  fingerprint-verified) duplicate draws of the frozen config; earliest-sub on
  ties. Never rely on Kaggle auto-select (winner's curse + 7% silent-zero rate).
- **September:** open-source repo prep (CC-BY/OSI) so a win is bankable.

## Active levers (each gated by an offline eval-geometry A/B before shipping)

1. Animation frames as code-queryable sandbox global (frame[-1] currently
   discarded; raw-frames-vs-scalars is the delta). Kill: levels flat AND query
   rate <20% on animating games.
2. Grinder retrigger on level-age (≥120 actions or ≥10 turns on a
   never-completed level) + win-path narration. Success: ≥2 of {m0r0, ls20,
   cn04} unlock; decisive metric = post-narration L2 completion <3× baseline.
3. Watchdog-as-heartbeat: default stall 900s → 600s (beats the 15-min scorecard
   stale-close race). Tail insurance for the finals.
4. SFT track, gated: decontaminated panel (no vc33/sc25/lp85 — run-8 trained on
   them); scale only on ≥+6 levels over 2 waves; tokens/turn must hold (the
   under-deliberation kill); $10-30 frontier-teacher census on
   dc22/m0r0/sk48/tr87 before any corpus buy; one live draw before scaling.
5. UPSCALE 4→8: 1-hour transcription probe first; A/B only on a flip.

## Instruments and hygiene

- Submission→bytes→patch-set LEDGER (docs/SUBMISSION-LEDGER.md): every past and
  future sub mapped to kernel version, notebook hash, patch set, pins,
  hypothesis, result. No conclusion may cite a live draw absent from the ledger.
- Eval-geometry A/B (stock vs v7) runs FIRST on quota reset — it gates every
  prompt-side ship decision. Then the decontaminated run-8 eval.
- Prompt/prefill token accounting added to the rig (gen-tokens alone is half
  the picture).

## Honest odds (on the record)

Levers sum +0.25-0.7 optimistic on a 0.929 base. Top-5 is the central case;
 #1 improved-but-not-favored. The mean-not-max framing and endgame mechanics
are what make whatever is built bankable.
