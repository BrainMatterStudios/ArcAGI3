# HANDOFF — 2026-07-27 — Independent audit closed; Phase-0 complete; protocol frozen

**Start here.** Supersedes HANDOFF-2026-07-26 everywhere they conflict.
Read in order: memory `arcagi3-audit-2026-07-26-corrections` (ground-truth corrections —
several older memories are amended in place) → `docs/REVIEW-2026-07-26-independent-audit.md`
(full 46-agent audit) → `docs/A1-PROTOCOL-2026-08.md` (**frozen** August gate protocol).

---

## 0. THE HEADLINE

Ahmed's "I suspect something is misrecorded" instinct was right, twice. A 46-agent
independent re-verification (all claims re-derived from primary sources, 3-vote verified)
found and fixed:
- **Wrong facts**: RESET costs **1 scored action** (not 0); the temp-0.3 sampling "defect"
  **never shipped**; pooled base mean is **0.918** (n=5), not 0.98; run-8's −12.5% val gate
  ran 43 rows (not 2) but is **fit evidence only** (100% train/val episode overlap).
- **Missed levers**: post-WIN RESET starts a **new play** (scorer takes max) — floor-safe
  efficiency replay on fully-won games; per-game score is capped at **completed-level weight
  share** → depth strictly dominates efficiency; ties break by **earliest submission**.
- **Two CRITICAL bugs in same-day code** (serving assert): peft prefix KeyError + vacuous
  endpoint check — either would have recreated the 1.26 LoRA-never-served disaster. Fixed,
  real-peft regression-tested (commits 215394f).
- **Confirmed verbatim**: final = **≤2 selected subs on a separate 55-game PRIVATE set**
  (110 hidden total), scored during the original run, almost certainly no rerun →
  **endgame = select the final 2 as DUPLICATES of the single best-evidenced config**;
  the public LB is a proxy, never a measurement instrument (+0.10 needs ~45 draws/arm).
- Also: **8 orphaned busy-loop processes** from an audit agent burned ~99 CPU-h overnight —
  new law: `ps` sweep after every agent workflow.

## 1. CURRENT STATE (all verified, all committed on `winning/duck-patched`)

| item | state |
|---|---|
| Pinned base distribution | n=5 {0.92, 1.14, 0.82, 0.75, **0.96**} → **mean 0.918, sd 0.149**, mean CI (0.73, 1.10) |
| Nightly farm draws #6-9 | ARMED through Jul 31 (detached `scratchpad/autosubmit_nightly.sh`; **check `pgrep -f autosubmit_nightly` after any reboot** — re-run detached if dead; log `scratchpad/autosubmit_nightly.log`) |
| Run-8 adapter artifacts | dataset `arc3-sft-k3-ckpts` **v2** (ckpt-8 + sft_adapter; v1 is POISONED — verify version on every mount) |
| serve-verify v2 builder | edits DONE (GATE_CKPT→checkpoint-8, sft_adapter glob, peft-prefix fix) — push after Aug 1 |
| Serving assert | `submission/_serve_verify_k3/serving_assert.py`, 12 tests, real-peft regression |
| Doctrine pack | `submission/_duck_doctrine/` (D1 field guide / D2 K3 playbook+anti-HUD / D3 guided-A7), 10 tests, env-toggled, all-off byte-identical |
| Two-tier ledger | `submission/_duck_fixes/ledger_fixes.py`, 10 tests, Stage-1 ReplayMockLLM PASS (applied-off byte-identical) |
| Pinned holdout | `scratchpad/holdout_arcint/` — 13 games, every one BFS- or witness-proved winnable (8 click / 5 movement); ff01/sy01/sq01 proved by `witness_probe.py` full engine WINs |
| A/B instrument | `scratchpad/rl_gate/ab_driver.py` (paired arms; validated vs mock incl. per-arm prompt content) |
| Scored-dataset ref | `scratchpad/taaf_scored_ref/` (committed copy of taaf-src-hybrid Jul-03) — **all measurement runs set `TAAF_ROOT=scratchpad/taaf_scored_ref`; the local `_adopt` tree is DRIFTED** |
| GPU quota | 0 until **Sat Aug 1 00:00 UTC** (30h/week shared) |
| Tests | 32/32 across the three new suites |

## 2. AUGUST — EXECUTE `docs/A1-PROTOCOL-2026-08.md` AS WRITTEN (thresholds frozen)

Gate 0 serve-verify v2 (~3-4h, both artifacts, timed merge rehearsal) → Gate 1 pilot
banding (1×13 base rollouts; keep games where base wins ≥1 but <all; freeze panel in §7)
→ Gate 2 checkpoint sweep (base/ckpt-8/step-15 server arms × 2 paired rollouts; adapter GO
= paired Δ≥+3, no game −2, scorecard non-inferior) → Gate 3 doctrine A/B (D1+D2, GO ≥+2;
D3 decided on official-dev evidence only) → stacked debut with in-sub serving assert,
**promptly** on green (earliest-submission tie-break), 7-draw stop rule vs base mean −0.10.
Example Gate-3 invocation is in ab_driver.py's docstring. Sampling always 0.6/0.95/top_k 20.

## 3. THE LAWS (additions; older ones stand)

11. **Audit and build against the DOWNLOADED scored dataset** (`taaf_scored_ref`), never
    the local `_adopt` tree (drifted ~Jul-05, never uploaded).
12. **Public LB is not an instrument.** All selection via paired offline A/Bs; slots are
    for farming the yardstick and gate-passed debuts only.
13. **Depth beats efficiency** (completed-share cap): capability > death-avoidance >
    efficiency, in that order, for every lever decision.
14. **Pre-registration is binding**: protocol thresholds change only via append-only
    amendments that demote results to exploratory.
15. **`ps aux` sweep after every multi-agent workflow** (the 99-CPU-hour busy-loop lesson).
16. RESET costs 1 scored action; only a post-WIN full reset (new play) is free — and that
    one is a LEVER (unexploited; requires run_complete ledger preservation if shipped).

## 4. PENDING-AHMED

- **Post the forum question** (rerun-at-selection + games-per-run + exact cap — text at the
  bottom of this doc). It de-risks the endgame selection strategy.
- Milestone-2: Sept 30 ($25K/$7.5K/$5K); final entry deadline Oct 26; winner license
  CC-BY 4.0 + open weights (binding Kaggle rules; easier than the CC0/MIT-0 we assumed).
- No new spend authorized; teacher-API (SCoRe v2) still Ahmed-gated; Claude-output
  training still forbidden.

## 5. FORUM QUESTION (post as-is, Discussion tab, tag the hosts)

> **Title: Three clarifications on final scoring mechanics**
>
> Hi organizers — three factual questions about how final scoring works, which I couldn't
> find in the rules or docs:
>
> 1. When we choose our 2 final submissions at the end of the competition, are their
>    private-leaderboard scores the ones already computed during each submission's
>    original scoring run, or are the selected notebooks re-run/re-scored at that point?
> 2. During one scored run, does a submission play the full hidden set used for both
>    leaderboards, or only the subset backing the public leaderboard? (Equivalently: how
>    many game environments does one notebook run see?)
> 3. Is the effective notebook wall-clock limit for scored runs exactly the 9 hours from
>    the Code Requirements, or is there additional headroom (one official page mentions
>    "under 12 hours")?
>
> Thanks — happy to be pointed at existing documentation if I missed it.
