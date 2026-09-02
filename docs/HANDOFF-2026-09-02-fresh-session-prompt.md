# Fresh-session prompt (amended 2026-09-02 after review of the 08-29 -> 09-02 session)

Amendments vs the prompt drafted at the end of the previous session, each from a
verified defect in that draft (review recorded in this session's transcript):
1. Fact 3 claimed "same-boot comparisons are readable". Every same-boot kernel ran stock
   first and graft second; no A/A control existed. Replaced with the true state.
2. Fact 2 called two single draws (1.71 vs 1.72) "FLAT" and named unseen-game comprehension
   as the strongest hypothesis. Two draws with sd ~0.3 each cannot distinguish 0 from +0.4;
   the cheaper explanation (the local A/B was an order artifact or noise) is now listed
   first and is under test.
3. "85 of ~110 hidden games are unseen" assumed the 25 public games sit inside the scored
   set. Nothing in the repo verifies that; it is now flagged unverified.
4. Added the operational facts that were missing: slots left, nothing armed, ledger fixed,
   pushes are NOT blocked by the auto-mode classifier (26/31 went through).

---

```
CONTEXT — ARC-AGI-3 Kaggle competition (arc-prize-2026-arc-agi-3), repo /Users/ahmed/Documents/ArcAGI3,
branch winning/duck-patched. Deadline Nov 2 (about 60 daily slots left). Goal: score MUCH higher — the
leaderboard top (cstl) is 7.51, the chasing pack 3.85–4.99, our best-ever 1.88. Read the current
leaderboard yourself first (kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --show).

YOUR MANDATE: conduct your own INDEPENDENT research to find a path to a dramatically higher score.
Do NOT trust the conclusions in memory files, handoff docs, or past session notes — they contain prior
sessions' mistakes and dead reasoning. Treat them as a catalog of what was TRIED, not what is TRUE.
Re-derive anything you rely on from primary data: raw traces, the scoring engine source, live submission
records, the Kaggle API, public datasets and repos.

HARD FACTS you may take as given (each verified against primary sources on 09-02; re-verify if in doubt):
1. One submission per UTC day; scored rerun = 9h box, ~110 games (the 25 public games are NOT verified
   to be inside the scored set — treat "unseen" arithmetic as unverified), one RTX Pro 6000 96GB, no
   internet. Scoring per level min(115, 100*(baseline/actions)^2), weighted (level+1), capped by
   completion share; game = max over plays; retries accumulate actions into the level's bucket.
2. Live draws on identical serving (27B "V22 stack"): stock 1.71 (sub 55927189), stock+TP9 graft 1.72
   (sub 55950252). Best live: 1.88 (Qwen3.8-Flash-Next, sub 55902917). Per-draw sd is ~0.3, so this pair
   cannot distinguish a zero lever from a +0.4 one. TP9's local "+50% levels" came from ONE same-boot
   kernel that ran stock first and TP9 second with NO stock-vs-stock control; the flight commit of the
   same TP9 arm drew 4.55 locally vs stock's 4.50. The cheapest explanation is that the local A/B was
   an order artifact or noise. Two control kernels (arc3-v22-aa, arc3-v22-ba-tp9) test exactly this;
   design and reading rules are pre-registered in docs/EXPERIMENT-2026-09-02-ab-instrument-control.md.
   Read their results before believing ANY local lever number.
3. Local instruments that exist: a two-phase same-boot kernel (submission/_v22_ab) whose phase-order
   validity is UNDER TEST; a serving load-bench pattern (arc3-servebench27); attested gated submission
   runners (scripts/submit_*.py). Cross-boot stock draws locally: 4.27, 7.14, 4.50, 4.55 — the local
   25-game mean has sd ~1.3, so single-boot comparisons across kernels are unreadable. Outputs of scored
   reruns are NOT retrievable; commit-run outputs are (REST kernels/output; kaggle CLI hangs on big logs).
4. External assets verified real: HuggingFace schema-harness/arc-agi-3-schema-traces (public, last
   modified 2026-07-16; 50 frontier-model trajectories at 95–99% RHAE on public games; world-model-as-code
   + backtest + BFS, ~5.9h/game); public repos of the >=78% public-25 tier (NIMI-research/Tycho,
   ryanbbrown/Retrodict) since July; ARC Prize Milestone 2 forces prize-track leaders to open-source by
   Sept 30 (staff post, discussion/713634). Five named forum self-reports say public-25 scores do not
   rank harnesses on the hidden set (5.0->1.4, 6.8->1.19, 3.8->0.9, 5.0+->1.6, 2.8->2.4).
5. Operational: nothing is armed for tonight unless this session's decision tree (experiment doc above)
   fires; docs/submission-ledger.json is current; kaggle kernels push works from the agent (auto-mode
   does not block it); two GPU sessions can run concurrently.
6. Start-here docs (evidence of record, not gospel): docs/EXPERIMENT-2026-09-02-ab-instrument-control.md,
   docs/HANDOFF-2026-08-29-packs-in-flight.md §13–§17, docs/research-2026-08-31/R9,R10,R11,J9,J10.

PROCESS REQUIREMENTS (each encodes a verified failure of the 08-29 -> 09-02 session):
- Evidence over assertion: never claim a mechanism, score, or dataset property you have not personally
  measured or fetched this session. Distinguish a max-statistic from a mean (a rival's best-of-48 was
  read as their expected score, projecting 2.2–2.65 for a draw that landed 1.71).
- Every comparison needs a control that isolates the variable: same boot AND counterbalanced order (or
  an A/A) — a warm server is a variable. Pre-register the reading rule and the noise scale (paired SE)
  BEFORE data arrives, against the correct baseline distribution.
- Lean instruments first: throughput with load generators (minutes), behavior with single-game traces
  (~1h), score with multi-game runs (hours). One variable per experiment — the prior session bundled
  five flags, then four prompt interventions, and could attribute neither.
- Test any graft/kernel against the EXACT target bundle locally before pushing (three GPU cycles were
  burned on seam mismatches and a missing install cell).
- Use fresh-context subagents for research and adversarial judge subagents to attack any build or
  conclusion before it ships. Never launch a Kaggle submission without my explicit approval.
- Approval gate: analysis and building are autonomous; git push, submissions, and anything irreversible
  require my go.

DELIVERABLE: your own ranked assessment of where the next 2–5 points actually come from — including
directions prior sessions did NOT pursue — each with the cheapest decisive experiment, its cost, and a
pre-registered decision rule. Challenge the framing itself if the evidence points elsewhere (model,
harness paradigm, data, or something no prior session considered). Then execute the top experiment.
```
