# June-30 ADOPT-AND-HARDEN PLAN — capture the winning offline method

**Context (validated this campaign):** our explorer + max-over-plays portfolio caps at ~0.33; the score gap
is 7 zero-games needing novel-mechanic solving (W1 wall), which NO buildable-offline approach crosses
(validated ~15 angles: learned CNN/RL at T4 scale, VLM read/steer/plan 3B+7B, real StochasticGoose, the
reference, portfolio/config/seed diversity, generic + per-game structured cracking). The leaders score
0.5–1.21 OFFLINE on this exact eval -> a winning offline method EXISTS; it is undisclosed until the June-30
Milestone-1 open-source requirement. This plan captures it fast.

## 0. The moment (00:00 UTC 2026-06-30 = ~the submission reset)
Prize-eligible teams MUST open-source (CC0/MIT-0) by June-30 to claim Milestone #1. Watch:
- Kaggle: `arc-prize-2026-arc-agi-3` discussion + the leaderboard team profiles (Tufa Labs, Tong Hui Kang,
  "the last dance", Redfield Rentals, SVG, Helmut AGI, Barada Sahu) -> their linked GitHub.
- ARC Prize: arcprize.org/blog, the ARC Prize Discord (#arc-agi-3), github.com/arcprize.
- GitHub search: "arc-agi-3" "arc-prize-2026" pushed:>=2026-06-29, sort by stars/recency.

## 1. Triage the dropped code (first 30 min)
Classify the winner's core into one of our integration slots:
- **standalone policy** (CNN/RL/search) -> drop into `learned_explorer.py` `Learner.act` (abstain-default).
- **graph/exploration heuristic** -> wrap behind the abstain-default so it can only ADD candidates (W3 firewall).
- **online-learned model (GPU)** -> route through the T4 fail-safe (`_gpu_ok` verifies a real CUDA op).
- **hosted-LLM/online API** -> ILLEGAL offline; if that's their method, it doesn't transfer -> log and skip.

## 2. Reproduce their number FIRST (do not integrate a misread)
Run their solution UNMODIFIED on our dev games via the proven offline-adapter pattern
(`scratchpad/ref_offline.py` for the 3rd-place agent; `scratchpad/goose_offline.py` for StochasticGoose --
both bypass the online API + feed our offline frames). Confirm it scores on games WE wall at (the 7
zero-games: re86, wa30, sb26, ka59, bp35, g50t, dc22). DECISIVE: if it clears any zero-game, it crosses our
wall -> harden it. If it walls identically, their win is game-specific/rule-exploiting -> adopt-inert, keep
the portfolio.

## 3. Target the zero-games (where the score lives)
Mechanics partially reverse-engineered this session (use to verify the adopted method actually solves them):
- **re86** targeting: cursor (color 0/9) moves A1-4; A5 snaps to the fixed white crosshair (color 11);
  yellow (4) fixed; ~95-step episode limit. Win condition NOT "cursor->yellow+A5" nor "align->white+A5".
- **dc22** multi-object maze: tiny avatar (color 14) + divider line of color-0 dots at col 32-33 + many shapes.
- **wa30** slide-until-wall (asymmetric deltas A1 -4 / A2 +7); **sb26/ka59/bp35/g50t** pattern/click puzzles.

## 4. Harden + integrate (hours, not days)
- Graft into the harness with **banked TransferExplorer + the +4 portfolio as the FALLBACK branch** (the
  winner's preview overfit-collapsed 12.58%->0.25%, so hardening for generalization = real ownable value).
- Add the winner's solver to the PORTFOLIO as an additive play (max-over-plays) so it CANNOT regress the
  games we already win -- it only adds the zero-games it cracks. This stacks the +4 portfolio under it.
- T4: push with `--accelerator NvidiaTeslaT4`; the kernel pipeline (`submission/devkit_build.py`,
  `planner_kernel_build.py`) + model-attach (`qwen-lm/qwen2.5-vl` pattern) are proven working.
- Gate: strict-superset on dev (no zero-game regression) + clears >=1 zero-game. Submit on a measured win
  (USER-GATED).

## 5. Assets ready (built this campaign)
- `portfolio_policy.py` (+4, strict-superset, full_reset safety floor) -- the fallback the winner stacks on.
- `learned_explorer.py` (T4 fail-safe + abstain-default Learner slot) -- the receive-slot.
- Offline adapters (`ref_offline.py`, `goose_offline.py`) -- run ANY external agent on our games in minutes.
- T4 kernel pipeline (verified across 5+ runs) + Kaggle model-attach pattern.
- Zero-game mechanic notes (above) -- point the adopted method at the right problems immediately.

## Honest expectation
If the dropped winner's method is offline-legal and crosses the zero-game wall, this gets us to leader-class
within a day. If every prize-eligible drop is hosted-LLM (illegal offline) or game-specific, the realistic
ceiling stays ~0.33 + the portfolio's margin, and the genuine answer is that the offline non-frontier
paradigm is harder than the leaderboard implies. Either way, this is the fastest path from a real lever.
