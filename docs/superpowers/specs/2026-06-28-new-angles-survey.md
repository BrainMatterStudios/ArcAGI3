# NEW-ANGLES SURVEY — LLMs / multimodal / world-models / scoring-structure (2026-06-28)

4 adversarial research agents (offline-LLM priors, world-models, foundation-policy+TTT, scoring-structure
+ creative), each loaded with the walls/graveyard/offline+T4 constraint and demanding cheap kill-tests.
They converged. This is the synthesis + ranked next-experiments.

## The unifying conclusion (all 4 agents, independently)
**The wall is INFORMATIONAL, not algorithmic.** The binding constraint is W1 (goal unobservable offline)
+ W2 (reward starved) — properties of the observation stream, not the search policy. So no scoring-structure
or search-algorithm cleverness on our side wins. Every surviving winning-EV angle reduces to ONE question:

**THE CONVEX-HULL GATE.** Can we build an offline generator whose SUPPORT contains the hidden mechanics?
Every offline rich-prior bet (curriculum→in-context policy; baked goal-hypothesis library) lives or dies
here. The strongest scale evidence — DeepMind **AdA** (500M params, meta-RL over 10^40 XLand tasks) —
shows in-context adaptation generalizes WITHIN the generator's support but does NOT cross the out-of-support
boundary. ARC-AGI-3 hidden games are adversarially out-of-support BY DESIGN. Our BET 3 already measured this
at small scale (novel affordance = hard 0.000). **The decisive cheap test: does scaling mechanic-family
count move held-out novel-class off the zero floor?** If flat at zero → the entire offline-prior program is
capped near 0.33; if rising → green-light a large curriculum build.

## External SOTA grounding (new, important — reframes our on-device-LLM kill)
- The actual ARC-AGI-3 winner ("Executable World Models", arXiv 2605.05138; Tufa-adjacent) uses **GPT-5.5
  high-reasoning ONLINE at inference**: maintain an executable Python world model, refine it by **next-state
  prediction error** (dense → escapes W2 without game reward), enumerate **goal-hypotheses from the LLM's
  prior** and test them interactively (→ escapes W1 by a prior over designer intent, NOT single-frame
  perception). 58% RHAE. **This is ONLINE → ILLEGAL for the Kaggle OFFLINE track.**
- Reframe of our killed on-device-LLM probe: that test asked a model to guess the trigger COLD from one
  frame — the wrong test. The winner never does that; it carries a goal PRIOR + cheap hypothesis-test loops.
  So "offline LLM prior" isn't dead the way we recorded — but the STRONG version needs frontier-scale
  reasoning unavailable on a T4. The legal question is how much of that prior we can BAKE offline.
- T4 reality: every T4-runnable open VLM (SmolVLM-2B, Qwen2.5-VL-3B/7B, Moondream) is WEAKER than the
  qwen2.5:14b/gemma3 we already killed at cold goal-guessing. No T4-resident model reasons at winner scale.

## Ranked surviving bets (EV-toward-WINNING, adversarially filtered)

### GATE (run first) — Convex-hull scaling probe. Cheap, decisive, gates everything below.
Extend the BET 3 synthetic engine from 4 → ~12-16 genuinely-distinct mechanic families; rerun the
leave-one-family-out in-context probe and plot held-out novel-class accuracy vs family-count. **Pre-reg:
flat at chance as families grow → offline-prior program capped (bank 0.33); clear upward trend → green-light
the big curriculum build.** ~1 day (reuses BET 3 harness). The single highest-value experiment on the board.

### Bet 1 — Offline-synthesized executable-world-model library + symbolic verifier-search (LEGAL proxy to the winner).
Use a frontier LLM OFFLINE to synthesize a frozen Python library of (dynamics primitives) + (goal-hypothesis
enumerators); ship as pure code; at eval run a T4 verifier-search: rank candidate world-models by next-frame
prediction error, plan the SHORTEST actions to confirm the top goal-hypothesis. Escapes W1 via baked goal
prior, W2 via dense prediction error. **Honesty flags:** (a) overlaps the killed symbolic-FM+planner —
survives ONLY if goal-hypothesis ENUMERATION is the genuinely-new part (verify the killed version didn't
enumerate goals); (b) the action-efficiency trap — testing k hypotheses costs actions and the squared metric
punishes it; (c) a hidden game with a goal-type outside the library = hard zero (the convex-hull gate again).
Medium cost/uncertainty. Gated on the convex-hull probe being non-zero.

### Bet 2 (CHEAP FALSIFIERS — afternoon each, run as settlers):
- **Monotone-invariant / "manufactured reward" probe (≤½ day, offline replay over existing traces):** mine
  scalar grid-functions (color counts, components, symmetry defect, region occupancy) that move monotonically
  toward observed level-ups; test on a held-out family whether ANY frame-scalar predicts level-progress (AUC
  vs level-ups). Decisive: no scalar predicts → progress variable unrendered (perception wall, e.g. ls20
  rotation) → KILL; one predicts → a manufactured dense reward the campaign never had. Only creative angle
  attacking W2 with a non-learned method.
- **VLM-interestingness AUC falsifier (~2h):** does a frozen T4-VLM's "closer-to-goal?" ranking beat AUC 0.5
  vs true reward-edge states? Expected ~0.5 (consistent with W1 + sub-1% frontier scores) → quick kill.
- **Portfolio oracle-ceiling settle (~2h, existing bakeoff traces):** mean_game[max_policy S] vs
  max_policy[mean_game S] on HOLDOUT. If gap < ~0.03 → diversity has no payoff (max of correlated zeros);
  multi-strategy portfolio under max-over-plays is NO-GO (distinct from killed multi-seed in mechanism,
  identical in outcome — budget-binding reproduces 0.18).

### Confirmed DEAD / re-killed (do not re-propose)
- MuZero/EfficientZero (learned reward starved by W2), Dreamer/latent-imagination + WM-intrinsic
  (curiosity/empowerment/disagreement = killed novelty class), Genie/video-prediction (dynamics without a
  goal; we already had 93-100% pred acc → +0 levels), distillation of a frontier model into a small policy
  (→ collapses to killed on-device-LLM on novel mechanics), small VLM-at-eval (weaker than already-killed),
  map-then-exploit geodesic replay (done this session — desync + fictional headroom), multi-SEED (0.18).
- Control-endogenous (inverse-dynamics) learned state key for faithful replay: the WM agent rated it the top
  WM lever, BUT it's an EFFICIENCY lever (levels already reached), and this session's multi-play PoC already
  showed faithful replay only helps already-capped levels + the headroom is fictional; plus ls20's goal
  variable is invisible in frames (control-endogenous latents discard exogenous state). DOWN-RANKED to ~dead.

## Bottom line
No wall-breaker was found — the wall is informational and the legal-offline ceiling near 0.33 stands UNLESS
the convex-hull gate opens. The one experiment that could change the strategic picture is the **convex-hull
scaling probe**; everything else is either a cheap falsifier (run to close threads) or gated on that probe.
The leader-class path remains BET 1 (adopt-and-harden the external #1, which is the online executable-WM we
are legally barred from running directly).
