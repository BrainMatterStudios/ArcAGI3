# NEXT-SESSION BUILD PLAN (from next-paradigm-research workflow wf_eab5ed08-c8d, 2026-06-28)

18 candidates, 2 survivors; honest verdict: no survivor escapes the SPIRIT of W1-W4; realistic path = adopt+harden the June-30 winner. Run the cheap kill-tests below WHILE waiting.

Files and cited lines verified: `agent.py` salient_click_targets at coarse step-8 (L49-60), `spatial_value_explorer.py` the 16→24 Conv2d click-value head that diverts (L45-60), `cai_prune_explorer.py` the ~51% no-op audit (L1-12), `synthgen/{canvas,genres}.py`, `learned_explorer.py`, `scripts/rhae_headroom.py` all present.

---

# NEXT-SESSION BUILD PLAN — ARC-AGI-3 OFFLINE

## Framing (read first)
Only two non-trivial survivors came in (#1 decode-tufa, #2 orthogonal). Both are real escapes of the *letter* of W1-W4, both carry the same dominant prior that has killed everything before them: **dev-efficiency wins are Kaggle-inert because the scored games wall at L0/L1**, so any lever that only pays off on *completed* levels converts to ~0 RHAE on the hidden set. The honest third "bet" is not a new idea — it is **adopt-and-harden the June-30 open-source winner** into the hardened harness. Ranked by EV-toward-*winning* (not EV-toward-a-nice-result), adopt-and-harden outranks both research arms.

---

## BET 1 — Adopt-and-Harden the June-30 open-source #1 (EV rank 1)
**Why it ranks first:** Winning needs a signal/method not in our observations. The Milestone-1 winner is, by construction, a method that *does* score on the hidden games (that is what won them #1). Our hardened harness (LearnedExplorer + T4 fail-safe + abstain-default Learner slot, tu93 stays L5) is purpose-built to *receive* an external policy without coverage collapse. This is the only path on the board whose ceiling is "leader-class" rather than "≤0.33 + a capped efficiency margin."

**Concrete spec (Day 1 the moment the source drops):**
1. Clone winner; classify their core into one of our slots: (a) standalone policy → drop into `learned_explorer.py` Learner interface; (b) frontier/exploration heuristic → wrap behind the abstain-default so it can only *add* candidates, never reorder/prune (W3 firewall); (c) online-learned model → route through the T4 fail-safe verify-real-CUDA-op-or-fallback path.
2. Run their solution **unmodified** on our dev trace set first (reproduce their claimed number on games we share), to confirm we're integrating a working artifact, not a misread.
3. Only then graft into the harness with the banked-0.33 floor wired as the fallback branch.

**Single cheapest Day-1 decisive kill-test:** Run the adopted policy against the harness's banked TransferExplorer on the dev games that wall at L0/L1 (the scored-set proxies). **Decisive:** if it clears L0/L1 on games TransferExplorer cannot, it escapes our wall and you harden it. If it walls at the identical levels, their win was either game-specific or rule-exploiting and **does not transfer to our harness** → it becomes the fallback's equal, not a win.

**What makes it a graveyard entry:** Winner turns out to be a hosted-LLM/online-API approach (illegal in OFFLINE track), or its gains are entirely on games disjoint from the hidden scored families, or it reduces to a method already in our graveyard (graph-explore / colour-transfer / on-device LLM). Then: log "adopt-inert, same wall," ship banked 0.33.

**Honest caveat:** This bet is *gated on an external event* (source publishing). It is rank-1 on ceiling, but you cannot start it until the drop. Bets 2/3 are what you build *while waiting*.

---

## BET 2 — Oracle-Reorder Ceiling Probe for the Click-Affordance Lever (merged #1 decode-tufa) (EV rank 2)
**This is a kill-test, not yet a build.** The proposal's own decisive weakness is that it only pays RHAE on levels already completed, and the level-up click is one change-cell among many (picking *which* is the W1-dead progress question). So we run **only the zero-training oracle ceiling first** — no Conv2d, no training. If the ceiling is thin, the entire arm dies in an afternoon with no code beyond a replay script.

**Concrete spec (the probe, ~half a day):**
- Reuse existing dev traces. Restrict to games that wall at L0/L1 (scored-set proxies).
- For every **completed** level: replay recorded clicks; count `pre_levelup_noop_clicks` = no-op clicks a *perfect* change-affordance oracle would have reordered to *after* the level-up click.
- Convert action-saving to RHAE delta (quadratic), weight by level. This is the **upper bound of the whole lever** — a real within-level Conv2d head can only do worse.
- Grounding facts already in repo: `agent.py` L49-60 (`salient_click_targets`, coarse_grid_step=8, the heuristic that already front-loads change-likely cells into 5 salience tiers, dumps the lattice to prio-9 — the head only adds value on prio-9 background and inert-vs-live disambiguation, a thin margin); `cai_prune_explorer.py` L1-12 (the 51% no-op audit this lever targets); `spatial_value_explorer.py` L45-60 (the click-VALUE head we already tried that *diverts* — the proposal must stay strictly an action-space-reduction prior, never a value signal, or it collapses into this known failure).

**Single cheapest Day-1 decisive kill-test:** the probe above **is** the kill-test. **Decisive thresholds, pre-registered:** oracle-reorder weighted-RHAE gain on scored-style completed levels — if **< ~0.05 absolute** (because levels complete via movement, the level-up click is already early under salience, or no-op fraction on the *completed-level prefix* is low) → **DEAD, no training, same wall as "dev-efficiency Kaggle-inert."** Only if gain is **large (e.g. > 0.15)** do you spend a day on the within-level head and test held-out AUC ≫ 0.5 on *untried coordinates of the same level*.

**What makes it a graveyard entry:** Oracle ceiling is thin (most likely outcome given the wall prior), OR the scored games complete L0/L1 via movement so no-op-click fraction is small, OR the head's held-out AUC ≈ 0.5 (can't beat the existing salience tiers). Entry: "click-affordance oracle ceiling = X RHAE, sub-threshold, capped lever on already-completed levels — inert on L0/L1-walled scored set."

---

## BET 3 — Held-Out-Family Synthetic Feasibility Probe for In-Context Goal-Inference (merged #2 orthogonal) (EV rank 3)
**Also run as a kill-test first, not the full build.** The full meta-RL build (procedural POMDP generator + RL² GRU/decision-transformer on dual-T4) is the highest build cost on the board, and ARC's adversarial-novel-by-design mandate is meta-RL's worst case (out-of-family generalization). **Do not build the generator or touch real games until the cheap synthetic probe clears.**

**Concrete spec (the probe, ~1-1.5 days):**
- Build a **minimal** generator: 3-4 mechanic families only (gate, switch, collect, reach) on the OFFLINE engine, reusing `synthgen/canvas.py` role-randomization so colour/shape identity carries **zero** surface signal.
- Train an in-context GRU / small decision-transformer over (frame, action, frame-change) history with **leave-one-mechanic-family-out**.
- Measure on the **never-trained held-out family**: in-context level-up rate and actions-to-levelup, synthetic→synthetic (no sim-to-real gap — the easiest possible version of the real problem).

**Single cheapest Day-1 decisive kill-test:** the held-out-family level-up rate vs a random-policy baseline. **Pre-register both:** (a) held-out-family level-up delta vs random, and (b) actions-to-levelup vs the 35× RHAE headroom (`scripts/rhae_headroom.py`) on solved held-out-family games. **Decisive:** if the meta-agent cannot infer a held-out *synthetic* family from interaction history — with zero sim-to-real gap — it has **no path** on adversarially-novel real games, and **W1's strong form is confirmed**. KILL at a fraction of full build cost. Only on a clear margin do you proceed to leave-one-*real*-game-out.

**What makes it a graveyard entry:** held-out-family level-up ≈ random (most likely — confirms W1-strong), OR it clears synthetic but the real-game leave-out collapses (sim-to-real on adversarial novelty). Entry: "in-context amortized inference cannot generalize across mechanic families even synthetically → W1 strong form confirmed, transfer hypothesis class closed."

---

## (3) THE SINGLE CHEAPEST DECISIVE FIRST EXPERIMENT
**Run BET 2's oracle-reorder ceiling probe.** It is the cheapest (no training, no new model, pure replay over existing dev traces), the most decisive (it is the *upper bound* of an entire arm), and it settles the live reduction risk shared by every efficiency lever in one afternoon. Sequence the week:
1. **Day-1 AM:** Bet-2 oracle probe → likely kills or de-risks the click arm by lunch.
2. **Day-1 PM → Day-2:** start Bet-3 minimal synthetic generator + held-out-family probe (the only arm that could *settle W1's strong form* and has a firewall floor).
3. **On June-30 drop:** pivot to Bet-1 adopt-and-harden immediately — it preempts both.

---

## (4) HONEST VERDICT
**No survivor genuinely escapes the *spirit* of W1-W4.** Both research arms escape the *letter* and are worth their cheap kill-tests, but:
- **Bet 2** escapes W1-W4 mechanically yet is **ceiling-bound, not wall-bound**: even at its oracle upper bound it pays RHAE only on *already-completed* levels, and the scored games wall at L0/L1 — the exact "dev-efficiency is Kaggle-inert" reduction that killed CAIPrune (0.28). Probability it converts to a real Kaggle gain: low. Worth running *only because the kill-test is an afternoon.*
- **Bet 3** is the only arm that attacks the actual wall (offline goal-ID), and its probe has genuine scientific value (settles W1 weak-vs-strong). But the base rate — adversarial-novel-by-design hidden games + sim-to-real on out-of-family generalization — makes a *winning* ceiling unlikely even if it clears the synthetic probe. Expected outcome: it confirms W1-strong and joins the graveyard, which is still valuable (it closes the meta-RL hypothesis class definitively).

**The offline non-LLM ceiling near 0.33 still stands.** The reproducible paradigm caps there because the signal needed to deploy our real capabilities (ls20 perception, 35× headroom, chain-recurrence) safely is not in the observations, and W4 says the help-vs-derail discriminator to gate those capabilities is itself unlearnable in-regime.

**The realistic path to winning is BET 1 — adopt-and-harden the June-30 open-source #1.** It is the only path whose ceiling is leader-class rather than "0.33 + capped margin," and our hardened harness exists precisely to absorb it without coverage collapse. Recommendation: run Bet 2's probe as the Day-1 cheap settler, spend the wait building Bet 3's W1-strong probe, and **hold dual-T4 capacity in reserve for fast Bet-1 integration the moment the winner publishes.** Do not invest a multi-day full build into either research arm before its kill-test clears — both are odds-on to join the graveyard.

Relevant files: `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/agent.py`, `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/cai_prune_explorer.py`, `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/spatial_value_explorer.py`, `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/learned_explorer.py`, `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/synthgen/{canvas,genres}.py`, `/Users/ahmed/Documents/ArcAGI3/scripts/rhae_headroom.py`.