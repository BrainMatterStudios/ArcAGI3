# ARC-AGI-3 Campaign Synthesis — 2026-08-04

Spot-verified this session before writing: git ancestry of the 0.80 draw's build (`bcef400` 08-03 04:31 IS an ancestor of antifreeze `3fc3f92` 08-03 17:25 — the "v7" live draw is v6), AUDIT 4 numbers (base n=8 mean 0.929 sd 0.195, E[max of 92]=1.412±0.084, P[any≥1.8]≈0.0004, RESULTS-2026-08-02-trace-audits.md:116-132), `frame[-1]` discard (taaf game.py:170), `MULTIMODAL_UPSCALE='4'` (both scored ref and current bundle setup_commands.json), `DEFAULT_STALE_MINUTES=15` (reference/arc-agi-toolkit/arc_agi/scorecard.py:24).

---

## 1. Verdict on the current plan: NEEDS MAJOR AMENDMENT — the objective, step 1, and the missing step 4 are wrong; step 2 is salvageable with gates

**The objective is mis-stated.** The final prize pays the MEAN of your config, not your best draw: private scores are locked at run time, never re-run, and public⊥private per draw (organizer-confirmed, arcagi3-kaggle-submission-contract.md). Every team's expected final ≈ config_mean + 0.56·sd from best-of-2 selection. Deflating rivals' public maxima the same way (Kojima 1.86 over 56 subs → implied mean ~1.43; sanity check: Tufa 1.45/94 deflates to ~0.98, matching their self-reported 0.77/1.21/1.30 spread) puts the real bar at a **config mean of ~1.5–1.6 for #1** (1.45 gives ~34% by simulation; rival improvement over 90 days pushes it up — state the range, not a point). "Reach 1.8" chases a draw your own bootstrap prices at 4-in-10,000; "raise mean +0.55–0.65" is an engineering problem. Re-price every kill/keep decision in mean-shift units.

**Step 1 (farm v7, test mean at n≥3, sample the max) is void three ways:**
- The n≥3 test has MDE ≈ +0.28–0.33 at sd 0.195; a plausible v7 effect (+0.1–0.15) needs 27–60 draws/arm. Your own July law ("~45 draws/arm for +0.10; the public LB is not an instrument") already said this. Step 1 reinstalls the LB as an instrument.
- "Sample the max" contributes ~nothing to the final (private draws unobservable; E[max of all 92 remaining slots] ≈ 1.41 — below the bar even if you spent every slot on it). Its only legitimate payout is Milestone 2 (public LB, Sept 30), where current arithmetic (E[max by then] ≈ 1.38 vs ~1.64 for 3rd) says it's out of reach anyway.
- **v7 has never been scored.** See section 2.

**Step 2 (SFT)** survives but its gate is currently contaminated (vc33 in both training set and behavioral panel), underpowered (2 waves resolve only ~2σ effects; the one resolved effect to date was −47%), and the teacher is half-blind (K3 fails 4/9 of the diagnosed never-unlocked games — scaling K3 data buys mostly what you can already reach). Amendments in section 3, item 7.

**Missing entirely: an endgame step.** Kaggle auto-selects finals by public score if you don't choose — a winner's-curse machine given sd 0.195 and the demonstrated ~7% silent-zero rate (three 3411-byte 0.00s). Your own 07-26 audit prescribed duplicate-final-2 doctrine and the plan silently dropped it. Also absent: Milestone-2 decision date, open-source prep (CC-BY/OSI or removal), earliest-sub tie-break exploitation.

**Amended plan skeleton:** (1) objective = config mean ≥1.5, submit 2 byte-identical duplicates of the best-evidenced config; (2) slots go to pre-registered discriminating hypotheses or endgame banking, never mean-tests at n<10; (3) unlock levers (section 3) are the main track, SFT gated; (4) endgame playbook: freeze ~Oct 20, last 6–10 slots = gated duplicates, pre-registered selection rule, Sept 20 Milestone-2 go/forfeit node, open-source repo started September.

## 2. The anomaly: mostly manufactured; the residual is small, real, and one rig run settles it

**Verified: none of the three sub-mean draws is v7.** 0.76 = the context/compaction arm — the mechanism class round 2 later measured as the campaign's most harmful (0/303 compactions succeeded, −8 levels). 0.78 = the depth pack, whose own rig run showed no gain. 0.80 = build `bcef400`, which git confirms predates antifreeze, the HUD slow-tick fix, and the playbook — i.e., v6; AB-ROUND3's own text says "the live v6 draws sample it." Local and live evidence are therefore broadly CONSISTENT: the predicted-harmful arm scored worst, the null arms scored base-typical-low. There is no contradiction to explain; there IS a config-identity bookkeeping failure — no ledger maps submission → exact notebook bytes → patch set, which is how "0.80 = v7" got believed.

**Residual tension:** P(all three ≤0.80 | arms ≥ base) ≈ 1.6% (≈6.5% after conditioning out the expected-harmful context arm). Mild evidence the shared new-stack machinery costs ~0.1 mean live. Best-supported untested mechanism: **eval geometry**. Local A/Bs ran 10 games @ 3600s (360 GPU-s/game); eval runs 28 @ 7920s (283 GPU-s/game) — eval agents are token-POORER (~0.79×), prompt-token-heavier arms pay more under 28-way contention, and prompt/prefill tokens (the mechanism's currency) have never been measured — gen-token accounting was only fixed 08-03. Compounding: the rig "cap" 1.242 was calibrated against the 1.27 max draw, not the 0.929 mean ("partly coincidental" per your own audit), and A/B gains concentrate in re86/ft09 while the hidden set is ~70% zero-level game-runs, so a real local +25% plausibly dilutes to +0.05–0.25 live. A weaker but free-to-check contributor: the scorecard auto-closes after 15 idle minutes (see item 8) and the rig never starts the cleanup thread, so live truncation is locally invisible.

**The settling tests (zero slots):**
1. **Eval-geometry A/B**: two ~2.2h kernels (not one), stock vs v7, 28 clones @ 7920s box, fixed token hook, read turns/game + tokens/turn + levels-ex-ft09. Pre-registered rule: turns drop >20% under v7 → strip lowest-value prompt components; turns hold → dilution+composition explains the draws and v7 ships on design grounds.
2. **Ledger first**: re-derive every past sub's exact bytes/pins from Kaggle kernel versions before any conclusion cites a live draw again.
3. **Next slot** = the first true v7 draw with the hypothesis pre-registered in the submission message (samples the untested config and the ~0.1 live-cost hypothesis simultaneously).

## 3. Ranked surviving ideas (EV in config-mean units; all survived two independent verification passes)

1. **Objective reframe + endgame playbook** (findings 1+7; skeptic 8, feas 7–8). EV: converts a 0.04% lottery into a +0.55 engineering target and insures the 2 rows where all value concentrates against the demonstrated 7% silent-zero rate. Cost: one session, zero GPU. **Next action:** rewrite plan objective; commit the selection rule (2 clean byte-identical duplicates, submit_gated.py byte-check mandatory, earliest-on-ties); calendar Sept 20 (Milestone 2 go/forfeit) and ~Oct 20 (freeze).
2. **Animation frames as sandbox data** (finding 5; skeptic 5, feas 5). The engine returns a frame LIST; game.py:170 keeps only `frame[-1]`, and the prompt literally promises the model "a short multi-frame animation" it never gets. lf52 animates on 30/30 actions; 8/9 never-unlocked games fail on mechanic misreads whose evidence lives in the discarded frames. Caveat from verification: v7's patch 2 already ships animation SCALARS (count/changed/bbox) and those didn't unlock g50t — the true lever is raw-frames-vs-scalars, and 27B query uptake is unproven. EV +0.05–0.2 mean. **Next action:** ~1 day — expose `last_animation` as a code-queryable sandbox global (near-zero tokens until queried, matching the existing current_frame/diff_frames pattern), one prompt line; A/B at eval geometry, levels-ex-ft09, ≥3 waves; kill if levels flat AND query rate <20% on animating games.
3. **Retrigger the patch11 grinder on level-age + win-path narration** (finding 6; skeptic 6.5, feas 6). Verified: the grinder went dormant solely because its trigger reads the watchdog 900s stall timer (duck_patches.py:1995) that an always-emitting LLM never trips (grinder_engagements=0 across all A/Bs); failed grinds cost exactly 0 score; m0r0=596 states/15 acts, ls20=13 acts with the rotation indicator outside the mask band, lf52=8 clicks. Value routes through narration (grind-won levels themselves score ~0 under squared efficiency) and replay-at-WIN (full-game wins only). Capability-independent — the one lever whose transfer story is immune to hidden-set-composition risk. EV +0.1–0.3. **Next action:** half-day rig run — level-age trigger (≥120 actions or ≥10 turns on a never-completed level), 9-game panel, instrument grinder budget sufficiency; success = ≥2 of {m0r0, ls20, cn04} unlock; then the decisive metric: post-narration L2 completion at <3× baseline actions.
4. **Eval-geometry instrument + prompt-token accounting** (finding 4; skeptic 5, feas 7). Not additive itself, but gates every ship decision and settles the residual anomaly (section 2). **Next action:** run before shipping anything else prompt-side; make 28@7920 the standard A/B geometry permanently.
5. **Slot doctrine** (finding 3; skeptic 6, feas 7). Every slot ships a ≥+0.2-expected config difference, serves a pre-registered discriminating hypothesis, or banks endgame duplicates; all arm selection happens offline on levels-ex-ft09, ≥3 waves; add one n=1-readable live fingerprint per sub (per-game unlock events) as a transfer probe. Zero cost, immediate.
6. **Stale-close heartbeat** (finding 8; skeptic 4, feas 6). DEFAULT_STALE_MINUTES=15 verified; the gateway force-closes idle scorecards at partial score; the local rig never starts the cleanup thread so this is structurally invisible locally. Sharpener from verification: your watchdog default (900s) EQUALS 15 min and races the server's 60s-granularity close — lowering it to ~600s makes the existing watchdog the heartbeat (RESET hits the game server, works during a vLLM wedge). Live gateway threshold UNVERIFIED (the timeout param path is unclamped). **Next action:** one-line watchdog change in the v7 build + 16-min-stall local repro with `on_scorecard_close` set. Tail insurance for the finals; near-free.
7. **SFT track gates** (finding 9; skeptic 6, feas 5). Before spending the quota or $50–150: (a) swap vc33/sc25/lp85 out of the behavioral panel (in: cn04 or lf52); (b) pre-register "scale only if ≥+6 levels over 2 waves" — explicitly a large-effects-only policy, since sub-+6 effects can't close this gap; (c) report tokens/turn and kill on under-deliberation, the same criterion that killed the synth adapter; (d) $10–30 frontier-teacher census (non-Anthropic) on dc22/m0r0/sk48/tr87 + 2 K3-solved controls BEFORE any corpus buy; (e) dated yes/no chase on the Anthropic approval; (f) require one live draw of any adapter config before scaling — a clean rig positive is not yet evidence of live gain.
8. **MULTIMODAL_UPSCALE 4→8** (finding 10; skeptic 5, feas 4). Verified untested input axis (scored=4, code default=16; tr87's glyphs reach the model at ~20px). Caveats from verification: the local A/B also runs at 4 (does NOT explain the anomaly), the len/3 estimator charges base64 URLs as phantom tokens so upscale 8 triggers history trimming — pair with an estimator exemption for image parts. EV +0.02–0.08, plausibly null. **Next action:** 1-hour transcription probe (render tr87 rule table / sk48 legend at 4/8/16, ask the served 27B to transcribe against known ground truth); only on a flip, one A/B wave pair.

## 4. Revised 90-day allocation (~90 slots, ~30h GPU/week, one engineer)

**Slots (~90):**
- ~10: pre-registered discriminating live draws — first true v7 (next slot), then one draw per lever that passes its offline A/B (animation, grinder, adapter), each with a hypothesis and an n=1-readable fingerprint in the sub message.
- ~55–60: ship-the-current-best-config draws that double as mean accumulation — but ONLY configs that won an offline eval-geometry A/B; no LB mean-testing at small n.
- ~10–15: endgame — freeze ~Oct 20; byte-identical duplicates of the frozen config via submit_gated.py; pre-registered selection.
- 0: max-farming for its own sake. Sept 20: explicit Milestone-2 go/forfeit decision (default forfeit on current arithmetic — don't publish the stack for nothing).

**GPU/rig hours (weekly):** first ~5h: eval-geometry stock-vs-v7 A/B (gates everything). Then standing eval-geometry A/B waves for animation-frames and grinder (~half-day each build + 3-wave A/Bs). SFT quota (4.5h/week): run-8 behavioral eval only AFTER the panel decontamination; corpus spend only after a clean positive AND a live draw.

**Engineering (rough weeks 1–3):** Week 1 — ledger reconstruction, plan rewrite + playbook commit, eval-geometry A/B, watchdog-heartbeat one-liner, upscale transcription probe. Week 2 — animation-frames sandbox lever + A/B; grinder retrigger + 9-game panel. Week 3 — win-path narration test; SFT panel fix + teacher census; integrate whatever won its A/B into the candidate final config. Weeks 4–11 — iterate on whichever of {animation, grinder+narration, SFT} shows a real offline effect, stacking levers; measure the finalist config's mean properly (n≥5–8 live draws) in October before freeze.

**Honest odds:** the identified levers sum to roughly +0.25–0.7 optimistic on a 0.93 base. Reaching mean ~1.5 requires most of them to hit; P(#1) is materially improved but not favored. Top-5 / a strong final is the realistic central case — which is exactly why the mean-not-max framing and the endgame mechanics matter: they're the difference between banking what you build and lotterying it.

## 5. Checked and solid — do not relitigate

- **Base distribution & farming ceiling:** n=8 mean 0.929, sd 0.195; E[max of 92 slots]=1.412±0.084; P[any≥1.8]≈0.0004 (re-verified in RESULTS file this session). Farming cannot win the final.
- **Final mechanics:** 2 selected subs, private scores locked at run time, no rerun, auto-select by public if unselected, earliest-sub tie-break, CC-BY/OSI open-source-or-removed, Milestone 2 terms — all verbatim in saved primary snapshots.
- **Draw identities:** 0.76=context arm, 0.78=depth pack, 0.80=v6 (`bcef400` ancestry re-verified this session). v7 has zero scored draws.
- **Gap decomposition:** 100% unlock-limited on public games; 8/9 never-unlocked games are goal-inference-blocked (tr87 perception + freeze confound). Unlock levers are the only track that closes the gap.
- **Harness A/B results stand:** WMR trio +4 levels (≈0.9σ — real but weak); graph/compact/playbook null-or-harmful; compaction 0/303. The 27B "receives but does not operationalize" injected heuristics.
- **Code facts:** frame[-1] discard (game.py:170, re-verified), grinder trigger dead-by-construction (duck_patches.py:1995), STALE_MINUTES=15 (scorecard.py:24, re-verified), UPSCALE=4 in the scored bundle (re-verified), the animation-scalar patch already ships, sandbox-global injection channel proven.
- **SFT contamination:** run-8 trained on vc33/sc25/lp85; vc33 is in the behavioral panel (HANDOFF-2026-07-26:68 + ab_adapter_eval_result.json).
- **Failed-search economics:** actions on never-completed levels cost exactly 0 score (2026-08-01 controlled measurement); replay-at-WIN verified live but fires only on full-game WIN.

**UNVERIFIED items relied on with flags:** rival sd≈0.2 assumption in the bar computation (sensitivity-checked, conclusion stable at 0.15–0.25); live gateway's actual stale-close threshold; sonpham frame-full efficacy and brevity-RL datum (external, no license — reimplement, don't copy); 27B uptake of sandbox animation queries and of vision at higher upscale (each has a designed cheap test above); eval-geometry token-starvation MAGNITUDE (the section-2 test exists to measure it). One prior claim to retire: "the 0.56 unlock gap matches the mean gap almost exactly" is a numerological coincidence — the audit's 0.56 was computed against the old 1.8 bar.

---

## POST-REVIEW CORRECTION (session verification, 2026-08-04)

The report's claim "0.80 = v6 / v7 has zero scored draws" (sections 2 and 5) is
REFUTED by direct evidence: pulling the kernel version bound to sub 55224522
(latest = v7, no pushes since) shows the notebook contains patch14 /
TAAF_ANTIFREEZE / TAAF_PLAYBOOK markers. The bcef400 ancestry inference was
misled (the bundle-internal git_status.txt tracks the upstream duck repo, not
ours). 0.80 IS a true v7 draw (n=1).

The meta-finding this error demonstrates is ACCEPTED and stands: submission →
bytes → patch-set identity is forensic, not ledgered. Ledger reconstruction is
week-1 work. All other verdicts accepted as written.
