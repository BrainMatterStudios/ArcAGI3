# Fable adversarial review — amendments to the depth reordering (2026-08-02)

VERDICT: direction stands (depth over efficiency, today). Three builds as specified:
detector likely HARMFUL, allocator likely inert, defect pack mixed. Amendments:

1. STALL DETECTOR AS SPECIFIED IS DEAD. The 12x separation is a cross-model
   artifact: completers are 27/29 Kimi-K3 teacher episodes; K3-only separation
   collapses to 0.89 vs 0.94. 8/29 completions happen AFTER action 30 -> the
   horizon-30 rule culls ~28% of L1 completions (~-0.3). Rebuild ONLY from duck rig
   episodes, horizon >= ~100, held-out FP validation.
2. "EFFICIENCY WORTH ZERO" IS NOT A LAW. All cap-bound rows are L1-only. The moment
   depth arrives, frontier levels are won slow and efficiency IS the margin. The
   +0.673 headline assumes cap-bound depth-6 (superhuman across 6 levels);
   realistic frontier efficiency gives ~+0.23/game. Also the 66-level 0.78x stat is
   teacher-contaminated; duck-specific evidence is 10 L1s.
3. G0: DEFER, NOT CANCEL. sd CI from n=8 is [0.129, 0.397] -> E[best of 90] spans
   1.25-1.91; the 1.41<1.50 mootness argument was a point estimate posing as a
   bound. Endgame selection of 2 finals needs rho eventually.
4. D1 REPAIRS REQUIRED BEFORE GPU: (a) game_over should ALSO clear current_plan
   (it references pre-death board state) while keeping world/goal/action models;
   (b) RESET un-strip needs a rate limiter (advertise only if >=N actions since
   level start; never within k of a completion) — harness already auto-resets on
   death, and a perseverating model + always-legal escape = reset-spam livelock;
   (c) ACTION7 guidance fine, but whether undo is a scored action is UNVERIFIED.
5. OBJECTIVE REFRAME: L1->L2 conversion (3x multiplier, observed mass) is the
   target; depth 6 is upside, not objective. No run of OUR model has ever exceeded
   depth 1. Leader's true mean ~1.5; real mean-gap ~0.2 = 8-10 L1->L2 conversions.
6. The blinded researcher's Go-Explore-on-stalled-levels survives (score-free on
   failed levels) but inherits amendment 1's detector requirements.
