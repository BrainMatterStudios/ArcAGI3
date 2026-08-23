# Search-theory import — working notes (2026-08-23)

## Measured this session (offline engine, ONLY_RESET_LEVELS=true)
- deepcopy(env): 0.8 ms, deterministic (frame-equal replay verified); raw step ~11.6k act/s
- tu93 FULL 9-level crack via snapshot BFS: 6.2 s (vs 85 s reset-replay in solve_floor)
- IW(1) sweep (120 s/game): L1+ on ar25, cd82, lf52, sp80, tu93(3), vc33 — seconds each;
  exhausts (width>1) everywhere else. Depths at-or-below human min (vc33 L1: 3 vs 7).
- nbfs (novelty-first, complete): ar25 L1+L2 (L2 depth 11 < human min 15) where BFS/IW fail.
- click coverage vs human winning clicks (within 2px of component centroid / within 5):
  ft09 .87/1.0 (cands 64-116), lf52 .88/.99, s5i5 .98/1.0, vc33 .97/1.0, sc25 .93/1.0,
  sb26 .82/.99, lp85 .78/.99, tn36 .63/.97, bp35 .68/.86, cd82 1.0/1.0;
  FAILURES: su15 .17/.27 (real lattice = 16x14 grid @4px, from source), r11l .47/.57,
  cn04 .09/.30 (clicks inside large comps, position matters).
- probe artifacts found: 16-click cap missed dc22 panel buttons + sc25 icon/slots + ft09 tiles.
- ft09 = lights-out constraint puzzle (commuting toggles): blind search dead regardless of
  coverage; needs model tier (July GF2-style solver exists).
- g50t: nbfs exhausts 1439 states (clone-recorder mechanic needs bespoke operator).
- cn04: 20k states in 41 s without L1 (state explosion, cell-position clicks needed).

## Live-lane cost model
- gateway ~130 act/s shared; grind budget 2700 s/run => ~350k actions.
- reset-replay BFS: testing 1 action at depth d costs d+1 actions.
- rollout methods (Go-Explore / Rollout-IW): amortize return cost over k-step rollout
  => ~(1 + d/k) actions per action tried, ~10x live throughput.

## Levels min-depth table (human winners' min per level) — see budgets_v2.json
full-crack candidates (all levels <=~26 min): cd82, sb26, ft09*, su15, lp85(-L8 50), tn36(-L7 55)
(*combinatorially dead despite shallow depth)

## LLM lane today (ours_38): 19 levels total over 25 games; 0-level games:
cd82 dc22 ft09 g50t ka59 m0r0 re86 sk48 sp80 tr87 wa30 (11 games).

## Head-to-head (120 s/game, 16-click cap, snapshot backend)
levels won: bfs 26 | iw1 8 | nbfs 29  (LLM duck-38 lane: 19)
nbfs >= bfs everywhere; iw1 solves fast but exhausts (width>1) — use novelty as ORDER not PRUNE.
nbfs full games: tu93 9/9 (4.2 s). nbfs notable: ls20 3, vc33 4, s5i5 2, m0r0 2, cd82 2, ar25 2, lf52 2.
Structural zeros at 16-cap: dc22/ft09/lp85/sc25 (cap artifact), su15/r11l (lattice/bg clicks),
cn04/re86 (20k state cap), tr87/bp35 (slow stepping), g50t (clone op), wa30 (model), sb26 (slow+cap), tn36 (combinatorial).

## Live-cost simulation rule
reset-replay cost ~= nodes * (depth/2 + 1) actions; gateway 130 act/s shared.
tu93 crack ~46k actions ~ 6 min. Rollout-style amortization ~8x cheaper.
