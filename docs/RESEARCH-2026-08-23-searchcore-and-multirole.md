# RESEARCH 2026-08-23 — search-core suite (MEASURED) + multi-role design

## A. Search theory sweep — measured tonight on the offline engine, all 25 fixtures

HEADLINE: the screening suite (120-240s/game budgets) already reached
**36 levels across 16/25 games + one full crack** — vs the LLM lane's 19
levels total. The 3-5/25 full-crack falsifier target is credible (cd82, sb26
rated high; r11l/su15/ls20/vc33 medium+). Raw results + working skeletons:
wave scratchpad searchtheory/ (probe.py, gex.py, results/*.json).

New engine facts: copy.deepcopy(env) is deterministic, 0.8ms → snapshot
search 11.6k act/s, 14x faster than reset-replay (tu93 full crack 6.2s vs
85s). Two-backend core required (snapshot offline / Rollout-IW-style
reset-replay live at ~130 act/s, one reset per ROLLOUT not per action =
8-10x the current grinder's efficiency).

Algorithm verdicts (all empirical): nbfs (BFWS-lite novelty-preferred,
complete) 29 levels > BFS 26 >> strict IW(1) 8 (ordering right, pruning
wrong); Go-Explore = the scheduler (archive + 1/sqrt visits + momentum
rollouts; exact masked hash as primary cell, coarse component-multiset tier
only past ~10k cells; solved cn04-L1 where nbfs explodes); run-length macros
(MUST emit intermediate states — completeness bug found empirically) +
composite ignition probes for inert roots (sc25 proven to need them);
4-tier click generator (centroids uncapped — the 16-cap alone cost dc22 3
levels; stride-interior; autocorrelation lattice — unlocked su15+r11l;
learned dead-click pruning); portfolio = successive-halving race on novel
states/100 actions + frame-0 archetype dispatch. REJECTED with reasons:
UCT/MCTS (no reward/stochasticity), bidirectional (no goal states),
SATzilla (no training distribution), IW(1)-strict, Atari downscaled cells.
Specialist tier (exists in-tree, port behind frame-only detectors): ft09
GF2 lights-out, tn36 pattern-goal, sc25 glyph, wa30 grab-drag A* — the
cheapest full-crack inventory (+2-6 games).

Build plan v1 (7-9 days CPU, falsifier-gated, ~950 LOC on graft_explorer
seams): core 2-backend -> nbfs + click tiers -> macros/ignition ->
Go-Explore scheduler -> portfolio racer -> specialists -> live graft.
FALSIFIER (offline, 25 fixtures): >=3 full cracks at <=45min/game generic
config; >=35 total levels (tonight's 36 is the floor); live-transfer gate:
replay-cost arithmetic fits the grind envelope (tu93 ~6min - fits).
Caveats: screening budgets, single seeds on 3 games, specialists dev-tuned.

## B. Multi-role collaboration — designed, literature-grounded, falsifier staged

BUILD: (1) zero-decode CONTRADICTION LEDGER (expect-queue events -> stated
facts in the world model); (2) BANKRUPTCY JUDGE — fresh-context grounded
rebuild call (cited contradictions + 3 structurally different hypotheses),
~1k decode/firing, 1-3/session vs 20-45k-decode spirals (sb26: one wrong
model held verbatim 15 turns). REJECTED: pre-batch LLM verifier (self-
critique degrades below GPT-4 scale — Huang ICLR24, CriticBench; the
environment already verifies via expect-queue) and debate (no win at
matched compute — Smit ICML24). Law: judge calls must be grounded in
observed facts + fresh context + concrete comparison framing. Falsifier
staged: trigger tuning partially run (naive-15 too hot, archetype windows
required); judge prompt-pack + answer keys buildable now; serve replay
costs cents. Corpus durably pinned: scratchpad/multirole_corpus/ (99MB).

## C. Status of the ladder
xd flew 1.50 (middle band; digest ~0 live). 08-24 armed: v12+effort_medium
ladder read #1 (attested svid 344335040 — builder's placeholder caught and
fixed at verification), mock rc=0, runner pid 10988.
