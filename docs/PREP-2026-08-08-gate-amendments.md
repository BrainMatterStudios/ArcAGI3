# Prep note 2026-08-08 — staged gate amendments (nothing pushed; all awaiting go)

Staged overnight per Ahmed's 2026-08-07 approval to *prepare* builds. Every item
below stops at the approval line: no `kaggle datasets version`, no
`kaggle kernels push` has been run.

## 1. Track B (35B capability gate) — serve-command fix STAGED

`scratchpad/taaf_scored_ref/setup_commands.json` (the bundle source) now includes
`--max-num-seqs 512` in the vLLM launch — the known Qwen3.6-35B-A3B-on-Blackwell
failure (default 1024 exceeds the model's cache blocks; external precedent required
512). Embedded PYSETUP re-validated (`ast.parse` OK); model refs unchanged
(cmechevalier 35B, TP=1, max-len 65536).

**To ship (after go):**
1. `kaggle datasets version -p <bundle build dir> -m "v3: --max-num-seqs 512"`
   for `ahmedmobasher86/arcagi3-bundle-35b` (rebuild the bundle dir from
   taaf_scored_ref the same way v2 was built — v2's builder session should be
   reproduced, NOT hand-assembled; verify with a fresh `kaggle datasets files` pull
   that setup_commands.json matches the local file byte-for-byte).
2. Re-push duck-sparse (v8) so the kernel binds bundle v3; re-verify remote==local
   and commit-log bundle resolution as done on 08-07.
3. The GATE kernel (not the live sub) runs first: same serve path + the
   protocol-compliance metric below + panel dc22/m0r0/sk48/tr87 + controls
   ft09/su15 per the frozen Track B design.

## 2. Track B — protocol-compliance metric (spec, to add to the gate kernel)

Count, per game and in total: analyzer turns attempted; turns yielding a
harness-accepted action on first parse; turns recovered by retry; turns dropped.
Print `PROTOCOL: game=<id> ok=<n>/<N> retried=<n> dropped=<n>` per game and a
final `PROTOCOL_TOTAL ok_rate=<x>`. GO bar (frozen design): ok_rate >= 0.95.
Purpose: a sonpham-style silent collapse (35B emitting unparseable actions) must
be measured, not inferred from a 0.000 score.

## 3. Track A closure — candidate amendments under evaluation tonight

- 6 new HUD regions are confirmed clocks (ar25 col63 · ft09 row63 · g50t
  row63/32-63 · sc25 cols62-63 · su15 row63 · tn36 row1). The LIVE mask is the
  dynamic `HudMaskTracker` (static per-game regions cannot transfer to hidden
  games), so the fix path is: fixture tests driving the tracker over offline-engine
  frames for those 6 games (agent building `test_hud_tracker_fixtures.py` now).
  If the tracker already converges: no code change, fixtures become regression
  tests. If it misses (columns/partial rows suspected): amend the tracker, prove
  it on fixtures, then it joins the Track A candidate arm.
- sonpham levers (no-impact detection, 60s-yield tempo): NOT built tonight;
  implement only with unit tests before any wave, else they wait for round 2.

## 4. Offline results feeding the build queue (all 2026-08-07, zero live actions)

contingency classifier GO 14/14 · A-not-B brake GO (hard brake safe on masked
games) · undo-residue NO-GO (peek() = 4-game tactic only) · HUD counters NO-GO
0/25 · human replays secured (scratchpad/human_replays/, 340 traces, all 25 games)
· archetype classifier accuracy test RUNNING (local, no GPU).

## 5. Approval line (explicit)

Awaiting Ahmed's go for, in order: (a) bundle v3 upload + duck-sparse re-push,
(b) Track A closure kernel pair push (with whatever fixture-driven mask amendment
survives), (c) Track B gate kernel push. GPU spend only after (a)-(c) approvals;
the 17.5h first-week ceiling from the campaign plan applies.
