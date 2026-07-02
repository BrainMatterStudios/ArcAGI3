# Wall-time attack on deep multi-level (2026-07-02)

The depth-weighted RHAE (see efficiency_lever_findings.md) means the ONLY thing that materially scores is
efficiently completing DEEP levels — which is wall-time-bound. This investigates cracking that wall-time.

## Bottleneck: prefix-replay
`reset()` ALWAYS returns to the L0 board, deterministically (verified, reset_semantics.py) — there is NO
in-place level restart. So to search level k, every node must replay the 0..k-1 prefix from reset. Cost ~
prefix_len x nodes; both grow with depth -> super-linear blowup (double-reset chain: 1.6s->92s/level, dies at L6).

## Offline breakthrough: deepcopy snapshot/restore
`copy.deepcopy(env)` works on the offline Arcade env: restore+apply == reset+prefix+apply (verified,
repeatable), and is ~constant-cost vs O(prefix_len) replay (4.5x faster at prefix_len=18, more at depth).
Snapshot-chain (snapshot_chain.py) starts each level's search from the PREVIOUS level's END snapshot -> the
prefix is NEVER replayed -> per-node cost independent of depth.

RESULT (tu93, move-only): solved the FULL game L1->L9 in 57s, FRESH-VERIFIED (levels_completed=9).
Per-level actions [18,10,19,17,29,30,14,21,29] vs baseline [19,16,34,42,123,80,14,23,111] -> official
depth-weighted score ~100/100 (deployed portfolio = 0.31). The score CEILING is algorithmically reachable.

## *** DEPLOYMENT CAVEAT (decisive): snapshot is OFFLINE-ONLY ***
The Kaggle eval agent uses the ARC-AGI-3-Agents Agent interface: take_action -> do_action_request(action) =
REST over ROOT_URL; the agent only receives FrameData and can reset()/act. There is NO in-process game object,
so deepcopy is impossible at eval. Snapshot is an OFFLINE ORACLE. At eval, multi-level is still reset()+replay
= wall-time-prohibitive. The original wall STANDS for deployment.

## Eval-compatible path (the real remaining lever)
Since prefix-replay can't be eliminated over REST, the only eval lever is REDUCING SEARCH NODES:
- gradient-bearing DISTANCE heuristic (agent->goal) via role-typed positions (role_typing.py) — boolean goal
  predicates gave NO gradient (closed_loop_results.md), so distance is the candidate.
- macro-actions / amortized (learned) policy to need fewer search actions per level.
- within-level search must be BFS-robust: novelty search FAILED lf52 L0 that plain BFS solved in 8 actions.

## Honest status
- Offline: multi-level SOLVED (snapshot oracle) — big, verified, but not shippable.
- Eval: wall-time wall stands; needs node-reduction (heuristic/amortization), not snapshotting.
Assets: reset_semantics.py, snapshot_test.py, snapshot_chain.py, ab_efficiency.py, l0_shortest.py, efficiency_audit.py.
