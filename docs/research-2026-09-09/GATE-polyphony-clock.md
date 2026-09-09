# GATE — Polyphony (A3/A4 port candidate) — 2026-09-09

Repo: `github.com/Mininglamp-AI/polyphony-arc-3` (default branch `master`).
Gate run BEFORE any port work, same protocol that killed NOOA.

## 1. Licence — PASS

MIT, read from the LICENSE file directly. The GitHub API reports `NOASSERTION`,
the same detection quirk seen on NOOA. No licence risk.

## 2. Loop shape — PASS (and it is our Stage-1)

Observe -> Edit -> Plan -> Act. The agent grows a Python "Heuristic System"
(state / engine / planner / verifier) in a per-game workspace. Per the README:
a policy "is accepted only when it reproduces those transitions exactly", then
"the agent searches it for an action sequence, plays that sequence against the
live game, and checks the frames again."

That acceptance test is exactly our Stage-1 backtest. Explicitly built for
"deployable open-weight models"; default `--model Qwen/Qwen3.6-27B`. So the
paradigm is not speculative for us — we have already measured its core step on
our own brain (docs/research-2026-09-09/PREREG-stage1-own-transitions.md, 4/4 green).

## 3. Clock — PASS (this is NOT what kills it)

Unlike NOOA, time is a first-class configurable. `agent.py`: "Per-game wall-clock
deadline (epoch seconds). TIME is the primary stop authority under a bounded
wall-clock budget"; the old step budget and stuck rule were deliberately removed.
`run_swarm.py --per-game-deadline-s` defaults to 1800 s, inside our 7,920 s.
There is no fixed turn timeout and no unbounded-turn assumption to violate.

## 4. Compute — **FAIL**, by 3-10x

The binding constraint is decode tokens, not wall clock and not call count.

Their own competition operating point (README):
`--parallel-nums 5 --per-game-deadline-s 14400 --max-tool-calls-per-send 40`
against `vllm serve Qwen/Qwen3.6-27B --tensor-parallel-size 8 --max-model-len 262144`.
That is 4 h/game with 5 games in flight across 8 GPUs = ~6.4 GPU-hours per game.
Ours is 8.8 GPU-hours for all 110 games = 0.08 GPU-hours per game.

Hardware-independent version, using OUR measured numbers on OUR brain:

| | calls | completion tokens | tokens/call |
|---|---|---|---|
| Stage-1 green model, dc22L1 | 2 | 47,410 | 23,705 |
| Stage-1 green model, dc22L2 | 3 | 55,356 | 18,452 |
| Stage-1 green model, dc22L1 (b) | 9 | 118,221 | 13,136 |
| Stage-1 green model, dc22L2 (b) | 2 | 40,886 | 20,443 |
| **mean per green model** | 4 | **65,468** | **16,367** |
| **stock, per whole game** | 64.2 | **83,094** | **1,293** |

* A Stage-1 call is **12.7x fatter** than a stock call. Writing a Python
  transition model requires long generation; that cost is intrinsic.
* **ONE verified level model = 79% of an entire game's decode budget** (best
  case 49%, worst case 142%).
* Call COUNT is not the problem: 2-9 calls fits easily inside our ~52.
  Tokens per call is the problem.

Polyphony needs, per level: collect transitions, edit the policy, backtest,
re-edit until green, search the policy for a plan, execute, re-verify. The green
model alone eats the game. The rest of the loop, and every level after the
first, has nothing left to spend.

The incremental path does not rescue it: the 9-call run shows the incremental
rounds still cost 13,136 tokens each, ~10x stock.

## 5. Can we buy the budget back? No.

Cutting concurrency 28 -> 5 (their number) buys 5.6x per-game compute but covers
5/28 of the games in the same wall clock, and the score averages over all 110
games, so uncovered games score zero. The Kaggle session cap leaves at most
~1.35x more wall clock. Neither reallocation closes a 3-10x gap.

## VERDICT — DEAD on compute, at the current geometry.

Licence clean, loop shape right, clock configurable, core step already proven
green on Flash-Next. It fails only because one verified executable model costs
about one whole game's generation on a single GPU serving 28 games at once.

This is a different death from NOOA (which violated the clock contract outright)
and from A1/A2 (engaged-and-flat). Polyphony is not refuted as an idea. It is
priced out of our hardware. Record it as reopenable if the per-game compute
allocation ever changes by an order of magnitude.

NEXT: Track D (Oct-1 absorption prep), which the revised plan calls certain value
and which is untouched.
