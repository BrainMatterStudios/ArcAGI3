# duck-throughput — duck-patched + stop over-reserving KV cache

Everything in `_duck_patched` (ACTION7 fix + animation metadata), plus **one serving change**.

## The problem

The bundled setup script starts vLLM with `--max-model-len 65536`. The agent can never use
that much context:

- `setup_commands.json` sets `LOCAL_ANALYZER_CONTEXT_WINDOW = 32768`
- `tool_agent.py:138` reads it into `_LOCAL_ANALYZER_CONTEXT_WINDOW`
- `tool_agent.py:972` caps the prompt at `window - reply_reserve - request_safety_margin`,
  then generates at most `reply_reserve` more

So prompt + completion is bounded by **32,768** while we reserve **65,536** of KV cache per
sequence. Exactly 2× waste, on the one resource that limits concurrency.

Why that matters: KV-cache capacity in tokens is fixed by the GPU, and the number of
concurrent sequences it supports is that capacity divided by `max-model-len`. A real duck run's
server log reports `Maximum concurrency for 65,536 tokens per request: 10.12x` while the solver
runs **28** games at once (`concurrency=28`, `benchmark_initial.pkl`). Generation throughput is
the binding constraint on total tokens delivered, and total tokens delivered is what buys depth
— which is the only thing the depth-weighted scorer pays for.

A survey of every public duck fork found **none** that tuned `max_model_len`, concurrency, or
per-game budget with a score to back it. This axis is untouched by the field.

## The change

Rewrite `VLLM_MAX_MODEL_LEN` inside the bundled setup script at notebook runtime:

```
[duck-throughput] max-model-len 65536 -> 40960 (analyzer window 32768 + 8192 headroom); KV reservation cut 38%
```

The target is **derived** (`ANALYZER_CONTEXT_WINDOW + 8192`), not hardcoded, so it stays
consistent if the analyzer window is retuned.

**Why 40960 and not 32768.** The true bound is 32,768, which would cut the reservation by 50%
instead of 38%. But the agent's token accounting is an *estimate* and vLLM's is exact; drift
between them returns HTTP 400 and kills the game. We lost two entire EWM gate runs to precisely
that failure mode. The last 12% of the win is not worth reintroducing it.

We rewrite the constant rather than hand-rolling the serve command, because inheriting the
bundle's serve is the one thing that has reliably worked (a hand-rolled serve is what killed
EWM gate v1).

## Fail-safe behaviour

Verified by direct execution of the helper:

| situation | behaviour |
|---|---|
| constants not found | logs a warning, serves **unchanged** |
| target not below current | logs, serves **unchanged** |
| more than one rewrite matched | logs a warning, serves **unchanged** |

It can degrade to the current duck; it cannot produce a broken serve command.

## Change 2 — adaptive per-game budget

The bundle ships `max_runtime_s_per_game=7920` with `concurrency=28`, which exactly fills a 9h
budget **if** the eval set is 110 games (110/28 = 4 waves × 7920 = 31,680s). Two things are
unresolved: the ARC-AGI-3 technical report says the competition set is **55** fully-private
environments (2 waves = 4.4h, leaving ~4.6h of GPU idle), and the total notebook cap has four
conflicting public figures.

Rather than bet on either number, the budget is derived at runtime from what the gateway
actually serves, right after `bm.games = _competition_games()`. The rule is **one-directional —
raise only, never lower**:

| games | waves | computed | effect |
|---|---|---|---|
| 25 | 1 | 28,620s | raise |
| 55 | 2 | 14,310s | raise — claims the idle half |
| 110 | 4 | 7,155s | **unchanged** |
| 140 | 5 | 5,724s | unchanged |

So a 110-game set is byte-for-byte the current behaviour and carries no regression risk; a
55-game set uses the hours we would otherwise idle. `_elapsed` subtracts real setup + vLLM serve
time, and the 0.9 factor leaves room for the solver to drain and the scorecard to close. Any
missing knob or exception leaves the budget untouched with a logged reason.

**Caveat:** there is still no stall detection (`solver.py:245-261`), so a longer per-game budget
also means a hopeless game holds its slot longer. Depth is the only thing that scores and most
games are not hopeless, so this should be net positive — but it is an argument for building the
stall detector next, not a reason to skip this.

## Not changed

Model, quantization, sampling parameters, `concurrency`, `max_runtime_s_per_game`, `n_passes`,
the game list, the submission path.

## Build and push

```
.venv/bin/python submission/_duck_patched/build_duck_patched.py       # prerequisite
.venv/bin/python submission/_duck_throughput/build_duck_throughput.py
kaggle kernels push -p submission/_duck_throughput --accelerator NvidiaRtxPro6000
```

## What is NOT claimed

The throughput gain is **unmeasured**. The 2× over-reservation is verified from source, and the
concurrency arithmetic follows from it, but the actual effect on tokens delivered and on score
can only be observed on the eval GPU. Ship this *after* `_duck_patched` has a score, so the two
changes have clean attribution.
