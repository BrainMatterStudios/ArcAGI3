# ARC-AGI-3 Agent — ARC Prize 2026

A general, training-free interactive agent for the
[ARC Prize 2026 / ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. Dropped into an unseen 64×64 grid game with no instructions, it explores,
infers the goal from the levels-completed reward, builds a world model, and plans to
clear levels — and must **generalize to unseen private games**.

See [PLAN.md](PLAN.md) for strategy and roadmap.

## Setup

```bash
uv sync            # Python 3.12, arc-agi 0.9.1 + arcengine 0.9.3
```

## Run the agent on the local dev games (fully offline)

```bash
uv run python -m arcagi3.runner                       # all local games
uv run python -m arcagi3.runner --game navg --budget 4000
```

## Test

```bash
uv run pytest
```

## Layout

- `src/arcagi3/perception.py` — frame → object-centric state (connected components,
  status-bar/counter masking, salient click targets, state hashing).
- `src/arcagi3/world_model.py` — directed state-transition graph (frontier search,
  shortest-path replay, reward tracking).
- `src/arcagi3/agent.py` — `ExplorerAgent`: graph-based exploration + exploitation.
- `src/arcagi3/runner.py` — offline runner / eval harness.
- `src/arcagi3/games/` — diverse local dev games (navigation, click, push) for
  developing & regression-testing generalization.

## License

[MIT-0](LICENSE) (no-attribution) — required for ARC Prize eligibility.
