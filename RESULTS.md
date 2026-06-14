# Real-game results (ARC-AGI-3 public games)

Measured at local speed (NORMAL mode: games downloaded and run locally, ~700+ actions/sec)
against the real public games via the ARC-AGI-3 API. Levels = levels completed within the
stated per-game action budget. At the actual Kaggle eval the budget is far larger (12h
notebook, games run in parallel threads at local gateway speed), so these dev numbers are
a conservative lower bound.

## Progress this R&D session

| milestone | result (real public games) |
|---|---|
| initial agent (toy-game heuristics) | **1 level** total across 25 games @ 4000 actions |
| + distractor-masking fix | multiple avatar games score: tu93 3/9, su15 1/9, m0r0 1/6, ls20 lvl 1 |
| + 5-tier click salience & coverage | click games score: vc33 1/7, lp85 1/8, r11l 1/6 |

The two fixes that mattered most, both grounded in the deep-research findings:

1. **Distractor-masking fix** — previously *any* color whose cells changed during probing
   was masked from the state key, which masked structural colors (maze walls) the avatar
   moves over. That made the agent blind to its own progress (doors opening) and collapsed
   the state graph (~147 states; 136 actions per new state). Now a color is only treated as
   an animated distractor if it rigidly translates with a constant, action-independent delta
   (a real counter/animation). Result on ls20: 147 → 3820 states, ~2 actions/new-state, and
   level 1 becomes solvable.

2. **5-tier click salience** (per arXiv 2512.24156) — reduce the 4096-cell click space to
   centroid clicks of connected-component segments, stratified into 5 tiers by size +
   color-rarity (wide flat status bars last), with full coverage of object-dense frames.

## Why coverage is the lever

Deep research (multi-source, verified) confirms that for ARC-AGI-3, *exploration coverage*
— not learned-policy quality — is the dominant lever within budget; frontier LLM agents
score <1% and underperform random. Our agent is a training-free graph explorer aligned
with the published SOTA (object segmentation, masked-frame state hashing, frontier search)
plus extras: a motion model (controllable-avatar detection + per-action displacement for
cheap coordinate navigation) and action-correlated animated-distractor detection.

## Reproduce

```bash
echo 'ARC_API_KEY=...' > .env            # from https://three.arcprize.org
uv run python -m arcagi3.validate_online --budget 30000        # all games
PYTHONPATH=src uv run python scripts/run_one.py ls20 60000      # single game, live progress
PYTHONPATH=src uv run python scripts/survey_games.py            # mechanic taxonomy
```
