# arc-3 — games catalog + internal run evaluation

Two things live in this repo, and they share one site (**https://arc3.sonpham.net**):

1. **Games** — a browsable, playable catalog of ~300 ARC-AGI-3 games: the 25 official
   ARC Prize Foundation games, our in-house custom games, and the Red Blue Pill community
   catalog. Public, no login, runs entirely in the browser.
2. **Internal runs** — the evaluation side: every benchmark run we've done of the
   ARC-AGI-3 duck harness, with a scoreboard, a per-turn run inspector, resource usage,
   and score-distribution ("signal") runs. Behind sign-in.

Everything under `docs/` is that site; everything else is the machinery that produces it —
the harness variants, the GCP launch kit, and the raw run logs.

---

## 1. Games

`docs/index.html` — the public half of the site.

**294 games in the catalog**, from three sources that all share the same
`environment_files/<code>/<version>/` layout and the same `ARCBaseGame` / `arcengine`
model:

| Source | Count | Origin |
|---|---:|---|
| `official` | 25 | the public ARC-AGI-3 games (`environment_files/` in this repo) |
| `custom` | 17 | built in-house (sibling repo `arc-agi-3`) |
| `redbluepill` | 252 | [theredbluepill/arc-interactive](https://github.com/theredbluepill/arc-interactive) |

Game *codes* collide across sources (`cr01`, `ft09`, `ls20`, `pt01`, `vc33` exist in more
than one catalog as entirely different games), so everything keys on the full `game_id`
with its version suffix (`ft09-0d8bbf25` vs `ft09-9ab2447a`) — never the bare code.

### How play works

Click a game and it runs **the real Python game** in your browser: Pyodide (WASM CPython +
numpy) in a Web Worker, with `arcengine`'s wheel pulled straight from PyPI and unzipped into
site-packages. The only thing fetched from us is the game's `.py` source text. Nothing is
sent to a server, nothing is recorded, there's no login and no leaderboard.

- Controls: WASD/arrows to move, `r` reset, `z` = ACTION5, `x`/`c` = ACTION7, click = CLICK.
- Undo, per-level jump strip, FPS control, live status/score.
- **Tile render modes** (`docs/static/games/arc_tiles.py`): `solid` is the engine's stock
  nearest-neighbour upscale; `tiles` gives every palette colour a fixed deterministic motif
  so a cell reads as an object with a texture rather than a flat blob; `random` reshuffles
  both motif and colour assignment per episode from a seed — behaviour untouched, appearance
  not, so an agent can't carry "blue == wall" across episodes. Same board, same rules.

The 16-colour board palette is deliberately duplicated between the thumbnail generator
(`scripts/build_games_manifest.py`) and the play page (`docs/static/js/games-play.js`), so
in-browser play renders pixel-identical to the static thumbnails and to recorded runs.

### Rebuilding the catalog

```bash
python scripts/build_games_manifest.py    # manifest.json + per-game src/ + thumbnails
```

Writes `docs/static/games/manifest.json`, `docs/static/games/src/<game_id>/<file>.py`, and
`docs/static/img/games/<game_id>.png`. Needs `arcengine` installed (for thumbnails and
tile-scale detection). Six titles are currently parked in `TEMPORARILY_REMOVED_CODES` —
their source and thumbnails stay on disk; delete the entry to bring one back.

---

## 2. Internal runs

`docs/internal.html` and friends — sign-in gated on the live site.

| Page | What it is |
|---|---|
| `internal.html` | **Scoreboard**: one row per run, one column per game, cell = that run's score |
| `viewer.html` | **Run inspector**: scrub every board state of a game and read the agent's full decision trace per turn |
| `harness.html` | Per-run harness facts (server, weights, env knobs) + real in-code tool-call counts |
| `signals.html` | **Signal runs**: one game played N times, as a box/whisker score distribution |
| `usage.html` | Per-run CPU / GPU / RAM / storage over the life of the run |

Static data from finished runs only — no live streaming. 37 runs are published today.

### The metric, and why we don't trust single runs

Score is the ARC-AGI-3 score (level depth is weighted, so depth beats efficiency).
Two rules, both learned the hard way:

- **Score on ex-`ft09`, never raw all-25.** One game (`ft09`) swings the 25-game average by
  ±1.0 on its own. Its own signal run — 25 passes of the same config — ranges 0.0 to 47.6
  (median 10.8, σ 11.2).
- **The 25-game mean is noise-dominated.** Its ~95% range on a fixed config is roughly
  0.45–2.67, so a single-run A/B cannot resolve a harness change. Replicate anything
  promising 2–3×.

### Where things stand

Reference points: Tufa's public-set score with this harness is **1.6002**; their semi-private
milestone score was 1.21. Our pristine reproduction of their stack (`tufa-exact-rung0`) scores
0.679, and that run is the baseline every other run's knobs are diffed against.

Best validated configs, by 2-pass mean ex-`ft09`:

| Config | ex-`ft09` | Note |
|---|---:|---|
| `ffa7gn` — frame-full + ACTION7 + animation + goal-guidance + **no-impact band** | **1.62** | current best; 21 levels |
| `ffa7gnh` — + HUD code-model | 1.60 | flat vs `ffa7gn`; the code model didn't add |
| `frame-full` alone | 1.44 | env-toggle `ARC3_FRAME_MODE=full` |
| `baseline-v12` (frozen reference) | 1.21 | 2 runs: 1.224, 1.188 |
| `ffa7g` — same stack, **no** no-impact band | 1.05 | the ablation control |
| `ffa7gnsg` — + state graph | ~1.11 | **regression**, defaulted OFF |

What the ablations actually said:

- **No-impact detection is the one clear win** (+55% ex-`ft09`, 21 vs 15 levels at equal
  action budget). It stops an explore action whose only board change is the game's
  deterministic HUD/moves band, killing wasted wall-presses. Both runs hit the ~132-min time
  cap, so the savings buy *progress*, not a lower action count.
- **The state graph lost three times running.** The model used its tools heavily but the
  plan-rejection block fired zero times, so we paid the query-turn cost (644 vs 366 tok/action)
  for nothing → fewer actions → fewer levels. A lean rerun that surfaces `untried_here` for
  free still lost on a 7-game A/B (1.533 on vs 2.061 off). Now opt-in via `ARC3_STATE_GRAPH=on`.
- **Model swaps all failed.** 35B-A3B MoE: 0.000 across 25 games. GLM-4.6V (SOTA visual
  grounding): 0.000. Gemma-4-31B (AIME 89.2): 0.156. A brevity-RL fine-tune cut thinking
  tokens 46% but the savings didn't convert to actions (1,710 vs 3,073) and depth collapsed.
  The 27B dense remains the harness's brain.
- **Serving matters, but less than tempo.** The vrfai compressed-tensors quant hits a
  pathological kernel path on vLLM 0.25 (3.4× slower than 0.19; ngram spec decode amplifies it
  to unusable), while spec decode on official weights is statistically tied with the pristine
  stack. Our early agent-side modifications (required ledger, outline renders, 900s yield) cost
  ~2.2× on identical serving — the tempo regime (60s yield, act-look-act) dominates everything
  else at this model scale.

### The custom-games pass

`20260726_054336_v12-ffa7gnsg-customgames17` is the first harness run against the **17 in-house
custom games** instead of the official 25, using the validated-best config. Mean 7.89 over 17
games, 24 levels, all completed without a crash. One outright win: `ps01` (Pouring Water Son, a
*live* real-time physics game) cleared its only level — the first evidence this turn-based
harness can handle continuous game semantics at all. Best partials: `cr01` (Crumbling Route)
9/10 levels, `sn02` (Sneeze) 6/7.

These numbers are **not comparable** to any all-25 or ex-`ft09` figure above: different games,
different level counts, different baselines (several needed `baseline_actions` patched to null
first — see `scripts/build_gcp_customgames_bundle.py`).

### Publishing a run

```bash
scripts/publish_run.sh <gcs-run-id> <log-dir-name>
```

Pulls logs from GCS → exports scoreboard + viewer JSON → commits/pushes → syncs the new data
into the Railway volume → verifies it's live. Requires a `HARNESS` entry for the run in
`scripts/export_runs_index.py` (that table is the single source of truth for server/weights/env
facts, which live on the VM, not in the run artifacts) and `$ARC3_SITE_DIR` pointing at the
railway-linked site checkout.

Supporting exporters: `export_viewer_data.py` (per-turn frames), `export_signal_runs.py`,
`export_usage.py`, `export_tool_calls.py`, `export_game_thumbs.py`.

---

## Harness variants

The rule that keeps `main` from drifting away from a known-good baseline:
**the baseline is an immutable artifact; experiments copy from it and never edit it in place.**

- `harnesses/baseline-v12/` is **frozen** — the exact source the `bundle-v12` GCS artifact runs
  (which is not any clean git commit, so this vendored copy is the only faithful record).
- A variant = copy of `baseline-v12` + its own patch → its own new-named bundle. Never mutate
  the baseline, never overwrite a shared GCS bundle.
- Small/additive and sharing the agent loop → an env-toggle stored as `patch/` + `MANIFEST.md`.
  Fundamentally different code (a two-agent world-model harness) → its own folder.
- Every folder carries a `MANIFEST.md`: what it derives from, the diff, env config, and its
  validated ex-`ft09` score(s).

See `harnesses/README.md`.

## Layout

```
docs/            the site — Games (public) + Internal runs (gated), plus exported run data
  static/games/  game manifest, per-game .py source, the Pyodide engine, tile shim
  data/          per-run exported JSON (scoreboard, viewer frames, usage, signals)
harnesses/       frozen baseline + one folder per variant, each with a MANIFEST
ARC3-Inference/  the duck harness itself (tool-using solver over TAAF); distill/ holds the
                 Phase-1 rejection-sampling SFT extractor
tufa-arc-agi-framework/, vendor-taaf-grafts/   upstream framework + our grafts
gcp/             spot-safe launch kit: restartable runs, GCS log sync, crash-loop guards,
                 one startup script per harness variant
logs/            complete artifacts of finished runs (transcripts, prompts, events, benchmark.json)
scripts/         catalog builder, run exporters, publish pipeline
kaggle/          the exact upstream notebook + its launch metadata
environment_files/  the 25 official games
```

Big raw request logs (`*_requests.jsonl`, multi-GB for thinking runs) live in
`gs://cellens-ai-artifacts/arc3-duck/` rather than git; a424-run request logs are included
gzipped.

## Provenance

The harness is a working fork of [Tufa Labs' ARC-AGI-3 Duck Harness](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner)
(June 30 milestone winner) by Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit,
Michal Tesnar and Stefano Viel, MIT-licensed. Commit `a2dddac` is pristine upstream; every
divergence since is one reviewed commit. Competition environment files are not redistributed
here.
