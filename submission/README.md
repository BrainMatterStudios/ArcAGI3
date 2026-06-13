# Kaggle submission

The agent is submitted via a one-click Kaggle notebook that runs our agent through the
official ARC-AGI-3-Agents framework against the gateway-served private games.

## Files

- `my_agent.py` — thin `MyAgent(Agent)` adapter around `arcagi3.policy.HybridPolicy`
  (reactive `choose_action`/`is_done`). Verified end-to-end by `tests/test_submission.py`.
- `build_notebook.py` — regenerates `notebook.ipynb` from `my_agent.py` + the harness.
- `notebook.ipynb` — the submission notebook (generated; do not hand-edit).

## How to submit

1. **Package the agent as a Kaggle dataset** named `arcagi3-agent`:
   upload this repo's `src/` directory (so the package is at `arcagi3/...`).
   ```bash
   # one-time, requires kaggle CLI + token
   kaggle datasets create -p src   # or update: kaggle datasets version -p src -m "update"
   ```
2. **Create the competition notebook** for *ARC Prize 2026 - ARC-AGI-3*:
   - Attach the competition data and the `arcagi3-agent` dataset.
   - Paste `notebook.ipynb` (or import it).
3. **Run / Submit.** At evaluation rerun (`KAGGLE_IS_COMPETITION_RERUN`), the notebook
   waits for the gateway, runs `python main.py --agent myagent`, and the gateway records
   the scorecard into the submission.

## How it runs at eval

- Install `arc-agi`/`arcengine` offline from the competition wheels (no internet).
- `OPERATION_MODE=online`, pointed at `http://gateway:8001` (games come from the gateway,
  not local files; `ENVIRONMENTS_DIR` empty).
- `my_agent.py` adds the `arcagi3-agent` dataset path to `sys.path` and imports the agent.
- Submission columns: `row_id, game_id, end_of_game, score` (score = levels completed).

## Notes

- GPU T4×2 available; the current agent is CPU/algorithmic (no model weights needed).
- Internal soft time budget: stop ~5 min before 8h; Kaggle hard limit is 12h.
