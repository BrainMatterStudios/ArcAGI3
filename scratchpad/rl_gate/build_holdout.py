"""build_holdout.py — materialize the pinned arc-interactive behavioral holdout.

Copies selected arc-interactive game packages into a frozen holdout dir and
fixes each metadata.json so `baseline_actions` has exactly `win_levels`
entries (the taaf harness asserts len == number_of_levels; community metadata
often carries a single game-level value). Padding repeats the last value —
fine for RELATIVE A/B comparison, which is this holdout's only use. Absolute
RHAE numbers computed against these padded baselines are NOT meaningful.

Each copied game is then load+reset smoke-tested through OUR pinned engine
(arc_agi offline Arcade — the same path run_rollout.py uses).

Run:  .venv/bin/python scratchpad/rl_gate/build_holdout.py \
          --games ul01,tt01 [--src <upstream env dir>] [--dest <holdout dir>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DEFAULT = REPO / "scratchpad/arc_interactive_upstream/environment_files"
DEST_DEFAULT = REPO / "scratchpad/holdout_arcint/environment_files"

# Official ARC-AGI-3 games bundled inside arc-interactive; NEVER holdout
# material (ft09/vc33 are in the run-8 training corpus; ls20 is official
# public dev; sb26-family excluded defensively).
FORBIDDEN = {"ft09", "ls20", "vc33", "tu93", "sb26"}


def probe_win_levels(env_dir: Path, stem: str) -> int:
    import arc_agi
    from arc_agi.base import OperationMode

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(env_dir))
    arcade.get_environments()
    game = arcade.make(stem)
    frame = game.reset()
    if frame is None:
        raise RuntimeError(f"{stem}: reset returned None")
    if frame.state.name not in ("NOT_FINISHED", "NOT_PLAYED"):
        raise RuntimeError(f"{stem}: unexpected state after reset: {frame.state.name}")
    return int(frame.win_levels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", required=True, help="comma-separated stems")
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    args = ap.parse_args()

    stems = [s.strip() for s in args.games.split(",") if s.strip()]
    bad = sorted(set(stems) & FORBIDDEN)
    if bad:
        sys.exit(f"refusing official/corpus games in holdout: {bad}")

    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for stem in stems:
        src_game = args.src / stem
        if not src_game.is_dir():
            sys.exit(f"{stem}: not found under {args.src}")
        dest_game = args.dest / stem
        if dest_game.exists():
            shutil.rmtree(dest_game)
        shutil.copytree(src_game, dest_game)

        # fix every version's metadata.json in the copy
        n_levels = probe_win_levels(args.src, stem)
        for meta_path in dest_game.glob("*/metadata.json"):
            meta = json.loads(meta_path.read_text())
            base = list(meta.get("baseline_actions") or [])
            if not base:
                base = [10]
            fixed = (base + [base[-1]] * n_levels)[:n_levels]
            meta["baseline_actions"] = fixed
            meta_path.write_text(json.dumps(meta, indent=2))

        # smoke: the FIXED copy must load+reset through our engine
        probed = probe_win_levels(args.dest, stem)
        assert probed == n_levels, f"{stem}: copy probe mismatch {probed} != {n_levels}"
        manifest[stem] = {"win_levels": n_levels,
                          "baseline_actions_padded": fixed}
        print(f"[holdout] {stem}: {n_levels} levels, baselines {fixed}")

    out = args.dest.parent / "holdout_manifest.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(manifest)
    out.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"[holdout] {len(stems)} games ready under {args.dest}")
    print(f"[holdout] manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
