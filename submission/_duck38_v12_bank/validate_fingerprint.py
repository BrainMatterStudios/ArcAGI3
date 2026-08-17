#!/usr/bin/env python3
"""Offline validation of the cross-clone transfer FINGERPRINT premise.

The transfer lever (thtennant fork, taaf_grafts.family_store) keys clone
families by blake2b-128 over (initial grid, number_of_levels, sorted
available action ids). Its entire value rests on three empirical claims,
none of which we accept untested (doctrine v2):

  T1  UNIQUENESS  — the 25 public games produce 25 distinct fingerprints
                    (a collision would let a solution replay into the WRONG
                    game — the one genuinely dangerous failure mode).
  T2  STABILITY   — reopening the same game N times yields the SAME
                    fingerprint (random initial frames would break keying;
                    an unstable game must fingerprint to None-equivalent,
                    i.e. simply never hit, which is safe).
  T3  CLONE HIT   — anonymized clones (built with the SAME mechanism the
                    competition simulator uses: cloned EnvironmentInfo with
                    scrubbed IDs) fingerprint identically to their source.

Run:  .venv/bin/python submission/_duck38_v12_bank/validate_fingerprint.py
"""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ONLY_RESET_LEVELS"] = "true"

GAMES = [
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
]


def fingerprint(initial_grid, number_of_levels: int, action_ids) -> str:
    """Mirror of the fork's family_store.fingerprint packing: deterministic
    bytes over (grid dims + cells, level count, sorted action ids)."""
    h = hashlib.blake2b(digest_size=16)
    rows = list(initial_grid)
    h.update(struct.pack("<II", len(rows), len(rows[0]) if rows else 0))
    for row in rows:
        h.update(bytes(int(c) & 0xFF for c in row))
    h.update(struct.pack("<I", int(number_of_levels)))
    for aid in sorted(int(a) for a in action_ids):
        h.update(struct.pack("<I", aid))
    return h.hexdigest()


def open_and_fingerprint(arcade, game_id: str):
    import arcengine

    env = arcade.make(game_id)
    assert env is not None, game_id
    resp = env.reset()
    assert resp is not None and resp.frame, game_id
    data = resp.frame[-1]
    rows = data.tolist() if hasattr(data, "tolist") else data
    grid = tuple(tuple(int(c) for c in row) for row in rows)
    info = env.environment_info
    levels = getattr(info, "number_of_levels", None)
    if levels in (None, 0):
        levels = int(resp.win_levels or 0)
    actions = [a.value if hasattr(a, "value") else int(a) for a in (resp.available_actions or [])]
    return fingerprint(grid, levels, actions), grid


def main() -> None:
    import arc_agi

    def fresh_arcade():
        return arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )

    # T1 uniqueness + first-open record
    first: dict[str, str] = {}
    arcade = fresh_arcade()
    for g in GAMES:
        fp, _ = open_and_fingerprint(arcade, g)
        first[g] = fp
    values = list(first.values())
    dupes = {v for v in values if values.count(v) > 1}
    assert not dupes, f"T1 FAIL: fingerprint collision across games: {dupes}"
    print(f"T1 PASS: 25 games -> {len(set(values))} distinct fingerprints")

    # T2 stability: reopen each game 3 more times in FRESH arcades
    unstable = []
    for trial in range(3):
        arcade = fresh_arcade()
        for g in GAMES:
            fp, _ = open_and_fingerprint(arcade, g)
            if fp != first[g]:
                unstable.append((g, trial))
    stable = [g for g in GAMES if g not in {u[0] for u in unstable}]
    print(f"T2: stable fingerprints on {len(stable)}/25 games across 4 opens"
          + (f"; UNSTABLE: {sorted({u[0][:4] for u in unstable})}" if unstable else ""))

    # T3 clone hit through the competition simulator's own cloning mechanism
    sys.path.insert(0, str(REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"))
    from taaf.competition_arcade import CompetitionArcadeServer

    subset = GAMES[:6]
    server = CompetitionArcadeServer(
        game_ids=subset,
        total_runs=12,  # each source game cloned ~2x with anonymized IDs
        environments_dir=str(REPO / "environment_files"),
    ).start()
    try:
        client = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=server.base_url,
            arc_api_key=server.api_key,
        )
        card_id = client.open_scorecard()
        # Ground truth from the SERVER side (the client is anonymized — its
        # environment_info carries no private tags, verified this run):
        truth: dict[str, str] = {}
        for info in server._arcade.available_environments:
            src = next((t.split(":", 1)[1] for t in (info.private_tags or [])
                        if t.startswith("taaf_source_game:")), None)
            if src:
                truth[info.game_id] = src
        assert truth, "simulator exposed no clone ground truth"
        hits = misses = wrong = 0
        for clone_id in server.exposed_game_ids:
            import arcengine

            env = client.make(clone_id, scorecard_id=card_id)
            resp = env.reset()
            data = resp.frame[-1]
            rows = data.tolist() if hasattr(data, "tolist") else data
            grid = tuple(tuple(int(c) for c in row) for row in rows)
            levels = int(resp.win_levels or 0)
            actions = [a.value if hasattr(a, "value") else int(a) for a in (resp.available_actions or [])]
            fp = fingerprint(grid, levels, actions)
            src = truth.get(clone_id)
            if src is None:
                continue
            if src in {g for g, f in first.items() if f == fp}:
                hits += 1
            elif fp in first.values():
                wrong += 1
            else:
                misses += 1
        print(f"T3: clones fingerprint-matched to source: {hits} hits, "
              f"{misses} misses (safe no-op), {wrong} WRONG matches")
        assert wrong == 0, "T3 FAIL: a clone matched the WRONG source game"
    finally:
        server.stop()

    print("FINGERPRINT VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
