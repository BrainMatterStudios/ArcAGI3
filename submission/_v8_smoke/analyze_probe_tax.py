#!/usr/bin/env python3
"""analyze_probe_tax.py — PRE-REGISTERED reads for smoke #2.

Written and committed 2026-08-26 20:2x CEST, while kernel arc3-v8-smoke v2
was still RUNNING and before any of its output existed. Every field it
consumes is already persisted by the kernel's report cell, so no re-push is
needed: the reads below are computed from v8_smoke_results.json.

WHY THESE READS (coordinator, smoke #2 arm review): the early probe's cost
lands on SCORE (efficiency), not on level count, so the original bar
("no regression > 1 level on vc33/dc22") does not test the thing that can
sink the lever.

READ 1  vc33 SCORE vs its comparators (v12 10.71; effort-smoke 4.73), with
        its level-1 action count separated into LLM actions and probe
        actions. Offline these are separable by construction: the harness
        counts only LLM-executed actions (the grinder steps the raw engine),
        so actions_per_level[0] is the no-probe count and the probe's
        contribution is the telemetry's early_probe_actions.

READ 2  Per game: probe actions charged, level-1 baseline, and the implied
        efficiency delta on every level the LLM completed.

READ 3  L = mean per-game score loss on NON-MATCHING games, and the
        break-even detection rate p* = L / (GAIN + L) with GAIN = 85.7
        (ft09 live 14.29 -> banked-crack projection 100.0).

KILL CRITERION, pre-registered: if the measured L implies p* > 15%, early
detection reverts to stall-gated (EXPLORER_V8_EARLY=0, ENVELOPE §A3) and
only the stall-gated specialist ships.

Three L estimates are reported, because they answer different questions:
  L_proj  PROJECTED, not observed: mean(final_score - score_if_probe_billed)
          over the non-matching games played. An OFFLINE smoke cannot observe
          this at all — without the competition guard a reset at
          action_count == 0 opens a fresh play, so offline max-over-plays
          hides the tax entirely. Treat this as a model check, and take
          L_measured (competition server) as the evidence.
  L_cond  the same mean restricted to games that completed >= 1 level.
  L_worst the coordinator's risk case, per game: assume the LLM completes
          level 1 at exactly its baseline (the best case the probe can
          spoil) and charge the observed probe actions on top.

An arithmetic bound worth stating: the probe only ever damages level 1's
term, whose weight is 1 of W = n(n+1)/2, so the loss on an n-level game is
bounded by 100/W — 4.76 on a 6-level game, 2.78 on an 8-level game, but 33
on a 2-level game. The level-count distribution of the hidden set therefore
drives L more than the probe's action count does; this script prints the
bound per game so the estimate is auditable rather than asserted.

Usage: python3 submission/_v8_smoke/analyze_probe_tax.py [results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT = HERE / "results" / "v8_smoke_results.json"

GAIN = 85.7          # ft09: 100.0 banked-crack projection - 14.29 live (v1)
# MEASURED on the authoritative competition server, 2026-08-27
# (test_competition_scoring.py::test_J_probe_tax_measured_on_the_competition_server):
# a declined 89-action probe on ft09 followed by its minimal 4-action level-1
# plan scored 0.9965 against a no-probe control of 4.7619 => 3.77 points, one
# play, matching the billed-together prediction to 2%.
MEASURED_TAX = 4.761904761904762 - 0.9964646791265169
KILL_P = 0.15        # pre-registered: break-even above this => revert
COMPARATORS = {"vc33": {"v12": 10.71, "effort_smoke": 4.73}}

# Measured probe cost (test_I_probe_cost_is_bounded_and_measured, real engine).
PROBE_COST_MEAN = 141        # mean of ft09 89, cn04 113, sk48 135, vc33 179, dc22 187
PROBE_COST_MAX = 187

# Every offline game's level count and level-1 baseline, read from the engine
# 2026-08-26 (env.step(RESET).win_levels + environment_info.baseline_actions on
# all 25 environment_files games). Needed because the probe can only ever
# damage level 1's term, whose weight is 1 of W = n(n+1)/2 — so the loss on an
# n-level game is bounded by 100/W REGARDLESS of how many actions the probe
# spends. The coordinator's "L could be 20+" case needs W <= 5, i.e. a 2-level
# game; the smallest game in the corpus has SIX levels (W=21, bound 4.76).
CORPUS = {
    "ar25": (8, 32), "bp35": (9, 21), "cd82": (6, 55), "cn04": (6, 29),
    "dc22": (6, 59), "ft09": (6, 43), "g50t": (7, 78), "ka59": (7, 28),
    "lf52": (10, 32), "lp85": (8, 17), "ls20": (7, 22), "m0r0": (6, 30),
    "r11l": (6, 22), "re86": (8, 26), "s5i5": (8, 20), "sb26": (8, 18),
    "sc25": (6, 36), "sk48": (8, 61), "sp80": (6, 39), "su15": (9, 22),
    "tn36": (7, 32), "tr87": (6, 54), "tu93": (9, 19), "vc33": (7, 7),
    "wa30": (9, 71),
}


def score_from(apl, baselines, levels_completed, n_levels):
    """taaf.game.GameRun._compute_final_score on arbitrary action counts."""
    total, weights, max_weights = 0.0, 0, 0
    for idx in range(int(n_levels or 0)):
        weight = idx + 1
        weights += weight
        acts = apl[idx] if idx < len(apl) else 0
        base = baselines[idx] if idx < len(baselines) else None
        score = (min(115.0, (float(base) / acts) ** 2 * 100)
                 if (idx < levels_completed and acts and base) else 0.0)
        if score > 0:
            max_weights += weight
        total += score * weight
    if not weights:
        return 0.0
    return min(total / weights, max_weights / weights * 100)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    data = json.loads(path.read_text())
    tel = (data.get("telemetry") or {}).get("games") or {}
    rows = data["games"]
    if rows and "actions_per_level" not in rows[0]:
        sys.exit(f"{path}: pre-v8.1 results schema (no actions_per_level) — "
                 "this analysis needs a smoke #2 run")

    print("=" * 78)
    print("PRE-REGISTERED READ 1 — vc33 SCORE (not levels) vs its comparators")
    for row in rows:
        if row["stem"] != "vc33":
            continue
        rec = tel.get(row["game_id"], {})
        probe = int(rec.get("early_probe_actions", 0) or 0)
        apl = list(row["actions_per_level"])
        l1_llm = apl[0] if apl else 0
        base1 = (row["base_actions_per_level"] or [None])[0]
        print(f"  vc33 harness score {row['final_score']} vs v12 "
              f"{COMPARATORS['vc33']['v12']} / effort-smoke "
              f"{COMPARATORS['vc33']['effort_smoke']}")
        print(f"  vc33 levels {row['levels_completed']}/{row['number_of_levels']}")
        print(f"  vc33 level-1 actions WITHOUT the probe (LLM only): {l1_llm}")
        print(f"  vc33 level-1 actions WITH the probe (live-equivalent): "
              f"{l1_llm + probe}   [probe charged {probe}, baseline {base1}]")
        print(f"  vc33 score if the probe were billed live: "
              f"{row.get('score_if_probe_billed')}")

    print("=" * 78)
    print("PRE-REGISTERED READ 2 — per game: probe charged, L1 baseline, "
          "efficiency delta on every level the LLM completed")
    for row in rows:
        rec = tel.get(row["game_id"], {})
        probe = int(rec.get("early_probe_actions", 0) or 0)
        apl = list(row["actions_per_level"])
        bases = row["base_actions_per_level"]
        print(f"  {row['game_id']}: probe={probe} actions, "
              f"L1 baseline={bases[0] if bases else None}, "
              f"levels={row['levels_completed']}/{row['number_of_levels']}, "
              f"grinder_engine_actions={rec.get('grinder_actions', 0)}, "
              f"detected={rec.get('early_detect')!r}")
        for idx in range(int(row["levels_completed"])):
            a = apl[idx] if idx < len(apl) else 0
            b = bases[idx] if idx < len(bases) else None
            if not (a and b):
                continue
            eff = min(115.0, (b / a) ** 2 * 100)
            charged = a + (probe if idx == 0 else 0)
            eff_live = min(115.0, (b / charged) ** 2 * 100)
            print(f"      L{idx + 1}: {a} actions vs baseline {b} -> "
                  f"{eff:.1f}%  |  with probe billed: {charged} actions -> "
                  f"{eff_live:.1f}%   (delta {eff_live - eff:+.1f} pts of that "
                  f"level's term)")

    print("=" * 78)
    print("PRE-REGISTERED READ 3 — L and the break-even detection rate p*")
    losses_obs, losses_cond, losses_worst, bounds = [], [], [], []
    for row in rows:
        rec = tel.get(row["game_id"], {})
        if rec.get("early_detect"):
            continue                       # matched game: it pays no net tax
        probe = int(rec.get("early_probe_actions", 0) or 0)
        n = int(row["number_of_levels"] or 0)
        weights = n * (n + 1) // 2
        bound = (100.0 / weights) if weights else 0.0
        bounds.append((row["game_id"], n, bound))
        loss = float(row["final_score"] or 0.0) - float(
            row.get("score_if_probe_billed") or 0.0)
        losses_obs.append(loss)
        if int(row["levels_completed"]) >= 1:
            losses_cond.append(loss)
        bases = row["base_actions_per_level"]
        if bases and probe:
            b1 = float(bases[0])
            best = score_from([b1] + [0] * (n - 1), bases, 1, n)
            taxed = score_from([b1 + probe] + [0] * (n - 1), bases, 1, n)
            losses_worst.append(best - taxed)

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    def pstar(loss):
        return loss / (GAIN + loss) if (GAIN + loss) else 1.0

    print("  per-game arithmetic bound on the tax (100 / W, W = n(n+1)/2):")
    for gid, n, bound in bounds:
        print(f"      {gid}: {n} levels -> W={n * (n + 1) // 2}, "
              f"max possible loss {bound:.2f} pts")
    for label, xs in (("L_measured (COMPETITION SERVER, ft09 A/B)", [MEASURED_TAX]),
                      ("L_proj  (projection over non-matching games played)", losses_obs),
                      ("L_cond  (projection, >=1 level completed)", losses_cond),
                      ("L_worst (projection, LLM finishes L1 at baseline)", losses_worst)):
        L = mean(xs)
        p = pstar(L)
        verdict = "CLEARS" if p <= KILL_P else "FAILS"
        print(f"  {label}: n={len(xs)} L={L:.2f} pts -> break-even p* = "
              f"{100 * p:.1f}%  [{verdict} the {100 * KILL_P:.0f}% kill line]")

    # --- corpus-wide worst case: EVERY game, LLM finishes L1 at baseline ----
    print("-" * 78)
    print("  CORPUS-WIDE worst case (all 25 offline games, engine-read level "
          f"counts + L1 baselines, probe = {PROBE_COST_MEAN} actions mean / "
          f"{PROBE_COST_MAX} max):")
    corpus_losses, corpus_losses_max = [], []
    worst_game = (None, 0.0)
    for stem, (n, b1) in CORPUS.items():
        weights = n * (n + 1) // 2
        for cost, bucket in ((PROBE_COST_MEAN, corpus_losses),
                             (PROBE_COST_MAX, corpus_losses_max)):
            eff = min(115.0, (b1 / (b1 + cost)) ** 2 * 100)
            loss = (100.0 - eff) / weights     # L1's term is weight 1 of W
            bucket.append(loss)
            if cost == PROBE_COST_MAX and loss > worst_game[1]:
                worst_game = (stem, loss)
    bound_max = max(100.0 / (n * (n + 1) // 2) for n, _ in CORPUS.values())
    print(f"      L1-term loss, mean probe: {mean(corpus_losses):.2f} pts "
          f"(max over games {max(corpus_losses):.2f})")
    print(f"      L1-term loss, max  probe: {mean(corpus_losses_max):.2f} pts "
          f"(worst game {worst_game[0]} at {worst_game[1]:.2f})")
    print(f"      ARITHMETIC CEILING on any corpus game: {bound_max:.2f} pts "
          f"(=100/W of the smallest game, {min(n for n, _ in CORPUS.values())} "
          f"levels) -> break-even p* can never exceed "
          f"{100 * bound_max / (GAIN + bound_max):.1f}% on corpus-shaped games")
    corpus_p = pstar(mean(corpus_losses_max))
    print(f"      corpus worst-case L={mean(corpus_losses_max):.2f} -> "
          f"p*={100 * corpus_p:.1f}% "
          f"[{'CLEARS' if corpus_p <= KILL_P else 'FAILS'} the kill line]")

    L_decisive = max(MEASURED_TAX, mean(losses_cond), mean(losses_worst),
                     mean(corpus_losses_max))
    p_decisive = pstar(L_decisive)
    kill = p_decisive > KILL_P
    print("-" * 78)
    print(f"  DECISIVE L (max of L_cond, L_worst) = {L_decisive:.2f} pts")
    print(f"  break-even detection rate p* = {100 * p_decisive:.1f}% "
          f"(gain per matched game = {GAIN})")
    print(f"  PRE-REGISTERED KILL CRITERION (p* > {100 * KILL_P:.0f}%): "
          f"{'TRIGGERED — revert to stall-gated (EXPLORER_V8_EARLY=0)' if kill else 'NOT triggered — early detection may ship'}")

    out = {
        "gain_per_matched_game": GAIN,
        "kill_p": KILL_P,
        "L_measured_competition_server": MEASURED_TAX,
        "L_proj": mean(losses_obs),
        "L_cond": mean(losses_cond),
        "L_worst": mean(losses_worst),
        "L_corpus_worst": mean(corpus_losses_max),
        "L_corpus_mean_probe": mean(corpus_losses),
        "corpus_arithmetic_ceiling": bound_max,
        "L_decisive": L_decisive,
        "break_even_p": p_decisive,
        "kill_triggered": kill,
        "per_game_loss_bound": [{"game_id": g, "levels": n, "bound": b}
                                for g, n, b in bounds],
    }
    (path.parent / "probe_tax_analysis.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", path.parent / "probe_tax_analysis.json")


if __name__ == "__main__":
    main()
