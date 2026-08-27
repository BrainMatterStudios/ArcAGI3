#!/usr/bin/env python3
"""build_duck38_v8.py — THE SCORED V8 ARM: the duck38-v12 base plus the
crack-or-nothing SearchCore graft, in its exact flight configuration.

ARM: the duck38-v12 base (Aug-07 community anim bundle + Qwen3.8-27B-FP8
repacked, boot-attested, flown live at 1.55) plus exactly ONE graft —
submission/_explorer_v8/graft_explorer_v8.py with its six-file flat bundle —
installed with the FLIGHT CONFIG below. effort_medium is DELIBERATELY ABSENT:
it was REVERTED as harmful live (pooled 1.12 vs 1.55, commit 6a8e11f).

THE FLIGHT CONFIG (``FLIGHT_CONFIG``; every line asserted into the notebook
exactly once, before the run cell, and every one set EXPLICITLY even where it
equals the code default so the run log states the whole config):

  EXPLORER_V8=1                      SearchCore portfolio replaces v7's grind
  EXPLORER_V8_EARLY=1                detection at game start, not after a stall
  EXPLORER_V8_EARLY_ENGAGE=1         engage on a hit, same envelope guards
  EXPLORER_V8_PRESCREEN=1            ZERO-ACTION frame-0 screen in front of
                                     everything — 0 probe actions on 24 of 25
                                     dev fixtures, ft09 retained (1 TP, 0 FN,
                                     0 FP)
  EXPLORER_V8_ENGAGE_SPECIALISTS=ft09_gf2
                                     engage ONLY the class MEASURED to crack;
                                     detection alone is not a crack predictor,
                                     and a failed engagement is billed into
                                     the play the LLM keeps using
  EXPLORER_V8_SPECIALIST_MAX_ACTIONS=4000
  EXPLORER_V8_STALL_GRIND=0          generic stall grind OFF — partial
                                     grinding measured NET-NEGATIVE (smoke #2:
                                     292 687 / 292 635 engine actions on
                                     dc22 / sk48 for 2 and 0 unlocks)
  EXPLORER_V8_BANK=1                 a crack scores only if replayed into a
                                     fresh play; the engine takes MAX over
                                     plays
  EXPLORER_V8_STOP_AFTER_CRACK=1     free the worker once the banked play lands

LIVE EVIDENCE (kernel arc3-v8-smoke version 4, RTX Pro 6000, COMPLETE
2026-08-27, results in submission/_v8_smoke/results/): 5/5 pre-registered
bars, graded on the ENGINE SCORECARD.
  * pre-screen DECLINED vc33 / dc22 / sk48 at exactly 0 probe actions, 0
    engagements, 0 grinder engine actions
  * ft09 admitted (strict_lattice n=16), detected ft09_gf2 in 89 probe
    actions, cracked 6/6 in 1233 engine actions, BANKED a 75-action replay:
    ENGINE score 100.0, plays [3.512, 100.0]
  * every envelope guard well inside cap (worst 1232/292500 actions in one
    engagement, 0.6 s of the 2700 s cumulative grind budget)
  * untouched games within draw variance of their v12 comparators
    (vc33 L3 21.43, sk48 L1 2.78, dc22 L0)

CONSTRUCTION: the tracked v12 notebook (submission/_duck38_v12/
arc3-duck38-v12.ipynb, code-cell sha256 dc2c36f8…, the bytes that flew 1.55)
is replicated EXACTLY — boot attestation cell, commit-smoke hook (offline
3-game 60-min box; the scored KAGGLE_IS_COMPETITION_RERUN path never enters
it), TRUE_SUBMISSION wiring — with ONE inserted cell immediately before the
run cell: the six-file bundle embedded inline (the xd / effort cell pattern)
plus install() behind a HARD assert on every half of the verdict. Rationale
for the hard gate: a scored run that silently plays stock v12 poisons the
read, while an ERROR costs no slot (2026-08-01 audit law).

The smoke's telemetry and report cells are DELIBERATELY NOT carried over —
they are an instrument, not the arm.

ENVELOPE: unchanged from the v7 flight that completed the 9 h box; every guard
is enforced on every engine call and was observed inside cap live.

Usage:
  python3 submission/_duck38_v8/build_duck38_v8.py
  cd submission/_duck38_v8 && \
    python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
KERNEL_SLUG = "arc3-duck38-v8"
BASE_CODE_SHA256 = "dc2c36f897428123b2b9e95d85470b57cf6d954910599be3908b0a38a8199615"

# The six files the v8 bundle needs flat in one directory (envelope doc §9;
# prescreen.py added by v8.3). Order below is the WRITE order in the cell —
# graft_explorer_v8 last, since it imports the rest.
BUNDLE_FILES = {
    "graft_explorer.py": SUB / "_explorer_floor" / "graft_explorer.py",
    "graft_bank.py": SUB / "_duck38_v12_bank" / "graft_bank.py",
    "search_core.py": SUB / "_search_core" / "search_core.py",
    "prescreen.py": SUB / "_search_core" / "prescreen.py",
    "specialists.py": SUB / "_search_core" / "specialists.py",
    "graft_explorer_v8.py": SUB / "_explorer_v8" / "graft_explorer_v8.py",
}

FLIGHT_CONFIG = {
    "EXPLORER": "1",
    "EXPLORER_V8": "1",
    "EXPLORER_V8_EARLY": "1",
    "EXPLORER_V8_EARLY_ENGAGE": "1",
    "EXPLORER_V8_PRESCREEN": "1",
    "EXPLORER_V8_ENGAGE_SPECIALISTS": "ft09_gf2",
    "EXPLORER_V8_SPECIALIST_MAX_ACTIONS": "4000",
    "EXPLORER_V8_STALL_GRIND": "0",
    "EXPLORER_V8_BANK": "1",
    "EXPLORER_V8_STOP_AFTER_CRACK": "1",
}

MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"
MARK_SMOKE = 'SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]'

GRAFT_CELL_HEAD = r'''# THE ONE DELTA vs flown v12: the explorer-v8 crack-or-nothing graft.
#
# Every flag is set EXPLICITLY, including the ones that equal the code
# default, so the scored run's own log states the entire configuration and a
# default drift cannot silently change the arm.
#
# WHY EACH ONE (all measured, docs/ENVELOPE-2026-08-26-v8.md + Addendum 3):
#   PRESCREEN        a zero-action frame-0 screen decides whether a game is
#                    even plausibly of an engageable class. 25 dev fixtures:
#                    1 TP (ft09), 0 FN, 0 FP, 0 engine actions, ~3 ms. Mean
#                    probe cost per game falls 125.4 -> 3.6 actions.
#   ENGAGE=ft09_gf2  detection is NOT a crack predictor. A detection that
#                    engages and fails spends engine actions billed into the
#                    SAME play the LLM keeps using (a failed engagement cannot
#                    open a fresh play — only a post-WIN reset escapes the
#                    competition guard, api.py:316-334). So engage only the
#                    class measured to crack its game.
#   STALL_GRIND=0    the generic stall grind cracked NOTHING in smoke #2 while
#                    spending 292 687 / 292 635 engine actions on dc22 / sk48.
#                    Partial grinding is net-negative; crack or spend nothing.
#   BANK=1           a crack scores only when replayed into a fresh play; the
#                    engine takes the MAX over plays. Live: ft09 plays
#                    [3.512, 100.0] -> environment score 100.0.
#
# effort_medium is DELIBERATELY ABSENT (reverted as harmful live, pooled 1.12
# vs 1.55, commit 6a8e11f) and is purged from the environment below along with
# the other stale grafts.
#
# LIVE GATE ALREADY PASSED: kernel arc3-v8-smoke version 4 (RTX Pro 6000,
# COMPLETE) ran this exact configuration and passed all five pre-registered
# bars on the ENGINE SCORECARD.
'''

GRAFT_CELL_TAIL = r'''
_V8_DIR = WORKING_DIR / "v8_bundle"
_V8_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in _V8_BUNDLE.items():
    (_V8_DIR / _name).write_text(_src, encoding="utf-8")
os.environ["EXPLORER_V8_CORE_DIR"] = str(_V8_DIR)
if str(_V8_DIR) not in sys.path:
    sys.path.insert(0, str(_V8_DIR))

import importlib as _importlib

_v8mod = _importlib.import_module("graft_explorer_v8")
_v8_status = _v8mod.install()
print("[explorer-v8]", _v8_status)
# HARD GATE on every half of the verdict. A scored run that silently plays
# stock v12 poisons the read; an ERROR costs no slot (2026-08-01 audit law).
# The smoke commit executed this identical path and passed these identical
# asserts.
assert "explorer: OK" in _v8_status, "v7 substrate must be live, got: " + repr(_v8_status)
assert "v8: OK" in _v8_status, "v8 grind must be live, got: " + repr(_v8_status)
assert "early: OK" in _v8_status, "early probe must be live, got: " + repr(_v8_status)
assert "prescreen: OK" in _v8_status, "the zero-action pre-screen must be live, got: " + repr(_v8_status)
assert "stop-after-crack" in _v8_status, "stop-after-crack must be live, got: " + repr(_v8_status)
print("[explorer-v8] flags:", {k: v for k, v in os.environ.items() if k.startswith("EXPLORER")})
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    base_code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert hashlib.sha256(base_code.encode()).hexdigest() == BASE_CODE_SHA256, (
        "base v12 notebook drifted from the flown-1.55 bytes — re-attest before building"
    )

    sources = {}
    for name, path in BUNDLE_FILES.items():
        text = path.read_text()
        assert text.strip(), name
        sources[name] = text
    assert "def install() -> str:" in sources["graft_explorer_v8.py"]
    assert "def _grind_v8" in sources["graft_explorer_v8.py"]
    assert "class _BankingKillSwitch" in sources["graft_bank.py"]
    assert "def screen(" in sources["prescreen.py"]

    run_hits = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and MARK_RUN in "".join(c["source"])
    ]
    assert len(run_hits) == 1, run_hits
    run_idx = run_hits[0]

    flag_lines = "".join(
        f'os.environ["{k}"] = "{v}"\n' for k, v in FLIGHT_CONFIG.items()
    )
    purge = (
        'for _stale in ("EFFORT_MEDIUM", "EFFORT_DEAD_RETRY", "YIELD_CARRYOVER",\n'
        '               "YIELD_SLICE_CAP"):\n'
        "    os.environ.pop(_stale, None)\n\n"
    )
    bundle_lines = ["_V8_BUNDLE = {\n"]
    for name in BUNDLE_FILES:
        bundle_lines.append(f"    {name!r}: {sources[name]!r},\n")
    bundle_lines.append("}\n")

    graft_cell_text = (
        GRAFT_CELL_HEAD + flag_lines + purge + "".join(bundle_lines) + GRAFT_CELL_TAIL
    )
    # The graft sources must ride the notebook byte-identically: embedded via
    # !r, and Python str repr round-trips by language guarantee.
    for name, text in sources.items():
        assert repr(text) in graft_cell_text, name

    nb["cells"].insert(
        run_idx,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": graft_cell_text.splitlines(keepends=True),
        },
    )

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined and MARK_SMOKE in joined
    assert joined.index(MARK_ATTEST) < joined.index(MARK_SMOKE)
    assert joined.index(MARK_SMOKE) < joined.index('os.environ["EXPLORER_V8"] = "1"')
    for flag, val in FLIGHT_CONFIG.items():
        line = f'os.environ["{flag}"] = "{val}"'
        assert joined.count(line) == 1, (flag, joined.count(line))
        assert joined.index(line) < joined.index(MARK_RUN), flag
    assert 'EFFORT_MEDIUM"] = "1"' not in joined, "effort_medium must stay OFF"
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined  # TRUE_SUBMISSION path intact
    assert "Qwen/Qwen3.8-27B-FP8" in joined         # served brain unchanged
    assert len(nb["cells"]) == 12                   # 11 v12 cells + the graft cell

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"ahmedmobasher86/{KERNEL_SLUG}",
                "title": KERNEL_SLUG,
                "code_file": f"{KERNEL_SLUG}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "machine_shape": "NvidiaRtxPro6000",
                "dataset_sources": [
                    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                    "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
                    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
                ],
                "kernel_sources": [],
                # LAW (serving-lab, 3 wasted pushes): the competition source is
                # what gates the RTX Pro 6000 pool — machine_shape +
                # --accelerator alone bind P100.
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "model_sources": [
                    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
                ],
            },
            indent=2,
        )
        + "\n"
    )

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(nb["cells"]), "notebook bytes:",
          (HERE / f"{KERNEL_SLUG}.ipynb").stat().st_size)


if __name__ == "__main__":
    main()
