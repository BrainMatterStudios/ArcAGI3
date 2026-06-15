"""Software-factory observe collectors for ArcAGI3 (config: observe.collectors).

This project has no live service / DB; the meaningful health signal is "is the
agent still importable and still winning the offline local games?" These collectors
shell out to the PROJECT's `uv` env (the factory CLI runs in its own pipx venv that
lacks arc_agi/numpy), so they work regardless of which interpreter invokes them.

The full 25-game real-game sweep stays a scheduled/manual gate (slow, needs
ARC_API_KEY) — these are the fast, offline, deterministic checks an observe pass runs.

Wire-up: `observe.collectors: arcagi3.factory_checks:collectors` in factory.config.yaml
(run `factory observe` with PYTHONPATH=src so this module is importable).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from software_factory.loop.collectors import CheckResult, CheckVerdict

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/ArcAGI3)
# Expected local-game floor (memory: salience wins all local games). FAIL if a game
# fails to win or the total drops below this.
_LOCAL_MIN_TOTAL = 24


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env={"PYTHONPATH": "src", "ARC_API_KEY": "local-dev", "PATH": _path()})
    return p.returncode, (p.stdout + p.stderr)


def _path() -> str:
    import os
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


class ImportSafetyCollector:
    """The agent must import with no network / GPU / API key (the eval contract)."""

    name = "import_safety"

    def scan(self, data) -> list[CheckResult]:  # noqa: ARG002 (offline; ignores DataAdapter)
        try:
            rc, out = _run(["uv", "run", "python", "-c",
                            "from arcagi3.salience_explorer import SalienceExplorer; "
                            "SalienceExplorer(); print('OK')"], timeout=120)
        except Exception as e:  # noqa: BLE001
            return [CheckResult(self.name, CheckVerdict.FAIL, {"error": repr(e)})]
        ok = rc == 0 and "OK" in out
        return [CheckResult(self.name, CheckVerdict.PASS if ok else CheckVerdict.FAIL,
                            {"returncode": rc, "tail": out[-300:]})]


class LocalGamesCollector:
    """Run the offline local games via the salience policy; FAIL on any regression."""

    name = "local_games"

    def scan(self, data) -> list[CheckResult]:  # noqa: ARG002
        try:
            rc, out = _run(["uv", "run", "python", "-m", "arcagi3.runner",
                            "--agent", "salience", "--budget", "16000",
                            "--games-dir", "src/arcagi3/games", "--quiet"], timeout=600)
        except Exception as e:  # noqa: BLE001
            return [CheckResult(self.name, CheckVerdict.FAIL, {"error": repr(e)})]
        m = re.search(r"TOTAL\s+levels\s+(\d+).*?wins\s+(\d+)/(\d+)", out, re.S)
        if rc != 0 or not m:
            return [CheckResult(self.name, CheckVerdict.FAIL,
                                {"returncode": rc, "tail": out[-400:]})]
        total, wins, games = int(m.group(1)), int(m.group(2)), int(m.group(3))
        verdict = (CheckVerdict.PASS if (wins == games and total >= _LOCAL_MIN_TOTAL)
                   else CheckVerdict.FAIL)
        return [CheckResult(self.name, verdict,
                            {"total_levels": total, "wins": wins, "games": games,
                             "floor": _LOCAL_MIN_TOTAL})]


class KaggleScoreCollector:
    """Read the competition leaderboard score (read-only; no submit) and emit a signal.

    Closes the submit->score->improve loop: parses `kaggle competitions submissions`,
    finds the latest COMPLETE public score and the best prior, and flags regressions or
    pending submissions. WARN (not FAIL) on transient/auth/network issues so a flaky
    network can't abort an observe pass (conventions §8 degrade-to-WARN).
    """

    name = "kaggle_score"
    COMP = "arc-prize-2026-arc-agi-3"

    def scan(self, data) -> list[CheckResult]:  # noqa: ARG002
        try:
            rc, out = _run(["kaggle", "competitions", "submissions", self.COMP, "--csv"],
                           timeout=120)
        except Exception as e:  # noqa: BLE001
            return [CheckResult(self.name, CheckVerdict.WARN, {"error": repr(e)})]
        if rc != 0:
            return [CheckResult(self.name, CheckVerdict.WARN,
                                {"returncode": rc, "tail": out[-300:]})]
        rows = self._parse_csv(out)
        if not rows:
            return [CheckResult(self.name, CheckVerdict.WARN, {"note": "no submissions parsed"})]
        scored = [r for r in rows if r["score"] is not None]
        unscored = [r for r in rows if r["score"] is None and r["status"]]
        errored = [r for r in unscored if "ERROR" in r["status"].upper()]
        pending = [r for r in unscored if "ERROR" not in r["status"].upper()
                   and "COMPLETE" not in r["status"].upper()]
        if not scored:
            return [CheckResult(self.name, CheckVerdict.WARN,
                                {"note": "no scored submission yet",
                                 "pending": len(pending), "errored": len(errored)})]
        latest = scored[0]                      # submissions list is newest-first
        best = max(r["score"] for r in scored)
        ev = {"latest_ref": latest["ref"], "latest_score": latest["score"],
              "best_score": best, "delta_vs_best": round(latest["score"] - best, 6),
              "pending": len(pending), "errored": len(errored), "scored_count": len(scored)}
        if latest["score"] < best:             # banked a worse agent than a prior best
            return [CheckResult(self.name, CheckVerdict.FAIL, ev)]
        return [CheckResult(self.name, CheckVerdict.PASS, ev)]

    @staticmethod
    def _parse_csv(text: str) -> list[dict]:
        import csv
        import io
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []
        out = []
        for row in csv.DictReader(io.StringIO("\n".join(lines))):
            keys = {k.lower(): k for k in row}
            sc = row.get(keys.get("publicscore", ""), "")
            try:
                score = float(sc) if sc not in ("", None) else None
            except ValueError:
                score = None
            out.append({"ref": row.get(keys.get("ref", ""), ""),
                        "status": row.get(keys.get("status", ""), ""), "score": score})
        return out


collectors = [ImportSafetyCollector(), LocalGamesCollector(), KaggleScoreCollector()]
