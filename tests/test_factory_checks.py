"""Tests for KaggleScoreCollector.

ERROR submissions must not count as pending.
"""
import sys
import types
from unittest.mock import patch


def _install_sf_stub() -> None:
    if "software_factory.loop.collectors" in sys.modules:
        return
    sf = types.ModuleType("software_factory")
    loop = types.ModuleType("software_factory.loop")
    coll = types.ModuleType("software_factory.loop.collectors")

    class _CheckVerdict:
        PASS = "PASS"
        FAIL = "FAIL"
        WARN = "WARN"

    class _CheckResult:
        def __init__(self, name, verdict, evidence=None):
            self.name = name
            self.verdict = verdict
            self.evidence = evidence or {}

    coll.CheckVerdict = _CheckVerdict
    coll.CheckResult = _CheckResult
    sf.loop = loop
    loop.collectors = coll
    sys.modules["software_factory"] = sf
    sys.modules["software_factory.loop"] = loop
    sys.modules["software_factory.loop.collectors"] = coll


_install_sf_stub()

from arcagi3.factory_checks import KaggleScoreCollector  # noqa: E402

_CSV_MIXED = chr(10).join([
    "ref,status,publicScore",
    "sub_1,complete,0.22",
    "sub_2,error,",
    "sub_3,error,",
    "sub_4,pending,",
    "",
])


def test_error_submissions_counted_separately_from_pending():
    with patch("arcagi3.factory_checks._run", return_value=(0, _CSV_MIXED)):
        results = KaggleScoreCollector().scan(None)
    ev = results[0].evidence
    assert ev["pending"] == 1
    assert ev["errored"] == 2


def test_verdict_unaffected_by_error_fix():
    with patch("arcagi3.factory_checks._run", return_value=(0, _CSV_MIXED)):
        results = KaggleScoreCollector().scan(None)
    assert results[0].verdict == "PASS"
