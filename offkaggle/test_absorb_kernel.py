#!/usr/bin/env python3
"""Tests for absorb_kernel.py. The load-bearing one is
test_reproduces_keith_attestation: it proves the hashing convention here is the
same one that produced submission/_keith_copy/ATTEST.json for our current live
base, so an Oct-1 absorption is comparable to the 09-05 one."""
import hashlib
import json
import types
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import absorb_kernel as ak  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KEITH = REPO / "submission" / "_keith_copy"


class AttestTests(unittest.TestCase):
    def test_reproduces_keith_attestation(self):
        """The recorded ATTEST.json for our live base must recompute exactly."""
        nb = KEITH / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
        recorded = json.loads((KEITH / "ATTEST.json").read_text())
        got = ak.attest(nb)
        self.assertEqual(got["file_sha256"], recorded["file_sha256"])
        self.assertEqual(got["code_cell_sha256"], recorded["code_cell_sha256"])

    def test_code_cell_hash_ignores_outputs(self):
        """Re-pulling the same code with different outputs must not change the
        code-cell hash — otherwise the identity is useless across pulls."""
        nb = {"cells": [
            {"cell_type": "code", "source": ["print(1)"], "outputs": [{"x": 1}],
             "execution_count": 3},
            {"cell_type": "markdown", "source": ["ignored"]},
        ]}
        h1 = ak.code_cell_sha256(nb)
        nb["cells"][0]["outputs"] = [{"x": 999}]
        nb["cells"][0]["execution_count"] = 77
        self.assertEqual(h1, ak.code_cell_sha256(nb))

    def test_markdown_excluded(self):
        a = {"cells": [{"cell_type": "code", "source": ["x=1"]}]}
        b = {"cells": [{"cell_type": "code", "source": ["x=1"]},
                       {"cell_type": "markdown", "source": ["notes"]}]}
        self.assertEqual(ak.code_cell_sha256(a), ak.code_cell_sha256(b))

    def test_source_may_be_a_string(self):
        a = {"cells": [{"cell_type": "code", "source": ["x=", "1"]}]}
        b = {"cells": [{"cell_type": "code", "source": "x=1"}]}
        self.assertEqual(ak.code_cell_sha256(a), ak.code_cell_sha256(b))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.src = json.loads((KEITH / "kernel-metadata.json").read_text())

    def test_matches_the_hand_built_copy(self):
        """copy_metadata must produce what we built by hand on 09-05."""
        want = json.loads((KEITH / "push_bothmounts" / "kernel-metadata.json").read_text())
        got = ak.copy_metadata(self.src, want["id"], want["code_file"])
        self.assertEqual(got, want)

    def test_regime_carried_verbatim(self):
        got = ak.copy_metadata(self.src, "me/copy", "copy.ipynb")
        for k in ak.REGIME_KEYS:
            self.assertEqual(got[k], self.src[k], f"{k} must be carried verbatim")

    def test_source_identity_dropped(self):
        got = ak.copy_metadata(self.src, "me/copy", "copy.ipynb")
        self.assertNotIn("id_no", got)
        self.assertNotIn("keywords", got)
        self.assertEqual(got["id"], "me/copy")

    def test_copy_forced_private(self):
        src = dict(self.src, is_private=False)
        self.assertTrue(ak.copy_metadata(src, "me/copy", "c.ipynb")["is_private"])

    def test_regime_diff_detects_a_swapped_model(self):
        b = dict(self.src, model_sources=["someone/other-model/1"])
        d = ak.regime_diff(self.src, b)
        self.assertIn("model_sources", d)
        self.assertNotIn("docker_image", d)

    def test_regime_diff_empty_when_same(self):
        self.assertEqual(ak.regime_diff(self.src, dict(self.src)), {})


class StageTests(unittest.TestCase):
    """stage() with a fake kaggle CLI, so the pipeline is tested offline."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _runner(self, cmd, **kw):
        dest = Path(cmd[cmd.index("-p") + 1])
        assert "-k" not in cmd, "slug is positional in the kaggle CLI, not -k"
        assert "/" in cmd[cmd.index("pull") + 1], "slug must follow 'pull' positionally"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(KEITH / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb",
                     dest / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb")
        shutil.copy2(KEITH / "kernel-metadata.json", dest / "kernel-metadata.json")

        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()

    def test_stage_is_byte_identical_and_attests(self):
        att = ak.stage("keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp",
                       "arc3-absorb-test", self.tmp, runner=self._runner,
                       now="2026-10-01T00:00Z")
        recorded = json.loads((KEITH / "ATTEST.json").read_text())
        self.assertEqual(att["file_sha256"], recorded["file_sha256"])
        self.assertEqual(att["code_cell_sha256"], recorded["code_cell_sha256"])
        self.assertTrue(att["byte_identical_notebook"])
        self.assertEqual(att["sources"]["docker"], recorded["sources"]["docker"])

        pushed = self.tmp / "push" / "arc3-absorb-test.ipynb"
        self.assertTrue(pushed.exists())
        self.assertEqual(pushed.read_bytes(),
                         (KEITH / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_bytes())

    def test_verify_passes_then_fails_after_an_edit(self):
        ak.stage("k/s", "arc3-absorb-test", self.tmp, runner=self._runner)
        self.assertTrue(ak.verify(self.tmp)["file_sha256_matches"])
        nb = self.tmp / "push" / "arc3-absorb-test.ipynb"
        nb.write_bytes(nb.read_bytes() + b" ")
        v = ak.verify(self.tmp)
        self.assertFalse(v["file_sha256_matches"])

    def test_stage_handles_a_script_kernel(self):
        """09-14 rehearsal: amanatar/arc-agi-3-hybrid-repl-agent is a `kernel_type: script`
        kernel (one .py, no .ipynb). The copy must be byte-identical with the same suffix and
        the attestation must hash the script as a one-cell notebook."""
        script = b"import os\nprint('hello')\n"

        def runner(cmd, **kw):
            dest = Path(cmd[cmd.index("-p") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "their-script.py").write_bytes(script)
            (dest / "kernel-metadata.json").write_text(json.dumps({
                "id": "k/s", "id_no": 7, "title": "s", "code_file": "their-script.py",
                "language": "python", "kernel_type": "script", "is_private": False,
                "enable_gpu": True, "dataset_sources": ["a/b"], "model_sources": ["m/n/1"],
                "docker_image": "gcr.io/x@sha256:abc", "machine_shape": "NvidiaRtxPro6000"}))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        att = ak.stage("k/s", "arc3-absorb-script", self.tmp, runner=runner, now="2026-09-14T18:00Z")
        copy = self.tmp / "push" / "arc3-absorb-script.py"
        self.assertTrue(copy.is_file() and copy.read_bytes() == script)
        self.assertEqual(att["kernel_type"], "script")
        self.assertEqual(att["n_code_cells"], 1)
        self.assertEqual(att["code_cell_sha256"], hashlib.sha256(script).hexdigest())
        meta = json.loads((self.tmp / "push" / "kernel-metadata.json").read_text())
        self.assertEqual(meta["code_file"], "arc3-absorb-script.py")
        self.assertEqual(meta["kernel_type"], "script")
        v = ak.verify(self.tmp)
        self.assertTrue(v["file_sha256_matches"] and v["code_cell_sha256_matches"])

    def test_pull_failure_is_loud(self):
        class R:
            returncode, stdout, stderr = 1, "", "403 Forbidden"

        with self.assertRaises(RuntimeError) as e:
            ak.stage("k/s", "x", self.tmp, runner=lambda *a, **k: R())
        self.assertIn("403", str(e.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
