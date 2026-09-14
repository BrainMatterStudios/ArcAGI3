import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transplant_serving as ts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JUNE = REPO / "submission" / "_absorb_june"


class TransplantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "push").mkdir()
        src = next(p for p in (JUNE / "push").iterdir() if p.suffix == ".ipynb")
        (self.tmp / "push" / src.name).write_bytes(src.read_bytes())
        (self.tmp / "push" / "kernel-metadata.json").write_bytes((JUNE / "push" / "kernel-metadata.json").read_bytes())

    def test_transplant_on_the_june_winner(self):
        att = ts.transplant(self.tmp, "arc3-transplant-test", now="2026-09-14T19:00Z")
        nb = json.loads((self.tmp / "push_transplant" / "arc3-transplant-test.ipynb").read_text())
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        joined = "\n".join(cells)
        self.assertEqual(att["substitutions"], {"bundle_pin": 1, "setup_env_bundle_dir": 1, "setup_commands_dir": 1})
        self.assertIn("BUNDLE_DIR = _find_bundle_dir_pinned()", joined)
        self.assertNotIn("BUNDLE_DIR = _find_bundle_dir()\n", joined)
        self.assertIn('SERVING_BUNDLE_DIR / "setup_commands.json"', joined)
        self.assertIn('env["TAAF_KAGGLE_BUNDLE_DIR"] = str(SERVING_BUNDLE_DIR)', joined)
        self.assertIn('"TAAF_VLLM_MTP_TOKENS": "0"', joined)
        self.assertIn('"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "10737418240"', joined)
        # the serving cell sits right after the discovery cell and before the setup-commands cell
        idx_find = next(i for i, c in enumerate(cells) if "_find_bundle_dir_pinned()" in c)
        idx_serv = next(i for i, c in enumerate(cells) if "SERVING_BUNDLE_DIR = _find_serving_bundle_dir()" in c)
        idx_cmd = next(i for i, c in enumerate(cells) if 'SERVING_BUNDLE_DIR / "setup_commands.json"' in c)
        self.assertTrue(idx_find < idx_serv < idx_cmd)
        # every cell still parses
        import ast
        for c in cells:
            ast.parse(c.replace("await bm.run", "bm.run"))
        meta = json.loads((self.tmp / "push_transplant" / "kernel-metadata.json").read_text())
        # metadata order is Kaggle's, not the notebook's; the agent bundle is pinned by the
        # notebook's own DATASET_SOURCES[0], so membership is what matters here
        self.assertIn("jeroencottaar/taaf-kaggle-source-share", meta["dataset_sources"])
        self.assertEqual(meta["dataset_sources"][-2:], ts.KEITH_DATASETS)
        self.assertEqual(meta["model_sources"], [ts.KEITH_MODEL])
        self.assertEqual(meta["docker_image"], ts.KEITH_DOCKER)
        self.assertTrue(meta["is_private"])
        # the staged copy itself is untouched
        src = next(p for p in (self.tmp / "push").iterdir() if p.suffix == ".ipynb")
        self.assertEqual(json.loads(src.read_text())["cells"].__len__() + 1, len(nb["cells"]))


if __name__ == "__main__":
    unittest.main()
