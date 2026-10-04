"""Verify the downloadable ONNX acquisition tools in the 0.3.0 ZIP contract."""

import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_layout", REPO / "scripts/verify-release-layout.py")
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)


class OnnxReleaseToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        for relative in ("scripts/prepare-onnx-models.py", "release/ppocrv6-onnx-manifest.json"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)

    def mutate(self, change):
        path = self.root / "release/ppocrv6-onnx-manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        change(manifest)
        path.write_text(json.dumps(manifest), "utf-8")

    def test_current_tools_pass(self):
        LAYOUT.verify_onnx_tools(self.root)

    def test_missing_downloader_fails(self):
        (self.root / "scripts/prepare-onnx-models.py").unlink()
        with self.assertRaises(SystemExit):
            LAYOUT.verify_onnx_tools(self.root)

    def test_unpinned_url_fails(self):
        self.mutate(lambda data: data["assets"][0].update(url="https://example.org/latest.onnx"))
        with self.assertRaises(SystemExit):
            LAYOUT.verify_onnx_tools(self.root)

    def test_duplicate_path_fails(self):
        self.mutate(lambda data: data["assets"].__setitem__(1, data["assets"][0].copy()))
        with self.assertRaises(SystemExit):
            LAYOUT.verify_onnx_tools(self.root)

    def test_invalid_hash_fails(self):
        self.mutate(lambda data: data["assets"][0].update(sha256="not-a-checksum"))
        with self.assertRaises(SystemExit):
            LAYOUT.verify_onnx_tools(self.root)


if __name__ == "__main__":
    unittest.main()
