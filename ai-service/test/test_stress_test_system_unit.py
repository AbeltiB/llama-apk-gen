import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / 'test'))

TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from stres_test_system import (
    _normalize_prompts,
    ensure_output_dirs,
    load_prompts_from_file,
    validate_export_payload,
)


class StressTestSystemUnitTests(unittest.TestCase):
    def test_validate_export_payload_missing_keys(self):
        errors = validate_export_payload({"code": "x"})
        self.assertTrue(any("Missing top-level keys" in e for e in errors))

    def test_normalize_prompts_deduplicates_and_filters_short(self):
        prompts = ["", "  hi  ", "Create a todo app", "create a todo app", "# comment"]
        normalized = _normalize_prompts(prompts, dedupe=True, min_len=10)
        self.assertEqual(normalized, ["Create a todo app"])

    def test_load_prompts_txt(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prompts.txt"
            p.write_text("#comment\nCreate a notes app\nCreate a notes app\n", encoding="utf-8")
            loaded = load_prompts_from_file(p)
            self.assertEqual(loaded, ["Create a notes app"])

    def test_load_prompts_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prompts.json"
            p.write_text(json.dumps({"prompts": ["Create calculator app", "Create timer app"]}), encoding="utf-8")
            loaded = load_prompts_from_file(p)
            self.assertEqual(len(loaded), 2)

    def test_ensure_output_dirs_creates_all(self):
        with tempfile.TemporaryDirectory() as td:
            dirs = ensure_output_dirs(Path(td))
            self.assertTrue(dirs["json"].exists())
            self.assertTrue(dirs["downloads"].exists())
            self.assertTrue(dirs["reports"].exists())


if __name__ == "__main__":
    unittest.main()
