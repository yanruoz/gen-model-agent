"""Tests for the Project 2 additions to loaders.py (image_shape) and profiler.py
(modality, non-negative-integer flag, class balance).

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import loaders  # noqa: E402
import profiler  # noqa: E402


class TestImageShape(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name) / "imgs"
        self.folder.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, n_features, shape_line):
        # Flattened 2x2 "images" in a CSV, plus a label column
        rows = ["label," + ",".join(f"p{i}" for i in range(n_features))]
        rows += [f"{k % 2}," + ",".join(str(k + i) for i in range(n_features)) for k in range(4)]
        (self.folder / "x.csv").write_text("\n".join(rows) + "\n")
        (self.folder / "DATA_DESCRIPTION.md").write_text(f"# d\n\n## Loading\nlabel_column: label\n{shape_line}\n")

    def test_declared_image_shape_is_parsed(self):
        self._write(4, "image_shape: 2,2,1")
        X, y, meta = loaders.load_generic(self.folder)
        self.assertEqual(meta["image_shape"], (2, 2, 1))
        self.assertEqual(profiler.profile_dataset(X, y, meta)["modality"], "image")

    def test_x_separator_also_accepted(self):
        self._write(4, "image_shape: 2x2")
        _, _, meta = loaders.load_generic(self.folder)
        self.assertEqual(meta["image_shape"], (2, 2))

    def test_shape_that_does_not_match_feature_count_is_refused(self):
        self._write(4, "image_shape: 3,3")
        with self.assertRaisesRegex(ValueError, "does not multiply"):
            loaders.load_generic(self.folder)


class TestProfileFields(unittest.TestCase):
    def test_tabular_without_labels(self):
        X = np.random.default_rng(0).normal(size=(50, 3))  # (50, 3) continuous
        p = profiler.profile_dataset(X, None, {})
        self.assertEqual(p["modality"], "tabular")
        self.assertFalse(p["is_nonneg_integer"])
        self.assertNotIn("imbalance_ratio", p)

    def test_counts_and_class_balance(self):
        X = np.random.default_rng(0).poisson(2.0, size=(40, 5)).astype(float)  # (40, 5) counts
        y = np.array(["a"] * 30 + ["b"] * 10)
        p = profiler.profile_dataset(X, y, {})
        self.assertTrue(p["is_nonneg_integer"])
        self.assertEqual(p["class_counts"], {"a": 30, "b": 10})
        self.assertEqual(p["imbalance_ratio"], 3.0)
        self.assertEqual(p["minority_class"], "b")


if __name__ == "__main__":
    unittest.main()
