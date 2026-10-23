"""Tests for scripts/profiler.py, focused on missing-value handling.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import profiler  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


class TestNoMissingValues(unittest.TestCase):
    def test_statistics_match_plain_numpy_exactly(self):
        X = np.random.default_rng(0).integers(0, 5, size=(50, 8)).astype(float)
        p = profiler.profile_dataset(X)
        self.assertFalse(p["has_missing"])
        self.assertEqual(p["n_missing_cells"], 0)
        self.assertEqual(p["missing_fraction"], 0.0)
        self.assertEqual(p["mean"], float(np.mean(X)))
        self.assertEqual(p["std"], float(np.std(X)))
        self.assertEqual(p["value_range"], [float(X.min()), float(X.max())])
        self.assertEqual(p["sparsity"], round(float(np.mean(X == 0)), 4))

    def test_no_missing_detail_keys_when_nothing_is_missing(self):
        p = profiler.profile_dataset(np.ones((4, 3)))
        for key in ("n_features_with_missing", "top_missing_features", "n_features_all_missing"):
            self.assertNotIn(key, p)

    def test_real_dataset_profile_is_unchanged_by_the_rewrite(self):
        committed = REPO / "outputs" / "pbmc" / "profile.json"
        raw = REPO / "data" / "pbmc" / "pbmc3k_raw.h5ad"
        if not (committed.exists() and raw.exists()):
            self.skipTest("pbmc data or committed profile not present")
        import loaders

        X, y, meta = loaders.load_dataset("pbmc")
        new = profiler.profile_dataset(X, y, meta)
        old = json.loads(committed.read_text())
        for key in ("n_samples", "n_features", "sparsity", "has_missing", "value_range", "mean", "std", "has_labels"):
            self.assertEqual(new[key], old[key], key)


class TestWithMissingValues(unittest.TestCase):
    def setUp(self):
        # 4 samples x 3 features; 3 of 12 cells missing, all in feature "b"/"c"
        self.X = np.array(
            [[1.0, np.nan, 0.0],
             [2.0, 4.0, 0.0],
             [3.0, np.nan, 6.0],
             [0.0, 8.0, np.nan]]
        )
        self.meta = {"feature_names": ["a", "b", "c"]}
        self.p = profiler.profile_dataset(self.X, metadata=self.meta)

    def test_statistics_are_over_observed_values_and_finite(self):
        observed = self.X[~np.isnan(self.X)]
        self.assertTrue(self.p["has_missing"])
        self.assertEqual(self.p["n_missing_cells"], 3)
        self.assertAlmostEqual(self.p["mean"], float(observed.mean()))
        self.assertAlmostEqual(self.p["std"], float(observed.std()))
        self.assertEqual(self.p["value_range"], [float(observed.min()), float(observed.max())])
        self.assertTrue(all(np.isfinite(v) for v in (self.p["mean"], self.p["std"], *self.p["value_range"])))

    def test_sparsity_is_zeros_among_observed_cells_only(self):
        # zeros: [0,2]=0, [1,2]=0, [3,0]=0 -> 3 zeros of 9 observed
        self.assertEqual(self.p["sparsity"], round(3 / 9, 4))

    def test_missing_counts_and_worst_features(self):
        self.assertEqual(self.p["n_features_with_missing"], 2)
        self.assertEqual(self.p["n_samples_with_missing"], 3)
        self.assertEqual(self.p["n_features_all_missing"], 0)
        self.assertEqual(self.p["max_feature_missing_fraction"], 0.5)
        self.assertEqual(self.p["top_missing_features"][0], {"feature": "b", "missing_fraction": 0.5})
        self.assertEqual(self.p["top_missing_features"][1], {"feature": "c", "missing_fraction": 0.25})
        self.assertEqual(len(self.p["top_missing_features"]), 2)  # fully observed features are not listed

    def test_profile_is_valid_json_with_no_nan_literal(self):
        json.dumps(self.p, allow_nan=False)  # raises ValueError if any nan/inf slipped in

    def test_feature_indices_are_used_when_there_are_no_names(self):
        p = profiler.profile_dataset(self.X)
        self.assertEqual(p["top_missing_features"][0]["feature"], 1)

    def test_entirely_missing_feature_is_counted(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        p = profiler.profile_dataset(X)
        self.assertEqual(p["n_features_all_missing"], 1)
        self.assertEqual(p["max_feature_missing_fraction"], 1.0)

    def test_top_list_is_capped(self):
        X = np.full((3, 20), np.nan)
        X[0, :] = 1.0  # every feature 2/3 missing
        p = profiler.profile_dataset(X)
        self.assertEqual(len(p["top_missing_features"]), profiler.TOP_MISSING_FEATURES)


class TestEveryValueMissing(unittest.TestCase):
    def test_no_crash_and_valid_json_with_nulls(self):
        p = profiler.profile_dataset(np.full((3, 2), np.nan))
        self.assertIsNone(p["mean"])
        self.assertIsNone(p["std"])
        self.assertIsNone(p["value_range"])
        self.assertIsNone(p["sparsity"])
        self.assertEqual(p["missing_fraction"], 1.0)
        json.dumps(p, allow_nan=False)


class TestRendering(unittest.TestCase):
    def test_summary_markdown_reports_missing_values_and_worst_feature(self):
        X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, np.nan]])
        p = profiler.profile_dataset(X, metadata={"feature_names": ["f1", "f2"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.md"
            profiler.write_profile_summary_md("demo", p, path)
            text = path.read_text()
        self.assertIn("Missing values:** Yes", text)
        self.assertIn("worst feature 'f2'", text)
        self.assertIn("Most-missing features", text)

    def test_summary_and_print_do_not_crash_when_stats_are_null(self):
        p = profiler.profile_dataset(np.full((2, 2), np.nan))
        with tempfile.TemporaryDirectory() as tmp:
            profiler.write_profile_summary_md("demo", p, Path(tmp) / "s.md")
        profiler.print_profile("demo", p)

    def test_no_missing_wording_is_unchanged(self):
        p = profiler.profile_dataset(np.ones((3, 2)))
        self.assertEqual(profiler._missing_text(p), "No")


if __name__ == "__main__":
    unittest.main()
