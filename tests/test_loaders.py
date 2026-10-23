"""Tests for the generic loader in scripts/loaders.py.

Run from the repo root:  python -m unittest discover -s tests -v
Uses temp directories (loaders.DATA_ROOT is redirected), so no real data is touched
except one regression check that the built-in pbmc loader still returns what it did.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import doctor  # noqa: E402
import fetch_data  # noqa: E402
import loaders  # noqa: E402


class LoaderTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_root = loaders.DATA_ROOT
        self._orig_max = loaders.MAX_DENSE_BYTES
        loaders.DATA_ROOT = self.root

    def tearDown(self):
        loaders.DATA_ROOT = self._orig_root
        loaders.MAX_DENSE_BYTES = self._orig_max
        self._tmp.cleanup()

    def make(self, name, description=None):
        folder = self.root / name
        folder.mkdir()
        if description is not None:
            (folder / "DATA_DESCRIPTION.md").write_text(description)
        return folder


class TestParseLoadingSpec(LoaderTestCase):
    def test_no_file_or_no_section_gives_empty_spec(self):
        self.assertEqual(loaders.parse_loading_spec(self.root / "missing.md"), {})
        folder = self.make("d", "# Title\n\n## What this is\nSomething.\n")
        self.assertEqual(loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md"), {})

    def test_plain_bullets_and_backticks_are_accepted(self):
        folder = self.make("d", "## Loading\n- file: `x.csv`\n* label_column: \"grp\"\nid_column: sample\n\n## Next\nfile: ignored.csv\n")
        spec = loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md")
        self.assertEqual(spec, {"file": "x.csv", "label_column": "grp", "id_column": "sample"})

    def test_url_and_md5_keys_are_parsed_including_digits_and_colons(self):
        folder = self.make("d", "## Loading\nfile: a.npz\nurl: https://example.org/a.npz?download=1\nmd5: 0123456789abcdef0123456789abcdef\n")
        spec = loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md")
        self.assertEqual(spec["url"], "https://example.org/a.npz?download=1")
        self.assertEqual(spec["md5"], "0123456789abcdef0123456789abcdef")

    def test_prose_with_a_colon_is_ignored_not_an_error(self):
        folder = self.make("d", "## Loading\nNote: the file is large.\nfile: x.csv\n")
        self.assertEqual(loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md"), {"file": "x.csv"})

    def test_misspelled_key_is_an_error_not_silently_dropped(self):
        folder = self.make("d", "## Loading\nlable_column: grp\n")
        with self.assertRaisesRegex(ValueError, "Unknown setting 'lable_column'"):
            loaders.parse_loading_spec(folder / "DATA_DESCRIPTION.md")


class TestTabular(LoaderTestCase):
    def test_csv_with_label_and_id_columns(self):
        folder = self.make("t", "## Loading\nlabel_column: grp\nid_column: sid\n")
        pd.DataFrame({"sid": ["a", "b", "c"], "f1": [1.0, 2.0, 3.0], "f2": [4, 5, 6], "grp": ["x", "y", "x"]}).to_csv(
            folder / "data.csv", index=False
        )
        X, y, meta = loaders.load_dataset("t")
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(list(y), ["x", "y", "x"])
        self.assertEqual(meta["feature_names"], ["f1", "f2"])
        self.assertEqual(meta["modality"], "tabular")

    def test_tsv_and_semicolon_delimiters_are_detected_without_a_description(self):
        folder = self.make("t")
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(folder / "d.tsv", sep="\t", index=False)
        X, y, _ = loaders.load_dataset("t")
        self.assertEqual(X.shape, (2, 2))
        self.assertIsNone(y)

        folder2 = self.make("t2")
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]}).to_csv(folder2 / "d.csv", sep=";", index=False)
        X2, _, _ = loaders.load_dataset("t2")
        self.assertEqual(X2.shape, (3, 2))

    def test_latin1_file_is_read(self):
        folder = self.make("t")
        (folder / "d.csv").write_bytes("caf\xe9,b\n1,2\n3,4\n".encode("latin-1"))
        X, _, meta = loaders.load_dataset("t")
        self.assertEqual(X.shape, (2, 2))

    def test_label_is_never_guessed(self):
        folder = self.make("t")
        pd.DataFrame({"a": [1.0, 2.0], "label": [0, 1]}).to_csv(folder / "d.csv", index=False)
        X, y, _ = loaders.load_dataset("t")
        self.assertEqual(X.shape, (2, 2))  # 'label' is just another feature unless declared
        self.assertIsNone(y)

    def test_declared_columns_that_do_not_exist_are_errors(self):
        folder = self.make("t", "## Loading\nlabel_column: nope\n")
        pd.DataFrame({"a": [1.0, 2.0]}).to_csv(folder / "d.csv", index=False)
        with self.assertRaisesRegex(ValueError, "not a column"):
            loaders.load_dataset("t")

    def test_non_numeric_feature_column_fails_loudly(self):
        folder = self.make("t")
        pd.DataFrame({"a": [1.0, 2.0], "color": ["red", "blue"]}).to_csv(folder / "d.csv", index=False)
        with self.assertRaisesRegex(ValueError, "non-numeric feature columns"):
            loaders.load_dataset("t")


class TestMissingTokens(LoaderTestCase):
    def test_common_missing_spellings_become_nan_in_feature_columns(self):
        folder = self.make("t")
        (folder / "d.csv").write_text("a,b,c\n1,?,3\n-,5,--\nna,6,none\n4,missing,.\n7,8,9\n")
        X, _, _ = loaders.load_dataset("t")
        self.assertEqual(X.shape, (5, 3))
        self.assertEqual(int(np.isnan(X).sum()), 7)
        self.assertEqual(X[4].tolist(), [7, 8, 9])  # a fully observed row is untouched

    def test_pandas_default_tokens_still_work(self):
        folder = self.make("t")
        (folder / "d.csv").write_text("a,b\n1,\n2,NaN\n3,NA\n")
        X, _, _ = loaders.load_dataset("t")
        self.assertEqual(int(np.isnan(X[:, 1]).sum()), 3)

    def test_label_and_id_columns_are_not_touched_by_the_extra_tokens(self):
        folder = self.make("t", "## Loading\nlabel_column: grp\nid_column: sid\n")
        (folder / "d.csv").write_text("sid,f1,grp\n-,1,none\n?,2,-\nna,3,?\n")
        X, y, _ = loaders.load_dataset("t")
        self.assertEqual(list(y), ["none", "-", "?"])
        self.assertFalse(np.isnan(X).any())

    def test_a_column_that_is_mostly_text_is_still_refused(self):
        folder = self.make("t")
        (folder / "d.csv").write_text("a,b\n1,red\n2,blue\n3,?\n")
        with self.assertRaisesRegex(ValueError, "non-numeric feature columns"):
            loaders.load_dataset("t")


class TestArrays(LoaderTestCase):
    def test_npy_2d_and_higher_dim_is_flattened_with_shape_recorded(self):
        folder = self.make("a")
        np.save(folder / "x.npy", np.arange(12.0).reshape(4, 3))
        X, y, meta = loaders.load_dataset("a")
        self.assertEqual(X.shape, (4, 3))
        self.assertNotIn("image_shape", meta)

        folder2 = self.make("a2")
        np.save(folder2 / "x.npy", np.zeros((5, 4, 4, 3)))
        X2, _, meta2 = loaders.load_dataset("a2")
        self.assertEqual(X2.shape, (5, 48))
        self.assertEqual(tuple(meta2["image_shape"]), (4, 4, 3))

    def test_1d_array_is_rejected(self):
        folder = self.make("a")
        np.save(folder / "x.npy", np.arange(5.0))
        with self.assertRaisesRegex(ValueError, "at least 2-D"):
            loaders.load_dataset("a")

    def test_npz_default_keys_and_column_vector_labels(self):
        folder = self.make("z")
        np.savez(folder / "d.npz", X=np.random.default_rng(0).random((6, 3)), y=np.arange(6).reshape(6, 1))
        X, y, _ = loaders.load_dataset("z")
        self.assertEqual(X.shape, (6, 3))
        self.assertEqual(y.shape, (6,))

    def test_npz_custom_keys_via_loading_section(self):
        folder = self.make("z", "## Loading\nx_key: imgs\ny_key: labs\n")
        np.savez(folder / "d.npz", imgs=np.zeros((4, 2, 2)), labs=np.array([0, 1, 0, 1]))
        X, y, meta = loaders.load_dataset("z")
        self.assertEqual(X.shape, (4, 4))
        self.assertEqual(list(y), [0, 1, 0, 1])

    def test_npz_missing_key_lists_what_exists(self):
        folder = self.make("z")
        np.savez(folder / "d.npz", train_images=np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, "train_images"):
            loaders.load_dataset("z")

    def test_npz_label_length_mismatch_is_an_error(self):
        folder = self.make("z")
        np.savez(folder / "d.npz", X=np.zeros((4, 2)), y=np.array([0, 1, 0]))
        with self.assertRaises(ValueError):
            loaders.load_dataset("z")


class TestH5ad(LoaderTestCase):
    def test_sparse_matrix_with_declared_label(self):
        import anndata as ad
        import scipy.sparse as sp

        folder = self.make("h", "## Loading\nlabel_column: cell_type\n")
        adata = ad.AnnData(
            X=sp.csr_matrix(np.eye(5, 4, dtype=np.float32)),
            obs=pd.DataFrame({"cell_type": list("aabba")}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        )
        adata.write_h5ad(folder / "d.h5ad")
        X, y, meta = loaders.load_dataset("h")
        self.assertEqual(X.shape, (5, 4))
        self.assertEqual(list(y), list("aabba"))
        self.assertEqual(meta["feature_names"][:2], ["g0", "g1"])

    def test_undeclared_label_is_none(self):
        import anndata as ad

        folder = self.make("h")
        ad.AnnData(X=np.ones((3, 2), dtype=np.float32)).write_h5ad(folder / "d.h5ad")
        _, y, _ = loaders.load_dataset("h")
        self.assertIsNone(y)


class TestFileSelectionAndLimits(LoaderTestCase):
    def test_no_data_file(self):
        self.make("e", "## What this is\nNothing here.\n")
        with self.assertRaisesRegex(ValueError, "No data file found"):
            loaders.load_dataset("e")

    def test_two_candidates_require_an_explicit_file(self):
        folder = self.make("m")
        pd.DataFrame({"a": [1.0, 2.0]}).to_csv(folder / "one.csv", index=False)
        pd.DataFrame({"a": [1.0, 2.0]}).to_csv(folder / "two.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Say which one"):
            loaders.load_dataset("m")
        (folder / "DATA_DESCRIPTION.md").write_text("## Loading\nfile: two.csv\n")
        X, _, meta = loaders.load_dataset("m")
        self.assertEqual(meta["source_file"], "two.csv")

    def test_declared_file_must_exist(self):
        self.make("m", "## Loading\nfile: gone.csv\n")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            loaders.load_dataset("m")

    def test_oversized_dense_matrix_is_refused_with_a_clear_message(self):
        folder = self.make("big")
        np.save(folder / "x.npy", np.zeros((100, 100)))
        loaders.MAX_DENSE_BYTES = 1000
        with self.assertRaisesRegex(ValueError, "dense .* float64 matrix"):
            loaders.load_dataset("big")

    def test_unknown_dataset_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset 'nope'"):
            loaders.load_dataset("nope")


class TestKnownDatasetsGetNoSpecialTreatment(unittest.TestCase):
    """pbmc and pathmnist load through the same generic path as any other dataset."""

    def test_there_is_no_registry_or_per_dataset_loader(self):
        self.assertFalse(hasattr(loaders, "LOADERS"))
        self.assertFalse(hasattr(loaders, "load_pbmc3k"))
        self.assertFalse(hasattr(loaders, "load_pathmnist"))

    def test_pbmc_via_generic_h5ad_path(self):
        if not (loaders.DATA_ROOT / "pbmc" / "pbmc3k_raw.h5ad").exists():
            self.skipTest("pbmc raw file not present locally (run scripts/fetch_data.py --dataset pbmc)")
        X, y, meta = loaders.load_dataset("pbmc")
        self.assertEqual(X.shape, (2700, 32738))
        self.assertIsNone(y)
        self.assertEqual(meta["source_file"], "pbmc3k_raw.h5ad")

    def test_pathmnist_via_generic_npz_path(self):
        if not (loaders.DATA_ROOT / "pathmnist" / "pathmnist.npz").exists():
            self.skipTest("pathmnist file not present locally (run scripts/fetch_data.py --dataset pathmnist)")
        X, y, meta = loaders.load_dataset("pathmnist")
        self.assertEqual(X.shape, (89996, 2352))
        self.assertEqual(len(np.unique(y)), 9)
        self.assertEqual(tuple(meta["image_shape"]), (28, 28, 3))


class TestFetchData(LoaderTestCase):
    def _source(self, content=b"hello,world\n1,2\n"):
        src = self.root / "source_file.csv"
        src.write_bytes(content)
        return src

    def _dataset(self, src, md5=None):
        import hashlib

        md5 = md5 if md5 is not None else hashlib.md5(src.read_bytes()).hexdigest()
        return self.make("f", f"## Loading\nfile: got.csv\nurl: {src.as_uri()}\nmd5: {md5}\n")

    def test_downloads_missing_file_and_verifies_md5(self):
        folder = self._dataset(self._source())
        self.assertIn("downloaded", fetch_data.fetch("f"))
        self.assertTrue((folder / "got.csv").exists())
        self.assertFalse((folder / "got.csv.part").exists())

    def test_existing_file_is_left_alone(self):
        folder = self._dataset(self._source())
        fetch_data.fetch("f")
        before = (folder / "got.csv").stat().st_mtime_ns
        self.assertIn("already present", fetch_data.fetch("f"))
        self.assertEqual((folder / "got.csv").stat().st_mtime_ns, before)

    def test_md5_mismatch_discards_the_download(self):
        folder = self._dataset(self._source(), md5="0" * 32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            fetch_data.fetch("f")
        self.assertFalse((folder / "got.csv").exists())
        self.assertFalse((folder / "got.csv.part").exists())

    def test_existing_file_with_wrong_md5_is_never_overwritten(self):
        folder = self._dataset(self._source(), md5="0" * 32)
        (folder / "got.csv").write_text("precious local edits")
        with self.assertRaisesRegex(ValueError, "Not overwriting"):
            fetch_data.fetch("f")
        self.assertEqual((folder / "got.csv").read_text(), "precious local edits")

    def test_failed_download_leaves_no_partial_file(self):
        folder = self.make("f", "## Loading\nfile: got.csv\nurl: file:///definitely/not/here.csv\n")
        with self.assertRaisesRegex(ValueError, "failed"):
            fetch_data.fetch("f")
        self.assertEqual(list(folder.glob("got.csv*")), [])

    def test_download_sends_a_non_default_user_agent(self):
        # Some hosts (e.g. exampledata.scverse.org) return HTTP 403 for urllib's default
        # "Python-urllib/x.y" User-Agent; the real request must not use it.
        import urllib.request
        seen = {}
        orig_urlopen = urllib.request.urlopen

        def spy(req, *a, **kw):
            seen["user_agent"] = req.get_header("User-agent")
            return orig_urlopen(req, *a, **kw)

        folder = self._dataset(self._source())
        urllib.request.urlopen = spy
        try:
            fetch_data.fetch("f")
        finally:
            urllib.request.urlopen = orig_urlopen
        self.assertEqual(seen.get("user_agent"), fetch_data.USER_AGENT)
        self.assertNotIn("python-urllib", seen.get("user_agent", "").lower())
        self.assertTrue((folder / "got.csv").exists())

    def test_refuses_non_http_file_schemes_and_missing_declarations(self):
        self.make("g", "## Loading\nfile: a.csv\nurl: ftp://example.org/a.csv\n")
        with self.assertRaisesRegex(ValueError, "must start with"):
            fetch_data.fetch("g")
        self.make("h", "## Loading\nfile: a.csv\n")
        with self.assertRaisesRegex(ValueError, "nothing to fetch"):
            fetch_data.fetch("h")


class TestDoctorDataFileCheck(LoaderTestCase):
    def test_missing_data_file_with_url_points_to_fetch_data(self):
        self.make("d", "## Loading\nfile: a.csv\nurl: https://example.org/a.csv\n")
        errors, checks = doctor.check_core("d")
        self.assertTrue(any("fetch_data.py --dataset d" in e for e in errors))

    def test_missing_data_file_without_url_gives_no_fetch_hint(self):
        self.make("d", "## Loading\nfile: a.csv\n")
        errors, _ = doctor.check_core("d")
        self.assertTrue(errors)
        self.assertFalse(any("fetch_data" in e for e in errors))

    def test_present_data_file_passes(self):
        folder = self.make("d", "## Loading\nfile: a.csv\n")
        (folder / "a.csv").write_text("x\n1\n")
        errors, _ = doctor.check_core("d")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
