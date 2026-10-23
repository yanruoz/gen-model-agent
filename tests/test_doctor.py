"""Tests for scripts/doctor.py's virtualenv detection and crash-safe import checking.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import doctor  # noqa: E402


def fake_completed(returncode, stderr=""):
    return subprocess.CompletedProcess(args=["python", "-c", "..."], returncode=returncode, stdout="", stderr=stderr)


class TestInVirtualenv(unittest.TestCase):
    def test_true_when_prefix_differs_from_base_prefix(self):
        with patch.object(sys, "prefix", "/some/venv"), patch.object(sys, "base_prefix", "/usr"):
            self.assertTrue(doctor._in_virtualenv())

    def test_false_when_prefix_equals_base_prefix(self):
        with patch.object(sys, "prefix", "/usr"), patch.object(sys, "base_prefix", "/usr"):
            self.assertFalse(doctor._in_virtualenv())

    def test_missing_base_prefix_attribute_is_treated_as_not_isolated(self):
        # sys.base_prefix always exists on 3.3+, but guard the fallback logic directly:
        # deleting the attribute must not raise, and must read as "not isolated".
        real_sys = doctor.sys
        fake = types.SimpleNamespace(prefix="/usr")  # no base_prefix attribute at all
        with patch.object(doctor, "sys", fake):
            self.assertFalse(doctor._in_virtualenv())
        self.assertIs(doctor.sys, real_sys)


class TestCheckImports(unittest.TestCase):
    def test_a_real_importable_module_is_ok(self):
        results = doctor._check_imports(["json"])
        self.assertEqual(results, [{"module": "json", "ok": True}])

    def test_a_nonexistent_module_is_a_normal_not_ok_with_an_error_message(self):
        results = doctor._check_imports(["definitely_not_a_real_package_xyz"])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"])
        self.assertIn("error", results[0])
        self.assertNotIn("crashed", results[0]["error"])

    def test_a_dotted_submodule_name_is_supported(self):
        results = doctor._check_imports(["os.path"])
        self.assertEqual(results, [{"module": "os.path", "ok": True}])

    def test_uses_the_current_interpreter(self):
        with patch.object(doctor.subprocess, "run", return_value=fake_completed(0)) as mock_run:
            doctor._check_imports(["numpy"])
        called_cmd = mock_run.call_args.args[0]
        self.assertEqual(called_cmd[0], sys.executable)
        self.assertIn("import numpy", called_cmd[-1])

    def test_a_negative_returncode_is_reported_as_a_crash_naming_the_signal(self):
        with patch.object(doctor.subprocess, "run", return_value=fake_completed(-11)):  # SIGSEGV
            results = doctor._check_imports(["umap"])
        self.assertFalse(results[0]["ok"])
        self.assertIn("crashed", results[0]["error"])
        self.assertIn("SIGSEGV", results[0]["error"])

    def test_an_unrecognized_negative_returncode_still_reports_a_crash(self):
        with patch.object(doctor.subprocess, "run", return_value=fake_completed(-999)):
            results = doctor._check_imports(["umap"])
        self.assertFalse(results[0]["ok"])
        self.assertIn("crashed", results[0]["error"])
        self.assertIn("signal 999", results[0]["error"])

    def test_a_normal_positive_failure_uses_the_last_stderr_line_not_a_crash_message(self):
        with patch.object(doctor.subprocess, "run", return_value=fake_completed(1, stderr="Traceback (...)\nModuleNotFoundError: No module named 'x'\n")):
            results = doctor._check_imports(["x"])
        self.assertFalse(results[0]["ok"])
        self.assertEqual(results[0]["error"], "ModuleNotFoundError: No module named 'x'")
        self.assertNotIn("crashed", results[0]["error"])

    def test_multiple_modules_are_each_checked_independently(self):
        with patch.object(
            doctor.subprocess, "run",
            side_effect=[fake_completed(0), fake_completed(-11), fake_completed(1, stderr="err")],
        ):
            results = doctor._check_imports(["a", "b", "c"])
        self.assertEqual([r["ok"] for r in results], [True, False, False])
        self.assertIn("crashed", results[1]["error"])
        self.assertNotIn("crashed", results[2]["error"])


class TestCheckCoreIncludesVenvCheck(unittest.TestCase):
    def test_venv_check_is_present_and_never_contributes_an_error_either_way(self):
        for in_venv in (True, False):
            with patch.object(doctor, "_in_virtualenv", return_value=in_venv), \
                 patch.object(doctor, "_check_imports", return_value=[{"module": "numpy", "ok": True}]):
                errors, checks = doctor.check_core()
            venv_checks = [c for c in checks if c["module"].startswith("isolated venv")]
            self.assertEqual(len(venv_checks), 1)
            self.assertEqual(venv_checks[0]["ok"], in_venv)
            self.assertEqual(errors, [], in_venv)  # never an error, regardless of the venv state

    def test_a_real_missing_package_alongside_the_venv_check_still_reports_as_an_error(self):
        with patch.object(doctor, "_check_imports", return_value=[{"module": "numpy", "ok": False, "error": "boom"}]):
            errors, checks = doctor.check_core()
        self.assertTrue(any("numpy" in e and "boom" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
