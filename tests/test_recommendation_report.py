"""Tests for validate_recommendation.py (metric citations, privacy wording) and report.py's number tracing.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from report import untraced_numbers  # noqa: E402
from validate_recommendation import validate_recommendation  # noqa: E402

R = "the latent space preserves held-out neighborhoods about as well as PCA"
OBJ = {"raw_text": "learn a representation", "effective_task": "representation_learning"}


def use(**kw):
    u = {"task": "representation_learning", "use": "Use the latent codes as clustering input for these cells",
         "model_key": "m_s0", "verdict": "conditional",
         "evidence": [{"metric": "m_s0.representation.trustworthiness", "value": 0.8073}],
         "rationale": R, "caveats": []}
    u.update(kw)
    return u


class TestRecommendation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mdir = Path(self._tmp.name)
        (self.mdir / "m_s0.json").write_text(json.dumps({"representation": {"trustworthiness": 0.8073}}))

    def tearDown(self):
        self._tmp.cleanup()

    def errs(self, rec, obj=OBJ):
        return validate_recommendation(rec, obj, self.mdir)[0]

    def test_valid(self):
        self.assertEqual(self.errs({"uses": [use()]}), [])

    def test_wrong_quoted_value(self):
        e = self.errs({"uses": [use(evidence=[{"metric": "m_s0.representation.trustworthiness", "value": 0.91}])]})
        self.assertTrue(any("metrics say" in x for x in e), e)

    def test_rounded_quote_is_accepted(self):
        self.assertEqual(self.errs({"uses": [use(evidence=[{"metric": "m_s0.representation.trustworthiness", "value": 0.81}])]}), [])

    def test_unknown_metric(self):
        e = self.errs({"uses": [use(evidence=[{"metric": "m_s0.fidelity.c2st_auc", "value": 0.5}])]})
        self.assertTrue(any("not found" in x for x in e), e)

    def test_recommended_needs_evidence(self):
        e = self.errs({"uses": [use(verdict="recommended", evidence=[])]})
        self.assertTrue(any("at least one metric" in x for x in e), e)

    def test_primary_task_must_be_covered(self):
        e = self.errs({"uses": [use(task="data_generation")]})
        self.assertTrue(any("primary task" in x for x in e), e)

    def test_privacy_claim_rejected(self):
        e = self.errs({"uses": [use(rationale="synthetic cells are anonymized and safe to share with collaborators")]})
        self.assertTrue(any("privacy claim" in x for x in e), e)

    def test_sharing_needs_disclaimer(self):
        obj = dict(OBJ, raw_text="generate synthetic cells for sharing")
        e = self.errs({"uses": [use()]}, obj)
        self.assertTrue(any("privacy_statement" in x for x in e), e)
        ok = {"uses": [use()], "privacy_statement": "This is not a formal privacy guarantee such as differential privacy."}
        self.assertEqual(self.errs(ok, obj), [])


class TestNumberTracing(unittest.TestCase):
    def artifacts(self):
        return {"profile": {"n_samples": 2700}, "meta": {}, "run_log": {}, "plan": {}, "recommendation": None,
                "metrics": {"k": {"fidelity": {"c2st_auc": 0.9457, "recall": 0.7667}}}, "runs": {}}

    def test_traced_numbers_pass(self):
        self.assertEqual(untraced_numbers("AUC 0.9457, rounded 0.95, recall 76.67%", self.artifacts()), [])

    def test_untraced_number_flagged(self):
        self.assertEqual(untraced_numbers("AUC 0.9457 and an invented 0.731", self.artifacts()), ["0.731"])


if __name__ == "__main__":
    unittest.main()
