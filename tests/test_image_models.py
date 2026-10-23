"""Shape and behavior tests for the image models, the downstream test's target logic, and the image
plan checks. Tiny inputs on CPU only.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from models.classifier import SmallCNN  # noqa: E402
from models.conv import CondDiffusion, CondGAN, ConvVAE, check_shape  # noqa: E402
from validate_plan import validate_plan  # noqa: E402

SHAPE = (8, 8, 3)
R = "a specific reason that cites this dataset's profile numbers"


class TestConvModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(6, 8 * 8 * 3)
        self.y = torch.tensor([0, 1, 2, 0, 1, 2])

    def test_cvae(self):
        m = ConvVAE(SHAPE, 4, 8, "bernoulli", n_classes=3, hidden=16)
        c = torch.nn.functional.one_hot(self.y, 3).float()
        loss, _, _ = m(self.x, c)
        self.assertTrue(torch.isfinite(loss))
        s = m.sample(5, c[:5])
        self.assertEqual(tuple(s.shape), (5, 192))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_gan(self):
        m = CondGAN(SHAPE, 4, 8, n_classes=3)
        self.assertEqual(tuple(m.D(self.x, self.y).shape), (6,))
        m.G.eval()
        s = m.sample(5, self.y[:5])
        self.assertEqual(tuple(s.shape), (5, 192))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_diffusion(self):
        m = CondDiffusion(SHAPE, 20, 8, n_classes=3)
        self.assertTrue(torch.isfinite(m(self.x, self.y)))
        s = m.sample(4, self.y[:4], steps=5, guidance=2.0)
        self.assertEqual(tuple(s.shape), (4, 192))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_classifier_features(self):
        m = SmallCNN(SHAPE, 3)
        self.assertEqual(tuple(m.features(self.x).shape), (6, 128))

    def test_shape_check(self):
        with self.assertRaises(ValueError):
            check_shape((10, 10, 3))


def _img_plan(**kw):
    hp = {"timesteps": 100, "base_channels": 8, "sampling_steps": 20, "guidance_scale": 2.0, "epochs": 5, "batch_size": 32, "lr": 2e-4}
    m = {"id": "cdiff", "family": "diffusion", "conditional": True, "input_space": "transformed", "hyperparameters": hp,
         "hyperparameter_reasons": {k: R for k in hp}, "seeds": [0], "serves_tasks": ["class_balancing"],
         "estimated_minutes": 5, "status": "planned", "reason": R}
    m.update(kw)
    return {"dataset": "imgs", "seed": 0, "budget_min": 20, "revision": 1,
            "split": {"test_fraction": 0.2, "stratify": True, "reason": R},
            "preprocessing": [{"step": "scale_unit_range", "reason": R}], "models": [m]}


class TestImagePlanChecks(unittest.TestCase):
    PROFILE = {"n_samples": 20000, "n_features": 192, "has_labels": True, "modality": "image", "is_nonneg_integer": True,
               "value_range": [0.0, 255.0], "imbalance_ratio": 1.6, "class_counts": {"0": 1, "1": 1}}
    OBJ = {"effective_task": "class_balancing", "secondary_tasks": [], "decision": "proceed"}

    def test_valid_image_plan(self):
        self.assertEqual(validate_plan(_img_plan(), self.OBJ, self.PROFILE)[0], [])

    def test_image_model_needs_transformed_input_and_unit_range(self):
        errs = validate_plan(_img_plan(input_space="raw"), self.OBJ, self.PROFILE)[0]
        self.assertTrue(any("transformed" in e for e in errs), errs)
        p = _img_plan(); p["preprocessing"] = []
        errs = validate_plan(p, self.OBJ, self.PROFILE)[0]
        self.assertTrue(any("scale_unit_range" in e for e in errs), errs)

    def test_image_hyperparameters_required(self):
        p = _img_plan()
        del p["models"][0]["hyperparameters"]["guidance_scale"]
        errs = validate_plan(p, self.OBJ, self.PROFILE)[0]
        self.assertTrue(any("guidance_scale" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
