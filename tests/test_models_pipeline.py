"""Tests for preprocess.py transforms and evaluation-space sizing, the VAE likelihood, and the flow's
exact log-density. Small synthetic inputs only; no dataset files needed.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import preprocess  # noqa: E402
from models.flow import RealNVP  # noqa: E402
from models.vae import VAE, nb_log_prob  # noqa: E402


class TestTransforms(unittest.TestCase):
    def test_fit_on_train_apply_to_new_data(self):
        rng = np.random.default_rng(0)
        train = rng.poisson(3.0, size=(50, 8)).astype(float)
        fitted = preprocess.fit_transforms(train, ["normalize_total", "log1p", "standardize"])
        out = preprocess.apply_transforms(train, fitted)
        np.testing.assert_allclose(out.mean(0), 0, atol=1e-9)  # standardize fit on the same data -> mean 0
        new = preprocess.apply_transforms(rng.poisson(3.0, size=(5, 8)).astype(float), fitted)
        self.assertEqual(new.shape, (5, 8))

    def test_invert_log1p_and_unit_range(self):
        x = np.array([[0.0, 10.0], [255.0, 3.0]])
        fitted = preprocess.fit_transforms(x, ["scale_unit_range"])
        np.testing.assert_allclose(preprocess.invert_transforms(preprocess.apply_transforms(x, fitted), fitted), x)
        fitted = preprocess.fit_transforms(x, ["log1p"])
        np.testing.assert_allclose(preprocess.invert_transforms(preprocess.apply_transforms(x, fitted), fitted), x)


class TestStableComponents(unittest.TestCase):
    def test_pure_noise_keeps_few_components(self):
        # Isotropic noise in 300 dims with 400 samples: every PCA direction is fitted noise and overfits.
        x = np.random.default_rng(0).normal(size=(400, 300))
        k, _ = preprocess.stable_components(x, 50, seed=0)
        self.assertLess(k, 10)

    def test_gradual_spectrum_keeps_generalizing_components(self):
        # Many moderately strong directions (PBMC-like): the strong ones generalize, so k is well above 2.
        rng = np.random.default_rng(0)
        z = rng.normal(size=(2000, 40)) * np.linspace(4, 1, 40)
        x = z @ rng.normal(size=(40, 200)) / 10 + rng.normal(size=(2000, 200))
        k, _ = preprocess.stable_components(x, 50, seed=0)
        self.assertGreaterEqual(k, 10)


class TestVAE(unittest.TestCase):
    def test_nb_log_prob_matches_scipy(self):
        x = torch.tensor([[0.0, 3.0, 10.0]]); mu = torch.tensor([[1.5, 2.0, 8.0]]); theta = torch.tensor([2.0, 0.5, 5.0])
        ours = nb_log_prob(x, mu, theta).numpy()[0]
        p = (theta / (theta + mu)).numpy()[0]  # scipy: n = theta, p = theta / (theta + mu)
        ref = stats.nbinom.logpmf(x.numpy()[0], theta.numpy(), p)
        np.testing.assert_allclose(ours, ref, rtol=1e-4)

    def test_nb_vae_forward_and_sample_shapes(self):
        torch.manual_seed(0)
        m = VAE(20, 4, [16], "nb")
        x = torch.poisson(torch.full((8, 20), 2.0))
        loss, ll, kl = m(x)
        self.assertTrue(torch.isfinite(loss))
        s = m.sample(5, library=torch.full((5, 1), 40.0))
        self.assertEqual(tuple(s.shape), (5, 20))
        self.assertTrue(torch.all(s >= 0) and torch.all(s == s.round()))  # counts


class TestFlow(unittest.TestCase):
    def test_log_prob_at_init_is_scaled_gaussian(self):
        # Coupling layers start at the identity, so log p(x) = sum log N(x_j; mean_j, std_j).
        torch.manual_seed(0)
        x = torch.randn(64, 3) * torch.tensor([1.0, 2.0, 0.5]) + 3.0
        f = RealNVP(3, 4, 16); f.set_normalization(x)
        ref = torch.distributions.Normal(f.x_mean, f.x_std).log_prob(x).sum(1)
        torch.testing.assert_close(f.log_prob(x), ref, rtol=1e-5, atol=1e-5)

    def test_sample_inverts_forward(self):
        torch.manual_seed(0)
        f = RealNVP(4, 4, 16)
        for p in f.parameters():
            p.data.normal_(0, 0.1)  # move off the identity
        z = torch.randn(10, 4)
        x = z.clone()
        for layer in reversed(f.layers):
            x = layer.inverse(x)
        back = x
        for layer in f.layers:
            back, _ = layer(back)
        torch.testing.assert_close(back, z, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
