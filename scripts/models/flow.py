"""RealNVP normalizing flow (affine coupling layers) for low-dimensional tabular data, e.g. PCA
   scores. Gives an exact log-likelihood, which a VAE (lower bound) and a GAN (none) don't.

   Inputs are standardized inside the model (mean/std fit on training data and stored as buffers),
   so log_prob is reported in the original input units (the Jacobian of the scaling is included).
"""
import math

import torch
import torch.nn as nn


class Coupling(nn.Module):
    """Transforms the unmasked half of x conditioned on the masked half."""

    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)  # start at the identity map for stable early training
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_masked):
        s, t = self.net(x_masked).chunk(2, dim=1)
        s = torch.tanh(s)  # bounded log-scale keeps training stable
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, x):          # data -> latent
        s, t = self._st(x * self.mask)
        return x * torch.exp(s) + t, s.sum(1)

    def inverse(self, z):          # latent -> data
        s, t = self._st(z * self.mask)
        return (z - t) * torch.exp(-s)


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers, hidden_dim):
        super().__init__()
        masks = [(torch.arange(dim) % 2 == i % 2).float() for i in range(n_layers)]
        self.layers = nn.ModuleList(Coupling(dim, hidden_dim, m) for m in masks)
        self.register_buffer("x_mean", torch.zeros(dim))
        self.register_buffer("x_std", torch.ones(dim))
        self.dim = dim

    def set_normalization(self, x):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))

    def log_prob(self, x):
        """Exact per-sample log-density in input units. (batch,)"""
        z = (x - self.x_mean) / self.x_std
        log_det = -torch.log(self.x_std).sum().expand(x.shape[0])
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        base = -0.5 * (z.pow(2).sum(1) + self.dim * math.log(2 * math.pi))
        return base + log_det

    def forward(self, x):
        """Loss = mean negative log-likelihood."""
        return -self.log_prob(x).mean()

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.dim, device=self.x_mean.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z * self.x_std + self.x_mean  # (n, dim)
