"""Variational autoencoder for tabular data, with a choice of likelihood:

   nb        negative binomial on raw counts, scVI-style: the decoder outputs per-feature proportions
             (softmax) that are scaled by each sample's library size (total count), plus a learned
             per-feature dispersion. Suits sparse, over-dispersed count data such as scRNA-seq.
   gaussian  per-feature mean + learned per-feature log-variance; for continuous/transformed data.
   bernoulli for data in [0, 1].

   Optional class conditioning (one-hot label concatenated to encoder and decoder inputs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()]
    return nn.Sequential(*layers)


def nb_log_prob(x, mu, theta, eps=1e-8):
    """log NB(x | mean mu, inverse dispersion theta), elementwise. Shapes (batch, d)."""
    log_theta_mu = torch.log(theta + mu + eps)
    return (theta * (torch.log(theta + eps) - log_theta_mu)
            + x * (torch.log(mu + eps) - log_theta_mu)
            + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))


class VAE(nn.Module):
    def __init__(self, n_in, latent_dim, hidden_dims, likelihood, n_classes=0):
        super().__init__()
        self.n_in, self.latent_dim, self.likelihood, self.n_classes = n_in, latent_dim, likelihood, n_classes
        h = list(hidden_dims)
        self.encoder = _mlp([n_in + n_classes] + h)
        self.mu_head, self.logvar_head = nn.Linear(h[-1], latent_dim), nn.Linear(h[-1], latent_dim)
        self.decoder = _mlp([latent_dim + n_classes] + h[::-1])
        self.out = nn.Linear(h[0], n_in)
        if likelihood == "nb":
            self.log_theta = nn.Parameter(torch.zeros(n_in))       # per-feature inverse dispersion
        elif likelihood == "gaussian":
            self.log_var = nn.Parameter(torch.zeros(n_in))         # per-feature variance

    def _enc_input(self, x):
        # Counts span orders of magnitude; the encoder sees log1p(counts), the likelihood sees raw counts.
        return torch.log1p(x) if self.likelihood == "nb" else x

    def encode(self, x, c=None):
        h = self._enc_input(x)
        if self.n_classes:
            h = torch.cat([h, c], dim=1)
        h = self.encoder(h)
        return self.mu_head(h), self.logvar_head(h).clamp(-10, 10)  # (batch, latent) x2

    def decode(self, z, c=None, library=None):
        h = torch.cat([z, c], dim=1) if self.n_classes else z
        out = self.out(self.decoder(h))  # (batch, n_in)
        if self.likelihood == "nb":
            return {"mu": F.softmax(out, dim=1) * library, "theta": torch.exp(self.log_theta)}
        if self.likelihood == "gaussian":
            return {"mean": out, "log_var": self.log_var}
        return {"logits": out}

    def log_lik(self, x, params):
        """Per-sample log-likelihood, summed over features. (batch,)"""
        if self.likelihood == "nb":
            return nb_log_prob(x, params["mu"], params["theta"]).sum(1)
        if self.likelihood == "gaussian":
            lv = params["log_var"]
            return (-0.5 * (math.log(2 * math.pi) + lv + (x - params["mean"]) ** 2 / torch.exp(lv))).sum(1)
        return -F.binary_cross_entropy_with_logits(params["logits"], x, reduction="none").sum(1)

    def forward(self, x, c=None, beta=1.0):
        """Returns (loss, recon_ll, kl), each averaged over the batch. loss = -(ELBO with KL weight beta)."""
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        library = x.sum(1, keepdim=True) if self.likelihood == "nb" else None
        ll = self.log_lik(x, self.decode(z, c, library))                        # (batch,)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)               # (batch,)
        return (-(ll - beta * kl)).mean(), ll.mean(), kl.mean()

    @torch.no_grad()
    def elbo(self, x, c=None, n_samples=1):
        """Per-sample ELBO (full KL weight). (batch,)"""
        mu, logvar = self.encode(x, c)
        library = x.sum(1, keepdim=True) if self.likelihood == "nb" else None
        lls = []
        for _ in range(n_samples):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            lls.append(self.log_lik(x, self.decode(z, c, library)))
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)
        return torch.stack(lls).mean(0) - kl

    @torch.no_grad()
    def sample(self, n, c=None, library=None, z=None):
        """Draws n samples. z defaults to prior draws; generate.py passes draws from a density fit to the
        training latents instead. For nb, `library` (n, 1) gives each sample's total count.
        Returns (n, n_in) samples drawn from the likelihood (not just its mean)."""
        z = z if z is not None else torch.randn(n, self.latent_dim, device=self.mu_head.weight.device)
        p = self.decode(z, c, library)
        if self.likelihood == "nb":
            # Gamma-Poisson mixture = negative binomial with mean mu and inverse dispersion theta.
            theta = p["theta"].expand_as(p["mu"])
            rate = torch.distributions.Gamma(theta, theta / (p["mu"] + 1e-8)).sample()
            return torch.poisson(rate)
        if self.likelihood == "gaussian":
            return p["mean"] + torch.randn_like(p["mean"]) * torch.exp(0.5 * p["log_var"])
        return torch.bernoulli(torch.sigmoid(p["logits"]))
