"""Small convolutional generative models for images, all optionally class-conditional:

   ConvVAE        conv encoder/decoder VAE; bernoulli (pixels in [0, 1]) or gaussian likelihood
   CondGAN        DCGAN-style generator + spectral-norm projection discriminator, hinge loss
   CondDiffusion  DDPM-trained noise-prediction UNet with classifier-free guidance, DDIM sampling

   Images arrive flattened in HWC order with values in [0, 1] (scale_unit_range); each model reshapes
   to NCHW internally and returns flattened HWC samples in [0, 1]. Height and width must be divisible
   by 4 (two stride-2 downsamplings).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_nchw(x, shape):
    h, w, c = shape
    return x.view(-1, h, w, c).permute(0, 3, 1, 2).contiguous()


def to_flat(x):
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)


def check_shape(shape):
    if len(shape) != 3 or shape[0] % 4 or shape[1] % 4:
        raise ValueError(f"conv models need (H, W, C) with H and W divisible by 4, got {shape}")


# ── Conditional VAE ──

class ConvVAE(nn.Module):
    def __init__(self, image_shape, latent_dim, base_channels, likelihood, n_classes=0, hidden=256):
        super().__init__()
        check_shape(image_shape)
        self.shape, self.latent_dim, self.likelihood, self.n_classes = tuple(image_shape), latent_dim, likelihood, n_classes
        h, w, c = self.shape
        b = base_channels
        self.feat = (2 * b, h // 4, w // 4)
        n_feat = 2 * b * (h // 4) * (w // 4)
        self.enc = nn.Sequential(nn.Conv2d(c, b, 4, 2, 1), nn.ReLU(), nn.Conv2d(b, 2 * b, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.enc_fc = nn.Sequential(nn.Linear(n_feat + n_classes, hidden), nn.ReLU())
        self.mu_head, self.logvar_head = nn.Linear(hidden, latent_dim), nn.Linear(hidden, latent_dim)
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim + n_classes, hidden), nn.ReLU(), nn.Linear(hidden, n_feat), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(2 * b, b, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(b, c, 4, 2, 1))
        if likelihood == "gaussian":
            self.log_var = nn.Parameter(torch.full((h * w * c,), -3.0))

    def _c(self, c):
        return [c] if self.n_classes else []

    def encode(self, x, c=None):
        h = self.enc(to_nchw(x, self.shape))
        h = self.enc_fc(torch.cat([h, *self._c(c)], 1))
        return self.mu_head(h), self.logvar_head(h).clamp(-10, 10)

    def decode_logits(self, z, c=None):
        h = self.dec_fc(torch.cat([z, *self._c(c)], 1)).view(-1, *self.feat)
        return to_flat(self.dec(h))  # (batch, H*W*C)

    def log_lik(self, x, out):
        if self.likelihood == "bernoulli":
            # Bernoulli cross-entropy on continuous [0, 1] intensities: a standard training objective for
            # image VAEs, though not a normalized density for non-binary pixels.
            return -F.binary_cross_entropy_with_logits(out, x, reduction="none").sum(1)
        mean = torch.sigmoid(out)
        return (-0.5 * (math.log(2 * math.pi) + self.log_var + (x - mean) ** 2 / torch.exp(self.log_var))).sum(1)

    def forward(self, x, c=None, beta=1.0):
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        ll = self.log_lik(x, self.decode_logits(z, c))
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)
        return (-(ll - beta * kl)).mean(), ll.mean(), kl.mean()

    @torch.no_grad()
    def elbo(self, x, c=None, n_samples=1):
        mu, logvar = self.encode(x, c)
        lls = [self.log_lik(x, self.decode_logits(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar), c)) for _ in range(n_samples)]
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)
        return torch.stack(lls).mean(0) - kl

    @torch.no_grad()
    def sample(self, n, c=None, z=None):
        """Returns the decoder MEAN image for each latent draw (pixel noise from the likelihood would only add
        independent per-pixel speckle), flattened HWC in [0, 1]. z defaults to prior draws; generate.py passes
        draws from a density fit to the training latents instead."""
        z = z if z is not None else torch.randn(n, self.latent_dim, device=self.mu_head.weight.device)
        return torch.sigmoid(self.decode_logits(z, c))


# ── Conditional GAN ──

class Generator(nn.Module):
    def __init__(self, image_shape, latent_dim, base_channels, n_classes):
        super().__init__()
        h, w, c = image_shape
        b = base_channels
        self.shape, self.latent_dim, self.n_classes = tuple(image_shape), latent_dim, n_classes
        self.emb = nn.Embedding(max(n_classes, 1), latent_dim)
        self.fc = nn.Sequential(nn.Linear(latent_dim, 2 * b * (h // 4) * (w // 4)), nn.BatchNorm1d(2 * b * (h // 4) * (w // 4)), nn.ReLU())
        # Nearest-neighbor upsampling + 3x3 conv instead of transposed convolutions, which leave
        # checkerboard artifacts (Odena et al., 2016).
        self.net = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, c, 3, padding=1), nn.Sigmoid())
        self.feat = (2 * b, h // 4, w // 4)

    def forward(self, z, y):
        h = z * self.emb(y) if self.n_classes else z  # multiplicative class conditioning
        return to_flat(self.net(self.fc(h).view(-1, *self.feat)))


class Discriminator(nn.Module):
    def __init__(self, image_shape, base_channels, n_classes):
        super().__init__()
        h, w, c = image_shape
        b = base_channels
        sn = nn.utils.spectral_norm
        self.shape, self.n_classes = tuple(image_shape), n_classes
        self.net = nn.Sequential(sn(nn.Conv2d(c, b, 4, 2, 1)), nn.LeakyReLU(0.2), sn(nn.Conv2d(b, 2 * b, 4, 2, 1)), nn.LeakyReLU(0.2),
                                 nn.Flatten())
        n_feat = 2 * b * (h // 4) * (w // 4)
        self.out = sn(nn.Linear(n_feat, 1))
        self.proj = sn(nn.Embedding(max(n_classes, 1), n_feat))  # projection discriminator (Miyato & Koyama, 2018)

    def forward(self, x, y):
        f = self.net(to_nchw(x * 2 - 1, self.shape))
        out = self.out(f).squeeze(1)
        if self.n_classes:
            out = out + (self.proj(y) * f).sum(1)
        return out


class CondGAN(nn.Module):
    def __init__(self, image_shape, latent_dim, base_channels, n_classes=0):
        super().__init__()
        check_shape(image_shape)
        self.G = Generator(image_shape, latent_dim, base_channels, n_classes)
        self.D = Discriminator(image_shape, base_channels, n_classes)
        self.latent_dim, self.n_classes = latent_dim, n_classes

    @torch.no_grad()
    def sample(self, n, y=None):
        """Generator BatchNorm uses batch statistics here, as during training: with the stored running
        statistics (eval mode) samples were measurably worse (higher Frechet distance, weaker class conditioning)."""
        dev = next(self.G.parameters()).device
        y = y if y is not None else torch.zeros(n, dtype=torch.long, device=dev)
        was_training = self.G.training
        self.G.train()
        out = self.G(torch.randn(n, self.latent_dim, device=dev), y)
        self.G.train(was_training)
        return out


# ── Conditional diffusion ──

def _sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
        return self.c2(F.silu(self.n2(h))) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, channels, base, n_classes):
        super().__init__()
        e = 4 * base
        self.t_mlp = nn.Sequential(nn.Linear(base, e), nn.SiLU(), nn.Linear(e, e))
        self.cls = nn.Embedding(n_classes + 1, e)  # last index = "no class" for classifier-free guidance
        self.inp = nn.Conv2d(channels, base, 3, padding=1)
        self.d1, self.d2 = ResBlock(base, base, e), ResBlock(2 * base, 2 * base, e)
        self.down1, self.down2 = nn.Conv2d(base, 2 * base, 4, 2, 1), nn.Conv2d(2 * base, 2 * base, 4, 2, 1)
        self.mid = ResBlock(2 * base, 2 * base, e)
        self.up2, self.up1 = nn.ConvTranspose2d(2 * base, 2 * base, 4, 2, 1), nn.ConvTranspose2d(2 * base, base, 4, 2, 1)
        self.u2, self.u1 = ResBlock(4 * base, 2 * base, e), ResBlock(2 * base, base, e)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, channels, 3, padding=1))
        self.base = base

    def forward(self, x, t, y):
        e = self.t_mlp(_sinusoidal(t, self.base)) + self.cls(y)
        h0 = self.d1(self.inp(x), e)              # (B, b, H, W)
        h1 = self.d2(self.down1(h0), e)           # (B, 2b, H/2, W/2)
        m = self.mid(self.down2(h1), e)           # (B, 2b, H/4, W/4)
        u = self.u2(torch.cat([self.up2(m), h1], 1), e)
        u = self.u1(torch.cat([self.up1(u), h0], 1), e)
        return self.out(u)


class CondDiffusion(nn.Module):
    def __init__(self, image_shape, timesteps, base_channels, n_classes=0, p_uncond=0.1):
        super().__init__()
        check_shape(image_shape)
        self.shape, self.T, self.n_classes, self.p_uncond = tuple(image_shape), timesteps, n_classes, p_uncond
        self.net = UNet(image_shape[2], base_channels, max(n_classes, 1))
        betas = torch.linspace(1e-4, 0.02, timesteps)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))  # cumulative signal fraction per step

    def forward(self, x, y):
        """Noise-prediction MSE at a random timestep (standard DDPM objective). Data scaled to [-1, 1].
        Class labels are dropped to the 'no class' token with probability p_uncond (for guidance)."""
        x0 = to_nchw(x, self.shape) * 2 - 1
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.abar[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        null = torch.full_like(y, self.net.cls.num_embeddings - 1)
        y_in = torch.where(torch.rand(y.shape, device=y.device) < self.p_uncond, null, y) if self.n_classes else null
        return F.mse_loss(self.net(xt, t, y_in), noise)

    @torch.no_grad()
    def sample(self, n, y=None, steps=50, guidance=2.0):
        """DDIM sampling (eta = 0) with classifier-free guidance; returns flattened HWC in [0, 1]."""
        dev = self.abar.device
        h, w, c = self.shape
        null = torch.full((n,), self.net.cls.num_embeddings - 1, dtype=torch.long, device=dev)
        y = y if (y is not None and self.n_classes) else null
        x = torch.randn(n, c, h, w, device=dev)
        ts = torch.linspace(self.T - 1, 0, steps).long().tolist()
        for i, t in enumerate(ts):
            tt = torch.full((n,), t, device=dev, dtype=torch.long)
            eps = self.net(x, tt, y)
            if self.n_classes and guidance != 1.0:
                eps_u = self.net(x, tt, null)                 # unconditional prediction
                eps = eps_u + guidance * (eps - eps_u)        # classifier-free guidance
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return to_flat((x.clamp(-1, 1) + 1) / 2)
