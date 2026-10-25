"""Draw synthetic samples from one trained model (and, for VAEs, encode real data into the latent space).

   Usage: venv/bin/python scripts/generate.py --dataset <d> --key <id>_s<seed> [--n <int>]

   Default n = size of the held-out split, so synthetic and held-out real sets are the same size for
   the two-sample metrics. Writes to outputs/<d>/samples/:
     <key>_trans.npy      samples in the shared transformed space (every model)     (n, d)
     <key>_raw.npy        raw-scale samples, when the model generates them (NB VAE)  (n, d)
     <key>_y.npy          requested class of each sample (conditional models)
     <key>_sampling.json  how the samples were drawn (shown in the report)
   and, for VAEs, outputs/<d>/latents/<key>_{train,test}.npy (posterior means)  (n, latent_dim)

   VAE latent sampling: new samples are decoded from draws of a Gaussian mixture fit to the TRAINING
   samples' latent codes (the aggregate posterior; "ex-post density estimation", Ghosh et al. 2020),
   not from the N(0, I) prior. Training codes do not fill the prior evenly, and decoding from its empty
   regions produced unrealistic in-between samples. For count models the mixture is fit jointly on
   [latent code, log library size], since total counts depend on the sample's state.

   Extra sample sets for objective tests (conditional models on labeled data):
     class_balancing / augmentation: target class(es) up to the largest training class
                                     (<key>_balance_trans.npy / _balance_y.npy)
     data_generation / augmentation: a training-sized labeled set for train-on-synthetic, test-on-real
                                     (<key>_tstr_trans.npy / _tstr_y.npy), training class mix, capped at TSTR_MAX
"""
import argparse
import json

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from model_io import load_checkpoint, model_input, outputs, pick_device
from preprocess import apply_transforms, load_prepared

SAMPLE_CHUNK = 500
GMM_COMPONENTS = 10      # mixture size for the latent density (unconditional)
GMM_PER_CLASS = 3        # mixture size per class (conditional VAEs)
TSTR_MAX = 10_000


def _encode(model, entry, prepared, split, ckpt, device, classes=None):
    """Posterior means for one split, in chunks. (n, latent_dim)"""
    conditional = bool(entry.get("conditional"))
    lookup = {c: i for i, c in enumerate(classes or [])}
    x = torch.from_numpy(model_input(entry, prepared, split, ckpt["input_pca"]))
    mus = []
    with torch.no_grad():
        for i in range(0, len(x), SAMPLE_CHUNK):
            xb, c = x[i:i + SAMPLE_CHUNK].to(device), None
            if conditional:
                yb = torch.tensor([lookup[v] for v in prepared[f"y_{split}"][i:i + SAMPLE_CHUNK]], device=device)
                c = torch.nn.functional.one_hot(yb, len(classes)).float()
            mus.append(model.encode(xb, c)[0].cpu())
    return torch.cat(mus).numpy().astype(np.float64)


class LatentSampler:
    """Gaussian mixture(s) over training latent codes; per class for conditional VAEs."""

    def __init__(self, z_train, seed, y_train=None, extra=None):
        self.seed, self.per_class = seed, y_train is not None
        data = z_train if extra is None else np.hstack([z_train, extra])
        if self.per_class:
            self.gmms = {c: GaussianMixture(min(GMM_PER_CLASS, max(1, (y_train == c).sum() // 50)), covariance_type="full",
                                            random_state=seed).fit(data[y_train == c]) for c in np.unique(y_train)}
        else:
            self.gmm = GaussianMixture(GMM_COMPONENTS, covariance_type="full", random_state=seed).fit(data)
        self.description = (f"Gaussian mixture fit to training latent codes ({GMM_PER_CLASS} components per class)" if self.per_class
                            else f"Gaussian mixture ({GMM_COMPONENTS} components) fit to training latent codes"
                            + (" jointly with log library size" if extra is not None else ""))

    def sample(self, n, labels=None):
        if not self.per_class:
            self.gmm.random_state = self.seed
            s, _ = self.gmm.sample(n)
            return s[np.random.default_rng(self.seed).permutation(n)]  # GMM.sample returns draws grouped by component
        out = np.zeros((n, next(iter(self.gmms.values())).means_.shape[1]))
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            g = self.gmms[c]; g.random_state = self.seed
            out[idx] = g.sample(len(idx))[0][np.random.default_rng(self.seed).permutation(len(idx))]
        return out


def _sample_images(model, entry, labels, classes, device, n, sampler=None):
    """Draw n images in chunks; labels (n,) class strings for conditional models, else None."""
    hp, fam, out = entry["hyperparameters"], entry["family"], []
    lookup = {c: i for i, c in enumerate(classes or [])}
    z_all = sampler.sample(n, labels) if sampler is not None else None
    for i in range(0, n, SAMPLE_CHUNK):
        m = min(SAMPLE_CHUNK, n - i)
        y = torch.tensor([lookup[v] for v in labels[i:i + m]], device=device) if labels is not None else None
        if fam == "vae":
            c = torch.nn.functional.one_hot(y, len(classes)).float() if y is not None else None
            z = torch.from_numpy(z_all[i:i + m]).float().to(device) if z_all is not None else None
            out.append(model.sample(m, c, z=z).cpu())
        elif fam == "gan":
            out.append(model.sample(m, y).cpu())
        else:
            out.append(model.sample(m, y, steps=hp["sampling_steps"], guidance=hp["guidance_scale"]).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def _objective_tasks(dataset):
    obj = json.loads((outputs(dataset) / "objective.json").read_text())
    return {obj.get("effective_task"), *obj.get("secondary_tasks", [])}


def generate_images(dataset, key, prepared, model, ckpt, n):
    entry, classes, seed = ckpt["entry"], ckpt.get("classes"), ckpt["seed"]
    device = pick_device("image")
    model = model.to(device)
    out_dir = outputs(dataset) / "samples"
    conditional = bool(entry.get("conditional"))
    sampler, info = None, {"latent_sampling": None}
    if entry["family"] == "vae":
        lat_dir = outputs(dataset) / "latents"
        lat_dir.mkdir(exist_ok=True)
        z = {s: _encode(model, entry, prepared, s, ckpt, device, classes) for s in ("train", "test")}
        for s in z:
            np.save(lat_dir / f"{key}_{s}.npy", z[s].astype(np.float32))
        sampler = LatentSampler(z["train"], seed, prepared["y_train"] if conditional else None)
        info["latent_sampling"] = sampler.description

    labels = prepared["y_test"][:n] if conditional else None  # held-out class mix
    np.save(out_dir / f"{key}_trans.npy", _sample_images(model, entry, labels, classes, device, n, sampler))
    if conditional:
        np.save(out_dir / f"{key}_y.npy", np.asarray(labels).astype(str))
        tasks = _objective_tasks(dataset)
        if tasks & {"class_balancing", "augmentation"}:
            from downstream_test import target_classes
            targets, counts = target_classes(dataset, prepared)
            bal = np.array([t for t in targets for _ in range(max(counts.values()) - counts[t])])
            if len(bal):
                np.save(out_dir / f"{key}_balance_trans.npy", _sample_images(model, entry, bal, classes, device, len(bal), sampler))
                np.save(out_dir / f"{key}_balance_y.npy", bal.astype(str))
                info["balance_samples"] = int(len(bal))
                print(f"[generate] {key}: {len(bal)} balancing samples for {targets}")
        if tasks & {"data_generation", "augmentation"}:
            # Training-sized labeled set with the training class mix, for train-on-synthetic, test-on-real.
            n_t = min(len(prepared["y_train"]), TSTR_MAX)
            y_t = np.random.default_rng(seed).choice(prepared["y_train"], size=n_t, replace=False)
            np.save(out_dir / f"{key}_tstr_trans.npy", _sample_images(model, entry, y_t, classes, device, n_t, sampler))
            np.save(out_dir / f"{key}_tstr_y.npy", y_t.astype(str))
            info["tstr_samples"] = int(n_t)
            print(f"[generate] {key}: {n_t} train-on-synthetic samples")
    (out_dir / f"{key}_sampling.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[generate] {key}: {n} samples -> {out_dir}")


def generate(dataset, key, n=None):
    prepared = load_prepared(dataset)
    fitted = prepared["state"]["fitted"]
    model, ckpt = load_checkpoint(dataset, key)
    entry, seed = ckpt["entry"], ckpt["seed"]
    n = int(n or prepared["meta"]["n_test"])
    torch.manual_seed(10_000 + seed)  # sampling seed derived from (but distinct from) the training seed
    rng = np.random.default_rng(10_000 + seed)
    out_dir = outputs(dataset) / "samples"
    out_dir.mkdir(exist_ok=True)
    if ckpt.get("image_shape"):
        return generate_images(dataset, key, prepared, model, ckpt, n)

    space, info = entry["input_space"], {"latent_sampling": None}
    if entry["family"] == "vae":
        lat_dir = outputs(dataset) / "latents"
        lat_dir.mkdir(exist_ok=True)
        z = {s: _encode(model, entry, prepared, s, ckpt, "cpu") for s in ("train", "test")}
        for s in z:
            np.save(lat_dir / f"{key}_{s}.npy", z[s].astype(np.float32))
        nb = entry["hyperparameters"]["likelihood"] == "nb"
        log_lib = np.log(ckpt["train_library"].astype(np.float64))[:, None] if nb else None
        sampler = LatentSampler(z["train"], 10_000 + seed, extra=log_lib)
        info["latent_sampling"] = sampler.description
        draws = sampler.sample(n)
        zs = torch.from_numpy(draws[:, :z["train"].shape[1]]).float()
        if nb:
            # Library size drawn jointly with the latent code, clipped to the observed training range.
            lib = np.clip(np.exp(draws[:, -1:]), ckpt["train_library"].min(), ckpt["train_library"].max())
            raw = model.sample(n, library=torch.from_numpy(lib).float(), z=zs).numpy()     # (n, d) counts
            np.save(out_dir / f"{key}_raw.npy", raw.astype(np.float32))
            trans = apply_transforms(raw.astype(np.float64), fitted)                     # (n, d)
        else:
            s = model.sample(n, z=zs).numpy()
            trans = s if space == "transformed" else apply_transforms(s.astype(np.float64), fitted)
    else:
        s = model.sample(n).numpy()                                       # (n, d_in)
        if space == "pca":
            pca = ckpt["input_pca"]
            # Back to transformed space, plus Gaussian residual noise outside the modeled components.
            trans = s @ pca["components"] + pca["mean"] + rng.normal(size=(n, len(pca["mean"]))) * pca["resid_std"]  # (n, d)
        elif space == "transformed":
            trans = s
        else:  # raw, non-VAE
            np.save(out_dir / f"{key}_raw.npy", s.astype(np.float32))
            trans = apply_transforms(s.astype(np.float64), fitted)
    np.save(out_dir / f"{key}_trans.npy", np.asarray(trans, dtype=np.float32))
    (out_dir / f"{key}_sampling.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[generate] {key}: {n} samples -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a trained model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True, help="<model_id>_s<seed>")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    generate(args.dataset, args.key, args.n)
