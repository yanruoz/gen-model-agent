"""Draw synthetic samples from one trained model (and, for VAEs, encode real data into the latent space).

   Usage: venv/bin/python scripts/generate.py --dataset <d> --key <id>_s<seed> [--n <int>]

   Default n = size of the held-out split, so synthetic and held-out real sets are the same size for
   the two-sample metrics. Writes to outputs/<d>/samples/:
     <key>_trans.npy   samples in the shared transformed space (every model)     (n, d)
     <key>_raw.npy     raw-scale samples, when the model generates them (NB VAE)  (n, d)
   and, for VAEs, outputs/<d>/latents/<key>_{train,test}.npy (posterior means)  (n, latent_dim)
"""
import argparse

import numpy as np
import torch

from model_io import load_checkpoint, model_input, outputs
from preprocess import apply_transforms, load_prepared


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

    space = entry["input_space"]
    if entry["family"] == "vae" and entry["hyperparameters"]["likelihood"] == "nb":
        # New cells get library sizes resampled from the training cells' observed totals.
        library = torch.from_numpy(rng.choice(ckpt["train_library"], size=n, replace=True)).unsqueeze(1)
        raw = model.sample(n, library=library).numpy()                   # (n, d) counts
        np.save(out_dir / f"{key}_raw.npy", raw.astype(np.float32))
        trans = apply_transforms(raw.astype(np.float64), fitted)          # (n, d)
    else:
        s = model.sample(n).numpy()                                       # (n, d_in)
        if space == "pca":
            pca = ckpt["input_pca"]
            # Back to transformed space, plus Gaussian residual noise outside the modeled components.
            trans = s @ pca["components"] + pca["mean"] + rng.normal(size=(n, len(pca["mean"]))) * pca["resid_std"]  # (n, d)
        elif space == "transformed":
            trans = s
        else:  # raw, non-NB
            np.save(out_dir / f"{key}_raw.npy", s.astype(np.float32))
            trans = apply_transforms(s.astype(np.float64), fitted)
    np.save(out_dir / f"{key}_trans.npy", np.asarray(trans, dtype=np.float32))

    if entry["family"] == "vae":
        lat_dir = outputs(dataset) / "latents"
        lat_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            for split in ("train", "test"):
                x = torch.from_numpy(model_input(entry, prepared, split, ckpt["input_pca"]))
                mu, _ = model.encode(x)
                np.save(lat_dir / f"{key}_{split}.npy", mu.numpy().astype(np.float32))
    print(f"[generate] {key}: {n} samples -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a trained model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True, help="<model_id>_s<seed>")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    generate(args.dataset, args.key, args.n)
