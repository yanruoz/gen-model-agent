"""Draw synthetic samples from one trained model (and, for VAEs, encode real data into the latent space).

   Usage: venv/bin/python scripts/generate.py --dataset <d> --key <id>_s<seed> [--n <int>]

   Default n = size of the held-out split, so synthetic and held-out real sets are the same size for
   the two-sample metrics. Writes to outputs/<d>/samples/:
     <key>_trans.npy   samples in the shared transformed space (every model)     (n, d)
     <key>_raw.npy     raw-scale samples, when the model generates them (NB VAE)  (n, d)
   and, for VAEs, outputs/<d>/latents/<key>_{train,test}.npy (posterior means)  (n, latent_dim)

   Image models: conditional models generate the held-out split's class mix (labels saved as
   <key>_y.npy). When the objective is class_balancing or augmentation, a conditional model also
   generates the target class(es) up to the largest training class (<key>_balance_trans.npy /
   <key>_balance_y.npy) for downstream_test.py.
"""
import json
import argparse

import numpy as np
import torch

from model_io import load_checkpoint, model_input, outputs, pick_device
from preprocess import apply_transforms, load_prepared


SAMPLE_CHUNK = 500


def _sample_images(model, entry, labels, classes, device, n):
    """Draw n images in chunks; labels (n,) class strings for conditional models, else None."""
    hp, fam, out = entry["hyperparameters"], entry["family"], []
    lookup = {c: i for i, c in enumerate(classes or [])}
    for i in range(0, n, SAMPLE_CHUNK):
        m = min(SAMPLE_CHUNK, n - i)
        y = torch.tensor([lookup[v] for v in labels[i:i + m]], device=device) if labels is not None else None
        if fam == "vae":
            c = torch.nn.functional.one_hot(y, len(classes)).float() if y is not None else None
            out.append(model.sample(m, c).cpu())
        elif fam == "gan":
            out.append(model.sample(m, y).cpu())
        else:
            out.append(model.sample(m, y, steps=hp["sampling_steps"], guidance=hp["guidance_scale"]).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def generate_images(dataset, key, prepared, model, ckpt, n):
    entry, classes = ckpt["entry"], ckpt.get("classes")
    device = pick_device("image")
    model = model.to(device)
    out_dir = outputs(dataset) / "samples"
    conditional = bool(entry.get("conditional"))
    labels = prepared["y_test"][:n] if conditional else None  # held-out class mix
    np.save(out_dir / f"{key}_trans.npy", _sample_images(model, entry, labels, classes, device, n))
    if conditional:
        np.save(out_dir / f"{key}_y.npy", np.asarray(labels).astype(str))
        obj = json.loads((outputs(dataset) / "objective.json").read_text())
        if obj.get("effective_task") in ("class_balancing", "augmentation"):
            from downstream_test import target_classes
            targets, counts = target_classes(dataset, prepared)
            bal = np.array([t for t in targets for _ in range(max(counts.values()) - counts[t])])
            if len(bal):
                np.save(out_dir / f"{key}_balance_trans.npy", _sample_images(model, entry, bal, classes, device, len(bal)))
                np.save(out_dir / f"{key}_balance_y.npy", bal.astype(str))
                print(f"[generate] {key}: {len(bal)} balancing samples for {targets}")
    if entry["family"] == "vae":
        lat_dir = outputs(dataset) / "latents"
        lat_dir.mkdir(exist_ok=True)
        lookup = {c: i for i, c in enumerate(classes or [])}
        with torch.no_grad():
            for split in ("train", "test"):
                x = torch.from_numpy(prepared[f"trans_{split}"])
                mus = []
                for i in range(0, len(x), SAMPLE_CHUNK):
                    xb = x[i:i + SAMPLE_CHUNK].to(device)
                    c = None
                    if conditional:
                        yb = torch.tensor([lookup[v] for v in prepared[f"y_{split}"][i:i + SAMPLE_CHUNK]], device=device)
                        c = torch.nn.functional.one_hot(yb, len(classes)).float()
                    mus.append(model.encode(xb, c)[0].cpu())
                np.save(lat_dir / f"{key}_{split}.npy", torch.cat(mus).numpy().astype(np.float32))
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

    space = entry["input_space"]
    if entry["family"] == "vae" and entry["hyperparameters"]["likelihood"] == "nb":
        # New samples get library sizes resampled from the training samples' observed totals.
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
