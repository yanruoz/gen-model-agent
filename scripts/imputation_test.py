"""Objective test for imputation: can the VAE recover masked values of HELD-OUT real samples better than
   simple imputers?

   Usage: venv/bin/python scripts/imputation_test.py --dataset <d> --keys k1,k2,...

   Masking (all generic, decided from the profile):
     sparse count data   10% of each sample's NONZERO entries are hidden (set to 0), the usual dropout
                         benchmark: zeros are already ambiguous, so only known nonzero values can be scored
     images              one random 8x8 square per image (all channels), filled with the training mean
     continuous tabular  20% of entries at random, filled with the training mean
   Imputers:
     VAE                 encode the masked sample, decode, take the decoder mean at masked entries; repeated
                         ITERATIONS times, feeding the imputed values back in (count VAE: library size =
                         the sample's observed total, as the decoder expects)
     mean                training mean of each feature (count data: scaled to the sample's observed total)
     knn                 k-nearest-neighbor imputation (k = 10) fit on training samples
   Error is measured on masked entries only: count data on the log1p(count) scale, images on [0, 1]
   pixels, continuous data on the transformed scale. Writes outputs/<d>/metrics/imputation.json.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.impute import KNNImputer

from model_io import load_checkpoint, model_input, outputs, pick_device
from preprocess import load_prepared

N_TEST = 1000
KNN_TRAIN = 3000
ITERATIONS = 3
SEED = 0


def _mask(prepared, rng):
    meta = prepared["meta"]
    n = min(N_TEST, meta["n_test"])
    count = meta["is_count_raw"] and not meta.get("image_shape")
    if count:
        x = prepared["raw_test"][:n].astype(np.float64)
        mask = np.zeros_like(x, dtype=bool)
        for i, row in enumerate(x):
            nz = np.flatnonzero(row)
            k = max(1, int(round(0.1 * len(nz))))
            mask[i, rng.choice(nz, k, replace=False)] = True
        return "count", x, mask
    x = prepared["trans_test"][:n].astype(np.float64)
    mask = np.zeros_like(x, dtype=bool)
    if meta.get("image_shape"):
        h, w, c = meta["image_shape"]
        for i in range(n):
            r0, c0 = rng.integers(0, h - 8 + 1), rng.integers(0, w - 8 + 1)
            m = np.zeros((h, w, c), dtype=bool); m[r0:r0 + 8, c0:c0 + 8, :] = True
            mask[i] = m.reshape(-1)
        return "image", x, mask
    mask = rng.random(x.shape) < 0.2
    return "continuous", x, mask


def _errors(true, imp, mask, kind):
    t, p = true[mask], imp[mask]
    if kind == "count":
        t, p = np.log1p(t), np.log1p(np.clip(p, 0, None))
    return {"rmse": round(float(np.sqrt(np.mean((t - p) ** 2))), 4), "mae": round(float(np.mean(np.abs(t - p))), 4)}


@torch.no_grad()
def vae_impute(dataset, key, prepared, kind, x, mask, fill):
    model, ckpt = load_checkpoint(dataset, key)
    entry = ckpt["entry"]
    if entry["family"] != "vae":
        return None
    dev = pick_device("image") if ckpt.get("image_shape") else "cpu"
    model = model.to(dev)
    classes = ckpt.get("classes")
    cur = np.where(mask, fill, x)
    for _ in range(ITERATIONS):
        outs = []
        for i in range(0, len(cur), 500):
            xb = torch.from_numpy(cur[i:i + 500]).float().to(dev)
            c = None
            if entry.get("conditional"):  # the held-out sample's own class is known
                lookup = {cl: j for j, cl in enumerate(classes)}
                yb = torch.tensor([lookup[v] for v in prepared["y_test"][i:i + 500]], device=dev)
                c = torch.nn.functional.one_hot(yb, len(classes)).float()
            mu, _ = model.encode(xb, c)
            if kind == "count":
                # Decoder proportions (library 1), then the total count implied by the observed entries:
                # observed total / (1 - predicted share of the masked features).
                mb = torch.from_numpy(mask[i:i + 500]).to(dev)
                rho = model.decode(mu, c, torch.ones(len(mu), 1, device=dev))["mu"]          # (b, d), rows sum to 1
                obs = torch.from_numpy((x[i:i + 500] * ~mask[i:i + 500]).sum(1, keepdims=True)).float().to(dev)
                lib = obs / (1 - (rho * mb).sum(1, keepdim=True)).clamp_min(1e-3)
                outs.append((rho * lib).cpu().numpy())
            elif ckpt.get("image_shape"):
                outs.append(torch.sigmoid(model.decode_logits(mu, c)).cpu().numpy())
            else:
                p = model.decode(mu, c)
                outs.append((p["mean"] if "mean" in p else torch.sigmoid(p["logits"])).cpu().numpy())
        cur = np.where(mask, np.concatenate(outs), x)
    return cur


def run(dataset, keys):
    prepared = load_prepared(dataset)
    rng = np.random.default_rng(SEED)
    kind, x, mask = _mask(prepared, rng)
    train = prepared["raw_train"].astype(np.float64) if kind == "count" else prepared["trans_train"].astype(np.float64)

    # ── Baselines ──
    if kind == "count":
        # Training mean profile scaled to each sample's observed total.
        prof = train.sum(0) / train.sum()
        obs_total = (x * ~mask).sum(1, keepdims=True)
        mean_imp = np.where(mask, prof[None, :] * obs_total / np.maximum(1 - (prof[None, :] * mask).sum(1, keepdims=True), 1e-6), x)
        fill = np.zeros_like(x)
        knn_space = lambda a: np.log1p(a)
    else:
        mean_imp = np.where(mask, train.mean(0)[None, :], x)
        fill = np.broadcast_to(train.mean(0)[None, :], x.shape)
        knn_space = lambda a: a
    sub = train[np.random.default_rng(SEED).choice(len(train), min(KNN_TRAIN, len(train)), replace=False)]
    knn = KNNImputer(n_neighbors=10).fit(knn_space(sub))
    q = knn_space(x).copy(); q[mask] = np.nan
    knn_imp = knn.transform(q)
    knn_imp = np.expm1(knn_imp) if kind == "count" else knn_imp
    results = {"baseline_mean": _errors(x, mean_imp, mask, kind), "baseline_knn": _errors(x, knn_imp, mask, kind)}
    for k in keys:
        imp = vae_impute(dataset, k, prepared, kind, x, mask, fill)
        results[k] = _errors(x, imp, mask, kind) if imp is not None else {"status": "skipped: only VAEs impute in this pipeline"}
    for k, v in results.items():
        print(f"[imputation] {k}: {v}", flush=True)
    out = {"mask_kind": kind, "n_samples": int(len(x)), "n_masked_entries": int(mask.sum()),
           "error_scale": {"count": "log1p(count)", "image": "pixel intensity in [0, 1]", "continuous": "transformed scale"}[kind],
           "results": results,
           "definitions": {
               "rmse": "Root mean squared error between true and imputed values, masked entries only (lower is better).",
               "mae": "Mean absolute error, masked entries only.",
               "baseline_mean": "Training mean of each feature (count data: training mean profile scaled to the sample's observed total).",
               "baseline_knn": "k-nearest-neighbor imputation (k = 10) from training samples.",
           },
           "note": "Masked values are real values hidden on purpose, so the truth is known. For count data only nonzero "
                   "entries were masked, which tests recovery of expressed values, not whether a zero is a dropout."}
    (outputs(dataset) / "metrics").mkdir(exist_ok=True)
    (outputs(dataset) / "metrics" / "imputation.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imputation test on masked held-out data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keys", required=True)
    args = parser.parse_args()
    run(args.dataset, [k for k in args.keys.split(",") if k])
