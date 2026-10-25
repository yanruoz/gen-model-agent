"""Objective test for anomaly_detection: can each model's score separate held-out real samples from
   corrupted ones better than simple non-generative baselines?

   Usage: venv/bin/python scripts/anomaly_test.py --dataset <d> --keys k1,k2,...

   No dataset here comes with labeled anomalies, so anomalies are made from HELD-OUT real samples with
   corruptions that have a concrete meaning (all generic, none dataset-specific):
     count data (non-negative integer, tabular):
       doublet           counts of two samples added together (in single-cell data, two cells captured as one)
       feature_shuffle   a sample's values permuted across features (right totals, wrong feature profile)
     images and continuous data (transformed scale):
       blend             average of two samples
       feature_shuffle   values permuted across features (for images: pixels)
       noise             Gaussian noise with 0.5 x each feature's training standard deviation
   Scores (higher = more anomalous):
     VAE: negative ELBO; flow: negative log-likelihood (PCA flow: plus the Gaussian residual it generates with);
     diffusion: mean denoising error over fixed timesteps. GANs have no likelihood and are not scored.
     Baselines: distance to the 5th nearest training sample, and PCA reconstruction error, both in the
     shared evaluation PCA space.
   Metric: AUROC (normal = held-out real, anomalous = corrupted held-out real), per corruption and mean.
   Writes outputs/<d>/metrics/anomaly.json.
"""
import argparse
import json
import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from model_io import load_checkpoint, outputs, pick_device
from preprocess import apply_transforms, load_prepared, to_eval_space

N_EACH = 1000
SEED = 0
DIFFUSION_TIMESTEPS = (20, 50, 100, 200)


def make_corruptions(prepared, rng):
    """Returns (normal, {name: anomalous}) as dicts with 'raw' (counts only) and 'trans' arrays."""
    meta = prepared["meta"]
    n = min(N_EACH, meta["n_test"] // 2)
    perm = rng.permutation(meta["n_test"])
    a_idx, b_idx = perm[:n], perm[n:2 * n]          # normal samples; partners for two-sample corruptions
    fitted = prepared["state"]["fitted"]
    count = meta["is_count_raw"] and not meta.get("image_shape")
    if count:
        raw = prepared["raw_test"].astype(np.float64)
        base = raw[a_idx]
        shuf = np.array([rng.permutation(r) for r in raw[b_idx]])
        anomalies = {"doublet": raw[a_idx] + raw[b_idx], "feature_shuffle": shuf}
        normal = {"raw": base, "trans": apply_transforms(base, fitted)}
        return normal, {k: {"raw": v, "trans": apply_transforms(v, fitted)} for k, v in anomalies.items()}
    tr = prepared["trans_test"].astype(np.float64)
    sd = prepared["trans_train"].std(0)
    lo, hi = (0.0, 1.0) if meta.get("image_shape") else (-np.inf, np.inf)
    anomalies = {"blend": 0.5 * (tr[a_idx] + tr[b_idx]),
                 "feature_shuffle": np.array([rng.permutation(r) for r in tr[b_idx]]),
                 "noise": np.clip(tr[b_idx] + rng.normal(size=tr[b_idx].shape) * 0.5 * sd, lo, hi)}
    return {"trans": tr[a_idx]}, {k: {"trans": v} for k, v in anomalies.items()}


def _model_x(entry, ckpt, arrays):
    space = entry["input_space"]
    if space == "raw":
        return arrays["raw"]
    if space == "transformed":
        return arrays["trans"]
    pca = ckpt["input_pca"]
    return (arrays["trans"] - pca["mean"]) @ pca["components"].T


@torch.no_grad()
def model_score(dataset, key, arrays):
    """Anomaly score per sample (higher = more anomalous)."""
    model, ckpt = load_checkpoint(dataset, key)
    entry = ckpt["entry"]
    dev = pick_device("image") if ckpt.get("image_shape") else "cpu"
    model = model.to(dev)
    x = np.asarray(_model_x(entry, ckpt, arrays), dtype=np.float32)
    out = []
    torch.manual_seed(SEED)
    for i in range(0, len(x), 500):
        xb = torch.from_numpy(x[i:i + 500]).to(dev)
        if entry["family"] == "vae":
            out.append(-model.elbo(xb, None, n_samples=3).cpu())
        elif entry["family"] == "flow":
            s = -model.log_prob(xb).cpu()
            if entry["input_space"] == "pca":
                # The PCA flow generates with independent Gaussian residuals; score them too.
                pca = ckpt["input_pca"]
                tb = arrays["trans"][i:i + 500]
                resid = tb - (x[i:i + 500] @ pca["components"] + pca["mean"])
                sd = np.maximum(pca["resid_std"], 1e-6)
                s = s + torch.from_numpy((0.5 * ((resid / sd) ** 2) + np.log(sd) + 0.5 * math.log(2 * math.pi)).sum(1)).float()
            out.append(s)
        elif entry["family"] == "diffusion":
            # Denoising error at fixed timesteps with fixed noise, without class information.
            from models.conv import to_nchw
            img = to_nchw(xb, model.shape) * 2 - 1
            null = torch.full((len(xb),), model.net.cls.num_embeddings - 1, device=dev, dtype=torch.long)
            g = torch.Generator(device="cpu").manual_seed(SEED)
            errs = []
            for t in DIFFUSION_TIMESTEPS:
                noise = torch.randn(img.shape, generator=g).to(dev)
                a = model.abar[t]
                xt = a.sqrt() * img + (1 - a).sqrt() * noise
                pred = model.net(xt, torch.full((len(xb),), t, device=dev, dtype=torch.long), null)
                errs.append(((pred - noise) ** 2).flatten(1).mean(1))
            out.append(torch.stack(errs).mean(0).cpu())
        else:
            return None
    return torch.cat(out).numpy()


def baseline_scores(prepared, arrays):
    state = prepared["state"]
    ev = to_eval_space(arrays["trans"], state)
    ev_tr = prepared["eval_train"].astype(np.float64)
    knn = NearestNeighbors(n_neighbors=5).fit(ev_tr).kneighbors(ev)[0][:, -1]
    recon = ev @ state["pca_components"] + state["pca_mean"]
    pca_err = ((arrays["trans"] - recon) ** 2).sum(1)
    return {"baseline_knn_distance": knn, "baseline_pca_reconstruction": pca_err}


def run(dataset, keys):
    prepared = load_prepared(dataset)
    rng = np.random.default_rng(SEED)
    normal, anomalies = make_corruptions(prepared, rng)
    scorers = {k: (lambda arr, k=k: model_score(dataset, k, arr)) for k in keys}
    scorers.update({b: (lambda arr, b=b: baseline_scores(prepared, arr)[b]) for b in ("baseline_knn_distance", "baseline_pca_reconstruction")})
    results = {}
    for name, fn in scorers.items():
        s_norm = fn(normal)
        if s_norm is None:
            results[name] = {"status": "skipped: this model family has no likelihood or reconstruction score"}
            continue
        per = {}
        for cname, arr in anomalies.items():
            s_anom = fn(arr)
            y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
            per[cname] = round(float(roc_auc_score(y, np.r_[s_norm, s_anom])), 4)
        results[name] = {"auroc": per, "auroc_mean": round(float(np.mean(list(per.values()))), 4)}
        print(f"[anomaly] {name}: AUROC {per} (mean {results[name]['auroc_mean']})", flush=True)
    out = {"n_normal": int(len(normal["trans"])), "n_per_corruption": int(len(normal["trans"])),
           "corruptions": list(anomalies), "results": results,
           "definitions": {
               "auroc": "Area under the ROC curve for telling corrupted held-out samples from uncorrupted ones using the score; 0.5 = chance, 1 = perfect.",
               "doublet": "Counts of two held-out samples added together.", "blend": "Average of two held-out samples.",
               "feature_shuffle": "A held-out sample's values permuted across features.",
               "noise": "Gaussian noise of 0.5 x each feature's training standard deviation added to a held-out sample.",
               "baseline_knn_distance": "Non-generative baseline: distance to the 5th nearest training sample in the evaluation PCA space.",
               "baseline_pca_reconstruction": "Non-generative baseline: squared error of reconstructing the sample from the evaluation PCA.",
           },
           "note": "Anomalies are synthetic corruptions of real held-out samples; real anomalies (e.g. low-quality samples) may be harder or easier to detect."}
    (outputs(dataset) / "metrics").mkdir(exist_ok=True)
    (outputs(dataset) / "metrics" / "anomaly.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly-detection test on corrupted held-out data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keys", required=True)
    args = parser.parse_args()
    run(args.dataset, [k for k in args.keys.split(",") if k])
