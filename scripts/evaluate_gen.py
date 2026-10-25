"""Evaluate one trained model against HELD-OUT real data. Metrics are fixed per modality and model
   capability here, never chosen by the agent. Every two-sample metric comes with a real-vs-real
   reference value (training subsample vs held-out) so "good" has a concrete yardstick.

   Usage: venv/bin/python scripts/evaluate_gen.py --dataset <d> --key <id>_s<seed>
   Writes outputs/<d>/metrics/<key>.json.

   Tabular metric groups:
     fidelity        (all samplers)  in the shared evaluation PCA space
     memorization    (all samplers)  distance to closest training record (DCR)
     feature_stats   (all samplers)  per-feature mean/variance agreement in transformed space
     likelihood      (VAE: ELBO, flow: exact log-likelihood), in the model's own input space
     representation  (VAE)           latent space vs. a PCA of the same size

   Image metric groups (evaluation space = features of a CNN trained on the real training images,
   see image_eval.py):
     fidelity             Frechet distance, MMD, C2ST, precision/recall, each with a real-vs-real reference
     memorization         DCR in the feature space
     memorization_pixels  DCR in raw pixel space (near-copies of training images)
     conditional          (conditional models) does the classifier recognize the requested class?
     feature_stats        per-pixel mean/variance agreement
     likelihood           (VAE) ELBO
"""
import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

from model_io import load_checkpoint, model_input, outputs
from preprocess import load_prepared, to_eval_space

KNN_K = 5          # neighbors for precision/recall (Kynkaanniemi et al., 2019)
TRUST_K = 15       # neighbors for trustworthiness / kNN overlap
C2ST_FOLDS = 5
REF_SEED = 0  # one real-vs-real reference subsample shared by all models
C2ST_REFERENCE_WARN = 0.65  # real-vs-real AUC above this flags a biased evaluation space

IMAGE_DEFINITIONS = {
    "frechet_distance": "Frechet distance between Gaussians fit to synthetic and held-out real features (the FID formula, computed on the evaluation classifier's features instead of Inception's). 0 = same mean and covariance; compare with frechet_distance_reference.",
    "frechet_distance_reference": "Same distance, training-set subsample vs held-out real: what two real samples of this size give.",
    "label_agreement": "Share of synthetic images that the evaluation classifier (trained on real images) assigns to the class they were generated for.",
    "label_agreement_reference": "The classifier's accuracy on held-out real images: the agreement a perfect class-conditional generator would reach.",
}

DEFINITIONS = {
    "mmd2": "Squared maximum mean discrepancy (RBF kernel, bandwidth = median distance among held-out real samples) between synthetic and held-out real samples in the evaluation PCA space. 0 = same distribution; compare with mmd2_reference. The estimator is unbiased, so values near 0 can be slightly negative.",
    "mmd2_reference": "Same MMD, but a training-set subsample vs held-out real: the value two real samples of this size give.",
    "c2st_auc": "Classifier two-sample test: cross-validated ROC AUC of a gradient-boosted classifier separating synthetic from held-out real samples. 0.5 = indistinguishable, 1.0 = perfectly separable.",
    "c2st_auc_reference": "Same test, training subsample vs held-out real (should be near 0.5).",
    "precision": f"Fraction of synthetic samples inside the {KNN_K}-NN radius of some held-out real sample (fidelity: are samples realistic?).",
    "recall": f"Fraction of held-out real samples inside the {KNN_K}-NN radius of some synthetic sample (coverage: is the data's variety reproduced?).",
    "dcr_median_synthetic": "DCR = distance to closest record: for one sample, the Euclidean distance (evaluation space) to its nearest training sample. This is the median DCR over synthetic samples.",
    "dcr_median_heldout": "Median DCR over held-out real samples: how close a genuinely new sample from the same population typically is to some training sample. This is the yardstick for the synthetic DCR.",
    "dcr_ratio": "dcr_median_synthetic / dcr_median_heldout. About 1 means synthetic samples are as far from the training data as new real samples are; well below 1 means they sit unusually close to training samples (possible copying); above 1 means they are farther away than real data (often a sign of off-distribution samples, not of privacy).",
    "too_close_fraction": "Fraction of synthetic samples whose DCR is below the 5th percentile of held-out DCRs. If the model does not copy, about 0.05 is expected (the same share as for new real samples); much higher means a group of synthetic samples are near-copies of training samples. Evidence about copying only, not a privacy guarantee.",
    "mean_corr": "Pearson correlation, across features, between per-feature means of synthetic and held-out real data (transformed space).",
    "var_corr": "Same, for per-feature variances.",
    "zero_frac_corr": "Same, for each feature's fraction of exact zeros (raw counts; only for models that generate counts).",
    "library_median_ratio": "Median total count per synthetic sample divided by the held-out median (counts only).",
    "train_mean": "Mean per-sample ELBO (VAE, a lower bound) or exact log-likelihood (flow) on training data, in the model's own input space. Not comparable across models with different input spaces.",
    "heldout_mean": "Same quantity on held-out real data; a large drop from train_mean indicates overfitting.",
    "trustworthiness": f"Trustworthiness (Venna and Kaski, 2001), between 0 and 1: for each held-out sample, look at its {TRUST_K} nearest neighbors in the latent space and penalize any that are NOT among its {TRUST_K} nearest neighbors in the reference space (full transformed data), more heavily the farther away they really are. 1 = every latent-space neighbor is a true neighbor; lower = the latent space puts unrelated samples next to each other.",
    "trustworthiness_pca_baseline": "Trustworthiness of a PCA (fit on training data) with as many dimensions as the latent space: the bar a learned representation is compared with.",
    "knn_overlap": f"Average fraction of each held-out sample's {TRUST_K} nearest neighbors in the full transformed space that are also among its {TRUST_K} nearest neighbors in the latent space.",
    "knn_overlap_pca_baseline": "Same, for the same-size PCA baseline.",
}


def _mmd2(x, y, gamma):
    """Unbiased squared MMD with RBF kernel exp(-gamma * ||a-b||^2). x (n, k), y (m, k)."""
    x, y = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)); kxx.fill_diagonal_(0)
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)); kyy.fill_diagonal_(0)
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
    n, m = len(x), len(y)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def _c2st(a, b, seed):
    """Shuffled stratified folds: with unshuffled folds, a sample set stored in a structured order
    (e.g. grouped by mixture component) leaves whole groups out of training and understates the AUC."""
    from sklearn.model_selection import StratifiedKFold
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    cv = StratifiedKFold(C2ST_FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def _knn_radii(x, k):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    return d[:, -1]  # distance to the k-th neighbor (excluding self)


def _coverage(points, ref, ref_radii):
    """Fraction of `points` inside the k-NN ball of at least one `ref` sample."""
    d = torch.cdist(torch.from_numpy(points).double(), torch.from_numpy(ref).double())  # (n_points, n_ref)
    return float((d <= torch.from_numpy(ref_radii).double()[None, :]).any(1).double().mean())


def _dcr(points, train):
    d, _ = NearestNeighbors(n_neighbors=1).fit(train).kneighbors(points)
    return d[:, 0]


def _knn_overlap(high, low, k):
    a = NearestNeighbors(n_neighbors=k + 1).fit(high).kneighbors(high, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(low).kneighbors(low, return_distance=False)[:, 1:]
    return float(np.mean([len(set(r1) & set(r2)) / k for r1, r2 in zip(a, b)]))


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _fidelity(f_syn, f_te, f_tr, seed, rng, with_fd=False):
    """Two-sample metrics between synthetic and held-out real features, with real-vs-real references."""
    n = min(len(f_syn), len(f_te))
    f_syn, f_te_n = f_syn[:n], f_te[:n]
    f_tr_sub = f_tr[np.random.default_rng(REF_SEED).choice(len(f_tr), size=n, replace=False)]  # same reference for every model
    d_te = torch.cdist(torch.from_numpy(f_te_n), torch.from_numpy(f_te_n))
    gamma = 1.0 / (2 * float(d_te[d_te > 0].median()) ** 2)
    out = {"n_compared": n}
    if with_fd:
        from image_eval import frechet_distance
        out.update(frechet_distance=_r(frechet_distance(f_syn, f_te_n)), frechet_distance_reference=_r(frechet_distance(f_tr_sub, f_te_n)))
    out.update(mmd2=_r(_mmd2(f_syn, f_te_n, gamma), 5), mmd2_reference=_r(_mmd2(f_tr_sub, f_te_n, gamma), 5),
               c2st_auc=_r(_c2st(f_syn, f_te_n, seed)), c2st_auc_reference=_r(_c2st(f_tr_sub, f_te_n, seed)),
               precision=_r(_coverage(f_syn, f_te_n, _knn_radii(f_te_n, KNN_K))),
               recall=_r(_coverage(f_te_n, f_syn, _knn_radii(f_syn, KNN_K))))
    if out["c2st_auc_reference"] > C2ST_REFERENCE_WARN:
        out["warning"] = (f"real-vs-real C2ST reference is {out['c2st_auc_reference']} (> {C2ST_REFERENCE_WARN}): the evaluation "
                          "space separates training from held-out real data, so fidelity metrics are biased against all models")
    return out, f_syn, f_te_n


def _memorization(points, ref_points, train):
    dcr_syn, dcr_te = _dcr(points, train), _dcr(ref_points, train)
    return {"dcr_median_synthetic": _r(np.median(dcr_syn)), "dcr_median_heldout": _r(np.median(dcr_te)),
            "dcr_ratio": _r(np.median(dcr_syn) / np.median(dcr_te)),
            "too_close_fraction": _r(np.mean(dcr_syn < np.percentile(dcr_te, 5))),
            "note": "Evidence about copying of training samples, not a formal privacy guarantee."}


def evaluate_image(dataset, key, prepared, model, ckpt):
    from image_eval import class_index, features, get_evaluator, predict
    entry, seed = ckpt["entry"], ckpt["seed"]
    rng = np.random.default_rng(seed)
    out = outputs(dataset)
    evaluator, ev_info = get_evaluator(dataset, prepared)
    syn = np.load(out / "samples" / f"{key}_trans.npy")                                   # (n_syn, H*W*C)
    x_te, x_tr = prepared["trans_test"], prepared["trans_train"]
    f_syn, f_te, f_tr = features(evaluator, syn), features(evaluator, x_te), features(evaluator, x_tr)  # (n, 128)
    metrics = {"key": key, "model_id": entry["id"], "family": entry["family"], "seed": seed, "conditional": bool(entry.get("conditional")),
               "input_space": entry["input_space"],
               "eval_space": f"{ev_info['feature_dim']}-d features of a CNN classifier trained on the real training images "
                             f"(held-out accuracy {ev_info['heldout_accuracy']})"}
    metrics["fidelity"], f_syn_n, f_te_n = _fidelity(f_syn, f_te, f_tr, seed, rng, with_fd=True)
    n = metrics["fidelity"]["n_compared"]
    metrics["memorization"] = _memorization(f_syn_n, f_te_n, f_tr)
    metrics["memorization_pixels"] = _memorization(syn[:n].astype(np.float64), x_te[:n].astype(np.float64), x_tr.astype(np.float64))
    metrics["memorization_pixels"]["note"] = "DCR in raw pixel space (scaled to [0, 1]): detects near-identical copies of training images."

    classes = prepared["meta"]["classes"]
    y_path = out / "samples" / f"{key}_y.npy"
    if y_path.exists():
        req = class_index(np.load(y_path).astype(str), classes)
        pred = predict(evaluator, syn)
        y_te_idx = class_index(prepared["y_test"], classes)
        pred_te = predict(evaluator, x_te)
        metrics["conditional"] = {
            "label_agreement": _r(np.mean(pred == req)),
            "label_agreement_reference": _r(np.mean(pred_te == y_te_idx)),
            "per_class_agreement": {c: _r(np.mean(pred[req == i] == i)) for i, c in enumerate(classes) if (req == i).any()},
            "per_class_heldout_accuracy": {c: _r(np.mean(pred_te[y_te_idx == i] == i)) for i, c in enumerate(classes)},
        }
    metrics["feature_stats"] = {"mean_corr": _r(np.corrcoef(syn.mean(0), x_te.mean(0))[0, 1]),
                                "var_corr": _r(np.corrcoef(syn.var(0), x_te.var(0))[0, 1])}
    if entry["family"] == "vae":
        dev = next(model.parameters()).device
        cond = bool(entry.get("conditional"))
        def elbo(x, y):
            vals = []
            for i in range(0, len(x), 500):
                xb = torch.from_numpy(x[i:i + 500]).to(dev)
                c = torch.nn.functional.one_hot(torch.from_numpy(class_index(y[i:i + 500], classes)).to(dev), len(classes)).float() if cond else None
                vals.append(model.elbo(xb, c, n_samples=2).cpu())
            return float(torch.cat(vals).mean())
        torch.manual_seed(seed)
        ll_tr, ll_te = elbo(x_tr, prepared["y_train"]), elbo(x_te, prepared["y_test"])
        metrics["likelihood"] = {"kind": f"ELBO (lower bound; {entry['hyperparameters']['likelihood']} pixel likelihood)", "space": "pixels",
                                 "train_mean": _r(ll_tr, 3), "heldout_mean": _r(ll_te, 3), "per_dimension_heldout": _r(ll_te / x_te.shape[1], 4)}
    defs = {**DEFINITIONS, **IMAGE_DEFINITIONS}
    metrics["definitions"] = {k: v for k, v in defs.items() if any(k in g for g in metrics.values() if isinstance(g, dict))}
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    (mdir / f"{key}.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    f_ = metrics["fidelity"]
    print(f"[evaluate] {key}: FD {f_['frechet_distance']} (ref {f_['frechet_distance_reference']}), C2ST {f_['c2st_auc']} "
          f"(ref {f_['c2st_auc_reference']}), label agreement {metrics.get('conditional', {}).get('label_agreement')}")
    return metrics


def evaluate(dataset, key):
    prepared = load_prepared(dataset)
    state = prepared["state"]
    model, ckpt = load_checkpoint(dataset, key)
    if ckpt.get("image_shape"):
        from model_io import pick_device
        return evaluate_image(dataset, key, prepared, model.to(pick_device("image")), ckpt)
    entry, seed = ckpt["entry"], ckpt["seed"]
    rng = np.random.default_rng(seed)
    out = outputs(dataset)
    metrics = {"key": key, "model_id": entry["id"], "family": entry["family"], "seed": seed,
               "input_space": entry["input_space"], "eval_space": f"PCA ({prepared['meta']['eval_components']} components) of transformed data, fit on training split"}

    # ── Fidelity, in the shared evaluation space ──
    trans_syn = np.load(out / "samples" / f"{key}_trans.npy")                    # (n_syn, d)
    ev_syn = to_eval_space(trans_syn.astype(np.float64), state)                    # (n_syn, k)
    ev_te, ev_tr = prepared["eval_test"].astype(np.float64), prepared["eval_train"].astype(np.float64)
    n = min(len(ev_syn), len(ev_te))
    ev_syn, ev_te_n = ev_syn[:n], ev_te[:n]
    ev_tr_sub = ev_tr[np.random.default_rng(REF_SEED).choice(len(ev_tr), size=n, replace=False)]  # same reference for every model

    d_te = torch.cdist(torch.from_numpy(ev_te_n), torch.from_numpy(ev_te_n))
    gamma = 1.0 / (2 * float(d_te[d_te > 0].median()) ** 2)
    radii_real, radii_syn = _knn_radii(ev_te_n, KNN_K), _knn_radii(ev_syn, KNN_K)
    metrics["fidelity"] = {
        "n_compared": n,
        "mmd2": _r(_mmd2(ev_syn, ev_te_n, gamma), 5), "mmd2_reference": _r(_mmd2(ev_tr_sub, ev_te_n, gamma), 5),
        "c2st_auc": _r(_c2st(ev_syn, ev_te_n, seed)), "c2st_auc_reference": _r(_c2st(ev_tr_sub, ev_te_n, seed)),
        "precision": _r(_coverage(ev_syn, ev_te_n, radii_real)), "recall": _r(_coverage(ev_te_n, ev_syn, radii_syn)),
    }

    # Safeguard: two real samples should be hard to tell apart. If they aren't, the evaluation space
    # itself separates fitted from unseen data (e.g. PCA overfitting) and every fidelity number is biased.
    if metrics["fidelity"]["c2st_auc_reference"] > C2ST_REFERENCE_WARN:
        metrics["fidelity"]["warning"] = (
            f"real-vs-real C2ST reference is {metrics['fidelity']['c2st_auc_reference']} (> {C2ST_REFERENCE_WARN}): "
            "the evaluation space separates training from held-out real data, so fidelity metrics are biased against all models")

    # ── Memorization: distance to closest training record ──
    dcr_syn, dcr_te = _dcr(ev_syn, ev_tr), _dcr(ev_te_n, ev_tr)
    metrics["memorization"] = {
        "dcr_median_synthetic": _r(np.median(dcr_syn)), "dcr_median_heldout": _r(np.median(dcr_te)),
        "dcr_ratio": _r(np.median(dcr_syn) / np.median(dcr_te)),
        "too_close_fraction": _r(np.mean(dcr_syn < np.percentile(dcr_te, 5))),
        "note": "Evidence about copying of training samples, not a formal privacy guarantee.",
    }

    # ── Per-feature statistics (transformed space; zeros/library on raw counts if generated) ──
    tr_te = prepared["trans_test"].astype(np.float64)
    fs = {"mean_corr": _r(np.corrcoef(trans_syn.mean(0), tr_te.mean(0))[0, 1]),
          "var_corr": _r(np.corrcoef(trans_syn.var(0), tr_te.var(0))[0, 1])}
    raw_path = out / "samples" / f"{key}_raw.npy"
    if raw_path.exists() and prepared["meta"]["is_count_raw"]:
        raw_syn, raw_te = np.load(raw_path), prepared["raw_test"]
        fs["zero_frac_corr"] = _r(np.corrcoef((raw_syn == 0).mean(0), (raw_te == 0).mean(0))[0, 1])
        fs["zero_frac_synthetic"] = _r((raw_syn == 0).mean()); fs["zero_frac_heldout"] = _r((raw_te == 0).mean())
        fs["library_median_ratio"] = _r(np.median(raw_syn.sum(1)) / np.median(raw_te.sum(1)))
    metrics["feature_stats"] = fs

    # ── Likelihood in the model's own input space ──
    x_tr = torch.from_numpy(model_input(entry, prepared, "train", ckpt["input_pca"]))
    x_te = torch.from_numpy(model_input(entry, prepared, "test", ckpt["input_pca"]))
    torch.manual_seed(seed)
    with torch.no_grad():
        f = (lambda x: model.elbo(x, n_samples=5)) if entry["family"] == "vae" else model.log_prob
        ll_tr, ll_te = float(f(x_tr).mean()), float(f(x_te).mean())
    metrics["likelihood"] = {"kind": "ELBO (lower bound)" if entry["family"] == "vae" else "exact log-likelihood",
                             "space": entry["input_space"], "train_mean": _r(ll_tr, 3), "heldout_mean": _r(ll_te, 3),
                             "per_dimension_heldout": _r(ll_te / x_te.shape[1], 4)}

    # ── Representation quality: VAE latent vs a same-size PCA, judged in the full transformed space ──
    lat_path = out / "latents" / f"{key}_test.npy"
    if lat_path.exists():
        from sklearn.decomposition import PCA
        lat = np.load(lat_path).astype(np.float64)                                 # (n_test, latent_dim)
        k_lat = lat.shape[1]
        ref = prepared["trans_test"].astype(np.float64)                            # (n_test, d) reference space
        pca_base = PCA(n_components=k_lat, random_state=seed).fit(prepared["trans_train"]).transform(ref)  # (n_test, k_lat)
        metrics["representation"] = {
            "latent_dim": k_lat, "reference_space": f"transformed data, all {ref.shape[1]} features (held-out samples)",
            "trustworthiness": _r(trustworthiness(ref, lat, n_neighbors=TRUST_K)),
            "trustworthiness_pca_baseline": _r(trustworthiness(ref, pca_base, n_neighbors=TRUST_K)),
            "knn_overlap": _r(_knn_overlap(ref, lat, TRUST_K)),
            "knn_overlap_pca_baseline": _r(_knn_overlap(ref, pca_base, TRUST_K)),
            "note": "Neighborhoods are Euclidean in the transformed space, which favors PCA (it is the best linear "
                    "Euclidean summary by construction); a latent close to the baseline is not a failure.",
        }

    metrics["definitions"] = {k: v for k, v in DEFINITIONS.items()
                              if any(k in g for g in metrics.values() if isinstance(g, dict))}
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    (mdir / f"{key}.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    f_ = metrics["fidelity"]
    print(f"[evaluate] {key}: C2ST AUC {f_['c2st_auc']} (ref {f_['c2st_auc_reference']}), "
          f"MMD2 {f_['mmd2']} (ref {f_['mmd2_reference']}), DCR ratio {metrics['memorization']['dcr_ratio']}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one trained model against held-out real data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.key)
