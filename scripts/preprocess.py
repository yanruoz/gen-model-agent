"""Applies plan.json's split + preprocessing block once and saves the result, so every model in the
   plan trains on exactly the same data (one shared preprocessing block, as in Project 1).

   Order is fixed: row steps -> train/test split -> feature selection (fit on train) -> transforms
   (fit on train) -> evaluation PCA (fit on train). Nothing is ever fit on the held-out split.

   Saves outputs/<d>/prepared/:
     raw_{train,test}.npy          selected features, untransformed            (n, d) float32
     trans_{train,test}.npy        after transform steps                       (n, d) float32
     eval_{train,test}.npy         evaluation PCA of trans                     (n, k) float32
     y_{train,test}.npy            labels, if any
     state.npz                     fitted transform + PCA parameters (to map synthetic raw -> trans -> eval)
     meta.json                     shapes, feature names, plan hash (for reuse)

   Usage: venv/bin/python scripts/preprocess.py outputs/<d>/plan.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from loaders import load_dataset
from plan_schema import EVAL_PCA_COMPONENTS, FEATURE_STEPS, ROW_STEPS, TRANSFORM_STEPS

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
# Bump when this file's output changes for the same plan, so cached prepared/ data is not reused.
PREP_VERSION = 5


def prep_hash(plan):
    """Identifies a preprocessing result: only the fields that change the prepared data."""
    key = {"version": PREP_VERSION, "dataset": plan["dataset"], "seed": plan["seed"], "split": {k: plan["split"].get(k) for k in ("test_fraction", "stratify")},
           "steps": [{k: s.get(k) for k in ("step", "params")} for s in plan["preprocessing"]]}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


# ── Transforms: fit on train, apply anywhere (incl. to synthetic raw samples) ──

def fit_transforms(raw_train, steps):
    """Returns a list of (step, params) with fitted parameters, applied in order."""
    fitted, x = [], raw_train
    for name in steps:
        if name == "normalize_total":
            # Scale each row to the median row total of the training data.
            target = float(np.median(x.sum(axis=1)))
            fitted.append((name, {"target_sum": target}))
        elif name == "log1p":
            fitted.append((name, {}))
        elif name == "standardize":
            mu, sd = x.mean(axis=0), x.std(axis=0)
            fitted.append((name, {"mean": mu, "std": np.where(sd > 0, sd, 1.0)}))
        elif name == "scale_unit_range":
            # One global min/max (not per feature) so pixel intensities keep their relative scale.
            fitted.append((name, {"min": float(x.min()), "max": float(x.max())}))
        x = apply_step(x, *fitted[-1])
    return fitted


def apply_step(x, name, p):
    if name == "normalize_total":
        totals = x.sum(axis=1, keepdims=True)  # (n, 1)
        return x / np.where(totals > 0, totals, 1.0) * p["target_sum"]
    if name == "log1p":
        return np.log1p(np.clip(x, 0, None))
    if name == "standardize":
        return (x - p["mean"]) / p["std"]
    if name == "scale_unit_range":
        return (x - p["min"]) / max(p["max"] - p["min"], 1e-12)
    raise ValueError(name)


def apply_transforms(x, fitted):
    for name, p in fitted:
        x = apply_step(x, name, p)
    return x


def invert_transforms(x, fitted):
    """Map transformed values back to the raw scale where that is defined (used for flow samples,
    which live in transformed space). normalize_total can't be undone per row (the original total
    is unknown), so values stay on the normalized scale; this is noted in the evaluation."""
    for name, p in reversed(fitted):
        if name == "log1p":
            x = np.expm1(x)
        elif name == "standardize":
            x = x * p["std"] + p["mean"]
        elif name == "scale_unit_range":
            x = x * (p["max"] - p["min"]) + p["min"]
    return x


# ── Evaluation-space size ──
STABLE_VARIANCE_RATIO = 0.85


def stable_components(trans_train, k_max, seed):
    """Number of PCA components that generalize to unseen data, decided on TRAINING data only.

    PCA fit on n samples in d ~ n dimensions overfits: later components carry more variance on the
    data they were fit to than on new data, so fitted and new data become separable in that space
    even when they come from the same distribution (a classifier can then separate training from
    held-out samples well above chance). Split the training data in half, fit PCA on
    one half, and keep the largest k for which the other half retains >= 85% of the fitted half's
    cumulative variance. Cumulative (not per-component) because neighboring components with similar
    variance swap order between samples, which makes single-component ratios noisy.
    Returns (k, cumulative variance ratios for k = 1..k_max)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trans_train))
    a, b = trans_train[idx[: len(idx) // 2]], trans_train[idx[len(idx) // 2:]]
    pca = PCA(n_components=k_max, random_state=seed).fit(a)
    cum = np.cumsum(pca.transform(b).var(0)) / np.cumsum(pca.transform(a).var(0))  # (k_max,)
    ok = np.flatnonzero(cum >= STABLE_VARIANCE_RATIO)
    k = int(ok[-1]) + 1 if ok.size else 2
    return max(k, 2), cum


# ── State save/load ──

def save_state(path, fitted, pca):
    arrays = {"pca_mean": pca.mean_, "pca_components": pca.components_, "pca_evr": pca.explained_variance_ratio_}
    names = []
    for i, (name, p) in enumerate(fitted):
        names.append(name)
        for k, v in p.items():
            arrays[f"t{i}_{k}"] = np.asarray(v)
    arrays["transform_names"] = np.array(names)
    np.savez(path, **arrays)


def load_state(prepared_dir):
    s = np.load(Path(prepared_dir) / "state.npz", allow_pickle=False)
    fitted = []
    for i, name in enumerate(s["transform_names"].tolist()):
        p = {k.split("_", 1)[1]: (s[k].item() if s[k].ndim == 0 else s[k]) for k in s.files if k.startswith(f"t{i}_")}
        fitted.append((name, p))
    return {"fitted": fitted, "pca_mean": s["pca_mean"], "pca_components": s["pca_components"], "pca_evr": s["pca_evr"]}


def to_eval_space(trans, state):
    """Project transformed-space samples (n, d) into the shared evaluation PCA space (n, k)."""
    return (trans - state["pca_mean"]) @ state["pca_components"].T


def load_prepared(dataset, root=OUTPUTS):
    d = Path(root) / dataset / "prepared"
    out = {"meta": json.loads((d / "meta.json").read_text()), "state": load_state(d)}
    for name in ("raw", "trans", "eval", "y"):
        for split in ("train", "test"):
            f = d / f"{name}_{split}.npy"
            if f.exists():
                out[f"{name}_{split}"] = np.load(f, allow_pickle=False)
    return out


# ── Main pipeline ──

def prepare(plan, out_root=OUTPUTS):
    out_dir = Path(out_root) / plan["dataset"] / "prepared"
    h = prep_hash(plan)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("prep_hash") == h:
        print(f"[preprocess] reusing {out_dir} (same preprocessing hash {h})")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(plan["dataset"])  # (n, d_all) float64
    if np.isnan(X).any():
        raise ValueError("data has missing values; no preprocessing step handles them (see data-profiling skill)")
    rng = np.random.default_rng(plan["seed"])
    steps = plan["preprocessing"]

    # ── Step 1: row steps ──
    kept = np.arange(len(X))  # original row index of every kept sample
    for s in (s for s in steps if s["step"] in ROW_STEPS):
        n = int(s["params"]["n_samples"])
        if n < len(X):
            if s["step"] == "stratified_subsample":
                idx, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=plan["seed"])
            else:
                idx = rng.choice(len(X), size=n, replace=False)
            idx = np.sort(idx)
            X, y, kept = X[idx], (y[idx] if y is not None else None), kept[idx]

    # ── Step 2: train/test split (held-out real data; never used for fitting anything) ──
    strat = y if (plan["split"].get("stratify") and y is not None) else None
    tr, te = train_test_split(np.arange(len(X)), test_size=plan["split"]["test_fraction"],
                              stratify=strat, random_state=plan["seed"])
    tr, te = np.sort(tr), np.sort(te)

    # ── Step 3: feature selection, ranked on the transformed training data ──
    transform_names = [s["step"] for s in steps if s["step"] in TRANSFORM_STEPS]
    feat_idx = np.arange(X.shape[1])
    for s in (s for s in steps if s["step"] in FEATURE_STEPS):
        k = int(s["params"]["n_features"])
        ranked_on = apply_transforms(X[tr], fit_transforms(X[tr], transform_names))  # (n_train, d_all)
        var = ranked_on.var(axis=0)
        feat_idx = np.sort(np.argsort(-var, kind="stable")[:k])
    raw_tr, raw_te = X[tr][:, feat_idx], X[te][:, feat_idx]  # (n_train, d), (n_test, d)

    # ── Step 4: transforms, refit on the selected features of the training split ──
    fitted = fit_transforms(raw_tr, transform_names)
    trans_tr, trans_te = apply_transforms(raw_tr, fitted), apply_transforms(raw_te, fitted)

    # ── Step 5: shared evaluation space (PCA of transformed, fit on train, stable components only) ──
    k_max = int(min(EVAL_PCA_COMPONENTS, trans_tr.shape[1], trans_tr.shape[0] // 2 - 1))
    k_eval, var_ratios = stable_components(trans_tr, k_max, plan["seed"])
    pca = PCA(n_components=k_eval, random_state=plan["seed"]).fit(trans_tr)
    eval_tr, eval_te = pca.transform(trans_tr), pca.transform(trans_te)  # (n_train, k), (n_test, k)

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    for name, arr in (("raw_train", raw_tr), ("raw_test", raw_te), ("trans_train", trans_tr), ("trans_test", trans_te),
                      ("eval_train", eval_tr), ("eval_test", eval_te)):
        np.save(out_dir / f"{name}.npy", f32(arr))
    if y is not None:
        np.save(out_dir / "y_train.npy", np.asarray(y[tr]).astype(str))
        np.save(out_dir / "y_test.npy", np.asarray(y[te]).astype(str))
    save_state(out_dir / "state.npz", fitted, pca)
    # Rows the subsample left out: unused REAL data, used only by headroom_check.py's extra-real-data arm.
    np.save(out_dir / "kept_idx.npy", kept)

    names = metadata.get("feature_names")
    meta = {
        "prep_hash": h, "dataset": plan["dataset"],
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": int(len(feat_idx)),
        "eval_components": k_eval, "eval_components_max": k_max,
        "eval_components_rule": f"largest k where an unseen half of the training data keeps >= {STABLE_VARIANCE_RATIO} of the fitted half's cumulative PCA variance",
        "eval_half_split_cumulative_ratios": [round(float(r), 3) for r in var_ratios],
        "eval_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "transforms": transform_names, "image_shape": metadata.get("image_shape") and list(metadata["image_shape"]),
        "has_labels": y is not None,
        "classes": sorted(set(np.asarray(y).astype(str).tolist())) if y is not None else None, "is_count_raw": bool(np.all(raw_tr == np.round(raw_tr)) and raw_tr.min() >= 0),
        "feature_names": [names[i] for i in feat_idx] if names else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[preprocess] train {raw_tr.shape}, test {raw_te.shape}, eval PCA {k_eval} comps "
          f"({meta['eval_variance_explained']:.1%} variance) -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply plan.json's split and preprocessing; save prepared arrays.")
    parser.add_argument("plan_json")
    args = parser.parse_args()
    prepare(json.loads(Path(args.plan_json).read_text()))
