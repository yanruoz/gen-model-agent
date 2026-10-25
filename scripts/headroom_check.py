"""Headroom check for class_balancing / augmentation, run BEFORE choosing generative models:
   is there anything for added training data to gain on this dataset?

   Usage: venv/bin/python scripts/headroom_check.py outputs/<d>/plan.json
   (plan.json needs only split + preprocessing at this point; models may still be empty.)

   Trains the same classifier, with HEADROOM_SEEDS seeds each, on:
     real            the real training split
     resampled       real + target class(es) oversampled by duplication up to the largest class
     extra_real      real + that many NEW real target-class samples, taken from rows the subsample left out
                     (only when a row subsample was applied and enough unused rows exist)
   and scores all of them on the held-out real split. extra_real is the best case for any generator:
   a perfect generator can at most produce new samples from the same distribution. If extra_real does not
   beat resampling, no generator can be expected to.

   Writes outputs/<d>/metrics/headroom.json. downstream_test.py reuses these arms (same preprocessing hash).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from image_eval import class_index, train_classifier
from model_io import outputs, pick_device
from preprocess import apply_transforms, load_prepared, prepare

HEADROOM_SEEDS = (0, 1, 2)
CLASSIFIER_EPOCHS = 20


def _score(model, x_te, y_te, target_idx, n_classes):
    from image_eval import predict
    from sklearn.metrics import f1_score
    pred = predict(model, x_te)
    per_class = f1_score(y_te, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return {"accuracy": round(float((pred == y_te).mean()), 4), "macro_f1": round(float(per_class.mean()), 4),
            "target_f1": round(float(per_class[target_idx].mean()), 4)}


def target_classes(dataset, prepared):
    obj = json.loads((outputs(dataset) / "objective.json").read_text())
    t = (obj.get("target") or {}).get("class")
    classes, y_tr = prepared["meta"]["classes"], prepared["y_train"]
    counts = {c: int((y_tr == c).sum()) for c in classes}
    if t in (None, "auto"):
        return [min(counts, key=counts.get)], counts
    if t == "all":
        return [c for c in classes if counts[c] < max(counts.values())], counts
    return [str(t)], counts


def summarize(runs):
    """[{target_f1, macro_f1, accuracy}, ...] over seeds -> mean and sd per metric."""
    out = {"n_seeds": len(runs), "runs": runs}
    for k in ("target_f1", "macro_f1", "accuracy"):
        v = np.array([r[k] for r in runs])
        out[f"{k}_mean"], out[f"{k}_sd"] = round(float(v.mean()), 4), round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
    return out


def fit_arm(x, y, prepared, t_idx, seeds=HEADROOM_SEEDS, label=""):
    meta = prepared["meta"]
    classes, shape = meta["classes"], tuple(meta["image_shape"])
    x_te, y_te = prepared["trans_test"], class_index(prepared["y_test"], classes)
    runs = []
    for s in seeds:
        m, info = train_classifier(x, y, shape, len(classes), s, CLASSIFIER_EPOCHS, pick_device("image"), patience=None)
        r = _score(m, x_te, y_te, t_idx, len(classes))
        r.update(seed=s, n_train=int(len(x)))
        runs.append(r)
        print(f"[headroom] {label} seed {s}: target F1 {r['target_f1']}, macro F1 {r['macro_f1']}", flush=True)
    return summarize(runs)


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    dataset = plan["dataset"]
    prepare(plan)
    prepared = load_prepared(dataset)
    meta = prepared["meta"]
    if not meta.get("has_labels") or not meta.get("image_shape"):
        raise SystemExit("headroom_check needs labeled image data (the classifier is a CNN)")
    classes = meta["classes"]
    targets, counts = target_classes(dataset, prepared)
    t_idx = [classes.index(t) for t in targets]
    n_add = {t: max(counts.values()) - counts[t] for t in targets}
    x_tr, y_tr = prepared["trans_train"], class_index(prepared["y_train"], classes)
    rng = np.random.default_rng(0)
    out = {"prep_hash": meta["prep_hash"], "targets": targets, "train_class_counts": counts, "n_added_per_target": n_add,
           "classifier": f"SmallCNN trained for exactly {CLASSIFIER_EPOCHS} epochs (cosine learning-rate decay, final weights), "
                         f"{len(HEADROOM_SEEDS)} seeds per training set", "arms": {}}

    out["arms"]["real"] = fit_arm(x_tr, y_tr, prepared, t_idx, label="real")
    dup = np.concatenate([rng.choice(np.flatnonzero(y_tr == classes.index(t)), n_add[t], replace=True) for t in targets])
    out["arms"]["resampled"] = fit_arm(np.concatenate([x_tr, x_tr[dup]]), np.concatenate([y_tr, y_tr[dup]]), prepared, t_idx, label="resampled")

    # ── Extra real data from rows the subsample left out (never part of train or held-out) ──
    from loaders import load_dataset
    X_all, y_all, _ = load_dataset(dataset)
    unused = np.setdiff1d(np.arange(len(X_all)), np.load(outputs(dataset) / "prepared" / "kept_idx.npy"))
    picks = []
    for t in targets:
        pool = unused[np.asarray(y_all)[unused].astype(str) == t]
        if len(pool) < n_add[t]:
            picks = None
            break
        picks.append(rng.choice(pool, n_add[t], replace=False))
    if picks:
        idx = np.concatenate(picks)
        x_extra = apply_transforms(X_all[idx], prepared["state"]["fitted"]).astype(np.float32)
        y_extra = class_index(np.asarray(y_all)[idx].astype(str), classes)
        out["arms"]["extra_real"] = fit_arm(np.concatenate([x_tr, x_extra]), np.concatenate([y_tr, y_extra]), prepared, t_idx, label="extra_real")
    else:
        out["extra_real_note"] = "not run: no row subsample was applied, or too few unused real samples of the target class"
    del X_all

    a = out["arms"]
    noise = a["real"]["target_f1_sd"]
    best_name = "extra_real" if "extra_real" in a else "resampled"
    gain_resampled = round(a["resampled"]["target_f1_mean"] - a["real"]["target_f1_mean"], 4)
    gain_best = round(a[best_name]["target_f1_mean"] - a["real"]["target_f1_mean"], 4)
    out["summary"] = {
        "noise_sd_target_f1": noise,
        "gain_resampled_over_real": gain_resampled,
        "gain_extra_real_over_real": gain_best if best_name == "extra_real" else None,
        "extra_real_minus_resampled": round(a["extra_real"]["target_f1_mean"] - a["resampled"]["target_f1_mean"], 4) if "extra_real" in a else None,
        "reading": ("New real target-class data improves target F1 over resampling by more than the seed noise: a good "
                    "generator has room to help." if "extra_real" in a and
                    a["extra_real"]["target_f1_mean"] - a["resampled"]["target_f1_mean"] > 2 * max(noise, a["resampled"]["target_f1_sd"])
                    else "New real target-class data does not beat resampling by more than the seed noise (or could not be "
                         "tested): a generator trained on this data is not expected to beat resampling."),
    }
    out["definitions"] = {
        "target_f1": "F1 score for the target class(es) on held-out real data; mean and sd over seeds.",
        "extra_real": "Real training data plus genuinely new real target-class samples from rows the subsample left out: the best case for any generator.",
        "noise_sd_target_f1": "Standard deviation of target F1 across seeds of the real-only classifier: differences smaller than about twice this are not meaningful.",
    }
    (outputs(dataset) / "metrics").mkdir(exist_ok=True)
    (outputs(dataset) / "metrics" / "headroom.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[headroom] {out['summary']['reading']}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom check before choosing generative models for balancing/augmentation.")
    parser.add_argument("plan_json")
    run(parser.parse_args().plan_json)
