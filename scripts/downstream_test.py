"""Objective-specific test for class_balancing / augmentation on labeled data: does adding synthetic
   samples to the real training set help a classifier on HELD-OUT REAL data, compared with doing nothing
   and with simple resampling?

   Usage: venv/bin/python scripts/downstream_test.py --dataset <d> --keys k1,k2,...

   Training sets (same classifier, same epochs, scored on the held-out real split):
     real                      real training data only, trained with two seeds: the spread between them
                               is the noise level a difference must clear to mean anything
     real + resampled          target class(es) oversampled by duplication up to the largest class
     real + synthetic (<key>)  target class(es) topped up to the largest class with that model's samples
   Target classes come from objective.json (target.class; "auto" or null = the smallest training class).
   Writes outputs/<d>/metrics/downstream.json; synthetic data never enters the held-out split.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from image_eval import class_index, predict, train_classifier
from model_io import outputs, pick_device
from preprocess import load_prepared

DOWNSTREAM_SEEDS = (0, 1)
DOWNSTREAM_EPOCHS = 20


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


def _score(model, x_te, y_te, target_idx, n_classes):
    pred = predict(model, x_te)
    per_class = f1_score(y_te, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return {"accuracy": round(float((pred == y_te).mean()), 4),
            "macro_f1": round(float(per_class.mean()), 4),
            "target_f1": round(float(per_class[target_idx].mean()), 4),
            "per_class_f1": [round(float(v), 4) for v in per_class]}


def run(dataset, keys):
    prepared = load_prepared(dataset)
    meta = prepared["meta"]
    classes, shape = meta["classes"], tuple(meta["image_shape"])
    x_tr, y_tr = prepared["trans_train"], class_index(prepared["y_train"], classes)
    x_te, y_te = prepared["trans_test"], class_index(prepared["y_test"], classes)
    targets, counts = target_classes(dataset, prepared)
    t_idx = [classes.index(t) for t in targets]
    n_add = {t: max(counts.values()) - counts[t] for t in targets}
    device = pick_device("image")
    rng = np.random.default_rng(0)
    out = {"targets": targets, "train_class_counts": counts, "n_added_per_target": n_add,
           "classifier": f"SmallCNN trained for exactly {DOWNSTREAM_EPOCHS} epochs (cosine learning-rate decay, final weights) on each training set",
           "results": {}}

    def fit_and_score(name, x, y, seed):
        # Fixed epochs, final weights: identical training for every training set (no early stopping).
        model, info = train_classifier(x, y, shape, len(classes), seed, DOWNSTREAM_EPOCHS, device, patience=None)
        res = _score(model, x_te, y_te, t_idx, len(classes))
        res.update(n_train=int(len(x)), seed=seed, epochs_run=info["epochs_run"])
        out["results"][name] = res
        print(f"[downstream] {name}: target F1 {res['target_f1']}, macro F1 {res['macro_f1']}, accuracy {res['accuracy']}")

    # ── Baselines ──
    for s in DOWNSTREAM_SEEDS:
        fit_and_score(f"real_seed{s}", x_tr, y_tr, s)
    extra = [rng.choice(np.flatnonzero(y_tr == classes.index(t)), size=n_add[t], replace=True) for t in targets]
    idx = np.concatenate([np.arange(len(x_tr)), *extra])
    fit_and_score("real_plus_resampled", x_tr[idx], y_tr[idx], DOWNSTREAM_SEEDS[0])

    # ── Real + synthetic target-class samples, per model ──
    for key in keys:
        f_x, f_y = outputs(dataset) / "samples" / f"{key}_balance_trans.npy", outputs(dataset) / "samples" / f"{key}_balance_y.npy"
        if not f_x.exists():
            out["results"][f"real_plus_{key}"] = {"status": "skipped: model produced no class-conditional balancing samples"}
            continue
        xs, ys = np.load(f_x), class_index(np.load(f_y).astype(str), classes)
        fit_and_score(f"real_plus_{key}", np.concatenate([x_tr, xs]), np.concatenate([y_tr, ys]), DOWNSTREAM_SEEDS[0])

    r = out["results"]
    seeds = [r[f"real_seed{s}"]["target_f1"] for s in DOWNSTREAM_SEEDS]
    out["noise_level_target_f1"] = round(abs(seeds[0] - seeds[1]), 4)
    out["notes"] = ("noise_level_target_f1 is the target-class F1 difference between two real-only runs that differ only in the "
                    "classifier's seed. A change smaller than this is indistinguishable from training noise. Two runs are a rough "
                    "yardstick, not a confidence interval.")
    out["definitions"] = {
        "target_f1": "F1 score (harmonic mean of precision and recall) for the target class(es), on held-out real data.",
        "macro_f1": "Unweighted mean of per-class F1 on held-out real data; checks that helping the target class did not hurt others.",
        "noise_level_target_f1": "Target-class F1 difference between two real-only classifier runs with different seeds.",
    }
    (outputs(dataset) / "metrics" / "downstream.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream class-balancing test on held-out real data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keys", required=True, help="comma-separated model keys whose balancing samples to test")
    args = parser.parse_args()
    run(args.dataset, [k for k in args.keys.split(",") if k])
