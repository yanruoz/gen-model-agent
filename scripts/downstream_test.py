"""Objective tests on labeled image data, scored on HELD-OUT REAL data, 3 classifier seeds per arm.

   Usage: venv/bin/python scripts/downstream_test.py --dataset <d> --keys k1,k2,...

   class_balancing / augmentation (does adding synthetic target-class samples help?):
     baseline arms real / resampled / extra_real come from headroom_check.py (metrics/headroom.json,
     same preprocessing hash; computed here if missing), plus
     real_plus_<key>   target class(es) topped up to the largest class with that model's samples
   data_generation / augmentation (could synthetic data replace the real data? train on synthetic, test on real):
     real_same_size    real training samples, as many as the synthetic set
     synthetic_<key>   that model's labeled synthetic set only (training class mix)
   Writes outputs/<d>/metrics/downstream.json; synthetic data never enters the held-out split.
"""
import argparse
import json

import numpy as np

from headroom_check import HEADROOM_SEEDS, fit_arm, target_classes  # noqa: F401  (target_classes used by generate.py)
from image_eval import class_index
from model_io import outputs
from preprocess import load_prepared


def run(dataset, keys):
    prepared = load_prepared(dataset)
    meta = prepared["meta"]
    classes = meta["classes"]
    obj = json.loads((outputs(dataset) / "objective.json").read_text())
    tasks = {obj.get("effective_task"), *obj.get("secondary_tasks", [])}
    x_tr, y_tr = prepared["trans_train"], class_index(prepared["y_train"], classes)
    out = {"tasks": sorted(t for t in tasks if t), "seeds_per_arm": len(HEADROOM_SEEDS)}
    samples = outputs(dataset) / "samples"

    # ── Class balancing: real + synthetic target-class samples ──
    if tasks & {"class_balancing", "augmentation"}:
        h_path = outputs(dataset) / "metrics" / "headroom.json"
        if not h_path.exists() or json.loads(h_path.read_text()).get("prep_hash") != meta["prep_hash"]:
            from headroom_check import run as headroom
            headroom(outputs(dataset) / "plan.json")
        h = json.loads(h_path.read_text())
        targets = h["targets"]
        t_idx = [classes.index(t) for t in targets]
        out["balancing"] = {"targets": targets, "n_added_per_target": h["n_added_per_target"], "classifier": h["classifier"],
                            "noise_sd_target_f1": h["summary"]["noise_sd_target_f1"], "arms": dict(h["arms"])}
        for key in keys:
            fx, fy = samples / f"{key}_balance_trans.npy", samples / f"{key}_balance_y.npy"
            if not fx.exists():
                continue
            xs, ys = np.load(fx), class_index(np.load(fy).astype(str), classes)
            out["balancing"]["arms"][f"real_plus_{key}"] = fit_arm(np.concatenate([x_tr, xs]), np.concatenate([y_tr, ys]),
                                                                   prepared, t_idx, label=f"real_plus_{key}")

    # ── Train on synthetic, test on real ──
    if tasks & {"data_generation", "augmentation"}:
        tstr_keys = [k for k in keys if (samples / f"{k}_tstr_trans.npy").exists()]
        if tstr_keys:
            n_t = len(np.load(samples / f"{tstr_keys[0]}_tstr_y.npy"))
            idx = np.random.default_rng(0).choice(len(x_tr), size=n_t, replace=False)
            all_idx = list(range(len(classes)))
            arms = {"real_same_size": fit_arm(x_tr[idx], y_tr[idx], prepared, all_idx, label="real_same_size")}
            for key in tstr_keys:
                xs = np.load(samples / f"{key}_tstr_trans.npy")
                ys = class_index(np.load(samples / f"{key}_tstr_y.npy").astype(str), classes)
                arms[f"synthetic_{key}"] = fit_arm(xs, ys, prepared, all_idx, label=f"synthetic_{key}")
            base = arms["real_same_size"]["accuracy_mean"]
            out["tstr"] = {"n_train": n_t, "arms": arms, "noise_sd_accuracy": arms["real_same_size"]["accuracy_sd"],
                           "utility_ratio": {k: round(v["accuracy_mean"] / base, 4) for k, v in arms.items() if k != "real_same_size"}}
    out["definitions"] = {
        "target_f1": "F1 score for the target class(es) on held-out real data; mean and sd over classifier seeds.",
        "macro_f1": "Unweighted mean of per-class F1 on held-out real data.",
        "utility_ratio": "Train-on-synthetic, test-on-real accuracy divided by the accuracy of the same classifier trained on the same number of real samples. 1 = synthetic data is as useful as real data for this classifier.",
        "noise_sd": "Standard deviation across classifier seeds; differences smaller than about twice this are not meaningful.",
    }
    (outputs(dataset) / "metrics" / "downstream.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Objective tests on held-out real data (balancing, train-on-synthetic).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keys", required=True, help="comma-separated model keys")
    args = parser.parse_args()
    run(args.dataset, [k for k in args.keys.split(",") if k])
