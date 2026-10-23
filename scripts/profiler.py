"""Dataset profiler (copied from Project 1, extended with modality, count-data, and class-balance fields).
   Fully generic: takes the common (X, y, metadata)
   interface from loaders.py and computes summary statistics. This file
   should never contain dataset-specific branching.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from loaders import load_dataset


TOP_MISSING_FEATURES = 5
# Rows checked for "all values are non-negative integers"; a subsample keeps this cheap on large image sets.
COUNT_CHECK_ROWS = 2000


def _missing_details(missing, metadata):
    """Per-feature/sample missing-value summary. `missing` is the boolean
    (n_samples, n_features) mask; only computed when something is missing."""
    n_samples, n_features = missing.shape
    per_feature = missing.mean(axis=0)  # (n_features,) fraction missing per feature
    names = (metadata or {}).get("feature_names")
    if not (isinstance(names, (list, tuple)) and len(names) == n_features):
        names = list(range(n_features))
    worst = np.argsort(-per_feature, kind="stable")[:TOP_MISSING_FEATURES]
    return {
        "n_features_with_missing": int((per_feature > 0).sum()),
        "n_samples_with_missing": int(missing.any(axis=1).sum()),
        "n_features_all_missing": int((per_feature == 1.0).sum()),
        "max_feature_missing_fraction": round(float(per_feature.max()), 4),
        "top_missing_features": [
            {"feature": names[i] if isinstance(names[i], int) else str(names[i]),
             "missing_fraction": round(float(per_feature[i]), 4)}
            for i in worst
            if per_feature[i] > 0
        ],
    }


def profile_dataset(X, y=None, metadata=None):
    n_samples, n_features = X.shape

    missing = np.isnan(X)  # (n_samples, n_features)
    n_missing = int(np.count_nonzero(missing))
    n_observed = X.size - n_missing

    if n_missing == 0:
        sparsity = float(np.mean(X == 0))
        value_range = [float(np.min(X)), float(np.max(X))]
        mean, std = float(np.mean(X)), float(np.std(X))
    elif n_observed == 0:
        # Every value missing: no statistics exist; report null rather than nan (invalid JSON).
        sparsity, value_range, mean, std = None, None, None, None
    else:
        # Statistics over observed values only, so one NaN doesn't poison every number
        # (and sparsity is the share of zeros among observed entries, not among all entries).
        sparsity = float(np.count_nonzero(X == 0)) / n_observed
        value_range = [float(np.nanmin(X)), float(np.nanmax(X))]
        mean, std = float(np.nanmean(X)), float(np.nanstd(X))

    profile = {
        "n_samples": n_samples,
        "n_features": n_features,
        "sparsity": None if sparsity is None else round(sparsity, 4),
        "has_missing": n_missing > 0,
        "n_missing_cells": n_missing,
        "missing_fraction": round(n_missing / max(X.size, 1), 6),
        "value_range": value_range,
        "mean": mean,
        "std": std,
        "has_labels": y is not None,
    }
    if n_missing > 0:
        profile.update(_missing_details(missing, metadata))

    # Modality drives model choice: image if the loader found or was told an image shape, else tabular.
    image_shape = (metadata or {}).get("image_shape")
    profile["modality"] = "image" if image_shape else "tabular"
    if image_shape:
        profile["image_shape"] = [int(v) for v in image_shape]

    # Count data (non-negative integers) suggests a count likelihood (e.g. negative binomial) over a Gaussian one.
    # Pixel intensities 0-255 also pass this check, so the model-selection skill reads it together with modality.
    rows = X[:COUNT_CHECK_ROWS]  # (<=2000, n_features)
    obs = rows[~np.isnan(rows)]
    profile["is_nonneg_integer"] = bool(obs.size and np.all(obs >= 0) and np.all(obs == np.round(obs)))

    if y is not None:
        classes, counts = np.unique(y, return_counts=True)
        profile["n_classes"] = int(len(classes))
        profile["class_counts"] = {str(c): int(n) for c, n in zip(classes, counts)}
        # max/min class size; 1.0 means perfectly balanced
        profile["imbalance_ratio"] = round(float(counts.max() / counts.min()), 3)
        profile["minority_class"] = str(classes[np.argmin(counts)])

    if metadata:
        # Long lists (e.g. tens of thousands of feature names) would make profile.json too large to read; keep a count and examples.
        profile["metadata"] = {
            k: ({"n": len(v), "first": [str(x) for x in v[:5]]} if isinstance(v, (list, tuple)) and len(v) > 20 else v)
            for k, v in metadata.items()
        }

    return profile


def _fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def _missing_text(profile):
    if not profile["has_missing"]:
        return "No"
    worst = profile.get("top_missing_features") or []
    text = (
        f"Yes: {profile['missing_fraction'] * 100:.2f}% of entries ({profile['n_missing_cells']:,}); "
        f"{profile['n_features_with_missing']:,} of {profile['n_features']:,} features and "
        f"{profile['n_samples_with_missing']:,} of {profile['n_samples']:,} samples affected"
    )
    if worst:
        text += f"; worst feature {worst[0]['feature']!r} at {worst[0]['missing_fraction'] * 100:.1f}%"
    if profile.get("n_features_all_missing"):
        text += f"; {profile['n_features_all_missing']} feature(s) entirely missing"
    return text


def _range_text(profile):
    vr = profile["value_range"]
    return "n/a" if vr is None else f"[{vr[0]:.2f}, {vr[1]:.2f}]"


def print_profile(name, profile):
    print(f"\n{'='*50}")
    print(f"  Dataset: {name}")
    print(f"{'='*50}")
    print(f"  Samples:        {profile['n_samples']:,}")
    print(f"  Features:       {profile['n_features']:,}")
    sp = profile["sparsity"]
    print(f"  Sparsity:       {'n/a' if sp is None else f'{sp*100:.1f}% zeros'}")
    print(f"  Missing values: {_missing_text(profile)}")
    print(f"  Value range:    {_range_text(profile)}")
    print(f"  Mean / Std:     {_fmt(profile['mean'], '.3f')} / {_fmt(profile['std'], '.3f')}")
    print(f"  Has labels:     {'Yes (' + str(profile['n_classes']) + ' classes)' if profile['has_labels'] else 'No'}")
    print(f"  Modality:       {profile['modality']}" + (f" {tuple(profile['image_shape'])}" if profile.get("image_shape") else ""))
    print(f"  Non-neg ints:   {profile['is_nonneg_integer']}")
    if profile.get("imbalance_ratio") is not None:
        print(f"  Imbalance:      {profile['imbalance_ratio']}x (minority class {profile['minority_class']})")

    if "metadata" in profile:
        print(f"  {'-'*46}")
        for k, v in profile["metadata"].items():
            if isinstance(v, (list, tuple)) and len(v) > 5:
                v_display = f"[{v[0]!r}, {v[1]!r}, ... ({len(v)} total)]"
            else:
                v_display = v
            print(f"  {k}: {v_display}")
    print(f"{'='*50}\n")


def write_profile_summary_md(name, profile, path):
    """Render a profile dict as a short Markdown summary, a quick-to-skim
    companion to profile.json (same JSON+MD pairing as validate_plan.py's
    reports)."""
    lines = [f"# Profile summary: {name}", ""]
    lines.append(f"- **Samples:** {profile['n_samples']:,}")
    lines.append(f"- **Features:** {profile['n_features']:,}")
    sp = profile["sparsity"]
    lines.append(f"- **Sparsity:** {'n/a' if sp is None else f'{sp * 100:.1f}% zeros'}")
    lines.append(f"- **Missing values:** {_missing_text(profile)}")
    if profile.get("top_missing_features"):
        worst = ", ".join(f"`{t['feature']}` {t['missing_fraction'] * 100:.1f}%" for t in profile["top_missing_features"])
        lines.append(f"- **Most-missing features:** {worst}")
    lines.append(f"- **Value range:** {_range_text(profile)}")
    lines.append(f"- **Mean / Std:** {_fmt(profile['mean'], '.3f')} / {_fmt(profile['std'], '.3f')}")
    if profile["has_labels"]:
        lines.append(f"- **Labels:** Yes ({profile['n_classes']} classes)")
    else:
        lines.append("- **Labels:** No")
    shape = f" {tuple(profile['image_shape'])}" if profile.get("image_shape") else ""
    lines.append(f"- **Modality:** {profile['modality']}{shape}")
    lines.append(f"- **All values non-negative integers:** {profile['is_nonneg_integer']}")
    if profile.get("imbalance_ratio") is not None:
        lines.append(f"- **Class imbalance (max/min):** {profile['imbalance_ratio']}x, minority class `{profile['minority_class']}`")
        lines.append(f"- **Class counts:** " + ", ".join(f"`{k}`: {v:,}" for k, v in profile["class_counts"].items()))

    if "metadata" in profile:
        lines.append("")
        lines.append("## Metadata")
        for k, v in profile["metadata"].items():
            if isinstance(v, (list, tuple)) and len(v) > 5:
                v_display = f"`{v[0]!r}`, `{v[1]!r}`, ... ({len(v)} total)"
            else:
                v_display = v
            lines.append(f"- **{k}:** {v_display}")

    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile a dataset; writes profile.json + profile_summary.md."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name: a folder under data/")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write profile.json (default: outputs/<dataset>/profile.json)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("outputs") / args.dataset / "profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(args.dataset)
    profile = profile_dataset(X, y, metadata)
    print_profile(args.dataset, profile)

    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, allow_nan=False)

    md_path = out_path.with_name("profile_summary.md")
    write_profile_summary_md(args.dataset, profile, md_path)

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")