"""Figures for one dataset, from artifacts only (runs/, samples/, latents/, prepared/).

   Usage: venv/bin/python scripts/visualize.py --dataset <d> --keys k1,k2,...

   Writes outputs/<d>/figures/:
     loss_curves.png            train / validation / held-out loss per model, kept epoch marked
     real_vs_synthetic.png      evaluation-space PC1 vs PC2, held-out real (gray) vs each model
     feature_stats.png          per-feature mean and variance, synthetic vs held-out real
     count_properties.png       total counts per sample and per-feature zero fraction (count models only)
     latent_<key>.png           2-D PCA of a VAE's latent codes for held-out samples
   Image data instead gets:
     sample_grid.png            per class (rows): real held-out images, then each model's samples
     loss_curves.png            as above; GANs show discriminator and generator losses
     label_agreement.png        per class: how often the real-data classifier assigns the requested class
     objective_tests.png        objective tests (balancing / train-on-synthetic / anomaly / imputation), if run
   Colors: held-out real data is always neutral gray; models take fixed categorical slots in plan
   order (never cycled), so a model keeps its color across every figure.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from model_io import outputs
from preprocess import load_prepared, to_eval_space

# Reference categorical palette (dataviz skill, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REAL = "#8a8a85"
INK, INK_MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.axisbelow": True,
})


def _color(i):
    return SERIES[i % len(SERIES)] if i < len(SERIES) else "#555555"


def _imgs(flat, shape):
    return np.clip(flat.reshape(-1, *shape), 0, 1)


def sample_grid(dataset, keys, prepared, fig_dir, per=6):
    shape = tuple(prepared["meta"]["image_shape"])
    classes = prepared["meta"].get("classes") or ["all"]
    y_te = prepared.get("y_test")
    sources = [("held-out real", prepared["trans_test"], y_te)]
    for k in keys:
        yk = outputs(dataset) / "samples" / f"{k}_y.npy"
        sources.append((k, np.load(outputs(dataset) / "samples" / f"{k}_trans.npy"), np.load(yk) if yk.exists() else None))
    rows, cols = len(classes), len(sources)
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * per * cols / 3, 1.55 * rows / 1.5 + 0.6), squeeze=False)
    for j, (name, x, y) in enumerate(sources):
        for i, c in enumerate(classes):
            ax = axes[i][j]
            idx = np.flatnonzero(y == c)[:per] if y is not None else np.arange(i * per, (i + 1) * per)
            if len(idx):
                tile = np.concatenate(list(_imgs(x[idx], shape)), axis=1)  # one row of `per` images
                ax.imshow(tile, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(f"class {c}", fontsize=7, rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title(name, fontsize=8, color=INK)
    fig.suptitle("Held-out real images and generated images, by class (rows)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "sample_grid.png")
    plt.close(fig)


def label_agreement(dataset, keys, colors, fig_dir):
    ms = {k: json.loads((outputs(dataset) / "metrics" / f"{k}.json").read_text()) for k in keys
          if (outputs(dataset) / "metrics" / f"{k}.json").exists()}
    ms = {k: m["conditional"] for k, m in ms.items() if "conditional" in m}
    if not ms:
        return
    classes = list(next(iter(ms.values()))["per_class_agreement"])
    ref = next(iter(ms.values())).get("per_class_heldout_accuracy", {})
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    w = 0.8 / (len(ms) + 1)
    xs = np.arange(len(classes))
    ax.bar(xs - 0.4 + w / 2, [ref.get(c, 0) for c in classes], width=w * 0.92, color=REAL, label="held-out real (reference)")
    for i, (k, m) in enumerate(ms.items(), 1):
        ax.bar(xs - 0.4 + w / 2 + i * w, [m["per_class_agreement"][c] for c in classes], width=w * 0.92, color=colors[k], label=k)
    ax.set_xticks(xs, [f"class {c}" for c in classes], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("classifier assigns requested class")
    ax.legend(fontsize=7, ncol=len(ms) + 1, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Label agreement: does a classifier trained on real images recognize the requested class?",
                 fontsize=9, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(fig_dir / "label_agreement.png")
    plt.close(fig)


def _arm_color(name, colors):
    for k, c in colors.items():
        if name.endswith(k):
            return c
    return REAL


def objective_tests(dataset, colors, fig_dir):
    """One panel per objective test that ran: balancing / train-on-synthetic (mean +/- sd over seeds),
    anomaly AUROC per score, imputation RMSE per imputer."""
    m = outputs(dataset) / "metrics"
    load = lambda n: json.loads((m / n).read_text()) if (m / n).exists() else None
    d, an, im = load("downstream.json") or {}, load("anomaly.json"), load("imputation.json")
    panels = []
    if d.get("balancing"):
        b = d["balancing"]
        panels.append((f"Class balancing: target-class F1 (class {', '.join(b['targets'])})",
                       [(n, r["target_f1_mean"], r["target_f1_sd"]) for n, r in b["arms"].items()], "F1 on held-out real images"))
    if d.get("tstr"):
        panels.append(("Train on synthetic, test on real: accuracy",
                       [(n, r["accuracy_mean"], r["accuracy_sd"]) for n, r in d["tstr"]["arms"].items()], "accuracy on held-out real images"))
    if an:
        panels.append(("Anomaly detection: mean AUROC over corruption types",
                       [(n, r["auroc_mean"], 0) for n, r in an["results"].items() if "auroc_mean" in r], "AUROC (0.5 = chance)"))
    if im:
        panels.append((f"Imputation: RMSE on masked entries ({im['error_scale']})",
                       [(n, r["rmse"], 0) for n, r in im["results"].items() if "rmse" in r], "RMSE (lower is better)"))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.6, sum(0.42 * len(p[1]) + 1.3 for p in panels)), squeeze=False)
    for ax, (title, rows, xlabel) in zip(axes[:, 0], panels):
        names = [r[0] for r in rows]
        vals = np.array([r[1] for r in rows]); sds = np.array([r[2] for r in rows])
        ax.barh(range(len(rows)), vals, xerr=np.where(sds > 0, sds, np.nan), color=[_arm_color(n, colors) for n in names],
                height=0.6, error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.4f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(rows)), names, fontsize=7); ax.invert_yaxis()
        lo, hi = float(np.min(vals - sds)), float(np.max(vals + sds))
        pad = 0.15 * (hi - lo + 1e-3)
        ax.set_xlim(max(0, lo - 3 * pad), hi + 3 * pad)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "objective_tests.png")
    plt.close(fig)


def loss_curves(dataset, keys, colors, fig_dir):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        run = json.loads((outputs(dataset) / "runs" / f"{key}.json").read_text())
        h = run["history"]
        ep = [r["epoch"] for r in h]
        if run["family"] == "gan":
            ax.plot(ep, [r["d_loss"] for r in h], color=colors[key], lw=1.6, label="discriminator")
            ax.plot(ep, [r["g_loss"] for r in h], color=colors[key], lw=1.6, ls="--", label="generator")
            ax.set_title(key, fontsize=9, color=INK); ax.set_xlabel("epoch"); ax.set_ylabel(run["loss_definition"])
            ax.legend(fontsize=7, loc="upper right")
            continue
        ax.plot(ep, [r["loss"] for r in h], color=colors[key], lw=1.6, label="train")
        ax.plot(ep, [r["val_loss"] for r in h], color=colors[key], lw=1.6, ls="--", label="validation (from train)")
        ho = [(r["epoch"], r["heldout_loss"]) for r in h if "heldout_loss" in r]
        ax.plot(*zip(*ho), color=REAL, lw=1.6, label="held-out (reporting only)")
        ax.axvline(run["best_epoch"], color=INK_MUTED, lw=0.8, ls=":")
        ax.annotate(f"kept epoch {run['best_epoch']}", (run["best_epoch"], 0.03), xycoords=("data", "axes fraction"),
                    fontsize=7, color=INK_MUTED, ha="left", va="bottom", xytext=(3, 0), textcoords="offset points")
        vals = [r["loss"] for r in h] + [r["val_loss"] for r in h]
        lo, hi = np.percentile(vals, [1, 90])  # clip the first epochs' spike so the end of training is readable
        ax.set_ylim(lo - 0.05 * (hi - lo), hi)
        ax.set_title(key, fontsize=9, color=INK)
        ax.set_xlabel("epoch")
        ax.set_ylabel(run["loss_definition"])
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Training curves", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png")
    plt.close(fig)


def real_vs_synthetic(dataset, keys, colors, prepared, fig_dir):
    image = bool(prepared["meta"].get("image_shape"))
    if image:
        # Images are evaluated in the evaluation classifier's feature space, so plot PCs of those features
        # (PCA fit on held-out real features), not of raw pixels.
        from image_eval import features, get_evaluator
        evaluator, _ = get_evaluator(dataset, prepared)
        f_te = features(evaluator, prepared["trans_test"])
        pca2 = PCA(n_components=2, random_state=0).fit(f_te)
        ev_te = pca2.transform(f_te)
        project = lambda k: pca2.transform(features(evaluator, np.load(outputs(dataset) / "samples" / f"{k}_trans.npy")))
        axis_name, title = "feature PC", "Held-out real vs synthetic images, first two PCs of the evaluation classifier's features"
    else:
        ev_te = prepared["eval_test"]
        project = lambda k: to_eval_space(np.load(outputs(dataset) / "samples" / f"{k}_trans.npy").astype(np.float64), prepared["state"])
        axis_name, title = "eval PC", "Held-out real vs synthetic samples, first two evaluation PCs"
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.5), squeeze=False, sharex=True, sharey=True)
    for ax, key in zip(axes[0], keys):
        syn = project(key)
        ax.scatter(ev_te[:, 0], ev_te[:, 1], s=6, c=REAL, alpha=0.55, lw=0, label="held-out real")
        ax.scatter(syn[:, 0], syn[:, 1], s=6, c=colors[key], alpha=0.55, lw=0, label=f"synthetic ({key})")
        ax.set_title(key, fontsize=9, color=INK)
        ax.set_xlabel(f"{axis_name}1")
        ax.legend(fontsize=7, loc="best", markerscale=2)
    axes[0][0].set_ylabel(f"{axis_name}2")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "real_vs_synthetic.png")
    plt.close(fig)


def feature_stats(dataset, keys, colors, prepared, fig_dir):
    te = prepared["trans_test"]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 6.6), squeeze=False)
    for j, key in enumerate(keys):
        syn = np.load(outputs(dataset) / "samples" / f"{key}_trans.npy")
        for i, (stat, name) in enumerate(((np.mean, "mean"), (np.var, "variance"))):
            ax = axes[i][j]
            r, s = stat(te, axis=0), stat(syn, axis=0)
            ax.scatter(r, s, s=5, c=colors[key], alpha=0.5, lw=0)
            lim = [min(r.min(), s.min()), max(r.max(), s.max())]
            ax.plot(lim, lim, color=INK_MUTED, lw=0.8, ls="--")
            ax.set_xlabel(f"held-out real, per-feature {name}")
            ax.set_ylabel(f"synthetic, per-feature {name}")
            ax.set_title(f"{key}: {name}", fontsize=9, color=INK)
    fig.suptitle("Per-feature statistics (transformed scale); dashed line = perfect agreement",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_stats.png")
    plt.close(fig)


def count_properties(dataset, keys, colors, prepared, fig_dir):
    count_keys = [k for k in keys if (outputs(dataset) / "samples" / f"{k}_raw.npy").exists()]
    if not count_keys or not prepared["meta"]["is_count_raw"]:
        return
    raw_te = prepared["raw_test"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    bins = np.linspace(np.log10(max(raw_te.sum(1).min(), 1)) - 0.3, np.log10(raw_te.sum(1).max()) + 0.3, 40)
    axes[0].hist(np.log10(raw_te.sum(1) + 1), bins=bins, color=REAL, alpha=0.6, label="held-out real")
    for k in count_keys:
        raw = np.load(outputs(dataset) / "samples" / f"{k}_raw.npy")
        axes[0].hist(np.log10(raw.sum(1) + 1), bins=bins, histtype="step", lw=1.6, color=colors[k], label=k)
        axes[1].scatter((raw_te == 0).mean(0), (raw == 0).mean(0), s=5, c=colors[k], alpha=0.5, lw=0, label=k)
    axes[0].set_xlabel("log10 total counts per sample (selected features)")
    axes[0].set_ylabel("samples")
    axes[0].legend(fontsize=7)
    axes[1].plot([0, 1], [0, 1], color=INK_MUTED, lw=0.8, ls="--")
    axes[1].set_xlabel("held-out real: fraction of zeros per feature")
    axes[1].set_ylabel("synthetic: fraction of zeros per feature")
    axes[1].legend(fontsize=7, markerscale=2)
    fig.suptitle("Count-level properties", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "count_properties.png")
    plt.close(fig)


def latent_plots(dataset, keys, prepared, fig_dir):
    for key in keys:
        path = outputs(dataset) / "latents" / f"{key}_test.npy"
        if not path.exists():
            continue
        lat = np.load(path)
        xy = PCA(n_components=2, random_state=0).fit_transform(lat)  # (n_test, 2)
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        y = prepared.get("y_test")
        if y is not None:
            for i, c in enumerate(sorted(set(y))):
                m = y == c
                ax.scatter(xy[m, 0], xy[m, 1], s=6, lw=0, alpha=0.7, c=_color(i), label=str(c))
            ax.legend(fontsize=6, markerscale=2, ncol=2, title="label")
            subtitle = "colored by label (labels were not used to train this model)"
        else:
            # No labels: color by sequencing depth, the main technical factor a latent space should not be driven by.
            depth = np.log10(prepared["raw_test"].sum(1) + 1)
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=6, lw=0, c=depth, cmap="Blues")
            fig.colorbar(sc, ax=ax, label="log10 total counts")
            subtitle = "colored by total counts (no labels in this dataset)"
        ax.set_xlabel("latent PC1"); ax.set_ylabel("latent PC2")
        ax.set_title(f"{key}: latent codes of held-out samples\n{subtitle}", fontsize=8.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"latent_{key}.png")
        plt.close(fig)


def visualize(dataset, keys):
    prepared = load_prepared(dataset)
    fig_dir = outputs(dataset) / "figures"
    fig_dir.mkdir(exist_ok=True)
    colors = {k: _color(i) for i, k in enumerate(keys)}
    loss_curves(dataset, keys, colors, fig_dir)
    if prepared["meta"].get("image_shape"):
        sample_grid(dataset, keys, prepared, fig_dir)
        label_agreement(dataset, keys, colors, fig_dir)
        objective_tests(dataset, colors, fig_dir)
        real_vs_synthetic(dataset, keys, colors, prepared, fig_dir)
        latent_plots(dataset, keys, prepared, fig_dir)
        print(f"[visualize] figures -> {fig_dir}")
        return
    real_vs_synthetic(dataset, keys, colors, prepared, fig_dir)
    objective_tests(dataset, colors, fig_dir)
    feature_stats(dataset, keys, colors, prepared, fig_dir)
    count_properties(dataset, keys, colors, prepared, fig_dir)
    latent_plots(dataset, keys, prepared, fig_dir)
    print(f"[visualize] figures -> {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make the figures for one dataset's trained models.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keys", required=True, help="comma-separated <model_id>_s<seed> keys, in plan order")
    args = parser.parse_args()
    visualize(args.dataset, args.keys.split(","))
