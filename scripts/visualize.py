"""Figures for one dataset, from artifacts only (runs/, samples/, latents/, prepared/).

   Usage: venv/bin/python scripts/visualize.py --dataset <d> --keys k1,k2,...

   Writes outputs/<d>/figures/:
     loss_curves.png            train / validation / held-out loss per model, kept epoch marked
     real_vs_synthetic.png      evaluation-space PC1 vs PC2, held-out real (gray) vs each model
     feature_stats.png          per-feature mean and variance, synthetic vs held-out real
     count_properties.png       total counts per sample and per-feature zero fraction (count models only)
     latent_<key>.png           2-D PCA of a VAE's latent codes for held-out samples
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
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150,
})


def _color(i):
    return SERIES[i % len(SERIES)] if i < len(SERIES) else "#555555"


def loss_curves(dataset, keys, colors, fig_dir):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        run = json.loads((outputs(dataset) / "runs" / f"{key}.json").read_text())
        h = run["history"]
        ep = [r["epoch"] for r in h]
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
    ev_te = prepared["eval_test"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.5), squeeze=False, sharex=True, sharey=True)
    for ax, key in zip(axes[0], keys):
        syn = to_eval_space(np.load(outputs(dataset) / "samples" / f"{key}_trans.npy").astype(np.float64), prepared["state"])
        ax.scatter(ev_te[:, 0], ev_te[:, 1], s=6, c=REAL, alpha=0.55, lw=0, label="held-out real")
        ax.scatter(syn[:, 0], syn[:, 1], s=6, c=colors[key], alpha=0.55, lw=0, label=f"synthetic ({key})")
        ax.set_title(key, fontsize=9, color=INK)
        ax.set_xlabel("eval PC1")
        ax.legend(fontsize=7, loc="best", markerscale=2)
    axes[0][0].set_ylabel("eval PC2")
    fig.suptitle("Held-out real vs synthetic samples, first two evaluation PCs", fontsize=11, color=INK, x=0.01, ha="left")
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
    real_vs_synthetic(dataset, keys, colors, prepared, fig_dir)
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
