"""Assemble reports/<dataset>/generated_report_<N>.pdf from the artifacts in outputs/<dataset>/.
   The only file in reports/ is the PDF; this script is the only thing that writes there.

   Usage: venv/bin/python scripts/report.py --dataset <d> --report-number <N>

   Pages (generated from JSON; the agent writes only findings.md). A Contents page with page numbers comes first:
     1 Summary: objective (verbatim), decision, models, headline metrics with a legend, recommendation verdicts
     2 How to read this report        abbreviations, seeds, evaluation space, references, memorization, tests
     3 Metric definitions             every metric and test quantity, before any full table
     4 Objective interpretation       objective.json (+ ambiguities)
     5 Data and preprocessing         profile.json, plan split/preprocessing with reasons
     6 Models and training            plan models: hyperparameters with reasons; run outcome, epochs, stopping
     7 Evaluation metrics             metrics/*.json, grouped, with real-vs-real references; objective tests
     8 Figures                        figures/*.png
     9 Findings                       findings.md (agent-written)
    10 Downstream use                 recommendation.json
    11 Critique and revision          critique.json, if any
    12 Limitations                    auto-collected + findings.md "## Limitations"
    13 Reproducibility                command, seeds, config hashes, versions, timing

   Before writing, every decimal number in findings.md is traced to a value in the JSON artifacts; any
   that can't be traced are listed and the script exits 1 without writing (use --allow-untraced to
   override, which prints them on the Limitations page). Each page is also saved as a PNG under
   outputs/<d>/report_pages/ so the result can be inspected visually.
"""
import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parent.parent
PAGE_SIZE = (8.5, 11)
plt.rcParams["figure.max_open_warning"] = 0  # a report holds ~20 page figures until written
TEXT_W = 488  # usable line width in points (8.5in page minus margins); lines are wrapped by measured width
LINE_H, MIN_Y = 0.0185, 0.05
INK, MUTED = "#1f1f1e", "#5f5e5a"


# ── Text page helpers (adapted from Project 1's report.py) ──

_T2P = TextToPath()
_FONTS = {"p": FontProperties(family="sans-serif", size=8.6), "m": FontProperties(family="monospace", size=7.2),
          "h": FontProperties(family="sans-serif", size=11, weight="bold"), "c": FontProperties(family="sans-serif", size=8)}


def _width(text, kind="p"):
    return _T2P.get_text_width_height_descent(text.replace("$", "S"), _FONTS[kind], ismath=False)[0]


def _wrap(text, kind="p", indent="", max_w=TEXT_W):
    """Greedy word wrap by rendered width, so wide characters or long headings never run off the page.
    Splits on single spaces, so aligned columns in monospace rows keep their spacing."""
    if _width(text, kind) <= max_w:
        return [text]
    lines, cur = [], ""
    for w in text.split(" "):
        cand = f"{cur} {w}" if cur else w
        if cur.strip() and _width(cand, kind) > max_w:
            lines.append(cur.rstrip())
            cur = indent + w
        else:
            cur = cand
    lines.append(cur.rstrip())
    out = []
    for line in lines:  # a single word wider than the line (e.g. a long path) is cut by characters
        while _width(line, kind) > max_w and len(line) > 1:
            k = len(line)
            while k > 1 and _width(line[:k], kind) > max_w:
                k -= 1
            out.append(line[:k]); line = indent + line[k:]
        out.append(line)
    return out


def _block_groups(heading, paras):
    """One group of lines per paragraph (the heading joins the first), so a short paragraph is never split
    across pages and a heading never sits alone at the bottom of a page."""
    groups = []
    for i, p in enumerate(paras or [""]):
        lines = []
        if heading and i == 0:
            hl = _wrap(heading, "h")
            lines = [("h", hl[0])] + [("hc", l) for l in hl[1:]]  # a long heading continues on the next line
        mono = p.startswith("```")
        body = p.strip("`") if mono else p
        for raw_line in body.splitlines() or [""]:
            if mono:
                lines += [("m", l) for l in _wrap(raw_line, "m", indent="    ")]
            else:
                indent = "  " if raw_line.startswith("- ") else ""
                lines += [("p", l) for l in _wrap(raw_line, indent=indent)]
        lines.append(("sgap" if p.startswith("- ") else "gap", ""))  # list items sit closer together
        groups.append(lines)
    return groups


def _height(kind):
    return {"h": LINE_H * 1.6, "hc": LINE_H * 1.25, "gap": LINE_H * 0.6, "sgap": LINE_H * 0.2}.get(kind, LINE_H)


def text_pages(title, blocks):
    """blocks: [(heading or None, [paragraph, ...])]. Paragraphs starting with ``` render monospace.
    Splits across pages as needed; never silently drops content."""
    pages, cur, y = [], [], 0.0
    capacity = 0.93 - MIN_Y
    for heading, paras in blocks:
        for group in _block_groups(heading, paras):
            g_h = sum(_height(k) for k, _ in group)
            if y + g_h > capacity and cur and g_h <= capacity / 4:  # start short paragraphs on a fresh page
                pages.append(cur); cur, y = [], 0.0
            for kind, text in group:  # long paragraphs still split line by line
                h = _height(kind)
                if y + h > capacity and cur:
                    pages.append(cur); cur, y = [], 0.0
                cur.append((kind, text)); y += h
    pages.append(cur)
    pages = [pg for pg in pages if any(kind not in ("gap", "sgap") for kind, _ in pg)] or [[]]  # no spacing-only pages
    figs = []
    for i, lines in enumerate(pages):
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_axes([0.04, 0.02, 0.92, 0.95]); ax.axis("off")  # use the full page, not default axes margins
        ax.text(0.05, 0.97, title + (" (cont.)" if i else ""), transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", color=INK)
        y = 0.93
        for kind, text in lines:
            if kind == "h":
                y -= LINE_H * 0.4
                ax.text(0.05, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=INK)
                y -= LINE_H * 1.2
            elif kind == "hc":
                ax.text(0.05, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=INK)
                y -= LINE_H * 1.25
            elif kind in ("gap", "sgap"):
                y -= _height(kind)
            else:
                ax.text(0.06, y, text, transform=ax.transAxes, va="top", color=INK,
                        fontsize=7.2 if kind == "m" else 8.6, family="monospace" if kind == "m" else "sans-serif")
                y -= LINE_H
        figs.append(fig)
    return figs


def figure_page(png, caption):
    img = plt.imread(png)
    h, w = img.shape[:2]
    fig_w = 8.0
    fig = plt.figure(figsize=(fig_w, min(fig_w * h / w + 0.8, 10.5)))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.9]); ax.axis("off")
    ax.imshow(img, interpolation="none")  # embed the 300-dpi source pixels as-is, no resampling
    fig.text(0.04, 0.02, "\n".join(_wrap(caption, "c", max_w=fig_w * 72 * 0.92)), fontsize=8, color=MUTED, va="bottom")
    return fig


# ── Artifact loading ──

def _load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()


def load_all(dataset):
    out = ROOT / "outputs" / dataset
    a = {"out": out, "profile": _load(out / "profile.json", {}), "objective": _load(out / "objective.json", {}),
         "plan": _load(out / "plan.json", {}), "run_log": _load(out / "run_log.json", {}),
         "recommendation": _load(out / "recommendation.json"), "critique": _load(out / "critique.json"),
         "findings": _load(out / "findings.md"), "meta": _load(out / "prepared" / "meta.json", {}),
         "desc": _load(ROOT / "data" / dataset / "DATA_DESCRIPTION.md", "")}
    ok = [e["key"] for e in a["run_log"].get("entries", []) if e.get("status") == "ok"]
    a["keys"] = ok
    a["metrics"] = {k: _load(out / "metrics" / f"{k}.json") for k in ok if (out / "metrics" / f"{k}.json").exists()}
    a["runs"] = {k: _load(out / "runs" / f"{k}.json") for k in ok if (out / "runs" / f"{k}.json").exists()}
    a["downstream"] = _load(out / "metrics" / "downstream.json")
    a["anomaly"] = _load(out / "metrics" / "anomaly.json")
    a["imputation"] = _load(out / "metrics" / "imputation.json")
    a["evaluator"] = _load(out / "evaluator" / "evaluator.json")
    a["is_image"] = bool(a["meta"].get("image_shape"))
    return a


# ── Number tracing ──

def _numbers_in(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, acc)


def untraced_numbers(text, a):
    """Decimal numbers (e.g. 0.81, 12.5, 81.2%) in `text` that match no value in the artifacts at the
    precision written. Percentages may also match a fraction (81.2% <-> 0.812)."""
    pool = []
    for src in (a["profile"], a["meta"], a["run_log"], list(a["metrics"].values()), list(a["runs"].values()),
                a["plan"], a["recommendation"] or {}, a["downstream"] or {}, a["evaluator"] or {},
                a["anomaly"] or {}, a["imputation"] or {}):
        _numbers_in(src, pool)
    missing = []
    for m in re.finditer(r"(?<![\w.])(-?\d+\.\d+)(%?)", text):
        s, pct = m.group(1), m.group(2) == "%"
        q, dec = float(s), len(s.split(".")[1])
        cands = [q, q / 100] if pct else [q]
        ok = any(any(round(v, d) == round(c, d) for v in pool) for c, d in zip(cands, [dec, dec + 2]))
        if not ok:
            missing.append(m.group(0))
    return sorted(set(missing))


# ── Page builders ──

def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        # Exactly as stored in the JSON artifacts (which are rounded to <= 5 decimals when written), so
        # text quoting the JSON matches the tables and small values like MMD are not rounded to 0.
        return f"{v:.5f}".rstrip("0").rstrip(".") or "0"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


# ── Readable names (the JSON field name is still shown once, on the glossary page, for tracing) ──

LABELS = {
    "n_compared": "samples compared", "mmd2": "MMD²", "mmd2_reference": "  MMD² reference",
    "c2st_auc": "C2ST AUC", "c2st_auc_reference": "  C2ST AUC reference", "precision": "precision", "recall": "recall",
    "frechet_distance": "Fréchet distance", "frechet_distance_reference": "  Fréchet distance reference",
    "label_agreement": "label agreement", "label_agreement_reference": "  label agreement reference",
    "dcr_median_synthetic": "median DCR, synthetic", "dcr_median_heldout": "median DCR, held-out real",
    "dcr_ratio": "DCR ratio", "too_close_fraction": "share too close",
    "mean_corr": "corr. of feature means", "var_corr": "corr. of feature variances",
    "zero_frac_corr": "corr. of zero fractions", "zero_frac_synthetic": "zero fraction, synthetic",
    "zero_frac_heldout": "zero fraction, held-out real", "library_median_ratio": "median library size ratio",
    "train_mean": "mean (log-)likelihood, training", "heldout_mean": "mean (log-)likelihood, held-out",
    "per_dimension_heldout": "held-out, per dimension", "latent_dim": "latent size",
    "trustworthiness": "trustworthiness", "trustworthiness_pca_baseline": "  same-size PCA",
    "knn_overlap": "kNN overlap", "knn_overlap_pca_baseline": "  same-size PCA",
    "target_f1": "target-class F1", "macro_f1": "macro F1", "accuracy": "accuracy", "utility_ratio": "utility ratio",
    "noise_sd": "seed standard deviation", "auroc": "AUROC", "auroc_mean": "mean AUROC", "rmse": "RMSE", "mae": "MAE",
    "baseline_mean": "baseline: mean", "baseline_knn": "baseline: kNN", "baseline_knn_distance": "baseline: kNN distance",
    "baseline_pca_reconstruction": "baseline: PCA reconstruction", "real_same_size": "real images",
    "real": "real only", "resampled": "real + resampled target class", "extra_real": "real + new real target images",
    "feature_shuffle": "feature shuffle", "doublet": "doublet", "blend": "blend", "noise": "noise",
    "anomaly": "anomaly test", "imputation": "imputation test",
}
GROUP_TITLES = {"fidelity": "Fidelity", "conditional": "Class conditioning", "memorization": "Memorization",
                "memorization_pixels": "Memorization (raw pixels)", "feature_stats": "Per-feature statistics",
                "likelihood": "Likelihood", "representation": "Representation"}
TASK_NAMES = {"proceed": "proceed", "proceed_adapted": "proceed, adapted", "decline": "decline"}
VERDICTS = {"recommended": "Recommended", "conditional": "Conditional", "not_recommended": "Not recommended"}
# Generic path segments dropped when a metric path like vae_nb_s0.fidelity.c2st_auc is shown in words.
_PATH_SKIP = {"metrics", "results", "arms", "fidelity", "memorization", "feature_stats", "representation", "likelihood",
              "conditional", "downstream", "tstr", "balancing"}

ABBREVIATIONS = [
    "AUC / AUROC: area under the ROC (receiver operating characteristic) curve; 0.5 = chance, 1 = perfect separation.",
    "C2ST: classifier two-sample test. A classifier is trained to tell synthetic from real samples; its AUC is the score.",
    "MMD²: squared maximum mean discrepancy, a kernel distance between two sets of samples (RBF = Gaussian kernel).",
    "DCR: distance to closest record, the distance from a sample to its nearest training sample.",
    "PCA / PCs: principal component analysis / principal components.",
    "kNN, 5-NN: k nearest neighbors; a 5-NN radius is the distance from a sample to its 5th nearest neighbor.",
    "ELBO: evidence lower bound, the VAE's training objective (a lower bound on log-likelihood). NLL: negative log-likelihood.",
    "VAE: variational autoencoder. NB: negative binomial (a count distribution). GAN: generative adversarial network.",
    "CNN: convolutional neural network. FID: Fréchet Inception Distance (here computed on the evaluation classifier's features).",
    "TSTR: train on synthetic, test on real. F1: harmonic mean of precision and recall for one class; macro F1 averages it over classes.",
    "RMSE / MAE: root mean squared error / mean absolute error.",
]


def _label(name):
    if name in LABELS:
        return LABELS[name].strip()
    if name.startswith("synthetic_"):
        return f"synthetic: {name[len('synthetic_'):]}"
    if name.startswith("real_plus_"):
        return f"real + synthetic: {name[len('real_plus_'):]}"
    return name.replace("_", " ")


def _task(t):
    return t.replace("_", " ") if t else "none"


def _path_words(path):
    """'vae_nb_s0.representation.trustworthiness' -> 'vae_nb_s0, trustworthiness'."""
    parts = [p for p in path.split(".") if p not in _PATH_SKIP]
    return ", ".join(p if re.search(r"_s\d+$", p) else _label(p) for p in parts)


def _readable_paths(text, a):
    """Replace metric paths inside free text (e.g. an evidence 'compared_to') with words."""
    roots = set(a["metrics"]) | {"anomaly", "imputation", "downstream"}
    return re.sub(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b",
                  lambda m: _path_words(m.group(0)) if m.group(0).split(".")[0] in roots else m.group(0), text)


def _seed_line(a):
    models = [(m["id"], m.get("seeds", [])) for m in a["plan"].get("models", [])]
    if not models:
        return None
    ex = f"{models[0][0]}_s{models[0][1][0]}" if models[0][1] else models[0][0]
    listing = "; ".join(f"{mid} (seed{'s' if len(sd) > 1 else ''} {', '.join(map(str, sd))})" for mid, sd in models)
    return (f"Run names are <model id>_s<seed>: {ex} is model {models[0][0]} trained with seed {models[0][1][0] if models[0][1] else '?'}. "
            f"Models in this report: {listing}.")


def _pca_models(a):
    return [m for m in a["plan"].get("models", []) if m.get("input_space") == "pca"]


def guide_pages(a):
    meta, blocks = a["meta"], [("Abbreviations", [f"- {x}" for x in ABBREVIATIONS])]
    line = _seed_line(a)
    if line:
        multi = any(len(m.get("seeds", [])) > 1 for m in a["plan"].get("models", []))
        why = ("A seed fixes the model's random initialization, mini-batch order, the validation slice used for early "
               "stopping, and the random draws when sampling; the held-out split and preprocessing are the same for every run. "
               + ("Models with two seeds are trained twice as a stability check: a conclusion that holds for both seeds does not "
                  "depend on one lucky or unlucky initialization. Two runs are far too few to estimate run-to-run variability, "
                  "so the spread between seeds is not a confidence interval and should not be read as one."
                  if multi else "Each model was trained with one seed, so stability across seeds was not checked."))
        blocks.append(("Run names and seeds", [line, why]))
    if meta and a["is_image"] and a["evaluator"]:
        ev = a["evaluator"]
        blocks.append(("Where realism is measured: the evaluation space", [
            f"Images are compared in the {ev.get('feature_dim')}-dimensional feature space of a small CNN classifier trained on the "
            f"real training images (held-out accuracy {ev.get('heldout_accuracy')}). This plays the role that Inception features play "
            "in the usual FID: realism is judged by features that matter for recognizing these images. An ImageNet network would be a "
            "poor judge of small, domain-specific images, and too heavy to run here. The classifier never sees synthetic or held-out "
            "images during training, and the same classifier scores every model.",
            "Its features only capture what helps tell the classes apart, so two image sets can look alike to it and still differ in "
            "ways it ignores. The sample grid figure is the visual check. The 'Memorization (raw pixels)' table repeats the DCR "
            "check on raw pixels."]))
    elif meta:
        blocks.append(("Where realism is measured: the evaluation space", [
            f"Synthetic and held-out real samples are compared in one shared space: the first {meta.get('eval_components')} "
            f"principal components (PCs) of the transformed data, fit on the training split. These {meta.get('eval_components')} PCs "
            f"explain {100 * meta.get('eval_variance_explained', 0):.1f}% of the variance of the transformed training data, so the "
            f"realism metrics (C2ST, MMD, precision/recall, DCR) say nothing about the remaining "
            f"{100 * (1 - meta.get('eval_variance_explained', 0)):.1f}%. Per-feature statistics are computed on all features instead.",
            f"Why only {meta.get('eval_components')}: PCA fitted to the training data overfits it in later components, which would "
            "let a classifier tell training data from new real data apart and bias every comparison. The pipeline keeps the "
            f"components that hold up on unseen data (rule: {meta.get('eval_components_rule', '')}).",
        ]))
    blocks.append(("Reference values", [
        "Every realism metric is shown with a real-vs-real reference: the same metric computed between a subsample of the "
        "training data and the held-out real data. That is the value two genuinely real samples of this size give, so a model "
        "is judged by how close it gets to the reference, not to a theoretical ideal."]))
    blocks.append(("How memorization is characterized", [
        "For each sample, DCR (distance to closest record) is the distance to its nearest training sample in the evaluation space. "
        "The held-out real samples give the yardstick: how close a new sample from the same population typically lands to some "
        "training sample. The DCR ratio is the median synthetic DCR divided by the median held-out DCR (about 1 = as far as new "
        "real data; well below 1 = suspiciously close; well above 1 = samples unlike the real data). The share too close counts "
        "synthetic samples closer than the 5th percentile of held-out DCRs (about 0.05 is expected without copying).",
        "This detects near-copies of training samples. It is not a privacy guarantee: it does not test membership inference, "
        "attribute inference, or any formal property such as differential privacy."]))
    if any("conditional" in m for m in a["metrics"].values()):
        blocks.append(("Do generated images belong to the requested class?", [
            "Label agreement is the share of synthetic images that the evaluation classifier assigns to the class they were generated "
            "for. Its reference is the classifier's accuracy on held-out real images, which is what a perfect class-conditional generator "
            "would reach. Low agreement means the model's class conditioning is weak, which matters directly for class balancing."]))
    d = a["downstream"] or {}
    if d.get("balancing"):
        blocks.append(("The class-balancing test", [
            "Before any generator was chosen, a headroom check trained the same classifier, with three seeds each, on real data "
            "only, on real data with the target class oversampled by duplication, and, when unused real data existed, on real data "
            "plus that many genuinely new real target-class samples. The last one is the best case for any generator: a perfect "
            "generator can at most produce new samples from the same distribution. Each model's synthetic target-class samples "
            "were then tested the same way. Everything is scored on held-out real images, which synthetic data never enters. "
            "Differences smaller than about twice the real-only seed standard deviation are not meaningful."]))
    if d.get("tstr"):
        blocks.append(("Could synthetic data replace the real data? (train on synthetic, test on real)", [
            "The same classifier is trained on each model's labeled synthetic images only, and on the same number of real training "
            "images, three seeds each, and both are scored on held-out real images. The utility ratio is synthetic accuracy divided by real "
            "accuracy: 1 means the synthetic data is as useful for training this classifier as real data."]))
    if a["anomaly"]:
        blocks.append(("The anomaly-detection test", [
            "No labeled anomalies exist, so anomalies are made from held-out real samples with corruptions of known meaning "
            f"({', '.join(a['anomaly']['corruptions'])}). Each model scores every sample (negative ELBO for VAEs, negative "
            "log-likelihood for flows, denoising error for diffusion models) and AUROC measures how well the score separates corrupted "
            "from uncorrupted samples (0.5 = chance). Two non-generative baselines are scored the same way, so a model is only useful "
            "here if it beats them."]))
    if a["imputation"]:
        blocks.append(("The imputation test", [
            f"Real values of held-out samples are hidden on purpose ({a['imputation']['mask_kind']} masking, "
            f"{a['imputation']['n_masked_entries']} entries), so the truth is known. Each imputer fills them in, and the error on the "
            f"hidden entries is measured on the {a['imputation']['error_scale']} scale. The VAE is compared with mean and "
            "k-nearest-neighbor imputation."]))
    if any("representation" in m for m in a["metrics"].values()):
        blocks.append(("How the learned representation is judged", [
            "Trustworthiness (0 to 1) asks whether samples that are neighbors in the latent space are also neighbors in the full "
            "transformed data; it is penalized when the latent space puts unrelated samples together. It is compared with a PCA of "
            "the same size, fit on the training data, which is the simplest alternative representation."]))
    pcs = _pca_models(a)
    others = [m for m in a["plan"].get("models", []) if m.get("input_space") != "pca"]
    if pcs and others and meta:
        for m in pcs:
            blocks.append((f"Caution when comparing {m['id']} with the other models", [
                f"{m['id']} is trained on the top {m['hyperparameters'].get('pca_components')} PCA components of the transformed data, "
                f"and realism is scored in a {meta.get('eval_components')}-component PCA of the same transformed data. The directions it "
                f"models directly are largely the directions the realism metrics look at, while "
                f"{', '.join(o['id'] for o in others)} must reproduce them through a model of all {meta.get('n_features')} features. "
                "The shared realism metrics may therefore favor it. Per-feature and count-level checks, which use all features, "
                "are a fairer comparison of what each model reproduces outside those components."]))
    return text_pages("How to read this report", blocks)


def summary_page(a, n):
    o, p = a["objective"], a["plan"]
    task_line = f"Task: {_task(o.get('task'))}" + (f" (also: {', '.join(map(_task, o['secondary_tasks']))})" if o.get("secondary_tasks") else "")
    task_line += f"  |  Decision: {TASK_NAMES.get(o.get('decision'), o.get('decision'))}"
    if o.get("decision") != "proceed":
        task_line += f"  |  Task actually run: {_task(o.get('effective_task'))}"
    blocks = [("Objective (as given)", [f"\"{o.get('raw_text', '')}\""]),
              ("Agent's decision", [task_line, o.get("decision_reason", "")])]
    rows = []
    for k in a["keys"]:
        m = a["metrics"].get(k, {})
        f = m.get("fidelity", {})
        if a["is_image"]:
            c = m.get("conditional", {})
            rows.append(f"{k:<24} Fréchet dist. {_fmt(f.get('frechet_distance'))} (ref {_fmt(f.get('frechet_distance_reference'))})"
                        f"  C2ST AUC {_fmt(f.get('c2st_auc'))} (ref {_fmt(f.get('c2st_auc_reference'))})"
                        + (f"  label agreement {_fmt(c.get('label_agreement'))} (ref {_fmt(c.get('label_agreement_reference'))})" if c else ""))
        else:
            rows.append(f"{k:<14} C2ST AUC {_fmt(f.get('c2st_auc'))} (ref {_fmt(f.get('c2st_auc_reference'))})"
                        f"  precision {_fmt(f.get('precision'))}  recall {_fmt(f.get('recall'))}"
                        f"  DCR ratio {_fmt(m.get('memorization', {}).get('dcr_ratio'))}")
    skipped = [e for e in a["run_log"].get("entries", []) if e.get("status") != "ok"]
    rows += [f"{e['key']:<18} {'not run (did not fit the budget)' if e['status'] == 'not_run_budget' else e['status']}" for e in skipped]
    guide = ["Run names are <model id>_s<seed>. 'How to read this report', right after this summary, defines every abbreviation "
             "and explains seeds and the references."]
    if a["is_image"]:
        legend = ("Fréchet distance: distance between real and synthetic feature distributions (lower is better). C2ST AUC: how well "
                  "a classifier tells synthetic from real (0.5 = indistinguishable). Label agreement: share of generated images "
                  "recognized as their requested class. ref = the value two real samples give.")
    else:
        legend = ("C2ST AUC: how well a classifier tells synthetic from real cells (0.5 = indistinguishable). Precision: share "
                  "of synthetic samples near real ones. Recall: share of real samples covered. DCR ratio: distance of synthetic "
                  "samples to training data relative to new real data (about 1 = no sign of copying). ref = the value two real "
                  "samples give.")
    blocks.append(("Models run (headline metrics; full tables on the metrics pages)",
                   guide + ["```" + "\n".join(rows or ["none"]) + "```", legend]))
    for title, lines in objective_test_tables(a, short=True):
        blocks.append((title, ["```" + "\n".join(lines) + "```"]))
    if a["recommendation"]:
        recs = [f"- {VERDICTS.get(u['verdict'], u['verdict'])}: {u['use']} ({u.get('model_key', '')})" for u in a["recommendation"]["uses"]]
        blocks.append(("Downstream-use verdicts", recs))
    if a["findings"]:
        summ = re.search(r"^##\s*Summary\s*$(.*?)(?=^##\s|\Z)", a["findings"], re.M | re.S)
        if summ:
            blocks.append(("Summary (agent-written)", [p.strip() for p in summ.group(1).strip().split("\n\n")]))
    title = f"Generated report {n}: {p.get('dataset', '')}"
    return text_pages(title, blocks)


def objective_page(a):
    o = a["objective"]
    blocks = [("Interpretation", [f"Primary task: {_task(o.get('task'))}",
                                  f"Secondary tasks: {', '.join(map(_task, o.get('secondary_tasks') or [])) or 'none'}",
                                  o.get("interpretation_reason", "")]),
              ("Success criterion", [o.get("success_criterion", "")]),
              ("Whether to model, and how", [f"Decision: {TASK_NAMES.get(o.get('decision'), o.get('decision'))} "
                                             f"(task actually run: {_task(o.get('effective_task'))}). The reason is given in the summary."])]
    if o.get("ambiguities"):
        blocks.append(("Other readings considered", [f"- {x}" for x in o["ambiguities"]]))
    return text_pages("Objective interpretation", blocks)


def data_page(a):
    pr, p, meta = a["profile"], a["plan"], a["meta"]
    what = re.search(r"^##\s*What this is\s*$(.*?)(?=^##\s|\Z)", a["desc"], re.M | re.S)
    prof = [f"{pr.get('n_samples', 0):,} samples x {pr.get('n_features', 0):,} features; modality {pr.get('modality')}"
            + (f" {tuple(pr['image_shape'])}" if pr.get("image_shape") else ""),
            f"Sparsity: {100 * (pr.get('sparsity') or 0):.1f}% zeros; value range {_fmt(pr.get('value_range'))}; non-negative integers: {pr.get('is_nonneg_integer')}",
            f"Labels: {'yes, ' + str(pr.get('n_classes')) + ' classes, largest/smallest class ratio ' + _fmt(pr.get('imbalance_ratio')) if pr.get('has_labels') else 'none'}"]
    blocks = []
    if what:
        blocks.append(("Dataset (from DATA_DESCRIPTION.md)", [" ".join(what.group(1).split())]))
    blocks.append(("Profile", prof))
    sp = p.get("split", {})
    blocks.append(("Held-out split", [f"Held-out fraction {sp.get('test_fraction')}, stratified by class: {'yes' if sp.get('stratify') else 'no'}"
                                      + (f" -> {meta.get('n_train')} training / {meta.get('n_test')} held-out samples" if meta else ""),
                                      f"Reason: {sp.get('reason', '')}"]))
    steps = [f"- {s['step']} {json.dumps(s.get('params')) if s.get('params') else ''}: {s.get('reason', '')}" for s in p.get("preprocessing", [])]
    blocks.append(("Preprocessing (applied identically for every model; fit on training data only)", steps or ["none"]))
    if meta and not a["is_image"]:
        blocks.append(("Evaluation space", [
            f"PCA with {meta.get('eval_components')} components, fit on the training split; see 'How to read this report'."]))
    return text_pages("Data and preprocessing", blocks)


def models_page(a):
    blocks = []
    for m in a["plan"].get("models", []):
        hp, rs = m.get("hyperparameters", {}), m.get("hyperparameter_reasons", {})
        entries = [e for e in a["run_log"].get("entries", []) if e.get("model_id") == m["id"]]
        if m.get("status") == "not_run_budget":
            outcome = "not run (did not fit the budget)"
        elif entries:
            outcome = "; ".join(f"{e['key']} {e.get('trained', 'ok') if e.get('status') == 'ok' else e.get('status')}" for e in entries)
        else:
            outcome = "not run"
        lines = [f"Family: {m['family']}{' (conditional)' if m.get('conditional') else ''}; input: {m.get('input_space')}; "
                 f"seeds {m.get('seeds')}; serves {', '.join(map(_task, m.get('serves_tasks', [])))}; "
                 f"estimated {m.get('estimated_minutes')} min/seed",
                 f"Outcome: {outcome}",
                 f"Why this model: {m.get('reason', '')}"]
        lines += [f"- {k} = {_fmt(v)}: {rs.get(k, '')}" for k, v in hp.items()]
        for k, r in a["runs"].items():
            if r.get("model_id") == m["id"]:
                lines.append(f"Training ({k}): {r['epochs_run']} of {r['epochs_planned']} max epochs, kept epoch {r['best_epoch']} "
                             f"({r['stop_reason']}); {r['minutes']} min on {r['device']}; {r['n_fit']} fit / {r['n_val']} validation samples"
                             + (f"; {r['n_parameters']:,} parameters" if r.get("n_parameters") else "")
                             + (f"; input PCA keeps {100 * r['input_pca_variance_explained']:.1f}% of variance" if r.get("input_pca_variance_explained") else ""))
        blocks.append((m["id"], lines))
    if a["plan"].get("evaluation_notes"):
        blocks.append(("Evaluation focus", [a["plan"]["evaluation_notes"]]))
    return text_pages("Models, hyperparameters, and training", blocks)


GROUPS = ["fidelity", "conditional", "memorization", "memorization_pixels", "feature_stats", "likelihood", "representation"]


def metrics_pages(a):
    keys = list(a["metrics"])
    if not keys:
        return text_pages("Evaluation metrics", [(None, ["No model produced metrics (see Limitations and run log)."])])
    blocks = []
    for g in GROUPS:
        names = []
        for k in keys:
            for n, v in a["metrics"][k].get(g, {}).items():
                if n not in names and not isinstance(v, (str, dict, list)):
                    names.append(n)
        if not names:
            continue
        w = max(len(LABELS.get(n, _label(n))) for n in names) + 2
        lines = [" " * w + "".join(f"{k:>16}" for k in keys)]
        for n in names:
            lines.append(f"{LABELS.get(n, _label(n)):<{w}}" + "".join(f"{_fmt(a['metrics'][k].get(g, {}).get(n)):>16}" for k in keys))
        notes = {a["metrics"][k][g].get("warning") or a["metrics"][k][g].get("note") for k in keys if g in a["metrics"][k]} - {None}
        if g == "likelihood":
            notes.add("Likelihoods are in each model's own input space and units; do not compare them across models with different input spaces.")
        blocks.append((GROUP_TITLES.get(g, g), ["```" + "\n".join(lines) + "```"] + sorted(notes)))
    if a["is_image"] and a["evaluator"]:
        blocks.insert(0, (None, ["Realism metrics use the evaluation classifier's feature space (see 'How to read this report'); "
                                 "per-pixel statistics use all pixels. Indented rows are the real-vs-real reference or baseline."]))
    elif a["meta"]:
        blocks.insert(0, (None, ["Realism metrics use the evaluation PCA space (see 'How to read this report'); per-feature "
                                 "statistics use all features. Indented rows are the real-vs-real reference or baseline."]))
    return text_pages("Evaluation metrics (held-out real data)", blocks)


def _arm_line(name, r, keys=("target_f1", "macro_f1", "accuracy")):
    cells = "".join(f"{_fmt(r.get(f'{k}_mean')):>9} +/- {_fmt(r.get(f'{k}_sd')):<7}" for k in keys)
    return f"{_label(name):<32}{r.get('runs', [{}])[0].get('n_train', ''):>7}  {cells}"


def objective_test_tables(a, short=False):
    """[(title, lines)] for every objective test that ran. short=True gives the page-1 version."""
    out = []
    d = a["downstream"] or {}
    if d.get("balancing"):
        b = d["balancing"]
        head = f"{'training set':<32}{'n':>7}  " + "".join(f"{k + ' (mean +/- sd)':<26}" for k in (("target F1",) if short else ("target F1", "macro F1", "accuracy")))
        rows = [head] + [_arm_line(n, r, ("target_f1",) if short else ("target_f1", "macro_f1", "accuracy")) for n, r in b["arms"].items()]
        out.append((f"Objective test: class balancing, target class {', '.join(b['targets'])} (held-out real images, {d.get('seeds_per_arm')} seeds per arm)", rows))
    if d.get("tstr"):
        t = d["tstr"]
        rows = [f"{'training set':<32}{'n':>7}  {'accuracy (mean +/- sd)':<26}{'macro F1 (mean +/- sd)':<26}"]
        rows += [_arm_line(n, r, ("accuracy", "macro_f1")) for n, r in t["arms"].items()]
        rows += [f"utility ratio (synthetic / real accuracy): " + ", ".join(f"{k.replace('synthetic_', '', 1)} {_fmt(v)}" for k, v in t["utility_ratio"].items())]
        out.append(("Objective test: train on synthetic, test on real (held-out real images)", rows))
    if a["anomaly"]:
        an = a["anomaly"]
        cs = an["corruptions"]
        rows = [f"{'score':<30}" + "".join(f"{_label(c):>17}" for c in cs) + f"{'mean AUROC':>12}"]
        for n, r in an["results"].items():
            rows.append(f"{_label(n):<30}" + ("".join(f"{_fmt(r['auroc'][c]):>17}" for c in cs) + f"{_fmt(r['auroc_mean']):>12}"
                                      if "auroc" in r else f"  {r.get('status', '')}"))
        out.append((f"Objective test: anomaly detection (AUROC, {an['n_normal']} normal vs {an['n_per_corruption']} corrupted per type)", rows))
    if a["imputation"]:
        im = a["imputation"]
        rows = [f"{'imputer':<30}{'RMSE':>10}{'MAE':>10}"]
        rows += [f"{_label(n):<30}" + (f"{_fmt(r['rmse']):>10}{_fmt(r['mae']):>10}" if "rmse" in r else f"  {r.get('status', '')}")
                 for n, r in im["results"].items()]
        out.append((f"Objective test: imputation ({im['mask_kind']} masking, error on {im['error_scale']} scale; lower is better)", rows))
    return out


def objective_test_pages(a):
    tables = objective_test_tables(a)
    if not tables:
        return []
    blocks = [(title, ["```" + "\n".join(lines) + "```"]) for title, lines in tables]
    d = a["downstream"] or {}
    if d.get("balancing"):
        b = d["balancing"]
        blocks.append((None, [f"Each target class was topped up to the largest training class "
                              f"({', '.join(f'{t}: +{v}' for t, v in b['n_added_per_target'].items())} samples). Classifier: {b['classifier']}. "
                              f"Real-only seed standard deviation of target F1: {_fmt(b.get('noise_sd_target_f1'))}."]))
    if a["anomaly"]:
        blocks.append((None, [a["anomaly"].get("note", "")]))
    if a["imputation"]:
        blocks.append((None, [a["imputation"].get("note", "")]))
    return text_pages("Objective tests on held-out real data", blocks)


def definitions_page(a):
    """Every metric and test quantity, defined before the tables that use it. The JSON field name is given in
    brackets so a number can be traced to its file."""
    defs, tests = {}, {}
    for m in a["metrics"].values():
        defs.update(m.get("definitions", {}))
    for t in (a["downstream"] or {}, a["anomaly"] or {}, a["imputation"] or {}):
        tests.update(t.get("definitions", {}))
    blocks = [("Model metrics", [f"- {_label(k)} [{k}]: {v}" for k, v in defs.items()] or ["No metrics."])]
    if tests:
        blocks.append(("Objective-test quantities", [f"- {_label(k)} [{k}]: {v}" for k, v in tests.items()]))
    return text_pages("Metric definitions", blocks)


CAPTIONS = {
    "loss_curves": "Training curves. The validation set is 10% of the training split and decides early stopping; the held-out curve is shown for reference only and never used to choose anything.",
    "real_vs_synthetic": "Held-out real samples (gray) and synthetic samples in the first two principal components of the shared evaluation space (for images: of the evaluation classifier's features). Overlap in two components is necessary but not sufficient; the C2ST and MMD metrics use all components.",
    "feature_stats": "Per-feature mean and variance of synthetic vs held-out real data on the transformed scale. Points on the dashed line agree exactly.",
    "count_properties": "Count-level checks for models that generate counts: total counts per sample, and per-feature fraction of zeros.",
    "sample_grid": "Held-out real images (first column block) and each model's samples, one row per class. Conditional models were asked for the class of the row; unconditional models' samples are not sorted by class.",
    "label_agreement": "Per class: share of each model's images that the evaluation classifier assigns to the requested class; gray bars are the classifier's accuracy on held-out real images (the ceiling).",
    "objective_tests": "Objective tests on held-out real data: each bar is the mean over classifier seeds (or the AUROC / error of one score), with error bars of one standard deviation where seeds were repeated.",
    "latent": "Two-dimensional PCA of a VAE's latent codes for held-out samples. A picture for orientation; neighbor-preservation numbers are on the metrics page.",
}


def figure_pages(a):
    figs = []
    for png in sorted((a["out"] / "figures").glob("*.png")):
        cap = next((c for k, c in CAPTIONS.items() if png.stem.startswith(k)), png.stem)
        if png.stem.startswith("latent_"):
            cap = f"{png.stem[len('latent_'):]}: " + cap
        figs.append(figure_page(png, cap))
    return figs


def findings_pages(a):
    text = a["findings"]
    if not text:
        return text_pages("Findings", [(None, ["findings.md was not written; the report has no written interpretation."])])
    blocks, heading, paras = [], None, []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        chunk = chunk.strip()
        if chunk.startswith("## "):
            first, _, rest = chunk.partition("\n")
            if paras or heading:
                blocks.append((heading, paras))
            heading, paras = first[3:].strip(), ([rest.strip()] if rest.strip() else [])
        elif chunk.startswith("# "):
            continue
        else:
            paras.append(" ".join(l.strip() for l in chunk.splitlines()) if not chunk.startswith("- ") else chunk)
    blocks.append((heading, paras))
    blocks = [(h, p) for h, p in blocks if h not in ("Summary", "Limitations")]  # shown on their own pages
    return text_pages("Findings", blocks)


def recommendation_pages(a):
    r = a["recommendation"]
    if not r:
        return text_pages("Downstream use", [(None, ["recommendation.json was not written."])])
    blocks = []
    for u in r["uses"]:
        lines = [f"Verdict: {VERDICTS.get(u['verdict'], u['verdict'])}  |  task: {_task(u.get('task'))}  |  model: {u.get('model_key', '')}",
                 u.get("rationale", "")]
        lines += [f"- evidence: {_path_words(e['metric'])} = {_fmt(e.get('value'))}"
                  + (f"; compared with {_readable_paths(e['compared_to'], a)}" if e.get("compared_to") else "") for e in u.get("evidence", [])]
        lines += [f"- caveat: {c}" for c in u.get("caveats", [])]
        blocks.append((u["use"], lines))
    if r.get("not_tested"):
        blocks.append(("Not tested (no recommendation either way)", [f"- {x}" for x in r["not_tested"]]))
    if r.get("privacy_statement"):
        blocks.append(("Privacy", [r["privacy_statement"]]))
    return text_pages("Downstream use of the generated data", blocks)


def critique_pages(a):
    c = a["critique"]
    if not c:
        return []
    blocks = [(None, ["An independent critic subagent reviewed the objective, plan, metrics, and training logs (not the agent's "
                      "own reasoning) before the recommendation and findings were written. Each finding's evidence is in "
                      f"outputs/{a['plan'].get('dataset')}/critique.json."]
               + ([f"The critic reviewed plan revision {a['plan']['revision'] - 1}. Numbers it quotes for models changed in "
                   "the revision differ from the tables above, which show the final runs."]
                  if a["plan"].get("critique_applied") and (a["plan"].get("revision") or 1) > 1 else [])
               + [c.get("summary", "")])]
    for f in c.get("findings", []):
        blocks.append((f"[{f.get('severity', '')}] {f.get('category', '')}", [f.get("issue", ""),
                                                                          f"Recommendation: {f.get('recommendation', '')}"]))
    if a["plan"].get("critique_applied"):
        blocks.append(("Revision made in response", [f"Plan revision {a['plan'].get('revision')}: {a['plan']['critique_applied']}"]))
    return text_pages("Critique and revision", blocks)


def limitations_pages(a, untraced):
    items = []
    for e in a["run_log"].get("entries", []):
        if e.get("status") == "failed":
            items.append(f"{e['key']} failed at the {e.get('failed_step')} step after one retry: {(e.get('error') or '').splitlines()[-1:]}")
        elif e.get("status") == "not_run_budget":
            items.append(f"{e['key']} was planned but not run because it did not fit the time budget.")
    for k, r in a["runs"].items():
        if "cap" in (r.get("stop_reason") or ""):
            items.append(f"{k} stopped at the wall-clock cap, before early stopping triggered; it may be undertrained.")
    for k, m in a["metrics"].items():
        if m.get("fidelity", {}).get("warning"):
            items.append(f"{k}: {m['fidelity']['warning']}")
    if _pca_models(a) and len(_pca_models(a)) < len(a["plan"].get("models", [])):
        items.append("Models trained on PCA scores are evaluated in a PCA space of the same data, which may favor them on the "
                     "shared realism metrics (see 'How to read this report').")
    if a["meta"] and not a["is_image"]:
        items.append(f"Fidelity is measured in a {a['meta'].get('eval_components')}-component PCA space "
                     f"({100 * (a['meta'].get('eval_variance_explained') or 0):.1f}% of training variance); differences outside it are not measured.")
    if a["is_image"]:
        items.append("Image realism is judged in the feature space of a classifier trained on the same real data; differences that do "
                     "not affect classification (fine texture, staining details) are not measured by these numbers.")
    if (a["downstream"] or {}).get("balancing") or (a["downstream"] or {}).get("tstr"):
        items.append("Classifier-based tests use one classifier architecture and three seeds per training set; three seeds give a "
                     "rough spread, not a confidence interval.")
    if a["anomaly"]:
        items.append("Anomalies in the anomaly test are synthetic corruptions of real samples; detection of real anomalies was not measured.")
    if not a["critique"]:
        items.append("No independent critique was run for this report.")
    if a["findings"]:
        lim = re.search(r"^##\s*Limitations\s*$(.*?)(?=^##\s|\Z)", a["findings"], re.M | re.S)
        if lim:
            items += [p.strip().lstrip("- ").strip() for p in re.split(r"\n\s*\n|\n- ", lim.group(1).strip()) if p.strip()]
    if untraced:
        items.append(f"findings.md quotes numbers that could not be traced to any artifact: {', '.join(untraced)}")
    return text_pages("Limitations", [(None, [f"- {x}" for x in items])])


def repro_pages(a, n):
    try:
        import numpy, sklearn, torch
        versions = f"python {platform.python_version()}, torch {torch.__version__}, numpy {numpy.__version__}, scikit-learn {sklearn.__version__}"
    except Exception:
        versions = f"python {platform.python_version()}"
    lines = [f"Command: ./run.sh {a['plan'].get('dataset')} \"{a['objective'].get('raw_text', '')}\" --report-number {n}",
             f"Plan seed {a['plan'].get('seed')}; plan revision {a['plan'].get('revision')}; budget {a['plan'].get('budget_min')} min; "
             f"pipeline time {a['run_log'].get('total_minutes')} min",
             f"Preprocessing hash {a['meta'].get('prep_hash')}", versions]
    lines += [f"- {k}: seed {r['seed']}, config hash {r['config_hash']}, {r['minutes']} min on {r['device']}" for k, r in a["runs"].items()]
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. All intermediate files: outputs/{a['plan'].get('dataset')}/.")
    return text_pages("Reproducibility", [(None, lines)])


def contents_page(sections, n, dataset):
    """One page listing each non-empty section and the page it starts on (the contents page is page 1)."""
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.95]); ax.axis("off")
    ax.text(0.05, 0.97, f"Generated report {n}: {dataset}", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", color=INK)
    ax.text(0.05, 0.925, "Contents", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", color=INK)
    y, page = 0.88, 2
    for title, figs in sections:
        if not figs:
            continue
        ax.text(0.07, y, title, transform=ax.transAxes, fontsize=10, va="top", color=INK)
        ax.text(0.90, y, str(page), transform=ax.transAxes, fontsize=10, va="top", ha="right", color=INK)
        ax.plot([0.07 + _width(title, "p") * 10 / 8.6 / (0.92 * 612) + 0.01, 0.865], [y - 0.008] * 2,
                transform=ax.transAxes, color=MUTED, lw=0.6, ls=(0, (1, 3)))  # dotted leader
        y -= 0.032
        page += len(figs)
    return [fig]


def build(dataset, n, allow_untraced=False):
    a = load_all(dataset)
    if not a["plan"] or not a["objective"]:
        sys.exit("ERROR: plan.json and objective.json are required")
    untraced = untraced_numbers(a["findings"] or "", a)
    if untraced and not allow_untraced:
        print("UNTRACED numbers in findings.md (not found in any artifact at the precision written):")
        for u in untraced:
            print(f"  {u}")
        sys.exit("ERROR: fix findings.md so every number comes from metrics/run/profile JSON, then rerun.")

    sections = [("Summary", summary_page(a, n)), ("How to read this report", guide_pages(a)),
                ("Metric definitions", definitions_page(a)), ("Objective interpretation", objective_page(a)),
                ("Data and preprocessing", data_page(a)), ("Models, hyperparameters, and training", models_page(a)),
                ("Evaluation metrics", metrics_pages(a)), ("Objective tests", objective_test_pages(a)),
                ("Figures", figure_pages(a)), ("Findings", findings_pages(a)),
                ("Downstream use of the generated data", recommendation_pages(a)),
                ("Critique and revision", critique_pages(a)), ("Limitations", limitations_pages(a, untraced)),
                ("Reproducibility", repro_pages(a, n))]
    pages = contents_page(sections, n, a["plan"].get("dataset", ""))
    for _, figs in sections:
        pages += figs
    for i, fig in enumerate(pages, 1):
        fig.text(0.5, 0.008, f"{i} / {len(pages)}", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    pdf_path = ROOT / "reports" / dataset / f"generated_report_{n}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = a["out"] / "report_pages"
    png_dir.mkdir(exist_ok=True)
    for old in png_dir.glob("page_*.png"):
        old.unlink()
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(pages, 1):
            pdf.savefig(fig)
            fig.savefig(png_dir / f"page_{i:02d}.png", dpi=70)
            plt.close(fig)
    print(f"Wrote {pdf_path} ({len(pages)} pages); page images in {png_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the generated report PDF for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-number", type=int, required=True)
    parser.add_argument("--allow-untraced", action="store_true", help="Build even if findings.md has untraceable numbers (listed under Limitations)")
    args = parser.parse_args()
    build(args.dataset, args.report_number, args.allow_untraced)
