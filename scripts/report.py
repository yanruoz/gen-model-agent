"""Assemble reports/<dataset>/generated_report_<N>.pdf from the artifacts in outputs/<dataset>/.
   The only file in reports/ is the PDF; this script is the only thing that writes there.

   Usage: venv/bin/python scripts/report.py --dataset <d> --report-number <N>

   Pages (generated from JSON; the agent writes only findings.md):
     1 Summary: objective (verbatim), decision, models, headline metrics, recommendation verdicts
     2 Objective interpretation       objective.json (+ ambiguities)
     3 Data and preprocessing         profile.json, plan split/preprocessing with reasons, evaluation space
     4 Models and training            plan models: hyperparameters with reasons; run sidecars: epochs, stopping
     5 Evaluation metrics             metrics/*.json, grouped, with real-vs-real references
     6 Metric definitions
     7 Figures                        figures/*.png
     8 Findings                       findings.md (agent-written)
     9 Downstream use                 recommendation.json
    10 Critique and revision          critique.json, if any
    11 Limitations                    auto-collected + findings.md "## Limitations"
    12 Reproducibility                command, seeds, config hashes, versions, timing

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
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
PAGE_SIZE = (8.5, 11)
plt.rcParams["figure.max_open_warning"] = 0  # a report holds ~20 page figures until written
WRAP = 92
LINE_H, MIN_Y = 0.0185, 0.05
INK, MUTED = "#1f1f1e", "#5f5e5a"


# ── Text page helpers (adapted from Project 1's report.py) ──

def _wrap(text, width=WRAP, indent=""):
    return textwrap.wrap(text, width=width, subsequent_indent=indent) or [""]


def _block_lines(heading, paras):
    lines = []
    if heading:
        lines.append(("h", heading))
    for p in paras:
        mono = p.startswith("```")
        body = p.strip("`") if mono else p
        for raw_line in body.splitlines() or [""]:
            if mono:
                lines.append(("m", raw_line))
            else:
                indent = "  " if raw_line.startswith("- ") else ""
                lines += [("p", l) for l in _wrap(raw_line, indent=indent)]
        lines.append(("gap", ""))
    return lines


def text_pages(title, blocks):
    """blocks: [(heading or None, [paragraph, ...])]. Paragraphs starting with ``` render monospace.
    Splits across pages as needed; never silently drops content."""
    pages, cur, y = [], [], 0.0
    capacity = 0.93 - MIN_Y
    for heading, paras in blocks:
        lines = _block_lines(heading, paras)
        for kind, text in lines:
            h = {"h": LINE_H * 1.6, "gap": LINE_H * 0.6}.get(kind, LINE_H)
            if y + h > capacity and cur:
                pages.append(cur); cur, y = [], 0.0
            cur.append((kind, text)); y += h
    pages.append(cur)
    pages = [pg for pg in pages if any(kind != "gap" for kind, _ in pg)] or [[]]  # no spacing-only pages
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
            elif kind == "gap":
                y -= LINE_H * 0.6
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
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.9]); ax.axis("off"); ax.imshow(img)
    fig.text(0.04, 0.02, "\n".join(_wrap(caption, 120)), fontsize=8, color=MUTED, va="bottom")
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
                a["plan"], a["recommendation"] or {}):
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


def summary_page(a, n):
    o, p = a["objective"], a["plan"]
    blocks = [("Objective (as given)", [f"\"{o.get('raw_text', '')}\""]),
              ("Agent's decision", [f"Task: {o.get('task')}  |  decision: {o.get('decision')}  |  task run: {o.get('effective_task')}"
                                    + (f"  |  also: {', '.join(o['secondary_tasks'])}" if o.get("secondary_tasks") else ""),
                                    o.get("decision_reason", "")])]
    rows = []
    for k in a["keys"]:
        m = a["metrics"].get(k, {})
        f = m.get("fidelity", {})
        rows.append(f"{k:<18} {m.get('family', ''):<6} C2ST {_fmt(f.get('c2st_auc'))} (real-vs-real {_fmt(f.get('c2st_auc_reference'))})"
                    f"  precision {_fmt(f.get('precision'))}  recall {_fmt(f.get('recall'))}"
                    f"  DCR ratio {_fmt(m.get('memorization', {}).get('dcr_ratio'))}")
    skipped = [e for e in a["run_log"].get("entries", []) if e.get("status") != "ok"]
    rows += [f"{e['key']:<18} {e['status']}" for e in skipped]
    blocks.append(("Models run (headline metrics; details on the metrics page)", ["```" + "\n".join(rows or ["none"]) + "```"]))
    if a["recommendation"]:
        recs = [f"- [{u['verdict']}] {u['use']} ({u.get('model_key', '')})" for u in a["recommendation"]["uses"]]
        blocks.append(("Downstream-use verdicts", recs))
    if a["findings"]:
        summ = re.search(r"^##\s*Summary\s*$(.*?)(?=^##\s|\Z)", a["findings"], re.M | re.S)
        if summ:
            blocks.append(("Summary (agent-written)", [p.strip() for p in summ.group(1).strip().split("\n\n")]))
    title = f"Generated report {n}: {p.get('dataset', '')}"
    return text_pages(title, blocks)


def objective_page(a):
    o = a["objective"]
    blocks = [("Interpretation", [f"Primary task: {o.get('task')}", f"Secondary tasks: {', '.join(o.get('secondary_tasks') or []) or 'none'}",
                                  o.get("interpretation_reason", "")]),
              ("Success criterion", [o.get("success_criterion", "")]),
              ("Whether to model, and how", [f"Decision: {o.get('decision')} (task actually run: {o.get('effective_task')})", o.get("decision_reason", "")])]
    if o.get("ambiguities"):
        blocks.append(("Other readings considered", [f"- {x}" for x in o["ambiguities"]]))
    return text_pages("Objective interpretation", blocks)


def data_page(a):
    pr, p, meta = a["profile"], a["plan"], a["meta"]
    what = re.search(r"^##\s*What this is\s*$(.*?)(?=^##\s|\Z)", a["desc"], re.M | re.S)
    prof = [f"{pr.get('n_samples', 0):,} samples x {pr.get('n_features', 0):,} features; modality {pr.get('modality')}"
            + (f" {tuple(pr['image_shape'])}" if pr.get("image_shape") else ""),
            f"Sparsity {_fmt(pr.get('sparsity'))}; value range {_fmt(pr.get('value_range'))}; non-negative integers: {pr.get('is_nonneg_integer')}",
            f"Labels: {'yes, ' + str(pr.get('n_classes')) + ' classes, imbalance ratio ' + _fmt(pr.get('imbalance_ratio')) if pr.get('has_labels') else 'none'}"]
    blocks = []
    if what:
        blocks.append(("Dataset (from DATA_DESCRIPTION.md)", [" ".join(what.group(1).split())]))
    blocks.append(("Profile", prof))
    sp = p.get("split", {})
    blocks.append(("Held-out split", [f"test_fraction {sp.get('test_fraction')}, stratified: {bool(sp.get('stratify'))}"
                                      + (f" -> {meta.get('n_train')} training / {meta.get('n_test')} held-out samples" if meta else ""),
                                      f"Reason: {sp.get('reason', '')}"]))
    steps = [f"- {s['step']} {json.dumps(s.get('params')) if s.get('params') else ''}: {s.get('reason', '')}" for s in p.get("preprocessing", [])]
    blocks.append(("Preprocessing (applied identically for every model; fit on training data only)", steps or ["none"]))
    if meta:
        blocks.append(("Evaluation space", [
            f"All models are scored in one shared space: PCA of the transformed data with {meta.get('eval_components')} components "
            f"({_fmt(meta.get('eval_variance_explained'))} of training variance), fit on the training split. The number of components "
            f"is the {meta.get('eval_components_rule', '')}: a PCA with more components overfits the training data, and a classifier "
            "could then tell training from held-out real samples apart, biasing every comparison."]))
    return text_pages("Data and preprocessing", blocks)


def models_page(a):
    blocks = []
    for m in a["plan"].get("models", []):
        hp, rs = m.get("hyperparameters", {}), m.get("hyperparameter_reasons", {})
        lines = [f"Family: {m['family']}{' (conditional)' if m.get('conditional') else ''}; input: {m.get('input_space')}; "
                 f"seeds {m.get('seeds')}; serves {', '.join(m.get('serves_tasks', []))}; status {m.get('status', 'planned')}; "
                 f"estimated {m.get('estimated_minutes')} min/seed",
                 f"Why this model: {m.get('reason', '')}"]
        lines += [f"- {k} = {_fmt(v)}: {rs.get(k, '')}" for k, v in hp.items()]
        for k, r in a["runs"].items():
            if r.get("model_id") == m["id"]:
                lines.append(f"Training ({k}): {r['epochs_run']} of {r['epochs_planned']} max epochs, kept epoch {r['best_epoch']} "
                             f"({r['stop_reason']}); {r['minutes']} min on {r['device']}; {r['n_fit']} fit / {r['n_val']} validation samples"
                             + (f"; input PCA keeps {_fmt(r['input_pca_variance_explained'])} of variance" if r.get("input_pca_variance_explained") else ""))
        blocks.append((m["id"], lines))
    if a["plan"].get("evaluation_notes"):
        blocks.append(("Evaluation focus", [a["plan"]["evaluation_notes"]]))
    return text_pages("Models, hyperparameters, and training", blocks)


GROUPS = ["fidelity", "memorization", "feature_stats", "likelihood", "representation"]


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
        w = max(len(n) for n in names) + 2
        lines = [" " * w + "".join(f"{k:>16}" for k in keys)]
        for n in names:
            lines.append(f"{n:<{w}}" + "".join(f"{_fmt(a['metrics'][k].get(g, {}).get(n)):>16}" for k in keys))
        notes = {a["metrics"][k][g].get("warning") or a["metrics"][k][g].get("note") for k in keys if g in a["metrics"][k]} - {None}
        if g == "likelihood":
            notes.add("Likelihoods are in each model's own input space and units; do not compare them across models with different input spaces.")
        blocks.append((g.replace("_", " ").capitalize(), ["```" + "\n".join(lines) + "```"] + sorted(notes)))
    return text_pages("Evaluation metrics (held-out real data)", blocks)


def definitions_page(a):
    defs = {}
    for m in a["metrics"].values():
        defs.update(m.get("definitions", {}))
    return text_pages("How to read the metrics", [(None, [f"- {k}: {v}" for k, v in defs.items()] or ["No metrics."])])


CAPTIONS = {
    "loss_curves": "Training curves. The validation set is 10% of the training split and decides early stopping; the held-out curve is shown for reference only and never used to choose anything.",
    "real_vs_synthetic": "Held-out real samples (gray) and synthetic samples in the first two components of the shared evaluation space. Overlap in two components is necessary but not sufficient; the C2ST and MMD metrics use all components.",
    "feature_stats": "Per-feature mean and variance of synthetic vs held-out real data on the transformed scale. Points on the dashed line agree exactly.",
    "count_properties": "Count-level checks for models that generate counts: total counts per sample, and per-feature fraction of zeros.",
    "latent": "Two-dimensional PCA of a VAE's latent codes for held-out samples. A picture for orientation; neighbor-preservation numbers are on the metrics page.",
}


def figure_pages(a):
    figs = []
    for png in sorted((a["out"] / "figures").glob("*.png")):
        cap = next((c for k, c in CAPTIONS.items() if png.stem.startswith(k)), png.stem)
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
        lines = [f"Verdict: {u['verdict'].upper()}  |  task: {u.get('task')}  |  model: {u.get('model_key', '')}", u.get("rationale", "")]
        lines += [f"- evidence: {e['metric']} = {_fmt(e.get('value'))}" + (f" (vs {e['compared_to']})" if e.get("compared_to") else "") for e in u.get("evidence", [])]
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
    blocks = [(None, [c.get("summary", "")])]
    for f in c.get("findings", []):
        blocks.append((f"[{f.get('severity', '')}] {f.get('category', '')}", [f.get("issue", ""), f"Recommendation: {f.get('recommendation', '')}"]))
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
    if a["meta"]:
        items.append(f"Fidelity is measured in a {a['meta'].get('eval_components')}-component PCA space "
                     f"({_fmt(a['meta'].get('eval_variance_explained'))} of training variance); differences outside it are not measured.")
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

    pages = (summary_page(a, n) + objective_page(a) + data_page(a) + models_page(a) + metrics_pages(a) + definitions_page(a)
             + figure_pages(a) + findings_pages(a) + recommendation_pages(a) + critique_pages(a)
             + limitations_pages(a, untraced) + repro_pages(a, n))

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
