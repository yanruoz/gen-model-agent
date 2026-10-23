---
name: data-profiling
description: Turn profile.json + DATA_DESCRIPTION.md into plan.json's split and preprocessing block for generative modeling. Used in step 4 of CLAUDE.md, after the objective is validated and before model selection.
---

# Preprocessing for generative models

Read `profile.json`, `profile_summary.md`, and `DATA_DESCRIPTION.md`. Decide the `split` block and the `preprocessing` list of `plan.json` (full schema in `model-selection/SKILL.md`). Every step needs a `reason` that cites this dataset (a number from the profile, a fact from the description).

`scripts/preprocess.py` runs these steps deterministically. It always does them in the same order: row steps, then the train/test split, then feature selection (fit on the training split only), then transforms (fit on the training split only). It saves three views of the data that models can choose between via `input_space`:

- `raw`: the selected features, untransformed (counts stay integers);
- `transformed`: after the transform steps;
- `pca`: PCA of `transformed`, fit on training data. Each model sets its own size with `pca_components`.

It also builds the **evaluation space** that every tabular model is scored in: PCA (up to 50 components) of `transformed`, fit on training data. Models with different input spaces are therefore compared in one common space.

## The split (always)

`split.test_fraction` between 0.1 and 0.4. Held-out real samples are the only thing synthetic data is ever compared against, so they must never be used for training or for fitting preprocessing. 0.2 is a reasonable starting point; give the reason in terms of this dataset's size. For example, for 2,700 cells, 0.2 leaves 540 held-out cells, enough for two-sample tests while keeping 2,160 for training. Set `stratify: true` when labels exist, so the held-out set keeps the class mix.

## Decision branches (pick the first that matches)

1. **Image** (`modality: image`). Use `scale_unit_range` (pixel 0–255 to [0, 1]), which suits bounded-output decoders and keeps the metric scale interpretable. If `n_samples` is far more than the time budget can train on (rough guide: above 30,000 with a 20–30 minute budget), add `stratified_subsample` (or `subsample` if unlabeled) first, and give the number and the time reasoning.
2. **Count matrix** (`modality: tabular`, `is_nonneg_integer: true`, high sparsity, e.g. scRNA-seq):
   - `select_top_variance` with `n_features` around 2,000 when `n_features` is in the tens of thousands. Ranking is done on library-size-normalized, log-transformed training data. Most genes are almost all zeros and add noise and training time, not signal.
   - Then `normalize_total` and `log1p`. These define the `transformed` view (and so the evaluation space) on the scale where cell-to-cell distances mean something. A count-likelihood model (negative binomial VAE) still trains on `raw` counts. The transforms do not affect it, except through the evaluation space.
3. **Continuous tabular** (not integer, or negative values present): `standardize` (zero mean, unit variance per feature, fit on training data), so no feature dominates a Gaussian likelihood or a distance just because of its units. Use `select_top_variance` only if `n_features` is in the thousands.

## Not supported (say so; don't work around it)

- **Missing values.** No generative model here trains on NaNs, and there is no imputation step. If `has_missing` is true, record it in the objective's `decision_reason` (decline, or adapt the task) rather than inventing a fill rule.
- **QC filtering of rows** (removing low-quality cells, doublets). Not available as a step. If `DATA_DESCRIPTION.md` warns about it (PBMC's raw file does), state it as a limitation in the report; don't silently drop rows.
- **Categorical features.** The loader refuses them (inherited from Project 1).
