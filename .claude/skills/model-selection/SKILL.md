---
name: model-selection
description: Choose generative model(s) and hyperparameters for the validated objective, and the full plan.json schema. Used in step 4 of CLAUDE.md, right after data-profiling decides split and preprocessing.
---

# Model selection and plan.json schema

## plan.json schema (authoritative)

```json
{
  "dataset": "pbmc",
  "seed": 0,
  "budget_min": 20,
  "split": {"test_fraction": 0.2, "stratify": false, "reason": "..."},
  "preprocessing": [
    {"step": "select_top_variance", "params": {"n_features": 2000}, "reason": "..."},
    {"step": "normalize_total", "reason": "..."},
    {"step": "log1p", "reason": "..."}
  ],
  "models": [
    {
      "id": "vae_nb",
      "family": "vae",
      "conditional": false,
      "input_space": "raw",
      "hyperparameters": {"latent_dim": 10, "hidden_dims": [256, 128], "epochs": 300,
                          "batch_size": 128, "lr": 0.001, "likelihood": "nb", "kl_warmup_epochs": 30},
      "hyperparameter_reasons": {"latent_dim": "...", "hidden_dims": "...", "...": "one per key"},
      "seeds": [0],
      "serves_tasks": ["representation_learning", "data_generation"],
      "estimated_minutes": 1.5,
      "status": "planned",
      "reason": "Why this family fits this objective and this data, and what it is compared against"
    }
  ],
  "evaluation_notes": "Which metrics matter most for this objective and why (informational)",
  "revision": 1,
  "critique_applied": null
}
```

- `id`: short, unique, `[a-z0-9_]`. Artifacts are keyed `<id>_s<seed>`, so seeds never overwrite each other.
- `serves_tasks`: the tasks from `objective.json` (`effective_task` and `secondary_tasks`) this model is for. The validator checks the family has the needed capability (`plan_schema.TASK_NEEDS` / `MODEL_CAPABILITIES`).
- `estimated_minutes`: per seed, from the timing table below. Planned minutes x seeds, summed over models, must fit `budget_min` (from the run prompt; default 20).
- `status`: `planned`, or `not_run_budget` for a model you would run but that doesn't fit the budget. Keep its estimate and reason, so the report says what was left out and why. This never blocks the run.
- Validate: `venv/bin/python scripts/validate_plan.py outputs/<dataset>/plan.json`.

## Step 1: which families (objective x modality)

Capabilities are fixed (`plan_schema.MODEL_CAPABILITIES`):

| | VAE | GAN | Flow | Diffusion |
|---|---|---|---|---|
| Exact likelihood | no (lower bound) | no | **yes** | no |
| Latent representation | **yes** | no | no (same dim as data) | no |
| Sample quality on images | blurry | sharper, unstable | weak on raw pixels | best, slowest |
| Stability | high | **low (needs >= 2 seeds)** | high | high |

Rules (the first match sets the primary model; add one comparator where the budget allows):

- **representation_learning** → VAE. On count data (`is_nonneg_integer`, tabular, sparse), use `likelihood: nb` on `input_space: raw`. Raw counts with a library-size-aware negative binomial decoder avoid the distortion that log-transforming sparse counts adds. On continuous data use `gaussian` on `transformed`.
- **data_generation, tabular** → the VAE above as primary, plus a **flow on `pca`** as a comparator: an exact-likelihood model with a different inductive bias, scored in the same evaluation space. A GAN is allowed only with a dataset-specific reason; on a few thousand rows it is the least stable choice.
- **anomaly_detection / uncertainty_quantification** → flow (exact likelihood), on `pca` for high-dimensional data, with the VAE's ELBO as a cross-check. A GAN is invalid here (no likelihood).
- **imputation** → VAE (reconstruction). Diffusion inpainting for images only if the budget allows.
- **class_balancing / augmentation on images** → conditional models compared: conditional VAE (fast baseline), conditional GAN with 2 seeds, conditional diffusion (or `not_run_budget`).

Don't add models just to fill the table: each one needs a reason tied to the objective, and it needs a comparison the evaluation can actually make.

## Step 2: hyperparameters (worked heuristics; every value needs a reason citing this data)

`n_train = n_samples_after_row_steps x (1 - test_fraction)`.

**VAE**
- `latent_dim`: 10 when `n_train` < 5,000; 16 up to 50,000; 32 above. A small latent space is enough for a few thousand samples and gives a representation that is easy to inspect. More dimensions mostly fit noise at this size.
- `hidden_dims`: `[256, 128]` when the input has >= 1,000 features, else `[128, 64]`.
- `batch_size`: 128 (for `n_train` < 1,000, use 64 so an epoch has more than a handful of steps).
- `epochs`: aim for roughly 5,000–10,000 gradient steps: `epochs ≈ 7,500 / ceil(n_train / batch_size)`, clipped to [50, 400]. For 2,160 cells at batch 128 (17 steps/epoch), that is about 440, so 400.
- `kl_warmup_epochs`: about 10% of epochs. Ramping the KL weight from 0 to 1 avoids posterior collapse early in training.
- `lr`: 1e-3 (Adam). Lower to 5e-4 if the loss curve from a previous run oscillates.

**Flow (RealNVP, affine coupling)**
- `pca_components`: density estimation needs many samples per dimension. Use 10–20 when `n_train` < 5,000, up to 50 above that, and state the variance this keeps (from the preprocessing/eval PCA if known).
- `n_layers`: 6–8 coupling layers. `hidden_dim`: 128 (256 above 20k samples).
- `epochs`, `batch_size`, `lr`: same step-count rule as the VAE; `lr` 1e-3.

**Seeds**: one seed is enough for a stable family (VAE, flow) under a tight budget; give it 2 if the budget allows and the claim depends on stability. GANs: always >= 2.

## Timing table (Apple M4 CPU/MPS; per seed)

Measured on this machine; use it for `estimated_minutes` and round up.

| Model | Data | Setting | Minutes |
|---|---|---|---|
| VAE (nb) | 2,160 x 2,000 counts | 400 epochs, batch 128 | TBD |
| Flow | 2,160 x 20 PCA | 8 layers, 400 epochs | TBD |
