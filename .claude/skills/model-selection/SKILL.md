---
name: model-selection
description: Choose generative model(s) and hyperparameters for the validated objective, and the full plan.json schema. Used in step 4 of AGENT.md, right after data-profiling decides split and preprocessing.
---

# Model selection and plan.json schema

## plan.json schema (authoritative)

```json
{
  "dataset": "<dataset>",
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
- **class_balancing / augmentation on images** → three conditional models compared (`conditional: true`, all on `input_space: transformed` after `scale_unit_range`): conditional VAE (fast baseline, expect blurry samples), conditional GAN with 2 seeds, conditional diffusion. If diffusion doesn't fit the budget, keep it with `status: not_run_budget`. `run_plan.py` then runs the downstream test automatically: the same classifier is trained on real data, real data with resampling, and real data plus each model's target-class samples, and all are scored on held-out real data. That test is what the recommendation rests on.
- **data_generation on images** → unconditional or conditional diffusion if the budget allows, otherwise GAN (2 seeds); a VAE only as a fast baseline.

Don't add models just to fill the table: each one needs a reason tied to the objective, and it needs a comparison the evaluation can actually make.

## Step 2: hyperparameters (worked heuristics; every value needs a reason citing this data)

`n_train = n_samples_after_row_steps x (1 - test_fraction)`.

**VAE**
- `latent_dim`: 10 when `n_train` < 5,000; 16 up to 50,000; 32 above. A small latent space is enough for a few thousand samples and gives a representation that is easy to inspect. More dimensions mostly fit noise at this size.
- `hidden_dims`: `[256, 128]` when the input has >= 1,000 features, else `[128, 64]`.
- `batch_size`: 128 (for `n_train` < 1,000, use 64 so an epoch has more than a handful of steps).
- `epochs`: a **maximum**. `train.py` always sets aside 10% of the training split for validation, stops when validation loss hasn't improved for 25 epochs, and keeps the best weights, so too many epochs costs time, not overfitting. Use 300–400 for a few thousand samples, and lower it only to fit the budget. On small datasets, held-out loss typically bottoms out well before the maximum.
- `kl_warmup_epochs`: about 10% of epochs. Ramping the KL weight from 0 to 1 avoids posterior collapse early in training.
- `lr`: 1e-3 (Adam). Lower to 5e-4 if the loss curve from a previous run oscillates.

**Flow (RealNVP, affine coupling)**
- `pca_components`: density estimation needs many samples per dimension. Use 10–20 when `n_train` < 5,000, up to 50 above that. The variance the components keep is recorded in the run sidecar (`input_pca_variance_explained`). When generating, the flow adds independent Gaussian noise with the training residual variance outside these components, so its samples are full-dimensional. Say in the reason that the flow models the top components and treats the rest as noise.
- `n_layers`: 6–8 coupling layers. `hidden_dim`: 128 (256 above 20k samples).
- `epochs`, `batch_size`, `lr`: as for the VAE (epochs is a maximum, early stopping applies); `lr` 1e-3.

**Image models** (28x28-scale images, a few thousand to tens of thousands of training images)
- Conditional VAE: `latent_dim` 32, `base_channels` 32, `hidden_dims` [256], `likelihood` "bernoulli" (pixels in [0, 1]), `kl_warmup_epochs` about 10% of epochs, `epochs` 60 (maximum; early stopping applies), `batch_size` 128, `lr` 1e-3. Expect blurry means; that is the known VAE trade-off, and the reason to compare it with sharper families.
- Conditional GAN: `latent_dim` 64, `base_channels` 64, `epochs` 20–30 (a fixed count: GANs are not early-stopped because their losses don't measure sample quality), `batch_size` 128, `lr` 2e-4, `seeds` [0, 1].
- Conditional diffusion: `timesteps` 400, `base_channels` 32, `sampling_steps` 50 (DDIM), `batch_size` 128, `lr` 2e-4. Diffusion keeps improving with training much longer than the other families, so give it the largest share of the budget that fits (`epochs` as the budget allows; early stopping applies). `guidance_scale` trades diversity for class fidelity; 2–4 is the usual range, and the upper end suits class balancing, where producing the requested class matters most.
- Subsampling: with a 20–30 min budget and these three families, about 20,000 images (stratified) keeps every model's training to minutes. State the time arithmetic in the reason.

**Seeds**: one seed is enough for a stable family (VAE, flow) under a tight budget; give it 2 if the budget allows and the claim depends on stability. GANs: always >= 2.

## Timing table (Apple M4 CPU/MPS; per seed)

Measured on this machine for training to the maximum epochs (generation + evaluation add about 0.2 min per model at this size). Scale roughly linearly with samples x features for the VAE and with samples for the flow. Use these for `estimated_minutes` and round up generously; the estimate should cover the maximum, even though early stopping usually ends sooner. These are timing benchmarks only, not evidence about how any particular dataset behaves.

| Model | Size | Setting | Minutes to max epochs |
|---|---|---|---|
| VAE (nb), CPU | ~2,000 samples x 2,000 features | 400 epochs, batch 128, [256, 128] | 0.9 |
| Flow (RealNVP), CPU | ~2,000 samples x 20 dims | 8 layers, hidden 128, 400 epochs | ~0.2 |

Images, Apple MPS, ~14,000 training images of 28x28x3 (per epoch; multiply by epochs, scale linearly with image count):

| Model | Setting | Seconds per epoch |
|---|---|---|
| Conditional VAE | base 32, latent 32 | ~2 |
| Conditional GAN | base 64, latent 64 | ~8 |
| Conditional diffusion | base 32, 400 timesteps | ~12–13 |

Not part of `budget_min` but part of the run's wall-clock time: diffusion sampling takes about 0.02 s per image (50 guided steps), each image model's evaluation about 1 min, training the evaluation classifier once about 1 min, and the class-balancing test about 4 min.
