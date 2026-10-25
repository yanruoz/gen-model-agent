---
name: downstream-use
description: Turn the evaluation results into outputs/<dataset>/recommendation.json, which says what the generated data (or learned representation) should and should not be used for, each verdict backed by a metric. Used in step 8 of AGENT.md, after run_plan.py (and the critique, if any) and before the report.
---

# Downstream-use recommendations

The objective asked for something specific. This step answers: **given what the evaluation measured, what can this output be used for, under what conditions, and what should it not be used for?** A recommendation is only as strong as the test behind it. If the evaluation didn't test a use, the answer for that use is "not tested", never "recommended".

Read first: `objective.json`, `plan.json`, `run_log.json`, every `metrics/*.json`, and the figures under `figures/`. Then write `outputs/<dataset>/recommendation.json` and run:

```bash
venv/bin/python scripts/validate_recommendation.py outputs/<dataset>/recommendation.json
```

## Schema

```json
{
  "uses": [
    {
      "task": "representation_learning",
      "use": "Use the VAE latent codes as the input to clustering or neighbor graphs of these samples",
      "model_key": "vae_nb_s0",
      "verdict": "recommended | conditional | not_recommended",
      "evidence": [
        {"metric": "vae_nb_s0.representation.trustworthiness", "value": 0.81,
         "compared_to": "vae_nb_s0.representation.trustworthiness_pca_baseline = 0.83"}
      ],
      "rationale": "What the evidence shows for this use, in one or two sentences",
      "caveats": ["Conditions or limits on the use"]
    }
  ],
  "not_tested": ["Uses someone might want but the evaluation did not test, and what test would be needed"],
  "privacy_statement": "Required when sharing, releasing, or privacy comes up (see below)"
}
```

- `metric` is a dotted path into `metrics/<key>.json`: `<key>.<group>.<name>`. The validator resolves it and checks `value` against it, so copy the value exactly, with the same rounding.
- Cover the primary task (`objective.effective_task`) and each secondary task with at least one use.
- When there are several models, say which one to use for what, and why. A verdict per model is fine.

## How to decide a verdict

Compare against the **reference** values stored next to each metric, not against an imagined ideal:

- **Fidelity:** `c2st_auc` vs `c2st_auc_reference`. A C2ST AUC close to the reference means synthetic samples are hard to tell from real ones. Well above it (e.g. 0.9 vs a reference of 0.55) means a classifier separates them easily. Then synthetic samples can't stand in for real data in analyses that are sensitive to the joint distribution, though they may still be fine for uses that only need marginals (check `feature_stats`). MMD works the same way against `mmd2_reference`. Precision and recall separately say whether samples are realistic (precision) and whether they cover the real variety (recall).
- **Representation:** `trustworthiness` and `knn_overlap` against their PCA baselines. A latent space that doesn't beat a same-size PCA isn't useless, but the recommendation must say a PCA would do about as well for neighbor-based uses. Recommend it only for what it adds (e.g. a count-aware model of the data, a generator for the same space).
- **Images:** fidelity is in the evaluation classifier's feature space. `frechet_distance` against `frechet_distance_reference` works like MMD. `conditional.label_agreement` against `label_agreement_reference` says whether the model actually produces the requested class. For class balancing, a model whose images aren't recognized as the target class can't help that class, whatever its other scores. Check `memorization_pixels` for near-copies too.
- **Class balancing / augmentation (`metrics/downstream.json`, `balancing`):** compare `arms.real_plus_<key>.target_f1_mean` with the `real` and `resampled` arms and, when present, `extra_real` (new real data: the best case for any generator), using the seed sd (`*_sd`, and `noise_sd_target_f1`). Synthetic data is `recommended` for balancing only if it beats resampling by more than about twice the sd without lowering `macro_f1_mean`. If `extra_real` doesn't beat resampling either, say that no generator could have, and recommend resampling. Cite paths like `downstream.balancing.arms.real_plus_cdiff_s0.target_f1_mean`.
- **Replacing real data (`downstream.tstr`):** `utility_ratio.synthetic_<key>` is synthetic-trained accuracy over real-trained accuracy at the same size. Near 1 (within the sd) supports using the synthetic set as stand-in training data; well below 1 means it loses information a classifier needs.
- **Anomaly detection (`metrics/anomaly.json`):** compare each model's `auroc` per corruption and `auroc_mean` with both baselines. A model is recommended for a kind of anomaly only where it beats the baselines. Say which corruption types it catches and which it misses, and that the anomalies are synthetic.
- **Imputation (`metrics/imputation.json`):** compare the model's `rmse`/`mae` with `baseline_mean` and `baseline_knn`. Recommend it only if it beats both, and state the scale the error is on.
- **Memorization:** `dcr_ratio` well below 1, or `too_close_fraction` far above 0.05, means synthetic samples sit close to training samples. Treat as a red flag for any sharing use.
- **Training:** read `run_log.json` and the run sidecars. A model stopped by the time cap, or with a large train-to-held-out gap, gets a caveat.
- `recommended` needs supporting evidence and no contradicting metric. `conditional` means "yes, if ..." and the condition goes in `caveats`. `not_recommended` is a normal, useful answer: say what failed.
- If a simpler alternative did as well (PCA baseline, resampling), say so in the rationale.

## Privacy (hard rule)

This pipeline measures **memorization** (distance from synthetic to training samples). That is evidence about one kind of leakage, not a privacy guarantee. Whenever the objective or a use mentions sharing, releasing, publishing, or privacy:

- `privacy_statement` must say plainly that there is **no formal privacy guarantee** (e.g. differential privacy), and what the memorization check does and does not show, with its numbers.
- Never write that synthetic data is anonymized, de-identified, privacy-preserving, private, or safe to share or release. The validator rejects these phrases.
- A sharing use can be `conditional` at most, with the condition being a proper privacy review or a formally private training method.
