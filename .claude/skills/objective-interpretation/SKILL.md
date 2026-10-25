---
name: objective-interpretation
description: Turn the user's free-text objective into outputs/<dataset>/objective.json (task, decision whether to model at all, success criterion), checked by validate_objective.py. Used in step 3 of AGENT.md, after profiling and before preprocessing/model selection.
---

# Objective interpretation

You get one free-text objective (from the `objective:` line of the run prompt) and the dataset's `profile.json` + `DATA_DESCRIPTION.md`. Your job is to decide **what is being asked**, **whether generative modeling is a sound way to do it on this data**, and **how success will be measured**. Write the result to `outputs/<dataset>/objective.json`, then run:

```bash
venv/bin/python scripts/validate_objective.py outputs/<dataset>/objective.json --objective "<objective text, verbatim>"
```

Fix and re-run until it passes. The validator enforces the data checks below in code; your job is to get the reasoning right, not just to pass.

## Schema

```json
{
  "raw_text": "<the objective, verbatim>",
  "task": "representation_learning",
  "secondary_tasks": ["data_generation"],
  "target": {"class": null},
  "success_criterion": "What measurable result on held-out real data would count as success",
  "interpretation_reason": "Which words in the objective map to which task, and why",
  "decision": "proceed | proceed_adapted | decline",
  "decision_reason": "Why generative modeling is (or is not) appropriate here, citing profile.json",
  "effective_task": "the task that will actually run (= task for proceed, another task for proceed_adapted, null for decline)",
  "ambiguities": ["Other readings of the objective, and why they were not chosen"]
}
```

## Step 1: map words to a task

Pick exactly one primary `task` from `plan_schema.OBJECTIVE_TASKS`. Add `secondary_tasks` only when the objective asks for two things explicitly ("learn a representation **and** generate samples").

| The objective talks about... | Task |
|---|---|
| synthetic / simulated / fake samples, "realistic data", sharing or releasing data | `data_generation` |
| "more training data", improving a downstream model in general | `augmentation` |
| rare / minority / under-represented classes, "balance" | `class_balancing` (needs labels) |
| missing values, dropout recovery, filling in, denoising | `imputation` |
| outliers, unusual / abnormal / out-of-distribution samples, QC flags | `anomaly_detection` |
| embedding, latent space, low-dimensional representation, features for clustering | `representation_learning` |
| confidence, likelihood, calibrated density, "how sure" | `uncertainty_quantification` |

If the objective is vague ("do something useful with this data"), pick the task that the data supports best and that the evaluation can actually measure, and list the other plausible readings in `ambiguities`. Never stop to ask: record the judgment call (AGENT.md rule 8 on assumptions).

## Which tasks can be tested here

Every task needs an objective test on held-out real data, or nobody can judge the result. Implemented tests (`plan_schema.TASK_SUPPORT`):

| Task | Tabular | Images | Test |
|---|---|---|---|
| data_generation | yes | yes | two-sample realism metrics; labeled images also get train-on-synthetic, test-on-real |
| representation_learning | yes | no | latent space vs a same-size PCA |
| class_balancing, augmentation | no | yes (labeled) | headroom check, then classifier on held-out real data |
| anomaly_detection | yes | yes | corrupted held-out samples, AUROC vs non-generative baselines |
| imputation | yes | yes | masked held-out values vs mean and kNN imputation |
| uncertainty_quantification | no | no | none |

If the requested task has no test for this data, the validator requires `decline` or `proceed_adapted` to a testable task. Say so plainly in `decision_reason`.

## Step 2: decide whether to model at all

Check each against `profile.json`, in order. The validator enforces the first three.

1. **Too little data:** `n_samples < 200` means `decline`.
2. **Needs labels it doesn't have:** `class_balancing` on a dataset with `has_labels: false` can't proceed as asked. Either `decline`, or `proceed_adapted` with an `effective_task` that doesn't need labels (usually `data_generation`). Say plainly in `decision_reason` which one you chose and why. Don't invent labels (e.g. clusters relabeled as classes) and call it class balancing.
3. **Nearly balanced classes:** for `class_balancing`, if `imbalance_ratio < 1.5`, the decision must be `proceed_adapted`. The run still trains, but the claim tested is "does synthesis beat simple resampling at all", not "synthesis fixes an imbalance". Between 1.5 and about 3, `proceed` is allowed, but the evaluation must still compare against resampling (the model-selection skill makes this part of the plan).
4. **A simpler method would do the job:** if the objective could be met without a generative model (e.g. "find outliers" on a 5-feature table where a z-score would do), say so in `decision_reason`. You may still proceed, but the report has to compare against that simpler baseline.
5. **Privacy wording:** if the objective mentions sharing, releasing, privacy, or anonymity, note in `ambiguities` that this pipeline checks memorization (nearest-neighbor distances), which is evidence, not a formal privacy guarantee such as differential privacy. Never state or imply that synthetic data is "private", "anonymized", or "safe to share".

## Step 3: success criterion

One sentence naming the measurement on **held-out real data** that decides success. Examples:
- representation_learning: "held-out reconstruction likelihood is close to the training value, and the latent space preserves nearest-neighbor structure (trustworthiness) of the log-normalized data".
- data_generation: "a classifier cannot tell synthetic from held-out real samples well (C2ST AUC close to its real-vs-real reference), and synthetic samples are no closer to training samples than held-out real samples are".
- class_balancing: "minority-class F1 on held-out real data improves over real data plus simple resampling by more than the seed noise".
- data_generation meant to replace real training data: "a classifier trained only on synthetic data reaches accuracy close to one trained on the same number of real samples (utility ratio near 1), with realism metrics near their real-vs-real references".
- anomaly_detection: "the model's anomaly score separates corrupted from uncorrupted held-out samples (AUROC) better than the non-generative baselines".
- imputation: "error on masked held-out values is lower than mean and kNN imputation".

## Target class

For `class_balancing`, `target.class` is a class label from `profile.json`'s `class_counts` keys, or `"auto"` for the smallest class (`profile.minority_class`). Use `null` for other tasks.
