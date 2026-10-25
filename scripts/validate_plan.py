"""Validates outputs/<dataset>/plan.json against the schema in .claude/skills/model-selection/SKILL.md,
   the validated objective.json, and profile.json. Deterministic, no LLM involved.
   Pattern from Project 1 (pass/fail JSON + Markdown report, lazy-reason rejection), rules rewritten
   for generative models.

   Usage: venv/bin/python scripts/validate_plan.py outputs/<d>/plan.json
   Reads objective.json and profile.json from the same folder. Exit 0 = pass.
"""
import argparse
import json
import sys
from pathlib import Path

from plan_schema import (
    IMAGE_HYPERPARAMS, IMPLEMENTED, INPUT_SPACES, KNOWN_PREPROCESSING_STEPS, LIKELIHOODS, MIN_SEEDS, MIN_TRAIN_SAMPLES,
    MODEL_FAMILIES, MODEL_STATUSES, NONNEG_ONLY_STEPS, OBJECTIVE_TASKS, REQUIRED_HYPERPARAMS,
    STEP_KIND_ORDER, TASK_FAMILIES, TEST_FRACTION_RANGE, is_lazy_reason, serves,
)


def _n_after_rows(plan, profile):
    """Samples left after row steps (subsampling), before the train/test split."""
    n = profile["n_samples"]
    for step in plan.get("preprocessing", []):
        if step.get("step") in ("subsample", "stratified_subsample"):
            n = min(n, int((step.get("params") or {}).get("n_samples", n)))
    return n


def _check_preprocessing(plan, profile, errors):
    steps = plan.get("preprocessing")
    if not isinstance(steps, list):
        errors.append("'preprocessing' must be a list (may be empty)")
        return
    last_kind = -1
    for i, step in enumerate(steps):
        name = step.get("step") if isinstance(step, dict) else None
        if name not in KNOWN_PREPROCESSING_STEPS:
            errors.append(f"preprocessing[{i}]: unknown step {name!r}; allowed {sorted(KNOWN_PREPROCESSING_STEPS)}")
            continue
        if is_lazy_reason(step.get("reason")):
            errors.append(f"preprocessing '{name}': reason is missing or too generic")
        kind = STEP_KIND_ORDER[name]
        if kind < last_kind:
            errors.append(f"preprocessing '{name}' is out of order: row steps, then feature selection, then transforms")
        last_kind = max(last_kind, kind)
        if name in NONNEG_ONLY_STEPS and profile.get("value_range") and profile["value_range"][0] < 0:
            errors.append(f"preprocessing '{name}' needs non-negative data, but the minimum value is {profile['value_range'][0]}")
        params = step.get("params") or {}
        if name == "select_top_variance":
            k = params.get("n_features")
            if not isinstance(k, int) or not (1 <= k <= profile["n_features"]):
                errors.append(f"select_top_variance: n_features must be an int in [1, {profile['n_features']}]")
        if name in ("subsample", "stratified_subsample"):
            k = params.get("n_samples")
            if not isinstance(k, int) or k < 2:
                errors.append(f"{name}: n_samples must be an int >= 2")
            if name == "stratified_subsample" and not profile.get("has_labels"):
                errors.append("stratified_subsample needs labels; use 'subsample' instead")


def _check_model(m, plan, profile, objective, n_train, errors, warnings):
    mid = m.get("id", "?")
    family = m.get("family")
    if not isinstance(m.get("id"), str) or not m["id"].replace("_", "").isalnum():
        errors.append(f"model {mid!r}: 'id' must be a short alphanumeric/underscore string")
    if family not in MODEL_FAMILIES:
        errors.append(f"model {mid}: family {family!r} not in {sorted(MODEL_FAMILIES)}")
        return
    status = m.get("status", "planned")
    if status not in MODEL_STATUSES:
        errors.append(f"model {mid}: status must be one of {sorted(MODEL_STATUSES)}")
    if is_lazy_reason(m.get("reason")):
        errors.append(f"model {mid}: reason is missing or too generic")

    # The model must serve at least one of the tasks this plan is for.
    tasks = m.get("serves_tasks") or []
    plan_tasks = {objective["effective_task"], *objective.get("secondary_tasks", [])}
    if not tasks:
        errors.append(f"model {mid}: 'serves_tasks' must list the task(s) it is for")
    for t in tasks:
        if t not in OBJECTIVE_TASKS or t not in plan_tasks:
            errors.append(f"model {mid}: serves_tasks entry {t!r} is not one of this run's tasks {sorted(plan_tasks)}")
        elif not serves(family, t):
            errors.append(f"model {mid}: a {family} lacks the capability task '{t}' needs (see plan_schema.TASK_NEEDS)")
        elif t in TASK_FAMILIES and family not in TASK_FAMILIES[t]:
            errors.append(f"model {mid}: the '{t}' test can only score {sorted(TASK_FAMILIES[t])} models")
        if t == "anomaly_detection" and m.get("conditional"):
            errors.append(f"model {mid}: anomaly scores must not depend on a class label; use an unconditional model")

    conditional = bool(m.get("conditional", False))
    if conditional and not profile.get("has_labels"):
        errors.append(f"model {mid}: conditional models need labels; this dataset has none")
    modality = profile.get("modality", "tabular")
    if status == "planned" and (family, modality, conditional) not in IMPLEMENTED:
        errors.append(
            f"model {mid}: ({family}, {modality}, conditional={conditional}) is not implemented. "
            f"Implemented: {sorted(IMPLEMENTED)}"
        )

    # Input space and likelihood must fit the data.
    space = m.get("input_space")
    if space not in INPUT_SPACES:
        errors.append(f"model {mid}: input_space must be one of {sorted(INPUT_SPACES)}")
    hp = m.get("hyperparameters") or {}
    reasons = m.get("hyperparameter_reasons") or {}
    needed = REQUIRED_HYPERPARAMS[family] | (IMAGE_HYPERPARAMS.get(family, set()) if modality == "image" else set())
    missing = needed - set(hp)
    if missing:
        errors.append(f"model {mid}: missing hyperparameters {sorted(missing)}")
    if modality == "image":
        # Conv models read pixels scaled to [0, 1] in their image shape: the 'transformed' view after scale_unit_range.
        if space != "transformed":
            errors.append(f"model {mid}: image models train on input_space 'transformed' (pixels after scale_unit_range)")
        if "scale_unit_range" not in [s.get("step") for s in plan.get("preprocessing", [])]:
            errors.append(f"model {mid}: image models need the scale_unit_range preprocessing step")
        if family == "diffusion" and isinstance(hp.get("timesteps"), int) and isinstance(hp.get("sampling_steps"), int) \
                and hp["sampling_steps"] > hp["timesteps"]:
            errors.append(f"model {mid}: sampling_steps cannot exceed timesteps")
    if space == "pca":
        missing_pca = "pca_components" not in hp
        if missing_pca:
            errors.append(f"model {mid}: input_space 'pca' needs a 'pca_components' hyperparameter")
    for key in hp:
        if is_lazy_reason(reasons.get(key)):
            errors.append(f"model {mid}: hyperparameter '{key}' has no specific reason in hyperparameter_reasons")
    if family == "vae":
        lik = hp.get("likelihood")
        if lik not in LIKELIHOODS:
            errors.append(f"model {mid}: likelihood must be one of {sorted(LIKELIHOODS)}")
        if lik == "nb":
            if space != "raw" or not profile.get("is_nonneg_integer"):
                errors.append(f"model {mid}: a negative binomial likelihood needs input_space 'raw' on non-negative integer data")
            elif profile.get("modality") == "image":
                errors.append(f"model {mid}: pixel intensities are not counts; use a gaussian or bernoulli likelihood")
        if lik == "bernoulli" and space == "pca":
            errors.append(f"model {mid}: a bernoulli likelihood needs data in [0, 1], not PCA scores")
    if space == "raw" and family != "vae":
        warnings.append(f"model {mid}: {family} on untransformed raw values; check the scale is suitable")

    # Seeds and data size.
    seeds = m.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        errors.append(f"model {mid}: 'seeds' must be a non-empty list of ints")
        seeds = []
    elif len(set(seeds)) != len(seeds):
        errors.append(f"model {mid}: duplicate seeds")
    if status == "planned" and len(seeds) < MIN_SEEDS.get(family, 1):
        errors.append(f"model {mid}: a {family} needs at least {MIN_SEEDS[family]} seeds (results vary a lot between seeds)")
    if status == "planned" and n_train < MIN_TRAIN_SAMPLES[family]:
        errors.append(f"model {mid}: {n_train} training samples is below the {MIN_TRAIN_SAMPLES[family]} minimum for a {family}")

    est = m.get("estimated_minutes")
    if not isinstance(est, (int, float)) or est <= 0:
        errors.append(f"model {mid}: 'estimated_minutes' (per seed) must be a positive number")
        est = 0
    return (est * len(seeds)) if status == "planned" else 0


def validate_plan(plan, objective, profile):
    errors, warnings = [], []
    for field in ("dataset", "seed", "budget_min", "split", "preprocessing", "models", "revision"):
        if field not in plan:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors, warnings

    if not isinstance(plan["seed"], int):
        errors.append("'seed' must be an int")

    # A held-out real split always exists; synthetic data never enters it.
    split = plan["split"]
    frac = split.get("test_fraction")
    lo, hi = TEST_FRACTION_RANGE
    if not isinstance(frac, (int, float)) or not (lo <= frac <= hi):
        errors.append(f"split.test_fraction must be in [{lo}, {hi}]")
        frac = 0.2
    if is_lazy_reason(split.get("reason")):
        errors.append("split: reason is missing or too generic")
    if split.get("stratify") and not profile.get("has_labels"):
        errors.append("split.stratify needs labels")

    _check_preprocessing(plan, profile, errors)

    models = plan["models"]
    if objective["decision"] == "decline":
        if models:
            errors.append("the objective was declined, so 'models' must be empty")
        return errors, warnings
    if not isinstance(models, list) or not models:
        errors.append("'models' must be a non-empty list")
        return errors, warnings
    ids = [m.get("id") for m in models]
    if len(set(ids)) != len(ids):
        errors.append("model ids must be unique")

    n_train = int(_n_after_rows(plan, profile) * (1 - frac))
    total = sum(_check_model(m, plan, profile, objective, n_train, errors, warnings) or 0 for m in models)

    # At least one model that will actually run must serve the primary (effective) task.
    task = objective["effective_task"]
    if not any(task in (m.get("serves_tasks") or []) and m.get("status", "planned") == "planned" for m in models):
        errors.append(f"no planned model serves the primary task '{task}'")

    budget = plan["budget_min"]
    if not isinstance(budget, (int, float)) or budget <= 0:
        errors.append("'budget_min' must be a positive number")
    elif total > budget:
        errors.append(
            f"planned models need ~{total:g} min (estimated_minutes x seeds) but the budget is {budget:g} min; "
            "shrink a model or mark it status 'not_run_budget' with its reason"
        )
    return errors, warnings


def write_reports(errors, warnings, plan_path):
    plan_path = Path(plan_path)
    report = {"passed": not errors, "errors": errors, "warnings": warnings}
    json_path = plan_path.with_name("plan_validation.json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Plan validation", "", f"**Result:** {'PASS' if not errors else 'FAIL'}"]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    plan_path.with_name("plan_validation.md").write_text("\n".join(lines) + "\n")
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate plan.json against objective.json and profile.json.")
    parser.add_argument("plan_json")
    args = parser.parse_args()

    plan_path = Path(args.plan_json)
    try:
        plan = json.loads(plan_path.read_text())
        objective = json.loads(plan_path.with_name("objective.json").read_text())
        profile = json.loads(plan_path.with_name("profile.json").read_text())
        obj_check = json.loads(plan_path.with_name("objective_validation.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: could not read plan/objective/profile/objective_validation: {e}")

    if not obj_check.get("passed"):
        errors, warnings = ["objective.json has not passed validate_objective.py"], []
    else:
        errors, warnings = validate_plan(plan, objective, profile)
    out = write_reports(errors, warnings, plan_path)
    print(f"{'PASS' if not errors else 'FAIL'}: {plan_path}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"Wrote {out}")
    sys.exit(0 if not errors else 1)
