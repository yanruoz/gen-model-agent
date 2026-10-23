"""Validates outputs/<dataset>/objective.json: the agent's structured reading of the user's
   free-text objective, checked against the dataset's profile.json.

   The "whether to model at all" checks live here, in code, so they can't be skipped:
   - a task that needs labels on a dataset without labels must be declined or adapted;
   - class balancing on nearly balanced classes must be framed as proceed_adapted;
   - too few samples for any generative model means decline.

   Usage: venv/bin/python scripts/validate_objective.py outputs/<d>/objective.json --objective "<text>"
   Writes objective_validation.json/.md next to objective.json; exit 0 = pass.
"""
import argparse
import json
import sys
from pathlib import Path

from plan_schema import (
    DECISIONS, MIN_SAMPLES_ANY_MODEL, NEAR_BALANCED_RATIO, OBJECTIVE_TASKS,
    TASKS_REQUIRING_LABELS, is_lazy_reason,
)

REQUIRED_FIELDS = {
    "raw_text": str, "task": str, "secondary_tasks": list, "success_criterion": str,
    "interpretation_reason": str, "decision": str, "decision_reason": str,
    "effective_task": (str, type(None)), "ambiguities": list,
}


def validate_objective(obj, profile, objective_text=None):
    errors, warnings = [], []

    for field, typ in REQUIRED_FIELDS.items():
        if field not in obj:
            errors.append(f"missing field '{field}'")
        elif not isinstance(obj[field], typ):
            errors.append(f"field '{field}' has the wrong type ({type(obj[field]).__name__})")
    if errors:
        return errors, warnings

    # The user's words are recorded verbatim, never paraphrased into the file.
    if objective_text is not None and obj["raw_text"].strip() != objective_text.strip():
        errors.append("raw_text must be the user's objective verbatim (as passed to --objective)")

    task, decision, effective = obj["task"], obj["decision"], obj["effective_task"]
    if task not in OBJECTIVE_TASKS:
        errors.append(f"task '{task}' is not one of {sorted(OBJECTIVE_TASKS)}")
    for t in obj["secondary_tasks"]:
        if t not in OBJECTIVE_TASKS:
            errors.append(f"secondary task '{t}' is not one of {sorted(OBJECTIVE_TASKS)}")
        elif t == task:
            errors.append(f"secondary task '{t}' repeats the primary task")
    if decision not in DECISIONS:
        errors.append(f"decision '{decision}' is not one of {sorted(DECISIONS)}")
    for field in ("interpretation_reason", "decision_reason", "success_criterion"):
        if is_lazy_reason(obj[field]):
            errors.append(f"'{field}' is empty or too generic; say what about this objective and dataset led here")
    if errors:
        return errors, warnings

    # effective_task: what will actually be run. Same as task for proceed, a different task for
    # proceed_adapted, None for decline.
    if decision == "proceed" and effective != task:
        errors.append("decision 'proceed' requires effective_task == task")
    if decision == "decline" and effective is not None:
        errors.append("decision 'decline' requires effective_task = null")
    if decision == "proceed_adapted" and effective not in OBJECTIVE_TASKS:
        errors.append("decision 'proceed_adapted' requires effective_task to be a valid task")

    # ── "Whether" checks against the data ──
    n = profile.get("n_samples", 0)
    if n < MIN_SAMPLES_ANY_MODEL and decision != "decline":
        errors.append(f"only {n} samples (< {MIN_SAMPLES_ANY_MODEL}); no generative model can be trained reliably, decision must be 'decline'")

    has_labels = bool(profile.get("has_labels"))
    if task in TASKS_REQUIRING_LABELS and not has_labels:
        if decision == "proceed":
            errors.append(f"task '{task}' needs class labels but this dataset has none; decision must be 'decline' or 'proceed_adapted'")
    if effective in TASKS_REQUIRING_LABELS and not has_labels:
        errors.append(f"effective_task '{effective}' needs class labels but this dataset has none")
    for t in obj["secondary_tasks"]:
        if t in TASKS_REQUIRING_LABELS and not has_labels:
            errors.append(f"secondary task '{t}' needs class labels but this dataset has none; drop it and note it in ambiguities")

    if task == "class_balancing" and has_labels:
        ratio = profile.get("imbalance_ratio")
        if ratio is not None and ratio < NEAR_BALANCED_RATIO and decision == "proceed":
            errors.append(
                f"classes are nearly balanced (max/min ratio {ratio} < {NEAR_BALANCED_RATIO}); decision must be "
                "'proceed_adapted' (test whether synthesis beats simple resampling) or 'decline'"
            )
        target = (obj.get("target") or {}).get("class")
        if target not in (None, "auto") and str(target) not in (profile.get("class_counts") or {}):
            errors.append(f"target class '{target}' is not a class in this dataset ({sorted(profile.get('class_counts', {}))})")

    if not obj["ambiguities"] and len(obj["raw_text"].split()) < 4:
        warnings.append("a very short objective usually has more than one reading; record alternatives in 'ambiguities'")
    return errors, warnings


def write_reports(errors, warnings, obj_path):
    obj_path = Path(obj_path)
    report = {"passed": not errors, "errors": errors, "warnings": warnings}
    json_path = obj_path.with_name("objective_validation.json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Objective validation", "", f"**Result:** {'PASS' if not errors else 'FAIL'}"]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    obj_path.with_name("objective_validation.md").write_text("\n".join(lines) + "\n")
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate objective.json against profile.json.")
    parser.add_argument("objective_json")
    parser.add_argument("--objective", required=True, help="The user's objective text, verbatim")
    parser.add_argument("--profile", default=None, help="Default: profile.json next to objective.json")
    args = parser.parse_args()

    obj_path = Path(args.objective_json)
    profile_path = Path(args.profile) if args.profile else obj_path.with_name("profile.json")
    try:
        obj = json.loads(obj_path.read_text())
        profile = json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: could not read inputs: {e}")

    errors, warnings = validate_objective(obj, profile, args.objective)
    out = write_reports(errors, warnings, obj_path)
    print(f"{'PASS' if not errors else 'FAIL'}: {obj_path}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"Wrote {out}")
    sys.exit(0 if not errors else 1)
