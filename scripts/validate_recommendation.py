"""Validates outputs/<dataset>/recommendation.json, the agent's downstream-use recommendations.

   Checks, all mechanical:
   - every cited metric exists in metrics/<key>.json and the quoted value matches it;
   - a "recommended" or "conditional" use cites at least one metric (no unsupported recommendations);
   - the primary task gets a verdict;
   - privacy: no wording that claims or implies privacy (e.g. "anonymized", "safe to share"), and when
     sharing/releasing/privacy comes up anywhere, a statement that there is no formal privacy guarantee.

   Usage: venv/bin/python scripts/validate_recommendation.py outputs/<d>/recommendation.json
   Writes recommendation_validation.json/.md next to it. Exit 0 = pass.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from plan_schema import is_lazy_reason

VERDICTS = {"recommended", "conditional", "not_recommended"}

# Wording that claims or implies a privacy property the pipeline cannot establish.
FORBIDDEN_PRIVACY = [
    r"privacy[- ]preserving", r"\banonymi[sz]ed\b", r"\bde-?identified\b", r"safe to (share|release|publish)",
    r"(guarantees?|ensures?|protects?|preserves?)\s+(patient |donor |individual )?privacy", r"\bprivacy[- ]safe\b",
    r"cannot be (re-?identified|traced)", r"no (risk|chance) of (re-?identification|leak)",
]
SHARING_WORDS = re.compile(r"\b(shar(e|ed|ing)|releas(e|ed|ing)|publish(ed|ing)?|privacy|private|anonym\w*|public)\b", re.I)
REQUIRED_DISCLAIMER = re.compile(r"\b(no|not an?|not any)\s+formal\s+privacy\s+guarantee", re.I)


def _resolve(metrics_dir, path):
    """'vae_nb_s0.fidelity.c2st_auc' -> value in metrics/vae_nb_s0.json, or raises KeyError."""
    key, *fields = path.split(".")
    f = metrics_dir / f"{key}.json"
    if not f.exists():
        raise KeyError(f"no metrics file for '{key}'")
    v = json.loads(f.read_text())
    for field in fields:
        if not isinstance(v, dict) or field not in v:
            raise KeyError(f"'{path}' not found in metrics/{key}.json")
        v = v[field]
    return v


def _matches(quoted, actual):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    if not isinstance(quoted, (int, float)) or isinstance(quoted, bool):
        return False
    decimals = len(repr(float(quoted)).split(".")[1].rstrip("0")) if "." in repr(float(quoted)) else 0
    return round(float(actual), decimals) == round(float(quoted), decimals) or abs(float(actual) - float(quoted)) < 1e-9


def validate_recommendation(rec, objective, metrics_dir):
    errors, warnings = [], []
    uses = rec.get("uses")
    if not isinstance(uses, list) or not uses:
        return ["'uses' must be a non-empty list"], warnings

    task = objective.get("effective_task")
    if task and not any(u.get("task") == task for u in uses):
        errors.append(f"no use addresses the primary task '{task}'")

    for i, u in enumerate(uses):
        label = f"uses[{i}] ({u.get('use', '?')[:40]})"
        if u.get("verdict") not in VERDICTS:
            errors.append(f"{label}: verdict must be one of {sorted(VERDICTS)}")
        for field in ("use", "rationale"):
            if is_lazy_reason(u.get(field)):
                errors.append(f"{label}: '{field}' is missing or too generic")
        evidence = u.get("evidence") or []
        if u.get("verdict") in ("recommended", "conditional") and not evidence:
            errors.append(f"{label}: a '{u.get('verdict')}' use must cite at least one metric as evidence")
        for ev in evidence:
            path = ev.get("metric", "")
            try:
                actual = _resolve(metrics_dir, path)
            except KeyError as e:
                errors.append(f"{label}: {e}")
                continue
            if "value" in ev and not _matches(ev["value"], actual):
                errors.append(f"{label}: quotes {path} = {ev['value']}, but metrics say {actual}")
        if not isinstance(u.get("caveats", []), list):
            errors.append(f"{label}: 'caveats' must be a list")

    # ── Privacy wording ──
    text = json.dumps(rec)
    for pat in FORBIDDEN_PRIVACY:
        m = re.search(pat, text, re.I)
        if m:
            errors.append(f"privacy claim not supported by any analysis here: '{m.group(0)}'. "
                          "The memorization check is evidence about copying, not a privacy guarantee.")
    mentions_sharing = bool(SHARING_WORDS.search(objective.get("raw_text", ""))) or \
        any(SHARING_WORDS.search(json.dumps({k: u.get(k) for k in ("use", "rationale")})) for u in uses)
    if mentions_sharing:
        stmt = rec.get("privacy_statement") or ""
        if not REQUIRED_DISCLAIMER.search(stmt):
            errors.append("sharing/privacy comes up, so 'privacy_statement' must say plainly that there is no "
                          "formal privacy guarantee (e.g. 'This is not a formal privacy guarantee such as differential privacy.')")
    return errors, warnings


def write_reports(errors, warnings, path):
    path = Path(path)
    report = {"passed": not errors, "errors": errors, "warnings": warnings}
    out = path.with_name("recommendation_validation.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Recommendation validation", "", f"**Result:** {'PASS' if not errors else 'FAIL'}"]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    path.with_name("recommendation_validation.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate recommendation.json against metrics and objective.")
    parser.add_argument("recommendation_json")
    args = parser.parse_args()
    path = Path(args.recommendation_json)
    try:
        rec = json.loads(path.read_text())
        objective = json.loads(path.with_name("objective.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: could not read inputs: {e}")
    errors, warnings = validate_recommendation(rec, objective, path.with_name("metrics"))
    out = write_reports(errors, warnings, path)
    print(f"{'PASS' if not errors else 'FAIL'}: {path}")
    for e in errors:
        print(f"  ERROR: {e}")
    print(f"Wrote {out}")
    sys.exit(0 if not errors else 1)
