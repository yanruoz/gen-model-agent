"""Deterministic preflight check (copied from Project 1; torch + device check added), mirrors validate_plan.py's pass/fail +
   JSON+Markdown report pattern. Scoped per-activity (core vs. report) so a
   report-only gap never blocks profiling through evaluation, per
   int-brain-lab/ibl-ai-agent's install skill (docs/design.md §1 item 8).
"""
import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

import loaders

CORE_PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "anndata", "matplotlib", "torch"]
REPORT_PACKAGES = ["matplotlib.backends.backend_pdf"]


def _in_virtualenv():
    """True if this interpreter is isolated from the system/base Python (venv or virtualenv).
    Informational only (never added to `errors`): a non-isolated interpreter can still have every
    package it needs and pass cleanly. What this guards against is run.sh's one auto-remediation
    step (a pinned `pip install`) running against a bare system/base conda interpreter instead of
    this project's own venv/ -- exactly what happened once and mutated a person's system-wide
    environment. Seeing this fact up front, before any install decision, is the point."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _check_imports(module_names):
    """Each import runs in its own subprocess (this same interpreter, via sys.executable), not
    in-process. A plain try/except ImportError can't catch a crash: `import umap` segfaulting
    (observed once, from an unrelated TensorFlow build sharing the same non-isolated environment)
    kills the whole process instead of raising an exception, which would otherwise take doctor.py
    itself down and leave a stale report on disk with no indication of what actually happened.
    Isolating each import means one crashing package is reported clearly, by name, and every other
    package still gets checked."""
    results = []
    for name in module_names:
        proc = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True)
        if proc.returncode == 0:
            results.append({"module": name, "ok": True})
            continue
        if proc.returncode < 0:
            try:
                sig_desc = signal.Signals(-proc.returncode).name
            except ValueError:
                sig_desc = f"signal {-proc.returncode}"
            error = (
                f"interpreter crashed importing '{name}' ({sig_desc}, exit code {proc.returncode}); "
                "not a normal ImportError, this environment is unsafe for this import"
            )
        else:
            stderr_lines = proc.stderr.strip().splitlines()
            error = stderr_lines[-1] if stderr_lines else f"import failed with exit code {proc.returncode}"
        results.append({"module": name, "ok": False, "error": error})
    return results


def check_core(dataset=None):
    checks = [{"module": "isolated venv (not system/base Python)", "ok": _in_virtualenv()}]
    checks += _check_imports(CORE_PACKAGES)
    errors = [f"Cannot import '{c['module']}': {c['error']}" for c in checks if not c["ok"] and "error" in c]
    if not errors:
        checks.append(check_device())

    if dataset:
        folder = loaders.DATA_ROOT / dataset
        desc_path = folder / "DATA_DESCRIPTION.md"
        exists = desc_path.exists()
        checks.append({"module": f"data/{dataset}/DATA_DESCRIPTION.md", "ok": exists})
        if not exists:
            errors.append(f"Missing {desc_path}")
        else:
            # The data file itself: resolvable per the '## Loading' section (or the single
            # data file in the folder). If missing but a url is declared, say how to get it.
            spec = {}
            try:
                spec = loaders.parse_loading_spec(desc_path)
                data_file = loaders._find_data_file(folder, spec)
                checks.append({"module": f"data/{dataset}/{data_file.name}", "ok": True})
            except ValueError as e:
                hint = ""
                if "url" in spec:
                    hint = f" A url is declared: run `venv/bin/python scripts/fetch_data.py --dataset {dataset}`."
                checks.append({"module": f"data file for {dataset}", "ok": False})
                errors.append(f"{e}{hint}")

    return errors, checks


def check_device():
    """Report the best available torch device. Informational only: no GPU is not an error, training
    falls back to MPS (Apple) or CPU with a smaller time budget. Runs in a subprocess for the same
    crash-safety reason as _check_imports."""
    code = ("import torch; print('cuda' if torch.cuda.is_available() else "
            "'mps' if torch.backends.mps.is_available() else 'cpu')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    device = proc.stdout.strip() if proc.returncode == 0 else "unknown (torch unavailable)"
    return {"module": f"torch device: {device}", "ok": True, "device": device}


def check_report():
    checks = _check_imports(REPORT_PACKAGES)
    errors = [f"Cannot import '{c['module']}': {c['error']}" for c in checks if not c["ok"]]
    return errors, checks


def write_reports(errors, checks, check_name):
    report = {"check": check_name, "passed": not errors, "errors": errors, "checks": checks}
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "doctor_report.json"
    md_path = out_dir / "doctor_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"# Doctor check: {check_name}", "", f"**Result:** {'PASS' if not errors else 'FAIL'}", "", "## Checks"]
    lines += [f"- {'OK' if c['ok'] else 'FAIL'}: {c['module']}" for c in checks]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic preflight check for required dependencies and inputs.")
    parser.add_argument("--check", choices=["core", "report", "all"], default="all")
    parser.add_argument("--dataset", default=None, help="If given, also checks data/<dataset>/DATA_DESCRIPTION.md exists (core check only)")
    args = parser.parse_args()

    all_errors, all_checks = [], []
    if args.check in ("core", "all"):
        e, c = check_core(args.dataset)
        all_errors += e
        all_checks += c
    if args.check in ("report", "all"):
        e, c = check_report()
        all_errors += e
        all_checks += c

    json_path, md_path = write_reports(all_errors, all_checks, args.check)

    print(f"{'PASS' if not all_errors else 'FAIL'}: doctor --check {args.check}")
    for c in all_checks:
        print(f"  {'OK' if c['ok'] else 'FAIL'}: {c['module']}")
    for e in all_errors:
        print(f"  ERROR: {e}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    sys.exit(0 if not all_errors else 1)
