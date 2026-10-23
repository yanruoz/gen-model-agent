"""Execute a validated plan.json: preprocess once, then for every planned model x seed:
   train -> generate -> evaluate, then make the figures. Pattern from Project 1's run_plan.py.

   Usage: venv/bin/python scripts/run_plan.py outputs/<d>/plan.json

   - Refuses a plan that hasn't passed validate_plan.py (plan_validation.json must say passed, and be newer).
   - Reuse before recomputing: a model/seed whose checkpoint sidecar config_hash matches (same
     preprocessing, hyperparameters, seed) is not retrained. Generation and evaluation always rerun
     (they take seconds to minutes), so metrics never go stale when evaluation code changes.
   - Each step runs as its own subprocess; a failing step is retried once, then that model/seed is
     skipped and recorded. One model failing never stops the others.
   - Models with status "not_run_budget" are recorded, not run.
   - Writes outputs/<d>/run_log.json with per-model/seed status, minutes, and errors.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from model_io import config_hash, outputs
from plan_schema import artifact_key
from preprocess import prep_hash, prepare

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def _run(args, label):
    """Run one step; retry once on failure. Returns (ok, error_tail)."""
    for attempt in (1, 2):
        proc = subprocess.run([PY, str(SCRIPTS / args[0]), *args[1:]], capture_output=True, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                if line.startswith("["):
                    print("   " + line)
            return True, None
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        print(f"   {label} failed (attempt {attempt}): {tail[-1] if tail else proc.returncode}")
    return False, "\n".join(tail)


def _reusable(dataset, key, expected_hash):
    run_path, ckpt_path = outputs(dataset) / "runs" / f"{key}.json", outputs(dataset) / "checkpoints" / f"{key}.pt"
    if not (run_path.exists() and ckpt_path.exists()):
        return False
    return json.loads(run_path.read_text()).get("config_hash") == expected_hash


def run_plan(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    dataset = plan["dataset"]
    val_path = plan_path.with_name("plan_validation.json")
    if not val_path.exists() or not json.loads(val_path.read_text()).get("passed") \
            or val_path.stat().st_mtime < plan_path.stat().st_mtime:
        sys.exit("ERROR: plan.json has not passed validate_plan.py since it was last edited; validate it first.")

    log = {"dataset": dataset, "revision": plan.get("revision"), "entries": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_start = time.time()
    if not plan["models"]:
        log["note"] = "objective declined; no models trained"
        (outputs(dataset) / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")
        print("[run_plan] objective declined: nothing to train")
        return log

    print("[run_plan] preprocessing")
    prepare(plan)
    ph = prep_hash(plan)

    jobs = [(m, s) for m in plan["models"] for s in m["seeds"]]
    done_keys = []
    for i, (m, seed) in enumerate(jobs, 1):
        key = artifact_key(m["id"], seed)
        entry = {"key": key, "model_id": m["id"], "family": m["family"], "seed": seed}
        if m.get("status", "planned") != "planned":
            entry.update(status="not_run_budget", reason=m.get("reason"))
            print(f"[{i}/{len(jobs)}] {key}: not run (budget)")
            log["entries"].append(entry)
            continue
        h = config_hash(m, seed, ph)
        t0 = time.time()
        steps = [(["generate.py", "--dataset", dataset, "--key", key], "generate"),
                 (["evaluate_gen.py", "--dataset", dataset, "--key", key], "evaluate")]
        if _reusable(dataset, key, h):
            entry["trained"] = "reused checkpoint (matching config hash)"
            print(f"[{i}/{len(jobs)}] {key}: reusing trained checkpoint ({h}); regenerating and re-evaluating")
        else:
            entry["trained"] = "trained this run"
            print(f"[{i}/{len(jobs)}] {key}: training {m['family']}...")
            steps.insert(0, (["train.py", "--dataset", dataset, "--model-id", m["id"], "--seed", str(seed)], "train"))
        for args, label in steps:
            ok, err = _run(args, label)
            if not ok:
                entry.update(status="failed", failed_step=label, error=err)
                break
        else:
            entry.update(status="ok", config_hash=h)
            done_keys.append(key)
        entry["minutes"] = round((time.time() - t0) / 60, 2)
        log["entries"].append(entry)

    if done_keys:
        print("[run_plan] figures")
        ok, err = _run(["visualize.py", "--dataset", dataset, "--keys", ",".join(done_keys)], "visualize")
        log["figures"] = "ok" if ok else f"failed: {err}"
    log["total_minutes"] = round((time.time() - t_start) / 60, 2)
    log["budget_min"] = plan.get("budget_min")
    (outputs(dataset) / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")
    n_ok = sum(e["status"] == "ok" for e in log["entries"])
    print(f"[run_plan] {n_ok}/{len(jobs)} model/seed runs available; {log['total_minutes']} min; wrote run_log.json")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a validated plan.json.")
    parser.add_argument("plan_json")
    args = parser.parse_args()
    run_plan(args.plan_json)
