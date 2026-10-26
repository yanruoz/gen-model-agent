"""Writes the settings file a pipeline run actually uses: .claude/pipeline-settings.json with every
'./' path rule also given as an absolute '//<repo>/' rule.

Why: './' rules resolve against the shell's current directory, which persists between Bash calls.
One `cd outputs/<d> && ls` made Write(./outputs/**) point at outputs/<d>/outputs/** and every
later write was denied (seen in a rehearsal run). Absolute rules don't move.

Usage: venv/bin/python bin/runtime_settings.py <out_path>   (run.sh puts it in a temp dir, outside
outputs/, so the agent can't edit its own permissions)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def absolutize(rules, repo=REPO):
    """['Write(./outputs/**)'] -> ['Write(./outputs/**)', 'Write(//<repo>/outputs/**)']; others unchanged."""
    out = []
    for r in rules:
        out.append(r)
        tool, _, rest = r.partition("(")
        if tool in ("Read", "Edit", "Write") and rest.startswith("./"):
            out.append(f"{tool}(/{repo.as_posix()}/{rest[2:]}")  # '//' prefix = absolute path
    return out


def build(repo=REPO):
    s = json.loads((repo / ".claude" / "pipeline-settings.json").read_text())
    p = s["permissions"]
    p["allow"], p["deny"] = absolutize(p["allow"], repo), absolutize(p["deny"], repo)
    return s


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(build(), indent=2) + "\n")
