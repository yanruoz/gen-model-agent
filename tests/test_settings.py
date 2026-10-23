"""Pipeline runs load both .claude/settings.json (dev) and .claude/pipeline-settings.json, because
skipping the project settings source would also skip CLAUDE.md. Deny beats allow across sources, so
every dev allow rule must be covered by a pipeline deny rule, or it would leak into pipeline runs.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import json
import re
import unittest
from fnmatch import fnmatchcase
from pathlib import Path

CLAUDE_DIR = Path(__file__).resolve().parent.parent / ".claude"


def _split(rule):
    """'Bash(git add *)' -> ('Bash', 'git add *'); 'WebSearch' -> ('WebSearch', None)."""
    m = re.fullmatch(r"(\w+)(?:\((.*)\))?", rule)
    return m.group(1), m.group(2)


def _covered(allow_rule, deny_rules):
    tool, spec = _split(allow_rule)
    for deny in deny_rules:
        d_tool, d_spec = _split(deny)
        if d_tool != tool:
            continue
        if d_spec is None or (spec is not None and fnmatchcase(spec, d_spec)):
            return True
    return False


class TestPipelineIsolation(unittest.TestCase):
    def setUp(self):
        self.dev = json.loads((CLAUDE_DIR / "settings.json").read_text())["permissions"]
        self.pipe = json.loads((CLAUDE_DIR / "pipeline-settings.json").read_text())["permissions"]

    def test_every_dev_allow_is_denied_in_pipeline(self):
        leaks = [r for r in self.dev.get("allow", []) if not _covered(r, self.pipe["deny"])]
        self.assertEqual(leaks, [], f"dev allow rules not denied by pipeline-settings.json: {leaks}")

    def test_agent_cannot_write_reports_or_data_directly(self):
        # reports/ is written only by scripts/report.py (run through Bash), never by Claude's own tools
        for rule in ("Edit(./reports/**)", "Write(./reports/**)", "Edit(./data/**)", "Write(./data/**)"):
            self.assertIn(rule, self.pipe["deny"])

    def test_env_is_unreadable_in_both(self):
        self.assertIn("Read(./.env)", self.dev["deny"])
        self.assertIn("Read(./.env)", self.pipe["deny"])

    def test_dev_denies_push(self):
        self.assertTrue(any(_split(r) == ("Bash", "git push *") for r in self.dev["deny"]))


if __name__ == "__main__":
    unittest.main()
