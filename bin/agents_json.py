"""run.sh helper (not an agent tool): turn .claude/agents/*.md (YAML-style frontmatter + prompt body)
   into the JSON that `claude --agents` expects. Pipeline runs use --setting-sources local, which does not
   load project subagents, so they are passed explicitly instead.

   Usage: venv/bin/python bin/agents_json.py   ->  prints {"<name>": {"description", "prompt", "tools"}}
"""
import json
import re
from pathlib import Path

agents = {}
for f in sorted((Path(__file__).resolve().parent.parent / ".claude" / "agents").glob("*.md")):
    text = f.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        continue
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    agents[meta["name"]] = {"description": meta.get("description", ""), "prompt": m.group(2).strip(),
                            "tools": [t.strip() for t in meta.get("tools", "").split(",") if t.strip()]}
print(json.dumps(agents))
