"""run.sh helper (not an agent tool): reads `claude -p --output-format stream-json` from stdin,
   saves every event to a transcript, prints one progress line per tool call, and writes the final
   result event (with permission denials, cost, turns) to agent_result.json.

   Usage: claude -p ... --output-format stream-json --verbose | python bin/agent_stream.py <out_dir>
   Exit code: 0 if the run finished without error, 1 otherwise.
"""
import json
import sys
import time
from pathlib import Path

out_dir = Path(sys.argv[1])
transcript = open(out_dir / "agent_transcript.jsonl", "w")
result, t0 = None, time.time()


def describe(name, inp):
    if name == "Bash":
        return inp.get("description") or inp.get("command", "")[:100]
    if name in ("Read", "Write", "Edit"):
        return inp.get("file_path", "").split("/outputs/")[-1].split("/gen-model-agent/")[-1]
    if name == "Agent":
        return f"subagent {inp.get('subagent_type', '')}: {inp.get('description', '')}"
    return json.dumps(inp)[:100]


for line in sys.stdin:
    transcript.write(line)
    transcript.flush()
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    if ev.get("type") == "assistant":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                mins = (time.time() - t0) / 60
                print(f"  [{mins:5.1f} min] {block['name']}: {describe(block['name'], block.get('input', {}))}", flush=True)
    elif ev.get("type") == "result":
        result = ev

if result is None:
    print("[agent] no result event received (claude exited early?)")
    sys.exit(1)
(out_dir / "agent_result.json").write_text(json.dumps(result, indent=2) + "\n")
print("[agent] result:", (result.get("result") or "").strip()[:1500])
denied = result.get("permission_denials") or []
print(f"[agent] turns {result.get('num_turns')}, {result.get('duration_ms', 0) / 60000:.1f} min, permission denials: {len(denied)}")
for d in denied:
    print("   -", d.get("tool_name"), json.dumps(d.get("tool_input"))[:200])
sys.exit(1 if result.get("is_error") else 0)
