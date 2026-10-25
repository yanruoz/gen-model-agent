#!/usr/bin/env bash
# Single-command entry point for the generative-modeling agent.
#
#   ./run.sh <dataset> "<objective>" [--report-number N] [--budget-min M] [--use-api-key]
#   ./run.sh <dataset> --smoke        # setup + a trivial headless call, checks no permission is denied
#
# Steps: check prerequisites -> load .env -> make venv/ if missing -> fetch the dataset's
# declared data file if missing -> run Claude Code headless with CLAUDE.md as the agent's
# instructions -> fail (nonzero exit) if the expected report PDF was not produced.
set -euo pipefail

# A full run takes up to ~40 min; on macOS, keep the machine from idle-sleeping until it ends
# (a sleep mid-run kills the headless session). Re-executes this script once under caffeinate.
if [[ -z "${GENAGENT_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export GENAGENT_CAFFEINATED=1
  exec caffeinate -is "$0" "$@"  # -i: no idle sleep; -s: no system sleep on AC power (lid must stay open)
fi

usage() {
  echo "usage: ./run.sh <dataset> \"<objective>\" [--report-number N] [--budget-min M] [--use-api-key]" >&2
  echo "       ./run.sh <dataset> --smoke" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
DATASET="$1"; shift
SMOKE=0; OBJECTIVE=""; REPORT_NUMBER=""; BUDGET_MIN=20; USE_API_KEY=0
if [[ "$1" == "--smoke" ]]; then SMOKE=1; shift; else OBJECTIVE="$1"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-number) REPORT_NUMBER="$2"; shift 2 ;;
    --budget-min)    BUDGET_MIN="$2"; shift 2 ;;
    --use-api-key)   USE_API_KEY=1; shift ;;
    *) usage ;;
  esac
done

# Always run from the repo root, so relative paths in CLAUDE.md and settings.json resolve.
cd "$(dirname "$0")"
OUT="outputs/$DATASET"
mkdir -p "$OUT"

# ── Step 1: prerequisites ──
command -v claude >/dev/null || { echo "ERROR: Claude Code ('claude') is not installed or not on PATH. See README." >&2; exit 1; }
[[ -f "data/$DATASET/DATA_DESCRIPTION.md" ]] || { echo "ERROR: data/$DATASET/DATA_DESCRIPTION.md not found." >&2; exit 1; }

# Auth: by default the run uses the Claude Code login. ANTHROPIC_API_KEY from the shell is always
# dropped; the key in .env is used only with --use-api-key (an explicit opt-in), never by default.
unset ANTHROPIC_API_KEY
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ $USE_API_KEY -eq 1 ]]; then
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "ERROR: --use-api-key given but .env has no ANTHROPIC_API_KEY" >&2; exit 1; }
  echo "[setup] auth: ANTHROPIC_API_KEY from .env (--use-api-key)"
else
  unset ANTHROPIC_API_KEY
  echo "[setup] auth: Claude Code login"
fi

# ── Step 2: isolated venv (never install into system/conda Python) ──
if [[ ! -x venv/bin/python ]]; then
  echo "[setup] creating venv/ and installing pinned requirements"
  python3 -m venv venv
  venv/bin/pip install -q -r requirements.txt
fi

# ── Step 3: data (fetch_data.py is a no-op when the file is present and its md5 matches) ──
venv/bin/python scripts/fetch_data.py --dataset "$DATASET"

# ── Step 4: headless agent run ──
if [[ $SMOKE -eq 1 ]]; then
  PROMPT="Smoke test only; do not follow the full pipeline. Run exactly these two commands, then reply with the single word DONE:
venv/bin/python scripts/doctor.py --check core --dataset $DATASET
venv/bin/python scripts/profiler.py --dataset $DATASET --out $OUT/profile.json
Then write the text 'smoke ok' to $OUT/smoke.txt using the Write tool."
else
  if [[ -z "$REPORT_NUMBER" ]]; then
    # Next free report number across reports/*/generated_report_*.pdf, starting at 1.
    REPORT_NUMBER=$(( $(ls reports/*/generated_report_*.pdf 2>/dev/null | wc -l) + 1 ))
  fi
  PROMPT="Run the full generative-modeling pipeline described in AGENT.md (it is also appended to your system prompt).
dataset: $DATASET
objective: $OBJECTIVE
report number: $REPORT_NUMBER
training budget (minutes, all models combined): $BUDGET_MIN"
fi

echo "[agent] claude -p (dataset=$DATASET${OBJECTIVE:+, objective=\"$OBJECTIVE\"})"
# Isolation: --setting-sources local loads no user or project settings and no CLAUDE.md (so neither the
# dev instructions nor anything they import reach the pipeline agent). The agent gets exactly:
#   AGENT.md (appended system prompt), .claude/pipeline-settings.json (permissions), the skill files
#   it reads by path, and the subagents in .claude/agents/ (passed explicitly with --agents).
# dontAsk: anything not allowlisted is denied, never prompted, so the run can't hang.
# --strict-mcp-config: no MCP servers. stream-json + bin/agent_stream.py: progress, transcript, result.
STATUS=0
AGENTS_JSON=$(venv/bin/python bin/agents_json.py)   # .claude/agents/*.md -> --agents (e.g. the gen-critic subagent)
claude -p "$PROMPT" \
  --append-system-prompt-file AGENT.md \
  --agents "$AGENTS_JSON" \
  --settings .claude/pipeline-settings.json \
  --setting-sources local \
  --permission-mode dontAsk \
  --strict-mcp-config \
  --output-format stream-json --verbose < /dev/null \
  | venv/bin/python bin/agent_stream.py "$OUT" || STATUS=$?

# ── Step 5: success check ──
if [[ $SMOKE -eq 1 ]]; then
  [[ -f "$OUT/profile.json" && -f "$OUT/smoke.txt" ]] || { echo "SMOKE FAIL: expected files missing" >&2; exit 1; }
  DENIED=$(venv/bin/python -c "import json;print(len(json.load(open('$OUT/agent_result.json')).get('permission_denials') or []))")
  [[ $STATUS -eq 0 && $DENIED -eq 0 ]] || { echo "SMOKE FAIL: status $STATUS, $DENIED permission denials" >&2; exit 1; }
  echo "SMOKE PASS"
  exit 0
fi
PDF="reports/$DATASET/generated_report_$REPORT_NUMBER.pdf"
[[ -f "$PDF" ]] || { echo "FAIL: $PDF was not produced" >&2; exit 1; }
[[ $STATUS -eq 0 ]] || { echo "FAIL: agent run ended with an error (see $OUT/agent_result.json)" >&2; exit 1; }
echo "DONE: $PDF"
