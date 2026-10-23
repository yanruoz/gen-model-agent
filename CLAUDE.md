# CLAUDE.md

Project 2 (BIOS 774): an AI agent for generative modeling. Spec: `PROJECT_REQUIREMENTS.md`.
Project 1 (dimension-reduction agent) lives at `../dim-reduction-agent`. Its handoff notes are imported below.

@docs/HANDOFF.md

## Project rules

- **Reuse Project 1 code by copying it in**, then adapting it here. Never import across folders or add `../dim-reduction-agent` to a path.
- **Single command:** the full pipeline must run with one command, given a dataset and a user objective.
- **Secrets:** API keys are read from `.env` (see `.env.example`). Never hardcode keys or commit `.env`.
- **Commit at each working milestone**, with a message that says what now works.
- **Deadline:** Mon Oct 26, 2026, 11:59 PM ET.

## How it runs

- Entry point: `./run.sh <dataset> "<objective>"`. It prepares `venv/` and the data, then runs Claude Code headless (`claude -p --permission-mode dontAsk`) with `AGENT.md` as the agent's instructions. Tools outside the allowlist in `.claude/pipeline-settings.json` are denied, never prompted. `.claude/settings.json` holds dev settings for interactive sessions only. Runs use the Claude Code login unless `--use-api-key` is passed.
- Every command goes through `venv/bin/python` / `venv/bin/pip`, never a system or conda interpreter (a bare `pip install` once modified a system-wide conda environment).
- The pipeline agent's instructions live in `AGENT.md`, which `run.sh` appends to the system prompt. Pipeline runs use `--setting-sources local`, so they never load this file or anything it imports. This file is for development sessions only. Change pipeline behavior in `AGENT.md`, the skills under `.claude/skills/`, or the scripts, not here.
- Decision logic: `.claude/skills/{objective-interpretation,data-profiling,model-selection,downstream-use,reporting}/SKILL.md`. Mechanical checks: `scripts/validate_{objective,plan,recommendation}.py`.
