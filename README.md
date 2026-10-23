# gen-model-agent

BIOS 774 Project 2: an AI agent that, given a dataset and a user objective, decides whether and how to apply generative modeling (VAE, GAN, normalizing flow, diffusion), trains and evaluates the chosen model(s), recommends downstream use, and writes a report.

Status: setup only. Usage, architecture, and results will be documented here once the pipeline runs end to end.

## Quick start (setup check)

Requires [Claude Code](https://claude.com/claude-code) installed and logged in, and Python 3.10+.

```bash
./run.sh pbmc --smoke   # creates venv/, fetches data, runs one headless Claude Code call
```

## Authentication

Pipeline runs use your **Claude Code login** by default. `run.sh` always drops any `ANTHROPIC_API_KEY` inherited from your shell, and it ignores the key in `.env` unless you opt in:

```bash
cp .env.example .env    # then set ANTHROPIC_API_KEY in .env
./run.sh pbmc "<objective>" --use-api-key
```

With `--use-api-key`, usage is billed to that API key instead of your Claude Code plan.

## Permissions

There are two settings files:

- `.claude/settings.json`: dev settings for interactive Claude Code sessions in this repo. Allows web search/fetch and `git add/commit/status/diff`. Denies `git push` (including force-push) and reading `.env`.
- `.claude/pipeline-settings.json`: passed to `claude -p` by `run.sh`. The agent can run only `venv/bin/python scripts/*`, read the repo, write inside `outputs/`, and call the critic subagent. It can't write to `data/` or `reports/` (only `scripts/report.py` writes reports), and it can't run git, use the web, install packages, or read `.env`. Anything not allowed is denied without a prompt.

Claude Code also loads the dev settings during pipeline runs, because skipping them would skip `CLAUDE.md` too. Deny rules take precedence, so `pipeline-settings.json` denies everything the dev file allows. `tests/test_settings.py` checks this.
