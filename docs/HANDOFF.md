# Handoff: Project 1 (dimension-reduction agent) → Project 2 (generative-modeling agent)

Written for a fresh agent starting Project 2 (STAI-X-style "AI agent for generative modeling": VAEs/GANs/flows/diffusion, PBMC + PathMNIST + maybe more, single-command pipeline, `generated_report_1/2`, 4-page `report.pdf`). Project 2's spec is structurally almost identical to Project 1's — same deliverable shape, same two backup datasets, same "agent makes substantive decisions, not a fixed pipeline" grading emphasis. This doc tells you what from Project 1 to reuse, adapt, or rebuild, and why. Everything below is read directly from the repo (code, git log, README, design.md, todo.md, skills, `.claude/agents/dr-critic.md`, `docs/manual_report_draft.md`) unless marked "inferred."

---

## 1. Overview

Project 1 is a Claude-Code-harness-only agent that, given one instruction naming a dataset folder (e.g. "do the data analysis on data/pbmc"), autonomously profiles the data, picks and justifies a preprocessing + dimension-reduction plan from an approved method list (PCA family, MDS/Isomap/LLE/Laplacian Eigenmaps/Diffusion Maps, t-SNE/UMAP), runs it through deterministic Python scripts, critiques its own plan once via a subagent, and produces a PDF report. There is **no LLM API call anywhere in the running pipeline** — Claude Code itself (reading `CLAUDE.md` + skill files) is the only "model," and every number/figure comes from NumPy/SciPy/scikit-learn/UMAP code. It was built and graded on two required datasets (PBMC3k scRNA-seq, PathMNIST images) plus a synthetic CSV used only for generic-loader testing.

## 2. Architecture

```
prompt: "do the data analysis on data/<name>"
   │
   ▼
CLAUDE.md (rules, 9-step execution sequence) ──reads──▶ data/<name>/DATA_DESCRIPTION.md
   │
   ├─ 1 scripts/doctor.py --check core         preflight: packages import, data file present
   ├─ 2 scripts/profiler.py                    → outputs/<name>/profile.json + profile_summary.md
   ├─ 3 skills (read, not executed):
   │      .claude/skills/data-profiling/SKILL.md     profile → preprocessing steps
   │      .claude/skills/method-selection/SKILL.md   profile → methods/hyperparams, plan.json schema
   ├─ 4 agent writes outputs/<name>/plan.json   every step/method has a dataset-specific "reason"
   ├─ 5 scripts/validate_plan.py                must pass before anything runs
   ├─ 6 scripts/run_plan.py                     reduce_dim → evaluate → (cluster) → visualize per method
   │                                            → embeddings/, metrics/, figures/, run_log.json
   ├─ 7 Agent(subagent_type="dr-critic")        fresh-context critic, plan+metrics+figures only
   │      → outputs/<name>/critique.json; agent makes exactly one plan revision in response
   ├─ 8 agent writes outputs/<name>/findings.md; scripts/report.py → reports/<name>/generated_report_<N>.pdf
   └─ 9 self-check checklist (end of CLAUDE.md)
```

**Control flow**: Claude Code itself is the orchestrator — there is no `main.py`/agent loop in code. `CLAUDE.md` is a numbered imperative script the model follows turn by turn; the "skills" (`.claude/skills/*/SKILL.md`) are loaded just-in-time (step 3) and contain the actual decision logic/schema, not `CLAUDE.md` itself. All computation is pushed into small deterministic CLI scripts under `scripts/`, each independently runnable and resumable via the JSON files it reads/writes — this is the single most important architectural pattern and is explicitly justified in `docs/design.md` §1 (adapted from reference repos "Sentinelle"/"biostat-superpowers"/"ibl-ai-agent").

**Exact trigger command**: there is no CLI entrypoint script. A run is triggered by starting Claude Code in the repo root and giving it a single natural-language instruction: `do the data analysis on data/pbmc`. `CLAUDE.md`'s "How you're invoked" section defines what happens if no dataset is named (stop, don't guess) or if a report number isn't given (count existing `reports/*/generated_report_*.pdf` across the repo, use next integer). For manual/debugging use, every step is also callable by hand (see README "Running the scripts by hand" and the Tool contracts table in `CLAUDE.md`).

**Key file/folder roles**:
- `CLAUDE.md` (repo root) — the only orchestration "brain"; rules, sequence, guard rails, fallback rules, output requirements, final checklist.
- `.claude/skills/{data-profiling,method-selection,reporting}/SKILL.md` — decision logic + `plan.json` schema, loaded just-in-time.
- `.claude/agents/dr-critic.md` — adversarial critic subagent definition (prompt only, no code; `tools: Read, Write`).
- `scripts/*.py` — deterministic tools; see §9 for the ones worth reusing.
- `data/<name>/DATA_DESCRIPTION.md` — hand-authored, per-dataset source of truth; never written by the pipeline.
- `outputs/<name>/` — all intermediate state (profile, plan, embeddings, metrics, figures, findings, run_log, critique) — regenerable, gitignored for large `.npy` files.
- `reports/<name>/generated_report_<N>.pdf` — final deliverable, built by `scripts/report.py` (matplotlib `PdfPages`, no external PDF lib dependency).

## 3. Folder/file structure

```
dim-reduction-agent/
├── CLAUDE.md                        orchestration brain: rules, 9-step sequence, guard rails, fallback
│                                     rules, tool contracts, output requirements, self-check checklist
├── README.md                        architecture, decision-making, setup, running, outputs, limitations
├── requirements.txt                 pinned deps, installed only into venv/ (never system/conda python)
├── report.pdf                       UNTRACKED, loose copy of the manual 4-page report at repo root
├── venv/                            project's own isolated Python env; created once, never bare system/conda
│
├── .claude/
│   ├── skills/                      decision logic, loaded just-in-time (not preloaded by CLAUDE.md)
│   │   ├── data-profiling/SKILL.md       profile.json + DATA_DESCRIPTION.md → preprocessing steps
│   │   ├── method-selection/SKILL.md     → methods/hyperparameters, defines plan.json's full schema
│   │   └── reporting/SKILL.md            → findings.md rules, what report.py generates on its own
│   └── agents/
│       └── dr-critic.md             adversarial critic subagent prompt (no code); tools: Read, Write
│
├── scripts/                         deterministic CLI tools; every one is `--help`-documented
│   ├── doctor.py                         preflight: package imports (subprocess-isolated), data file present
│   ├── fetch_data.py                     downloads one dataset's declared, checksummed url (only network access)
│   ├── loaders.py                        generic loader: csv/tsv/txt/npy/npz/h5ad, driven by ## Loading spec
│   ├── profiler.py                       → profile.json + profile_summary.md
│   ├── plan_schema.py                    shared constants: ALLOWED_METHODS, METHOD_ROLES, preprocessing steps
│   ├── validate_plan.py                  schema/guard-rail checks on plan.json, pass/fail JSON+Markdown report
│   ├── reduce_dim.py                     one DR method per call; preprocessing registry + method registry
│   ├── evaluate.py                       per-method metrics (trustworthiness, PCA variance, MDS stress, ...)
│   ├── cluster.py                        k-means + density cross-check, for coloring unlabeled plots only
│   ├── visualize.py                      2D scatter / scree plots → figures/*.png
│   ├── run_plan.py                       orchestrates reduce_dim → evaluate → cluster → visualize per plan
│   ├── report.py                         assembles reports/<name>/generated_report_<N>.pdf
│   └── test_api.py                       DEAD CODE: direct anthropic.Anthropic() smoke test, never deleted
│
├── tests/                           ~160 unit tests, one file per script/concern; run via
│                                     `venv/bin/python -m unittest discover -s tests`
│
├── data/<name>/                     one folder per dataset; data files themselves are git-ignored
│   ├── pbmc/
│   │   ├── DATA_DESCRIPTION.md           hand-authored, the one source of truth; pipeline never writes this
│   │   └── pbmc3k_raw.h5ad                (fetched via fetch_data.py; git-ignored)
│   └── pathmnist/
│       ├── DATA_DESCRIPTION.md
│       └── pathmnist.npz                  (fetched via fetch_data.py; git-ignored)
│
├── outputs/<name>/                  INTERMEDIATE/working artifacts, regenerable, not itself a deliverable
│   ├── profile.json, profile_summary.md       from profiler.py
│   ├── plan.json, plan_validation.*           the agent's plan + validate_plan.py's result
│   ├── embeddings/<method>.npy + .json        embedding + actual hyperparams/seed/runtime (.npy git-ignored)
│   ├── metrics/<method>.json                  trustworthiness, diagnostics, label silhouette
│   ├── metrics/clustering.json                only if plan has a clustering block
│   ├── figures/<method>.png, *_scree.png      scatter plots + PCA-family scree plots
│   ├── findings.md                            the one part of the report the agent writes by hand
│   ├── critique.json                          dr-critic's structured verdict (only if critique ran)
│   └── run_log.json                           per-method status: ok / reused / fell back / skipped
│
├── reports/<name>/
│   └── generated_report_<N>.pdf     FINAL deliverable only; nothing else lives in this folder
│
├── docs/
│   ├── design.md                         research-and-design doc written before implementation started
│   ├── todo.md                           running status/priority list, kept updated across the build
│   ├── report.tex / report.pdf           LaTeX source + build of the manual 4-page report (built with tectonic)
│   └── manual_report_draft.md            UNTRACKED draft of that report's prose, with checked numbers
│
├── resources/                       course-provided materials (assignment text, reference links); the
│                                     actual PDFs/docx are gitignored ("not ours to redistribute publicly")
│
├── notebooks/                       empty by design; CLAUDE.md forbids executing notebooks, agent never writes here
│
├── dim-reduction-agent-source.zip   UNTRACKED — a prior Canvas-submission snapshot, not pipeline code
└── dim-reduction-agent-source/      UNTRACKED — that zip's extracted contents, same caveat
```

**Two-folder split, `outputs/` vs. `reports/` (deliberate, stated in `docs/design.md` §3.1)**: `outputs/<name>/` is the pipeline's own scratch/working directory — every stage reads/writes it, and it's what makes each stage independently resumable — but it is not a deliverable itself. `report.py` reads everything out of `outputs/<name>/` and renders one self-contained PDF, figures embedded inline, directly into `reports/<name>/`; nothing is copied to the repo root, and the 1-vs-2 report-number assignment is a CLI argument (`--report-number`), never hardcoded per dataset.

**Untracked items currently sitting in the working tree** (visible in `git status`, not part of the designed layout above): `report.pdf` and `docs/manual_report_draft.md` at/near the root, plus `dim-reduction-agent-source.zip` and its extracted `dim-reduction-agent-source/` folder — all leftovers from preparing a Canvas submission, not pipeline files. If Project 2 reuses this repo, don't mistake them for structure to carry forward.

## 4. Decision logic: rule-based, not LLM-API-driven

This is the single biggest design choice to understand before starting Project 2, and it was a **deliberate, explicit constraint carried into the project** (`docs/design.md` §2: "Harness-only... no LangChain/API planner"), not a default. Concretely:

- There is **no** call to `anthropic.Anthropic()` or any other LLM API inside the executed pipeline. `scripts/test_api.py` is the one place such a call exists, and `docs/design.md` (§0) explicitly says it should be **deleted** once the harness approach was confirmed working — it never was deleted (dead code, still present; see §8 Lessons learned).
- All "intelligence" is Claude Code itself reading `CLAUDE.md` + the skill files and making choices a human would otherwise make by hand: which preprocessing steps, which methods, which hyperparameters, how to phrase the written findings. These choices are **constrained by fixed rule-based decision trees** written in the skills (e.g. `data-profiling/SKILL.md`'s numbered branches keyed on `profile.json` properties like sparsity/value-range/missingness; `method-selection/SKILL.md`'s method-role table and size thresholds), so the LLM's job is "apply this decision tree and justify the result in writing," not "decide from scratch."
- All numeric computation (the embeddings, metrics, clustering) is deterministic scikit-learn/NumPy/SciPy code with fixed seeds — the LLM never computes a number itself, only reads results and writes about them. This separation ("the model's job is to decide and to write, not to compute," README) is enforced structurally: `reduce_dim.py`/`evaluate.py`/`cluster.py` are plain scripts invoked by filename + JSON params, and the report can only quote numbers that exist in `metrics/*.json` (checked by the `reporting` skill's final-check step and the CLAUDE.md self-check checklist).
- A second layer of "judgment" is a subagent (`dr-critic`), dispatched with a **narrowed context** (only `plan.json` + `metrics/*.json` + figure paths, explicitly not the builder's own reasoning or `findings.md`) to adversarially re-check five fixed categories (manifold overread risk, suspicious metrics, preprocessing mismatch, unjustified hyperparameters, label leakage into reasoning). This pattern — fresh-context critic, narrow inputs, structured JSON output, exactly one revision loop — is directly reusable for Project 2's generative models (e.g. a critic that checks for mode collapse claims without cross-checking, FID/metric sanity, leakage of labels into an unsupervised generative objective).

**For Project 2**: the spec explicitly allows (but does not require) LLM APIs ("OpenAI APIs, Anthropic APIs, ... are all acceptable"). Given Project 1's approach worked and was graded on "quality of modeling decisions," the strong default is to **keep the harness-only pattern** (Claude Code + CLAUDE.md + skills + deterministic scripts) rather than introducing a separate API-based planner — it's the same project shape (inspect data → decide → run deterministic code → evaluate → report), just swapping "dimension-reduction methods" for "generative models" and adding "generate synthetic samples" + "recommend downstream use" as new decision points. This is a judgment call to flag to the user, not something to silently assume; see §10.

## 5. Tools, libraries, and external services

**Pinned dependencies** (`requirements.txt`, installed only into the project's own `venv/`, never system Python):
```
numpy==2.5.3
pandas==3.0.5
scipy==1.18.1
scikit-learn==1.9.1
anndata==0.13.3.post0
umap-learn==0.5.12
matplotlib==3.11.1
```
`docs/design.md` §0 also records a slightly different, now-superseded version snapshot (pandas 3.0.5 same, scanpy 1.12.4, medmnist 3.0.2 — these two are **not** in the pinned `requirements.txt`, meaning `scanpy`/`medmnist` were used only to originally fetch/inspect data, not as pipeline dependencies; the pipeline's own `loaders.py`/`fetch_data.py` replaced that need). README confirms versions are pinned specifically because "t-SNE and UMAP layouts are somewhat version-sensitive."

**No LLM API is called by the running pipeline.** `test_api.py` (dead code, see §8) is the only file that imports `anthropic.Anthropic()`, reading `ANTHROPIC_API_KEY` from the environment — if Project 2 does want a direct API call pattern (vs. pure harness), this file is the one place in Project 1 that shows how the key was expected to be configured (plain env var, never hardcoded). No other external service, API key, or secret appears anywhere in the repo.

**Network access** is isolated to one explicit, narrow path: `scripts/fetch_data.py --dataset <name>` downloads exactly the `url:`/checks the `md5:` declared in that dataset's own `DATA_DESCRIPTION.md`; nothing else in the pipeline touches the network. A real bug was found and fixed here (commit `60ab76a`): `exampledata.scverse.org` returns HTTP 403 for urllib's default User-Agent, fixed by sending an explicit one.

## 6. Environment and setup

- **Python**: "Requires Python 3.10 or newer (developed on 3.13)" (README). `__pycache__` artifacts confirm `cpython-313`.
- **No GPU usage anywhere** — everything is CPU scikit-learn/NumPy/SciPy/UMAP. This is a major point of divergence for Project 2: VAEs/GANs/diffusion models typically want PyTorch/TensorFlow and will often need GPU for anything beyond tiny models/datasets — **nothing in Project 1 tells you how to set up a GPU environment**, that's new groundwork for Project 2.
- **Environment isolation is a hard, explicitly-enforced rule**, not a suggestion — it caused a real incident worth internalizing: a bare `pip install -r requirements.txt` once ran outside `venv/` and installed into the user's system-wide conda environment, upgrading numpy/pandas/scikit-learn there and breaking unrelated tools, and made `import umap` crash from a stale TensorFlow build that had no reason to be anywhere near the project (see `CLAUDE.md`'s "Required execution sequence" preamble and commit `e3f8dc4` "Fix root cause of a live incident: pin every command to venv/, harden doctor.py against a segfault"). **Every single command, including the ones the agent runs itself, must go through `venv/bin/python`/`venv/bin/pip`, never a bare or system/conda interpreter.** This rule should be carried into Project 2's CLAUDE.md verbatim — it's even more important there since PyTorch/TF installs are heavier and more version-sensitive than this project's dependencies.
- **Self-healing setup**: `scripts/doctor.py --check {core,report,all}` is the one deterministic preflight tool. On failure, the agent gets exactly **one bounded fix attempt** (install pinned requirements, or fetch the one declared data file), then must stop and document rather than retry indefinitely or improvise a workaround (CLAUDE.md rules 6, non-negotiable rule set; added commit `065df1f`). `doctor.py` runs each package import in its own subprocess specifically so a crashing import (not just an absent one) is caught and reported rather than segfaulting the whole check (commit `e3f8dc4`).
- **Setup quirk worth knowing**: the README documents a "clean-numbering reset" — `rm -rf outputs/pbmc outputs/pathmnist reports/pbmc reports/pathmnist` — needed to force report numbering back to 1/2 and to force the agent to redo the full pipeline instead of reusing cached `outputs/` artifacts (commit `def0299`).

## 7. Datasets

Two required datasets, each with a hand-authored `DATA_DESCRIPTION.md` (never machine-generated) as the one source of truth for structure/modality/labels — this file is read by both the human and the agent, and is the **only** place preprocessing-relevant facts about a dataset live outside `profile.json`.

- **PBMC3k** (`data/pbmc/`): Scanpy's raw `pbmc3k_raw.h5ad`, 2,700 cells × 32,738 genes, raw UMI counts (non-negative integers), ~97.4% sparse, **no labels** (fully unsupervised). Loaded via `x_key`/`y_key`-style `## Loading` spec in its `DATA_DESCRIPTION.md` (`file: pbmc3k_raw.h5ad`, declared `url`/`md5` for `fetch_data.py`).
- **PathMNIST** (`data/pathmnist/`): MedMNIST v2, 89,996 flattened 28×28×3 (2,352-feature) histology patches, pixel intensities 0–255, 9 tissue-class labels (mild imbalance, 7,886–12,885 per class), **labels present but evaluation/coloring-only, never a feature**. Loaded from `pathmnist.npz` via declared `x_key: train_images`/`y_key: train_labels`.

**Loading mechanism (directly reusable for Project 2, see §9)**: one *generic* loader (`scripts/loaders.py:load_dataset`/`load_generic`) handles csv/tsv/txt/npy/npz/h5ad uniformly, driven only by an optional `## Loading` section (`file`, `label_column`, `id_column`, `delimiter`, `x_key`, `y_key`, `url`, `md5`) in that dataset's own `DATA_DESCRIPTION.md`. There is **no per-dataset code anywhere** — this was treated as a design invariant and tested by running an unregistered synthetic CSV through the pipeline unmodified (`docs/manual_report_draft.md` §1). Key constraint: **numeric features only** — a dataset with non-numeric columns is refused with an error naming them (not encoded, not dropped silently); this is a known, stated limitation, not a bug, and Project 2's data-inspection step should probably keep or explicitly revisit this constraint, since generative models for tabular data often *do* need categorical handling.

**Gotchas carried over from real runs** (would resurface in Project 2 if reusing the loader/profiler as-is):
- Missing-value handling exists but is single-value imputation only (median/mean/constant); no MAR/MNAR check.
- Dense-only pipeline: sparse inputs are densified; a hard size refusal exists above ~4 GB dense.
- No QC/row filtering (cells or samples) — only column/feature-level steps are supported structurally.
- `fetch_data.py` needs an explicit User-Agent header for at least one of the two dataset hosts (scverse); Zenodo (PathMNIST's host) was fine as-is.

## 8. Results

(From `docs/manual_report_draft.md` §4, which is the draft of the hand-written 4-page `report.pdf` — the authoritative, checked-against-`metrics/*.json` numbers; **note**: `outputs/pbmc/` and `outputs/pathmnist/` in the current working tree only contain a few stray `.npy` embeddings, not the full `profile.json`/`plan.json`/`metrics/` set — those were cleared/regenerated across multiple live runs and the final numbers only persist in the generated PDFs and this draft text, not as live JSON in the repo right now.)

- **PBMC3k**: preprocessed with `normalize_total` → `log1p` → `select_top_variance` (2,000 genes). PCA (50 comp.) trustworthiness 0.9344 (29.04% variance, >200 comps needed for 90%). Kernel PCA (explicit `gamma=0.0017`, not sklearn's default) 0.8824. Diffusion Maps (10 comp.) 0.8466. UMAP (15 neighbors) 0.7740. t-SNE (perplexity 30) 0.8023. Data-derived clustering (k=4 via automatic density cross-check, not plain silhouette) found a 1,330/685/348/337-cell split; every embedding showed three well-separated regions plus one gradient-like region plus a small outlier group.
- **PathMNIST**: preprocessed with `scale_unit_range` only (pixels → [0,1]; no manifold-method subsampling needed since methods used were chosen for scale). PCA (50 comp.) trustworthiness 0.9820 (79.07% variance, 178 comps for 90%, first PC alone 54.65%). Sparse PCA (20 comp., the required PCA-variant above the 5,000-sample O(N²) threshold) 0.9435. UMAP (30 neighbors) 0.8127. t-SNE (perplexity 30) 0.8347. Label silhouette **negative for every method** (−0.0336 to −0.0662) — raw-pixel distance does not separate the 9 tissue types well on average, consistent with the known lossy 28×28 downsampling from 224×224.
- **Critique outcomes**: pbmc's critique (6 findings) led to one real investigation (Diffusion Maps with a narrower kernel collapsed to a single point; reverted, documented as a limitation instead of "fixed"). PathMNIST's critique (5 findings) was fully addressed by rewriting vague `reason` text, including one finding considered and explicitly *rejected* (no available preprocessing step removes staining variation without also removing brightness that likely helps separate tissue types).

## 9. Lessons learned (from git history, todo.md, design.md, dead code)

- **"Standard choice" as a reason is explicitly banned and enforced by a validator**, not just a style guideline — `validate_plan.py`'s `_is_lazy_reason` check exists because early plans would have accepted vague justifications; this is worth carrying into Project 2's plan-validation design directly (hyperparameter/model choices for VAEs/GANs/diffusion are *very* easy to justify lazily).
- **Silhouette-alone k-selection was wrong in a documented, reproducible way** (commit `4194184`, `962ca9c`): it kept picking the coarsest split and visibly under-colored a real group on the live pbmc run; the fix (DBSCAN-nominated density groups cross-checked against k-means on the real embedding) is a nontrivial piece of applied judgment, not a one-line fix — the `dr-critic` prompt even has a dedicated paragraph telling the critic *not* to re-litigate this with plain silhouette. If Project 2 needs any kind of automatic model/cluster-count/latent-dim selection, expect the "obvious" metric-argmax approach to need a similar correction.
- **A `visualization_only`-style hard boundary mattered enough to be mechanically enforced**, not just documented: UMAP/t-SNE embeddings can never feed another computation, and `validate_plan.py` + `cluster.py` both refuse it at the code level, not just in a comment. Project 2 has a close analog: a GAN/VAE's generated samples or latent codes used only for *visualization/evaluation* vs. ones genuinely fed into downstream analysis need the same kind of explicit, code-enforced boundary, especially since the spec asks for "recommendations for downstream use of the generated data" — getting this boundary wrong (treating fake data as if it were real, or over-trusting a metric) is a classic generative-modeling pitfall the grading rubric likely watches for.
- **Artifacts are keyed by method name only** — a second run of the same method at a different seed/hyperparameter can't be kept alongside the first (stated limitation, `README.md` "Limitations"; workaround used was adding a *different* method, t-SNE alongside UMAP, rather than two UMAP seeds). If Project 2 needs multiple seeds/runs of the *same* generative model (common for GAN training-run comparisons), this naming scheme needs to be fixed up front, not inherited as-is.
- **No run history** — rerunning overwrites `run_log.json`/`metrics/clustering.json`; a revision's "before" state isn't recoverable except from the `reason`/critique text. Worth deciding deliberately for Project 2 whether training-run history (loss curves, intermediate checkpoints) needs to survive reruns — for generative models this is usually more valuable than it was for one-shot dimension-reduction metrics.
- **Dead code**: `scripts/test_api.py` was explicitly marked for deletion in `docs/design.md` §0 ("Delete once CLAUDE.md/skills work... a direct API call has no role in the pipeline") and was never deleted. It's harmless (not imported by anything, not run by CLAUDE.md) but is a concrete example that a "delete this later" design note doesn't enforce itself — if Project 2 reuses this repo's hygiene habits, don't assume a stated TODO got done; check.
- **`select_hvg` → `select_top_variance` rename** (commit `6f04580`): a method name that implied more biological sophistication ("highly variable genes," a real scanpy concept with mean-variance trend fitting) than the implementation actually did (plain top-variance ranking) was renamed once the mismatch was noticed. Worth the same scrutiny in Project 2 — e.g. don't let a function named `generate_samples` quietly mean "just fit one Gaussian," name things for what they actually do.
- **Everything gets built "pluggable, built to be added later without touching what exists"** (design.md §3.7: the critic and a synthetic smoke-test dataset were explicitly deferred and *did* get added later, cleanly, because they only read/write files that already existed). This pattern worked in practice (the critic was in fact added in one pass late in the timeline, commit `cd33694`, without needing to touch `reduce_dim.py`/`evaluate.py`/etc.) — a good model for sequencing Project 2's own build order (core pipeline first, critic/extra-dataset/stretch items deliberately last).

## 10. Reusable components

**Reuse as-is (generic, no dataset- or method-specific logic):**
- `scripts/loaders.py` (`load_dataset`, `load_generic`, `parse_loading_spec`) — the entire generic data-loading layer (csv/tsv/txt/npy/npz/h5ad, `## Loading`-spec-driven, label/id-column separation, delimiter sniffing, NA-token handling, numeric-only refusal with named offending columns). Project 2 needs the exact same "inspect an unknown dataset" entry point; there's no generative-modeling-specific reason to rewrite this.
- `scripts/profiler.py` (`profile_dataset`, `write_profile_summary_md`) — n_samples/n_features/sparsity/missingness/value-range/label profiling into `profile.json` + a human-readable `.md`. Directly satisfies Project 2's "inspect the dataset" requirement; may want new fields (e.g. image vs. tabular flag is already partially inferrable from shape/value-range, but Project 2's spec explicitly calls out "image or tabular data" detection — check whether that needs to be made explicit rather than inferred).
- `scripts/doctor.py` (`check_core`, `check_report`, subprocess-isolated import checking) — reuse the pattern and most of the code directly; just update the package list (PyTorch/Diffusers/etc. instead of umap-learn/anndata).
- `scripts/fetch_data.py` — reuse as-is; the `url`/`md5`-declared-in-`DATA_DESCRIPTION.md` fetch pattern has no dimension-reduction-specific content.
- `scripts/validate_plan.py` + `scripts/plan_schema.py` pattern (not the literal method lists) — the *shape* (allowed-methods set, required-role-per-method mapping, lazy-reason rejection, ordered-steps enforcement, pass/fail JSON+Markdown report) is exactly what Project 2 needs for validating a "which generative model + hyperparameters + preprocessing" plan before executing it. Rewrite the actual `ALLOWED_METHODS`/`METHOD_ROLES` contents for VAE/GAN/flow/diffusion, keep the validation machinery.
- `scripts/run_plan.py`'s orchestration pattern (loop over plan entries, skip-if-artifact-already-matches reuse check, retry-once-then-fallback, per-step progress line, `run_log.json`) — reuse the *pattern* directly; training a generative model is more expensive than fitting PCA, so the artifact-reuse-before-recompute discipline (CLAUDE.md rule 13) matters even more for Project 2.
- `CLAUDE.md`'s overall structure (mission/evaluation-setting/non-negotiable-rules/how-invoked/approved-methods/execution-sequence/guard-rails/fallback-rules/tool-contracts/output-requirements/skills-index/self-check/final-response-style) — reuse this section skeleton wholesale for Project 2's own `CLAUDE.md`; it's already been through one full design-research pass (`docs/design.md` §1) distilling what worked from three reference repos, no need to redo that research.
- `.claude/agents/dr-critic.md`'s pattern (fresh context, narrow explicit inputs, five fixed checked categories, structured JSON verdict, exactly-one-revision-loop, "never gates the final report") — reuse the mechanism; rewrite the five categories for generative-modeling failure modes (e.g. mode collapse without multi-seed evidence, FID/metric computed on too few samples, a VAE's KL term silently dominating, synthetic data proposed for a downstream use the evaluation never actually tested).

**Adapt (same shape, different content needed):**
- `.claude/skills/data-profiling/SKILL.md`'s decision-tree *pattern* (profile property → named modality → preprocessing steps, every branch requires a dataset-specific reason) — the actual branches (sparse-count/bounded-range/continuous/wide) are dimension-reduction-flavored; Project 2 needs branches keyed to generative-model fit (tabular vs. image vs. sequence, continuous vs. discrete, class-balanced vs. imbalanced-for-class-balancing-objective, etc.) and also needs to fold in *objective* interpretation (the user's stated goal: imputation/anomaly-detection/class-balancing/representation-learning/uncertainty — Project 1 has no analog to this axis at all, since DR has no "user objective" parameter).
- `.claude/skills/method-selection/SKILL.md`'s pattern (fixed method-role table, size-gated choice between alternatives, concrete hyperparameter heuristics with worked numbers, "this should typically land on N methods" guidance) — the role-table concept (`general_purpose`/`local_structure_only`/`visualization_only`) has a plausible generative analog (e.g. "distribution-fitting"/"sample-generation-only"/"representation-learning") worth designing deliberately rather than copying verbatim.
- `.claude/skills/reporting/SKILL.md`'s rules (every quoted number must trace to a metrics JSON, don't compare incommensurable outputs as a ranking, state null/negative results plainly, no claims the data can't support) — all directly portable to a generative-modeling report; just add the new required sections (recommendations for downstream use, generated-sample quality criteria).
- `scripts/reduce_dim.py`'s *registry pattern* (one script, `--method` dispatch to private `_run_<method>` functions, each returns embedding + sidecar params JSON with actual seed/runtime) — same shape works for "one script, `--model` dispatch to `_train_vae`/`_train_gan`/etc.", but the actual training code is new (PyTorch, not scikit-learn one-liners) and much heavier (checkpointing, loss curves, training time budgets).
- `scripts/evaluate.py`'s method→metric fixed-mapping pattern — the mapping itself (trustworthiness/explained-variance/stress) is DR-specific; Project 2 needs its own fixed mapping (FID/KID for images, some likelihood-based or MMD-style metric for tabular, reconstruction error for VAEs, etc.) chosen deliberately, not inherited.

**Rewrite (DR-specific, not reusable):**
- `scripts/reduce_dim.py`'s actual method bodies, `scripts/cluster.py` (k-means + density cross-check is a DR/EDA-specific coloring mechanism), `scripts/visualize.py`'s scatter-plot-specific logic (generative-model outputs need sample grids / reconstruction comparisons / training curves, not 2D scatter), `plan_schema.ALLOWED_METHODS`/`METHOD_ROLES` contents (keep the mechanism, not the values), and both `DATA_DESCRIPTION.md` files' framing-for-DR language (the facts about the two datasets are reusable, but the "this is a fully unsupervised task" framing needs rewriting once there's a stated generative objective per run).

## 11. Known weaknesses / things to fix or decide deliberately this time

- **Dead code never cleaned up** (`scripts/test_api.py`) — do an actual pass to delete anything marked "delete once X" rather than letting it linger; it's cheap and was skipped once already.
- **No GPU/training-infrastructure story exists in Project 1 at all** — this is new work for Project 2, not a gap to inherit. PyTorch installs, checkpointing, and training-time budgets (GAN/diffusion training can run far longer than scikit-learn's PCA/UMAP calls) need a deliberate plan; `doctor.py`'s "check per-activity, not one blanket gate" pattern should probably extend to a GPU-availability check that degrades to CPU/smaller-model gracefully rather than failing.
- **Artifact keying by method-name-only and no run history** (§8) should be fixed from the start if Project 2 expects to compare multiple seeds/hyperparameter settings of the *same* model type, which is far more likely to matter for GANs (notoriously seed-sensitive) than it was for PCA.
- **The "harness-only, no LLM API" choice was right for Project 1 but is a decision to make freshly for Project 2, not copy blindly** — flag it to the user early (see §3) rather than assuming. The STAI-X spec for Project 2 explicitly mentions OpenAI/Anthropic APIs and LangGraph as acceptable, which reads as a slightly more open invitation toward agent frameworks than Project 1's framing; worth a deliberate decision, stated as a judgment call in whatever becomes this project's own `CLAUDE.md`, exactly the way Project 1's own `CLAUDE.md` rule 11 would demand.
- **No categorical-feature support** — Project 1 refuses non-numeric columns outright. Generative modeling for tabular data (a very plausible dataset choice for Project 2, especially anything dissertation-related) commonly needs mixed continuous/categorical handling; decide explicitly whether to lift this restriction rather than inheriting it silently.
- **`outputs/<name>/` in the current repo is in a stale/partial state** (only a few stray `.npy` files remain; `profile.json`/`plan.json`/`metrics/` are gone for both datasets even though the final PDFs and `docs/manual_report_draft.md` describe a specific completed run). If Project 2 is a separate repo, this doesn't matter; if it reuses this repo's `outputs/`/`reports/` folders, regenerate rather than trust what's currently on disk. Also note two untracked, leftover artifacts sitting in the working tree right now that are not part of the pipeline and should not be mistaken for project files: `dim-reduction-agent-source.zip` and the extracted `dim-reduction-agent-source/` folder (a prior Canvas-submission snapshot), plus a loose `report.pdf` and `docs/manual_report_draft.md` at/near the root — these are submission artifacts, not pipeline code.
