---
name: reporting
description: Write findings.md and build reports/<dataset>/generated_report_<N>.pdf, then check the rendered pages. Used in step 9 of AGENT.md, after recommendation.json passes validation.
---

# Reporting

`scripts/report.py` builds the PDF from the JSON artifacts: summary, a "How to read this report" page (run names and seeds, the evaluation space and its variance explained, reference values, how memorization is measured, and a caution for models trained in PCA space), objective interpretation, data and preprocessing (with every `reason`), models and training, metrics tables, metric definitions, figures, recommendation, critique, limitations, and reproducibility. **You write one file: `outputs/<dataset>/findings.md`**, the written interpretation. Since `reason` fields are printed verbatim, a vague reason in `plan.json` becomes a vague sentence in the report. Fix the plan, not the findings.

## Steps

1. **Read before writing:** `objective.json`, `plan.json`, `run_log.json`, `runs/*.json` (training outcomes), every `metrics/*.json`, `recommendation.json`, and **look at every PNG in `figures/`** with the Read tool. Don't infer what a figure shows from the numbers.

2. **Write `findings.md`.** Markdown, `## ` headings, blank lines between paragraphs. About one to two pages. Sections, in this order:
   - `## Summary`: 3–5 sentences, shown on page 1. What was asked, what the agent decided, the main result in numbers, and the bottom-line recommendation.
   - `## What each model produced`: per model, the headline metrics, always next to their real-vs-real reference (C2ST vs reference, MMD vs reference), precision and recall, memorization, and training outcome (kept epoch, why it stopped).
   - `## What the figures show`: what you saw in each figure, described plainly ("synthetic samples fill the space between the real clusters").
   - `## How the models compare`: only on metrics computed in the same space (fidelity, memorization, feature stats). **Never compare likelihoods across models with different input spaces**; say they are not comparable. Name which model is better for which purpose, not an overall winner, unless one is better on every shared metric. If one model is trained on PCA scores and realism is scored in a PCA space, say that this may favor it on the realism metrics, and weigh the per-feature checks accordingly.
   - `## Answer to the objective`: did the run meet the success criterion in `objective.json`? Yes, partly, or no, with the numbers.
   - `## Limitations`: anything specific to this run that a reader must know (merged into the Limitations page).

3. **Build:**
   ```bash
   venv/bin/python scripts/doctor.py --check report
   venv/bin/python scripts/report.py --dataset <dataset> --report-number <N>
   ```
   `report.py` traces every decimal number in `findings.md` to the JSON artifacts at the precision written. If it lists untraced numbers, fix `findings.md` (copy values exactly as they appear in the JSON or the report tables) and rerun. Never use `--allow-untraced` to get past it.

4. **Check the rendered pages.** `report.py` saves every page as `outputs/<dataset>/report_pages/page_NN.png`. Read the summary page, the metrics page, the findings pages, and at least one figure page. Check: no text runs off the page edge, no page is blank, every figure shows a figure, and the summary matches the findings. Fix and rebuild if anything is wrong. A script that ran without errors does not prove the PDF is right.

## Rules for the findings

- **Name runs as the report does** (`<model id>_s<seed>`) and treat a second seed as a stability check: say whether the conclusion holds for both seeds, never present the two values as a range or an interval.

- **Every number comes from an artifact** (`metrics/*.json`, `runs/*.json`, `profile.json`, `prepared/meta.json`, `run_log.json`), written with the same rounding as there. Don't compute new statistics in prose.
- **Always give the reference next to a fidelity number.** "C2ST AUC 0.93 (real-vs-real reference 0.56)" says something; "C2ST AUC 0.93" alone doesn't.
- **State negative results plainly.** "A classifier separates synthetic from real samples almost perfectly" is a finding. Don't soften it into "reasonable fidelity".
- **No privacy claims.** Memorization metrics are evidence about copying, not a privacy guarantee. Same forbidden phrases as `downstream-use/SKILL.md`.
- **No claims the data can't support.** For unlabeled data, don't name types or classes the data doesn't label; describe structure in the figures only, as exploratory.
- **Plain, direct language.** Short sentences, no promotional wording, no unexplained jargon.
