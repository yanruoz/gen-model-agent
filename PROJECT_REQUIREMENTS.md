# Project 2: AI Agent for Generative Modeling (BIOS 774)

**Due:** October 26, 2026, 11:59 PM EST (submit via Canvas)
**Late policy:** 20% off the total possible score per day (or part of a day) late. More than 5 days late gets no credit.

## Goal

Build an AI agent that automatically runs generative modeling on real-world datasets. This is **not** a benchmarking study. The agent should decide **whether and how** to apply generative modeling, based on the dataset and the user objective.

Inputs: a new dataset + a user objective.
Agent outputs: a chosen workflow, trained model(s), evaluation of generated data, downstream-use recommendations, and a final report.

The agent must make real, data- and objective-dependent decisions (model choice, hyperparameters, evaluation criteria, downstream use). It should not run one fixed pipeline for every input.

## Required Agent Steps

1. **Inspect the dataset**: number of samples, feature types, class balance, missing values, image vs. tabular, etc.
2. **Interpret the user objective**: e.g., data generation, imputation, anomaly detection, class balancing, representation learning, uncertainty quantification.
3. **Select one or more generative models** suited to the data and objective.
4. **Select hyperparameters.**
5. **Train the model(s)** and generate synthetic samples when appropriate.
6. **Evaluate generated data** with appropriate quantitative or qualitative criteria.
7. **Recommend downstream use** of the generated data for the given objective.
8. **Produce a final analysis report** summarizing decisions and findings.

## Generative Models

The agent should be able to use the models covered in class, including (but not limited to):

- VAEs
- GANs
- Normalizing Flows
- Diffusion Models

It does not need to apply every model to every dataset. It should pick the appropriate model(s) and **explain its choices**.

## Datasets

Evaluate on **at least two real datasets**, ideally with different characteristics. Options:

- Own data (ideally related to dissertation), or
- Backup datasets:
  - PathMNIST or other MedMNIST variants (image)
  - Single-cell RNA-seq data, e.g., Scanpy 3K PBMC (tabular / high-dimensional)

## Deliverables

### 1. Source code
- Runs the **complete pipeline with a single command**.
- Minimal manual intervention after providing a dataset and a user objective.

### 2. Generated outputs (for each of the two datasets)
The agent must automatically produce:
- synthetic data or other generated outputs (when appropriate)
- evaluation results for the selected model(s)
- a summary of the selected models and training decisions
- recommendations for downstream use of the generated data
- a final analysis report

Format is up to us. The two reports must be named:
- `generated_report_1`
- `generated_report_2`

### 3. Written report
- PDF named **`report`**, **at most 4 pages** (excluding references and appendices).
- Must include:
  - overview of the system architecture
  - how the agent makes generative modeling decisions
  - tools or AI models used
  - experimental results on all datasets
  - strengths and limitations of the system
- Focus on the **design and behavior of the agent**, not an exhaustive benchmark of individual models.

## Grading Criteria

1. Degree of automation and autonomy of the pipeline
2. Quality of the analysis and modeling decisions made by the agent
3. Correct use of generative models
4. Quality of the generated outputs
5. Quality of the downstream-use recommendations
6. Quality of the generated reports and interpretations
7. Robustness on different datasets
8. Overall software design and reproducibility

Emphasis: an **intelligent, reusable generative modeling system**, not the best synthetic samples on one dataset.

## Allowed Tools

Any publicly available packages or AI models, e.g.:
PyTorch, TensorFlow, Hugging Face Diffusers, MONAI, OpenAI APIs, Anthropic APIs, LangGraph, or other agent frameworks.
Implementation language (Python-based) and agent architecture are up to us.

## Submission Checklist

- [ ] Source code (single-command pipeline)
- [ ] `generated_report_1` (dataset 1 outputs + report)
- [ ] `generated_report_2` (dataset 2 outputs + report)
- [ ] `report.pdf` (max 4 pages, excluding references/appendices)
- [ ] Uploaded to Canvas by Oct 26, 2026, 11:59 PM EST
