"""Shared constants for objective.json and plan.json. Imported by validate_objective.py,
   validate_plan.py, preprocess.py, train.py and evaluate_gen.py so the task list, model
   capabilities, and preprocessing steps can't drift apart between them.

   Pattern copied from Project 1's plan_schema.py (fixed tables + validator), contents rewritten
   for generative modeling.
"""

# ── Objectives ──
# What the user can ask for. The agent maps free text onto exactly one primary task
# (plus optional secondary tasks); validate_objective.py rejects anything else.
OBJECTIVE_TASKS = {
    "data_generation",            # realistic synthetic samples (e.g. for sharing, simulation)
    "augmentation",               # extra training data for a downstream model
    "class_balancing",            # synthetic samples for under-represented classes
    "imputation",                 # fill in missing or masked values
    "anomaly_detection",          # score how unusual each sample is
    "representation_learning",    # a low-dimensional latent space for downstream analysis
    "uncertainty_quantification", # calibrated density / likelihood estimates
}
TASKS_REQUIRING_LABELS = {"class_balancing"}
DECISIONS = {"proceed", "proceed_adapted", "decline"}

# Below this many samples no generative model is trained; the objective must be declined.
MIN_SAMPLES_ANY_MODEL = 200
# class_balancing below this max/min class-size ratio: classes are close to balanced, so the
# decision must be proceed_adapted (run, but framed as "check whether synthesis beats resampling").
NEAR_BALANCED_RATIO = 1.5

# ── Models ──
MODEL_FAMILIES = {"vae", "gan", "flow", "diffusion"}

# Fixed capability profile per family. validate_plan.py rejects a model assigned to a task it
# has no capability for (e.g. a GAN for anomaly detection: it has no likelihood).
MODEL_CAPABILITIES = {
    "vae":       {"sampling", "latent", "likelihood_bound", "reconstruction", "conditional"},
    "gan":       {"sampling", "conditional"},
    "flow":      {"sampling", "exact_likelihood", "conditional"},
    "diffusion": {"sampling", "reconstruction", "conditional"},
}
# A task is served by a model having ANY one of the listed capability sets (all of the set).
TASK_NEEDS = {
    "data_generation":            [{"sampling"}],
    "augmentation":               [{"sampling"}],
    "class_balancing":            [{"sampling", "conditional"}],
    "imputation":                 [{"reconstruction"}],
    "anomaly_detection":          [{"exact_likelihood"}, {"likelihood_bound"}],
    "representation_learning":    [{"latent"}],
    "uncertainty_quantification": [{"exact_likelihood"}, {"likelihood_bound"}],
}

# Minimum training-set size per family. GANs and diffusion need more data to train stably.
MIN_TRAIN_SAMPLES = {"vae": 200, "flow": 200, "gan": 1000, "diffusion": 1000}
# GANs are seed-sensitive: one run is not evidence of anything.
MIN_SEEDS = {"gan": 2}

# (family, modality, conditional) combinations with working code. Grows as models are added;
# validate_plan.py refuses anything else so a plan can't ask for a model that doesn't exist.
IMPLEMENTED = {
    ("vae", "tabular", False),
    ("flow", "tabular", False),
}

# What the model is trained on (built by preprocess.py, fit on the training split only):
# raw = selected features, untransformed (counts stay counts); transformed = after the plan's
# transform steps; pca = PCA of transformed, n components set by the model's pca_components.
INPUT_SPACES = {"raw", "transformed", "pca"}
LIKELIHOODS = {"nb", "gaussian", "bernoulli"}

MODEL_STATUSES = {"planned", "not_run_budget"}

# Hyperparameters each family's training code reads; every one needs its own reason in the plan.
REQUIRED_HYPERPARAMS = {
    "vae":       {"latent_dim", "hidden_dims", "epochs", "batch_size", "lr", "likelihood", "kl_warmup_epochs"},
    "flow":      {"n_layers", "hidden_dim", "epochs", "batch_size", "lr"},
    "gan":       {"latent_dim", "epochs", "batch_size", "lr"},
    "diffusion": {"timesteps", "epochs", "batch_size", "lr"},
}

# ── Preprocessing ──
# Three kinds, which must appear in this order: rows, then feature selection, then transforms.
ROW_STEPS = {"stratified_subsample", "subsample"}
FEATURE_STEPS = {"select_top_variance"}
TRANSFORM_STEPS = {"normalize_total", "log1p", "standardize", "scale_unit_range"}
KNOWN_PREPROCESSING_STEPS = ROW_STEPS | FEATURE_STEPS | TRANSFORM_STEPS
STEP_KIND_ORDER = {**{s: 0 for s in ROW_STEPS}, **{s: 1 for s in FEATURE_STEPS}, **{s: 2 for s in TRANSFORM_STEPS}}
# Steps that only make sense on non-negative data (counts, intensities).
NONNEG_ONLY_STEPS = {"normalize_total", "log1p"}

TEST_FRACTION_RANGE = (0.1, 0.4)
# Default PCA size of the shared tabular evaluation space (capped by the data's own size).
EVAL_PCA_COMPONENTS = 50

# "standard choice"/"default" alone is never a reason (Project 1 rule 12, carried over).
LAZY_REASONS = {"", "standard choice", "default", "standard", "n/a", "na", "common choice", "typical"}
MIN_REASON_CHARS = 25


def is_lazy_reason(reason):
    return (not isinstance(reason, str)
            or reason.strip().lower().rstrip(".") in LAZY_REASONS
            or len(reason.strip()) < MIN_REASON_CHARS)


def serves(family, task):
    caps = MODEL_CAPABILITIES[family]
    return any(need <= caps for need in TASK_NEEDS[task])


def artifact_key(model_id, seed):
    """Every artifact (checkpoint, samples, metrics) is keyed by model id AND seed, so several
    seeds of one model can coexist (a Project 1 limitation fixed here)."""
    return f"{model_id}_s{seed}"


assert set(TASK_NEEDS) == OBJECTIVE_TASKS
assert set(MODEL_CAPABILITIES) == MODEL_FAMILIES
