"""Tests for validate_objective.py (incl. the "whether to model" checks) and validate_plan.py.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_objective import validate_objective  # noqa: E402
from validate_plan import validate_plan  # noqa: E402

R = "a specific reason that cites this dataset's profile numbers"

PBMC_PROFILE = {"n_samples": 2700, "n_features": 32738, "has_labels": False, "modality": "tabular",
                "is_nonneg_integer": True, "value_range": [0.0, 419.0]}
IMG_PROFILE = {"n_samples": 89996, "n_features": 2352, "has_labels": True, "modality": "image",
               "is_nonneg_integer": True, "value_range": [0.0, 255.0], "imbalance_ratio": 1.634,
               "class_counts": {str(i): 9000 for i in range(9)}}


def objective(**kw):
    o = {"raw_text": "learn a representation of these cells", "task": "representation_learning",
         "secondary_tasks": [], "target": {}, "success_criterion": R, "interpretation_reason": R,
         "decision": "proceed", "decision_reason": R, "effective_task": "representation_learning",
         "ambiguities": []}
    o.update(kw)
    return o


def vae(**kw):
    m = {"id": "vae_nb", "family": "vae", "conditional": False, "input_space": "raw",
         "hyperparameters": {"latent_dim": 10, "hidden_dims": [128], "epochs": 100, "batch_size": 128,
                             "lr": 1e-3, "likelihood": "nb", "kl_warmup_epochs": 10},
         "seeds": [0], "serves_tasks": ["representation_learning"], "estimated_minutes": 2,
         "status": "planned", "reason": R}
    m["hyperparameter_reasons"] = {k: R for k in m["hyperparameters"]}
    m.update(kw)
    return m


def plan(models=None, **kw):
    p = {"dataset": "pbmc", "seed": 0, "budget_min": 20, "revision": 1,
         "split": {"test_fraction": 0.2, "stratify": False, "reason": R},
         "preprocessing": [{"step": "select_top_variance", "params": {"n_features": 2000}, "reason": R}],
         "models": [vae()] if models is None else models}
    p.update(kw)
    return p


class TestObjective(unittest.TestCase):
    def ok(self, o, profile=PBMC_PROFILE, text=None):
        errors, _ = validate_objective(o, profile, text)
        self.assertEqual(errors, [])

    def bad(self, o, pattern, profile=PBMC_PROFILE, text=None):
        errors, _ = validate_objective(o, profile, text)
        self.assertTrue(any(pattern in e for e in errors), errors)

    def test_valid_objective_passes(self):
        self.ok(objective(), text="learn a representation of these cells")

    def test_raw_text_must_be_verbatim(self):
        self.bad(objective(), "verbatim", text="something else")

    def test_unknown_task(self):
        self.bad(objective(task="clustering", effective_task="clustering"), "not one of")

    def test_lazy_reason_rejected(self):
        self.bad(objective(decision_reason="default"), "decision_reason")

    def test_labels_needed_but_absent_cannot_proceed(self):
        self.bad(objective(task="class_balancing", effective_task="class_balancing"), "needs class labels")

    def test_labels_needed_but_absent_can_be_declined(self):
        self.ok(objective(task="class_balancing", decision="decline", effective_task=None))

    def test_labels_needed_but_absent_can_be_adapted_to_unlabeled_task(self):
        self.ok(objective(task="class_balancing", decision="proceed_adapted", effective_task="data_generation"))

    def test_adapted_task_still_cannot_need_labels(self):
        self.bad(objective(task="class_balancing", decision="proceed_adapted", effective_task="class_balancing"),
                 "needs class labels")

    def test_nearly_balanced_classes_cannot_plainly_proceed(self):
        prof = dict(IMG_PROFILE, imbalance_ratio=1.2)
        self.bad(objective(task="class_balancing", effective_task="class_balancing"), "nearly balanced", profile=prof)
        self.ok(objective(task="class_balancing", decision="proceed_adapted", effective_task="class_balancing"), profile=prof)

    def test_mild_imbalance_above_threshold_can_proceed(self):
        self.ok(objective(task="class_balancing", effective_task="class_balancing"), profile=IMG_PROFILE)

    def test_unknown_target_class(self):
        self.bad(objective(task="class_balancing", effective_task="class_balancing", target={"class": "99"}),
                 "not a class", profile=IMG_PROFILE)

    def test_too_few_samples_must_decline(self):
        self.bad(objective(), "decline", profile=dict(PBMC_PROFILE, n_samples=150))

    def test_decision_effective_task_consistency(self):
        self.bad(objective(effective_task="data_generation"), "effective_task == task")
        self.bad(objective(decision="decline"), "effective_task = null")


class TestPlan(unittest.TestCase):
    def errors(self, p, o=None, profile=PBMC_PROFILE):
        return validate_plan(p, o or objective(), profile)[0]

    def bad(self, p, pattern, o=None, profile=PBMC_PROFILE):
        errs = self.errors(p, o, profile)
        self.assertTrue(any(pattern in e for e in errs), errs)

    def test_valid_plan_passes(self):
        self.assertEqual(self.errors(plan()), [])

    def test_gan_cannot_do_anomaly_detection(self):
        o = objective(task="anomaly_detection", effective_task="anomaly_detection")
        gan = vae(id="gan", family="gan", serves_tasks=["anomaly_detection"], seeds=[0, 1])
        self.bad(plan([gan]), "lacks the capability", o)

    def test_gan_needs_two_seeds(self):
        o = objective(task="data_generation", effective_task="data_generation")
        gan = vae(id="gan", family="gan", serves_tasks=["data_generation"], input_space="transformed")
        self.bad(plan([gan]), "at least 2 seeds", o)

    def test_unimplemented_model_rejected_unless_not_run(self):
        o = objective(task="data_generation", effective_task="data_generation")
        diff = vae(id="diff", family="diffusion", serves_tasks=["data_generation"], input_space="transformed")
        self.bad(plan([vae(serves_tasks=["data_generation"]), diff]), "not implemented", o)

    def test_budget_overrun_rejected_and_not_run_budget_accepted(self):
        big = vae(estimated_minutes=15, seeds=[0, 1])
        self.bad(plan([big]), "budget")
        o = objective(task="data_generation", effective_task="data_generation")
        diff = vae(id="diff", family="diffusion", serves_tasks=["data_generation"], input_space="transformed",
                   estimated_minutes=60, status="not_run_budget",
                   hyperparameters={"timesteps": 1000, "epochs": 50, "batch_size": 128, "lr": 2e-4})
        diff["hyperparameter_reasons"] = {k: R for k in diff["hyperparameters"]}
        self.assertEqual(self.errors(plan([vae(serves_tasks=["data_generation"]), diff]), o), [])

    def test_every_hyperparameter_needs_a_reason(self):
        m = vae()
        m["hyperparameter_reasons"]["lr"] = "default"
        self.bad(plan([m]), "'lr'")

    def test_nb_needs_raw_counts(self):
        self.bad(plan([vae(input_space="transformed")]), "negative binomial")
        self.bad(plan([vae()]), "negative binomial", profile=dict(PBMC_PROFILE, is_nonneg_integer=False))

    def test_conditional_needs_labels(self):
        self.bad(plan([vae(conditional=True)]), "need labels")

    def test_model_must_serve_a_run_task(self):
        self.bad(plan([vae(serves_tasks=["imputation"])]), "not one of this run's tasks")

    def test_primary_task_must_be_covered(self):
        o = objective(secondary_tasks=["data_generation"])
        flow = vae(id="flow", family="flow", serves_tasks=["data_generation"], input_space="pca",
                   hyperparameters={"n_layers": 6, "hidden_dim": 128, "epochs": 100, "batch_size": 128,
                                    "lr": 1e-3, "pca_components": 20})
        flow["hyperparameter_reasons"] = {k: R for k in flow["hyperparameters"]}
        self.bad(plan([flow]), "primary task", o)

    def test_pca_input_needs_components(self):
        o = objective(task="data_generation", effective_task="data_generation")
        flow = vae(id="flow", family="flow", serves_tasks=["data_generation"], input_space="pca",
                   hyperparameters={"n_layers": 6, "hidden_dim": 128, "epochs": 100, "batch_size": 128, "lr": 1e-3})
        flow["hyperparameter_reasons"] = {k: R for k in flow["hyperparameters"]}
        self.bad(plan([flow]), "pca_components", o)

    def test_preprocessing_order_and_names(self):
        p = plan(preprocessing=[{"step": "log1p", "reason": R},
                                {"step": "select_top_variance", "params": {"n_features": 10}, "reason": R}])
        self.bad(p, "out of order")
        self.bad(plan(preprocessing=[{"step": "magic", "reason": R}]), "unknown step")

    def test_declined_objective_needs_empty_models(self):
        o = objective(task="class_balancing", decision="decline", effective_task=None)
        self.bad(plan(), "must be empty", o)
        self.assertEqual(self.errors(plan(models=[]), o), [])

    def test_test_fraction_range(self):
        p = plan()
        p["split"] = dict(p["split"], test_fraction=0.9)
        self.bad(p, "test_fraction")

    def test_too_few_training_samples_for_family(self):
        p = plan(preprocessing=[{"step": "subsample", "params": {"n_samples": 220}, "reason": R}])
        self.bad(p, "below the 200 minimum")


if __name__ == "__main__":
    unittest.main()
