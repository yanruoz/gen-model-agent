"""Shared helpers for train.py / generate.py / evaluate_gen.py: build a model from its plan entry,
   pick the training input for its input_space, choose a device, and save/load checkpoints.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.flow import RealNVP  # noqa: E402
from models.vae import VAE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def outputs(dataset):
    return ROOT / "outputs" / dataset


def load_plan(dataset):
    return json.loads((outputs(dataset) / "plan.json").read_text())


def model_entry(plan, model_id):
    for m in plan["models"]:
        if m["id"] == model_id:
            return m
    raise ValueError(f"no model '{model_id}' in plan.json (have {[m['id'] for m in plan['models']]})")


def config_hash(entry, seed, prep_hash):
    """Identifies a trained artifact: everything that changes training, nothing that doesn't (reasons)."""
    key = {k: entry.get(k) for k in ("family", "conditional", "input_space", "hyperparameters")}
    key.update(seed=seed, prep_hash=prep_hash)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def pick_device(modality):
    """Small MLPs on tabular data run faster on CPU than on MPS (transfer overhead dominates);
    convolutional image models benefit from MPS/CUDA when present."""
    if modality == "image":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def fit_input_pca(trans_train, n_components, seed):
    """PCA used as a model's own input space (input_space 'pca'); separate from the eval PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=seed).fit(trans_train)
    # Per-feature std of what the kept components don't explain. Generation adds independent Gaussian
    # noise with this std, so a model on PCA scores still generates full-dimensional data instead of
    # samples confined to a k-dimensional subspace (which any classifier would spot immediately).
    resid = trans_train - pca.inverse_transform(pca.transform(trans_train))  # (n_train, d)
    return {"mean": pca.mean_.astype(np.float32), "components": pca.components_.astype(np.float32),
            "evr": float(pca.explained_variance_ratio_.sum()), "resid_std": resid.std(0).astype(np.float32)}


def model_input(entry, prepared, split, input_pca=None):
    """The (n, d) float32 array a model trains on / is scored on, for its input_space."""
    space = entry["input_space"]
    if space == "raw":
        return prepared[f"raw_{split}"]
    trans = prepared[f"trans_{split}"]
    if space == "transformed":
        return trans
    return ((trans - input_pca["mean"]) @ input_pca["components"].T).astype(np.float32)


def build(entry, n_in, n_classes=0):
    hp = entry["hyperparameters"]
    if entry["family"] == "vae":
        return VAE(n_in, hp["latent_dim"], hp["hidden_dims"], hp["likelihood"], n_classes)
    if entry["family"] == "flow":
        return RealNVP(n_in, hp["n_layers"], hp["hidden_dim"])
    raise ValueError(f"family {entry['family']} not implemented")


def save_checkpoint(path, model, extra):
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(dataset, key, device="cpu"):
    ckpt = torch.load(outputs(dataset) / "checkpoints" / f"{key}.pt", map_location=device, weights_only=False)
    model = build(ckpt["entry"], ckpt["n_in"], ckpt.get("n_classes", 0))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
