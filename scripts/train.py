"""Train one model from plan.json at one seed.

   Usage: venv/bin/python scripts/train.py --dataset <d> --model-id <id> --seed <int>

   Writes outputs/<d>/checkpoints/<id>_s<seed>.pt and outputs/<d>/runs/<id>_s<seed>.json (config hash,
   loss history, epochs run, minutes, device).

   Early stopping: 10% of the TRAINING split is set aside as a validation set. Training stops when
   validation loss hasn't improved for PATIENCE epochs (for VAEs, counted only after KL warm-up ends),
   and the best-validation weights are kept. `epochs` in the plan is therefore a maximum.
   A wall-clock cap of 3 x estimated_minutes (at least 2 min) also stops training.
   The held-out TEST split's loss is logged for the curves only; it never picks a model or an epoch.
"""
import argparse
import json
import math
import time

import numpy as np
import torch

from model_io import (build, config_hash, fit_input_pca, load_plan, model_entry, model_input, outputs,
                      pick_device, save_checkpoint)
from plan_schema import artifact_key
from preprocess import load_prepared

EVAL_EVERY = 5      # epochs between held-out (test) loss evaluations, for the curves only
VAL_FRACTION = 0.1  # share of the training split used for early stopping
PATIENCE = 25       # epochs without validation improvement before stopping


def _batches(n, batch_size, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


@torch.no_grad()
def _heldout_loss(model, family, x, chunk=1024):
    """Mean held-out loss on the same scale as the training loss (negative ELBO or NLL)."""
    model.eval()
    vals = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        vals.append(-(model.elbo(xb) if family == "vae" else model.log_prob(xb)))
    model.train()
    return float(torch.cat(vals).mean())


def train(dataset, model_id, seed):
    plan = load_plan(dataset)
    entry = model_entry(plan, model_id)
    prepared = load_prepared(dataset)
    meta = prepared["meta"]
    key = artifact_key(model_id, seed)
    hp, family = entry["hyperparameters"], entry["family"]
    modality = "image" if meta.get("image_shape") else "tabular"
    device = pick_device(modality)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Inputs for this model's input_space (PCA input fit on train only) ──
    input_pca = fit_input_pca(prepared["trans_train"], hp["pca_components"], seed) if entry["input_space"] == "pca" else None
    x_all = torch.from_numpy(model_input(entry, prepared, "train", input_pca))            # (n_train, d_in)
    x_te = torch.from_numpy(model_input(entry, prepared, "test", input_pca)).to(device)   # (n_test, d_in)
    n_in = x_all.shape[1]

    # ── Validation split carved from TRAIN for early stopping (test stays untouched) ──
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(VAL_FRACTION * len(x_all))))
    x_val, x_tr = x_all[perm[:n_val]].to(device), x_all[perm[n_val:]].to(device)          # (n_val, d_in), (n_fit, d_in)

    model = build(entry, n_in).to(device)
    if family == "flow":
        model.set_normalization(x_tr)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    cap_min = max(2.0, 3.0 * float(entry.get("estimated_minutes", 1)))
    gen = torch.Generator().manual_seed(seed)
    history, t0, stopped_early, stop_reason = [], time.time(), False, "max epochs reached"
    best_val, best_epoch, best_state = float("inf"), 0, None
    warm = max(1, int(hp.get("kl_warmup_epochs", 0)))
    epochs = int(hp["epochs"])

    print(f"[train] {key}: {family} on {entry['input_space']} {tuple(x_tr.shape)}, {epochs} epochs, device={device}")
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / warm) if family == "vae" else None  # KL warm-up: weight 0 -> 1
        sums, nb = {"loss": 0.0, "recon_ll": 0.0, "kl": 0.0}, 0
        for idx in _batches(len(x_tr), hp["batch_size"], gen):
            xb = x_tr[idx.to(device)]
            if family == "vae":
                loss, ll, kl = model(xb, beta=beta)
                sums["recon_ll"] += ll.item(); sums["kl"] += kl.item()
            else:
                loss = model(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sums["loss"] += loss.item(); nb += 1
        if not math.isfinite(sums["loss"]):
            raise RuntimeError(f"loss became non-finite at epoch {epoch}")
        row = {"epoch": epoch, **{k: v / nb for k, v in sums.items() if family == "vae" or k == "loss"}}
        if family == "vae":
            row["beta"] = beta
        row["val_loss"] = _heldout_loss(model, family, x_val)
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            row["heldout_loss"] = _heldout_loss(model, family, x_te)
        history.append(row)

        # Early stopping on validation loss; a VAE's loss is only comparable once beta has reached 1.
        if family != "vae" or beta >= 1.0:
            if row["val_loss"] < best_val:
                best_val, best_epoch = row["val_loss"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= PATIENCE:
                stopped_early, stop_reason = True, f"validation loss did not improve for {PATIENCE} epochs"
                break
        if (time.time() - t0) / 60 > cap_min:
            stopped_early, stop_reason = True, f"wall-clock cap {cap_min:.1f} min reached"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train] {key}: stopped at epoch {history[-1]['epoch']} ({stop_reason}); keeping epoch {best_epoch}")

    minutes = (time.time() - t0) / 60
    ck_dir, run_dir = outputs(dataset) / "checkpoints", outputs(dataset) / "runs"
    ck_dir.mkdir(exist_ok=True); run_dir.mkdir(exist_ok=True)
    extra = {"entry": entry, "n_in": n_in, "seed": seed, "input_pca": input_pca}
    if family == "vae" and hp["likelihood"] == "nb":
        # Empirical library sizes of the training cells; resampled when generating new cells.
        extra["train_library"] = prepared["raw_train"].sum(1).astype(np.float32)
    save_checkpoint(ck_dir / f"{key}.pt", model.cpu(), extra)

    final = history[-1]
    best_row = next((r for r in history if r["epoch"] == best_epoch), final)
    heldout_best = _heldout_loss(model, family, x_te)  # test loss of the kept weights (reporting only)
    sidecar = {
        "key": key, "model_id": model_id, "seed": seed, "family": family,
        "config_hash": config_hash(entry, seed, meta["prep_hash"]), "device": device,
        "epochs_planned": epochs, "epochs_run": final["epoch"], "best_epoch": best_epoch,
        "stopped_early": stopped_early, "stop_reason": stop_reason,
        "time_cap_min": cap_min, "minutes": round(minutes, 3),
        "n_fit": int(len(x_tr)), "n_val": int(len(x_val)), "n_in": n_in,
        "input_pca_variance_explained": input_pca and round(input_pca["evr"], 4),
        "kept_train_loss": best_row["loss"], "kept_val_loss": best_row.get("val_loss"),
        "kept_heldout_loss": round(heldout_best, 4),
        "loss_definition": "negative ELBO per sample" if family == "vae" else "negative log-likelihood per sample",
        "history": history,
    }
    (run_dir / f"{key}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"[train] {key}: done in {minutes:.2f} min; kept epoch {best_epoch}: train {best_row['loss']:.3f}, "
          f"val {best_row.get('val_loss', float('nan')):.3f}, held-out {heldout_best:.3f}")
    return sidecar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one plan.json model at one seed.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train(args.dataset, args.model_id, args.seed)
