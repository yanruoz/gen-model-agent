"""Train one model from plan.json at one seed.

   Usage: venv/bin/python scripts/train.py --dataset <d> --model-id <id> --seed <int>

   Writes outputs/<d>/checkpoints/<id>_s<seed>.pt and outputs/<d>/runs/<id>_s<seed>.json (config hash,
   loss history, epochs run, minutes, device).

   VAE, flow, diffusion: 10% of the TRAINING split is set aside as a validation set. Training stops when
   validation loss hasn't improved for `patience` epochs (for VAEs, counted only after KL warm-up ends)
   and the best-validation weights are kept, so `epochs` in the plan is a maximum.
   patience = 25 epochs, or a fifth of the maximum when that is smaller (at least 5).
   GAN: trained for the planned number of epochs and the final weights are kept. GAN losses do not
   measure sample quality, so there is nothing sound to early-stop on.
   All families: a wall-clock cap of 3 x estimated_minutes (at least 2 min) also stops training.
   The held-out TEST split's loss is logged for the curves only; it never picks a model or an epoch.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_io import (build, config_hash, fit_input_pca, load_plan, model_entry, model_input, outputs,
                      pick_device, save_checkpoint)
from plan_schema import artifact_key
from preprocess import load_prepared

EVAL_EVERY = 5      # epochs between held-out (test) loss evaluations, for the curves only
VAL_FRACTION = 0.1  # share of the training split used for early stopping
MAX_PATIENCE = 25


def _batches(n, batch_size, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _cond(family, conditional, y_idx, n_classes):
    """Conditioning input: one-hot for VAEs, class index for GAN/diffusion, None if unconditional."""
    if not conditional:
        return None
    return F.one_hot(y_idx, n_classes).float() if family == "vae" else y_idx


@torch.no_grad()
def _eval_loss(model, family, x, c, chunk=512):
    """Mean loss on held-aside data, on the training loss's scale. Diffusion loss is stochastic (random
    timestep and noise), so it is evaluated under a fixed seed to make epochs comparable."""
    model.eval()
    vals = []
    with torch.random.fork_rng(devices=[]):  # CPU generator only; device RNGs are reseeded by manual_seed
        torch.manual_seed(0)
        for i in range(0, len(x), chunk):
            xb, cb = x[i:i + chunk], (c[i:i + chunk] if c is not None else None)
            if family == "vae":
                vals.append(-model.elbo(xb, cb))
            elif family == "flow":
                vals.append(-model.log_prob(xb))
            else:
                vals.append(model(xb, cb if cb is not None else torch.zeros(len(xb), dtype=torch.long, device=xb.device)).expand(len(xb)))
    model.train()
    return float(torch.cat(vals).mean())


def _train_likelihood_family(model, family, hp, x_tr, c_tr, x_val, c_val, x_te, c_te, key, cap_min, seed):
    """VAE / flow / diffusion: minimize the model's loss with early stopping on the validation slice."""
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    gen = torch.Generator().manual_seed(seed)
    epochs = int(hp["epochs"])
    patience = max(5, min(MAX_PATIENCE, epochs // 5))
    warm = max(1, int(hp.get("kl_warmup_epochs", 0)))
    history, t0, best_val, best_epoch, best_state = [], time.time(), float("inf"), 0, None
    stop_reason = "max epochs reached"
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / warm) if family == "vae" else None  # KL warm-up: weight 0 -> 1
        sums, nb = {"loss": 0.0, "recon_ll": 0.0, "kl": 0.0}, 0
        for idx in _batches(len(x_tr), hp["batch_size"], gen):
            idx = idx.to(x_tr.device)
            xb, cb = x_tr[idx], (c_tr[idx] if c_tr is not None else None)
            if family == "vae":
                loss, ll, kl = model(xb, cb, beta=beta)
                sums["recon_ll"] += ll.item(); sums["kl"] += kl.item()
            elif family == "flow":
                loss = model(xb)
            else:
                loss = model(xb, cb if cb is not None else torch.zeros(len(xb), dtype=torch.long, device=xb.device))
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
        row["val_loss"] = _eval_loss(model, family, x_val, c_val)
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            row["heldout_loss"] = _eval_loss(model, family, x_te, c_te)
        history.append(row)
        if family != "vae" or beta >= 1.0:
            if row["val_loss"] < best_val:
                best_val, best_epoch = row["val_loss"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                stop_reason = f"validation loss did not improve for {patience} epochs"
                break
        if (time.time() - t0) / 60 > cap_min:
            stop_reason = f"wall-clock cap {cap_min:.1f} min reached"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:  # cap hit before warm-up ended: keep the last weights
        best_epoch = history[-1]["epoch"]
    return history, best_epoch, stop_reason, patience, _eval_loss(model, family, x_te, c_te)


def _train_gan(model, hp, x_tr, y_tr, cap_min, seed):
    """Hinge-loss GAN (Miyato et al. 2018 setup): one discriminator step per generator step."""
    opt_g = torch.optim.Adam(model.G.parameters(), lr=hp["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=hp["lr"], betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    dev = x_tr.device
    history, t0, stop_reason = [], time.time(), "planned epochs completed (GANs are not early-stopped)"
    epochs = int(hp["epochs"])
    for epoch in range(1, epochs + 1):
        sd, sg, nb = 0.0, 0.0, 0
        for idx in _batches(len(x_tr), hp["batch_size"], gen):
            idx = idx.to(dev)
            xb = x_tr[idx]
            yb = y_tr[idx] if y_tr is not None else torch.zeros(len(idx), dtype=torch.long, device=dev)
            z = torch.randn(len(idx), model.latent_dim, device=dev)
            fake = model.G(z, yb)
            loss_d = F.relu(1 - model.D(xb, yb)).mean() + F.relu(1 + model.D(fake.detach(), yb)).mean()
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            loss_g = -model.D(fake, yb).mean()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            sd += loss_d.item(); sg += loss_g.item(); nb += 1
        if not (math.isfinite(sd) and math.isfinite(sg)):
            raise RuntimeError(f"GAN loss became non-finite at epoch {epoch}")
        history.append({"epoch": epoch, "d_loss": sd / nb, "g_loss": sg / nb})
        if (time.time() - t0) / 60 > cap_min:
            stop_reason = f"wall-clock cap {cap_min:.1f} min reached"
            break
    return history, history[-1]["epoch"], stop_reason


def train(dataset, model_id, seed):
    plan = load_plan(dataset)
    entry = model_entry(plan, model_id)
    prepared = load_prepared(dataset)
    meta = prepared["meta"]
    key = artifact_key(model_id, seed)
    hp, family = entry["hyperparameters"], entry["family"]
    image_shape = tuple(meta["image_shape"]) if meta.get("image_shape") else None
    device = pick_device("image" if image_shape else "tabular")
    conditional = bool(entry.get("conditional"))
    classes = meta.get("classes") or []
    n_classes = len(classes) if conditional else 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Inputs for this model's input_space (PCA input fit on train only) ──
    input_pca = fit_input_pca(prepared["trans_train"], hp["pca_components"], seed) if entry["input_space"] == "pca" else None
    x_all = torch.from_numpy(model_input(entry, prepared, "train", input_pca))          # (n_train, d_in)
    x_te = torch.from_numpy(model_input(entry, prepared, "test", input_pca)).to(device) # (n_test, d_in)
    lookup = {c: i for i, c in enumerate(classes)}
    y_all = torch.tensor([lookup[v] for v in prepared["y_train"]]) if conditional else None
    y_te = torch.tensor([lookup[v] for v in prepared["y_test"]]).to(device) if conditional else None
    n_in = x_all.shape[1]

    # ── Validation slice carved from TRAIN (test stays untouched) ──
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(VAL_FRACTION * len(x_all))))
    va, fi = perm[:n_val], perm[n_val:]
    x_val, x_tr = x_all[va].to(device), x_all[fi].to(device)                          # (n_val, d_in), (n_fit, d_in)
    y_val, y_tr = (y_all[va].to(device), y_all[fi].to(device)) if conditional else (None, None)

    model = build(entry, n_in, n_classes, image_shape).to(device)
    if family == "flow":
        model.set_normalization(x_tr)
    cap_min = max(2.0, 3.0 * float(entry.get("estimated_minutes", 1)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {key}: {family}{' (conditional)' if conditional else ''} on {entry['input_space']} {tuple(x_tr.shape)}, "
          f"{n_params:,} parameters, up to {hp['epochs']} epochs, device={device}")
    t0 = time.time()
    if family == "gan":
        history, best_epoch, stop_reason = _train_gan(model, hp, x_tr, y_tr, cap_min, seed)
        patience, heldout = None, None
    else:
        c = lambda y: _cond(family, conditional, y, n_classes)
        history, best_epoch, stop_reason, patience, heldout = _train_likelihood_family(
            model, family, hp, x_tr, c(y_tr), x_val, c(y_val), x_te, c(y_te), key, cap_min, seed)
    minutes = (time.time() - t0) / 60
    print(f"[train] {key}: stopped at epoch {history[-1]['epoch']} ({stop_reason}); keeping epoch {best_epoch}")

    ck_dir, run_dir = outputs(dataset) / "checkpoints", outputs(dataset) / "runs"
    ck_dir.mkdir(exist_ok=True); run_dir.mkdir(exist_ok=True)
    extra = {"entry": entry, "n_in": n_in, "seed": seed, "input_pca": input_pca, "image_shape": image_shape,
             "n_classes": n_classes, "classes": classes if conditional else None}
    if family == "vae" and hp.get("likelihood") == "nb":
        # Empirical library sizes (total counts) of the training samples; resampled when generating.
        extra["train_library"] = prepared["raw_train"].sum(1).astype(np.float32)
    save_checkpoint(ck_dir / f"{key}.pt", model.cpu(), extra)

    best_row = next((r for r in history if r["epoch"] == best_epoch), history[-1])
    loss_def = {"vae": "negative ELBO per sample", "flow": "negative log-likelihood per sample",
                "diffusion": "noise-prediction MSE (DDPM objective)", "gan": "hinge losses (discriminator, generator)"}[family]
    sidecar = {
        "key": key, "model_id": model_id, "seed": seed, "family": family, "conditional": conditional,
        "config_hash": config_hash(entry, seed, meta["prep_hash"]), "device": device, "n_parameters": n_params,
        "epochs_planned": int(hp["epochs"]), "epochs_run": history[-1]["epoch"], "best_epoch": best_epoch,
        "stopped_early": history[-1]["epoch"] < int(hp["epochs"]), "stop_reason": stop_reason, "patience": patience,
        "time_cap_min": cap_min, "minutes": round(minutes, 3),
        "n_fit": int(len(x_tr)), "n_val": int(len(x_val)), "n_in": n_in,
        "input_pca_variance_explained": input_pca and round(input_pca["evr"], 4),
        "kept_train_loss": best_row.get("loss"), "kept_val_loss": best_row.get("val_loss"),
        "kept_heldout_loss": None if heldout is None else round(heldout, 4),
        "loss_definition": loss_def, "history": history,
    }
    (run_dir / f"{key}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"[train] {key}: done in {minutes:.2f} min")
    return sidecar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one plan.json model at one seed.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train(args.dataset, args.model_id, args.seed)
