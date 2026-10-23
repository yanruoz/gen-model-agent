"""Image evaluation helpers, shared by evaluate_gen.py (image branch) and downstream_test.py.

   Evaluation space for images: the penultimate-layer features (128-d) of a small CNN trained to
   classify the REAL training images. This is a domain-specific stand-in for the Inception network
   behind FID: an ImageNet network is a poor judge of 28x28 histology patches and too heavy for this
   machine. The classifier is trained once per dataset (cached in outputs/<d>/evaluator/), with a
   validation slice from the training split for early stopping; the held-out split is used only to
   report its accuracy.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_io import outputs, pick_device
from models.classifier import SmallCNN

EVAL_SEED = 1234          # fixed: the evaluator must not change between model runs
EVAL_MAX_EPOCHS = 30
EVAL_PATIENCE = 6


def class_index(y, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[v] for v in y], dtype=np.int64)


def train_classifier(x, y, image_shape, n_classes, seed, max_epochs, device, patience=EVAL_PATIENCE,
                     x_val=None, y_val=None, batch_size=128, lr=1e-3):
    """Train SmallCNN on (x, y) with cosine learning-rate decay. If no validation set is given, 10% of x is
    held out. patience=None trains all max_epochs and keeps the final weights (used for the downstream
    test, so every training set gets identical training); otherwise early-stops on validation accuracy.
    Returns (model in eval mode, info dict)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    if x_val is None:
        perm = torch.randperm(len(x), generator=g)
        n_val = max(1, len(x) // 10)
        x_val, y_val, x, y = x[perm[:n_val]], y[perm[:n_val]], x[perm[n_val:]], y[perm[n_val:]]
    x_val, y_val = torch.as_tensor(x_val).to(device), torch.as_tensor(y_val).to(device)
    model = SmallCNN(image_shape, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay over the maximum epochs: a fixed rate left validation accuracy too noisy to early-stop on.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best, best_state, best_epoch, t0 = -1.0, None, 0, time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=g).split(batch_size):
            loss = F.cross_entropy(model(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = accuracy(model, x_val, y_val)
        if patience is None:
            best, best_epoch = acc, epoch
            continue
        if acc > best:
            best, best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), {"val_accuracy": round(best, 4), "best_epoch": best_epoch, "epochs_run": epoch,
                          "minutes": round((time.time() - t0) / 60, 3)}


@torch.no_grad()
def predict(model, x, chunk=1024):
    dev = next(model.parameters()).device
    return torch.cat([model(torch.as_tensor(x[i:i + chunk]).to(dev)).argmax(1).cpu() for i in range(0, len(x), chunk)]).numpy()


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return float((torch.as_tensor(predict(model, x)) == torch.as_tensor(y).cpu()).float().mean())


@torch.no_grad()
def features(model, x, chunk=1024):
    dev = next(model.parameters()).device
    return torch.cat([model.features(torch.as_tensor(x[i:i + chunk]).to(dev)).cpu() for i in range(0, len(x), chunk)]).numpy().astype(np.float64)


def get_evaluator(dataset, prepared):
    """Load the cached evaluator classifier, or train it on the real training split."""
    meta = prepared["meta"]
    d = outputs(dataset) / "evaluator"
    shape, classes = tuple(meta["image_shape"]), meta["classes"]
    ck, info_path = d / "classifier.pt", d / "evaluator.json"
    device = pick_device("image")
    if ck.exists() and info_path.exists() and json.loads(info_path.read_text()).get("prep_hash") == meta["prep_hash"]:
        model = SmallCNN(shape, len(classes))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        return model.to(device).eval(), json.loads(info_path.read_text())
    d.mkdir(exist_ok=True)
    y_tr = class_index(prepared["y_train"], classes)
    model, info = train_classifier(prepared["trans_train"], y_tr, shape, len(classes), EVAL_SEED, EVAL_MAX_EPOCHS, device)
    info.update(prep_hash=meta["prep_hash"], seed=EVAL_SEED, feature_dim=128,
                heldout_accuracy=round(accuracy(model, prepared["trans_test"], class_index(prepared["y_test"], classes)), 4),
                description="SmallCNN (3 conv blocks, 128-d penultimate layer) trained on the real training split; "
                            "its penultimate features define the image evaluation space.")
    torch.save(model.state_dict(), ck)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"[evaluator] trained: validation accuracy {info['val_accuracy']}, held-out accuracy {info['heldout_accuracy']}")
    return model, info


def frechet_distance(a, b):
    """Frechet distance between Gaussians fit to feature sets a (n, k) and b (m, k) (the FID formula)."""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    # Small diagonal term: ReLU features that are always 0 make the covariances singular (standard FID practice).
    eps = 1e-6 * np.eye(a.shape[1])
    s1, s2 = np.cov(a, rowvar=False) + eps, np.cov(b, rowvar=False) + eps
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))
