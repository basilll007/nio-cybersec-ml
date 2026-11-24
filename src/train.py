"""Training / evaluation utilities for the Part A detectors."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score

from .models import make_mlp
from .utils import get_device, set_seed

MLP_CFG = {"epochs": 40, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-5, "patience": 6}
RF_CFG = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1}
LR_CFG = {"C": 1.0, "max_iter": 3000, "solver": "lbfgs"}


class TorchMLPClassifier:
    """sklearn-like wrapper around the PyTorch MLP (Adam, early stopping on val loss)."""

    def __init__(self, n_classes: int, seed: int = 0, device=None, **cfg):
        self.k = n_classes
        self.seed = seed
        self.device = device or get_device()
        self.cfg = {**MLP_CFG, **cfg}
        self.model = None
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        set_seed(self.seed)
        d = X.shape[1]
        self.model = make_mlp(d, self.k, self.seed, self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        if X_val is not None:
            Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        n = Xt.shape[0]
        for ep in range(self.cfg["epochs"]):
            self.model.train()
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, self.cfg["batch_size"]):
                b = perm[i:i + self.cfg["batch_size"]]
                loss = F.cross_entropy(self.model(Xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    vl = F.cross_entropy(self.model(Xv), yv).item()
                self.history.append(vl)
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.cfg["patience"]:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, Xt.shape[0], 8192):
            out.append(F.softmax(self.model(Xt[i:i + 8192]), dim=-1).cpu())
        return torch.cat(out).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_model(name: str, n_classes: int, seed: int):
    if name == "LR":
        return LogisticRegression(C=LR_CFG["C"], max_iter=LR_CFG["max_iter"], solver=LR_CFG["solver"], random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=RF_CFG["n_estimators"], max_depth=RF_CFG["max_depth"],
                                      min_samples_leaf=RF_CFG["min_samples_leaf"], n_jobs=-1, random_state=seed)
    if name == "MLP":
        return TorchMLPClassifier(n_classes, seed=seed)
    raise ValueError(name)


def fit_model(model, X, y, X_val=None, y_val=None):
    t0 = time.perf_counter()
    if isinstance(model, TorchMLPClassifier):
        model.fit(X, y, X_val, y_val)
    else:
        model.fit(X, y)
    return time.perf_counter() - t0


def classification_metrics(y_true, proba, n_classes: int) -> dict:
    y_pred = proba.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out


def permutation_importance(predict_fn, X, y, n_repeats: int = 5, seed: int = 0, metric=None):
    """Mean drop in macro-F1 when one feature column is shuffled (test set)."""
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro"))
    rng = np.random.default_rng(seed)
    base = metric(y, predict_fn(X))
    drops = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
            drops[j, r] = base - metric(y, predict_fn(Xp))
    return base, drops.mean(1), drops.std(1)
