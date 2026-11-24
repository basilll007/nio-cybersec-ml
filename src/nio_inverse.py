"""Neural *Inverse* Optimization (Ch10 'data primer') for tabular / embedding classifiers.

Formulation (follows the Ch10 README, attempt 4):
  * reference:  theta_ref = SGD_T(theta_0, D_real)            (T full-batch SGD steps, lr eta)
  * twin:       theta_P(primer) = SGD_T(theta_0, D_primer)    (same theta_0, same T, same eta),
                unrolled with torch.autograd.grad(create_graph=True) and torch.func.functional_call
  * NIO loss:   L = sum_l  MSE(theta_P^(l), theta_ref^(l)) / MSE(theta_ref^(l), theta_0^(l))
                (per-tensor normalisation = ||theta_P - theta_ref||^2 / ||theta_ref - theta_0||^2)
  * Adam updates ONLY the primer (inputs, and optionally soft labels); the twin's weights are never
    optimised directly - they are always the result of T SGD steps on the primer.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def soft_ce(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    return -(y_soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def one_hot(y: torch.Tensor, k: int) -> torch.Tensor:
    return F.one_hot(y.long(), k).float()


def sgd_unroll(model: nn.Module, theta0: dict, X: torch.Tensor, Y: torch.Tensor, T: int, lr: float,
               create_graph: bool = False, weight_decay: float = 0.0, return_traj: bool = False):
    """T full-batch SGD steps from theta0 on (X, soft labels Y).  With create_graph=True the result is
    differentiable w.r.t. X and Y (this is the unrolled inner loop of NIO)."""
    params = {n: p.detach().clone().requires_grad_(True) for n, p in theta0.items()}
    traj = []
    for _ in range(T):
        loss = soft_ce(functional_call(model, params, (X,)), Y)
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph)
        params = {n: p - lr * (g + weight_decay * p) for (n, p), g in zip(params.items(), grads)}
        if not create_graph:
            params = {n: p.detach().requires_grad_(True) for n, p in params.items()}
        if return_traj:
            traj.append(loss.item())
    params = params if create_graph else {n: p.detach() for n, p in params.items()}
    return (params, traj) if return_traj else params


def train_reference(model: nn.Module, theta0: dict, X: torch.Tensor, y: torch.Tensor, k: int, T: int, lr: float,
                    weight_decay: float = 0.0, dp_sigma: float | None = None, clip_norm: float = 1.0,
                    lot_size: int = 256, seed: int = 0):
    """Train the reference ('released') model.

    dp_sigma is None  -> plain full-batch SGD (optionally with weight decay).
    dp_sigma >= 0     -> DP-SGD-style: each step samples a lot of `lot_size` rows, clips every per-example
                         gradient to L2 norm `clip_norm`, sums, adds N(0, (dp_sigma*clip_norm)^2) noise per
                         coordinate and divides by the lot size.  No privacy accounting is performed.
    """
    Y = one_hot(y, k)
    if dp_sigma is None:
        return sgd_unroll(model, theta0, X, Y, T, lr, create_graph=False, weight_decay=weight_decay)

    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    params = {n: p.detach().clone() for n, p in theta0.items()}
    names = list(params.keys())

    def loss_one(p, x, yy):
        out = functional_call(model, p, (x.unsqueeze(0),))
        return soft_ce(out, yy.unsqueeze(0))

    per_ex_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
    n = X.shape[0]
    for _ in range(T):
        idx = torch.randperm(n, generator=g)[:lot_size].to(X.device)
        gs = per_ex_grad(params, X[idx], Y[idx])  # dict name -> (L, *shape)
        flat = torch.cat([gs[nm].reshape(lot_size, -1) for nm in names], dim=1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        flat = flat * torch.clamp(clip_norm / norms, max=1.0)
        summed = flat.sum(0)
        if dp_sigma > 0:
            summed = summed + dp_sigma * clip_norm * torch.randn(summed.shape, generator=g).to(X.device)
        upd = summed / lot_size
        off = 0
        for nm in names:
            sz = params[nm].numel()
            gstep = upd[off:off + sz].view_as(params[nm]); off += sz
            params[nm] = params[nm] - lr * (gstep + weight_decay * params[nm])
    return params


def weight_match_loss(params: dict, ref: dict, theta0: dict, normalize: bool = True) -> torch.Tensor:
    total = 0.0
    for n in ref:
        mse = ((params[n] - ref[n]) ** 2).mean()
        if normalize:
            mse = mse / (((ref[n] - theta0[n]) ** 2).mean() + 1e-12)
        total = total + mse
    return total


def balanced_labels(P: int, k: int) -> torch.Tensor:
    return torch.arange(P) % k


class Primer(nn.Module):
    """P learnable rows in (standardised) feature space + fixed balanced labels or learnable soft labels."""

    def __init__(self, P: int, d: int, k: int, seed: int, learn_labels: bool = False, init_scale: float = 1.0):
        super().__init__()
        g = torch.Generator().manual_seed(20_000 + seed)
        self.x = nn.Parameter(init_scale * torch.randn(P, d, generator=g))
        self.learn_labels = learn_labels
        self.register_buffer("y_hard", balanced_labels(P, k))
        if learn_labels:
            self.label_logits = nn.Parameter(3.0 * one_hot(self.y_hard, k))  # starts near the balanced labels
        self.k = k

    def labels(self) -> torch.Tensor:
        if self.learn_labels:
            return F.softmax(self.label_logits, dim=-1)
        return one_hot(self.y_hard, self.k)


def run_nio(model: nn.Module, theta0: dict, theta_ref: dict, d: int, k: int, P: int, T: int, lr_inner: float,
            iters: int = 300, nio_lr: float = 0.05, seed: int = 0, learn_labels: bool = False,
            device=None, log_every: int = 1) -> dict:
    device = device or next(iter(theta0.values())).device
    primer = Primer(P, d, k, seed, learn_labels).to(device)
    opt = torch.optim.Adam(primer.parameters(), lr=nio_lr)
    losses, raw = [], []
    t0 = time.perf_counter()
    for it in range(iters):
        Y = primer.labels()
        params = sgd_unroll(model, theta0, primer.x, Y, T, lr_inner, create_graph=True)
        loss = weight_match_loss(params, theta_ref, theta0, normalize=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            losses.append(loss.item())
            with torch.no_grad():
                raw.append(float(weight_match_loss(params, theta_ref, theta0, normalize=False)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    with torch.no_grad():
        x = primer.x.detach().clone()
        y = primer.labels().detach().clone()
    # final (post-update) twin and its weight error
    params = sgd_unroll(model, theta0, x, y, T, lr_inner, create_graph=False)
    final = float(weight_match_loss(params, theta_ref, theta0, normalize=True))
    return {"x": x, "y": y, "losses": losses, "raw_losses": raw, "final_loss": final, "seconds": secs,
            "n_tensors": len(theta_ref), "params": params}


@torch.no_grad()
def evaluate_params(model: nn.Module, params: dict, X: torch.Tensor, y: np.ndarray, k: int) -> dict:
    from .train import classification_metrics
    logits = functional_call(model, params, (X,))
    proba = F.softmax(logits, dim=-1).cpu().numpy()
    m = classification_metrics(y, proba, k)
    m.pop("confusion", None)
    return m


def relative_weight_error(params: dict, ref: dict, theta0: dict) -> float:
    """Mean over tensors of ||theta - theta_ref||^2 / ||theta_ref - theta_0||^2 (1.0 = no closer than theta_0)."""
    with torch.no_grad():
        return float(weight_match_loss(params, ref, theta0, normalize=True)) / len(ref)
