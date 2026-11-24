"""Neural *Input* Optimization (Ch7) with a trained detector in the loop.

Policy x = sigmoid(z) in [0,1]^3:  x1 firewall, x2 automated-response aggressiveness, x3 = tau (alert threshold).
  beta  = beta0 * (1 - 0.8 x1)
  gamma = 0.05 + 0.45 * x2 * q * TPR(tau)          (q = 1 for the real detector; q < 1 = hypothetical degradation)
  TPR(tau) = mean_i sigmoid((s_i - tau)/0.02) over held-out malware rows of the type,
  FPR(tau) = same over held-out benign rows (s = detector malware probability)
  L = max_t I(t) + lambda (x1^2 + x2^2) + mu * x2 * FPR(tau)
All constants are illustrative assumptions (as in the Ch7 README), not measured network properties."""
from __future__ import annotations

import time

import numpy as np
import torch

from .sir import STEPS, peak_only, simulate

DEFAULTS = {"beta0": 0.5, "firewall_eff": 0.8, "gamma_base": 0.05, "gamma_gain": 0.45, "temp": 0.02,
            "lam": 0.05, "mu": 0.5, "adam_lr": 0.05, "iters": 300, "z_init": 0.0,
            "default_policy": [0.3, 0.4, 0.5], "grid_n": 21, "tv_windows": 10, "tv_smooth": 0.05}


class ScoreBank:
    """Padded detector scores for a batch of C configurations (malware rows of one type, benign rows)."""

    def __init__(self, mal_list, ben_list):
        self.C = len(mal_list)
        self.mal, self.mal_mask = self._pad(mal_list)
        self.ben, self.ben_mask = self._pad(ben_list)

    @staticmethod
    def _pad(arrs):
        n = max(len(a) for a in arrs)
        out = torch.zeros(len(arrs), n); mask = torch.zeros(len(arrs), n)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = torch.from_numpy(np.array(a, dtype=np.float32)); mask[i, :len(a)] = 1
        return out, mask

    def soft_rates(self, tau: torch.Tensor, temp: float, idx=None):
        """tau: (B,) where row b uses config idx[b] (default: idx = arange(C)), or (C, G) grid."""
        mal, mm, ben, bm = self.mal, self.mal_mask, self.ben, self.ben_mask
        if idx is not None:
            mal, mm, ben, bm = mal[idx], mm[idx], ben[idx], bm[idx]
        if tau.ndim == 1:
            tpr = (torch.sigmoid((mal - tau[:, None]) / temp) * mm).sum(-1) / mm.sum(-1)
            fpr = (torch.sigmoid((ben - tau[:, None]) / temp) * bm).sum(-1) / bm.sum(-1)
        else:  # (C, G)
            tpr = (torch.sigmoid((mal[:, None, :] - tau[..., None]) / temp) * mm[:, None, :]).sum(-1) / mm.sum(-1)[:, None]
            fpr = (torch.sigmoid((ben[:, None, :] - tau[..., None]) / temp) * bm[:, None, :]).sum(-1) / bm.sum(-1)[:, None]
        return tpr, fpr

    def hard_rates(self, tau: np.ndarray, idx):
        tpr, fpr = [], []
        for b, c in enumerate(idx):
            m = self.mal[c][self.mal_mask[c] > 0].numpy(); s = self.ben[c][self.ben_mask[c] > 0].numpy()
            tpr.append(float((m >= tau[b]).mean())); fpr.append(float((s >= tau[b]).mean()))
        return np.array(tpr), np.array(fpr)


def rates(x1, x2, tpr, c, q=1.0):
    beta = c["beta0"] * (1 - c["firewall_eff"] * x1)
    gamma = c["gamma_base"] + c["gamma_gain"] * x2 * q * tpr
    return beta, gamma


def objective(x, bank: ScoreBank, idx, lam, mu, c, q=1.0, return_parts=False):
    """x: (B,3) policies; idx: (B,) config index; lam, mu, q: scalars or (B,) tensors."""
    x1, x2, tau = x[:, 0], x[:, 1], x[:, 2]
    tpr, fpr = bank.soft_rates(tau, c["temp"], idx)
    beta, gamma = rates(x1, x2, tpr, c, q)
    _, I, _ = simulate(beta, gamma)
    peak = I.max(-1).values
    deploy = x1 ** 2 + x2 ** 2
    fq = x2 * fpr
    loss = peak + lam * deploy + mu * fq
    if return_parts:
        return loss, {"peak": peak, "deploy_cost": deploy, "false_quarantine_cost": fq, "tpr": tpr, "fpr": fpr,
                      "beta": beta, "gamma": gamma}
    return loss


def optimize_fixed(bank, idx, lam, mu, c=DEFAULTS, q=1.0, iters=None, lr=None, record=True):
    """Adam on z (sigmoid-constrained) for a batch of independent problems (Adam is element-wise, so this is
    exactly equivalent to running each problem separately)."""
    iters = iters or c["iters"]; lr = lr or c["adam_lr"]
    B = len(idx)
    z = torch.nn.Parameter(torch.full((B, 3), float(c["z_init"])))
    opt = torch.optim.Adam([z], lr=lr)
    hist = {"loss": [], "x": []}
    t0 = time.perf_counter()
    for _ in range(iters):
        x = torch.sigmoid(z)
        loss = objective(x, bank, idx, lam, mu, c, q)
        opt.zero_grad(); loss.sum().backward(); opt.step()
        if record:
            hist["loss"].append(loss.detach().numpy().copy()); hist["x"].append(x.detach().numpy().copy())
    secs = time.perf_counter() - t0
    with torch.no_grad():
        x = torch.sigmoid(z)
        loss, parts = objective(x, bank, idx, lam, mu, c, q, return_parts=True)
    return {"x": x.numpy(), "loss": loss.numpy(), "parts": {k: v.numpy() for k, v in parts.items()},
            "hist_loss": np.array(hist["loss"]), "hist_x": np.array(hist["x"]), "seconds": secs}


@torch.no_grad()
def grid_peaks(bank: ScoreBank, c=DEFAULTS, n=None, q=1.0):
    """Brute force: peak infection for every (x1, x2, tau) on an n^3 grid and every config.
    Returns grid values, peak[C, n, n, n], tpr[C, n], fpr[C, n]."""
    n = n or c["grid_n"]
    g = torch.linspace(0, 1, n)
    taus = g[None, :].repeat(bank.C, 1)
    tpr, fpr = bank.soft_rates(taus, c["temp"])                                   # (C, n)
    x1 = g[:, None, None]; x2 = g[None, :, None]
    beta = (c["beta0"] * (1 - c["firewall_eff"] * x1)).expand(n, n, n)
    peaks = []
    for ci in range(bank.C):  # one config at a time keeps memory small
        gamma = (c["gamma_base"] + c["gamma_gain"] * x2 * q * tpr[ci][None, None, :]).expand(n, n, n)
        peaks.append(peak_only(beta.reshape(-1), gamma.reshape(-1)).reshape(n, n, n))
    return g.numpy(), torch.stack(peaks).numpy(), tpr.numpy(), fpr.numpy()


def grid_optimum(g, peak_c, fpr_c, lam, mu):
    """Best grid policy for one config: loss = peak + lam (x1^2+x2^2) + mu x2 FPR(tau)."""
    x1 = g[:, None, None]; x2 = g[None, :, None]
    loss = peak_c + lam * (x1 ** 2 + x2 ** 2) + mu * x2 * fpr_c[None, None, :]
    i = np.unravel_index(np.argmin(loss), loss.shape)
    return {"x": [float(g[i[0]]), float(g[i[1]]), float(g[i[2]])], "loss": float(loss[i]), "peak": float(peak_c[i])}


def optimize_time_varying(bank, idx, lam, mu, c=DEFAULTS, iters=None, lr=None, z_init=None):
    """x1(t), x2(t) piecewise-constant over W windows, tau fixed.  Cost terms are averaged over windows
    (so a constant schedule has exactly the fixed-policy cost) + smoothness penalty rho * sum (x_{k+1}-x_k)^2."""
    iters = iters or c["iters"]; lr = lr or c["adam_lr"]
    W, rho = c["tv_windows"], c["tv_smooth"]
    B = len(idx)
    if z_init is None:
        z12 = torch.nn.Parameter(torch.full((B, 2, W), float(c["z_init"])))
        z3 = torch.nn.Parameter(torch.full((B,), float(c["z_init"])))
    else:
        z12 = torch.nn.Parameter(torch.as_tensor(z_init[0]).clone().float())
        z3 = torch.nn.Parameter(torch.as_tensor(z_init[1]).clone().float())
    opt = torch.optim.Adam([z12, z3], lr=lr)
    rep = STEPS // W
    hist = []

    def evaluate(z12, z3):
        x12 = torch.sigmoid(z12); tau = torch.sigmoid(z3)
        tpr, fpr = bank.soft_rates(tau, c["temp"], idx)
        x1 = x12[:, 0].repeat_interleave(rep, -1); x2 = x12[:, 1].repeat_interleave(rep, -1)
        beta = c["beta0"] * (1 - c["firewall_eff"] * x1)
        gamma = c["gamma_base"] + c["gamma_gain"] * x2 * tpr[:, None]
        _, I, _ = simulate(beta, gamma)
        peak = I.max(-1).values
        deploy = (x12[:, 0] ** 2 + x12[:, 1] ** 2).mean(-1)
        fq = x12[:, 1].mean(-1) * fpr
        smooth = ((x12[..., 1:] - x12[..., :-1]) ** 2).sum((-1, -2))
        obj = peak + lam * deploy + mu * fq
        return obj + rho * smooth, obj, {"peak": peak, "deploy_cost": deploy, "false_quarantine_cost": fq,
                                         "smooth": smooth, "I": I, "x12": x12, "tau": tau}

    for _ in range(iters):
        total, obj, _ = evaluate(z12, z3)
        opt.zero_grad(); total.sum().backward(); opt.step()
        hist.append(total.detach().numpy().copy())
    with torch.no_grad():
        total, obj, parts = evaluate(z12, z3)
    return {"total": total.numpy(), "objective": obj.numpy(), "hist_total": np.array(hist),
            **{k: v.numpy() for k, v in parts.items()}}


@torch.no_grad()
def trajectories(x, bank, idx, c=DEFAULTS, q=1.0):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    tpr, fpr = bank.soft_rates(x[:, 2], c["temp"], idx)
    beta, gamma = rates(x[:, 0], x[:, 1], tpr, c, q)
    S, I, R = simulate(beta, gamma)
    return S.numpy(), I.numpy(), R.numpy()
