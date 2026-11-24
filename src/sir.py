"""SIR malware-propagation model (Ch7 README), batched and differentiable.

Euler integration, dt = 0.1, 500 steps, S0 = 0.99, I0 = 0.01, R0 = 0.  The state is recorded *before* each
update (exactly as in the README code), so the trajectory has 500 entries and peak = max_t I(t).
beta / gamma may be scalars, (B,) tensors, or (B, steps) tensors (time-varying policies)."""
from __future__ import annotations

import torch

DT, STEPS, S0, I0 = 0.1, 500, 0.99, 0.01


def simulate(beta, gamma, S0: float = S0, I0: float = I0, dt: float = DT, steps: int = STEPS):
    beta = torch.as_tensor(beta, dtype=torch.float32)
    gamma = torch.as_tensor(gamma, dtype=torch.float32)
    tv_b = beta.ndim >= 1 and beta.shape[-1] == steps and beta.ndim == 2
    tv_g = gamma.ndim >= 1 and gamma.shape[-1] == steps and gamma.ndim == 2
    bshape = torch.broadcast_shapes(beta.shape[:-1] if tv_b else beta.shape, gamma.shape[:-1] if tv_g else gamma.shape)
    dev = beta.device
    S = torch.full(bshape, S0, device=dev); I = torch.full(bshape, I0, device=dev); R = torch.zeros(bshape, device=dev)
    hS, hI, hR = [], [], []
    for t in range(steps):
        b = beta[..., t] if tv_b else beta
        g = gamma[..., t] if tv_g else gamma
        hS.append(S); hI.append(I); hR.append(R)
        inf = b * S * I
        rec = g * I
        S = S - dt * inf
        I = I + dt * (inf - rec)
        R = R + dt * rec
    return torch.stack(hS, -1), torch.stack(hI, -1), torch.stack(hR, -1)


def peak_infection(beta, gamma, **kw):
    _, I, _ = simulate(beta, gamma, **kw)
    return I.max(-1).values


@torch.no_grad()
def peak_only(beta, gamma, S0: float = S0, I0: float = I0, dt: float = DT, steps: int = STEPS):
    """Running-max version of simulate() for large brute-force grids (no trajectory stored)."""
    beta = torch.as_tensor(beta, dtype=torch.float32)
    gamma = torch.as_tensor(gamma, dtype=torch.float32)
    shape = torch.broadcast_shapes(beta.shape, gamma.shape)
    S = torch.full(shape, S0); I = torch.full(shape, I0)
    peak = I.clone()
    for _ in range(steps):
        peak = torch.maximum(peak, I)
        inf = beta * S * I
        S = S - dt * inf
        I = I + dt * (inf - gamma * I)
    return peak
