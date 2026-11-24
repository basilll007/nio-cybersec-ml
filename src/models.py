"""Model definitions.  The MLP is the detector used in Parts A, B and C (55->64->32->k in the spec;
the input width is the number of non-constant MalMem features, 52)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple = (64, 32)):
        super().__init__()
        dims = (in_dim, *hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mlp(in_dim: int, out_dim: int, seed: int, device=None) -> MLP:
    """Deterministic initialisation: the same seed always gives the same theta_0."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    m = MLP(in_dim, out_dim)
    torch.random.set_rng_state(state)
    return m.to(device) if device is not None else m


def param_dict(model: nn.Module, detach: bool = True) -> dict:
    return {n: (p.detach().clone() if detach else p) for n, p in model.named_parameters()}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
