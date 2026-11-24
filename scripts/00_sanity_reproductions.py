"""Sanity checks before extending the course code:
 (1) Ch10 NIO Dataset Primer toy 'abc' experiment (README attempt 4, verbatim logic, CPU, seed 42);
 (2) Ch7 NIO Malware Dynamics policy optimisation (README NIO block, verbatim logic, CPU).
Results are compared with the numbers printed in the two READMEs."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from src.utils import RESULTS, log_runtime, save_json


def ch10_abc() -> dict:
    torch.manual_seed(42)
    text = "abc" * 200
    vocab = "abc"
    data = torch.tensor([vocab.index(c) for c in text])
    X = torch.stack([data[i:i + 4] for i in range(len(data) - 4)])
    Y = torch.stack([data[i + 1:i + 5] for i in range(len(data) - 4)])

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(3, 8)
            self.attn = nn.MultiheadAttention(8, 1, batch_first=True)
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            h = self.embed(x) if x.ndim == 2 else x @ self.embed.weight
            mask = torch.triu(torch.ones(h.size(1), h.size(1)), diagonal=1).bool()
            h, _ = self.attn(h, h, h, attn_mask=mask)
            return self.fc(h)

    STEPS, LR = 20, 0.1
    gpt1 = TinyGPT()
    initial_weights = copy.deepcopy(gpt1.state_dict())
    opt = torch.optim.SGD(gpt1.parameters(), lr=LR)
    for _ in range(STEPS):
        loss = F.cross_entropy(gpt1(X).reshape(-1, 3), Y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    ref = {n: p.detach().clone() for n, p in gpt1.named_parameters()}
    gpt2 = TinyGPT(); gpt2.load_state_dict(initial_weights)
    start = {n: p.detach().clone() for n, p in gpt2.named_parameters()}
    primer = nn.Parameter(torch.randn(6, 3))
    nio_opt = torch.optim.Adam([primer], lr=0.1)

    def train_gpt2(probs):
        px = torch.stack([probs[:4], probs[1:5]]); py = torch.stack([probs[1:5], probs[2:6]])
        w = {n: p.clone().requires_grad_(True) for n, p in start.items()}
        for _ in range(STEPS):
            l = -(py * F.log_softmax(functional_call(gpt2, w, (px,)), dim=-1)).sum(-1).mean()
            g = torch.autograd.grad(l, tuple(w.values()), create_graph=True)
            w = {n: p - LR * gg for (n, p), gg in zip(w.items(), g)}
        return w

    losses, primers = [], {}
    t0 = time.perf_counter()
    for it in range(200):
        probs = F.softmax(primer, dim=-1)
        w = train_gpt2(probs)
        loss = sum(((w[n] - ref[n]) ** 2).mean() for n in w)
        nio_opt.zero_grad(); loss.backward(); nio_opt.step()
        losses.append(loss.item())
        if it % 20 == 0:
            primers[it] = "".join(vocab[i] for i in probs.argmax(1))
    secs = time.perf_counter() - t0
    final_probs = F.softmax(primer.detach(), dim=-1)
    fw = train_gpt2(final_probs)
    with torch.no_grad():
        l1 = F.cross_entropy(gpt1(X).reshape(-1, 3), Y.reshape(-1)).item()
        l2 = F.cross_entropy(functional_call(gpt2, fw, (X,)).reshape(-1, 3), Y.reshape(-1)).item()
    readme = {"weight_diff": {0: 0.03518, 20: 0.00559, 40: 0.00390, 60: 0.00227, 80: 0.00138, 100: 0.00111,
                              120: 0.00100, 140: 0.00092, 160: 0.00087, 180: 0.00082},
              "final_primer": "cabcac", "gpt1_eval_loss": 0.86926, "gpt2_eval_loss": 0.91245}
    ours = {"weight_diff": {k: losses[k] for k in readme["weight_diff"]},
            "final_primer": "".join(vocab[i] for i in final_probs.argmax(1)),
            "gpt1_eval_loss": l1, "gpt2_eval_loss": l2,
            "reduction_0_to_180_pct": 100 * (1 - losses[180] / losses[0]), "primer_every_20": primers,
            "nio_seconds": secs, "loss_curve": losses}
    return {"readme": readme, "ours": ours, "torch_version": torch.__version__}


def ch7_policy() -> dict:
    policy = torch.nn.Parameter(torch.tensor([0.3, 0.4]))
    optimizer = torch.optim.Adam([policy], lr=0.02)
    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        firewall, recovery = torch.sigmoid(policy)
        beta = 0.5 * (1 - 0.8 * firewall)
        gamma = 0.1 + 0.4 * recovery
        S, I, R = torch.tensor(0.99), torch.tensor(0.01), torch.tensor(0.0)
        infections = []
        for _t in range(500):
            infections.append(I)
            dS = -beta * S * I; dI = beta * S * I - gamma * I; dR = gamma * I
            S = S + 0.1 * dS; I = I + 0.1 * dI; R = R + 0.1 * dR
        peak = torch.stack(infections).max()
        cost = 0.05 * (firewall ** 2 + recovery ** 2)
        loss = peak + cost
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    readme = {"firewall": 0.470, "recovery": 0.493, "peak_infection_pct": 1.07, "final_loss": 0.0339}
    ours = {"firewall": firewall.item(), "recovery": recovery.item(), "peak_infection_pct": 100 * peak.item(),
            "final_loss": losses[-1], "loss_curve": losses,
            "iter_loss_within_1pct_of_final": next(i for i, l in enumerate(losses) if abs(l - losses[-1]) <= 0.01 * losses[-1])}
    # README Experiment 1: manual policy 0.3 / 0.4, and the two corner cases (both 0, both 1)
    def manual(fw, rc):
        beta, gamma = 0.5 * (1 - 0.8 * fw), 0.1 + 0.4 * rc
        S, I, R = 0.99, 0.01, 0.0; peak = 0.0
        for _ in range(500):
            peak = max(peak, I)
            dS = -beta * S * I; dI = beta * S * I - gamma * I; dR = gamma * I
            S += 0.1 * dS; I += 0.1 * dI; R += 0.1 * dR
        return 100 * peak, 100 * R
    ours["manual_policy_0.3_0.4_peak_pct"], ours["manual_policy_0.3_0.4_final_R_pct"] = manual(0.3, 0.4)
    ours["policy_0_0_peak_pct"], ours["policy_0_0_final_R_pct"] = manual(0.0, 0.0)
    ours["policy_1_1_peak_pct"], ours["policy_1_1_final_R_pct"] = manual(1.0, 1.0)
    return {"readme": readme, "ours": ours}


def main():
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    r10 = ch10_abc()
    print("Ch10 abc:", {k: v for k, v in r10["ours"].items() if k not in ("loss_curve", "primer_every_20")})
    save_json(r10, RESULTS / "sanity_ch10_abc.json")
    r7 = ch7_policy()
    print("Ch7 SIR:", {k: v for k, v in r7["ours"].items() if k != "loss_curve"})
    save_json(r7, RESULTS / "sanity_ch7_sir.json")
    log_runtime("00_sanity_reproductions", time.perf_counter() - t0, "Ch10 abc primer + Ch7 SIR NIO, CPU")


if __name__ == "__main__":
    main()
