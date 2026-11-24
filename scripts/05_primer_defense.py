"""Part B3 - Can the defender make NIO leakage harder?  Privacy/utility trade-off.

The released MalMem binary MLP is retrained with
  * DP-SGD-style updates: per-example gradient clipping (C=1), Gaussian noise sigma*C on the summed lot
    gradient, lot size 256; sigma in {0, 0.5, 1, 2} (spec) plus {4, 8} (extension).  NO privacy accounting
    (no epsilon is claimed) - this is only a heuristic defence.
  * weight decay (full-batch SGD) wd in {0.01, 0.03, 0.1}.
NIO is then run against each defended reference (attacker uses the plain recipe: full-batch SGD, T, lr;
for weight decay we also run an attacker that knows wd).  We report reference utility vs. leakage."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.func import functional_call

from src.data_malmem import load_raw, make_split
from src.leakage import class_structure, nn_distances, nn_summary, reference_subset
from src.models import make_mlp, param_dict
from src.nio_inverse import Primer, evaluate_params, one_hot, relative_weight_error, sgd_unroll, train_reference, \
    weight_match_loss
from src.utils import RESULTS, SEEDS, get_device, log_runtime, save_json, set_seed

CFG = {"T": 20, "lr_inner": 0.5, "nio_lr": 0.1, "nio_iters": 300, "P_values": [16, 64],
       "dp_sigmas": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], "dp_clip": 1.0, "dp_lot": 256,
       "weight_decays": [0.01, 0.03, 0.1], "split": "group", "task": "binary"}


def run_nio_wd(model, theta0, theta_ref, d, k, P, T, lr, iters, nio_lr, seed, wd, dev):
    """Same as run_nio, but the twin's inner SGD uses weight decay `wd` (informed attacker)."""
    primer = Primer(P, d, k, seed).to(dev)
    opt = torch.optim.Adam(primer.parameters(), lr=nio_lr)
    losses = []
    t0 = time.perf_counter()
    for _ in range(iters):
        params = sgd_unroll(model, theta0, primer.x, primer.labels(), T, lr, create_graph=True, weight_decay=wd)
        loss = weight_match_loss(params, theta_ref, theta0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    secs = time.perf_counter() - t0
    x, y = primer.x.detach().clone(), primer.labels().detach().clone()
    params = sgd_unroll(model, theta0, x, y, T, lr, weight_decay=wd)
    return {"x": x, "y": y, "losses": losses, "params": params, "seconds": secs,
            "final_loss": float(weight_match_loss(params, theta_ref, theta0)), "n_tensors": len(theta_ref)}


@torch.no_grad()
def loss_mia_auc(model, params, X_in, y_in, X_out, y_out) -> float:
    """Loss-threshold membership inference on the released model: AUC of (-loss) for members vs non-members."""
    li = F.cross_entropy(functional_call(model, params, (X_in,)), y_in, reduction="none").cpu().numpy()
    lo = F.cross_entropy(functional_call(model, params, (X_out,)), y_out, reduction="none").cpu().numpy()
    return float(roc_auc_score(np.r_[np.ones(len(li)), np.zeros(len(lo))], -np.r_[li, lo]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.quick:
        cfg.update({"nio_iters": 5, "P_values": [16], "dp_sigmas": [0.0, 2.0], "weight_decays": [0.1]})
    torch.set_num_threads(4)
    cpu, gdev = torch.device("cpu"), get_device()
    t_all = time.perf_counter()
    df = load_raw()
    from src.nio_inverse import run_nio

    rows, curves = [], []
    for seed in args.seeds:
        set_seed(seed)
        split = make_split(df, cfg["split"], seed)
        Xtr_np, Xte_np = split.X_train, split.X_test
        ytr_np, yte_np = split.y("train", "binary"), split.y("test", "binary")
        Xtr, Xte, ytr = torch.tensor(Xtr_np), torch.tensor(Xte_np), torch.tensor(ytr_np)
        yte = torch.tensor(yte_np)
        d, k, T = Xtr.shape[1], 2, cfg["T"]
        model = make_mlp(d, k, seed, cpu)
        theta0 = param_dict(model)
        # members for MIA: random train rows, same count as the held-out test set
        rng = np.random.default_rng(seed)
        mi = rng.choice(len(Xtr_np), size=len(Xte_np), replace=False)
        X_S = Xtr_np[reference_subset(len(Xtr_np), len(Xte_np), seed)]

        defences = [("none", {}, None)]
        defences += [(f"dp_sigma={s:g}", {"dp_sigma": s, "clip_norm": cfg["dp_clip"], "lot_size": cfg["dp_lot"]}, None)
                     for s in cfg["dp_sigmas"]]
        defences += [(f"wd={w:g}", {"weight_decay": w}, w) for w in cfg["weight_decays"]]
        # reference training on the GPU when available (per-example gradients for the DP variant)
        gm = make_mlp(d, k, seed, gdev)
        th0_g = {n: p.to(gdev) for n, p in theta0.items()}
        Xtr_g, ytr_g = Xtr.to(gdev), ytr.to(gdev)
        for name, kw, wd in defences:
            t0 = time.perf_counter()
            ref_g = train_reference(gm, th0_g, Xtr_g, ytr_g, k, T, cfg["lr_inner"], seed=seed, **kw)
            ref_secs = time.perf_counter() - t0
            theta_ref = {n: p.detach().to(cpu) for n, p in ref_g.items()}
            ref_m = evaluate_params(model, theta_ref, Xte, yte_np, k)
            mia = loss_mia_auc(model, theta_ref, Xtr[mi], ytr[mi], Xte, yte)
            attackers = [("plain", 0.0)] + ([("knows_wd", wd)] if wd else [])
            for P in cfg["P_values"]:
                for att, att_wd in attackers:
                    if att == "plain":
                        res = run_nio(model, theta0, theta_ref, d, k, P, T, cfg["lr_inner"], iters=cfg["nio_iters"],
                                      nio_lr=cfg["nio_lr"], seed=seed, device=cpu)
                    else:
                        res = run_nio_wd(model, theta0, theta_ref, d, k, P, T, cfg["lr_inner"], cfg["nio_iters"],
                                         cfg["nio_lr"], seed, att_wd, cpu)
                    twin = evaluate_params(model, res["params"], Xte, yte_np, k)
                    px, py = res["x"].numpy(), res["y"].numpy()
                    dtr, dho = nn_distances(px, X_S, Xte_np)
                    nn = nn_summary(dtr, dho)
                    st = class_structure(px, py, Xtr_np, ytr_np, split.features)
                    row = {"seed": seed, "defence": name, "attacker": att, "P": P,
                           "dp_sigma": kw.get("dp_sigma", np.nan), "weight_decay": kw.get("weight_decay", 0.0),
                           "ref_accuracy": ref_m["accuracy"], "ref_macro_f1": ref_m["macro_f1"],
                           "ref_roc_auc": ref_m.get("roc_auc"), "ref_train_seconds": ref_secs,
                           "mia_auc": mia,
                           "nio_rel_weight_error": res["final_loss"] / res["n_tensors"],
                           "twin_accuracy": twin["accuracy"], "twin_macro_f1": twin["macro_f1"],
                           "twin_roc_auc": twin.get("roc_auc"),
                           "nn_ratio_median": nn["ratio_median"], "nn_frac_closer_to_train": nn["frac_closer_to_train"],
                           "nn_mw_p_less": nn["mw_p_less"], "nn_wilcoxon_p_less": nn["wilcoxon_p_less"],
                           "nn_binom_p_greater": nn["binom_p_greater"], "struct_diff_cosine": st["diff_cosine_mean"],
                           "struct_top15_overlap": st["top_overlap_mean"], "nio_seconds": res["seconds"],
                           "d_train": dtr.tolist(), "d_holdout": dho.tolist()}
                    rows.append(row)
                    curves.append({"seed": seed, "defence": name, "attacker": att, "P": P, "losses": res["losses"]})
                    print(f"seed={seed} {name:12s} att={att:8s} P={P:2d} ref_acc={ref_m['accuracy']:.4f} mia={mia:.3f} "
                          f"relerr={row['nio_rel_weight_error']:.3f} twin_acc={twin['accuracy']:.4f} "
                          f"cos={st['diff_cosine_mean']:.3f} NNratio={nn['ratio_median']:.3f}", flush=True)

    df_rows = pd.DataFrame(rows)
    df_rows.drop(columns=["d_train", "d_holdout"]).to_csv(RESULTS / "b3_runs.csv", index=False)
    nn_long = [{"seed": r["seed"], "defence": r["defence"], "attacker": r["attacker"], "P": r["P"], "d_train": a, "d_holdout": b}
               for r in rows for a, b in zip(r["d_train"], r["d_holdout"])]
    pd.DataFrame(nn_long).to_csv(RESULTS / "b3_nn_distances.csv", index=False)
    summ = []
    num = ["ref_accuracy", "ref_macro_f1", "mia_auc", "nio_rel_weight_error", "twin_accuracy", "twin_macro_f1",
           "nn_ratio_median", "nn_frac_closer_to_train", "struct_diff_cosine", "struct_top15_overlap", "nio_seconds"]
    for key, g in df_rows.groupby(["defence", "attacker", "P"], sort=False):
        r = dict(zip(["defence", "attacker", "P"], key))
        r["dp_sigma"] = g["dp_sigma"].iloc[0]; r["weight_decay"] = g["weight_decay"].iloc[0]
        for c in num:
            r[f"{c}_mean"] = float(g[c].mean()); r[f"{c}_std"] = float(g[c].std(ddof=0))
        dtr = np.concatenate([np.array(v) for v in g["d_train"]]); dho = np.concatenate([np.array(v) for v in g["d_holdout"]])
        pooled = nn_summary(dtr, dho)
        r["nn_pooled_ratio_median"] = pooled["ratio_median"]; r["nn_pooled_mw_p_less"] = pooled["mw_p_less"]
        r["nn_pooled_wilcoxon_p_less"] = pooled["wilcoxon_p_less"]; r["nn_pooled_frac_closer_to_train"] = pooled["frac_closer_to_train"]
        r["nn_pooled_binom_p_greater"] = pooled["binom_p_greater"]
        summ.append(r)
    pd.DataFrame(summ).to_csv(RESULTS / "b3_summary.csv", index=False)
    save_json({"config": cfg, "curves": curves}, RESULTS / "b3_meta.json")
    dt = time.perf_counter() - t_all
    log_runtime("B3_primer_defense" + ("_quick" if args.quick else ""), dt,
                f"{len(rows)} NIO runs against DP-style / weight-decay defended references")
    print(f"B3 done in {dt:.0f}s")


if __name__ == "__main__":
    main()
