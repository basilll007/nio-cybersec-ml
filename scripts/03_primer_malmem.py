"""Part B1 - Neural Inverse Optimization (Ch10 data primer) against a released MalMem MLP detector.

Threat model (as in Ch10): the auditor knows the architecture, theta_0, the training recipe (T full-batch SGD
steps, lr) and the released weights theta_ref, but not the training data.  NIO learns P synthetic rows whose
T-step SGD training from theta_0 reproduces theta_ref.  We then ask (a) how useful the primer is as training
data (test accuracy of a twin retrained on it) versus a random real subset / Gaussian noise / the full data,
and (b) whether the primer leaks individual training records or only class-level structure."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data_malmem import TYPES, load_raw, make_split
from src.leakage import class_structure, memorised_control, nn_distances, nn_summary, pca_projection, reference_subset
from src.models import make_mlp, param_dict
from src.nio_inverse import Primer, evaluate_params, one_hot, relative_weight_error, run_nio, sgd_unroll, train_reference
from src.utils import RESULTS, SEEDS, log_runtime, mean_std, save_json, set_seed

CFG = {
    "lr_inner": 0.5,          # SGD lr for reference AND twin (same recipe, as in Ch10)
    "nio_lr": 0.1,            # Adam lr on the primer (Ch10 used 0.1)
    "nio_iters": 300,
    "P_binary": [2, 4, 8, 16, 32, 64],
    "P_4class": [4, 8, 16, 32, 64],
    "T_values": [10, 20, 40],
    "T_main": 20,
    "primer_init": "N(0, I) in standardised feature space",
    "nn_test": "DCR: fixed train reference subset S with |S| = |held-out test set|",
    "memorised_control_noise_sd": [0.005, 0.05],
    "n_control_rows": 64,
    "split": "group",
    "device_nio": "cpu",
}


def balanced_subset(y: np.ndarray, P: int, k: int, rng) -> np.ndarray:
    idx = []
    for i in range(P):
        c = i % k
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool))
    # avoid duplicates
    idx = list(dict.fromkeys(idx))
    while len(idx) < P:
        c = len(idx) % k
        cand = rng.choice(np.flatnonzero(y == c))
        if cand not in idx:
            idx.append(cand)
    return np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test (few iterations)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.quick:
        cfg.update({"nio_iters": 5, "P_binary": [2, 16], "P_4class": [4], "T_values": [10, 20]})
    torch.set_num_threads(4)
    dev = torch.device(cfg["device_nio"])
    t_all = time.perf_counter()
    df = load_raw()

    runs, curves, nn_rows, controls, structure, pca_rows = [], [], [], [], [], []
    pca_var = None
    for seed in args.seeds:
        set_seed(seed)
        split = make_split(df, cfg["split"], seed)
        Xtr_np, Xva_np, Xte_np = split.X_train, split.X_val, split.X_test
        Xtr, Xte = torch.tensor(Xtr_np, device=dev), torch.tensor(Xte_np, device=dev)
        d = Xtr.shape[1]
        rng = np.random.default_rng(seed)
        X_S = Xtr_np[reference_subset(len(Xtr_np), len(Xte_np), seed)]  # size-matched train reference set

        # calibration controls for the nearest-neighbour (DCR) membership test
        nc = cfg["n_control_rows"]
        ctrl_sets = {
            "memorised near-copies sd=0.005 (train)": memorised_control(X_S, nc, seed, 0.005),
            "perturbed copies sd=0.05 (train)": memorised_control(X_S, nc, seed, 0.05),
            "nonmember_rows (val)": Xva_np[rng.choice(len(Xva_np), nc, replace=False)],
            "gaussian_noise": rng.standard_normal((nc, d)).astype(np.float32),
        }
        for name, pts in ctrl_sets.items():
            dtr, dho = nn_distances(pts, X_S, Xte_np)
            controls.append({"seed": seed, "set": name, **nn_summary(dtr, dho)})
            for a, b in zip(dtr, dho):
                nn_rows.append({"seed": seed, "task": "-", "T": 0, "P": 0, "variant": "control", "set": name,
                                "d_train": a, "d_holdout": b})

        for task, k in [("binary", 2), ("4class", 4)]:
            ytr_np, yte_np = split.y("train", task), split.y("test", task)
            ytr = torch.tensor(ytr_np, device=dev)
            model = make_mlp(d, k, seed, dev)
            theta0 = param_dict(model)
            Ts = cfg["T_values"] if task == "binary" else [cfg["T_main"]]
            Ps = cfg["P_binary"] if task == "binary" else cfg["P_4class"]
            for T in Ts:
                t0 = time.perf_counter()
                theta_ref = train_reference(model, theta0, Xtr, ytr, k, T, cfg["lr_inner"])
                ref_secs = time.perf_counter() - t0
                ref_m = evaluate_params(model, theta_ref, Xte, yte_np, k)
                init_m = evaluate_params(model, theta0, Xte, yte_np, k)
                for P in Ps:
                    # ---- baselines at the same P (retrained from theta_0 for T steps) ----
                    sub_idx = balanced_subset(ytr_np, P, k, rng)
                    sub_params = sgd_unroll(model, theta0, Xtr[sub_idx], one_hot(ytr[sub_idx], k), T, cfg["lr_inner"])
                    sub_m = evaluate_params(model, sub_params, Xte, yte_np, k)
                    noise = Primer(P, d, k, seed).to(dev)  # identical to the NIO starting point
                    noise_params = sgd_unroll(model, theta0, noise.x.detach(), noise.labels(), T, cfg["lr_inner"])
                    noise_m = evaluate_params(model, noise_params, Xte, yte_np, k)
                    ndtr, ndho = nn_distances(noise.x.detach().cpu().numpy(), X_S, Xte_np)
                    noise_nn = nn_summary(ndtr, ndho)
                    st_noise = class_structure(noise.x.detach().cpu().numpy(), noise.labels().cpu().numpy(), Xtr_np, ytr_np, split.features)
                    st_sub = class_structure(Xtr_np[sub_idx], np.eye(k)[ytr_np[sub_idx]], Xtr_np, ytr_np, split.features)

                    variants = ["fixed"]
                    if task == "binary" and T == cfg["T_main"]:
                        variants.append("soft")
                    for variant in variants:
                        res = run_nio(model, theta0, theta_ref, d, k, P, T, cfg["lr_inner"], iters=cfg["nio_iters"],
                                      nio_lr=cfg["nio_lr"], seed=seed, learn_labels=(variant == "soft"), device=dev)
                        twin_m = evaluate_params(model, res["params"], Xte, yte_np, k)
                        px, py = res["x"].cpu().numpy(), res["y"].cpu().numpy()
                        dtr, dho = nn_distances(px, X_S, Xte_np)
                        nn = nn_summary(dtr, dho)
                        st = class_structure(px, py, Xtr_np, ytr_np, split.features)
                        row = {"seed": seed, "task": task, "T": T, "P": P, "variant": variant,
                               "nio_seconds": res["seconds"], "ref_train_seconds": ref_secs,
                               "nio_rel_weight_error": res["final_loss"] / res["n_tensors"],
                               "nio_rel_weight_error_iter0": res["losses"][0] / res["n_tensors"],
                               "subset_rel_weight_error": relative_weight_error(sub_params, theta_ref, theta0),
                               "noise_rel_weight_error": relative_weight_error(noise_params, theta_ref, theta0),
                               "primer_max_abs_feature": float(np.abs(px).max()),
                               "primer_mean_norm": float(np.linalg.norm(px, axis=1).mean()),
                               "real_train_mean_norm": float(np.linalg.norm(Xtr_np, axis=1).mean()),
                               "soft_label_entropy": float(-(py * np.log(py + 1e-12)).sum(1).mean())}
                        for pref, m in [("primer", twin_m), ("subset", sub_m), ("noise", noise_m),
                                        ("fulldata", ref_m), ("theta0", init_m)]:
                            for kk, vv in m.items():
                                row[f"{pref}_{kk}"] = vv
                        for kk, vv in nn.items():
                            row[f"nn_{kk}"] = vv
                        for kk, vv in noise_nn.items():
                            row[f"noise_nn_{kk}"] = vv
                        row["struct_diff_cosine"] = st["diff_cosine_mean"]
                        row["struct_top15_overlap"] = st["top_overlap_mean"]
                        row["struct_class_mean_cosine"] = float(np.mean(st["class_mean_cosine"]))
                        row["noise_struct_diff_cosine"] = st_noise["diff_cosine_mean"]
                        row["noise_struct_top15_overlap"] = st_noise["top_overlap_mean"]
                        row["subset_struct_diff_cosine"] = st_sub["diff_cosine_mean"]
                        row["subset_struct_top15_overlap"] = st_sub["top_overlap_mean"]
                        runs.append(row)
                        curves.append({"seed": seed, "task": task, "T": T, "P": P, "variant": variant,
                                       "losses": res["losses"], "raw_losses": res["raw_losses"]})
                        structure.append({"seed": seed, "task": task, "T": T, "P": P, "variant": variant, **st})
                        for a, b in zip(dtr, dho):
                            nn_rows.append({"seed": seed, "task": task, "T": T, "P": P, "variant": variant,
                                            "set": "nio_primer", "d_train": a, "d_holdout": b})
                        if variant == "fixed":
                            for a, b in zip(ndtr, ndho):
                                nn_rows.append({"seed": seed, "task": task, "T": T, "P": P, "variant": variant,
                                                "set": "noise_primer", "d_train": a, "d_holdout": b})
                        print(f"seed={seed} {task} T={T:2d} P={P:2d} {variant:5s} relerr={row['nio_rel_weight_error']:.3f} "
                              f"acc primer={twin_m['accuracy']:.4f} subset={sub_m['accuracy']:.4f} "
                              f"noise={noise_m['accuracy']:.4f} full={ref_m['accuracy']:.4f} "
                              f"NNratio={nn['ratio_median']:.3f} ({res['seconds']:.1f}s)", flush=True)
                        if seed == args.seeds[0] and task == "binary" and T == cfg["T_main"] and variant == "fixed":
                            pca_rows.append(("nio_primer_P%d" % P, px, py.argmax(1)))
                            if P == 16:
                                pca_rows.append(("noise_primer_P16", noise.x.detach().cpu().numpy(),
                                                 noise.labels().argmax(1).cpu().numpy()))
                if seed == args.seeds[0] and task == "binary" and T == cfg["T_main"] and pca_rows and pca_var is None:
                    rows, pca_var = pca_projection(Xtr_np, ytr_np, {n: (p, l) for n, p, l in pca_rows}, seed=seed)
                    pd.DataFrame(rows).to_csv(RESULTS / "b1_pca.csv", index=False)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(RESULTS / "b1_runs.csv", index=False)
    pd.DataFrame(nn_rows).to_csv(RESULTS / "b1_nn_distances.csv", index=False)
    pd.DataFrame(controls).to_csv(RESULTS / "b1_nn_controls.csv", index=False)
    save_json(curves, RESULTS / "b1_curves.json")
    save_json(structure, RESULTS / "b1_structure.json")

    # grouped summary (mean +- std over seeds)
    metric_cols = [c for c in runs_df.columns if c not in ("seed", "task", "T", "P", "variant") and
                   pd.api.types.is_numeric_dtype(runs_df[c])]
    summ = []
    for key, g in runs_df.groupby(["task", "T", "P", "variant"]):
        row = dict(zip(["task", "T", "P", "variant"], key))
        row["n_seeds"] = len(g)
        for c in metric_cols:
            if g[c].notna().all():
                row[f"{c}_mean"] = float(g[c].mean()); row[f"{c}_std"] = float(g[c].std(ddof=0))
        summ.append(row)
    pd.DataFrame(summ).to_csv(RESULTS / "b1_summary.csv", index=False)

    # pooled NN test per (task, T, variant) across P and seeds, and per (task,T,P,variant) pooled over seeds
    from src.leakage import nn_summary as _ns
    nn_df = pd.DataFrame(nn_rows)
    pooled = []
    for key, g in nn_df.groupby(["task", "T", "P", "variant", "set"]):
        pooled.append({**dict(zip(["task", "T", "P", "variant", "set"], key)), **_ns(g["d_train"].to_numpy(), g["d_holdout"].to_numpy())})
    for key, g in nn_df[nn_df.variant != "control"].groupby(["task", "T", "variant", "set"]):
        pooled.append({**dict(zip(["task", "T", "variant", "set"], key)), "P": "all",
                       **_ns(g["d_train"].to_numpy(), g["d_holdout"].to_numpy())})
    pd.DataFrame(pooled).to_csv(RESULTS / "b1_nn_pooled.csv", index=False)

    # timing: NIO vs normal training
    a1 = RESULTS / "a1_summary.csv"
    a1_mlp = None
    if a1.exists():
        a = pd.read_csv(a1)
        r = a[(a.split == "group") & (a.task == "binary") & (a.model == "MLP")]
        if len(r):
            a1_mlp = float(r["fit_seconds_mean"].iloc[0])
    timing = []
    for (task, T), g in runs_df[runs_df.variant == "fixed"].groupby(["task", "T"]):
        timing.append({"task": task, "T": T, "nio_seconds_mean_over_P": float(g["nio_seconds"].mean()),
                       "nio_seconds_by_P": g.groupby("P")["nio_seconds"].mean().to_dict(),
                       "reference_fullbatch_T_steps_seconds": float(g["ref_train_seconds"].mean()),
                       "a1_mlp_adam_training_seconds_gpu": a1_mlp})
    save_json({"config": cfg, "timing": timing, "pca_explained_variance": pca_var}, RESULTS / "b1_meta.json")
    dt = time.perf_counter() - t_all
    log_runtime("B1_primer_malmem" + ("_quick" if args.quick else ""), dt,
                f"{len(runs)} NIO runs; P sweep x T sweep x seeds; binary + 4-class; soft-label variant")
    print(f"B1 done in {dt:.0f}s")


if __name__ == "__main__":
    main()
