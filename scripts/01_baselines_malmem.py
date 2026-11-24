"""Part A1 - baseline detectors on CIC-MalMem-2022.

Random stratified split vs group split (all dumps of one binary in one split), binary and 4-class tasks,
LR / RF / PyTorch MLP, 3 seeds; leave-one-malware-type-out (LOTO) detectors; permutation importance.
Also writes the MLP detector scores that Part C consumes (data/scores/, not committed)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_malmem import MALWARE_TYPES, TYPES, dataset_stats, load_raw, make_split, transform_rows
from src.train import LR_CFG, MLP_CFG, RF_CFG, classification_metrics, fit_model, make_model, permutation_importance
from src.utils import DATA, RESULTS, SEEDS, log_runtime, mean_std, save_json

MODELS = ["LR", "RF", "MLP"]
PLUGINS = ["pslist", "dlllist", "handles", "ldrmodules", "malfind", "psxview", "modules", "svcscan", "callbacks"]
SCORES_DIR = DATA / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


def per_type_detection(meta: pd.DataFrame, p_mal: np.ndarray) -> dict:
    """TPR at 0.5 and ROC-AUC (type vs benign) for each malware type in a test set."""
    out = {}
    ben = (meta["Class"] == "Benign").to_numpy()
    out["FPR@0.5"] = float((p_mal[ben] >= 0.5).mean())
    for t in MALWARE_TYPES:
        m = (meta["MalwareType"] == t).to_numpy()
        if m.sum() == 0:
            continue
        y = np.r_[np.ones(m.sum()), np.zeros(ben.sum())]
        s = np.r_[p_mal[m], p_mal[ben]]
        out[t] = {"TPR@0.5": float((p_mal[m] >= 0.5).mean()), "AUC_vs_benign": float(roc_auc_score(y, s)), "n": int(m.sum())}
    return out


def main():
    t_all = time.perf_counter()
    df = load_raw()
    stats = dataset_stats(df)
    records, loto_records, importances, split_sizes, plugin_records = [], [], [], [], []

    for seed in SEEDS:
        for kind in ["random", "group"]:
            split = make_split(df, kind, seed)
            split_sizes.append({"seed": seed, "split": kind, "train": len(split.X_train), "val": len(split.X_val),
                                "test": len(split.X_test),
                                "test_type_counts": split.meta_test["MalwareType"].value_counts().to_dict()})
            for task, k in [("binary", 2), ("4class", 4)]:
                ytr, yva, yte = split.y("train", task), split.y("val", task), split.y("test", task)
                for name in MODELS:
                    model = make_model(name, k, seed)
                    fit_s = fit_model(model, split.X_train, ytr, split.X_val, yva)
                    proba = model.predict_proba(split.X_test)
                    rec = {"seed": seed, "split": kind, "task": task, "model": name, "fit_seconds": fit_s,
                           **classification_metrics(yte, proba, k)}
                    if task == "binary":
                        rec["per_type"] = per_type_detection(split.meta_test, proba[:, 1])
                    records.append(rec)
                    print(f"seed={seed} {kind:6s} {task:6s} {name:3s} acc={rec['accuracy']:.4f} "
                          f"f1={rec['macro_f1']:.4f} ({fit_s:.1f}s)")
                    if kind == "group" and task == "binary" and name == "MLP":
                        sc = split.meta_test[["Class", "MalwareType", "Family", "Group"]].copy()
                        sc["p_malware"] = proba[:, 1]
                        sc.to_csv(SCORES_DIR / f"known_seed{seed}.csv", index=False)
                    if kind == "group" and task == "4class" and name in ("RF", "MLP"):
                        base, imp_m, imp_s = permutation_importance(model.predict, split.X_test, yte, n_repeats=5, seed=seed)
                        for f, m_, s_ in zip(split.features, imp_m, imp_s):
                            importances.append({"seed": seed, "model": name, "task": task, "feature": f,
                                                "drop_macro_f1_mean": m_, "drop_macro_f1_std": s_, "base_macro_f1": base})

        # leave-one-malware-type-out detectors (group split, binary)
        for unseen in MALWARE_TYPES:
            seen = ["Benign"] + [t for t in MALWARE_TYPES if t != unseen]
            split = make_split(df, "group", seed, train_mask_types=seen)
            ytr, yva = split.y("train", "binary"), split.y("val", "binary")
            # evaluate on test-split rows of the unseen type + test benign; also on *all* rows of the unseen type
            all_unseen = df[df["MalwareType"] == unseen]
            X_all_unseen = transform_rows(all_unseen, split)
            for name in MODELS:
                model = make_model(name, 2, seed)
                fit_s = fit_model(model, split.X_train, ytr, split.X_val, yva)
                p_test = model.predict_proba(split.X_test)[:, 1]
                p_all_unseen = model.predict_proba(X_all_unseen)[:, 1]
                meta = split.meta_test
                ben = (meta["Class"] == "Benign").to_numpy()
                um = (meta["MalwareType"] == unseen).to_numpy()
                seen_mal = ((meta["Class"] == "Malware") & (meta["MalwareType"] != unseen)).to_numpy()
                y_u = np.r_[np.ones(um.sum()), np.zeros(ben.sum())]
                rec = {"seed": seed, "unseen_type": unseen, "model": name, "fit_seconds": fit_s,
                       "unseen_TPR@0.5_test": float((p_test[um] >= 0.5).mean()),
                       "unseen_AUC_test": float(roc_auc_score(y_u, np.r_[p_test[um], p_test[ben]])),
                       "unseen_TPR@0.5_allrows": float((p_all_unseen >= 0.5).mean()),
                       "unseen_AUC_allrows": float(roc_auc_score(np.r_[np.ones(len(p_all_unseen)), np.zeros(ben.sum())],
                                                                   np.r_[p_all_unseen, p_test[ben]])),
                       "seen_TPR@0.5_test": float((p_test[seen_mal] >= 0.5).mean()),
                       "FPR@0.5_test": float((p_test[ben] >= 0.5).mean()),
                       "n_unseen_test": int(um.sum()), "n_unseen_all": int(len(p_all_unseen))}
                # macro-F1 / accuracy on benign + unseen-type test rows (binary)
                yb = np.r_[np.ones(um.sum()), np.zeros(ben.sum())].astype(int)
                pb = np.r_[p_test[um], p_test[ben]]
                m = classification_metrics(yb, np.c_[1 - pb, pb], 2)
                rec.update({"unseen_accuracy_test": m["accuracy"], "unseen_macro_f1_test": m["macro_f1"]})
                loto_records.append(rec)
                print(f"seed={seed} LOTO unseen={unseen:10s} {name:3s} TPR={rec['unseen_TPR@0.5_test']:.4f} "
                      f"AUC={rec['unseen_AUC_test']:.4f} FPR={rec['FPR@0.5_test']:.4f}")
                if name == "MLP":
                    sc = meta.loc[ben | um, ["Class", "MalwareType", "Family", "Group"]].copy()
                    sc["p_malware"] = p_test[ben | um]
                    sc.to_csv(SCORES_DIR / f"loto_{unseen}_seed{seed}.csv", index=False)

        # single-telemetry-source detectors: one MLP per Volatility plugin (group split, binary).  Answers
        # "which memory-forensics plugin alone suffices?" and yields a genuinely weaker detector for Part C.
        split = make_split(df, "group", seed)
        ytr, yva, yte = split.y("train", "binary"), split.y("val", "binary"), split.y("test", "binary")
        for plugin in PLUGINS:
            ix = [j for j, f in enumerate(split.features) if f.split(".")[0] == plugin]
            if not ix:
                continue
            model = make_model("MLP", 2, seed)
            fit_s = fit_model(model, split.X_train[:, ix], ytr, split.X_val[:, ix], yva)
            p_val = model.predict_proba(split.X_val[:, ix])[:, 1]
            p_te = model.predict_proba(split.X_test[:, ix])[:, 1]
            ben = yte == 0
            rec = {"seed": seed, "plugin": plugin, "n_features": len(ix), "fit_seconds": fit_s,
                   "val_AUC": float(roc_auc_score(yva, p_val)), "test_AUC": float(roc_auc_score(yte, p_te)),
                   "test_TPR@0.5": float((p_te[~ben] >= 0.5).mean()), "test_FPR@0.5": float((p_te[ben] >= 0.5).mean()),
                   "test_accuracy": float(((p_te >= 0.5) == (yte == 1)).mean())}
            plugin_records.append(rec)
            print(f"seed={seed} plugin={plugin:10s} ({len(ix):2d} feats) val_AUC={rec['val_AUC']:.4f} test_AUC={rec['test_AUC']:.4f} "
                  f"TPR={rec['test_TPR@0.5']:.3f} FPR={rec['test_FPR@0.5']:.3f}")
            sc = split.meta_test[["Class", "MalwareType", "Family", "Group"]].copy()
            sc["p_malware"] = p_te
            sc.to_csv(SCORES_DIR / f"plugin_{plugin}_seed{seed}.csv", index=False)

    # ---- summaries ----
    rec_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("confusion", "per_type")} for r in records])
    metric_cols = ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc", "roc_auc_ovr_macro", "fit_seconds"]
    summ = []
    for (kind, task, name), g in rec_df.groupby(["split", "task", "model"]):
        row = {"split": kind, "task": task, "model": name}
        for c in metric_cols:
            if c in g and g[c].notna().all():
                ms = mean_std(g[c]); row[f"{c}_mean"] = ms["mean"]; row[f"{c}_std"] = ms["std"]
        summ.append(row)
    pd.DataFrame(summ).to_csv(RESULTS / "a1_summary.csv", index=False)

    # per-type detection for the group-split binary detectors (the "known-type" detectors)
    pt = []
    for r in records:
        if r["task"] == "binary":
            for t in MALWARE_TYPES:
                pt.append({"seed": r["seed"], "split": r["split"], "model": r["model"], "type": t,
                           "TPR@0.5": r["per_type"][t]["TPR@0.5"], "AUC_vs_benign": r["per_type"][t]["AUC_vs_benign"],
                           "FPR@0.5": r["per_type"]["FPR@0.5"]})
    pt_df = pd.DataFrame(pt)
    pt_df.to_csv(RESULTS / "a1_per_type_detection.csv", index=False)

    loto_df = pd.DataFrame(loto_records)
    loto_df.to_csv(RESULTS / "a1_loto.csv", index=False)
    lsum = []
    for (u, name), g in loto_df.groupby(["unseen_type", "model"]):
        row = {"unseen_type": u, "model": name}
        for c in ["unseen_TPR@0.5_test", "unseen_AUC_test", "unseen_TPR@0.5_allrows", "unseen_AUC_allrows",
                  "seen_TPR@0.5_test", "FPR@0.5_test", "unseen_macro_f1_test"]:
            ms = mean_std(g[c]); row[f"{c}_mean"] = ms["mean"]; row[f"{c}_std"] = ms["std"]
        k = pt_df[(pt_df.split == "group") & (pt_df.model == name) & (pt_df.type == u)]
        ms = mean_std(k["TPR@0.5"]); row["known_TPR@0.5_mean"] = ms["mean"]; row["known_TPR@0.5_std"] = ms["std"]
        ms = mean_std(k["AUC_vs_benign"]); row["known_AUC_mean"] = ms["mean"]; row["known_AUC_std"] = ms["std"]
        lsum.append(row)
    pd.DataFrame(lsum).to_csv(RESULTS / "a1_loto_summary.csv", index=False)

    imp_df = pd.DataFrame(importances)
    imp_df.to_csv(RESULTS / "a1_perm_importance_raw.csv", index=False)
    imp_sum = (imp_df.groupby(["model", "feature"])["drop_macro_f1_mean"].agg(["mean", "std"]).reset_index()
               .sort_values(["model", "mean"], ascending=[True, False]))
    imp_sum.to_csv(RESULTS / "a1_perm_importance.csv", index=False)

    conf = {}
    for r in records:
        key = f"{r['split']}|{r['task']}|{r['model']}"
        conf.setdefault(key, []).append(r["confusion"])
    conf = {k: {"per_seed": v, "sum": np.sum(np.array(v), axis=0).tolist()} for k, v in conf.items()}
    save_json({"labels_4class": TYPES, "labels_binary": ["Benign", "Malware"], "confusion": conf}, RESULTS / "a1_confusion.json")
    pl = pd.DataFrame(plugin_records)
    pl.to_csv(RESULTS / "a1_plugin_detectors.csv", index=False)
    plsum = pl.groupby("plugin").agg(n_features=("n_features", "first"), val_AUC_mean=("val_AUC", "mean"),
                                     test_AUC_mean=("test_AUC", "mean"), test_AUC_std=("test_AUC", "std"),
                                     TPR_mean=("test_TPR@0.5", "mean"), TPR_std=("test_TPR@0.5", "std"),
                                     FPR_mean=("test_FPR@0.5", "mean"), FPR_std=("test_FPR@0.5", "std"),
                                     acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", "std")).reset_index()
    plsum = plsum.sort_values("val_AUC_mean")
    # Part C "weak detector": the weakest *informative* plugin detector, chosen on validation AUC (> 0.6) only
    informative = plsum[plsum.val_AUC_mean > 0.6]
    plsum["selected_as_weak_detector"] = plsum.plugin == informative.iloc[0]["plugin"]
    plsum.to_csv(RESULTS / "a1_plugin_detectors_summary.csv", index=False)
    save_json(records, RESULTS / "a1_records.json")
    save_json({"dataset": stats, "split_sizes": split_sizes,
               "hyperparameters": {"MLP": MLP_CFG, "RF": RF_CFG, "LR": LR_CFG}}, RESULTS / "a1_dataset_stats.json")
    dt = time.perf_counter() - t_all
    log_runtime("A1_baselines_malmem", dt, "3 seeds x {random,group} x {binary,4class} x {LR,RF,MLP} + LOTO + perm. importance")
    print(f"A1 done in {dt:.0f}s")


if __name__ == "__main__":
    main()
