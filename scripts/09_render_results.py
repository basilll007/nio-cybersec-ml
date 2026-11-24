"""Render RESULTS.md from docs/RESULTS.template.md.

Every number in RESULTS.md is produced here from files in results/ (written by scripts 00-06).  Placeholders:
  {{table:NAME}}  -> a markdown table built by a function in TABLES
  {{NAME}}        -> a scalar formatted by a function in VALUES
Rendering fails loudly if a placeholder cannot be resolved."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import RESULTS, ROOT, load_json

R = RESULTS


def ms(m, s, d=3, pct=False):
    if pct:
        return f"{100 * m:.{d}f} ± {100 * s:.{d}f}"
    return f"{m:.{d}f} ± {s:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


# ------------------------------------------------------------------ loaders (cached)
_c = {}


def csv(name):
    if name not in _c:
        _c[name] = pd.read_csv(R / name)
    return _c[name]


def js(name):
    if name not in _c:
        _c[name] = load_json(R / name)
    return _c[name]


# ------------------------------------------------------------------ tables
def t_sanity_ch10():
    r = js("sanity_ch10_abc.json")
    rows = [{"quantity": f"weight difference @ iter {k}", "README": f"{v:.5f}", "ours": f"{r['ours']['weight_diff'][str(k)]:.5f}"}
            for k, v in [(int(a), b) for a, b in r["readme"]["weight_diff"].items()]]
    rows += [{"quantity": "final primer (argmax)", "README": r["readme"]["final_primer"], "ours": r["ours"]["final_primer"]},
             {"quantity": "GPT 1 eval loss", "README": f"{r['readme']['gpt1_eval_loss']:.5f}", "ours": f"{r['ours']['gpt1_eval_loss']:.5f}"},
             {"quantity": "GPT 2 (primer only) eval loss", "README": f"{r['readme']['gpt2_eval_loss']:.5f}", "ours": f"{r['ours']['gpt2_eval_loss']:.5f}"},
             {"quantity": "reduction iter 0 → 180", "README": "≈ 97.7%", "ours": f"{r['ours']['reduction_0_to_180_pct']:.1f}%"}]
    return md(pd.DataFrame(rows))


def t_sanity_ch7():
    r = js("sanity_ch7_sir.json")
    o, m = r["ours"], r["readme"]
    rows = [{"quantity": "optimised firewall x₁", "README": f"{m['firewall']:.3f}", "ours": f"{o['firewall']:.3f}"},
            {"quantity": "optimised recovery x₂", "README": f"{m['recovery']:.3f}", "ours": f"{o['recovery']:.3f}"},
            {"quantity": "peak infection", "README": f"{m['peak_infection_pct']:.2f}%", "ours": f"{o['peak_infection_pct']:.2f}%"},
            {"quantity": "final loss", "README": f"{m['final_loss']:.4f}", "ours": f"{o['final_loss']:.4f}"},
            {"quantity": "manual policy (0.3, 0.4) peak", "README": "(not printed)", "ours": f"{o['manual_policy_0.3_0.4_peak_pct']:.2f}%"}]
    return md(pd.DataFrame(rows))


def t_dataset():
    st = js("a1_dataset_stats.json")
    d = st["dataset"]
    sz = pd.DataFrame(st["split_sizes"])
    rows = [("rows (memory dumps)", f"{d['rows']:,}"),
            ("features: raw / after dropping constants", f"{d['n_features_raw']} / {d['n_features_kept']}"),
            ("constant columns dropped", ", ".join(f"`{c}`" for c in d["constant_dropped"])),
            ("log1p-transformed count features", str(len(d["count_log1p"]))),
            ("binary classes Benign : Malware", f"{d['class_counts']['Benign']:,} : {d['class_counts']['Malware']:,} (1 : 1)"),
            ("4-class counts Benign / Ransomware / Spyware / Trojan",
             " / ".join(f"{d['type_counts'][t]:,}" for t in ["Benign", "Ransomware", "Spyware", "Trojan"])),
            ("4-class imbalance (largest : smallest)", f"{max(d['type_counts'].values()) / min(d['type_counts'].values()):.2f} : 1"),
            ("malware families", str(len(d["family_counts"]) - 1)),
            ("distinct malware binaries (SHA-256 in Filename)", f"{d['n_malware_binaries']:,}"),
            ("dumps per binary (mean / max)", f"{d['dumps_per_binary_mean']:.2f} / {d['dumps_per_binary_max']}"),
            ("exact duplicate feature rows", f"{d['n_duplicate_feature_rows']:,}")]
    for kind in ["random", "group"]:
        s = sz[sz.split == kind]
        rows.append((f"{kind} split train / val / test (mean over 3 seeds)",
                     f"{s.train.mean():,.0f} / {s.val.mean():,.0f} / {s.test.mean():,.0f}"))
    rows.append(("face-detection fallback rate", "n/a (deepfake track dropped)"))
    return md(pd.DataFrame(rows, columns=["statistic", "value"]))


def t_a1_main():
    s = csv("a1_summary.csv")
    rows = []
    for task in ["binary", "4class"]:
        for split in ["random", "group"]:
            for m in ["LR", "RF", "MLP"]:
                r = s[(s.task == task) & (s.split == split) & (s.model == m)].iloc[0]
                auc = ms(r.roc_auc_mean, r.roc_auc_std, 4) if task == "binary" else ms(r.roc_auc_ovr_macro_mean, r.roc_auc_ovr_macro_std, 4)
                rows.append({"task": task, "split": split, "model": m, "accuracy": ms(r.accuracy_mean, r.accuracy_std, 4),
                             "macro-F1": ms(r.macro_f1_mean, r.macro_f1_std, 4),
                             "ROC-AUC (4-class: OvR macro)": auc})
    return md(pd.DataFrame(rows))


def t_a1_loto():
    l = csv("a1_loto_summary.csv")
    rows = []
    for t in ["Ransomware", "Spyware", "Trojan"]:
        for m in ["LR", "RF", "MLP"]:
            r = l[(l.unseen_type == t) & (l.model == m)].iloc[0]
            rows.append({"unseen type": t, "model": m,
                         "TPR known-type detector": ms(r["known_TPR@0.5_mean"], r["known_TPR@0.5_std"], 4),
                         "TPR unseen type (test rows)": ms(r["unseen_TPR@0.5_test_mean"], r["unseen_TPR@0.5_test_std"], 4),
                         "TPR unseen type (all rows)": ms(r["unseen_TPR@0.5_allrows_mean"], r["unseen_TPR@0.5_allrows_std"], 4),
                         "AUC unseen vs benign": ms(r["unseen_AUC_test_mean"], r["unseen_AUC_test_std"], 4),
                         "FPR (benign test)": ms(r["FPR@0.5_test_mean"], r["FPR@0.5_test_std"], 4)})
    return md(pd.DataFrame(rows))


def t_a1_plugins():
    d = csv("a1_plugin_detectors_summary.csv").sort_values("test_AUC_mean", ascending=False)
    rows = [{"Volatility plugin": r.plugin, "# features": r.n_features, "test ROC-AUC": ms(r.test_AUC_mean, r.test_AUC_std, 4),
             "TPR@0.5": ms(r.TPR_mean, r.TPR_std, 3), "FPR@0.5": ms(r.FPR_mean, r.FPR_std, 3),
             "accuracy": ms(r.acc_mean, r.acc_std, 3), "C1 weak detector": "**yes**" if r.selected_as_weak_detector else ""}
            for _, r in d.iterrows()]
    return md(pd.DataFrame(rows))


def t_a1_importance():
    imp = csv("a1_perm_importance.csv")
    out = []
    rf = imp[imp.model == "RF"].sort_values("mean", ascending=False).head(15).reset_index(drop=True)
    mlp = imp[imp.model == "MLP"].sort_values("mean", ascending=False).head(15).reset_index(drop=True)
    for i in range(15):
        out.append({"rank": i + 1, "RF feature": f"`{rf.feature[i]}`", "RF ΔmacroF1": ms(rf["mean"][i], rf["std"][i], 4),
                    "MLP feature": f"`{mlp.feature[i]}`", "MLP ΔmacroF1": ms(mlp["mean"][i], mlp["std"][i], 4)})
    return md(pd.DataFrame(out))


def _b1(task, T, variant="fixed"):
    s = csv("b1_summary.csv")
    return s[(s.task == task) & (s["T"] == T) & (s.variant == variant)].sort_values("P")


def t_b1_main():
    f = _b1("binary", 20); so = _b1("binary", 20, "soft").set_index("P")
    rows = []
    for _, r in f.iterrows():
        sr = so.loc[r.P]
        rows.append({"P": int(r.P), "NIO primer (fixed labels)": ms(r.primer_accuracy_mean, r.primer_accuracy_std),
                     "NIO primer (soft labels)": ms(sr.primer_accuracy_mean, sr.primer_accuracy_std),
                     "random real subset": ms(r.subset_accuracy_mean, r.subset_accuracy_std),
                     "Gaussian-noise primer": ms(r.noise_accuracy_mean, r.noise_accuracy_std),
                     "full data (θ_ref)": ms(r.fulldata_accuracy_mean, r.fulldata_accuracy_std),
                     "rel. weight error primer": ms(r.nio_rel_weight_error_mean, r.nio_rel_weight_error_std),
                     "rel. weight error subset": ms(r.subset_rel_weight_error_mean, r.subset_rel_weight_error_std)})
    return md(pd.DataFrame(rows))


def t_b1_auc():
    f = _b1("binary", 20)
    rows = [{"P": int(r.P), "primer ROC-AUC": ms(r.primer_roc_auc_mean, r.primer_roc_auc_std, 4),
             "subset ROC-AUC": ms(r.subset_roc_auc_mean, r.subset_roc_auc_std, 4),
             "noise ROC-AUC": ms(r.noise_roc_auc_mean, r.noise_roc_auc_std, 4),
             "full-data ROC-AUC": ms(r.fulldata_roc_auc_mean, r.fulldata_roc_auc_std, 4)} for _, r in f.iterrows()]
    return md(pd.DataFrame(rows))


def t_b1_4class():
    f = _b1("4class", 20)
    rows = [{"P": int(r.P), "NIO primer acc": ms(r.primer_accuracy_mean, r.primer_accuracy_std),
             "NIO primer macro-F1": ms(r.primer_macro_f1_mean, r.primer_macro_f1_std),
             "random subset acc": ms(r.subset_accuracy_mean, r.subset_accuracy_std),
             "noise acc": ms(r.noise_accuracy_mean, r.noise_accuracy_std),
             "full data acc": ms(r.fulldata_accuracy_mean, r.fulldata_accuracy_std),
             "full data macro-F1": ms(r.fulldata_macro_f1_mean, r.fulldata_macro_f1_std),
             "rel. weight error": ms(r.nio_rel_weight_error_mean, r.nio_rel_weight_error_std)} for _, r in f.iterrows()]
    return md(pd.DataFrame(rows))


def t_b1_T():
    s = csv("b1_summary.csv")
    d = s[(s.task == "binary") & (s.variant == "fixed")]
    rows = []
    for T in sorted(d["T"].unique()):
        g = d[d["T"] == T].sort_values("P")
        row = {"T": int(T), "full-data acc": ms(g.fulldata_accuracy_mean.iloc[0], g.fulldata_accuracy_std.iloc[0])}
        for _, r in g.iterrows():
            row[f"P={int(r.P)} acc / rel.err"] = f"{r.primer_accuracy_mean:.3f} / {r.nio_rel_weight_error_mean:.2f}"
        row["NIO s/run"] = f"{g.nio_seconds_mean.mean():.1f}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def t_b1_leak():
    p = csv("b1_nn_pooled.csv")
    rows = []

    def add(label, r):
        rows.append({"point set": label, "n": int(r.n_points), "median d_train": f"{r.d_train_median:.3f}",
                     "median d_held-out": f"{r.d_holdout_median:.3f}", "median ratio": f"{r.ratio_median:.3f}",
                     "share closer to train": f"{r.frac_closer_to_train:.3f}", "binomial p (>0.5)": pfmt(r.binom_p_greater),
                     "Mann–Whitney p": pfmt(r.mw_p_less), "paired Wilcoxon p": pfmt(r.wilcoxon_p_less)})
    allp = p[p.P.astype(str) == "all"]
    for (task, T, var), lab in [(("binary", 10, "fixed"), "NIO primer, binary, T=10 (all P)"),
                                (("binary", 20, "fixed"), "NIO primer, binary, T=20 (all P)"),
                                (("binary", 20, "soft"), "NIO primer soft labels, binary, T=20"),
                                (("binary", 40, "fixed"), "NIO primer, binary, T=40 (all P)"),
                                (("4class", 20, "fixed"), "NIO primer, 4-class, T=20 (all P)")]:
        r = allp[(allp.task == task) & (allp["T"].astype(int) == T) & (allp.variant == var) & (allp.set == "nio_primer")]
        if len(r):
            add(lab, r.iloc[0])
    r = allp[(allp.task == "binary") & (allp["T"].astype(int) == 20) & (allp.set == "noise_primer")]
    if len(r):
        add("Gaussian-noise primer (NIO start), binary T=20", r.iloc[0])
    ctrl = p[p.variant == "control"]
    for s_, lab in [("memorised near-copies sd=0.005 (train)", "**positive control**: near-copies of train rows (sd 0.005)"),
                    ("perturbed copies sd=0.05 (train)", "positive control: perturbed train rows (sd 0.05)"),
                    ("nonmember_rows (val)", "null control: real non-member rows (val)"),
                    ("gaussian_noise", "null control: Gaussian noise")]:
        r = ctrl[ctrl.set == s_]
        if len(r):
            add(lab, r.iloc[0])
    return md(pd.DataFrame(rows))


def t_b1_struct():
    f = _b1("binary", 20)
    f4 = _b1("4class", 20).set_index("P")
    rows = []
    for _, r in f.iterrows():
        row = {"P": int(r.P),
               "primer diff-cosine": ms(r.struct_diff_cosine_mean, r.struct_diff_cosine_std, 2),
               "primer top-15 overlap": ms(r.struct_top15_overlap_mean, r.struct_top15_overlap_std, 1),
               "random-subset diff-cosine": ms(r.subset_struct_diff_cosine_mean, r.subset_struct_diff_cosine_std, 2),
               "random-subset top-15 overlap": ms(r.subset_struct_top15_overlap_mean, r.subset_struct_top15_overlap_std, 1),
               "noise diff-cosine": ms(r.noise_struct_diff_cosine_mean, r.noise_struct_diff_cosine_std, 2),
               "noise top-15 overlap": ms(r.noise_struct_top15_overlap_mean, r.noise_struct_top15_overlap_std, 1)}
        if r.P in f4.index:
            q = f4.loc[r.P]
            row["4-class primer diff-cosine"] = ms(q.struct_diff_cosine_mean, q.struct_diff_cosine_std, 2)
        else:
            row["4-class primer diff-cosine"] = "–"
        rows.append(row)
    return md(pd.DataFrame(rows))


def t_b1_features():
    st = js("b1_structure.json")
    sel = [x for x in st if x["task"] == "binary" and x["T"] == 20 and x["P"] == 16 and x["variant"] == "fixed"]
    rows = []
    for x in sel:
        c = x["contrasts"][0]
        rows.append({"seed": x["seed"], "diff-cosine": f"{c['diff_cosine']:.2f}", "top-15 overlap": c["top_overlap"],
                     "sign agreement on real top-15": f"{c['sign_agreement_on_real_top']:.2f}",
                     "recovered top features (in real top-15 order)": ", ".join(f"`{f}`" for f in c["recovered_top_features"][:8])})
    return md(pd.DataFrame(rows))


def t_b1_timing():
    m = js("b1_meta.json")
    rows = []
    for t in m["timing"]:
        rows.append({"task": t["task"], "T": t["T"], "NIO run (300 it., CPU), s": f"{t['nio_seconds_mean_over_P']:.1f}",
                     "reference: T full-batch SGD steps (CPU), s": f"{t['reference_fullbatch_T_steps_seconds']:.3f}",
                     "NIO / reference": f"{t['nio_seconds_mean_over_P'] / max(t['reference_fullbatch_T_steps_seconds'], 1e-9):,.0f}×",
                     "A1 MLP (Adam, early stop, GPU), s": f"{t['a1_mlp_adam_training_seconds_gpu']:.1f}" if t["a1_mlp_adam_training_seconds_gpu"] else "n/a"})
    return md(pd.DataFrame(rows))


def t_b3(P=16):
    s = csv("b3_summary.csv")
    d = s[s.P == P]
    rows = []
    for _, r in d.iterrows():
        rows.append({"defence": r.defence.replace("dp_sigma=", "DP σ=").replace("none", "none (plain SGD)"),
                     "attacker": "knows wd" if r.attacker == "knows_wd" else "plain recipe",
                     "released-model acc": ms(r.ref_accuracy_mean, r.ref_accuracy_std, 4),
                     "loss-MIA AUC": ms(r.mia_auc_mean, r.mia_auc_std, 3),
                     "NIO rel. weight error": ms(r.nio_rel_weight_error_mean, r.nio_rel_weight_error_std, 3),
                     "primer-twin acc": ms(r.twin_accuracy_mean, r.twin_accuracy_std, 3),
                     "struct diff-cosine": ms(r.struct_diff_cosine_mean, r.struct_diff_cosine_std, 2),
                     "NN share closer to train (pooled)": f"{r.nn_pooled_frac_closer_to_train:.3f}",
                     "NN Wilcoxon p (pooled)": pfmt(r.nn_pooled_wilcoxon_p_less)})
    return md(pd.DataFrame(rows))


def t_b3_64():
    return t_b3(64)


def _c1():
    return csv("c1_main_summary.csv")


def _detname(d):
    return d.replace("plugin:", "") + ("-only" if d.startswith("plugin:") else "")


def t_c1_main():
    s = _c1(); weak = js("c1_meta.json")["weak_detector"]
    d = s[s.detector.isin(["known", "novel (LOTO)", weak])]
    rows = []
    for t in ["Ransomware", "Spyware", "Trojan"]:
        for det in ["known", "novel (LOTO)", weak]:
            r = d[(d.type == t) & (d.detector == det)].iloc[0]
            rows.append({"type": t, "detector": _detname(det), "detector AUC": f"{r.detector_AUC_mean:.4f}",
                         "x₁ firewall": ms(r.x1_firewall_mean, r.x1_firewall_std, 3),
                         "x₂ response": ms(r.x2_response_mean, r.x2_response_std, 3),
                         "x₃ τ": ms(r.x3_tau_mean, r.x3_tau_std, 3),
                         "TPR@τ / FPR@τ": f"{r['hard_TPR@tau_mean']:.3f} / {r['hard_FPR@tau_mean']:.3f}",
                         "peak I (default policy)": f"{r.default_peak_pct_mean:.2f}%",
                         "peak I (NIO)": f"{r.peak_pct_mean:.3f}%",
                         "loss NIO": ms(r.loss_mean, r.loss_std, 4), "loss grid 21³": f"{r.grid_loss_mean:.4f}",
                         "gap NIO−grid": f"{r.rel_gap_pct_mean:+.2f}%"})
    return md(pd.DataFrame(rows))


def t_c1_pareto():
    p = csv("c1_pareto.csv")
    d = p[p.detector == "known"]
    weak = js("c1_meta.json")["weak_detector"]
    w = p[p.detector == weak]
    rows = []
    for (l, m), g in d.groupby(["lambda", "mu"]):
        gw = w[(w["lambda"] == l) & (w["mu"] == m)]
        rows.append({"λ": round(float(l), 3), "μ": round(float(m), 3), "x₁ / x₂ / τ (all-feature detector)":
                     f"{g.x1_firewall.mean():.3f} / {g.x2_response.mean():.3f} / {g.x3_tau.mean():.3f}",
                     "peak I": f"{g.peak_pct.mean():.3f}%", "deploy cost": f"{g.deploy_cost.mean():.3f}",
                     f"x₁ / x₂ / τ ({_detname(weak)})": f"{gw.x1_firewall.mean():.3f} / {gw.x2_response.mean():.3f} / {gw.x3_tau.mean():.3f}",
                     f"peak I ({_detname(weak)})": f"{gw.peak_pct.mean():.3f}%",
                     f"false-quarantine cost ({_detname(weak)})": f"{gw.false_quarantine_cost.mean():.4f}"})
    return md(pd.DataFrame(rows))


def t_c1_time_varying():
    t = csv("c1_time_varying.csv")
    weak = js("c1_meta.json")["weak_detector"]
    rows = []
    for det in ["known", "novel (LOTO)", weak]:
        for init in ["cold", "warm"]:
            g = t[(t.detector == det) & (t.init == init)]
            rows.append({"detector": _detname(det), "init": init, "n configs": len(g),
                         "objective time-varying": ms(g.tv_objective.mean(), g.tv_objective.std(ddof=0), 4),
                         "objective best fixed": ms(g.fixed_best_loss.mean(), g.fixed_best_loss.std(ddof=0), 4),
                         "improvement": ms(g.improvement_vs_best_fixed_pct.mean(), g.improvement_vs_best_fixed_pct.std(ddof=0), 2) + "%",
                         "peak TV / fixed": f"{g.tv_peak_pct.mean():.3f}% / {g.fixed_peak_pct.mean():.3f}%",
                         "deploy cost TV / fixed": f"{g.tv_deploy_cost.mean():.3f} / {g.fixed_deploy_cost.mean():.3f}"})
    return md(pd.DataFrame(rows))


def t_c1_quality():
    q = csv("c1_detector_quality.csv")
    q = q.assign(q=q.q.round(3))
    rows = []
    for qq, g in q.groupby("q", sort=False):
        rows.append({"q (TPR scale)": qq, "x₁ firewall": ms(g.x1_firewall.mean(), g.x1_firewall.std(ddof=0), 3),
                     "x₂ response": ms(g.x2_response.mean(), g.x2_response.std(ddof=0), 3),
                     "peak I": f"{g.peak_pct.mean():.3f}%", "loss": f"{g.loss.mean():.4f}"})
    return md(pd.DataFrame(rows))


def t_c1_detpolicy():
    d = csv("c1_detector_to_policy.csv").sort_values("detector_AUC")
    rows = [{"detector": _detname(r.detector), "AUC": f"{r.detector_AUC:.4f}", "x₁": f"{r.x1:.3f}", "x₂": f"{r.x2:.3f}",
             "τ": f"{r.tau:.3f}", "TPR@τ": f"{r.TPR_at_tau:.3f}", "FPR@τ": f"{r.FPR_at_tau:.3f}",
             "peak I": f"{r.peak_pct:.3f}%", "loss": f"{r.loss:.4f}"} for _, r in d.iterrows()]
    return md(pd.DataFrame(rows))


def _gaps():
    m = csv("c1_main.csv").assign(run="main (λ=0.05, μ=0.5)", rel=lambda d: d.rel_gap_pct)
    pa = csv("c1_pareto.csv")
    pa = pa.assign(run="λ/μ sweep", rel=100 * pa.gap_nio_minus_grid / pa.grid_loss)
    return pd.concat([m[["detector", "detector_AUC", "run", "rel"]], pa[["detector", "detector_AUC", "run", "rel"]]])


def t_c1_gaps():
    g = _gaps()
    rows = []
    for det, d in g.groupby("detector"):
        rows.append({"detector": _detname(det), "mean AUC": f"{d.detector_AUC.mean():.4f}", "runs": len(d),
                     "median gap": f"{d.rel.median():+.2f}%", "worst gap": f"{d.rel.max():+.2f}%",
                     "runs > +1% above grid": int((d.rel > 1).sum())})
    return md(pd.DataFrame(rows).sort_values("mean AUC", ascending=False))


def t_runtime():
    rl = csv("runlog.csv")
    rows = [{"step": r.step, "wall-clock": f"{r.seconds / 60:.1f} min" if r.seconds >= 60 else f"{r.seconds:.0f} s",
             "finished": r.timestamp, "what ran": r.note} for _, r in rl.iterrows()]
    return md(pd.DataFrame(rows))


TABLES = {k[2:]: v for k, v in globals().items() if k.startswith("t_") and callable(v)}


# ------------------------------------------------------------------ scalar values
def values() -> dict:
    v = {}
    s = csv("a1_summary.csv")

    def a1(task, split, model, col):
        r = s[(s.task == task) & (s.split == split) & (s.model == model)].iloc[0]
        return r[f"{col}_mean"], r[f"{col}_std"]

    for split in ["random", "group"]:
        for m in ["LR", "RF", "MLP"]:
            v[f"a1_4c_{split}_{m}_f1"] = ms(*a1("4class", split, m, "macro_f1"), 3)
            v[f"a1_bin_{split}_{m}_acc"] = ms(*a1("binary", split, m, "accuracy"), 4)
    l = csv("a1_loto_summary.csv")
    for t in ["Ransomware", "Spyware", "Trojan"]:
        r = l[(l.unseen_type == t) & (l.model == "MLP")].iloc[0]
        v[f"a1_loto_{t}_tpr"] = ms(r["unseen_TPR@0.5_test_mean"], r["unseen_TPR@0.5_test_std"], 4)
        v[f"a1_known_{t}_tpr"] = ms(r["known_TPR@0.5_mean"], r["known_TPR@0.5_std"], 4)
    pl = csv("a1_plugin_detectors_summary.csv")
    weakp = pl[pl.selected_as_weak_detector].iloc[0]
    v["weak_plugin"] = weakp.plugin
    v["weak_plugin_auc"] = ms(weakp.test_AUC_mean, weakp.test_AUC_std, 3)
    v["weak_plugin_tpr"] = f"{weakp.TPR_mean:.3f}"; v["weak_plugin_fpr"] = f"{weakp.FPR_mean:.3f}"
    v["n_plugins_auc_ge_0999"] = str(int((pl.test_AUC_mean >= 0.999).sum()))
    imp = csv("a1_perm_importance.csv")
    v["top_rf_feature"] = imp[imp.model == "RF"].sort_values("mean", ascending=False).feature.iloc[0]

    b = csv("b1_summary.csv")

    def b1(task, T, P, var, col):
        r = b[(b.task == task) & (b["T"] == T) & (b.P == P) & (b.variant == var)].iloc[0]
        return r[f"{col}_mean"], r[f"{col}_std"]

    for P in [2, 4, 8, 16, 32, 64]:
        v[f"b1_P{P}_primer"] = ms(*b1("binary", 20, P, "fixed", "primer_accuracy"))
        v[f"b1_P{P}_subset"] = ms(*b1("binary", 20, P, "fixed", "subset_accuracy"))
        v[f"b1_P{P}_noise"] = ms(*b1("binary", 20, P, "fixed", "noise_accuracy"))
        v[f"b1_P{P}_relerr"] = ms(*b1("binary", 20, P, "fixed", "nio_rel_weight_error"), 2)
        v[f"b1_P{P}_cos"] = ms(*b1("binary", 20, P, "fixed", "struct_diff_cosine"), 2)
    v["b1_full"] = ms(*b1("binary", 20, 16, "fixed", "fulldata_accuracy"))
    for P in [4, 8, 16, 32, 64]:
        v[f"b1_4c_P{P}_primer"] = ms(*b1("4class", 20, P, "fixed", "primer_accuracy"))
        v[f"b1_4c_P{P}_subset"] = ms(*b1("4class", 20, P, "fixed", "subset_accuracy"))
    v["b1_4c_full"] = ms(*b1("4class", 20, 16, "fixed", "fulldata_accuracy"))
    f = b[(b.task == "binary") & (b["T"] == 20) & (b.variant == "fixed")]
    v["b1_primer_beats_subset_n"] = str(int((f.primer_accuracy_mean > f.subset_accuracy_mean).sum()))
    allruns = csv("b1_runs.csv")
    v["b1_n_runs"] = str(len(allruns))
    v["b1_primer_norm"] = f"{allruns.primer_mean_norm.mean():.2f}"
    v["b1_real_norm"] = f"{allruns.real_train_mean_norm.mean():.2f}"
    v["b1_primer_dtrain"] = f"{allruns.nn_d_train_median.median():.2f}"
    ctrl = csv("b1_nn_pooled.csv")
    c = ctrl[(ctrl.variant == "control") & (ctrl.set == "nonmember_rows (val)")].iloc[0]
    v["b1_real_nn"] = f"{c.d_holdout_median:.3f}"
    pp = ctrl[(ctrl.P.astype(str) == "all") & (ctrl.task == "binary") & (ctrl["T"].astype(int) == 20) &
              (ctrl.variant == "fixed") & (ctrl.set == "nio_primer")].iloc[0]
    v["b1_leak_share"] = f"{pp.frac_closer_to_train:.3f}"; v["b1_leak_n"] = str(int(pp.n_points))
    v["b1_leak_binom"] = pfmt(pp.binom_p_greater); v["b1_leak_wil"] = pfmt(pp.wilcoxon_p_less); v["b1_leak_mw"] = pfmt(pp.mw_p_less)
    v["b1_leak_ratio"] = f"{pp.ratio_median:.3f}"
    pc = ctrl[(ctrl.variant == "control") & (ctrl.set == "memorised near-copies sd=0.005 (train)")].iloc[0]
    v["b1_pos_share"] = f"{pc.frac_closer_to_train:.3f}"; v["b1_pos_wil"] = pfmt(pc.wilcoxon_p_less)
    v["b1_pos_ratio"] = f"{pc.ratio_median:.3f}"
    tm = js("b1_meta.json")["timing"]
    t20 = next(t for t in tm if t["task"] == "binary" and t["T"] == 20)
    v["b1_nio_secs"] = f"{t20['nio_seconds_mean_over_P']:.1f}"
    v["b1_ref_secs"] = f"{t20['reference_fullbatch_T_steps_seconds']:.3f}"
    v["b1_cost_ratio"] = f"{t20['nio_seconds_mean_over_P'] / t20['reference_fullbatch_T_steps_seconds']:,.0f}"

    s3 = csv("b3_summary.csv")

    def b3(defn, att="plain", P=16):
        return s3[(s3.defence == defn) & (s3.attacker == att) & (s3.P == P)].iloc[0]

    for dname, key in [("none", "none"), ("dp_sigma=0", "dp0"), ("dp_sigma=2", "dp2"), ("dp_sigma=8", "dp8"), ("wd=0.1", "wd01")]:
        r = b3(dname)
        v[f"b3_{key}_ref"] = ms(r.ref_accuracy_mean, r.ref_accuracy_std, 4)
        v[f"b3_{key}_twin"] = ms(r.twin_accuracy_mean, r.twin_accuracy_std, 3)
        v[f"b3_{key}_relerr"] = ms(r.nio_rel_weight_error_mean, r.nio_rel_weight_error_std, 3)
        v[f"b3_{key}_mia"] = f"{r.mia_auc_mean:.3f}"
    r = b3("wd=0.1", "knows_wd")
    v["b3_wd01_informed_relerr"] = ms(r.nio_rel_weight_error_mean, r.nio_rel_weight_error_std, 3)
    v["b3_wd01_informed_twin"] = ms(r.twin_accuracy_mean, r.twin_accuracy_std, 3)
    v["b3_mia_max"] = f"{s3.mia_auc_mean.max():.3f}"; v["b3_mia_min"] = f"{s3.mia_auc_mean.min():.3f}"
    v["b3_n_tests"] = str(len(s3))
    imin = s3.nn_pooled_wilcoxon_p_less.idxmin()
    v["b3_min_p"] = pfmt(s3.loc[imin, "nn_pooled_wilcoxon_p_less"])
    v["b3_min_p_cfg"] = f"{s3.loc[imin, 'defence'].replace('dp_sigma=', 'DP σ=')}, {s3.loc[imin, 'attacker']}, P = {s3.loc[imin, 'P']}"
    v["b3_bonf"] = pfmt(0.05 / len(s3))
    v["b3_n_bonf"] = str(int((s3.nn_pooled_wilcoxon_p_less < 0.05 / len(s3)).sum()))
    hi = s3[s3.dp_sigma >= 4]
    v["b3_twin_s4_min"] = f"{hi.twin_accuracy_mean.min():.3f}"
    r8 = b3("dp_sigma=8")
    v["b3_dp8_relerr"] = ms(r8.nio_rel_weight_error_mean, r8.nio_rel_weight_error_std, 3)

    cm = _c1()
    k = cm[(cm.type == "Spyware") & (cm.detector == "known")].iloc[0]
    n = cm[(cm.type == "Spyware") & (cm.detector == "novel (LOTO)")].iloc[0]
    weak = js("c1_meta.json")["weak_detector"]
    w = cm[(cm.type == "Spyware") & (cm.detector == weak)].iloc[0]
    for key, r in [("known", k), ("loto", n), ("weak", w)]:
        v[f"c1_spy_{key}_x"] = f"({r.x1_firewall_mean:.3f}, {r.x2_response_mean:.3f}, τ={r.x3_tau_mean:.3f})"
        v[f"c1_spy_{key}_peak"] = f"{r.peak_pct_mean:.3f}%"
        v[f"c1_spy_{key}_x1"] = f"{r.x1_firewall_mean:.3f}"
    v["c1_default_peak"] = f"{k.default_peak_pct_mean:.2f}%"
    main = csv("c1_main.csv")
    v["c1_gap_mean"] = f"{main.rel_gap_pct.mean():+.2f}%"; v["c1_gap_max"] = f"{main.rel_gap_pct.max():+.2f}%"
    v["c1_n_configs"] = str(len(main))
    v["c1_iters_conv"] = f"{main.iters_to_1pct.median():.0f}"
    pa = csv("c1_pareto.csv")
    v["c1_pareto_n"] = str(len(pa)); v["c1_pareto_gap_max"] = f"{pa.gap_nio_minus_grid.max():+.5f}"
    v["c1_pareto_worse_n"] = str(int((pa.gap_nio_minus_grid > 1e-6).sum()))
    tv = csv("c1_time_varying.csv")
    best_tv = tv.groupby(["seed", "type", "detector"]).improvement_vs_best_fixed_pct.max()
    v["c1_tv_impr"] = ms(best_tv.mean(), best_tv.std(ddof=0), 2) + "%"
    qd = csv("c1_detector_quality.csv")
    q = qd.assign(q=qd.q.round(3)).groupby("q").x1_firewall.mean()
    v["c1_q1_x1"] = f"{q.loc[1.0]:.3f}"; v["c1_q01_x1"] = f"{q.loc[0.1]:.3f}"; v["c1_q05_x1"] = f"{q.loc[0.5]:.3f}"
    g = _gaps()
    kl = g[g.detector.isin(["known", "novel (LOTO)"])]
    v["c1_kl_max"] = f"{kl.rel.max():+.2f}%"; v["c1_kl_median"] = f"{kl.rel.median():+.2f}%"; v["c1_kl_n"] = str(len(kl))
    v["c1_n_gt1"] = str(int((g.rel > 1).sum())); v["c1_n_runs"] = str(len(g))
    pl = g[~g.detector.isin(["known", "novel (LOTO)"])]
    v["c1_pl_n_gt1"] = str(int((pl.rel > 1).sum())); v["c1_pl_n"] = str(len(pl))
    wi = csv("c1_pareto.csv").assign(rel=lambda d: 100 * d.gap_nio_minus_grid / d.grid_loss).sort_values("rel").iloc[-1]
    v["c1_worst_rel"] = f"{wi.rel:+.1f}%"
    v["c1_worst_cfg"] = f"{_detname(wi.detector)}, {wi.type}, λ = {wi['lambda']:.2g}, μ = {wi.mu:.2g}"
    # --- extra scalars used in the narrative ---
    c7 = js("sanity_ch7_sir.json")["ours"]
    v["c7_p00"] = f"{c7['policy_0_0_peak_pct']:.2f}%"; v["c7_p11"] = f"{c7['policy_1_1_peak_pct']:.2f}%"
    v["c7_manual"] = f"{c7['manual_policy_0.3_0.4_peak_pct']:.2f}%"
    v["b1_noise_min"] = f"{f.noise_accuracy_mean.min():.2f}"; v["b1_noise_max"] = f"{f.noise_accuracy_mean.max():.2f}"
    v["b1_cos_min"] = f"{f.struct_diff_cosine_mean.min():.2f}"; v["b1_cos_max"] = f"{f.struct_diff_cosine_mean.max():.2f}"
    v["b1_overlap_min"] = f"{f.struct_top15_overlap_mean.min():.1f}"; v["b1_overlap_max"] = f"{f.struct_top15_overlap_mean.max():.1f}"
    bf = b[(b.task == "binary") & (b.variant == "fixed")]
    v["b1_T_min_acc"] = f"{bf.primer_accuracy_mean.min():.3f}"
    v["b1_T40_cost"] = f"{bf[bf['T'] == 40].nio_seconds_mean.mean() / bf[bf['T'] == 20].nio_seconds_mean.mean():.1f}"
    r16 = f[f.P == 16].iloc[0]
    v["b1_P16_gap"] = f"{100 * (r16.fulldata_accuracy_mean - r16.primer_accuracy_mean):.1f}"
    v["b3_dp_drop"] = f"{100 * (b3('none').ref_accuracy_mean - b3('dp_sigma=2').ref_accuracy_mean):.1f}"
    v["c1_pareto_secs"] = f"{js('c1_meta.json')['seconds']['pareto_batched']:.0f}"
    v["a1_bin_min"] = f"{100 * s[s.task == 'binary'].accuracy_mean.min():.2f}%"
    v["a1_bin_max"] = f"{100 * s[s.task == 'binary'].accuracy_mean.max():.2f}%"
    lm = l[l.model == "MLP"]
    v["a1_loto_max_miss"] = f"{100 * (1 - lm['unseen_TPR@0.5_test_mean'].min()):.2f}%"
    return v


def main():
    tpl = (ROOT / "docs" / "RESULTS.template.md").read_text(encoding="utf-8")
    vals = values()
    missing = []

    def sub_table(m):
        name = m.group(1)
        if name not in TABLES:
            missing.append("table:" + name); return m.group(0)
        return TABLES[name]()

    def sub_val(m):
        name = m.group(1)
        if name not in vals:
            missing.append(name); return m.group(0)
        return vals[name]

    out = re.sub(r"\{\{table:([A-Za-z0-9_]+)\}\}", sub_table, tpl)
    out = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", sub_val, out)
    if missing:
        raise SystemExit(f"unresolved placeholders: {missing}")
    (ROOT / "RESULTS.md").write_text(out, encoding="utf-8")
    print("RESULTS.md written", len(out), "chars")


if __name__ == "__main__":
    main()
