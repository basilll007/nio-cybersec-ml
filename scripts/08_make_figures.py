"""Make every figure from the files in results/ (no figure reads data/ or re-runs a model)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotting import (AQUA, BLUE, CAT, ENTITY, FULLDATA, GRID, INK, INK2, MUTED, ORANGE, SEQ_BLUE, VIOLET, YELLOW,
                          panel_label, save, seq, setup)
from src.utils import FIGURES, RESULTS, load_json

TYPES4 = ["Benign", "Ransomware", "Spyware", "Trojan"]
MTYPES = ["Ransomware", "Spyware", "Trojan"]


def have(*names):
    ok = all((RESULTS / n).exists() for n in names)
    if not ok:
        print("skip (missing):", names)
    return ok


# ----------------------------------------------------------------------------------------------- A1
def fig01():
    if not have("a1_dataset_stats.json"):
        return
    st = load_json(RESULTS / "a1_dataset_stats.json")["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [0.7, 1, 2.2]})
    ax = axes[0]
    cc = st["class_counts"]
    ax.bar(["Benign", "Malware"], [cc["Benign"], cc["Malware"]], color=[BLUE, ORANGE], width=0.6)
    ax.set_ylabel("Memory dumps (rows)")
    panel_label(ax, "a")
    ax = axes[1]
    tc = st["type_counts"]
    ax.bar(TYPES4, [tc[t] for t in TYPES4], color=[ENTITY[t] for t in TYPES4], width=0.6)
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=20)
    panel_label(ax, "b")
    ax = axes[2]
    fam = {k: v for k, v in st["family_counts"].items() if k != "Benign"}
    order = sorted(fam, key=lambda k: (k.split("-")[0], -fam[k]))
    ax.barh(range(len(order)), [fam[k] for k in order], color=[ENTITY[k.split("-")[0]] for k in order], height=0.7)
    ax.set_yticks(range(len(order)), [k.split("-", 1)[1] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("Rows (benign family = %d rows, not shown)" % st["family_counts"]["Benign"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENTITY[t]) for t in MTYPES]
    ax.legend(handles, MTYPES, loc="lower right")
    panel_label(ax, "c")
    save(fig, "fig01_malmem_dataset")


def fig02():
    if not have("a1_confusion.json"):
        return
    conf = load_json(RESULTS / "a1_confusion.json")["confusion"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, model, lab in zip(axes, ["RF", "MLP"], "ab"):
        cm = np.array(conf[f"group|4class|{model}"]["sum"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff"] + SEQ_BLUE), vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100 * cmn[i, j]:.1f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), TYPES4, rotation=25); ax.set_yticks(range(4), TYPES4)
        ax.set_xlabel(f"Predicted ({model})"); ax.set_ylabel("True")
        ax.grid(False)
        panel_label(ax, lab)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Row-normalised fraction")
    for ext in ("png", "pdf"):  # colorbar spans both panels, so skip tight_layout here
        fig.savefig(FIGURES / f"fig02_malmem_confusion.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig); print("saved fig02_malmem_confusion")


def fig03():
    if not have("a1_perm_importance.csv"):
        return
    imp = pd.read_csv(RESULTS / "a1_perm_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, model, col, lab in zip(axes, ["RF", "MLP"], [BLUE, ORANGE], "ab"):
        d = imp[imp.model == model].sort_values("mean", ascending=False).head(15)
        ax.barh(range(len(d)), d["mean"], xerr=d["std"], color=col, height=0.7, error_kw={"ecolor": INK2, "lw": 1})
        ax.set_yticks(range(len(d)), d["feature"]); ax.invert_yaxis()
        ax.set_xlabel(f"Drop in 4-class macro-F1 when permuted ({model})")
        panel_label(ax, lab)
    save(fig, "fig03_malmem_importance")


def fig04():
    if not have("a1_summary.csv", "a1_loto_summary.csv"):
        return
    s = pd.read_csv(RESULTS / "a1_summary.csv")
    l = pd.read_csv(RESULTS / "a1_loto_summary.csv")
    models = ["LR", "RF", "MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    w = 0.38
    ax = axes[0]
    for k, (split, col) in enumerate([("random", ORANGE), ("group", BLUE)]):
        d = s[(s.task == "4class") & (s.split == split)].set_index("model").loc[models]
        ax.bar(np.arange(3) + (k - 0.5) * w, d["macro_f1_mean"], w, yerr=d["macro_f1_std"], color=col,
               label=f"{split} split", error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xticks(range(3), models); ax.set_ylabel("4-class macro-F1 (test)"); ax.set_ylim(0.5, 0.9); ax.legend(loc="upper left")
    panel_label(ax, "a")
    ax = axes[1]
    for k, (split, col) in enumerate([("random", ORANGE), ("group", BLUE)]):
        d = s[(s.task == "binary") & (s.split == split)].set_index("model").loc[models]
        ax.bar(np.arange(3) + (k - 0.5) * w, 100 * (1 - d["accuracy_mean"]), w, yerr=100 * d["accuracy_std"], color=col,
               label=f"{split} split", error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xticks(range(3), models); ax.set_ylabel("Binary test error (%)"); ax.legend(loc="upper right")
    panel_label(ax, "b")
    ax = axes[2]
    for k, (det, col, mcol, scol) in enumerate([("known type (group split)", BLUE, "known_TPR@0.5_mean", "known_TPR@0.5_std"),
                                                ("unseen type (LOTO)", ORANGE, "unseen_TPR@0.5_test_mean", "unseen_TPR@0.5_test_std")]):
        d = l[l.model == "MLP"].set_index("unseen_type").loc[MTYPES]
        ax.bar(np.arange(3) + (k - 0.5) * w, 100 * (1 - d[mcol]), w, yerr=100 * d[scol], color=col, label=det,
               error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xticks(range(3), MTYPES); ax.set_ylabel("Missed malware, MLP (1 - TPR@0.5, %)"); ax.legend(loc="upper left")
    panel_label(ax, "c")
    save(fig, "fig04_split_honesty")


# ----------------------------------------------------------------------------------------------- B1
def fig06():
    if not have("b1_curves.json"):
        return
    cur = load_json(RESULTS / "b1_curves.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task, Ps, lab in [(axes[0], "binary", [2, 4, 8, 16, 32, 64], "a"), (axes[1], "4class", [4, 8, 16, 32, 64], "b")]:
        cols = seq(len(Ps))
        for P, col in zip(Ps, cols):
            L = np.array([c["losses"] for c in cur if c["task"] == task and c["T"] == 20 and c["P"] == P and c["variant"] == "fixed"])
            if len(L) == 0:
                continue
            L = L / 6.0  # 6 parameter tensors -> mean relative weight error
            m, sd = L.mean(0), L.std(0)
            it = np.arange(len(m))
            ax.plot(it, m, color=col, label=f"P = {P}", lw=1.6)
            ax.fill_between(it, np.maximum(m - sd, 1e-4), m + sd, color=col, alpha=0.15, lw=0)
        ax.axhline(1.0, color=MUTED, ls=":", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("NIO iteration"); ax.set_ylabel(f"Relative weight error ({task}, T = 20)")
        ax.legend(ncol=2)
        panel_label(ax, lab)
    save(fig, "fig06_primer_convergence")


def _b1sum():
    return pd.read_csv(RESULTS / "b1_summary.csv")


def fig07():
    if not have("b1_summary.csv"):
        return
    s = _b1sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task, lab in [(axes[0], "binary", "a"), (axes[1], "4class", "b")]:
        d = s[(s.task == task) & (s["T"] == 20)]
        f = d[d.variant == "fixed"].sort_values("P")
        P = f["P"].to_numpy()
        ax.errorbar(P, f["primer_accuracy_mean"], f["primer_accuracy_std"], color=ENTITY["primer"], marker="o", ms=6,
                    capsize=3, label="NIO primer (fixed labels)")
        so = d[d.variant == "soft"].sort_values("P")
        if len(so):
            ax.errorbar(so["P"] * 1.06, so["primer_accuracy_mean"], so["primer_accuracy_std"], color=VIOLET, marker="s",
                        ms=6, capsize=3, ls="--", lw=1.6, label="NIO primer (learnable soft labels)")
        ax.errorbar(P, f["subset_accuracy_mean"], f["subset_accuracy_std"], color=ENTITY["subset"], marker="^", ms=6,
                    capsize=3, label="Random real subset (balanced)")
        ax.errorbar(P, f["noise_accuracy_mean"], f["noise_accuracy_std"], color=ENTITY["noise"], marker="D", ms=5,
                    capsize=3, label="Gaussian-noise primer")
        fm, fs = f["fulldata_accuracy_mean"].mean(), f["fulldata_accuracy_std"].mean()
        ax.axhline(fm, color=FULLDATA, ls="--", lw=1.2, label=f"Full data reference ({fm:.3f})")
        ax.axhspan(fm - fs, fm + fs, color=FULLDATA, alpha=0.08, lw=0)
        chance = 0.5 if task == "binary" else f["theta0_accuracy_mean"].mean()
        if task == "binary":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(P, [str(p) for p in P])
        ax.set_xlabel("Primer size P (rows)"); ax.set_ylabel(f"Test accuracy of twin retrained from θ₀ ({task})")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if task == "binary" else "center right")
        panel_label(ax, lab)
    save(fig, "fig07_primer_vs_baselines")


def fig08():
    if not have("b1_summary.csv"):
        return
    s = _b1sum()
    d = s[(s.task == "binary") & (s.variant == "fixed")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    Ts = sorted(d["T"].unique())
    cols = seq(len(Ts))
    for T, col in zip(Ts, cols):
        f = d[d["T"] == T].sort_values("P")
        axes[0].errorbar(f["P"], f["primer_accuracy_mean"], f["primer_accuracy_std"], color=col, marker="o", capsize=3, label=f"T = {T}")
        axes[0].axhline(f["fulldata_accuracy_mean"].mean(), color=col, ls="--", lw=1)
        axes[1].errorbar(f["P"], f["nio_rel_weight_error_mean"], f["nio_rel_weight_error_std"], color=col, marker="o", capsize=3, label=f"T = {T}")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("Primer size P (rows)")
        ax.legend()
    axes[0].set_ylabel("Twin test accuracy (binary)\n(dashed: full-data reference at same T)")
    axes[1].set_ylabel("Final relative weight error\n‖θ_P − θ_ref‖² / ‖θ_ref − θ₀‖²")
    axes[1].set_yscale("log")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save(fig, "fig08_primer_T_sweep")


def fig09():
    if not have("b1_nn_distances.csv"):
        return
    nn = pd.read_csv(RESULTS / "b1_nn_distances.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    p = nn[(nn.set == "nio_primer") & (nn.task == "binary") & (nn["T"] == 20) & (nn.variant == "fixed")]
    for col, name, c in [("d_train", "primer → nearest training row", ENTITY["train"]),
                         ("d_holdout", "primer → nearest held-out row", ENTITY["holdout"])]:
        v = np.sort(p[col].to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=name)
    ax.set_xscale("log"); ax.set_xlabel("Euclidean NN distance (standardised features)"); ax.set_ylabel("ECDF over primer rows")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.legend(loc="lower right")
    panel_label(ax, "a")
    ax = axes[1]
    groups = [("NIO primer\n(binary, T=20)", p),
              ("noise primer\n(NIO start)", nn[(nn.set == "noise_primer") & (nn.task == "binary") & (nn["T"] == 20)]),
              ("near-copies of\ntrain rows\n(sd 0.005)", nn[nn.set == "memorised near-copies sd=0.005 (train)"]),
              ("perturbed copies\n(sd 0.05)", nn[nn.set == "perturbed copies sd=0.05 (train)"]),
              ("non-member\nreal rows (val)", nn[nn.set == "nonmember_rows (val)"])]
    data = [np.log10(np.maximum(g.d_train / g.d_holdout, 1e-3)) for _, g in groups]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops={"marker": ".", "markersize": 3, "markeredgecolor": MUTED})
    for patch, c in zip(bp["boxes"], [BLUE, AQUA, ORANGE, YELLOW, VIOLET]):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor(INK2)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_xticks(range(1, len(groups) + 1), [g[0] for g in groups], fontsize=7.5)
    ax.axhline(0, color=INK2, ls=":", lw=1)
    ax.set_ylabel("log10( d_train / d_held-out )\n< 0 means closer to training data")
    panel_label(ax, "b")
    save(fig, "fig09_leakage_nn")


def fig10():
    if not have("b1_pca.csv"):
        return
    pc = pd.read_csv(RESULTS / "b1_pca.csv")
    meta = load_json(RESULTS / "b1_meta.json")
    ev = meta.get("pca_explained_variance") or [np.nan, np.nan]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=False, sharey=False)
    real = pc[pc.set == "real"]
    for ax, sets, lab in [(axes[0], ["nio_primer_P16", "noise_primer_P16"], "a"), (axes[1], ["nio_primer_P64"], "b")]:
        for l, c, name in [(0, BLUE, "real benign"), (1, ORANGE, "real malware")]:
            r = real[real.label == l]
            ax.scatter(r.pc1, r.pc2, s=4, color=c, alpha=0.25, lw=0, label=name)
        for s_, mk in zip(sets, ["*", "X"]):
            d = pc[pc.set == s_]
            for l, c in [(0, BLUE), (1, ORANGE)]:
                dd = d[d.label == l]
                ax.scatter(dd.pc1, dd.pc2, s=120 if mk == "*" else 60, marker=mk, color=c, edgecolor=INK, lw=0.8,
                           label=f"{s_.replace('_', ' ')} ({'benign' if l == 0 else 'malware'} label)")
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var.)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var.)")
        ax.legend(fontsize=7, loc="best", markerscale=0.8)
        panel_label(ax, lab)
    save(fig, "fig10_primer_pca")


# ----------------------------------------------------------------------------------------------- B3
def fig11():
    if not have("b3_summary.csv"):
        return
    s = pd.read_csv(RESULTS / "b3_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    dp = s[s.defence.str.startswith("dp") & (s.attacker == "plain")]
    none = s[(s.defence == "none") & (s.attacker == "plain")]
    ax = axes[0]
    for P, col, ls in [(16, BLUE, "-"), (64, SEQ_BLUE[6], "--")]:
        d = dp[dp.P == P].sort_values("dp_sigma")
        ax.errorbar(d.dp_sigma, d.twin_accuracy_mean, d.twin_accuracy_std, color=col, ls=ls, marker="o", capsize=3,
                    label=f"leakage: NIO-primer twin acc. (P={P})")
    d = dp[dp.P == 16].sort_values("dp_sigma")
    ax.errorbar(d.dp_sigma, d.ref_accuracy_mean, d.ref_accuracy_std, color=FULLDATA, marker="s", capsize=3,
                label="utility: released model acc.")
    ax.axhline(none[none.P == 16].ref_accuracy_mean.iloc[0], color=FULLDATA, ls=":", lw=1, label="undefended model acc.")
    ax.set_xscale("symlog", linthresh=0.5); ax.set_xticks([0, 0.5, 1, 2, 4, 8], ["0", "0.5", "1", "2", "4", "8"])
    ax.set_xlabel("DP-style noise multiplier σ (clip C = 1, lot 256)"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.6, 1.005)
    ax.legend(fontsize=7.5, loc="lower left")
    panel_label(ax, "a")
    ax = axes[1]
    for P, col, ls in [(16, BLUE, "-"), (64, SEQ_BLUE[6], "--")]:
        d = dp[dp.P == P].sort_values("dp_sigma")
        ax.errorbar(d.dp_sigma, d.nio_rel_weight_error_mean, d.nio_rel_weight_error_std, color=col, ls=ls, marker="o",
                    capsize=3, label=f"NIO weight-match error (P={P})")
        ax.axhline(none[none.P == P].nio_rel_weight_error_mean.iloc[0], color=col, ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.5); ax.set_xticks([0, 0.5, 1, 2, 4, 8], ["0", "0.5", "1", "2", "4", "8"])
    ax.set_ylim(0, None)
    ax.set_xlabel("DP-style noise multiplier σ"); ax.set_ylabel("Relative weight error, higher = harder to invert\n(dotted: undefended)")
    ax.legend(fontsize=7.5)
    panel_label(ax, "b")
    ax = axes[2]
    d16 = s[(s.P == 16)]
    for _, r in d16.iterrows():
        if r.defence == "none":
            c, mk = FULLDATA, "s"
        elif r.defence.startswith("dp"):
            c, mk = BLUE, "o"
        else:
            c, mk = ORANGE, "^" if r.attacker == "plain" else "v"
        ax.errorbar(r.ref_accuracy_mean, r.twin_accuracy_mean, yerr=r.twin_accuracy_std,
                    color=c, marker=mk, ms=7, capsize=2, ls="none", elinewidth=1)
        if r.defence.startswith("dp"):
            lab, off = r.defence.replace("dp_sigma=", "σ="), (6, -3)
        elif r.defence == "none":
            lab, off = "", (0, 0)
        else:
            lab = r.defence
            off = (-6, -14) if r.attacker == "plain" else (-6, 8)
        if lab:
            ax.annotate(lab, (r.ref_accuracy_mean, r.twin_accuracy_mean), fontsize=7, color=INK2, xytext=off,
                        textcoords="offset points", ha="left" if off[0] > 0 else "right")
    ax.set_ylim(0.6, 1.005)
    ax.set_xlabel("Utility: released-model test accuracy"); ax.set_ylabel("Leakage: primer-twin test accuracy (P=16)")
    ax.legend(handles=[Line2D([], [], color=FULLDATA, marker="s", ls="none", label="undefended"),
                       Line2D([], [], color=BLUE, marker="o", ls="none", label="DP-style σ"),
                       Line2D([], [], color=ORANGE, marker="^", ls="none", label="weight decay (attacker unaware)"),
                       Line2D([], [], color=ORANGE, marker="v", ls="none", label="weight decay (attacker knows wd)")],
              fontsize=7.5, loc="best")
    panel_label(ax, "c")
    save(fig, "fig11_defense_tradeoff")


# ----------------------------------------------------------------------------------------------- C1
def fig13():
    if not have("c1_trajectories.json"):
        return
    tr = load_json(RESULTS / "c1_trajectories.json")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    t = np.arange(500) * 0.1
    for j, typ in enumerate(MTYPES):
        r = next(x for x in tr if x["type"] == typ and x["detector"] == "known")
        for comp in ["S", "R"]:
            axes[0, j].plot(t, r["default"][comp], color=ENTITY[comp], ls="--", lw=1.5)
            axes[0, j].plot(t, r["nio"][comp], color=ENTITY[comp], ls="-", lw=2)
        axes[1, j].plot(t, 100 * np.array(r["default"]["I"]), color=ENTITY["I"], ls="--", lw=1.5)
        axes[1, j].plot(t, 100 * np.array(r["nio"]["I"]), color=ENTITY["I"], ls="-", lw=2)
        d, n = r["default_policy"], r["nio_policy"]
        axes[0, j].text(0.98, 0.5, f"{typ}\ndefault x=({d[0]:.2f}, {d[1]:.2f}, τ={d[2]:.2f})\nNIO x=({n[0]:.2f}, {n[1]:.2f}, τ={n[2]:.2f})",
                        transform=axes[0, j].transAxes, ha="right", va="center", fontsize=8, color=INK2)
        axes[1, j].set_xlabel("Time (model units; dt = 0.1, 500 steps)")
        panel_label(axes[0, j], "abc"[j])
    axes[0, 0].set_ylabel("Fraction of hosts (S, R)"); axes[1, 0].set_ylabel("Infected I(t) (%)")
    handles = [Line2D([], [], color=ENTITY["S"], label="S susceptible"), Line2D([], [], color=ENTITY["R"], label="R recovered/cleaned"),
               Line2D([], [], color=ENTITY["I"], label="I infected"), Line2D([], [], color=INK2, ls="--", label="default policy"),
               Line2D([], [], color=INK2, ls="-", label="NIO policy")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig13_sir_before_after")


def fig14():
    if not have("c1_convergence.json"):
        return
    cv = load_json(RESULTS / "c1_convergence.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for typ, col in zip(MTYPES, [ORANGE, AQUA, YELLOW]):
        for det, ls in [("known", "-"), ("novel (LOTO)", "--")]:
            r = next(x for x in cv if x["type"] == typ and x["detector"] == det)
            ax.plot(r["loss"], color=col, ls=ls, lw=1.6, label=f"{typ}, {det}")
    ax.set_yscale("log"); ax.set_xlabel("NIO iteration"); ax.set_ylabel("Loss  max I + λ‖x₁,₂‖² + μ x₂ FPR")
    ax.legend(fontsize=7.5, ncol=1)
    panel_label(ax, "a")
    ax = axes[1]
    for det, ls in [("known", "-"), ("novel (LOTO)", "--")]:
        r = next(x for x in cv if x["type"] == "Spyware" and x["detector"] == det)
        for key, col, name in [("x1", BLUE, "x₁ firewall"), ("x2", ORANGE, "x₂ response"), ("tau", AQUA, "x₃ threshold τ")]:
            ax.plot(r[key], color=col, ls=ls, lw=1.6, label=f"{name} ({det})")
    ax.set_xlabel("NIO iteration"); ax.set_ylabel("Policy value (Spyware)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    panel_label(ax, "b")
    save(fig, "fig14_policy_convergence")


def fig15():
    if not have("c1_pareto.csv", "c1_grid_cloud.csv", "c1_grid_front.csv", "c1_meta.json"):
        return
    p = pd.read_csv(RESULTS / "c1_pareto.csv").round({"lambda": 3, "mu": 3}); cl = pd.read_csv(RESULTS / "c1_grid_cloud.csv"); fr = pd.read_csv(RESULTS / "c1_grid_front.csv")
    weak = load_json(RESULTS / "c1_meta.json")["weak_detector"]
    lams = sorted(p["lambda"].unique()); mus = sorted(p["mu"].unique())
    lcol = dict(zip(lams, seq(len(lams))))
    mk = dict(zip(mus, ["o", "s", "^", "D"]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey="row")
    for row, det in enumerate(["known", weak]):
        for j, typ in enumerate(MTYPES):
            ax = axes[row, j]
            c = cl[(cl.type == typ) & (cl.detector == det)]
            ax.scatter(c.deploy_cost, c.peak_pct, s=6, color=MUTED, alpha=0.35, lw=0)
            f = fr[(fr.type == typ) & (fr.detector == det)]
            ax.step(f.deploy_cost, f.peak_pct, where="post", color=INK, lw=1.2)
            d = p[(p.type == typ) & (p.detector == det) & (p.seed == 0)]
            for _, r in d.iterrows():
                ax.scatter(r.deploy_cost, r.peak_pct, s=55, marker=mk[r.mu], color=lcol[r["lambda"]], edgecolor=INK, lw=0.6, zorder=3)
            ax.set_yscale("log"); ax.set_xlabel("Deployment cost x₁² + x₂²")
            dname = "all-feature detector" if det == "known" else f"{det.split(':')[1]}-only detector"
            ax.text(0.97, 0.95, f"{typ}, {dname}", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8.5)
            panel_label(ax, "abcdef"[3 * row + j])
        axes[row, 0].set_ylabel("Peak infection max I(t) (%)")
    h = [Line2D([], [], color=lcol[l], marker="o", ls="none", ms=7, label=f"NIO λ={l:g}") for l in lams]
    h += [Line2D([], [], color=INK2, marker=mk[m], ls="none", ms=7, mfc="white", label=f"μ={m:g}") for m in mus]
    h += [Line2D([], [], color=INK, lw=1.2, label="21³ grid Pareto front"),
          Line2D([], [], color=MUTED, marker="o", ls="none", ms=4, label="grid policies (best τ)")]
    fig.legend(handles=h, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    save(fig, "fig15_pareto")


def fig16():
    if not have("c1_time_varying_schedules.json"):
        return
    tv = load_json(RESULTS / "c1_time_varying_schedules.json")
    r = next(x for x in tv if x["type"] == "Spyware" and x["detector"] == "known")
    W = len(r["x1"])
    t = np.arange(500) * 0.1
    edges = np.linspace(0, 50, W + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for key, col, name, fi in [("x1", BLUE, "x₁ firewall", 0), ("x2", ORANGE, "x₂ response", 1)]:
        ax.stairs(r[key], edges, color=col, lw=2, label=f"{name}, time-varying")
        ax.axhline(r["fixed_policy"][fi], color=col, ls="--", lw=1.3, label=f"{name}, best fixed")
    ax.set_xlabel("Time"); ax.set_ylabel("Policy value (Spyware, known detector)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(t, 100 * np.array(r["I_tv"]), color=BLUE, lw=2, label="time-varying policy")
    ax.plot(t, 100 * np.array(r["I_fixed"]), color=ORANGE, lw=1.6, ls="--", label="best fixed policy (NIO)")
    ax.set_xlabel("Time"); ax.set_ylabel("Infected I(t) (%)"); ax.legend()
    panel_label(ax, "b")
    save(fig, "fig16_time_varying_policy")


def fig18():
    if not have("c1_detector_quality.csv", "c1_detector_to_policy.csv"):
        return
    q = pd.read_csv(RESULTS / "c1_detector_quality.csv")
    q["q"] = q["q"].round(3)
    g = q.groupby("q").agg(x1=("x1_firewall", "mean"), x1s=("x1_firewall", "std"), x2=("x2_response", "mean"),
                           x2s=("x2_response", "std"), pk=("peak_pct", "mean"), pks=("peak_pct", "std")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    ax.errorbar(g.q, g.x1, g.x1s, color=BLUE, marker="o", capsize=3, label="x₁ firewall")
    ax.errorbar(g.q, g.x2, g.x2s, color=ORANGE, marker="s", capsize=3, label="x₂ automated response")
    ax.set_xlabel("Hypothetical detector quality q\n(effective TPR = q·TPR(τ); q = 1 is the real detector)")
    ax.set_ylabel("NIO-optimal policy"); ax.set_ylim(0, 1.02); ax.legend(); ax.invert_xaxis()
    panel_label(ax, "a")
    ax = axes[1]
    ax.errorbar(g.q, g.pk, g.pks, color=INK, marker="o", capsize=3)
    ax.set_xlabel("Hypothetical detector quality q"); ax.set_ylabel("Peak infection at optimum (%)"); ax.invert_xaxis()
    panel_label(ax, "b")
    ax = axes[2]
    d = pd.read_csv(RESULTS / "c1_detector_to_policy.csv")
    ax.scatter(d.detector_AUC, d.x1, color=BLUE, marker="o", s=40, label="x₁ firewall", zorder=3)
    ax.scatter(d.detector_AUC, d.x2, color=ORANGE, marker="s", s=40, label="x₂ automated response", zorder=3)
    for _, r in d.iterrows():
        name = r.detector.replace("plugin:", "")
        if r.detector_AUC < 0.995:
            ax.annotate(name, (r.detector_AUC, r.x1), fontsize=7, color=INK2, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Real detector ROC-AUC (type vs benign, test)"); ax.set_ylabel("NIO-optimal policy")
    ax.set_ylim(0, 1.02); ax.legend(loc="center left")
    panel_label(ax, "c")
    save(fig, "fig18_detector_quality")


def fig19():
    if not have("a1_plugin_detectors_summary.csv"):
        return
    d = pd.read_csv(RESULTS / "a1_plugin_detectors_summary.csv").sort_values("test_AUC_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    y = np.arange(len(d))
    labels = [f"{p} ({n})" for p, n in zip(d.plugin, d.n_features)]
    axes[0].barh(y, d.test_AUC_mean, xerr=d.test_AUC_std, color=BLUE, height=0.65, error_kw={"ecolor": INK2, "lw": 1})
    axes[0].set_yticks(y, labels); axes[0].set_xlabel("Test ROC-AUC (binary, group split)"); axes[0].set_xlim(0.4, 1.0)
    axes[0].axvline(0.5, color=MUTED, ls=":", lw=1)
    w = 0.38
    axes[1].barh(y + w / 2, 100 * (1 - d.TPR_mean), w, xerr=100 * d.TPR_std, color=ORANGE, label="missed malware (1 − TPR)")
    axes[1].barh(y - w / 2, 100 * d.FPR_mean, w, xerr=100 * d.FPR_std, color=AQUA, label="false alarms (FPR)")
    axes[1].set_xlabel("Rate at threshold 0.5 (%)"); axes[1].legend(loc="lower right")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save(fig, "fig19_plugin_detectors")


def main():
    setup()
    for f in [fig01, fig02, fig03, fig04, fig06, fig07, fig08, fig09, fig10, fig11, fig13, fig14, fig15, fig16, fig18, fig19]:
        try:
            f()
        except Exception as e:  # keep going; report
            import traceback
            print(f"FAILED {f.__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
