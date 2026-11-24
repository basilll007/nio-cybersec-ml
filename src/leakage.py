"""Leakage audit for NIO primers: does a primer reveal *records* or only *structure*?

1. Distance-to-closest-record (DCR) membership test, the standard synthetic-data privacy check: a fixed random
   subset S of the training rows with |S| = |held-out| is drawn, and for each primer row we compute the NN
   distance to S (d_train) and to the held-out rows (d_holdout).  Under "no memorisation" a primer row is equally
   likely to be closer to either set (share = 0.5).  Reported: distance ratio d_train/d_holdout, share closer to
   train with a one-sided binomial test, the one-sided Mann-Whitney U test on the two distance samples (spec),
   and a paired one-sided Wilcoxon signed-rank test on log(d_train/d_holdout) (more powerful, rows are paired).
   Calibration controls: near-copies of training rows in S (row + N(0, 0.05^2) noise; positive control that
   any real memorisation would look like), real validation rows (non-members, null control), Gaussian noise.
2. Structure test: primer class means vs real class means (cosine similarity of the class-difference vector,
   overlap of the top features ranked by |mean difference|, Spearman rank correlation).
3. PCA projection of real data with primer points overlaid.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import binomtest, mannwhitneyu, spearmanr, wilcoxon
from sklearn.decomposition import PCA


def _nn_dist(A: np.ndarray, B: np.ndarray, device: str = "cpu", chunk: int = 4096) -> np.ndarray:
    At = torch.as_tensor(A, dtype=torch.float32, device=device)
    out = torch.full((At.shape[0],), float("inf"), device=device)
    for i in range(0, B.shape[0], chunk):
        Bt = torch.as_tensor(B[i:i + chunk], dtype=torch.float32, device=device)
        out = torch.minimum(out, torch.cdist(At, Bt).min(dim=1).values)
    return out.cpu().numpy()


def reference_subset(n_train: int, n_hold: int, seed: int) -> np.ndarray:
    """Indices of the fixed, size-matched training reference set S."""
    rng = np.random.default_rng(50_000 + seed)
    return np.sort(rng.choice(n_train, size=min(n_hold, n_train), replace=False))


def nn_distances(points: np.ndarray, X_ref_train: np.ndarray, X_hold: np.ndarray, device: str = "cpu"):
    """NN distance of each point to the size-matched training reference set and to the held-out set."""
    return _nn_dist(points, X_ref_train, device), _nn_dist(points, X_hold, device)


def memorised_control(X_ref_train: np.ndarray, n: int, seed: int, noise: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(60_000 + seed)
    rows = X_ref_train[rng.choice(len(X_ref_train), n, replace=False)]
    return (rows + noise * rng.standard_normal(rows.shape)).astype(np.float32)


def nn_summary(d_train: np.ndarray, d_hold: np.ndarray) -> dict:
    d_train = np.asarray(d_train, dtype=float); d_hold = np.asarray(d_hold, dtype=float)
    ratio = d_train / np.maximum(d_hold, 1e-12)
    closer = int(np.sum(d_train < d_hold))
    out = {"n_points": int(len(d_train)),
           "d_train_median": float(np.median(d_train)), "d_holdout_median": float(np.median(d_hold)),
           "ratio_median": float(np.median(ratio)), "ratio_mean": float(np.mean(ratio)),
           "frac_closer_to_train": float(closer / max(len(d_train), 1)),
           "binom_p_greater": float(binomtest(closer, len(d_train), 0.5, alternative="greater").pvalue) if len(d_train) else None}
    if len(d_train) >= 2 and not np.allclose(d_train, d_hold):
        out["mw_U"], out["mw_p_less"] = [float(v) for v in mannwhitneyu(d_train, d_hold, alternative="less")]
        diff = np.log(np.maximum(d_train, 1e-12)) - np.log(np.maximum(d_hold, 1e-12))
        out["wilcoxon_p_less"] = float(wilcoxon(diff, alternative="less").pvalue) if np.any(diff != 0) else None
    else:
        out.update({"mw_U": None, "mw_p_less": None, "wilcoxon_p_less": None})
    return out


def class_structure(primer_x: np.ndarray, primer_y_soft: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                    features: list, top: int = 15) -> dict:
    """Compare primer class means with real class means in standardised feature space."""
    k = primer_y_soft.shape[1]
    w = primer_y_soft / np.maximum(primer_y_soft.sum(0, keepdims=True), 1e-12)
    pm = (w.T @ primer_x)                                     # (k, d) soft-label weighted primer means
    rm = np.stack([X_train[y_train == c].mean(0) for c in range(k)])
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out = {"class_mean_cosine": [cos(pm[c], rm[c]) for c in range(k)]}
    # contrast of every class against class 0 (benign): which features separate the classes?
    contrasts = []
    for c in range(1, k):
        dp, dr = pm[c] - pm[0], rm[c] - rm[0]
        tp = set(np.argsort(-np.abs(dp))[:top]); tr = set(np.argsort(-np.abs(dr))[:top])
        rho = spearmanr(np.abs(dp), np.abs(dr)).statistic
        sign_agree = float(np.mean(np.sign(dp[list(tr)]) == np.sign(dr[list(tr)])))
        contrasts.append({"class": c, "diff_cosine": cos(dp, dr), "top_overlap": len(tp & tr), "top_k": top,
                          "spearman_absdiff": float(rho), "sign_agreement_on_real_top": sign_agree,
                          "recovered_top_features": [features[i] for i in sorted(tp & tr, key=lambda i: -abs(dr[i]))]})
    out["contrasts"] = contrasts
    out["diff_cosine_mean"] = float(np.mean([c["diff_cosine"] for c in contrasts]))
    out["top_overlap_mean"] = float(np.mean([c["top_overlap"] for c in contrasts]))
    return out


def pca_projection(X_train: np.ndarray, y_train: np.ndarray, extra: dict, n_show: int = 4000, seed: int = 0):
    """Fit PCA(2) on training data; return projected sample of real rows plus every `extra` point set."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=2, random_state=seed).fit(X_train)
    idx = rng.choice(len(X_train), size=min(n_show, len(X_train)), replace=False)
    rows = [{"set": "real", "label": int(y_train[i]), "pc1": float(a), "pc2": float(b)}
            for i, (a, b) in zip(idx, pca.transform(X_train[idx]))]
    for name, (pts, labels) in extra.items():
        for (a, b), l in zip(pca.transform(pts), labels):
            rows.append({"set": name, "label": int(l), "pc1": float(a), "pc2": float(b)})
    return rows, pca.explained_variance_ratio_.tolist()
