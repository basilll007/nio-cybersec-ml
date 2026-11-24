"""CIC-MalMem-2022 loading, preprocessing and splitting.

Source: Hugging Face `bvk/CIC-MalMem-2022` loaded with `datasets.load_dataset` (Apache-2.0).
Columns: 55 VolMemLyzer features, `Class` (Benign/Malware), `Category` (Benign or <Type>-<Family>),
`Filename` (<Type>-<Family>-<sha256>-<dump>.raw for malware, 'Benign' for every benign row).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import DATA

HF_ID = "bvk/CIC-MalMem-2022"
CACHE_PATH = DATA / "MalMem2022.parquet"
TYPES = ["Benign", "Ransomware", "Spyware", "Trojan"]
MALWARE_TYPES = ["Ransomware", "Spyware", "Trojan"]
_FNAME_RE = re.compile(r"^(?P<type>[A-Za-z]+)-(?P<family>[A-Za-z0-9]+)-(?P<hash>[0-9a-fA-F]{16,})-(?P<dump>\d+)\.raw$")


def download() -> pd.DataFrame:
    """Fetch the dataset with Hugging Face `datasets` and cache a local parquet copy in data/."""
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from datasets import load_dataset
    ds = load_dataset(HF_ID)
    df = ds["train"].to_pandas()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def load_raw() -> pd.DataFrame:
    df = download().copy()
    df["MalwareType"] = df["Category"].map(lambda c: "Benign" if c == "Benign" else c.split("-")[0])
    df["Family"] = df["Category"].map(lambda c: "Benign" if c == "Benign" else c.split("-", 1)[1])
    groups = []
    n_unparsed = 0
    for i, (fn, cls) in enumerate(zip(df["Filename"], df["Class"])):
        if cls == "Benign":
            groups.append(f"benign_{i}")  # no binary identity available for benign dumps
            continue
        m = _FNAME_RE.match(str(fn))
        if m is None:
            n_unparsed += 1
            groups.append(f"unparsed_{fn}")
        else:
            groups.append(m.group("hash").lower())
    df["Group"] = groups
    df.attrs["n_unparsed_filenames"] = n_unparsed
    assert set(df["MalwareType"]) == set(TYPES), set(df["MalwareType"])
    return df


def feature_info(df: pd.DataFrame) -> dict:
    """Feature columns, the constant columns dropped, and which columns are counts (log1p)."""
    feats = [c for c in df.columns if c not in ("Class", "Category", "Filename", "MalwareType", "Family", "Group")]
    nun = df[feats].nunique()
    constant = [c for c in feats if nun[c] <= 1]
    kept = [c for c in feats if c not in constant]
    # count features = integer-typed, non-negative columns (process/handle/dll/... counts)
    counts = [c for c in kept if pd.api.types.is_integer_dtype(df[c]) and df[c].min() >= 0]
    return {"all": feats, "constant_dropped": constant, "kept": kept, "count_log1p": counts}


def _group_split_indices(df: pd.DataFrame, seed: int, val: float = 0.15, test: float = 0.15):
    """GroupShuffleSplit run inside each Category stratum (benign rows are singleton groups),
    so all dumps of one malware binary land in exactly one split and family balance is kept."""
    tr, va, te = [], [], []
    # a few hashes occur under two family labels: stratify by the group's first label so that a
    # binary can never straddle two splits
    stratum = df.groupby("Group")["Category"].transform("first")
    for k, (_, sub) in enumerate(df.groupby(stratum, sort=True)):
        idx = sub.index.to_numpy()
        g = sub["Group"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed + 1000 * k)
        a, b = next(gss.split(idx, groups=g))
        te.append(idx[b])
        idx2, g2 = idx[a], g[a]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed + 1000 * k + 1)
        c, d = next(gss2.split(idx2, groups=g2))
        tr.append(idx2[c]); va.append(idx2[d])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def _random_split_indices(df: pd.DataFrame, seed: int, val: float = 0.15, test: float = 0.15):
    idx = df.index.to_numpy()
    strat = df["Category"].to_numpy()
    a, te = train_test_split(idx, test_size=test, random_state=seed, stratify=strat)
    tr, va = train_test_split(a, test_size=val / (1 - test), random_state=seed, stratify=df.loc[a, "Category"].to_numpy())
    return tr, va, te


def split_indices(df: pd.DataFrame, kind: str, seed: int):
    if kind == "random":
        return _random_split_indices(df, seed)
    if kind == "group":
        return _group_split_indices(df, seed)
    raise ValueError(kind)


@dataclass
class MalMemSplit:
    kind: str
    seed: int
    features: list
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    scaler: StandardScaler = field(repr=False, default=None)
    extra: dict = field(default_factory=dict)

    def y(self, part: str, task: str) -> np.ndarray:
        meta = getattr(self, f"meta_{part}")
        if task == "binary":
            return (meta["Class"] == "Malware").to_numpy().astype(np.int64)
        if task == "4class":
            return meta["MalwareType"].map({t: i for i, t in enumerate(TYPES)}).to_numpy().astype(np.int64)
        raise ValueError(task)


def _transform(df: pd.DataFrame, info: dict) -> np.ndarray:
    X = df[info["kept"]].astype(np.float64).copy()
    for c in info["count_log1p"]:
        X[c] = np.log1p(X[c])
    return X.to_numpy()


def make_split(df: pd.DataFrame, kind: str, seed: int, train_mask_types: list | None = None) -> MalMemSplit:
    """Build a preprocessed split.  `train_mask_types`: if given, only rows whose MalwareType is in this
    list are kept in train/val (used for leave-one-malware-type-out); test keeps every type."""
    info = feature_info(df)
    tr, va, te = split_indices(df, kind, seed)
    if train_mask_types is not None:
        keep = set(train_mask_types)
        tr = np.array([i for i in tr if df.at[i, "MalwareType"] in keep])
        va = np.array([i for i in va if df.at[i, "MalwareType"] in keep])
    Xall = _transform(df, info)
    scaler = StandardScaler().fit(Xall[tr])  # fit on train only
    meta_cols = ["Class", "Category", "MalwareType", "Family", "Group", "Filename"]
    return MalMemSplit(
        kind=kind, seed=seed, features=info["kept"],
        X_train=scaler.transform(Xall[tr]).astype(np.float32),
        X_val=scaler.transform(Xall[va]).astype(np.float32),
        X_test=scaler.transform(Xall[te]).astype(np.float32),
        meta_train=df.loc[tr, meta_cols].reset_index(drop=True),
        meta_val=df.loc[va, meta_cols].reset_index(drop=True),
        meta_test=df.loc[te, meta_cols].reset_index(drop=True),
        scaler=scaler,
        extra={"train_idx": tr, "val_idx": va, "test_idx": te, "info": info},
    )


def transform_rows(df_rows: pd.DataFrame, split: MalMemSplit) -> np.ndarray:
    """Apply a split's preprocessing (log1p + its train-fitted scaler) to arbitrary rows."""
    return split.scaler.transform(_transform(df_rows, split.extra["info"])).astype(np.float32)


def dataset_stats(df: pd.DataFrame) -> dict:
    info = feature_info(df)
    mal = df[df["Class"] == "Malware"]
    return {
        "rows": int(len(df)),
        "n_features_raw": len(info["all"]),
        "n_features_kept": len(info["kept"]),
        "constant_dropped": info["constant_dropped"],
        "count_log1p": info["count_log1p"],
        "class_counts": df["Class"].value_counts().to_dict(),
        "type_counts": df["MalwareType"].value_counts().to_dict(),
        "family_counts": df["Category"].value_counts().to_dict(),
        "n_malware_binaries": int(mal["Group"].nunique()),
        "dumps_per_binary_mean": float(mal.groupby("Group").size().mean()),
        "dumps_per_binary_max": int(mal.groupby("Group").size().max()),
        "n_unparsed_filenames": int(df.attrs.get("n_unparsed_filenames", 0)),
        "n_duplicate_feature_rows": int(df.duplicated(subset=info["kept"]).sum()),
    }
