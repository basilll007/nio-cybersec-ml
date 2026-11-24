"""Shared helpers: paths, seeding, device selection, JSON/CSV I/O, run-time logging."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_runtime(step: str, seconds: float, note: str = "") -> None:
    """Append one line to results/runlog.csv (used for the 'Work done' log)."""
    path = RESULTS / "runlog.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("timestamp,step,seconds,device,note\n")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        note = note.replace(",", ";")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{step},{seconds:.1f},{dev},{note}\n")


@contextmanager
def timed(step: str, note: str = ""):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log_runtime(step, dt, note)
    print(f"[{step}] done in {dt:.1f}s")


def mean_std(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=0)), "n": int(a.size),
            "values": [float(v) for v in a]}


def fmt_ms(d: dict, digits: int = 3) -> str:
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"
