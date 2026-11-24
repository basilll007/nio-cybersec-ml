"""Matplotlib style + save helpers.  Colour-blind-safe categorical palette (validated default order from the
dataviz reference palette; scatter/all-pairs use is capped at the first three slots), one-hue sequential ramp
for ordinal quantities (P, sigma), recessive grid, no baked-in titles, 300-dpi PNG + vector PDF."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import FIGURES

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
FULLDATA = "#0b0b0b"

# entity -> colour (fixed, never by rank)
ENTITY = {"primer": BLUE, "subset": ORANGE, "noise": AQUA, "fulldata": FULLDATA,
          "known": BLUE, "novel (LOTO)": ORANGE, "S": BLUE, "I": ORANGE, "R": AQUA,
          "Benign": BLUE, "Ransomware": ORANGE, "Spyware": AQUA, "Trojan": YELLOW,
          "train": BLUE, "holdout": ORANGE}


def seq(n: int):
    """n ordinal colours from the blue ramp (light -> dark), never lighter than step 250."""
    if n == 1:
        return [SEQ_BLUE[3]]
    import numpy as np
    idx = np.linspace(0, len(SEQ_BLUE) - 1, n).round().astype(int)
    return [SEQ_BLUE[i] for i in idx]


def setup():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.prop_cycle": matplotlib.cycler(color=CAT),
    })


def save(fig, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved", name)


def panel_label(ax, text: str):
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)
