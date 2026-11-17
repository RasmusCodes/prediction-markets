"""Shared figure style for the paper (reference data-viz palette, light surface, validated slots 1-3 all-pairs).

Rules followed: categorical hues in fixed order (blue, orange, aqua) and never cycled; at most three series per
chart; one y-axis per chart; thin marks, hairline grid; text in ink colours, never series colours.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3: pass all-pairs CVD/normal-vision checks
BLUE, ORANGE, AQUA = SERIES
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VENUE_COLOR = {"Kalshi": BLUE, "Polymarket": ORANGE, "Simulation": AQUA}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "text.parse_math": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> str:
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    return path
