"""Shared publication-figure style for every plot in README.md.

One font, one colour palette, one set of axes conventions, one output
device — so the whole README figure set is visually coherent.  The look
is the one established by ``validation/pnas_letter_v2_figure.py`` (PNAS
two-column width, Okabe-Ito colour-blind-safe palette, sans-serif,
minimal in-figure text, vector PDF + raster PNG output).

Usage in a figure script::

    from validation._pubstyle import apply, save_fig, panel_label
    from validation import _pubstyle as ps
    apply()                              # install rcParams — call once
    fig, ax = plt.subplots(figsize=(ps.WIDTH_2COL, 3.4))
    ax.plot(x, y, color=ps.C_OMIN1)
    panel_label(ax, "A")
    save_fig(fig, "validation/outputs/myfig")   # writes .png AND .pdf

Design rules every README figure follows:
  * sans-serif, ~7-9 pt; no oversized titles — explanation belongs in
    the README caption / a compact legend, not painted on the axes;
  * top and right spines off; a faint dashed grid only where it helps;
  * the Okabe-Ito palette below — never ad-hoc colours;
  * vector PDF for the LaTeX/PDF build, 400-dpi PNG for GitHub.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- figure widths (inches) — PNAS column sizes -----------------------
WIDTH_1COL  = 3.42      # 8.7 cm   — single column
WIDTH_1HALF = 4.49      # 11.4 cm  — 1.5 column
WIDTH_2COL  = 7.00      # 17.8 cm  — full two-column width


# --- Okabe-Ito colour-blind-safe palette ------------------------------
BLACK     = "#000000"
BLUE      = "#0072B2"
VERMILION = "#D55E00"
ORANGE    = "#E69F00"
SKYBLUE   = "#56B4E9"
GREEN     = "#009E73"
YELLOW    = "#F0E442"
PURPLE    = "#CC79A7"
GREY      = "#5A5A5A"
LIGHTGREY = "#D9D9D9"
PALEGREY  = "#F2F2F2"

#: default colour cycle for multi-series plots
CYCLE = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKYBLUE, GREY]

# --- semantic aliases used across the recount figures -----------------
C_DATA   = BLACK       # observed / empirical data — the reference truth
C_OMIN1  = BLUE        # the unfiltered / Ωmin = 1 fit
C_OMIN4  = VERMILION   # the Csűrös Ωmin = 4 fit
C_BROWN  = PURPLE      # the MAP-Brownian fit
C_NEUTRAL = GREY       # neutral lines / sweeps
C_VIOLIN_FACE = LIGHTGREY
C_VIOLIN_EDGE = GREY
C_SHADE  = PALEGREY    # shaded sub-filter / reference regions


def apply() -> None:
    """Install the shared rcParams.  Call once before building a figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.6,
        "lines.markeredgewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "pdf.fonttype": 42,     # embed editable TrueType text in the PDF
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04,
                 fontsize: float = 11.0) -> None:
    """Bold panel letter just outside the top-left corner of ``ax``."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="left")


def grid(ax, axis: str = "both") -> None:
    """Apply the standard faint dashed grid (behind the data)."""
    ax.grid(True, axis=axis, which="major", lw=0.4, alpha=0.30,
            linestyle="--", zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, stem, dpi: int = 400, pad: float = 0.02) -> None:
    """Write ``<stem>.pdf`` (vector, for the LaTeX build) and
    ``<stem>.png`` (raster preview, for GitHub).  ``stem`` is a path
    with no extension; parent directories are created as needed."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi,
                bbox_inches="tight", pad_inches=pad)
