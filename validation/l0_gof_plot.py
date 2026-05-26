"""Intuitive plot of the L(0) conditional goodness-of-fit results.

Three panels side-by-side (unconditional / cond Omega>=2 / cond Omega>=3),
5 representative arc269 fits per panel:

    mc=1 no filter   - bold black, the reference
    sum >= 2         - orange  (loose filter)
    sum >= 4         - vermilion (Csurös canonical)
    sum >= 7         - dark vermilion (strict)
    sum >= 10        - darkest (strictest)

Each line: pred/emp ratio at sum bins k = 1..7 and >=8, log-y.

Horizontal dashed line at ratio = 1 (perfect fit).
Shaded green band at [0.5, 2.0] showing "within 2x of perfect".

The headline picture:

  - mc=1 line hugs y=1 in every panel and at every bin.
  - The filtered-fit lines start far below y=1 at small k and cross
    above y=1 at large k -- monotonically shifted from the data's
    shape.  The displacement grows with filter strength.
  - The conditional panels visually re-anchor each line so that the
    misfit is shape-only (mass is conserved); the filtered fits
    still fail.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from pathlib import Path
from validation._shared import load_dataset
from validation import _pubstyle as ps
from recount.native_backend import unobserved_logL0_native

OUT = Path("validation/outputs")


def _darken(hexcol, factor):
    """Return ``hexcol`` scaled toward black by ``factor`` (0..1)."""
    r, g, b = mcolors.to_rgb(hexcol)
    return (r * factor, g * factor, b * factor)


# Filter-strength gradient anchored on the shared palette: mc=1 is the
# black reference, sum>=4 is the Csurős fit so it takes C_OMIN4
# (vermilion) for consistency with the other GoF figures; the looser
# and stricter filters fan out warm around it.
C_SUM2  = ps.ORANGE                       # loosest filter
C_SUM4  = ps.C_OMIN4                      # Csurős canonical
C_SUM7  = _darken(ps.C_OMIN4, 0.72)       # strict
C_SUM10 = _darken(ps.C_OMIN4, 0.45)       # strictest

# Selected representative fits (subset of 11 available)
FITS = [
    # (label, dirname, fit's filter Omega_min, F_visible, line colour, lw, zorder)
    ("mc = 1 (no filter)", "brownian_arc269",              1, 90_243, ps.C_DATA, 2.4, 20),
    ("sum ≥ 2",            "brownian_arc269_omin4_sum2",   2, 24_904, C_SUM2,    1.6,  8),
    ("sum ≥ 4 (Csurös)",   "brownian_arc269_omin4_sum4",   4, 11_554, C_SUM4,    1.9, 12),
    ("sum ≥ 7",            "brownian_arc269_omin4_sum7",   7,  7_465, C_SUM7,    1.6,  9),
    ("sum ≥ 10",           "brownian_arc269_omin4_sum10", 10,  5_879, C_SUM10,   1.6,  7),
]

CONDITIONS = [
    (1, "unconditional",            "all bins  (k = 1, 2, ..., ≥ 8)"),
    (2, "conditioned on Ω ≥ 2",     "singletons excluded"),
    (3, "conditioned on Ω ≥ 3",     "1+2-copy families excluded"),
]

BINS = list(range(1, 8))   # 1..7
LAST = ">= 8"

# Empirical histogram of arc269 sum bins
tree, profiles, _, _, _ = load_dataset("arc269")
sums = profiles.sum(axis=1)
emp_bins = np.bincount(sums, minlength=100)
emp_ge_8 = int((sums >= 8).sum())
emp_at_k = {k: int(emp_bins[k]) for k in BINS}
emp_at_k[LAST] = emp_ge_8
emp_totals = {kc: int((sums >= kc).sum()) for kc in (1, 2, 3)}

# Compute L(0, 1..11) for every fit
def fit_stats(dirname, F_vis, Omin):
    rates = np.load(OUT / dirname / "arc269_sigma1.0_final_rates.npz")
    g, l, d, t = rates["gain"], rates["loss"], rates["dup"], rates["length"]
    L0 = [float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies=N)))
          for N in range(1, 12)]
    Fstar = F_vis / (1 - L0[Omin - 1])
    P = [L0[0]] + [L0[k] - L0[k - 1] for k in range(1, 11)]
    return dict(L0=L0, Fstar=Fstar, P=P, Omin=Omin, F_vis=F_vis)


def pred_count(stats, k, cond):
    """Predicted count at bin k under conditioning at cond.

    cond = 1: unconditional, F* * P(Omega=k).
    cond > 1: rescaled to empirical >= cond total.
    """
    L0 = stats["L0"]
    if k == LAST:
        # P(Omega >= 8) = 1 - L(0, 8)
        if cond == 1:
            return stats["Fstar"] * (1 - L0[7])
        return emp_totals[cond] * (1 - L0[7]) / (1 - L0[cond - 1])
    if k < cond:
        return None
    if cond == 1:
        return stats["Fstar"] * stats["P"][k]
    return emp_totals[cond] * stats["P"][k] / (1 - L0[cond - 1])


def main():
    stats = {label: fit_stats(d, F, Omin) for label, d, Omin, F, *_ in FITS}

    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_2COL, 3.2), sharey=True)

    x_pos = np.arange(len(BINS) + 1)
    x_labels = [str(k) for k in BINS] + [LAST]

    for panel, (ax, (cond, ptitle, psub)) in enumerate(zip(axes, CONDITIONS)):
        ax.axhspan(0.5, 2.0, color=ps.GREEN, alpha=0.12, lw=0, zorder=0)
        ax.axhline(1.0, color=ps.GREY, lw=0.9, linestyle='--', zorder=1)

        for (label, _, _, _, color, lw, zorder) in FITS:
            s = stats[label]
            xs, ys = [], []
            for i, k in enumerate(BINS + [LAST]):
                p = pred_count(s, k, cond)
                if p is None:
                    continue
                emp = emp_at_k[k]
                ratio = p / emp if emp > 0 else np.nan
                xs.append(i)
                ys.append(ratio)
            is_mc1 = label.startswith("mc = 1")
            ax.plot(xs, ys, color=color, lw=lw,
                    marker='s' if is_mc1 else 'o',
                    markersize=5.0 if is_mc1 else 3.8,
                    markeredgecolor='white',
                    markeredgewidth=0.5,
                    label=label,
                    zorder=zorder, alpha=1.0 if is_mc1 else 0.93)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("family size, k")
        ax.set_yscale("log")
        ax.set_ylim(0.005, 10)
        ps.grid(ax)
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        ax.set_title(f"{ptitle}\n{psub}", pad=6)
        ps.panel_label(ax, "ABC"[panel], dy=1.20)

    axes[0].set_ylabel("predicted / empirical count")

    # Compact shared legend below; the filter-strength ordering is the
    # legend order.  Explanatory text lives in the README caption.
    leg_handles = [
        mlines.Line2D([], [], color=c, lw=lw,
                      marker=('s' if label.startswith("mc = 1") else 'o'),
                      markersize=(5.0 if label.startswith("mc = 1") else 3.8),
                      markeredgecolor='white',
                      markeredgewidth=0.5, label=label)
        for (label, _, _, _, c, lw, _) in FITS
    ]
    fig.legend(handles=leg_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=len(FITS),
               columnspacing=1.5)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.235, top=0.86,
                        wspace=0.14)
    ps.save_fig(fig, OUT / "l0_gof_ratio_panels")
    plt.close(fig)
    print("  wrote l0_gof_ratio_panels.png + .pdf")


if __name__ == "__main__":
    main()
