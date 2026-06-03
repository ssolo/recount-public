"""Focused L(0) goodness-of-fit plot for the canonical min=4 configurations.

Three panels side-by-side, each showing predicted vs empirical absolute
family counts per total-copy-count bin k:

  Panel A: mc = 1 no filter (reference — the fit that matches the data)
  Panel B: sum >= 4 (Csurős canonical full-tree filter)
  Panel C: union-min4 (sum >= 4 in some subset's leaves)

Two figures are produced, same format:

  l0_gof_min4_focus.png      bins k = 1..7 and >= 8
  l0_gof_min4_focus_k30.png  bins k = 1..29 and >= 30  (tail view)

Bars below a fit's own filter threshold (k < Omin) are hatched: the
model never saw those families during fitting, so the predicted
count there is the rate process's self-prediction of the sub-filter
stratum.  Explanatory text lives in the README caption.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from validation._shared import load_dataset
from validation import _pubstyle as ps
from recount.native_backend import unobserved_logL0_native

OUT = Path("validation/outputs")

# Three focal fits: the mc=1 reference (panel A, the fit that matches
# the data) first, then the two Omega_min=4 filtered fits (panels B
# and C).  Predicted-bar colour matches the log-log plot per fit:
# C_OMIN1 (blue) for mc=1, C_OMIN4 (vermilion) for the Csurős sum>=4
# fit, C_BROWN (purple) for union-min4.
FITS = [
    # (label, dirname, F_visible, Omega_min, predicted-bar colour)
    ("mc = 1  (reference)",
     "brownian_arc269", 90_243, 1, ps.C_OMIN1),
    ("sum ≥ 4  (Csurős)",
     "brownian_arc269_omin4_sum4", 11_554, 4, ps.C_OMIN4),
    ("union-min4",
     "brownian_arc269_union_subsets_min4", 11_148, 4, ps.C_BROWN),
]

KMAX = 31               # compute L(0, 1..KMAX+1)

# Empirical histogram of arc269 sum bins
tree, profiles, _, _, _ = load_dataset("arc269")
sums = profiles.sum(axis=1)
emp_bins = np.bincount(sums, minlength=KMAX + 2)


def get_rates(dirname):
    z = np.load(OUT / dirname / "arc269_sigma1.0_final_rates.npz")
    return z["gain"], z["loss"], z["dup"], z["length"]


def fit_l0(fit_dir, F_vis, Omin):
    """L(0,1..KMAX+1) under the fitted rates, plus F* at the fit's filter."""
    g, l, d, t = get_rates(fit_dir)
    L0 = [float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies=N)))
          for N in range(1, KMAX + 2)]               # L0[i] = L(0, i+1)
    Fstar = F_vis / (1 - L0[Omin - 1])
    return dict(L0=L0, Fstar=Fstar, L0_at_1=L0[0],
                L0_at_filter=L0[Omin - 1], Omin=Omin, F_vis=F_vis)


def make_figure(stats_by_label, k_last, outfile):
    """One 3-panel figure with bins k = 1..k_last-1 and ">=k_last".

    Explanation lives in the README caption, not on the axes.
    """
    bins = list(range(1, k_last))
    tail = f">={k_last}"
    all_bins = bins + [tail]
    x_pos = np.arange(len(all_bins), dtype=float)
    x_labels = [str(k) for k in bins] + [f"≥ {k_last}"]
    show_ratios = (len(all_bins) <= 9)   # only label every bar pair when sparse

    emp = [int(emp_bins[k]) for k in bins] + [int((sums >= k_last).sum())]

    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_2COL, 3.1), sharey=True)
    BAR_W = 0.40

    for panel, (ax, (label, dirname, F_vis, Omin, pred_color)) in enumerate(
            zip(axes, FITS)):
        s = stats_by_label[label]
        L0, Fstar = s["L0"], s["Fstar"]
        pred = [Fstar * (L0[k] - L0[k - 1]) for k in bins]
        pred.append(Fstar * (1 - L0[k_last - 1]))

        ax.bar(x_pos - BAR_W / 2, emp, BAR_W, color=ps.C_DATA,
               edgecolor='white', linewidth=0.4, zorder=4)
        for xi, k, p in zip(x_pos, all_bins, pred):
            is_sub = isinstance(k, int) and k < Omin
            ax.bar(xi + BAR_W / 2, p, BAR_W, color=pred_color,
                   edgecolor='white', linewidth=0.4,
                   hatch='///' if is_sub else None, alpha=0.92, zorder=5)

        if show_ratios:
            # pred/emp ratio for each bar pair, on one tidy row just
            # above the data so the labels never collide with the bars
            for xi, e, p in zip(x_pos, emp, pred):
                r = p / e if e > 0 else float("inf")
                r_lbl = f"×{r:.2f}" if r >= 0.01 else f"×{r:.3f}"
                ax.text(xi, 0.93, r_lbl, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=6, color=ps.GREY,
                        rotation=90, zorder=10)

        if Omin > 1:
            ax.axvline(Omin - 1.5, color=ps.GREY, linestyle=':',
                       linewidth=1.0, zorder=2)
            # label hugs the filter line, just right of it; placed low on
            # the sparse figure (the ratio-label row fills the top there),
            # high on the dense figure (no ratio row to avoid)
            ax.text(Omin - 1.3, 0.04 if show_ratios else 0.92,
                    f"filter Ω ≥ {Omin}", transform=ax.get_xaxis_transform(),
                    ha='left', va='bottom' if show_ratios else 'top',
                    fontsize=7, color=ps.GREY, rotation=90)

        if show_ratios:
            # sparse figure: a tick at every bin
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
        else:
            # dense (tail) figure: thin to every 5th bin + the tail bar
            keep = [i for i, k in enumerate(bins) if k % 5 == 0]
            keep += [len(all_bins) - 1]
            ax.set_xticks([x_pos[i] for i in keep])
            ax.set_xticklabels([x_labels[i] for i in keep])
        ax.set_xlabel("family size, k")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 6e6 if show_ratios else 2.5e5)
        ps.grid(ax, axis='y')
        ax.grid(True, axis='y', which='minor', alpha=0.12, linestyle=':')
        ax.set_title(label, pad=6)
        ps.panel_label(ax, "ABC"[panel], dx=-0.13, dy=1.13)

    axes[0].set_ylabel("number of gene families")

    legend_handles = [
        mpatches.Patch(color=ps.C_DATA, label='empirical  E(k)'),
        mpatches.Patch(color=ps.C_OMIN1, label='predicted  (mc = 1)'),
        mpatches.Patch(color=ps.C_OMIN4, label='predicted  (sum ≥ 4)'),
        mpatches.Patch(color=ps.C_BROWN, label='predicted  (union-min4)'),
        mpatches.Patch(facecolor='white', edgecolor=ps.C_DATA, hatch='///',
                       label='bin below the fit\'s own filter'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=5, columnspacing=1.2)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.245, top=0.87,
                        wspace=0.18)
    ps.save_fig(fig, outfile.with_suffix(""))
    plt.close(fig)
    print(f"  wrote {outfile.name} + .pdf")


def main():
    stats = {label: fit_l0(d, F, Omin)
             for (label, d, F, Omin, _) in FITS}

    make_figure(stats, k_last=8, outfile=OUT / "l0_gof_min4_focus.png")
    make_figure(stats, k_last=30, outfile=OUT / "l0_gof_min4_focus_k30.png")

    # Text summary (k = 1..7, >=8) for the README
    lines = ["Canonical min=4 GoF + mc=1 reference (arc269): predicted vs empirical per bin\n"]
    for label, dirname, F_vis, Omin, _ in FITS:
        s = stats[label]
        L0, Fstar = s["L0"], s["Fstar"]
        lines.append(
            f"\n{label}  (F={F_vis:,}, L(0,1)={s['L0_at_1']:.4f}, "
            f"L(0,{Omin})={s['L0_at_filter']:.4f}, F*={Fstar:,.0f})")
        lines.append("  k       empirical   predicted   pred/emp")
        for k in range(1, 8):
            p = Fstar * (L0[k] - L0[k - 1])
            e = int(emp_bins[k])
            lines.append(f"  {k:>4}  {e:>10,}  {p:>10,.1f}   "
                         f"{p/e if e>0 else float('inf'):>7.3f}")
        p = Fstar * (1 - L0[7])
        e = int((sums >= 8).sum())
        lines.append(f"  {'>=8':>4}  {e:>10,}  {p:>10,.1f}   {p/e:>7.3f}")
    Path(OUT / "l0_gof_min4_focus.txt").write_text("\n".join(lines) + "\n")
    print("  wrote l0_gof_min4_focus.txt")


if __name__ == "__main__":
    main()
