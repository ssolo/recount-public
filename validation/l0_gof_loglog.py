"""Log-log sum-bin count histogram: predicted vs empirical.

Shows exactly the quantity that goes into the GoF ratio: F* · P(Ω=k)
(predicted family count at each sum bin) and E(k) (empirical count at
each sum bin), both on log-log axes so the full empirical tail (k up
to ~1000) is visible alongside the singleton-dominated head.

Three panels:
  Panel A: mc = 1 no filter -- the reference fit that DOES fit
  Panel B: sum >= 4 (Csurös full-tree filter) -- the canonical
           Ωmin = 4 fit
  Panel C: union-min4 -- the alternative Ωmin = 4 fit

Behind every panel the full sum >= 1..10 observation-threshold family
is drawn as faint thin grey lines, so each focal fit is read against
the whole sweep of thresholds, not just on its own.

Same axes across all three panels for direct comparison.  The
predicted curves are cached to /tmp; delete the cache to recompute.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from validation._shared import load_dataset
from validation import _pubstyle as ps
from recount.native_backend import unobserved_logL0_native

OUT = Path("validation/outputs")
CACHE = Path("/tmp/l0_gof_loglog_cache.npz")

# Three focal fits, one per panel.  Reference (mc=1) first so the reader
# sees "this is what a fit that matches the data looks like" before the
# two filter-4 fits that don't.  Predicted-curve colour: C_OMIN1 (blue)
# for the mc=1 reference, C_OMIN4 (vermilion) for the Csurős sum>=4 fit,
# C_BROWN (purple) for the union-min4 fit.  Each key indexes the cached
# curve dict built by compute_curves().
FITS = [
    ("mc = 1  (reference)",  "thr1",  1, ps.C_OMIN1),
    ("sum ≥ 4  (Csurős)",    "thr4",  4, ps.C_OMIN4),
    ("union-min4",           "union", 4, ps.C_BROWN),
]

# Compute L(0, N) up to this k.  Empirical sum-bin mass past k≈50 is
# very sparse (< 200 families per bin); the native L(0,N) call is
# O(N) per call but the inside-tensor build scales steeply with W so
# K_MAX=60 keeps the script runnable in ~minutes rather than tens of
# minutes while still covering the empirically informative tail.
K_MAX = 60

# Empirical histogram
tree, profiles, _, _, _ = load_dataset("arc269")
sums = profiles.sum(axis=1)
emp_bins = np.bincount(sums, minlength=K_MAX + 2)


def get_rates(dirname):
    z = np.load(OUT / dirname / "arc269_sigma1.0_final_rates.npz")
    return z["gain"], z["loss"], z["dup"], z["length"]


def predicted_curve(dirname, F_vis, Omin, k_max=K_MAX):
    """Return predicted family counts F* · P(Ω = k) at each k = 1..k_max."""
    g, l, d, t = get_rates(dirname)
    L0 = np.array([float(np.exp(unobserved_logL0_native(
        tree, g, l, d, t, min_copies=N))) for N in range(1, k_max + 2)])
    Fstar = F_vis / (1 - L0[Omin - 1])
    # P(Ω = k) = L(0, k+1) - L(0, k) for k >= 1; index L0[k-1] is L(0, k)
    P = L0[1:] - L0[:-1]   # P[i] = P(Ω = i+1)
    return Fstar * P


def threshold_dir(N):
    """Canonical fit subdir for the sum >= N filter (N = 1 is no filter)."""
    return "brownian_arc269" if N == 1 else f"brownian_arc269_omin4_sum{N}"


def compute_curves():
    """Predicted F*·P(Ω=k) for every sum>=N threshold (N = 1..10) plus the
    union-min4 fit.  Heavy (native L(0,N) calls); cached to /tmp."""
    if CACHE.exists():
        print(f"  using cached curves ({CACHE})")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = {}
    for N in range(1, 11):
        F_vis = int((sums >= N).sum())
        print(f"  threshold sum>={N:<2d} (F = {F_vis:,}) ...")
        out[f"thr{N}"] = predicted_curve(threshold_dir(N), F_vis, N)
    print("  union-min4 ...")
    out["union"] = predicted_curve(
        "brownian_arc269_union_subsets_min4", 11_148, 4)
    np.savez(CACHE, **out)
    return out


def main():
    curves = compute_curves()
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_2COL, 3.2),
                             sharey=True, sharex=True)

    ks = np.arange(1, K_MAX + 1)
    emp_counts = emp_bins[1:K_MAX + 1].astype(float)
    thr_curves = [curves[f"thr{N}"] for N in range(1, 11)]

    for panel, (ax, (label, key, Omin, pred_color)) in enumerate(
            zip(axes, FITS)):

        # Sub-filter shading (k < Omin)
        if Omin > 1:
            ax.axvspan(0.7, Omin - 0.5, color=ps.C_SHADE, lw=0, zorder=0)

        # Faint thin-line fan: the full sum>=1..10 threshold family, so the
        # focal fit is seen against every other observation threshold.
        for tc in thr_curves:
            nz = tc > 1e-10
            ax.plot(ks[nz], tc[nz], color=ps.GREY, lw=0.6, alpha=0.30,
                    zorder=3)

        # Empirical: scatter + line (skip zero bins for the line)
        emp_nz = emp_counts > 0
        ax.plot(ks[emp_nz], emp_counts[emp_nz], color=ps.C_DATA,
                marker='o', markersize=3.2, lw=0.9,
                markeredgecolor='white', markeredgewidth=0.4,
                zorder=10, alpha=0.95)
        # Focal predicted: line + markers, emphasized
        pred = curves[key]
        pred_nz = pred > 1e-10
        ax.plot(ks[pred_nz], pred[pred_nz], color=pred_color,
                marker='s', markersize=2.8, lw=1.4,
                markeredgecolor='white', markeredgewidth=0.3,
                zorder=8, alpha=0.95)

        # Mark fit's filter threshold
        if Omin > 1:
            ax.axvline(Omin - 0.5, color=ps.GREY, linestyle=':', lw=1.0,
                       zorder=2)
            ax.text(Omin + 0.2, 1.0, f"Ω ≥ {Omin}", color=ps.GREY,
                    fontsize=7, va='bottom', ha='left')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.7, K_MAX * 1.05)
        ax.set_ylim(0.3, 2.5e5)
        ps.grid(ax)
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        ax.set_xlabel("family size, k")
        ax.set_title(label, pad=6)
        ps.panel_label(ax, "ABC"[panel], dx=-0.13, dy=1.13)

    axes[0].set_ylabel("number of gene families")

    # Compact shared legend below the panels (the explanatory text lives
    # in the README caption, not painted on the figure).
    leg_handles = [
        mlines.Line2D([], [], color=ps.C_DATA, marker='o', markersize=3.6,
                      lw=1.0, markeredgecolor='white', markeredgewidth=0.4,
                      label='empirical  E(k)'),
        mlines.Line2D([], [], color=ps.C_OMIN1, marker='s', markersize=3.4,
                      lw=1.4, label='predicted, mc = 1'),
        mlines.Line2D([], [], color=ps.C_OMIN4, marker='s', markersize=3.4,
                      lw=1.4, label='predicted, sum ≥ 4'),
        mlines.Line2D([], [], color=ps.C_BROWN, marker='s', markersize=3.4,
                      lw=1.4, label='predicted, union-min4'),
        mlines.Line2D([], [], color=ps.GREY, lw=0.9, alpha=0.7,
                      label='every sum ≥ 1…10 threshold'),
    ]
    fig.legend(handles=leg_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=5, columnspacing=1.1,
               handletextpad=0.5, fontsize=6.8)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.255, top=0.87,
                        wspace=0.16)
    ps.save_fig(fig, OUT / "l0_gof_loglog")
    plt.close(fig)
    print("  wrote l0_gof_loglog.png + .pdf")


if __name__ == "__main__":
    main()
