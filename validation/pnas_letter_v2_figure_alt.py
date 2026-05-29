"""Alternative Figure 1 for the PNAS letter v2.

Same two-panel layout as validation/pnas_letter_v2_figure.py, but panel
A shows BOTH ancestral metrics together — the inferred number of gene
*families* and the inferred number of gene *copies* at the arc269 root
— as the observation threshold Ωmin sweeps 1..10, and carries two
reference violins so the 269 sampled genomes are shown both by family
count and by copy count.  Gene families and gene copies are told apart
by colour (green vs amber).  The Ωmin = 1 and Ωmin = 4 points on the
family sweep are picked out in blue / vermilion to link panel A to
panel B's two fitted curves.  Panel B is the Ω ≥ 2 posterior-predictive
check; both predicted curves are drawn as plain lines.

Outputs: docs/figures/pnas_letter_v2_fig1_alt.{pdf,png}

Run:  PYTHONPATH=. python3 validation/pnas_letter_v2_figure_alt.py

The heavy step (panel B's unobserved-L0 calls) is cached to /tmp so
restyling is instant; delete the cache to recompute.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from validation import _pubstyle as ps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "outputs"
FIGDIR = ROOT / "docs" / "figures"
CACHE = Path("/tmp/pnas_v2_alt_figdata.npz")

K_MAX = 60
FIT_OMIN1 = "brownian_arc269"
FIT_OMIN4 = "brownian_arc269_omin4_sum4"
OMIN = r"$\Omega_{\mathrm{min}}$"

# panel A — the two ancestral metrics, told apart by colour
C_FAM = ps.GREEN        # gene families
C_COP = ps.ORANGE       # gene copies
# panel B — the posterior-predictive check
C_DATA, C_OMIN1, C_OMIN4, C_SHADE = ps.C_DATA, ps.C_OMIN1, ps.C_OMIN4, ps.C_SHADE


# ======================================================================
# data
# ======================================================================
def compute() -> dict:
    from validation._shared import load_dataset
    from recount.native_backend import unobserved_logL0_native

    print("loading arc269 ...")
    tree, profiles, *_ = load_dataset("arc269")
    sums = profiles.sum(axis=1)

    # panel B — observed family-size histogram + conditional predicted curves
    emp = np.bincount(sums, minlength=K_MAX + 2)[1:K_MAX + 1].astype(float)
    n_ge2 = float((sums >= 2).sum())

    def cond_predicted(dirname: str) -> np.ndarray:
        z = np.load(OUT / dirname / "arc269_sigma1.0_final_rates.npz")
        g, l, d, t = z["gain"], z["loss"], z["dup"], z["length"]
        L0 = np.array([
            float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies=N)))
            for N in range(1, K_MAX + 2)])
        P = L0[1:] - L0[:-1]
        return n_ge2 * P / (1.0 - L0[1])

    print("panel B — conditional predicted curves ...")
    pred1 = cond_predicted(FIT_OMIN1)
    pred4 = cond_predicted(FIT_OMIN4)

    # panel A — root gene families AND gene copies as Ωmin sweeps 1..10
    sweep_fam, sweep_cop = [], []
    for N in range(1, 11):
        dd = "brownian_arc269" if N == 1 else f"brownian_arc269_omin4_sum{N}"
        f = json.load(open(OUT / dd / "arc269_sigma1.0_summary.json"))["final"]
        sweep_fam.append(f["root_families_corr"])
        sweep_cop.append(f["root_copies_corr"])

    # panel A — per-genome family count and per-genome total copy count
    genome_fam = (profiles > 0).sum(axis=0).astype(float)
    genome_cop = profiles.sum(axis=0).astype(float)

    out = dict(emp=emp, pred1=pred1, pred4=pred4,
               sweep_fam=np.array(sweep_fam), sweep_cop=np.array(sweep_cop),
               genome_fam=genome_fam, genome_cop=genome_cop)
    np.savez(CACHE, **out)
    return out


def load_data() -> dict:
    if CACHE.exists():
        print(f"using cached data ({CACHE})")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    return compute()


# ======================================================================
# figure
# ======================================================================
def _violin(ax, data, pos, color, width=0.92):
    """A soft single violin in `color` with a median bar and a faint
    strip of the underlying genome points."""
    vp = ax.violinplot([data], positions=[pos], widths=width,
                       showmeans=False, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.32)
        body.set_linewidth(0.8)
    vp["cmedians"].set_color(color)
    vp["cmedians"].set_linewidth(1.4)
    rng = np.random.default_rng(0)
    jit = pos + (rng.random(data.size) - 0.5) * width * 0.44
    ax.scatter(jit, data, s=2.4, color=color, alpha=0.40, lw=0, zorder=5)


def panel_sweep(ax, d):
    """Inferred root gene families AND gene copies vs Ωmin, with the
    269-genome family- and copy-count violins for reference."""
    xs = np.arange(1, 11)
    fam, cop = d["sweep_fam"], d["sweep_cop"]
    gfam, gcop = d["genome_fam"], d["genome_cop"]
    x_vio_f, x_vio_c = 11.35, 12.45

    # the two root sweeps — copies above, families below
    ax.plot(xs, cop, "-o", color=C_COP, lw=1.7, ms=4.2, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(xs, fam, "-o", color=C_FAM, lw=1.7, ms=4.2, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5)

    # Ωmin = 1 and Ωmin = 4 highlight dots on the family sweep — they
    # tie panel A to panel B's two predicted curves (matching colours)
    ax.plot(1, fam[0], "o", color=C_OMIN1, ms=7.5, zorder=6,
            markeredgecolor="white", markeredgewidth=0.9)
    ax.plot(4, fam[3], "o", color=C_OMIN4, ms=7.5, zorder=6,
            markeredgecolor="white", markeredgewidth=0.9)

    # value labels at the Ωmin = 1 and Ωmin = 10 endpoints
    def label(x, y, color, dy):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", va="center", fontsize=6.8,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))
    label(1, cop[0], C_COP, 11)
    label(10, cop[9], C_COP, 11)
    label(1, fam[0], C_FAM, 14)
    label(10, fam[9], C_FAM, -11)

    # divider, then the two reference violins (the 269 sampled genomes)
    ax.axvline(10.5, color="#CFCFCF", lw=0.7, ls=(0, (2, 2)), zorder=1)
    _violin(ax, gcop, x_vio_c, C_COP)
    _violin(ax, gfam, x_vio_f, C_FAM)

    ax.set_xlim(0.3, 13.2)
    ax.set_ylim(0, max(cop.max(), gcop.max()) * 1.12)
    ax.set_xticks(list(range(1, 11)) + [(x_vio_f + x_vio_c) / 2])
    ax.set_xticklabels([str(n) for n in range(1, 11)] + ["269\ngenomes"])
    ax.set_ylabel("number of gene families / gene copies")
    ax.set_xlabel(f"observation threshold,  {OMIN}")
    ax.xaxis.set_label_coords(0.40, -0.155)
    ps.grid(ax, axis="y")

    handles = [
        Line2D([], [], color=C_COP, marker="o", lw=1.7, ms=4.5,
               markeredgecolor="white", markeredgewidth=0.5,
               label="gene copies in LACA"),
        Line2D([], [], color=C_FAM, marker="o", lw=1.7, ms=4.5,
               markeredgecolor="white", markeredgewidth=0.5,
               label="gene families in LACA"),
        Line2D([], [], color=C_OMIN1, marker="o", lw=0, ms=7.0,
               markeredgecolor="white", markeredgewidth=0.9,
               label=f"{OMIN} = 1 fit (panel B)"),
        Line2D([], [], color=C_OMIN4, marker="o", lw=0, ms=7.0,
               markeredgecolor="white", markeredgewidth=0.9,
               label=f"{OMIN} = 4 fit (panel B)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.99),
              frameon=False, handlelength=1.7, labelspacing=0.4)


def panel_ppc(ax, d):
    """Ω ≥ 2 posterior-predictive check — both predictions as plain lines."""
    ks = np.arange(2, K_MAX + 1)
    emp = d["emp"][1:]
    pred1 = d["pred1"][1:]
    pred4 = d["pred4"][1:]

    ax.axvspan(1.7, 3.5, color=C_SHADE, lw=0, zorder=0)

    m1 = pred1 > 1e-9
    ax.plot(ks[m1], pred1[m1], "-", color=C_OMIN1, lw=1.9, alpha=0.95,
            solid_capstyle="round", zorder=3)
    m4 = pred4 > 1e-9
    ax.plot(ks[m4], pred4[m4], "-", color=C_OMIN4, lw=1.9, alpha=0.95,
            solid_capstyle="round", zorder=4)

    me = emp > 0
    ax.plot(ks[me], emp[me], "-", color=C_DATA, lw=0.6, alpha=0.45, zorder=5)
    ax.plot(ks[me], emp[me], "o", color=C_DATA, ms=3.6, zorder=6,
            markeredgecolor="white", markeredgewidth=0.45)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.7, 70)
    ax.set_ylim(0.6, 2.6e4)
    ax.set_xlabel("family size  (total gene copies)")
    ax.set_ylabel("number of gene families per size bin")
    ps.grid(ax)
    ax.grid(True, which="minor", lw=0.3, alpha=0.12)

    handles = [
        Line2D([], [], color=C_DATA, marker="o", ms=4, lw=1.0,
               markeredgecolor="white", markeredgewidth=0.4, label="observed"),
        Line2D([], [], color=C_OMIN1, lw=2.2, label=f"predicted, {OMIN} = 1 fit"),
        Line2D([], [], color=C_OMIN4, lw=2.2, label=f"predicted, {OMIN} = 4 fit"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handlelength=1.9, handletextpad=0.6, labelspacing=0.4)


def main() -> int:
    d = load_data()
    ps.apply()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.5))   # PNAS 2-col
    panel_sweep(axA, d)   # panel A — root families + copies sweep
    panel_ppc(axB, d)     # panel B — posterior-predictive check

    for ax in (axA, axB):                       # align the two y-labels
        ax.yaxis.set_label_coords(-0.15, 0.5)
    # align the two x-label baselines — panel A's sits low (y=-0.155) to
    # clear its two-line "269 genomes" tick, so drop panel B's to match
    axB.xaxis.set_label_coords(0.5, -0.155)
    for ax, letter in ((axA, "A"), (axB, "B")):
        ps.panel_label(ax, letter, dx=-0.175, dy=1.045)

    fig.subplots_adjust(left=0.094, right=0.986, bottom=0.175,
                        top=0.955, wspace=0.315)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / "pnas_letter_v2_fig1_alt.pdf"
    png = FIGDIR / "pnas_letter_v2_fig1_alt.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)

    print("\nroot sweep — families:  " + "  ".join(f"{v:,.0f}" for v in d["sweep_fam"]))
    print("root sweep — copies:    " + "  ".join(f"{v:,.0f}" for v in d["sweep_cop"]))
    gf, gc = d["genome_fam"], d["genome_cop"]
    print(f"269 genomes — families: median {np.median(gf):,.0f}, "
          f"range {gf.min():,.0f}-{gf.max():,.0f}")
    print(f"269 genomes — copies:   median {np.median(gc):,.0f}, "
          f"range {gc.min():,.0f}-{gc.max():,.0f}")
    print(f"wrote {pdf.relative_to(ROOT)}")
    print(f"wrote {png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
