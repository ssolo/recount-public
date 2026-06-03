"""LACA CORE analysis: how is the GLD-inferred LACA CORE partitioned
across modern subclade genomes?

LACA CORE = { f : P(f present at arc269 root) >= 0.5 } under the
matched-universe arc269 Brownian fit on the union of per-subclade min=4
families (F = 11,148; fit at validation/outputs/brownian_arc269_union_subsets_min4/).

Outputs (under validation/outputs/):

  - subset_core_partition.png/.pdf   — per-subset CORE (each subset's
    own ancestor from per-clade Brownian-extend ML at min=4).
  - laca_core_subset_distribution.png/.pdf
                                     — focal figure: fraction of LACA
    CORE present in ≥ x fraction of each subclade's modern genomes.
  - laca_core_tables.txt              — text-form tables (counts +
    percentages) of LACA CORE presence per subclade.

If the union per-family posteriors are missing (the union fit was
added after the standard validation pipeline), they are computed
in-place from the saved rates and the union-filtered profile.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from recount.native_backend import per_family_posteriors_native
from validation._shared import load_dataset
from validation import _pubstyle as ps

OUT = Path("validation/outputs")
UNION_DIR = OUT / "brownian_arc269_union_subsets_min4"

SUBSETS = [("dpann80", "DPANN (D80)"),
           ("proteo75", "Proteoarchaea (P75)"),
           ("eury114", "Methanobacteriati (E114)"),
           ("ed194", "Euryarchaeota (ED194)")]
COLORS = {"dpann80": ps.BLUE, "proteo75": ps.ORANGE,
          "eury114": ps.GREEN, "ed194": ps.VERMILION}
TIER_COLORS = ["#0d3a1f", "#3a8c4d", "#7dc278", "#c2e3a6",
               "#f7e297", "#e69c5c", "#c44747"]
TIER_LABELS = ["universal", "≥90 %", "50–90 %", "25–50 %",
               "10–25 %", "1–10 %", "= 1 leaf"]


def ensure_union_per_family(tree269, profiles269):
    """Generate the union-min4 per-family per-node posteriors if missing."""
    out_npz = UNION_DIR / "arc269_per_family_FxN.npz"
    if out_npz.exists():
        return out_npz
    print("  union per-family file missing — generating from saved rates...")
    F_orig = profiles269.shape[0]
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    union = np.zeros(F_orig, dtype=bool)
    for ds, _ in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        union |= profiles269[:, sub_li].sum(axis=1) >= 4
    prof_union = profiles269[union].astype(np.int32, copy=True)
    d = np.load(UNION_DIR / "arc269_sigma1.0_final_rates.npz")
    t0 = time.time()
    copies, present = per_family_posteriors_native(
        tree269, d["gain"], d["loss"], d["dup"], d["length"],
        prof_union, num_threads=16,
    )
    print(f"  computed in {time.time()-t0:.1f}s")
    np.savez_compressed(out_npz, present=present, copies=copies,
                        F=int(prof_union.shape[0]),
                        N=int(tree269.num_nodes),
                        dataset="arc269_union_subsets_min4",
                        threshold=0.5, union_mask=union)
    return out_npz


def tier_breakdown(per_fam_pres, L_S):
    return [
        int((per_fam_pres == L_S).sum()),
        int(((per_fam_pres >= 0.9*L_S) & (per_fam_pres < L_S)).sum()),
        int(((per_fam_pres >= 0.5*L_S) & (per_fam_pres < 0.9*L_S)).sum()),
        int(((per_fam_pres >= 0.25*L_S) & (per_fam_pres < 0.5*L_S)).sum()),
        int(((per_fam_pres >= 0.1*L_S) & (per_fam_pres < 0.25*L_S)).sum()),
        int(((per_fam_pres >= 2) & (per_fam_pres < 0.1*L_S)).sum()),
        int((per_fam_pres == 1).sum()),
    ]


def pairwise_jaccard(core_leaf, L_S, n_sample=400, rng_seed=2026):
    rng = np.random.default_rng(rng_seed)
    pairs = min(n_sample, L_S * (L_S - 1) // 2)
    Js = []
    for _ in range(pairs):
        i, j = rng.choice(L_S, 2, replace=False)
        a = core_leaf[:, i]; b = core_leaf[:, j]
        inter = (a & b).sum(); u = (a | b).sum()
        Js.append(inter / u if u else 0)
    return np.array(Js)


def figure1_subset_core(per_subset_data):
    """Single-axes plot in the same style as figure2_laca_distribution,
    but using each subclade's OWN root CORE (per-subset Brownian-extend
    ML fit at min=4) as the denominator, rather than the LACA CORE."""
    fig, ax = plt.subplots(figsize=(ps.WIDTH_2COL, 3.7))
    xs = np.linspace(0.0, 1.0, 501)
    for ds, label in SUBSETS:
        d = per_subset_data[ds]
        L_S = d["L_S"]; cs = d["core_size"]
        per_fam_pres = d["per_fam_pres"]
        # For each x in [0,1], count CORE families present in
        # >= ceil(x * L_S) of the subclade's L_S leaves; normalise
        # by the subclade's own CORE size (cs).
        ys = np.array([(per_fam_pres >= int(np.ceil(x * L_S))).sum()
                       for x in xs]) / cs
        # Per-leaf retention r_v of THIS subclade's CORE
        # (per_fam_pres counts # leaves a family is present at;
        #  we want per-leaf count of CORE families present)
        # leaf_count[v] = # CORE families present at leaf v.
        # Compute from leaf_core_count saved earlier:
        # we have d["leaf_med"] but not the full array; recompute from
        # per_fam_pres + L_S is non-trivial. Use mean_r and leaf_med
        # for headline numbers; r_min etc. from leaf scan below.
        # Reconstruct leaf-presence via the per_fam_pres only gives
        # totals; we need leaf_core_count which isn't stored. So
        # report mean retention, median, and the standard threshold
        # fractions instead.
        univ_frac = (per_fam_pres == L_S).sum() / cs
        n90_frac = (per_fam_pres >= int(np.ceil(0.9 * L_S))).sum() / cs
        n50_frac = (per_fam_pres >= int(np.ceil(0.5 * L_S))).sum() / cs
        mean_r = d["mean_r"]
        color = COLORS[ds]
        ax.plot(xs, ys, color=color, lw=1.8,
                label=f"{label}   (CORE = {cs:,};  mean retention {100*mean_r:.0f} %)")
        ax.scatter([0.5, 0.9, 1.0], [n50_frac, n90_frac, univ_frac],
                   color=color, s=20, zorder=10,
                   edgecolor='white', linewidths=0.5)

    ax.axvspan(0.5, 1.0, color=ps.C_SHADE, zorder=0)
    ax.text(0.965, 0.955, "strict tail\n(≥ half of subclade genomes)",
            transform=ax.transAxes, fontsize=6.6, ha='right', va='top',
            color=ps.GREY)
    ax.axhline(1.0, color=ps.GREY, lw=0.7, alpha=0.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("present in at least this fraction of the subclade's modern genomes")
    ax.set_ylabel("fraction of the subclade's OWN CORE present\n"
                  "(denominator = each subclade's per-root CORE size)")
    ax.set_xticks([0, 0.1, 0.25, 0.5, 0.9, 1.0])
    ax.set_xticklabels(["any", "≥10 %", "≥25 %", "≥50 %", "≥90 %", "universal"])
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{int(100*y)} %" for y in np.linspace(0, 1, 11)])
    ps.grid(ax)
    ax.legend(loc='lower left', title="subclade", title_fontsize=7.5)
    fig.tight_layout()
    ps.save_fig(fig, OUT / "subset_core_partition")
    plt.close(fig)
    print("  wrote subset_core_partition.png + .pdf")


def figure2_laca_distribution(present, laca_core, tree269, LACA, LACA_STRICT, SOFT_FM):
    """Focus on median + least-complete tail (not the most-complete tail)."""
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    fig, ax = plt.subplots(figsize=(ps.WIDTH_2COL, 3.7))
    xs = np.linspace(0.0, 1.0, 501)
    for ds, label in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        L_S = len(sub_li)
        leaf_present = present[:, sub_li] >= 0.5
        core_leaf = leaf_present[laca_core]
        per_fam_pres = core_leaf.sum(axis=1)
        ys = np.array([(per_fam_pres >= int(np.ceil(x * L_S))).sum()
                       for x in xs]) / LACA
        leaf_count = core_leaf.sum(axis=0)
        r = leaf_count / LACA
        r_min, r_p10 = float(r.min()), float(np.percentile(r, 10))
        r_med = float(np.median(r))
        univ_frac = (per_fam_pres == L_S).sum() / LACA
        n90_frac = (per_fam_pres >= 0.9*L_S).sum() / LACA
        n50_frac = (per_fam_pres >= 0.5*L_S).sum() / LACA
        color = COLORS[ds]
        ax.plot(xs, ys, color=color, lw=1.8,
                label=f"{label}   (L_S = {L_S};  median retention {100*r_med:.0f} %)")
        ax.scatter([0.5, 0.9, 1.0], [n50_frac, n90_frac, univ_frac],
                   color=color, s=20, zorder=10,
                   edgecolor='white', linewidths=0.5)

    # Shade the strict tail (>=50%)
    ax.axvspan(0.5, 1.0, color=ps.C_SHADE, zorder=0)
    ax.text(0.965, 0.955, "strict tail\n(≥ half of subclade genomes)",
            transform=ax.transAxes, fontsize=6.6, ha='right', va='top',
            color=ps.GREY)
    ax.axhline(1.0, color=ps.GREY, lw=0.7, alpha=0.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("present in at least this fraction of the subclade's modern genomes")
    ax.set_ylabel(f"fraction of the LACA CORE  ({LACA} families)  present")
    ax.set_xticks([0, 0.1, 0.25, 0.5, 0.9, 1.0])
    ax.set_xticklabels(["any", "≥10 %", "≥25 %", "≥50 %", "≥90 %", "universal"])
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{int(100*y)} %" for y in np.linspace(0, 1, 11)])
    ps.grid(ax)
    ax.legend(loc='lower left', title="subclade", title_fontsize=7.5)
    fig.tight_layout()
    ps.save_fig(fig, OUT / "laca_core_subset_distribution")
    plt.close(fig)
    print("  wrote laca_core_subset_distribution.png + .pdf")


def write_tables(present, laca_core, tree269, LACA):
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    THRESHOLDS = [0.0, 0.10, 0.25, 0.50, 0.90, 1.0]
    lines = []
    lines.append(f"LACA CORE (P_root >= 0.5) = {LACA}")
    lines.append(f"LACA strict CORE (P_root >= 0.9) = {(present[:, tree269.root] >= 0.9).sum()}")
    lines.append("")
    lines.append("Absolute counts (number of LACA CORE families present at >= X% of subclade genomes):")
    hdr = f"  {'subclade':<22s} {'L_S':>4s}  " + "  ".join(
        f"{'any' if t == 0 else f'>={int(100*t):d}%':>8s}" for t in THRESHOLDS) + "    universal"
    lines.append(hdr)
    for ds, label in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        L_S = len(sub_li)
        leaf_present = present[:, sub_li] >= 0.5
        core_leaf = leaf_present[laca_core]
        per_fam_pres = core_leaf.sum(axis=1)
        row = [f"  {label:<22s} {L_S:>4d} "]
        for t in THRESHOLDS:
            k = int(np.ceil(t * L_S)) if t > 0 else 1
            row.append(f"{int((per_fam_pres >= k).sum()):>8d}")
        row.append(f"      {int((per_fam_pres == L_S).sum())}")
        lines.append(" ".join(row))
    lines.append("")
    lines.append("Percentages of LACA CORE:")
    lines.append(hdr)
    for ds, label in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        L_S = len(sub_li)
        leaf_present = present[:, sub_li] >= 0.5
        core_leaf = leaf_present[laca_core]
        per_fam_pres = core_leaf.sum(axis=1)
        row = [f"  {label:<22s} {L_S:>4d} "]
        for t in THRESHOLDS:
            k = int(np.ceil(t * L_S)) if t > 0 else 1
            frac = (per_fam_pres >= k).sum() / LACA
            row.append(f"{100*frac:>7.1f}%")
        univ_frac = (per_fam_pres == L_S).sum() / LACA
        row.append(f"   {100*univ_frac:.2f}%")
        lines.append(" ".join(row))

    # Cross-subclade presence
    lines.append("")
    lines.append("Cross-subclade presence of LACA CORE families (each row counts families present in N of 4 subclades):")
    sub_present = {}
    for ds, _ in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        sub_present[ds] = (present[:, sub_li] >= 0.5)[laca_core].any(axis=1)
    stacked = np.column_stack([sub_present[ds] for ds, _ in SUBSETS])
    counts = stacked.sum(axis=1)
    for n in (0, 1, 2, 3, 4):
        c = int((counts == n).sum())
        lines.append(f"  present in {n} of 4 subclades: {c:>5d} ({100*c/LACA:.1f} %)")

    # Median + least-complete tail + LACA-total estimator
    lines.append("")
    lines.append("Per-subclade extant genome size and LACA CORE retention "
                 "(focus: median and lower tail):")
    lines.append(f"  G = total extant family count over full F=90,243 universe; "
                 f"r = fraction of LACA CORE retained at the leaf.")
    lines.append(f"  {'subclade':<22s} {'L_S':>4s} {'G_min':>6s} {'G_p10':>6s} "
                 f"{'G_p25':>6s} {'G_med':>6s} {'r_min':>6s} {'r_p10':>6s} "
                 f"{'r_p25':>6s} {'r_med':>6s}")
    profiles269_ = load_dataset("arc269")[1]
    est_rows = []
    for ds, label in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in sub.leaf_names if n in name_to_idx]
        L_S = len(sub_li)
        G_extant = (profiles269_[:, sub_li] > 0).sum(axis=0)
        leaf_present = present[:, sub_li] >= 0.5
        core_leaf = leaf_present[laca_core]
        leaf_core_count = core_leaf.sum(axis=0)
        r = leaf_core_count / LACA
        G = (int(G_extant.min()), int(np.percentile(G_extant, 10)),
             int(np.percentile(G_extant, 25)), int(np.median(G_extant)))
        rv = (float(r.min()), float(np.percentile(r, 10)),
              float(np.percentile(r, 25)), float(np.median(r)))
        lines.append(f"  {label:<22s} {L_S:>4d} "
                     f"{G[0]:>6d} {G[1]:>6d} {G[2]:>6d} {G[3]:>6d} "
                     f"{rv[0]:>6.3f} {rv[1]:>6.3f} {rv[2]:>6.3f} {rv[3]:>6.3f}")
        est_rows.append((label, L_S, G[3], rv[3]))

    lines.append("")
    lines.append("LACA-total simple estimator (upper bound assuming all extant "
                 "families are LACA-derived):")
    lines.append(f"  LACA_total ≈ G_extant / r,  using median G_extant and "
                 f"median r per subclade.  Reference: model LACA CORE = {LACA}.")
    lines.append(f"  {'subclade':<22s} {'G_med':>6s} {'r_med':>6s} "
                 f"{'LACA_total est':>15s} {'ratio to CORE':>14s}")
    for label, L_S, G_med, r_med in est_rows:
        est = G_med / r_med if r_med > 0 else 0
        lines.append(f"  {label:<22s} {G_med:>6d} {r_med:>6.3f} "
                     f"{est:>15,.0f} {est/LACA:>13.2f}×")

    txt = "\n".join(lines)
    print()
    print(txt)
    (OUT / "laca_core_tables.txt").write_text(txt + "\n")
    print(f"\n  wrote {OUT / 'laca_core_tables.txt'}")


def main():
    ps.apply()
    print("Loading arc269...")
    tree269, profiles269, _, _, _ = load_dataset("arc269")
    ensure_union_per_family(tree269, profiles269)
    z = np.load(UNION_DIR / "arc269_per_family_FxN.npz", allow_pickle=True)
    present = z["present"]
    p_laca = present[:, tree269.root]
    laca_core = p_laca >= 0.5
    LACA = int(laca_core.sum())
    LACA_STRICT = int((p_laca >= 0.9).sum())
    SOFT_FM = float(p_laca.sum())
    print(f"  LACA CORE = {LACA};  strict = {LACA_STRICT};  soft root_fm = {SOFT_FM:.0f}")

    # ---- per-subset data for Figure 1 ----
    print("\nBuilding per-subset CORE data...")
    per_subset_data = {}
    for ds, label in SUBSETS:
        sub, _, _, _, _ = load_dataset(ds)
        L_S = sub.num_leaves
        z_ds = np.load(OUT / f"brownian_extend_{ds}/{ds}_per_family_FxN.npz",
                       allow_pickle=True)
        p = z_ds["present"]
        F_total = p.shape[0]
        p_root = p[:, sub.root]
        core = p_root >= 0.5
        core_size = int(core.sum())
        leaf_present = p[:, :L_S] >= 0.5
        core_leaf = leaf_present[core]
        per_fam_pres = core_leaf.sum(axis=1)
        leaf_core_count = core_leaf.sum(axis=0)
        Js = pairwise_jaccard(core_leaf, L_S)
        per_subset_data[ds] = dict(
            label=label, L_S=L_S, F_total=F_total, core_size=core_size,
            per_fam_pres=per_fam_pres,
            leaf_med=int(np.median(leaf_core_count)),
            mean_r=leaf_core_count.mean() / core_size,
            med_J=float(np.median(Js)),
            tiers=tier_breakdown(per_fam_pres, L_S))

    print("\nFigure 1: per-subset CORE partition")
    figure1_subset_core(per_subset_data)

    print("\nFigure 2: LACA CORE distribution across subclade genomes")
    figure2_laca_distribution(present, laca_core, tree269, LACA, LACA_STRICT, SOFT_FM)

    print("\nTables:")
    write_tables(present, laca_core, tree269, LACA)


if __name__ == "__main__":
    main()
