"""Matched-min rate-path plots and pooled per-branch rate scatter.

For each filter min N ∈ {1..7}, produce:

  - Per-subset 2×2 rate-path plot (γ, μ, λ, t) along arc269 root → median-fm
    leaf path, overlaying arc269 sum>=N (full-tree, red) vs subset sum>=N
    (per-clade, purple). At N=4, the Csurös subset-only ML reference is
    also shown (gray dashed).
  - Pooled per-branch rate scatter across all 4 subsets: arc269 sum>=N rate
    on the y-axis vs subset sum>=N rate on the x-axis at every LCA-matched
    internal node — one 1×3 figure with γ, λ, t panels (μ omitted, always 1).

Naming:
  rate_path_min<N>_<ds>.png/.pdf
  rate_pooled_min<N>.png/.pdf

NOTE — the arc269 sum>=N filter is over ALL 269 leaves; the subset sum>=N
filter is over THAT subset's leaves. So "matched min N" means the SAME
filter THRESHOLD but the family universes differ — arc269 sees more
families because it sums across more leaves. This is the closest
apples-to-apples we can do without re-running arc269 separately on
each per-subset min=N universe (which would be 4×7=28 extra fits).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from validation._shared import load_dataset
from validation import _pubstyle as ps

OUT = Path("validation/outputs")
DATASETS = ["dpann80", "proteo75", "eury114", "ed194"]
SUBSET_LABEL = {"dpann80": "DPANN (D80)", "proteo75": "Proteoarchaea (P75)",
                "eury114": "Methanobacteriati (E114)", "ed194": "Euryarchaeota (ED194)"}

# --- series colours (Okabe-Ito, via _pubstyle) ------------------------
# arc269 full-tree MAP Brownian — the prominent series;
# subset MAP Brownian — the MAP-Brownian per-clade fit (C_BROWN);
# Csurös subset-only ML — neutral reference baseline.
C_ARC269 = ps.C_OMIN4
C_SUBSET = ps.C_BROWN
C_CSUROS = ps.C_NEUTRAL

MIN_LEVELS = (1, 2, 3, 4, 5, 6, 7)
RATE_COLS = ["gain_rate", "loss_rate", "dup_rate", "branch_length"]
RATE_TITLES = {
    "gain_rate":     "gain rate (γ)",
    "loss_rate":     "loss rate (μ) — fixed at 1 by convention",
    "dup_rate":      "dup rate (λ) — sub-critical cap at ~1",
    "branch_length": "branch length (t) — subset t collapses multi-edges",
}


def load_branches(path: Path) -> dict:
    if not path.exists(): return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try: out[int(r["node_index"])] = r
            except (KeyError, ValueError): continue
    return out


def arc269_fit_path(n: int) -> Path:
    if n == 1: return OUT / "brownian_arc269/arc269_map_sigma1.0.branches.csv"
    return OUT / f"brownian_arc269_omin4_sum{n}/arc269_map_sigma1.0.branches.csv"


def subset_fit_path(ds: str, n: int) -> Path:
    if n == 1: return OUT / f"brownian_{ds}_full/{ds}_map_sigma1.0.branches.csv"
    if n == 4: return OUT / f"brownian_{ds}/{ds}_map_sigma1.0.branches.csv"
    return OUT / f"brownian_{ds}_sum{n}/{ds}_map_sigma1.0.branches.csv"


def csuros_fit_path(ds: str) -> Path:
    return OUT / f"reproduction/{ds}_reproduce.branches.csv"


def read_rate(rows, idx, col):
    if idx not in rows: return None
    try:
        v = rows[idx][col]
        if v == "inf": return np.inf
        f = float(v)
        return f if np.isfinite(f) else None
    except (KeyError, ValueError): return None


def map_subset_to_arc269(subtree, tree269):
    arc269_name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    children = [[] for _ in range(subtree.num_nodes)]
    for v in range(subtree.num_nodes):
        if subtree.parent[v] >= 0:
            children[subtree.parent[v]].append(v)
    leaves_below = [None] * subtree.num_nodes
    for u in range(subtree.num_nodes):
        if u < subtree.num_leaves:
            leaves_below[u] = [u]
        else:
            leaves_below[u] = []
            for c in children[u]:
                leaves_below[u].extend(leaves_below[c])
    mapping = {}
    for u in range(subtree.num_nodes):
        names = [subtree.leaf_names[l] for l in leaves_below[u]]
        idxs = [arc269_name_to_idx[n] for n in names if n in arc269_name_to_idx]
        if not idxs: continue
        ancs = []
        for li in idxs:
            s = set(); v = li
            while v >= 0:
                s.add(int(v)); v = tree269.parent[v]
            ancs.append(s)
        mapping[u] = min(set.intersection(*ancs))
    return mapping


def pick_median_leaf(tree269, subtree, profiles269, rng):
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    sub_leaf_idxs = [name_to_idx[n] for n in subtree.leaf_names if n in name_to_idx]
    fms = np.array([int((profiles269[:, li] > 0).sum()) for li in sub_leaf_idxs])
    median_fm = int(np.median(fms))
    diffs = np.abs(fms - median_fm)
    chosen_pos = int(rng.choice(np.where(diffs == diffs.min())[0]))
    chosen_li = sub_leaf_idxs[chosen_pos]
    return chosen_li, tree269.leaf_names[chosen_li]


def arc269_path(tree269, leaf_idx):
    p, v = [], leaf_idx
    while v >= 0:
        p.append(v); v = tree269.parent[v]
    return p[::-1]


def plot_path_min(ds: str, n: int, tree269, profiles269, rng):
    """Per-subset 2×2 rate-path plot at filter min=N (arc269 vs subset)."""
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name = pick_median_leaf(tree269, subtree, profiles269, rng)
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    leaf_depth = len(path) - 1

    arc269_rows = load_branches(arc269_fit_path(n))
    subset_rows = load_branches(subset_fit_path(ds, n))
    csuros_rows = load_branches(csuros_fit_path(ds)) if n == 4 else {}
    if not arc269_rows or not subset_rows:
        print(f"  SKIP {ds} min={n}: missing branches data")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH_2COL, 5.0))

    for ax, col, letter in zip(axes.flat, RATE_COLS, "ABCD"):
        # arc269 full-tree
        d_pts, vals = [], []
        for d, av in enumerate(path):
            r = read_rate(arc269_rows, av, col)
            if r is not None: d_pts.append(d); vals.append(r)
        ax.plot(d_pts, vals, marker='o', color=C_ARC269, lw=1.6,
                markersize=4.0, markeredgecolor='white', markeredgewidth=0.5,
                zorder=8)

        # Subset (via arc2sub)
        d_pts, vals = [], []
        for d, av in enumerate(path):
            s = arc2sub.get(av)
            if s is None: continue
            r = read_rate(subset_rows, s, col)
            if r is None: continue
            d_pts.append(d); vals.append(r)
        ax.plot(d_pts, vals, marker='s', color=C_SUBSET, lw=1.4, linestyle=':',
                zorder=9, markersize=4.2, markeredgecolor='white',
                markeredgewidth=0.5)

        # Csurös ref (only at N=4)
        if csuros_rows:
            d_pts, vals = [], []
            for d, av in enumerate(path):
                s = arc2sub.get(av)
                if s is None: continue
                r = read_rate(csuros_rows, s, col)
                if r is None: continue
                d_pts.append(d); vals.append(r)
            ax.plot(d_pts, vals, marker='D', color=C_CSUROS, lw=1.4,
                    linestyle='--', zorder=10, markersize=4.0,
                    markerfacecolor='none', markeredgecolor=C_CSUROS,
                    markeredgewidth=1.1)

        ax.set_title(RATE_TITLES[col])
        ax.set_xlabel(f"depth  (0 = LACA / arc269 root, {leaf_depth} = leaf)")
        ax.set_ylabel(col.replace("_", " "))
        ps.grid(ax)
        ps.panel_label(ax, letter)
        # Reasonable y-limits (same heuristic as the legacy plot).
        if col == "loss_rate":         ax.set_ylim(0.5, 1.5)
        elif col == "dup_rate":         ax.set_ylim(0, 1.05)
        else:
            all_y = ax.get_ylim()
            if all_y[1] > 50: ax.set_ylim(0, 50)
            elif all_y[0] < 0: ax.set_ylim(0, all_y[1])

    fig.tight_layout()
    ps.save_fig(fig, OUT / f"rate_path_min{n}_{ds}")
    plt.close(fig)
    print(f"  wrote rate_path_min{n}_{ds}.png + .pdf")
    return True


def plot_pooled_min(n: int, tree269):
    """Pooled scatter across 4 subsets at min=N: arc269 vs subset rate at every
    LCA-matched internal node. One 1×3 figure with γ / λ / t panels."""
    arc269_rows = load_branches(arc269_fit_path(n))
    if not arc269_rows:
        print(f"  SKIP pooled min={n}: arc269 fit missing")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_2COL, 2.7))
    cols = ["gain_rate", "dup_rate", "branch_length"]
    titles = ["gain γ", "dup λ", "branch length t"]

    # Per-subset distinct marker colours (Okabe-Ito)
    ds_color = {"dpann80": ps.BLUE, "proteo75": ps.ORANGE,
                "eury114": ps.GREEN, "ed194": ps.VERMILION}

    counts = {ds: 0 for ds in DATASETS}
    for ds in DATASETS:
        subtree, _, _, _, _ = load_dataset(ds)
        subset_rows = load_branches(subset_fit_path(ds, n))
        if not subset_rows: continue
        sub2arc = map_subset_to_arc269(subtree, tree269)
        for ax, col in zip(axes, cols):
            xs, ys = [], []
            for s, a in sub2arc.items():
                if s < subtree.num_leaves or s == subtree.root: continue
                rs = read_rate(subset_rows, s, col)
                ra = read_rate(arc269_rows, a, col)
                if rs is None or ra is None: continue
                if np.isinf(rs) or np.isinf(ra): continue
                xs.append(rs); ys.append(ra)
            if not xs: continue
            counts[ds] = max(counts[ds], len(xs))
            ax.scatter(xs, ys, s=12, alpha=0.55, color=ds_color[ds],
                       edgecolor='none',
                       label=SUBSET_LABEL[ds] if ax is axes[0] else None)

    # Diagonal y=x line on each panel
    for ax, col, title, letter in zip(axes, cols, titles, "ABC"):
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0], 1e-6)
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.0)
        ax.plot([lo, hi], [lo, hi], '--', color=ps.GREY, lw=0.9,
                alpha=0.7, zorder=5)
        ax.set_xlabel(f"subset sum≥{n} {col.replace('_',' ')}")
        ax.set_ylabel(f"arc269 sum≥{n} {col.replace('_',' ')}")
        ax.set_title(title)
        ps.grid(ax)
        ps.panel_label(ax, letter, dx=-0.26)
        # log-scale gain (heavy tail), linear for dup (bounded), log for length (heavy tail)
        if col in ("gain_rate","branch_length"):
            try: ax.set_xscale('log'); ax.set_yscale('log')
            except Exception: pass

    axes[0].legend(loc='upper left', title="subset", title_fontsize=7.5)
    fig.tight_layout()
    ps.save_fig(fig, OUT / f"rate_pooled_min{n}")
    plt.close(fig)
    print(f"  wrote rate_pooled_min{n}.png + .pdf ({sum(counts.values())} pooled tuples)")
    return True


def write_shared_legend():
    handles = [
        mlines.Line2D([], [], marker='o', color=C_ARC269, lw=1.6, markersize=5,
                      markeredgecolor='white', markeredgewidth=0.5,
                      label='arc269 MAP Brownian, sum≥N (full-tree, F=90,243-filtered)'),
        mlines.Line2D([], [], marker='s', color=C_SUBSET, lw=1.4, linestyle=':',
                      markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                      label='subset MAP Brownian, sum≥N (per-clade, subset-leaves-filtered)'),
        mlines.Line2D([], [], marker='D', color=C_CSUROS, lw=1.4, linestyle='--',
                      markersize=4.5, markerfacecolor='none',
                      markeredgecolor=C_CSUROS, markeredgewidth=1.1,
                      label='Csurös subset-only ML (reference baseline, shown only at sum≥4)'),
    ]
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 0.6))
    fig.legend(handles=handles, loc='center', ncol=1, frameon=False)
    ps.save_fig(fig, OUT / "rate_path_min_legend")
    plt.close(fig)
    print(f"  wrote rate_path_min_legend.png + .pdf")


def main():
    ps.apply()
    print("Loading arc269...")
    tree269, profiles269, _, _, _ = load_dataset("arc269")

    for n in MIN_LEVELS:
        print(f"\n=== min={n} ===")
        rng = np.random.default_rng(2026)
        for ds in DATASETS:
            plot_path_min(ds, n, tree269, profiles269, rng)
        plot_pooled_min(n, tree269)

    print("\n--- shared legend ---")
    write_shared_legend()
    print("\nDONE.")


if __name__ == "__main__":
    main()
