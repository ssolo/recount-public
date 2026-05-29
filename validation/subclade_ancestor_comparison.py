"""For the arc269 full-tree fits, read off the per-node copies/families at
the 4 subclade-ancestor nodes (D80, P75, E114, ED194 ancestors inside
arc269) and compare to the subset-only ML fits.

Question being answered: are the subclade-ancestor reconstructions
consistent between (a) fitting the subset clade alone (Csurös' Java
approach) vs (b) fitting the full 269-leaf tree and reading off the
subclade-ancestor node?

Plots two variants in one pass:
  * BOUNDED   — fits from validation/outputs/{simple_ml,sota_ml,map_sigma1,
                profile_likelihood_arc269} (Csurös-exact dup logit + log γ ≤ 33)
  * UNCONSTRAINED — fits from validation/outputs_no_dup_cap/* (no sub-critical
                    cap, no gain cap — the historical snapshot)

Within each variant, two plot sets are produced:
  * Set 1 (suffix=''): cold ML + multistart-polish ML + MAP σ=1 cold
  * Set 2 (suffix='_with_map_polish'): set 1 + MAP σ=1 15-start global
Per-subset path plots use the median-fm leaf within each subset's arc269
clade (random tiebreak, fixed seed). Right panel uses linear y-scale for
dense ticks.
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator

from validation._shared import load_dataset


# Subsets and their nominal LACA-quote values (Csurös' published ML on the subset alone).
SUBSETS = {
    "dpann80": {"label": "DPANN (D80, 80 leaves)"},
    "proteo75": {"label": "Proteoarchaea (P75, 75 leaves)"},
    "eury114": {"label": "Methanobacteriati (E114, 114 leaves)"},
    "ed194": {"label": "Euryarchaeota (ED194, 194 leaves)"},
}


def _fit_paths(root: str) -> dict:
    """Build the standard 4-fit path dict from a results root directory.

    For the BOUNDED variant the cold ML is now produced into bounded_csuros/
    by the bounded fit queue (Csurös-recipe ML with sub-critical Yule cap).
    Fall back to simple_ml/ if bounded_csuros/ doesn't have it. The other
    fit slots are unchanged."""
    cold_bounded = Path(f"{root}/bounded_csuros/arc269_ml.branches.csv")
    cold_simple  = Path(f"{root}/simple_ml/arc269_ml.branches.csv")
    return {
        "cold_ML": cold_bounded if cold_bounded.exists() else cold_simple,
        "multistart_polish_ML": Path(f"{root}/sota_ml/arc269_ml.branches.csv"),
        "MAP_sigma1_cold": Path(f"{root}/map_sigma1/arc269_map_sigma1.0.branches.csv"),
        "MAP_sigma1_global": Path(f"{root}/profile_likelihood_arc269/arc269_map_sigma1.0.branches.csv"),
        "MAP_brownian":   Path(f"{root}/brownian_arc269/arc269_map_sigma1.0.branches.csv"),
        "ML_subset_seed": Path(f"{root}/arc269_seeded_ml/arc269_ml.branches.csv"),
    }


VARIANTS = [
    # (variant_name, arc269 fit-paths dict, subset-only results-root, variant_suffix)
    ("bounded (Csurös-exact dup logit + log γ ≤ 33)",
     _fit_paths("validation/outputs"), "validation/outputs", ""),
    ("unconstrained (no dup cap, no gain cap — historical)",
     _fit_paths("validation/outputs_no_dup_cap"),
     "validation/outputs_no_dup_cap", "_unconstrained"),
]

# Per-variant subset-only fits — cold ML / multistart-polish ML / MAP σ=1.
# Each entry is (display_label, csv_template, line_color, marker, linestyle).
# csv_template gets `.format(root=<subset_only_root>, ds=<ds>)`.
SUBSET_ONLY_VARIANTS = [
    # Subset MAP Brownian σ=1 — the per-clade Brownian-prior fit (mode 4 of
    # the canonical queue, on this subset's Csurös-pre-filtered profile).
    ("Subset MAP Brownian σ=1",         "{root}/brownian_{ds}/{ds}_map_sigma1.0.branches.csv",
     "tab:purple", "s", ":"),
    # Subset Brownian-extend ML — mode 2: warm-start from mode 4's rates,
    # drop the prior, short pure-ML fine-tune.
    ("Subset Brownian-extend ML",       "{root}/brownian_extend_{ds}/{ds}_ml.branches.csv",
     "tab:olive",  "v", "-."),
]

# Common across both variants.
ARC269_FITS = _fit_paths("validation/outputs")  # default, kept for backwards-compat imports
SET1_ARC269 = ["cold_ML", "multistart_polish_ML", "MAP_sigma1_cold",
               "MAP_brownian", "ML_subset_seed"]
SET2_ARC269 = SET1_ARC269 + ["MAP_sigma1_global"]
FITS_LABEL = {
    "subset_only": "Subset-only ML\n(Csurös, bit-perfect)",
    "cold_ML": "arc269 bare ML (no prior)",
    "multistart_polish_ML": "arc269 multistart-polish ML",
    "MAP_sigma1_cold": "arc269 MAP σ=1 (independent prior)",
    "MAP_sigma1_global": "arc269 MAP σ=1 (15-start global)",
    "MAP_brownian": "arc269 MAP Brownian σ=1",
    "ML_subset_seed": "arc269 ML from subset seed",
}
FITS_COLOR = {
    "subset_only": "gray",
    "cold_ML": "tab:red",
    "multistart_polish_ML": "tab:orange",
    "MAP_sigma1_cold": "tab:cyan",
    "MAP_sigma1_global": "tab:green",
    "MAP_brownian": "tab:brown",
    "ML_subset_seed": "tab:pink",
}


def find_lca_in_arc269(tree269, subset_leaf_names: list[str]) -> int:
    """Find the LCA (an internal node index in arc269) of a set of leaf names."""
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    leaf_indices = [name_to_idx[n] for n in subset_leaf_names if n in name_to_idx]
    if not leaf_indices:
        raise ValueError("No subset leaves found in arc269")
    parent = tree269.parent
    ancestors_per_leaf = []
    for li in leaf_indices:
        ancs = set()
        v = li
        while v >= 0:
            ancs.add(int(v))
            v = parent[v]
        ancestors_per_leaf.append(ancs)
    common = set.intersection(*ancestors_per_leaf)
    # We want the deepest of the common ancestors (closest to leaves) = LOWEST index in common.
    return min(common)


def load_branches_csv(path: Path) -> dict:
    """Returns {node_index: row_dict} from a branches.csv."""
    rows = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows[int(row["node_index"])] = row
    return rows


def _descendant_leaves(tree, u: int) -> list[int]:
    """All leaf indices descended from internal node u (inclusive if u is leaf)."""
    if u < tree.num_leaves:
        return [u]
    # Children-first walk via parent array
    leaves = []
    stack = [u]
    children = [[] for _ in range(tree.num_nodes)]
    for v in range(tree.num_nodes):
        if tree.parent[v] >= 0:
            children[tree.parent[v]].append(v)
    while stack:
        v = stack.pop()
        if v < tree.num_leaves:
            leaves.append(v)
        else:
            stack.extend(children[v])
    return leaves


def map_subset_to_arc269(subset_tree, tree269) -> dict:
    """For each subset node u, find the corresponding arc269 node = LCA in arc269
    of all leaves descended from u in the subset tree.

    Returns dict {subset_node_idx: arc269_node_idx}.
    """
    mapping = {}
    arc269_name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    # Pre-compute leaf descendants for each subset node
    # (faster: process bottom-up and union children's leaf sets)
    children = [[] for _ in range(subset_tree.num_nodes)]
    for v in range(subset_tree.num_nodes):
        if subset_tree.parent[v] >= 0:
            children[subset_tree.parent[v]].append(v)
    leaves_below = [None] * subset_tree.num_nodes
    for u in range(subset_tree.num_nodes):  # post-order: leaves first by indexing convention
        if u < subset_tree.num_leaves:
            leaves_below[u] = [u]
        else:
            leaves_below[u] = []
            for c in children[u]:
                leaves_below[u].extend(leaves_below[c])
    for u in range(subset_tree.num_nodes):
        leaf_names = [subset_tree.leaf_names[l] for l in leaves_below[u]]
        leaf_indices_in_arc = [arc269_name_to_idx[n] for n in leaf_names if n in arc269_name_to_idx]
        if not leaf_indices_in_arc:
            continue
        # LCA in arc269
        ancestors_per = []
        for li in leaf_indices_in_arc:
            ancs = set()
            v = li
            while v >= 0:
                ancs.add(int(v))
                v = tree269.parent[v]
            ancestors_per.append(ancs)
        common = set.intersection(*ancestors_per)
        mapping[u] = min(common)  # deepest common ancestor = lowest index
    return mapping


def _compute_results(arc269_fits: dict, tree269, subclade_ancestor: dict) -> dict:
    """Read each fit's branches.csv and extract cp/fm at every subclade ancestor + the root."""
    results = {}
    for fit_name, path in arc269_fits.items():
        if not path.exists():
            print(f"  SKIP {fit_name}: {path} missing")
            continue
        rows = load_branches_csv(path)
        results[fit_name] = {}
        for ds, lca_idx in subclade_ancestor.items():
            r = rows[lca_idx]
            results[fit_name][ds] = {
                "cp": float(r["copies_node_corrected"]),
                "fm": float(r["families_present_corrected"]),
            }
        r = rows[tree269.root]
        results[fit_name]["arc269_root"] = {
            "cp": float(r["copies_node_corrected"]),
            "fm": float(r["families_present_corrected"]),
        }
    return results


def _make_bar_chart(arc269_fits_in_order: list[str], suffix: str,
                    results: dict, subset_fits: dict, subclades: list,
                    out_dir: Path):
    """Bar chart: cp + fm at subclade ancestors."""
    fits_order = ["subset_only"] + arc269_fits_in_order
    fig, (ax_cp, ax_fm) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(subclades))
    width = 0.8 / len(fits_order)
    for i, fit_name in enumerate(fits_order):
        if fit_name == "subset_only":
            cps = [subset_fits.get(ds, {}).get("cp", 0) for ds in subclades]
            fms = [subset_fits.get(ds, {}).get("fm", 0) for ds in subclades]
        else:
            cps = [results.get(fit_name, {}).get(ds, {}).get("cp", 0) for ds in subclades]
            fms = [results.get(fit_name, {}).get(ds, {}).get("fm", 0) for ds in subclades]
        offset = (i - (len(fits_order) - 1) / 2.0) * width
        ax_cp.bar(x + offset, cps, width, label=FITS_LABEL[fit_name], color=FITS_COLOR[fit_name])
        ax_fm.bar(x + offset, fms, width, label=FITS_LABEL[fit_name], color=FITS_COLOR[fit_name])
    for ax in (ax_cp, ax_fm):
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in subclades])
        ax.set_xlabel("Subclade ancestor (subset clade root / corresponding arc269 internal node)",
                      fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    ax_cp.set_ylabel("Posterior root copies (Ωmin-corrected)", fontsize=10)
    ax_cp.set_title("Subclade-ancestor copies: subset-only vs arc269")
    ax_fm.set_ylabel("Posterior root families (Ωmin-corrected)", fontsize=10)
    ax_fm.set_title("Subclade-ancestor families: subset-only vs arc269")
    plt.tight_layout()
    out_png = out_dir / f"subclade_ancestor_comparison{suffix}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def _make_path_plot(ds: str, arc269_fits_in_order: list[str], suffix: str,
                    arc269_fits: dict, path_info: dict, subset_label: str,
                    out_dir: Path,
                    csuros_subset_path: list[tuple[int, float, float]] | None = None,
                    our_subset_paths: list[tuple[str, list[tuple[int, float, float]], str, str, str]] | None = None,
                    subset_leaf_cps: list[int] | None = None,
                    subset_leaf_fms: list[int] | None = None,
                    ymax_cp: float | None = None,
                    ymax_fm: float | None = None):
    """Per-subset path plot: arc269 root → median-sized leaf of subset ds.

    ``csuros_subset_path``: per-subset-node (depth, cp, fm) on the path —
    Csurös' bit-perfect recompute.

    ``our_subset_paths``: list of (label, points, color, marker, linestyle) —
    OUR independent subset-only ML fits at this subset (cold / multistart / MAP).
    Each entry plotted as its own line. Empty `points` entries are skipped.

    ``subset_leaf_cps`` / ``subset_leaf_fms``: distributions of cp/fm across
    ALL leaves of the subset (raw profile sums). Plotted as a thin violin
    around the gold-star leaf marker so the reader sees where the selected
    median leaf sits in the rest of the subset.
    """
    info = path_info
    path = info["path"]
    depths = info["depths"]
    leaf_name = info["leaf_name"]
    leaf_depth = info["leaf_depth"]
    obs_cp = info["obs_cp"]
    obs_fm = info["obs_fm"]
    anc_depth = info["anc_depth"]
    anc_idx = info["anc_idx"]

    fig, (ax_cp, ax_fm) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    # All non-gold-star legend entries suppressed — shared legend lives
    # in the README caption above the plot block. The dataset-specific
    # gold-star OBSERVED leaf marker keeps its individual legend entry
    # below since it carries data unique to each plot.
    for fit_name in arc269_fits_in_order:
        csv_path = arc269_fits.get(fit_name)
        if csv_path is None or not csv_path.exists():
            continue
        rows = load_branches_csv(csv_path)
        cps = [float(rows[i]["copies_node_corrected"]) for i in path]
        fms = [float(rows[i]["families_present_corrected"]) for i in path]
        color = FITS_COLOR[fit_name]
        ax_cp.plot(depths, cps, marker='o', color=color, lw=1.5)
        ax_fm.plot(depths, fms, marker='o', color=color, lw=1.5)

    if csuros_subset_path:
        csuros_d  = [d for (d, cp, fm) in csuros_subset_path]
        csuros_cp = [cp for (d, cp, fm) in csuros_subset_path]
        csuros_fm = [fm for (d, cp, fm) in csuros_subset_path]
        ax_cp.plot(csuros_d, csuros_cp, marker='D', markersize=9, linestyle='--',
                   color='gray', lw=1.8, zorder=11,
                   markerfacecolor='none', markeredgecolor='dimgray',
                   markeredgewidth=1.5)
        ax_fm.plot(csuros_d, csuros_fm, marker='D', markersize=9, linestyle='--',
                   color='gray', lw=1.8, zorder=11,
                   markerfacecolor='none', markeredgecolor='dimgray',
                   markeredgewidth=1.5)

    if our_subset_paths:
        for (_label, points, color, marker, linestyle) in our_subset_paths:
            if not points:
                continue
            our_d  = [d for (d, cp, fm) in points]
            our_cp = [cp for (d, cp, fm) in points]
            our_fm = [fm for (d, cp, fm) in points]
            ax_cp.plot(our_d, our_cp, marker=marker, markersize=7, linestyle=linestyle,
                       color=color, lw=2.0, zorder=9)
            ax_fm.plot(our_d, our_fm, marker=marker, markersize=7, linestyle=linestyle,
                       color=color, lw=2.0, zorder=9)

    for ax in (ax_cp, ax_fm):
        if anc_depth is not None:
            ax.axvline(anc_depth, color='gray', linestyle=':', alpha=0.7)
            ax.text(anc_depth + 0.1, ax.get_ylim()[1] if False else 1.0,
                    f"{ds.upper()} ancestor\n(node {anc_idx}, depth {anc_depth})",
                    fontsize=8, color='gray', va='bottom')
        ax.set_xlabel(f"Distance from root  (0 = LACA / arc269 root, {leaf_depth} = leaf)",
                      fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

    # Left panel (copies): linear scale, dense ticks.
    ax_cp.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    # Visible violin + scatter dots for ALL subset-leaf copy values,
    # placed at leaf_depth so you see where the chosen median leaf sits
    # in the rest of the subset's genomes.
    if subset_leaf_cps and len(subset_leaf_cps) > 2:
        cp_arr = np.array(subset_leaf_cps, dtype=float)
        vp = ax_cp.violinplot([cp_arr], positions=[leaf_depth],
                              widths=0.8, showmeans=False, showextrema=True,
                              showmedians=True)
        for body in vp['bodies']:
            body.set_alpha(0.55); body.set_facecolor('khaki'); body.set_edgecolor('goldenrod')
        for key in ('cmins', 'cmaxes', 'cmedians', 'cbars'):
            if key in vp:
                vp[key].set_alpha(0.8); vp[key].set_color('goldenrod')
        # Jittered scatter of every subset leaf so outliers are explicit.
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(cp_arr))
        ax_cp.scatter(leaf_depth + jitter, cp_arr, s=10, color='peru',
                      alpha=0.6, zorder=5)
    ax_cp.scatter([leaf_depth], [obs_cp], marker='*', s=320,
                  facecolor='gold', edgecolor='black', linewidths=1.5,
                  zorder=10, label=f'OBSERVED leaf copies = {obs_cp:,}')
    ax_cp.set_ylabel("Posterior expected copies at node (Ωmin-corrected)", fontsize=10)
    ax_cp.set_title(f"Copies along path: arc269 root → {subset_label} leaf '{leaf_name}'\n"
                    f"(observed leaf copies = {obs_cp:,})", fontsize=10)
    ax_cp.legend(fontsize=9, loc='upper right', framealpha=0.9)
    # y-axis: prefer shared cap (so all 4 subset plots in this set share the
    # same scale); else fall back to legacy "start from smallest genome".
    if ymax_cp is not None:
        ax_cp.set_ylim(0, ymax_cp)
    else:
        cp_lo = min(subset_leaf_cps) if subset_leaf_cps else None
        if cp_lo is not None and cp_lo > 0:
            ax_cp.set_ylim(bottom=cp_lo * 0.9)

    # Right panel (families): linear scale, dense ticks.
    ax_fm.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    if subset_leaf_fms and len(subset_leaf_fms) > 2:
        fm_arr = np.array(subset_leaf_fms, dtype=float)
        vp = ax_fm.violinplot([fm_arr], positions=[leaf_depth],
                              widths=0.8, showmeans=False, showextrema=True,
                              showmedians=True)
        for body in vp['bodies']:
            body.set_alpha(0.55); body.set_facecolor('khaki'); body.set_edgecolor('goldenrod')
        for key in ('cmins', 'cmaxes', 'cmedians', 'cbars'):
            if key in vp:
                vp[key].set_alpha(0.8); vp[key].set_color('goldenrod')
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(fm_arr))
        ax_fm.scatter(leaf_depth + jitter, fm_arr, s=10, color='peru',
                      alpha=0.6, zorder=5)
    ax_fm.scatter([leaf_depth], [obs_fm], marker='*', s=320,
                  facecolor='gold', edgecolor='black', linewidths=1.5,
                  zorder=10, label=f'OBSERVED leaf families = {obs_fm:,}')
    ax_fm.set_ylabel("Posterior families present at node (Ωmin-corrected, linear)", fontsize=10)
    ax_fm.set_title(f"Families along path: arc269 root → {subset_label} leaf '{leaf_name}'\n"
                    f"(observed leaf families = {obs_fm:,})", fontsize=10)
    ax_fm.legend(fontsize=9, loc='upper right', framealpha=0.9)
    if ymax_fm is not None:
        ax_fm.set_ylim(0, ymax_fm)
    else:
        ax_fm.set_ylim(bottom=0)

    plt.suptitle(f"arc269 root → {ds.upper()} median-sized leaf path  "
                 f"(F=90,243; median fm at {info['subset_leaf_count']} {ds} leaves = "
                 f"{info['median_fm_in_subset']})",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_png = out_dir / f"arc269_path_{ds}{suffix}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["all", "bounded", "unconstrained"],
                    default="all",
                    help="Which variant to regenerate (default: all)")
    args = ap.parse_args()

    out_dir = Path("validation/outputs")

    # --- Variant-independent setup ---
    tree269, profiles, _, _, _ = load_dataset("arc269")
    print(f"arc269: {tree269.num_leaves} leaves, {tree269.num_nodes} nodes, root={tree269.root}")

    # Subclade-ancestor node indices in arc269
    subclade_ancestor = {}
    for ds in SUBSETS:
        subtree, _, _, _, _ = load_dataset(ds)
        lca = find_lca_in_arc269(tree269, subtree.leaf_names)
        subclade_ancestor[ds] = lca
        print(f"  {ds}: subset has {subtree.num_leaves} leaves → LCA in arc269 = node {lca}")
    print(f"  arc269 root = node {tree269.root}")

    # Csurös' subset-only reproductions: per-node cp/fm for ALL subset nodes.
    # subset_fits[ds] = {"cp": value_at_root, "fm": value_at_root}  (legacy, kept for bar chart)
    # subset_node_values[ds] = {subset_node_idx: {"cp": ..., "fm": ...}}
    # subset_to_arc269[ds]   = {subset_node_idx: arc269_node_idx}
    subset_fits = {}
    subset_node_values = {}
    subset_to_arc269 = {}
    for ds in SUBSETS:
        subtree, _, _, _, _ = load_dataset(ds)
        path = Path(f"validation/outputs/reproduction/{ds}_reproduce.branches.csv")
        if not path.exists():
            print(f"  SKIP subset {ds}: {path} missing")
            continue
        rows = load_branches_csv(path)
        root_idx = subtree.root
        r = rows[root_idx]
        subset_fits[ds] = {
            "cp": float(r["copies_node_corrected"]),
            "fm": float(r["families_present_corrected"]),
        }
        # Full per-node values for the subset tree
        subset_node_values[ds] = {
            i: {"cp": float(row["copies_node_corrected"]),
                "fm": float(row["families_present_corrected"])}
            for i, row in rows.items()
        }
        # Map subset internal nodes → arc269 internal nodes
        subset_to_arc269[ds] = map_subset_to_arc269(subtree, tree269)
        print(f"  {ds}: {len(subset_to_arc269[ds])} subset nodes mapped to arc269")

    # Per-subset median-fm leaf + path (same for any variant)
    rng = np.random.default_rng(2026)
    per_subset_path = {}
    for ds in SUBSETS:
        subtree_s, _, _, _, _ = load_dataset(ds)
        name_to_idx269 = {n: i for i, n in enumerate(tree269.leaf_names)}
        subset_leaf_idxs = [name_to_idx269[n] for n in subtree_s.leaf_names if n in name_to_idx269]
        if not subset_leaf_idxs:
            print(f"  SKIP per-subset path {ds}: no leaves found in arc269")
            continue
        fms_per_leaf = np.array([int((profiles[:, li] > 0).sum()) for li in subset_leaf_idxs])
        median_val = int(np.median(fms_per_leaf))
        diffs = np.abs(fms_per_leaf - median_val)
        tied_positions = np.where(diffs == diffs.min())[0]
        chosen_pos = int(rng.choice(tied_positions))
        chosen_li = subset_leaf_idxs[chosen_pos]
        chosen_name = tree269.leaf_names[chosen_li]
        p = []
        v = chosen_li
        while v >= 0:
            p.append(v)
            v = tree269.parent[v]
        p = p[::-1]
        anc_idx = subclade_ancestor[ds]
        per_subset_path[ds] = {
            "leaf_idx": chosen_li,
            "leaf_name": chosen_name,
            "path": p,
            "depths": list(range(len(p))),
            "leaf_depth": len(p) - 1,
            "obs_cp": int(profiles[:, chosen_li].sum()),
            "obs_fm": int((profiles[:, chosen_li] > 0).sum()),
            "anc_idx": anc_idx,
            "anc_depth": p.index(anc_idx) if anc_idx in p else None,
            "median_fm_in_subset": median_val,
            "subset_leaf_count": len(subset_leaf_idxs),
        }
        # Distributions across ALL subset leaves (raw profile sums) — used for the
        # violin plot around the chosen-leaf gold star.
        per_subset_path[ds]["all_leaf_cps"] = [int(profiles[:, li].sum()) for li in subset_leaf_idxs]
        per_subset_path[ds]["all_leaf_fms"] = [int((profiles[:, li] > 0).sum()) for li in subset_leaf_idxs]
        print(f"  {ds}: median fm at subset leaves = {median_val}; "
              f"chose '{chosen_name}' (fm={per_subset_path[ds]['obs_fm']}, "
              f"cp={per_subset_path[ds]['obs_cp']}, path depth={per_subset_path[ds]['leaf_depth']})")

    subclades = list(SUBSETS.keys())

    def _build_our_subset_paths(ds: str, arc2sub: dict,
                                arc_path: list[int],
                                subset_only_root: str):
        """Build a list of (label, points, color, marker, linestyle) for the
        three subset-only variants (cold / multi / MAP) at this subset, rooted
        at `subset_only_root`. Variants with no fresh CSV produce empty points
        (skipped by the plotter)."""
        out = []
        for label, tmpl, color, marker, linestyle in SUBSET_ONLY_VARIANTS:
            csv_path = Path(tmpl.format(root=subset_only_root, ds=ds))
            pts = []
            if csv_path.exists():
                rows = load_branches_csv(csv_path)
                for d, arc_node in enumerate(arc_path):
                    s = arc2sub.get(arc_node)
                    if s is not None and s in rows:
                        pts.append((
                            d,
                            float(rows[s]["copies_node_corrected"]),
                            float(rows[s]["families_present_corrected"]),
                        ))
            out.append((label, pts, color, marker, linestyle))
        return out

    # --- Per-variant: results + plots ---
    for variant_name, arc269_fits, subset_only_root, variant_suffix in VARIANTS:
        if args.variant != "all":
            wanted = "_unconstrained" if args.variant == "unconstrained" else ""
            if variant_suffix != wanted:
                continue
        print(f"\n========== Variant: {variant_name} ==========")
        results = _compute_results(arc269_fits, tree269, subclade_ancestor)

        # Print comparison table
        print(f"{'subclade':<10} | {'subset-only (Csurös)':>20} | {'arc269 cold ML':>17} | "
              f"{'arc269 MS-polish':>18} | {'arc269 MAP σ=1 (cold)':>22} | "
              f"{'arc269 MAP σ=1 (global)':>24}")
        for ds in SUBSETS:
            row = f"{ds:<10} | "
            s = subset_fits.get(ds)
            row += f"{s['cp']:>6,.0f} cp / {s['fm']:>5,.0f} fm | " if s else f"{'n/a':>20} | "
            for fit_name in ["cold_ML", "multistart_polish_ML",
                             "MAP_sigma1_cold", "MAP_sigma1_global"]:
                r = results.get(fit_name, {}).get(ds)
                row += (f"{r['cp']:>6,.0f} / {r['fm']:>5,.0f}  | " if r
                        else f"{'n/a':>17} | ")
            print(row)

        # Save JSON
        out_json = out_dir / f"subclade_ancestor_comparison{variant_suffix}.json"
        out_json.write_text(json.dumps({
            "variant": variant_name,
            "subset_only_csuros": subset_fits,
            "arc269_fits_at_subclade_ancestors": results,
            "subclade_to_arc269_node": subclade_ancestor,
        }, indent=2))
        print(f"  wrote {out_json}")

        # Bar chart + per-subset paths for set 1 (no MAP polish)
        _make_bar_chart(SET1_ARC269, variant_suffix, results, subset_fits, subclades, out_dir)
        # Pass 1: collect every line + violin data point across all 4 subsets,
        # so each plot can use a shared y-axis cap. Pass 2: actually render.
        set1_payloads = {}
        for ds in SUBSETS:
            if ds not in per_subset_path: continue
            sub2arc = subset_to_arc269.get(ds, {})
            arc2sub = {a: s for s, a in sub2arc.items()}
            sub_vals = subset_node_values.get(ds, {})
            csuros_path_pts = []
            for d, arc_node in enumerate(per_subset_path[ds]["path"]):
                s = arc2sub.get(arc_node)
                if s is not None and s in sub_vals:
                    csuros_path_pts.append((d, sub_vals[s]["cp"], sub_vals[s]["fm"]))
            our_subset_paths = _build_our_subset_paths(
                ds, arc2sub, per_subset_path[ds]["path"], subset_only_root)
            set1_payloads[ds] = (csuros_path_pts, our_subset_paths)
        # Gather extents across all subsets for shared caps.
        all_cp, all_fm = [], []
        for ds, (csu, ours) in set1_payloads.items():
            for fit_name in SET1_ARC269:
                csv_path = arc269_fits.get(fit_name)
                if csv_path is None or not csv_path.exists(): continue
                rows = load_branches_csv(csv_path)
                for i in per_subset_path[ds]["path"]:
                    all_cp.append(float(rows[i]["copies_node_corrected"]))
                    all_fm.append(float(rows[i]["families_present_corrected"]))
            for (_d, cp, fm) in csu: all_cp.append(cp); all_fm.append(fm)
            for (_l, pts, *_r) in ours:
                for (_d, cp, fm) in pts: all_cp.append(cp); all_fm.append(fm)
            all_cp.extend(per_subset_path[ds]["all_leaf_cps"])
            all_fm.extend(per_subset_path[ds]["all_leaf_fms"])
            all_cp.append(per_subset_path[ds]["obs_cp"])
            all_fm.append(per_subset_path[ds]["obs_fm"])
        cap_cp = max(all_cp) * 1.05 if all_cp else None
        cap_fm = max(all_fm) * 1.05 if all_fm else None
        print(f"  Set 4 (mixed-min) shared y-caps: cp={cap_cp:.0f} fm={cap_fm:.0f}")
        for ds, (csuros_path_pts, our_subset_paths) in set1_payloads.items():
            _make_path_plot(ds, SET1_ARC269, variant_suffix, arc269_fits,
                            per_subset_path[ds], SUBSETS[ds]["label"], out_dir,
                            csuros_subset_path=csuros_path_pts,
                            our_subset_paths=our_subset_paths,
                            subset_leaf_cps=per_subset_path[ds]["all_leaf_cps"],
                            subset_leaf_fms=per_subset_path[ds]["all_leaf_fms"],
                            ymax_cp=cap_cp, ymax_fm=cap_fm)

        # Set 2 (with MAP polish line) — only if the 15-start global rates exist.
        if arc269_fits["MAP_sigma1_global"].exists():
            set2_suffix = variant_suffix + "_with_map_polish"
            _make_bar_chart(SET2_ARC269, set2_suffix, results, subset_fits, subclades, out_dir)
            for ds in SUBSETS:
                if ds not in per_subset_path:
                    continue
                sub2arc = subset_to_arc269.get(ds, {})
                arc2sub = {a: s for s, a in sub2arc.items()}
                sub_vals = subset_node_values.get(ds, {})
                csuros_path_pts = []
                for d, arc_node in enumerate(per_subset_path[ds]["path"]):
                    s = arc2sub.get(arc_node)
                    if s is not None and s in sub_vals:
                        csuros_path_pts.append((d, sub_vals[s]["cp"], sub_vals[s]["fm"]))
                our_subset_paths2 = _build_our_subset_paths(
                    ds, arc2sub, per_subset_path[ds]["path"], subset_only_root)
                _make_path_plot(ds, SET2_ARC269, set2_suffix, arc269_fits,
                                per_subset_path[ds], SUBSETS[ds]["label"], out_dir,
                                csuros_subset_path=csuros_path_pts,
                                our_subset_paths=our_subset_paths2,
                                subset_leaf_cps=per_subset_path[ds]["all_leaf_cps"],
                                subset_leaf_fms=per_subset_path[ds]["all_leaf_fms"])
        else:
            print(f"  SKIP set-2 plots: {arc269_fits['MAP_sigma1_global']} missing")

    _write_shared_legend(out_dir)


def _write_shared_legend(out_dir: Path):
    """Standalone legend for the arc269_path_<ds>.png series. Embedded
    once in the README above the 4-subset plot stack."""
    import matplotlib.lines as mlines
    handles = []
    for k in ("cold_ML", "MAP_sigma1_cold", "MAP_brownian", "ML_subset_seed"):
        handles.append(mlines.Line2D([], [], marker='o', color=FITS_COLOR[k],
            lw=1.5, markersize=7, label=FITS_LABEL[k].replace("\n"," ")))
    # subset-only variants
    for label, _, color, mark, ls in SUBSET_ONLY_VARIANTS:
        handles.append(mlines.Line2D([], [], marker=mark, color=color, lw=2.0,
            linestyle=ls, markersize=7, label=label))
    # Csurös subset-only baseline (hollow gray diamond)
    handles.append(mlines.Line2D([], [], marker='D', color='gray', lw=1.8,
        linestyle='--', markersize=9, markerfacecolor='none',
        markeredgecolor='dimgray', markeredgewidth=1.5,
        label='Csurös subset-only ML (published rates, reference)'))
    # observed leaf
    handles.append(mlines.Line2D([], [], marker='*', color='gold', lw=0,
        markersize=18, markeredgecolor='black', markeredgewidth=1.5,
        label='OBSERVED leaf cp/fm (gold ★, value labelled in each per-plot legend)'))

    fig = plt.figure(figsize=(16, 1.6))
    fig.legend(handles=handles, loc='center', ncol=3, frameon=False,
               fontsize=10, columnspacing=2.0)
    out_png = out_dir / "arc269_path_legend.png"
    out_pdf = out_dir / "arc269_path_legend.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote shared legend {out_png}")


if __name__ == "__main__":
    main()
