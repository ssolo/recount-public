"""Per-subset reproduction-of-Csurös trajectory plots.

A simplified variant of Set 1 (subset reproduction path) showing only
three lines per subset panel — the three fits relevant to the
"reproduction of Csurős' published rates" claim:

  - Csurös subset-only ML at min = 4  (gray dashed diamond, reference)
  - Our Brown-extend ML at min = 4    (olive ◀ — warm-started from MAP Brownian)
  - Our MAP Brownian σ = 1 at min = 4 (purple ■ — the recommended baseline)

Per panel: cp + fm along the path arc269-root → subset's median-fm
leaf, with the gold-star OBSERVED leaf cp/fm and a violin of the
distribution over the *other* leaves of the subset.

This is the bare-bones "do we reproduce Csurős?" view — no sum-sweep
gradient, no full-tree comparison, just the three fits at the
canonical Ωmin = 4 setting.

Outputs:
  validation/outputs/subset_reproduction_path_{ds}.{png,pdf}      (4 panels)
  validation/outputs/subset_reproduction_path_legend.{png,pdf}    (shared legend)
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
from validation._shared import load_dataset
from validation import _pubstyle as ps
from validation.brownian_split_path_plots import (
    OUT, SUBSET_LABEL, DATASETS,
    C_CSUROS, C_BROWN, C_EXTEND,
    load_branches, get_L0_for_fit, map_subset_to_arc269,
    median_fm_leaf, per_leaf_stats, subset_min4_mask, arc269_path,
    _line_data, _set_extents,
    plot_lines, add_observed_violin, setup_axes,
    add_subclade_ancestor_vline, finalize_axes,
)

# importing brownian_split_path_plots already calls ps.apply(); call it
# again here so this script also installs the shared style if run alone.
ps.apply()


def _reproduction_data(ds, tree269, profiles269, rng):
    """Load the three fits' branches.csv + the leaf distribution."""
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name, sub_leaf_idxs = median_fm_leaf(
        tree269, subtree, profiles269, rng)
    mask = subset_min4_mask(profiles269, sub_leaf_idxs)
    cps, fms, obs_cp, obs_fm = per_leaf_stats(
        profiles269, sub_leaf_idxs, chosen_li, mask)
    F_mask = int(mask.sum())
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    paths = {
        "csuros_sub":       OUT / f"reproduction/{ds}_reproduce.branches.csv",
        "subset_brown_min4": OUT / f"brownian_{ds}/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_ext":  OUT / f"brownian_extend_{ds}/{ds}_ml.branches.csv",
    }
    branches = {k: load_branches(p) for k, p in paths.items()}
    L0s = {k: get_L0_for_fit(p) for k, p in paths.items()}
    return {"subtree": subtree, "chosen_li": chosen_li, "chosen_name": chosen_name,
            "path": path, "arc2sub": arc2sub, "F_mask": F_mask,
            "cps": cps, "fms": fms, "obs_cp": obs_cp, "obs_fm": obs_fm,
            "branches": branches, "L0s": L0s}


def _reproduction_extract_line_extents(data, mode="observed"):
    b = data["branches"]; p = data["path"]; a = data["arc2sub"]; L0s = data["L0s"]
    out = []
    for key in ("csuros_sub", "subset_brown_min4", "subset_brown_ext"):
        out.append(_line_data(b[key], p, a, use_sub=True, mode=mode, L0=L0s[key]))
    return out


def plot_reproduction_set(ds, tree269, profiles269, rng, ymax_cp=None, ymax_fm=None,
                          data=None, mode="observed"):
    if data is None:
        data = _reproduction_data(ds, tree269, profiles269, rng)
    chosen_name = data["chosen_name"]; F_mask = data["F_mask"]; path = data["path"]
    leaf_depth = len(path) - 1
    b = data["branches"]; arc2sub = data["arc2sub"]; L0s = data["L0s"]

    fig = plt.figure(figsize=(ps.WIDTH_2COL, 3.7))
    fig.suptitle(
        f"{SUBSET_LABEL[ds]}  ·  reproduction of Csurös' min = 4 fit  ·  "
        f"arc269 root to median-fm leaf ‘{chosen_name}’   (F = {F_mask:,})")
    ax_cp, ax_fm = setup_axes(fig, leaf_depth, mode=mode)

    lines = [
        # 1. Csurős subset-only ML reference — grey dashed open diamond
        dict(branches=b["csuros_sub"], L0=L0s["csuros_sub"],
             color=C_CSUROS, lw=1.6, linestyle='--',
             marker='D', markersize=6, markerfacecolor='none',
             markeredgecolor=C_CSUROS, markeredgewidth=1.1, zorder=8, use_sub=True),
        # 2. Our MAP Brownian σ=1 — purple solid square
        dict(branches=b["subset_brown_min4"], L0=L0s["subset_brown_min4"],
             color=C_BROWN, lw=1.8, linestyle='-',
             marker='s', markersize=6, markerfacecolor=C_BROWN,
             markeredgecolor=C_BROWN, markeredgewidth=0.8,
             alpha=1.0, zorder=10, use_sub=True),
        # 3. Our Brown-extend ML — green dash-dot down-triangle
        dict(branches=b["subset_brown_ext"], L0=L0s["subset_brown_ext"],
             color=C_EXTEND, lw=1.6, linestyle='-.',
             marker='v', markersize=6, markerfacecolor=C_EXTEND,
             markeredgecolor=C_EXTEND, markeredgewidth=0.8,
             zorder=9, use_sub=True),
    ]
    plot_lines(ax_cp, ax_fm, lines, list(range(len(path))),
               arc2sub=arc2sub, path=path, mode=mode)
    add_observed_violin(ax_cp, ax_fm, data["cps"], data["fms"],
                        data["obs_cp"], data["obs_fm"], leaf_depth)
    finalize_axes(ax_cp, ax_fm, ymax_cp=ymax_cp, ymax_fm=ymax_fm)
    add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub)
    fig.tight_layout()
    suffix = "_l0corr" if mode == "l0corr" else ""
    ps.save_fig(fig, OUT / f"subset_reproduction_path_{ds}{suffix}")
    plt.close(fig)
    print(f"  {ds} {mode}: wrote subset_reproduction_path_{ds}{suffix}.png + .pdf")


def write_reproduction_legend():
    handles = [
        mlines.Line2D([], [], marker='D', color=C_CSUROS, lw=1.6, linestyle='--',
                      markersize=6, markerfacecolor='none',
                      markeredgecolor=C_CSUROS, markeredgewidth=1.1,
                      label='Csurős subset-only ML  (reference baseline; min = 4 from '
                            'Csurős 2026 PNAS bundled rates)'),
        mlines.Line2D([], [], marker='s', color=C_BROWN, lw=1.8,
                      markersize=6, markerfacecolor=C_BROWN,
                      markeredgecolor=C_BROWN, markeredgewidth=0.8,
                      label='our MAP Brownian σ = 1  (recommended baseline; '
                            'tree-Brownian autocorrelated log-rate prior)'),
        mlines.Line2D([], [], marker='v', color=C_EXTEND, lw=1.6, linestyle='-.',
                      markersize=6, markerfacecolor=C_EXTEND,
                      markeredgecolor=C_EXTEND, markeredgewidth=0.8,
                      label='our Brown-extend ML  (warm-started from MAP Brownian, '
                            'short pure-ML fine-tune)'),
        mlines.Line2D([], [], marker='*', color=ps.ORANGE, lw=0, markersize=12,
                      markeredgecolor=ps.C_DATA, markeredgewidth=0.9,
                      label='observed cp/fm at the subset\'s median-fm leaf '
                            '(what the fits should reproduce at the leaf tip)'),
        mlines.Line2D([], [], marker=None, color=ps.C_VIOLIN_FACE, lw=7,
                      label='violin — observed cp/fm over the other leaves of the subset'),
    ]
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 1.45))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False,
               columnspacing=1.8)
    ps.save_fig(fig, OUT / "subset_reproduction_path_legend")
    plt.close(fig)
    print("  legend: wrote subset_reproduction_path_legend.png + .pdf")


def main():
    print("Loading arc269...")
    tree269, profiles269, _, _, _ = load_dataset("arc269")
    rng = np.random.default_rng(2026)
    data_by_ds = {ds: _reproduction_data(ds, tree269, profiles269, rng)
                  for ds in DATASETS}

    # Shared y-cap across the 4 subsets per mode
    for mode in ("observed", "l0corr"):
        lines = {ds: _reproduction_extract_line_extents(d, mode=mode)
                 for ds, d in data_by_ds.items()}
        viol = {ds: (d["cps"], d["fms"], d["obs_cp"], d["obs_fm"])
                for ds, d in data_by_ds.items()}
        cap_cp, cap_fm = _set_extents(lines, viol)
        print(f"\n== mode={mode} == cap_cp={cap_cp:.0f}  cap_fm={cap_fm:.0f}")
        for ds in DATASETS:
            plot_reproduction_set(ds, tree269, profiles269, None, mode=mode,
                                  ymax_cp=cap_cp, ymax_fm=cap_fm,
                                  data=data_by_ds[ds])

    print("\n--- legend ---")
    write_reproduction_legend()
    print("\nDONE.")


if __name__ == "__main__":
    main()
