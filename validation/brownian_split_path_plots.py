"""Three sets of per-subset Brownian-only path plots, split from the
omnibus brownian_only_path_plot.py for clarity.

For each subset (dpann80, proteo75, eury114, ed194) plot the
arc269 root → median-fm leaf path with three views:

  SET 1 — `subset_min4_path_{ds}.png`
      Subset-only fits at the Csurös-matched min=4 condition,
      with intermediate sum>=2 / sum>=3 subset Brownian variants
      and the Csurös subset-only ML reference.
      No arc269 line — pure subset reproduction comparison.

  SET 2 — `min1_path_{ds}.png`
      Subset Brownian over the FULL subset complement (mc=1, no filter)
      vs arc269 canonical mc=1 (full F=90,243 fit). NO Csurös line —
      Csurös never fits at min=1, so the comparison would be
      apples-to-oranges.

  SET 3 — `union_subset_path_{ds}.png`
      Csurös subset min=4 reference vs a NEW arc269 Brownian fit
      trained on the UNION of per-subset min=4 families (= families
      with sum>=4 over the leaves of at least one of the four
      subsets). This is the closest like-for-like comparison
      between Csurös' per-clade pipeline and a single full-tree
      Brownian fit on the same family universe.

All three sets include the gold violin/star showing the observed
leaf cp/fm at depth=leaf_depth.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from validation._shared import load_dataset
from validation import _pubstyle as ps

ps.apply()   # install the shared README figure style — call once

OUT = Path("validation/outputs")
DATASETS = ["dpann80", "proteo75", "eury114", "ed194"]
SUBSET_LABEL = {"dpann80": "DPANN (D80)", "proteo75": "Proteoarchaea (P75)",
                "eury114": "Methanobacteriati (E114)", "ed194": "Euryarchaeota (ED194)"}

# --- semantic colours for the fit families (from the shared palette) --
# Each fit maps to one Okabe-Ito colour so every README path plot reads
# the same:  Csurös reference = neutral grey, MAP Brownian = purple,
# Brown-extend ML = green, arc269 canonical mc=1 = blue (it IS the
# Omega_min = 1 fit), arc269 union-of-subsets = vermilion.
C_CSUROS  = ps.C_NEUTRAL    # grey   — Csurös subset-only ML reference
C_BROWN   = ps.C_BROWN      # purple — MAP Brownian sigma = 1
C_EXTEND  = ps.GREEN        # green  — Brown-extend ML (warm-started)
C_ARC_M1  = ps.C_OMIN1      # blue   — arc269 canonical full-tree mc = 1
C_ARC_UNI = ps.VERMILION    # verm.  — arc269 fit on union-of-subsets min=4

# Shared style for the sum>=N family-size filter sweep, used by BOTH
# Set 1 (per-clade subset-only fits, plot_subset_min4_set) and Set 1.5
# (one arc269 full-tree fit per threshold, plot_arc269_sumsweep_set), so
# the two figures can be read against each other threshold-for-threshold:
# identical colour + marker per N.  Emphasis is by geometry, not colour —
#   sum>=4  (Csurös' canonical filter): solid, bold, large dark-edged D,
#                                       in the palette vermilion;
#   sum>=1  (no filter, the sweep anchor): solid, prominent o, in black;
#   every other threshold: thin de-emphasized blue ribbon, small marker.
SUM_SWEEP_STYLE = {
    1:  dict(color=ps.C_DATA,    marker='o', linestyle='-', lw=1.8, markersize=6.0, alpha=1.00, zorder=8),
    2:  dict(color=ps.SKYBLUE,   marker='s', linestyle=':', lw=1.1, markersize=4.0, alpha=0.80, zorder=4),
    3:  dict(color=ps.SKYBLUE,   marker='^', linestyle=':', lw=1.1, markersize=4.0, alpha=0.80, zorder=4),
    4:  dict(color=ps.VERMILION, marker='D', linestyle='-', lw=2.0, markersize=7.5, alpha=1.00, zorder=9,
             markerfacecolor=ps.VERMILION, markeredgecolor=ps.C_DATA, markeredgewidth=0.9),
    5:  dict(color=ps.SKYBLUE,   marker='v', linestyle=':', lw=1.1, markersize=4.0, alpha=0.70, zorder=4),
    6:  dict(color=ps.SKYBLUE,   marker='X', linestyle=':', lw=1.1, markersize=4.0, alpha=0.70, zorder=4),
    7:  dict(color=ps.SKYBLUE,   marker='P', linestyle=':', lw=1.1, markersize=4.0, alpha=0.70, zorder=4),
    8:  dict(color=ps.SKYBLUE,   marker='h', linestyle=':', lw=1.1, markersize=4.0, alpha=0.70, zorder=4),
    9:  dict(color=ps.SKYBLUE,   marker='d', linestyle=':', lw=1.1, markersize=4.0, alpha=0.70, zorder=4),
    10: dict(color=ps.SKYBLUE,   marker='*', linestyle=':', lw=1.1, markersize=5.0, alpha=0.70, zorder=4),
}


def _sweep_handles():
    """Legend handles for the sum>=1..10 filter sweep — identical styling in
    the Set 1 and Set 1.5 legends so the two plots read together."""
    keys = ("color", "marker", "linestyle", "lw", "markersize", "alpha",
            "markerfacecolor", "markeredgecolor", "markeredgewidth")
    hs = []
    for n in range(1, 11):
        st = SUM_SWEEP_STYLE[n]
        if n == 1:
            lbl = "sum ≥ 1  (no filter — sweep anchor)"
        elif n == 4:
            lbl = "sum ≥ 4  (Csurös' canonical filter — emphasized line)"
        elif n == 10:
            lbl = "sum ≥ 10  (most stringent)"
        else:
            lbl = f"sum ≥ {n}"
        hs.append(mlines.Line2D([], [], label=lbl,
                                **{k: st[k] for k in keys if k in st}))
    return hs


def load_branches(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["node_index"])] = r
            except (KeyError, ValueError):
                continue
    return out


def get_L0_for_fit(branches_csv_path: Path) -> float:
    """Find the L(0) for a fit, either from its summary.json or by deriving it
    from the observed/corrected ratio at any non-leaf node in branches.csv.

    Returns L(0) ∈ [0, 1). Defaults to 0 if not derivable."""
    import json
    # Try sibling summary JSON first.
    cand_summaries = list(branches_csv_path.parent.glob("*_summary*.json"))
    cand_summaries += list(branches_csv_path.parent.glob("*sigma1.0_summary.json"))
    for s in cand_summaries:
        try:
            d = json.load(open(s))
            L0 = float(d.get("final", {}).get("L0", 0.0))
            if L0 > 0: return L0
        except Exception: pass
    # Else derive from observed/corrected ratio.
    try:
        rows = load_branches(branches_csv_path)
        for r in rows.values():
            obs = float(r.get("copies_node_observed", 0))
            cor = float(r.get("copies_node_corrected", 0))
            if obs > 0 and cor > obs:
                return max(0.0, 1.0 - obs / cor)
    except Exception: pass
    return 0.0


def map_subset_to_arc269(subtree, tree269) -> dict:
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
        if not idxs:
            continue
        ancs = []
        for li in idxs:
            s = set(); v = li
            while v >= 0:
                s.add(int(v)); v = tree269.parent[v]
            ancs.append(s)
        mapping[u] = min(set.intersection(*ancs))
    return mapping


def median_fm_leaf(tree269, subtree, profiles269, rng):
    """Pick a representative leaf for the path (consistent across all 3 sets).

    Returns (chosen_li, chosen_name, sub_leaf_idxs). The chosen leaf is picked
    based on the FULL F=90,243 profile median fm so all 3 sets share the same
    representative leaf. Per-leaf cp/fm distributions and the chosen leaf's
    observed counts are computed separately by per_leaf_stats() with whatever
    family-universe mask the specific plot requires."""
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    sub_leaf_idxs = [name_to_idx[n] for n in subtree.leaf_names if n in name_to_idx]
    fms_full = np.array([int((profiles269[:, li] > 0).sum()) for li in sub_leaf_idxs])
    median_fm = int(np.median(fms_full))
    diffs = np.abs(fms_full - median_fm)
    chosen_pos = int(rng.choice(np.where(diffs == diffs.min())[0]))
    chosen_li = sub_leaf_idxs[chosen_pos]
    chosen_name = tree269.leaf_names[chosen_li]
    return chosen_li, chosen_name, sub_leaf_idxs


def per_leaf_stats(profiles, sub_leaf_idxs, chosen_li, mask=None):
    """Per-leaf cp/fm distribution and chosen-leaf observation, restricted to
    families where `mask` is True. mask=None means use all families (no filter)."""
    p = profiles[mask] if mask is not None else profiles
    fms = np.array([int((p[:, li] > 0).sum()) for li in sub_leaf_idxs])
    cps = np.array([int(p[:, li].sum()) for li in sub_leaf_idxs])
    obs_cp = int(p[:, chosen_li].sum())
    obs_fm = int((p[:, chosen_li] > 0).sum())
    return cps, fms, obs_cp, obs_fm


def subset_min4_mask(profiles, sub_leaf_idxs):
    """Mask: families with sum(profile over subset leaves) >= 4 (Csurös' filter)."""
    return profiles[:, sub_leaf_idxs].sum(axis=1) >= 4


def union_min4_mask(profiles, tree269, dataset_names=("dpann80","proteo75","eury114","ed194")):
    """Mask: union of per-subset min=4 families across all 4 subsets."""
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    union = np.zeros(profiles.shape[0], dtype=bool)
    for ds in dataset_names:
        subtree, _, _, _, _ = load_dataset(ds)
        sub_li = [name_to_idx[n] for n in subtree.leaf_names if n in name_to_idx]
        union |= profiles[:, sub_li].sum(axis=1) >= 4
    return union


def arc269_path(tree269, leaf_idx):
    path = []
    v = leaf_idx
    while v >= 0:
        path.append(v); v = tree269.parent[v]
    return path[::-1]


def _branch_vals(node_row, mode, L0):
    """Return (cp, fm) for one branches.csv row, in the requested mode.

      mode='observed' — literal posterior over OBSERVED families (uncorrected).
      mode='l0corr'   — L(0)-amplified posterior. Prefers the CSV's
                        `*_corrected` column when it was computed bit-for-bit
                        from Csurös' Pn-tensor recursion (mc>1 fits — leaves
                        ARE amplified by that path because P(leaf cp ≥ 1 |
                        unobserved) > 0 at Ωmin>1). For mc=1 fits where the
                        CSV `*_corrected` equals `*_observed` (Ωmin=1 code
                        path is gated off), falls back to the multiplicative
                        approximation observed × 1/(1−L0) at INTERNAL nodes
                        and leaves leaves unchanged (Ωmin=1 unobserved = empty
                        profile → leaves have ξ=0 by definition).
    """
    obs_cp = float(node_row.get("copies_node_observed",
                                 node_row.get("copies_node_corrected", 0)))
    obs_fm = float(node_row.get("families_present_observed",
                                 node_row.get("families_present_corrected", 0)))
    if mode == "observed":
        return obs_cp, obs_fm
    # mode == 'l0corr'
    cor_cp = float(node_row.get("copies_node_corrected", obs_cp))
    cor_fm = float(node_row.get("families_present_corrected", obs_fm))
    if cor_cp > obs_cp or cor_fm > obs_fm:
        # CSV's corrected column already has proper Csurös-style amplification
        # (this is the mc>1 code path including Csurös reproduction fits).
        return cor_cp, cor_fm
    # CSV equals observed at this node — mc=1 fit. Apply multiplicative L(0)
    # amplification at internal nodes only. Leaves stay as observed (Ωmin=1
    # unobserved profile has ξ=0 at every leaf by definition).
    is_leaf = str(node_row.get("is_leaf", "False")).lower() == "true"
    if 0 < L0 < 1 and not is_leaf:
        amp = 1.0 / (1.0 - L0)
        return obs_cp * amp, obs_fm * amp
    return obs_cp, obs_fm


def plot_lines(ax_cp, ax_fm, lines, depths, arc2sub=None, path=None, mode="observed"):
    """`lines` is a list of dicts with keys: branches, L0 (optional, used in
    'l0corr' mode), color, lw, ls, marker, markersize, markerfacecolor,
    markeredgecolor, markeredgewidth, alpha, use_sub (True = look up via
    arc2sub mapping, False = direct arc269 node).

    mode='observed' uses *_observed columns; mode='l0corr' applies the
    multiplicative 1/(1-L(0)) amplification per line.
    """
    for line in lines:
        b = line["branches"]
        if not b: continue
        L0 = float(line.get("L0", 0.0))
        d_pts, cp_pts, fm_pts = [], [], []
        for d, av in enumerate(path):
            if line.get("use_sub", False):
                node = arc2sub.get(av)
            else:
                node = av
            if node is None or node not in b: continue
            try:
                cp, fm = _branch_vals(b[node], mode, L0)
                d_pts.append(d); cp_pts.append(cp); fm_pts.append(fm)
            except (KeyError, ValueError):
                continue
        if not d_pts: continue
        kw = {k: line[k] for k in ("color","lw","linestyle","marker","markersize",
                                    "markerfacecolor","markeredgecolor",
                                    "markeredgewidth","alpha","zorder") if k in line}
        ax_cp.plot(d_pts, cp_pts, **kw)
        ax_fm.plot(d_pts, fm_pts, **kw)


def add_observed_violin(ax_cp, ax_fm, cps_per_leaf, fms_per_leaf, obs_cp, obs_fm, leaf_depth):
    for ax, all_vals, chosen, label in [
        (ax_cp, cps_per_leaf, obs_cp, "copies"),
        (ax_fm, fms_per_leaf, obs_fm, "families"),
    ]:
        vp = ax.violinplot([all_vals.astype(float)], positions=[leaf_depth],
                           widths=0.8, showmeans=False, showextrema=True, showmedians=True)
        for body in vp['bodies']:
            body.set_alpha(0.95); body.set_facecolor(ps.C_VIOLIN_FACE)
            body.set_edgecolor(ps.C_VIOLIN_EDGE); body.set_linewidth(0.7)
        for key in ('cmins', 'cmaxes', 'cmedians', 'cbars'):
            if key in vp:
                vp[key].set_color(ps.C_VIOLIN_EDGE); vp[key].set_linewidth(0.9)
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(all_vals))
        ax.scatter(leaf_depth + jitter, all_vals, s=5, color=ps.C_VIOLIN_EDGE,
                   alpha=0.35, lw=0, zorder=5)
        ax.scatter([leaf_depth], [chosen], marker='*', s=200,
                   facecolor=ps.ORANGE, edgecolor=ps.C_DATA, linewidths=0.9,
                   zorder=10, label=f'observed leaf {label} = {chosen:,}')


def setup_axes(fig, leaf_depth, title_suffix="", mode="observed"):
    ax_cp, ax_fm = fig.subplots(1, 2)
    suf = "uncorrected" if mode == "observed" else "L(0)-amplified"
    ax_cp.set_title(f"posterior copies per node  ({suf})")
    ax_fm.set_title(f"posterior families present per node  ({suf})")
    for ax in (ax_cp, ax_fm):
        ax.set_xlabel(f"depth   (0 = LACA / arc269 root, {leaf_depth} = leaf)")
        ps.grid(ax)
    ax_cp.set_ylabel("posterior expected copies")
    ax_fm.set_ylabel("posterior families present")
    return ax_cp, ax_fm


def _annotation_y_va(ax, x_lo, x_hi):
    """Pick a top- or bottom-anchored position for the subclade-ancestor
    label: whichever side has more empty vertical room across the x-span the
    label covers. Some panels run their path lines high near the line (so the
    bottom is clear), others run them low (so the top is clear) — decided per
    panel from the data actually plotted."""
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    lo_frac = hi_frac = None
    if span > 0:
        for ln in ax.get_lines():
            for xv, yv in zip(ln.get_xdata(), ln.get_ydata()):
                if yv != yv:                       # skip NaN gaps
                    continue
                if x_lo <= xv <= x_hi:
                    f = (yv - y0) / span
                    lo_frac = f if lo_frac is None else min(lo_frac, f)
                    hi_frac = f if hi_frac is None else max(hi_frac, f)
    if lo_frac is None:                            # no path data under the label
        return 0.035, 'bottom'
    return (0.965, 'top') if (1.0 - hi_frac) > lo_frac else (0.035, 'bottom')


def add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub, label=None):
    """Mark the depth at which the subclade's LCA (= subset root in arc269)
    sits on the path with a gray dotted vertical line. Helps the reader see
    where 'arc269-internal nodes' (above this line) end and 'subset-internal
    nodes' (below this line) begin.

    Call this *after* finalize_axes, so the y-limits are final: the label is
    anchored at the top or the bottom of each panel depending on where that
    panel's path lines sit near the line (see _annotation_y_va)."""
    if not arc2sub: return
    # Subset root maps to the SHALLOWEST arc269 node in arc2sub.values(); i.e.
    # the smallest arc269 node index among the path's nodes that are in arc2sub.
    # Equivalently: the first depth at which av is in arc2sub.
    anc_depth = None
    for d, av in enumerate(path):
        if av in arc2sub:
            anc_depth = d
            break
    if anc_depth is None or anc_depth == 0: return
    for ax in (ax_cp, ax_fm):
        y_frac, va = _annotation_y_va(ax, anc_depth, anc_depth + 4.0)
        ax.axvline(anc_depth, color=ps.C_NEUTRAL, linestyle=(0, (2, 2)),
                   alpha=0.7, lw=0.7, zorder=1)
        ax.text(anc_depth + 0.12, y_frac,
                label or f"subclade ancestor\n(depth {anc_depth})",
                color=ps.C_NEUTRAL, va=va, ha='left', fontsize=6.5,
                transform=ax.get_xaxis_transform(), zorder=12,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.75))


def finalize_axes(ax_cp, ax_fm, ymax_cp=None, ymax_fm=None):
    """Call after all lines + violin are drawn — sets bottom=0, autoscales top
    (or uses the explicit ymax_cp / ymax_fm caps if provided — used to share
    y-axis limits across the 4 subsets in a given set)."""
    for ax in (ax_cp, ax_fm):
        ax.relim()
        ax.autoscale_view()
        ax.legend(loc='upper right', labelspacing=0.4, borderaxespad=0.5)
    ax_cp.set_ylim(0, ymax_cp if ymax_cp is not None else ax_cp.get_ylim()[1])
    ax_fm.set_ylim(0, ymax_fm if ymax_fm is not None else ax_fm.get_ylim()[1])


def _line_data(branches, path, arc2sub=None, use_sub=False, mode="observed", L0=0.0):
    """Extract (cp, fm) values for a line — for ymax computation without rendering."""
    cps, fms = [], []
    if not branches:
        return cps, fms
    for av in path:
        node = arc2sub.get(av) if use_sub else av
        if node is None or node not in branches: continue
        try:
            cp, fm = _branch_vals(branches[node], mode, L0)
            cps.append(cp); fms.append(fm)
        except (KeyError, ValueError): continue
    return cps, fms


def _set_extents(set_lines_per_ds, violin_data_per_ds):
    """Compute (cp_max, fm_max) over all 4 subsets in a set so the y-axis
    caps can be shared. Adds a 5% headroom above the largest observed value."""
    cp_vals, fm_vals = [], []
    for lines in set_lines_per_ds.values():
        for cps, fms in lines:
            cp_vals.extend(cps); fm_vals.extend(fms)
    for cps_leaf, fms_leaf, obs_cp, obs_fm in violin_data_per_ds.values():
        cp_vals.extend(cps_leaf.tolist()); cp_vals.append(obs_cp)
        fm_vals.extend(fms_leaf.tolist()); fm_vals.append(obs_fm)
    cp_max = max(cp_vals) * 1.05 if cp_vals else 1.0
    fm_max = max(fm_vals) * 1.05 if fm_vals else 1.0
    return cp_max, fm_max


# ===== SET 1: subset-only min=4 + sum>=2/3 sweep + Csurös ref =====
def _set1_data(ds, tree269, profiles269, rng):
    """Build SET 1 lines + violin data for ds. Returns dict with everything
    needed to either compute extents or actually render the plot.

    Set 1 spans the full subset sum-sweep: sum>=1 (full subset complement,
    no filter) through sum>=7 (most stringent), plus Brown-extend ML at
    min=4 + Csurös subset-only ML at min=4 as references."""
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name, sub_leaf_idxs = median_fm_leaf(tree269, subtree, profiles269, rng)
    mask = subset_min4_mask(profiles269, sub_leaf_idxs)
    cps, fms, obs_cp, obs_fm = per_leaf_stats(profiles269, sub_leaf_idxs, chosen_li, mask)
    F_mask = int(mask.sum())
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    paths = {
        # sum>=1 (no filter) over the full subset complement
        "subset_brown_sum1": OUT / f"brownian_{ds}_full/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum2": OUT / f"brownian_{ds}_sum2/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum3": OUT / f"brownian_{ds}_sum3/{ds}_map_sigma1.0.branches.csv",
        # sum>=4 (= Csurös min=4 canonical, brownian_<ds>/)
        "subset_brown_sum4": OUT / f"brownian_{ds}/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum5": OUT / f"brownian_{ds}_sum5/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum6": OUT / f"brownian_{ds}_sum6/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum7": OUT / f"brownian_{ds}_sum7/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum8": OUT / f"brownian_{ds}_sum8/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum9": OUT / f"brownian_{ds}_sum9/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_sum10": OUT / f"brownian_{ds}_sum10/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_ext":  OUT / f"brownian_extend_{ds}/{ds}_ml.branches.csv",
        "csuros_sub":        OUT / f"reproduction/{ds}_reproduce.branches.csv",
    }
    branches = {k: load_branches(p) for k, p in paths.items()}
    L0s      = {k: get_L0_for_fit(p) for k, p in paths.items()}
    return {"subtree": subtree, "chosen_li": chosen_li, "chosen_name": chosen_name,
            "path": path, "arc2sub": arc2sub, "F_mask": F_mask,
            "cps": cps, "fms": fms, "obs_cp": obs_cp, "obs_fm": obs_fm,
            "branches": branches, "L0s": L0s}


def _set1_extract_line_extents(data, mode="observed"):
    """Return list of (line_cps, line_fms) tuples for ymax computation."""
    b = data["branches"]; p = data["path"]; a = data["arc2sub"]; L0s = data["L0s"]
    out = []
    for key in ("subset_brown_sum1","subset_brown_sum2","subset_brown_sum3",
                "subset_brown_sum4","subset_brown_sum5","subset_brown_sum6",
                "subset_brown_sum7","subset_brown_sum8","subset_brown_sum9",
                "subset_brown_sum10","subset_brown_ext","csuros_sub"):
        out.append(_line_data(b[key], p, a, use_sub=True, mode=mode, L0=L0s[key]))
    return out


def plot_subset_min4_set(ds, tree269, profiles269, rng, ymax_cp=None, ymax_fm=None,
                         data=None, mode="observed"):
    if data is None:
        data = _set1_data(ds, tree269, profiles269, rng)
    chosen_name = data["chosen_name"]; F_mask = data["F_mask"]; path = data["path"]
    leaf_depth = len(path) - 1
    b = data["branches"]; arc2sub = data["arc2sub"]; L0s = data["L0s"]

    fig = plt.figure(figsize=(ps.WIDTH_2COL, 3.7))
    fig.suptitle(
        f"{SUBSET_LABEL[ds]}  ·  Set 1  ·  arc269 root to median-fm leaf "
        f"‘{chosen_name}’   (F = {F_mask:,})")
    ax_cp, ax_fm = setup_axes(fig, leaf_depth, mode=mode)

    # Sum-sweep (sum>=1..10) drawn with the SHARED SUM_SWEEP_STYLE, so this
    # per-clade subset-only plot is colour/marker-identical to the arc269
    # full-tree Set 1.5 plot threshold-for-threshold.  sum>=4 (Csurös'
    # canonical filter) is emphasized; sum>=1 (no filter) is the prominent
    # anchor; the intermediate thresholds are de-emphasized ribbons.
    lines = []
    for n in range(1, 11):
        key = f"subset_brown_sum{n}"
        if not b[key]:
            continue
        lines.append(dict(branches=b[key], L0=L0s[key], use_sub=True,
                           **SUM_SWEEP_STYLE[n]))
    lines += [
        dict(branches=b["subset_brown_ext"],  L0=L0s["subset_brown_ext"],
             color=C_EXTEND,  lw=1.6, linestyle='-.',
             marker='v', markersize=6, markerfacecolor=C_EXTEND,
             markeredgecolor=C_EXTEND, markeredgewidth=0.8, zorder=9, use_sub=True),
        dict(branches=b["csuros_sub"], L0=L0s["csuros_sub"],
             color=C_CSUROS,       lw=1.6, linestyle='--',
             marker='D', markersize=6, markerfacecolor='none',
             markeredgecolor=C_CSUROS, markeredgewidth=1.1, zorder=10, use_sub=True),
    ]
    plot_lines(ax_cp, ax_fm, lines, list(range(len(path))), arc2sub=arc2sub, path=path, mode=mode)
    add_observed_violin(ax_cp, ax_fm, data["cps"], data["fms"],
                        data["obs_cp"], data["obs_fm"], leaf_depth)
    finalize_axes(ax_cp, ax_fm, ymax_cp=ymax_cp, ymax_fm=ymax_fm)
    add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub)
    fig.tight_layout()
    suffix = "_l0corr" if mode == "l0corr" else ""
    ps.save_fig(fig, OUT / f"subset_min4_path_{ds}{suffix}")
    plt.close(fig)
    print(f"  SET1 {ds} {mode}: wrote subset_min4_path_{ds}{suffix}.png + .pdf")


# ===== SET 1.5: arc269 full-tree fits sweep across sum>=N filter thresholds =====
# Goes between Set 1 (subset-only) and Set 2 (min=1 no filter). Shows how the
# single arc269 Brownian fit's posterior on each subset's path SHIFTS as we
# tighten the sum>=N pre-filter — from sum>=1 (= canonical no-filter, full
# F=90,243) all the way to sum>=7 (most stringent). The arc269 mc=1 canonical
# fit is the sum>=1 case (every family with at least one copy somewhere).
ARC269_SUM_FITS_DIRS = {
    1: "brownian_arc269",                       # = canonical mc=1 no filter
    2: "brownian_arc269_omin4_sum2",
    3: "brownian_arc269_omin4_sum3",
    4: "brownian_arc269_omin4_sum4",
    5: "brownian_arc269_omin4_sum5",
    6: "brownian_arc269_omin4_sum6",
    7: "brownian_arc269_omin4_sum7",
    8: "brownian_arc269_omin4_sum8",
    9: "brownian_arc269_omin4_sum9",
    10: "brownian_arc269_omin4_sum10",
}


def _set15_data(ds, tree269, profiles269, rng):
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name, sub_leaf_idxs = median_fm_leaf(tree269, subtree, profiles269, rng)
    # Violin/star use FULL F=90,243 universe (matches the sum>=1 line).
    cps, fms, obs_cp, obs_fm = per_leaf_stats(profiles269, sub_leaf_idxs, chosen_li, mask=None)
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    paths = {n: OUT / f"{d}/arc269_map_sigma1.0.branches.csv"
             for n, d in ARC269_SUM_FITS_DIRS.items()}
    branches = {n: load_branches(p) for n, p in paths.items()}
    L0s      = {n: get_L0_for_fit(p) for n, p in paths.items()}
    return {"subtree": subtree, "chosen_li": chosen_li, "chosen_name": chosen_name,
            "path": path, "arc2sub": arc2sub, "F_full": int(profiles269.shape[0]),
            "cps": cps, "fms": fms, "obs_cp": obs_cp, "obs_fm": obs_fm,
            "branches": branches, "L0s": L0s}


def _set15_extract_line_extents(data, mode="observed"):
    b = data["branches"]; p = data["path"]; L0s = data["L0s"]
    return [_line_data(b[n], p, None, use_sub=False, mode=mode, L0=L0s[n])
            for n in sorted(b.keys()) if b[n]]


def plot_arc269_sumsweep_set(ds, tree269, profiles269, rng, ymax_cp=None, ymax_fm=None,
                              data=None, mode="observed"):
    if data is None:
        data = _set15_data(ds, tree269, profiles269, rng)
    chosen_name = data["chosen_name"]; F_full = data["F_full"]; path = data["path"]
    leaf_depth = len(path) - 1
    b = data["branches"]; L0s = data["L0s"]; arc2sub = data["arc2sub"]

    fig = plt.figure(figsize=(ps.WIDTH_2COL, 3.7))
    fig.suptitle(
        f"{SUBSET_LABEL[ds]}  ·  Set 1.5  ·  arc269 root to median-fm leaf "
        f"‘{chosen_name}’")
    ax_cp, ax_fm = setup_axes(fig, leaf_depth, mode=mode)

    lines = []
    for n in sorted(b.keys()):
        if not b[n]: continue
        lines.append(dict(branches=b[n], L0=L0s[n], **SUM_SWEEP_STYLE[n]))
    plot_lines(ax_cp, ax_fm, lines, list(range(len(path))), arc2sub=arc2sub, path=path, mode=mode)
    add_observed_violin(ax_cp, ax_fm, data["cps"], data["fms"],
                        data["obs_cp"], data["obs_fm"], leaf_depth)
    finalize_axes(ax_cp, ax_fm, ymax_cp=ymax_cp, ymax_fm=ymax_fm)
    add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub)
    fig.tight_layout()
    suffix = "_l0corr" if mode == "l0corr" else ""
    ps.save_fig(fig, OUT / f"arc269_sumsweep_path_{ds}{suffix}")
    plt.close(fig)
    print(f"  SET1.5 {ds} {mode}: wrote arc269_sumsweep_path_{ds}{suffix}.png + .pdf")


def write_arc269_sumsweep_legend():
    handles = _sweep_handles()
    handles.append(mlines.Line2D([], [], marker='*', color=ps.ORANGE, lw=0, markersize=12,
                                 markeredgecolor=ps.C_DATA, markeredgewidth=0.9,
                                 label='observed leaf cp/fm'))
    handles.append(mlines.Line2D([], [], marker=None, color=ps.C_VIOLIN_FACE, lw=7,
                                 label="violin — observed leaf cp/fm across "
                                       "the subclade's other leaves"))
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 1.55))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False,
               columnspacing=1.8)
    ps.save_fig(fig, OUT / "arc269_sumsweep_path_legend")
    plt.close(fig)
    print("  SET1.5 legend: wrote arc269_sumsweep_path_legend.png + .pdf")


def write_subset_min4_legend():
    handles = _sweep_handles() + [
        mlines.Line2D([], [], marker='v', color=C_EXTEND, lw=1.6, linestyle='-.',
                      markersize=6, markerfacecolor=C_EXTEND, markeredgewidth=0.8,
                      label='Brown-extend ML  (min = 4; warm-started from MAP Brownian)'),
        mlines.Line2D([], [], marker='D', color=C_CSUROS, lw=1.6, linestyle='--',
                      markersize=6, markerfacecolor='none', markeredgecolor=C_CSUROS,
                      markeredgewidth=1.1, label='Csurös subset-only ML  (min = 4 reference baseline)'),
        mlines.Line2D([], [], marker='*', color=ps.ORANGE, lw=0, markersize=12,
                      markeredgecolor=ps.C_DATA, markeredgewidth=0.9,
                      label='observed leaf cp/fm  (value labelled per panel)'),
        mlines.Line2D([], [], marker=None, color=ps.C_VIOLIN_FACE, lw=7,
                      label="violin — observed leaf cp/fm across the subclade's other leaves"),
    ]
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 1.85))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False,
               columnspacing=1.8)
    ps.save_fig(fig, OUT / "subset_min4_path_legend")
    plt.close(fig)
    print("  SET1 legend: wrote subset_min4_path_legend.png + .pdf")


# ===== SET 2: subset mc=1 (full complement) + arc269 mc=1 canonical, no Csurös =====
def _set2_data(ds, tree269, profiles269, rng):
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name, sub_leaf_idxs = median_fm_leaf(tree269, subtree, profiles269, rng)
    cps, fms, obs_cp, obs_fm = per_leaf_stats(profiles269, sub_leaf_idxs, chosen_li, mask=None)
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    paths = {
        "subset_brown_full":     OUT / f"brownian_{ds}_full/{ds}_map_sigma1.0.branches.csv",
        "subset_brown_ext_full": OUT / f"brownian_extend_{ds}_full/{ds}_ml.branches.csv",
        "arc269_brown":          OUT / "brownian_arc269/arc269_map_sigma1.0.branches.csv",
    }
    branches = {k: load_branches(p) for k, p in paths.items()}
    L0s      = {k: get_L0_for_fit(p) for k, p in paths.items()}
    return {"subtree": subtree, "chosen_li": chosen_li, "chosen_name": chosen_name,
            "path": path, "arc2sub": arc2sub, "F_full": int(profiles269.shape[0]),
            "cps": cps, "fms": fms, "obs_cp": obs_cp, "obs_fm": obs_fm,
            "branches": branches, "L0s": L0s}


def _set2_extract_line_extents(data, mode="observed"):
    b = data["branches"]; p = data["path"]; a = data["arc2sub"]; L0s = data["L0s"]
    return [
        _line_data(b["arc269_brown"],          p, a, use_sub=False, mode=mode, L0=L0s["arc269_brown"]),
        _line_data(b["subset_brown_full"],     p, a, use_sub=True,  mode=mode, L0=L0s["subset_brown_full"]),
        _line_data(b["subset_brown_ext_full"], p, a, use_sub=True,  mode=mode, L0=L0s["subset_brown_ext_full"]),
    ]


def plot_min1_set(ds, tree269, profiles269, rng, ymax_cp=None, ymax_fm=None,
                  data=None, mode="observed"):
    if data is None:
        data = _set2_data(ds, tree269, profiles269, rng)
    chosen_name = data["chosen_name"]; F_full = data["F_full"]; path = data["path"]
    leaf_depth = len(path) - 1
    b = data["branches"]; arc2sub = data["arc2sub"]; L0s = data["L0s"]

    fig = plt.figure(figsize=(ps.WIDTH_2COL, 3.7))
    fig.suptitle(
        f"{SUBSET_LABEL[ds]}  ·  Set 2  ·  arc269 root to median-fm leaf "
        f"‘{chosen_name}’")
    ax_cp, ax_fm = setup_axes(fig, leaf_depth, mode=mode)

    lines = [
        dict(branches=b["arc269_brown"], L0=L0s["arc269_brown"],
             color=C_ARC_M1, lw=1.8, marker='o', markersize=6),
        dict(branches=b["subset_brown_full"], L0=L0s["subset_brown_full"],
             color=C_BROWN, lw=1.6, linestyle='-',
             marker='s', markersize=6, markerfacecolor=C_BROWN,
             markeredgecolor=C_BROWN, markeredgewidth=0.8, zorder=9, use_sub=True),
        dict(branches=b["subset_brown_ext_full"], L0=L0s["subset_brown_ext_full"],
             color=C_EXTEND, lw=1.6, linestyle='-.',
             marker='v', markersize=6, markerfacecolor=C_EXTEND,
             markeredgecolor=C_EXTEND, markeredgewidth=0.8, zorder=9, use_sub=True),
    ]
    plot_lines(ax_cp, ax_fm, lines, list(range(len(path))), arc2sub=arc2sub, path=path, mode=mode)
    add_observed_violin(ax_cp, ax_fm, data["cps"], data["fms"],
                        data["obs_cp"], data["obs_fm"], leaf_depth)
    finalize_axes(ax_cp, ax_fm, ymax_cp=ymax_cp, ymax_fm=ymax_fm)
    add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub)
    fig.tight_layout()
    suffix = "_l0corr" if mode == "l0corr" else ""
    ps.save_fig(fig, OUT / f"min1_path_{ds}{suffix}")
    plt.close(fig)
    print(f"  SET2 {ds} {mode}: wrote min1_path_{ds}{suffix}.png + .pdf")


def write_min1_legend():
    handles = [
        mlines.Line2D([], [], marker='o', color=C_ARC_M1, lw=1.8, markersize=6,
                      label='arc269 MAP Brownian  (canonical full-tree, F = 90,243, mc = 1)'),
        mlines.Line2D([], [], marker='s', color=C_BROWN, lw=1.6, markersize=6,
                      markerfacecolor=C_BROWN, markeredgewidth=0.8,
                      label='subset MAP Brownian over FULL subset complement  (mc = 1, no filter)'),
        mlines.Line2D([], [], marker='v', color=C_EXTEND, lw=1.6, linestyle='-.',
                      markersize=6, markerfacecolor=C_EXTEND, markeredgewidth=0.8,
                      label='subset Brown-extend ML over FULL subset complement  (mc = 1, prior dropped)'),
        mlines.Line2D([], [], marker='*', color=ps.ORANGE, lw=0, markersize=12,
                      markeredgecolor=ps.C_DATA, markeredgewidth=0.9,
                      label='observed leaf cp/fm'),
        mlines.Line2D([], [], marker=None, color=ps.C_VIOLIN_FACE, lw=7,
                      label="violin — observed leaf cp/fm across the subclade's other leaves"),
    ]
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 1.35))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False,
               columnspacing=1.8)
    ps.save_fig(fig, OUT / "min1_path_legend")
    plt.close(fig)
    print("  SET2 legend: wrote min1_path_legend.png + .pdf")


# ===== SET 3: Csurös subset min=4 ref + arc269 union-of-subsets-min=4 =====
def _set3_data(ds, tree269, profiles269, rng, union_mask):
    subtree, _, _, _, _ = load_dataset(ds)
    chosen_li, chosen_name, sub_leaf_idxs = median_fm_leaf(tree269, subtree, profiles269, rng)
    cps, fms, obs_cp, obs_fm = per_leaf_stats(profiles269, sub_leaf_idxs, chosen_li, union_mask)
    F_union = int(union_mask.sum())
    path = arc269_path(tree269, chosen_li)
    arc2sub = {a: s for s, a in map_subset_to_arc269(subtree, tree269).items()}
    paths = {
        "csuros_sub":       OUT / f"reproduction/{ds}_reproduce.branches.csv",
        "subset_brown_ext": OUT / f"brownian_extend_{ds}/{ds}_ml.branches.csv",
        "subset_brown_min4": OUT / f"brownian_{ds}/{ds}_map_sigma1.0.branches.csv",
        "arc_union":        OUT / "brownian_arc269_union_subsets_min4/arc269_map_sigma1.0.branches.csv",
    }
    branches = {k: load_branches(p) for k, p in paths.items()}
    L0s      = {k: get_L0_for_fit(p) for k, p in paths.items()}
    return {"subtree": subtree, "chosen_li": chosen_li, "chosen_name": chosen_name,
            "path": path, "arc2sub": arc2sub, "F_union": F_union,
            "cps": cps, "fms": fms, "obs_cp": obs_cp, "obs_fm": obs_fm,
            "branches": branches, "L0s": L0s}


def _set3_extract_line_extents(data, mode="observed"):
    b = data["branches"]; p = data["path"]; a = data["arc2sub"]; L0s = data["L0s"]
    return [
        _line_data(b["arc_union"],         p, a, use_sub=False, mode=mode, L0=L0s["arc_union"]),
        _line_data(b["subset_brown_min4"], p, a, use_sub=True,  mode=mode, L0=L0s["subset_brown_min4"]),
        _line_data(b["subset_brown_ext"],  p, a, use_sub=True,  mode=mode, L0=L0s["subset_brown_ext"]),
        _line_data(b["csuros_sub"],        p, a, use_sub=True,  mode=mode, L0=L0s["csuros_sub"]),
    ]


def plot_union_set(ds, tree269, profiles269, rng, union_mask=None,
                   ymax_cp=None, ymax_fm=None, data=None, mode="observed"):
    if data is None:
        if union_mask is None:
            union_mask = union_min4_mask(profiles269, tree269)
        data = _set3_data(ds, tree269, profiles269, rng, union_mask)
    chosen_name = data["chosen_name"]; F_union = data["F_union"]; path = data["path"]
    leaf_depth = len(path) - 1
    b = data["branches"]; arc2sub = data["arc2sub"]; L0s = data["L0s"]

    fig = plt.figure(figsize=(ps.WIDTH_2COL, 3.7))
    fig.suptitle(
        f"{SUBSET_LABEL[ds]}  ·  Set 3  ·  arc269 root to median-fm leaf "
        f"‘{chosen_name}’   (F = {F_union:,})")
    ax_cp, ax_fm = setup_axes(fig, leaf_depth, mode=mode)

    lines = [
        dict(branches=b["arc_union"], L0=L0s["arc_union"],
             color=C_ARC_UNI, lw=1.8, marker='o', markersize=6,
             markerfacecolor=C_ARC_UNI, markeredgecolor=C_ARC_UNI, markeredgewidth=0.8),
        dict(branches=b["subset_brown_min4"], L0=L0s["subset_brown_min4"],
             color=C_BROWN, lw=1.6, linestyle='-',
             marker='s', markersize=6, markerfacecolor=C_BROWN,
             markeredgecolor=C_BROWN, markeredgewidth=0.8, zorder=9, use_sub=True),
        dict(branches=b["subset_brown_ext"], L0=L0s["subset_brown_ext"],
             color=C_EXTEND, lw=1.6, linestyle='-.',
             marker='v', markersize=6, markerfacecolor=C_EXTEND,
             markeredgecolor=C_EXTEND, markeredgewidth=0.8, zorder=9, use_sub=True),
        dict(branches=b["csuros_sub"], L0=L0s["csuros_sub"],
             color=C_CSUROS, lw=1.6, linestyle='--',
             marker='D', markersize=6, markerfacecolor='none',
             markeredgecolor=C_CSUROS, markeredgewidth=1.1, zorder=10, use_sub=True),
    ]
    plot_lines(ax_cp, ax_fm, lines, list(range(len(path))), arc2sub=arc2sub, path=path, mode=mode)
    add_observed_violin(ax_cp, ax_fm, data["cps"], data["fms"],
                        data["obs_cp"], data["obs_fm"], leaf_depth)
    finalize_axes(ax_cp, ax_fm, ymax_cp=ymax_cp, ymax_fm=ymax_fm)
    add_subclade_ancestor_vline(ax_cp, ax_fm, path, arc2sub)
    fig.tight_layout()
    suffix = "_l0corr" if mode == "l0corr" else ""
    ps.save_fig(fig, OUT / f"union_path_{ds}{suffix}")
    plt.close(fig)
    print(f"  SET3 {ds} {mode}: wrote union_path_{ds}{suffix}.png + .pdf")


def write_union_legend():
    handles = [
        mlines.Line2D([], [], marker='o', color=C_ARC_UNI, lw=1.8, markersize=6,
                      label='arc269 MAP Brownian on UNION of per-subset min = 4 families'),
        mlines.Line2D([], [], marker='s', color=C_BROWN, lw=1.6, linestyle='-',
                      markersize=6, markerfacecolor=C_BROWN, markeredgewidth=0.8,
                      label='subset MAP Brownian  (per-clade min = 4)'),
        mlines.Line2D([], [], marker='v', color=C_EXTEND, lw=1.6, linestyle='-.',
                      markersize=6, markerfacecolor=C_EXTEND, markeredgewidth=0.8,
                      label='subset Brown-extend ML  (per-clade min = 4, prior dropped after warm-start)'),
        mlines.Line2D([], [], marker='D', color=C_CSUROS, lw=1.6, linestyle='--',
                      markersize=6, markerfacecolor='none', markeredgecolor=C_CSUROS,
                      markeredgewidth=1.1, label='Csurös subset-only ML  (per-clade min = 4 reference)'),
        mlines.Line2D([], [], marker='*', color=ps.ORANGE, lw=0, markersize=12,
                      markeredgecolor=ps.C_DATA, markeredgewidth=0.9,
                      label='observed leaf cp/fm'),
        mlines.Line2D([], [], marker=None, color=ps.C_VIOLIN_FACE, lw=7,
                      label="violin — observed leaf cp/fm across the subclade's other leaves"),
    ]
    fig = plt.figure(figsize=(ps.WIDTH_2COL, 1.5))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False,
               columnspacing=1.8)
    ps.save_fig(fig, OUT / "union_path_legend")
    plt.close(fig)
    print("  SET3 legend: wrote union_path_legend.png + .pdf")


def main():
    print("Loading arc269...")
    tree269, profiles269, _, _, _ = load_dataset("arc269")
    print("Computing union-of-per-subset-min=4 mask (used by Set 3)...")
    umask = union_min4_mask(profiles269, tree269)
    print(f"  union F={int(umask.sum()):,}")

    # PASS 1 — gather all line + violin data once per (set, ds).
    set1_data, set15_data, set2_data, set3_data = {}, {}, {}, {}
    rng = np.random.default_rng(2026)
    for ds in DATASETS:
        set1_data[ds]  = _set1_data(ds, tree269, profiles269, rng)
        set15_data[ds] = _set15_data(ds, tree269, profiles269, rng)
        set2_data[ds]  = _set2_data(ds, tree269, profiles269, rng)
        set3_data[ds]  = _set3_data(ds, tree269, profiles269, rng, umask)

    def _extents(data_by_ds, extract_fn, mode):
        lines = {ds: extract_fn(d, mode=mode) for ds, d in data_by_ds.items()}
        viol  = {ds: (d["cps"], d["fms"], d["obs_cp"], d["obs_fm"])
                 for ds, d in data_by_ds.items()}
        return _set_extents(lines, viol)

    # PASS 2 — render each plot in BOTH modes (observed + l0corr), with the
    # mode-specific shared y-cap across the 4 subsets per set.
    for mode in ("observed", "l0corr"):
        cap_cp1,  cap_fm1  = _extents(set1_data,  _set1_extract_line_extents,  mode)
        cap_cp15, cap_fm15 = _extents(set15_data, _set15_extract_line_extents, mode)
        cap_cp2,  cap_fm2  = _extents(set2_data,  _set2_extract_line_extents,  mode)
        cap_cp3,  cap_fm3  = _extents(set3_data,  _set3_extract_line_extents,  mode)
        print(f"\n== mode={mode} == caps  Set1: {cap_cp1:.0f}/{cap_fm1:.0f}  "
              f"Set1.5: {cap_cp15:.0f}/{cap_fm15:.0f}  "
              f"Set2: {cap_cp2:.0f}/{cap_fm2:.0f}  "
              f"Set3: {cap_cp3:.0f}/{cap_fm3:.0f}")
        for ds in DATASETS:
            plot_subset_min4_set(ds, tree269, profiles269, None, mode=mode,
                                 ymax_cp=cap_cp1, ymax_fm=cap_fm1, data=set1_data[ds])
            plot_arc269_sumsweep_set(ds, tree269, profiles269, None, mode=mode,
                                      ymax_cp=cap_cp15, ymax_fm=cap_fm15, data=set15_data[ds])
            plot_min1_set(ds, tree269, profiles269, None, mode=mode,
                          ymax_cp=cap_cp2, ymax_fm=cap_fm2, data=set2_data[ds])
            plot_union_set(ds, tree269, profiles269, None, union_mask=umask, mode=mode,
                           ymax_cp=cap_cp3, ymax_fm=cap_fm3, data=set3_data[ds])

    print("\n--- legends ---")
    write_subset_min4_legend()
    write_arc269_sumsweep_legend()
    write_min1_legend()
    write_union_legend()

    print("\nDONE.")


if __name__ == "__main__":
    main()
