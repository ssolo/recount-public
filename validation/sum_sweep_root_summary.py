"""Two summary plots over the sum-filter sweep: root copies and root families
at each MAP Brownian sigma=1 fit, per dataset (arc269 + 4 focal subsets).

Reads validation/outputs/brownian_<ds>{,_full,_sumN,_omin4_sumN}/<ds>_sigma1.0_summary.json
for each (dataset, N) and produces:

  validation/outputs/sum_sweep_root_copies.png    -- root copies vs N
  validation/outputs/sum_sweep_root_families.png  -- root families vs N

Both raw (observed posterior) and L(0)-amplified curves overlaid.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from validation import _pubstyle as ps

OUT = Path("/Users/ssolo/src/recount/validation/outputs")

DATASETS = [
    ("dpann80",  "DPANN (D80)",             ps.BLUE),
    ("proteo75", "Proteoarchaea (P75)",     ps.ORANGE),
    ("eury114",  "Methanobacteriati (E114)", ps.GREEN),
    ("ed194",    "Euryarchaeota (ED194)",   ps.VERMILION),
    ("arc269",   "arc269 full-tree (N=269)", ps.C_DATA),
]

# Range of sum thresholds plotted
N_VALUES = list(range(1, 11))


def fit_dir(dataset, N):
    if dataset == "arc269":
        if N == 1: return OUT / "brownian_arc269"
        return OUT / f"brownian_arc269_omin4_sum{N}"
    else:
        if N == 1: return OUT / f"brownian_{dataset}_full"
        if N == 4: return OUT / f"brownian_{dataset}"
        return OUT / f"brownian_{dataset}_sum{N}"


def load_summary(dataset, N):
    """Return (F, L0, root_cp, root_fm, cpf) or None if missing."""
    d = fit_dir(dataset, N)
    # canonical filename
    candidates = [d / f"{dataset}_sigma1.0_summary.json",
                  d / f"{dataset}_summary.json"]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                j = json.load(f)
            final = j.get("final", {})
            return dict(
                F=j.get("F_kept") or j.get("F"),
                L0=final.get("L0"),
                cp=final.get("root_copies_corr"),
                fm=final.get("root_families_corr"),
                cpf=final.get("copies_per_family"),
            )
    return None


def collect():
    """Return dict[dataset] = list of (N, summary_or_None)."""
    out = {ds: [] for ds, _, _ in DATASETS}
    for ds, _, _ in DATASETS:
        for N in N_VALUES:
            out[ds].append((N, load_summary(ds, N)))
    return out


def plot_metric(data, metric_key, ylabel, l0_amp=False, outfile=None):
    """metric_key: 'cp' or 'fm'. l0_amp: if True, multiply by 1/(1-L0).
    Log-y; legend outside on the right; value labels only at the two
    endpoints (N=1 and the rightmost present N)."""
    fig, ax = plt.subplots(figsize=(ps.WIDTH_2COL, 3.6))
    for di, (ds, label, color) in enumerate(DATASETS):
        Ns, ys = [], []
        for N, s in data[ds]:
            if s is None or s.get(metric_key) is None: continue
            y = s[metric_key]
            if l0_amp and s.get("L0") is not None:
                y = y / (1.0 - s["L0"])
            Ns.append(N); ys.append(y)
        if not Ns: continue
        is_arc = ds == "arc269"
        ax.plot(Ns, ys,
                color=color,
                marker='s' if is_arc else 'o',
                markersize=5.0 if is_arc else 4.0,
                linewidth=2.0 if is_arc else 1.5,
                alpha=1.0 if is_arc else 0.9,
                label=label,
                zorder=10 if is_arc else 5,
                markeredgecolor='white',
                markeredgewidth=0.5)
        # value labels only at the two endpoints — placed just outside
        # the data (left endpoint to the left, right endpoint to the
        # right), with a small per-series vertical stagger so the five
        # curves' near-coincident labels do not collide.
        dy = (di - (len(DATASETS) - 1) / 2.0) * 4.5
        for i in (0, len(Ns) - 1):
            N_i, y_i = Ns[i], ys[i]
            is_left = i == 0
            ax.annotate(f"{y_i:,.0f}", (N_i, y_i),
                        textcoords="offset points",
                        xytext=(-8 if is_left else 8, dy),
                        ha='right' if is_left else 'left', va='center',
                        fontsize=6.6, color=color, fontweight='bold')
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([f"≥{n}" for n in N_VALUES])
    ax.set_xlabel("sum-filter threshold N  (family kept iff Σ leaf copies ≥ N)")
    ax.set_ylabel(ylabel + "  (log scale)")
    ax.set_yscale("log")
    ps.grid(ax)
    ax.grid(True, axis="both", which="minor", lw=0.3, alpha=0.12,
            linestyle=":", zorder=0)
    ax.set_xlim(0.0, 11.0)
    # Legend on the right outside the plot — no overlap with data
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              title="dataset", title_fontsize=7.5)
    if outfile:
        ps.save_fig(fig, outfile.with_suffix(""))
        print(f"  wrote {outfile} + .pdf")
    plt.close(fig)


def write_table(data, outfile):
    """Markdown table: rows = N, cols = (dataset cp, fm) plus L0/cpf."""
    lines = []
    lines.append("Per-dataset sum-filter sweep — root posterior counts under MAP Brownian σ=1")
    lines.append("")
    lines.append("`raw / L(0)-amp` for cp and fm at every (dataset, N). cpf shown raw.\n")
    header = "| N | F | L(0) |" + " | ".join(f"{lbl} cp" for _, lbl, _ in DATASETS for kind in ("cp",)) + " |"
    # Actually let me write per-dataset tables instead of one wide one — too many cols
    for ds, label, _ in DATASETS:
        lines.append(f"\n### {label}\n")
        lines.append("| N | F | L(0) | root cp (raw / amp) | root fm (raw / amp) | cpf |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for N, s in data[ds]:
            if s is None:
                lines.append(f"| {N} | — | — | — | — | — |")
                continue
            amp = 1.0 / (1.0 - s["L0"]) if s["L0"] is not None else 1.0
            cp_raw, cp_amp = s["cp"], s["cp"] * amp
            fm_raw, fm_amp = s["fm"], s["fm"] * amp
            lines.append(f"| {N} | {s['F']:,} | {s['L0']:.4f} | "
                         f"{cp_raw:,.0f} / {cp_amp:,.0f} | "
                         f"{fm_raw:,.0f} / {fm_amp:,.0f} | "
                         f"{s['cpf']:.2f} |")
    out = "\n".join(lines) + "\n"
    Path(outfile).write_text(out)
    print(f"  wrote {outfile}")


def main():
    ps.apply()
    data = collect()
    n_missing = sum(1 for ds, _, _ in DATASETS
                    for N, s in data[ds] if s is None)
    n_total = len(DATASETS) * len(N_VALUES)
    print(f"summaries loaded: {n_total - n_missing}/{n_total}  (missing: {n_missing})")
    plot_metric(data, "cp",
                ylabel="root posterior copies (raw)",
                l0_amp=False,
                outfile=OUT / "sum_sweep_root_copies.png")
    plot_metric(data, "fm",
                ylabel="root posterior families (raw)",
                l0_amp=False,
                outfile=OUT / "sum_sweep_root_families.png")
    plot_metric(data, "cp",
                ylabel="root posterior copies (L(0)-amplified)",
                l0_amp=True,
                outfile=OUT / "sum_sweep_root_copies_l0amp.png")
    plot_metric(data, "fm",
                ylabel="root posterior families (L(0)-amplified)",
                l0_amp=True,
                outfile=OUT / "sum_sweep_root_families_l0amp.png")
    write_table(data, OUT / "sum_sweep_root_summary.md")


if __name__ == "__main__":
    main()
