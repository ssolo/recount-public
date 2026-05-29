"""Monte Carlo validation for PNAS Letter Figure 1A.

The figure's family-size prediction is computed from adjacent
unobserved-profile probabilities:

    P(Omega = k) = L0(k + 1) - L0(k),

where ``L0(m) = P{Omega < m}`` is evaluated by
``unobserved_logL0_native``.  This script independently forward-simulates
families under the fitted GLD rates and checks that the simulated
family-size histogram agrees with those L0-implied probabilities.

It validates the quantity plotted in ``validation/pnas_letter_v2_figure.py``:
the predicted histogram after conditioning on Omega >= 2.

Run:
    python3 validation/pnas_l0_family_size_sim.py --families 250000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recount.native_backend import unobserved_logL0_native
from recount.rates import GLDRates
from validation import _pubstyle as ps
from validation._shared import load_dataset
from validation.simulator import _per_edge_params, _preorder, _sample_gain


OUT = ROOT / "validation" / "outputs"
RESULTS = ROOT / "validation" / "results"

FITS = [
    ("omega_min_1", "brownian_arc269"),
    ("omega_min_4", "brownian_arc269_omin4_sum4"),
]


def _load_rates(tree, dirname: str) -> GLDRates:
    z = np.load(OUT / dirname / "arc269_sigma1.0_final_rates.npz")
    return GLDRates(
        tree=tree,
        gain=z["gain"].copy(),
        loss=z["loss"].copy(),
        dup=z["dup"].copy(),
        length=z["length"].copy(),
    )


def _simulate_sizes(tree, rates: GLDRates, families: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_surv, q_raw, is_polya, gain = _per_edge_params(tree, rates)
    preorder = _preorder(tree)
    counts = np.zeros(tree.num_nodes, dtype=np.int64)
    sizes = np.zeros(families, dtype=np.int64)

    for f in range(families):
        counts[tree.root] = _sample_gain(
            rng, gain[tree.root], q_raw[tree.root],
            bool(is_polya[tree.root]), 0,
        )
        for v in preorder:
            if v == tree.root:
                continue
            parent = int(tree.parent[v])
            parent_count = int(counts[parent])
            if parent_count > 0 and p_surv[v] > 0.0:
                survivors = int(rng.binomial(parent_count, min(float(p_surv[v]), 1.0)))
            else:
                survivors = 0
            gains = _sample_gain(
                rng, gain[v], q_raw[v], bool(is_polya[v]), survivors,
            )
            counts[v] = survivors + gains
        sizes[f] = int(counts[:tree.num_leaves].sum())
    return sizes


def _l0_probabilities(tree, rates: GLDRates, max_k: int) -> np.ndarray:
    l0 = np.array([
        math.exp(unobserved_logL0_native(
            tree, rates.gain, rates.loss, rates.dup, rates.length, min_copies=m,
        ))
        for m in range(1, max_k + 2)
    ])
    return np.diff(l0)  # index k-1 = P(Omega = k)


def _binomial_z(obs: int, n: int, p: float) -> float:
    var = n * p * (1.0 - p)
    if var <= 0.0:
        return 0.0 if obs == n * p else float("inf")
    return (obs - n * p) / math.sqrt(var)


def _summarize_fit(label: str, dirname: str, tree, families: int,
                   max_k: int, seed: int) -> dict:
    rates = _load_rates(tree, dirname)
    print(f"# {label}: simulating {families:,} raw families from {dirname}", flush=True)
    t0 = time.time()
    sizes = _simulate_sizes(tree, rates, families, seed)
    wall = time.time() - t0

    p = _l0_probabilities(tree, rates, max_k)
    p0 = math.exp(unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, min_copies=1,
    ))
    p0_or_1 = math.exp(unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, min_copies=2,
    ))
    p_ge2 = 1.0 - p0_or_1

    hist = np.bincount(sizes, minlength=max_k + 2)
    raw_checks = []
    raw_bins = [
        ("Omega=0", int((sizes == 0).sum()), p0),
        ("Omega=1", int((sizes == 1).sum()), p0_or_1 - p0),
        ("Omega>=2", int((sizes >= 2).sum()), p_ge2),
    ]
    for name, obs, prob in raw_bins:
        raw_checks.append({
            "bin": name,
            "simulated": obs,
            "expected": families * prob,
            "probability": prob,
            "z": _binomial_z(obs, families, prob),
            "ratio": obs / (families * prob) if prob > 0 else None,
        })

    # Figure 1A condition: compare P(Omega=k | Omega>=2), k=2..7 and >=8.
    cond_n = int((sizes >= 2).sum())
    cond_checks = []
    for k in range(2, 8):
        prob = p[k - 1] / p_ge2
        obs = int(hist[k])
        exp = cond_n * prob
        cond_checks.append({
            "bin": f"Omega={k}",
            "simulated": obs,
            "expected": exp,
            "probability_cond_ge2": prob,
            "z": _binomial_z(obs, cond_n, prob),
            "ratio": obs / exp if prob > 0 else None,
        })
    prob_tail = (1.0 - math.exp(unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, min_copies=8,
    ))) / p_ge2
    obs_tail = int((sizes >= 8).sum())
    exp_tail = cond_n * prob_tail
    cond_checks.append({
        "bin": "Omega>=8",
        "simulated": obs_tail,
        "expected": exp_tail,
        "probability_cond_ge2": prob_tail,
        "z": _binomial_z(obs_tail, cond_n, prob_tail),
        "ratio": obs_tail / exp_tail if prob_tail > 0 else None,
    })

    max_abs_z = max(abs(row["z"]) for row in raw_checks + cond_checks)
    print(f"# {label}: wall={wall:.1f}s, max |z|={max_abs_z:.2f}", flush=True)
    for row in raw_checks:
        print(
            f"  {row['bin']:<9} sim={row['simulated']:>8,} "
            f"exp={row['expected']:>10.1f} ratio={row['ratio']:.4f} "
            f"z={row['z']:+.2f}",
            flush=True,
        )
    print("  conditional on Omega>=2:", flush=True)
    for row in cond_checks:
        print(
            f"    {row['bin']:<8} sim={row['simulated']:>6,} "
            f"exp={row['expected']:>8.1f} ratio={row['ratio']:.4f} "
            f"z={row['z']:+.2f}",
            flush=True,
        )

    return {
        "label": label,
        "rates_dir": dirname,
        "families": families,
        "seed": seed,
        "wall_s": wall,
        "max_abs_z": max_abs_z,
        "raw_checks": raw_checks,
        "conditional_ge2_checks": cond_checks,
        "histogram": {
            "k": list(range(1, max_k + 1)),
            "raw_simulated": hist[1:max_k + 1].astype(int).tolist(),
            "raw_expected": (families * p).tolist(),
            "conditional_ge2_k": list(range(2, max_k + 1)),
            "conditional_ge2_simulated": hist[2:max_k + 1].astype(int).tolist(),
            "conditional_ge2_expected": (cond_n * p[1:] / p_ge2).tolist(),
            "raw_tail_ge_max_plus_1": int((sizes >= max_k + 1).sum()),
            "conditional_ge2_tail_ge_max_plus_1": int((sizes >= max_k + 1).sum()),
            "conditional_ge2_sample_size": cond_n,
        },
    }


def _plot_ratio(results: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        row["bin"].replace("Omega", r"$\Omega$")
        for row in results["fits"][0]["conditional_ge2_checks"]
    ]
    x = np.arange(len(labels))
    colors = {"omega_min_1": ps.C_OMIN1, "omega_min_4": ps.C_OMIN4}
    titles = {
        "omega_min_1": r"unfiltered fit ($\Omega_{\min}=1$)",
        "omega_min_4": r"filtered fit ($\Omega_{\min}=4$)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_2COL, 3.1), sharey=True)
    for ax, fit, letter in zip(axes, results["fits"], "AB"):
        rows = fit["conditional_ge2_checks"]
        ratio = np.array([r["ratio"] for r in rows], dtype=float)
        expected = np.array([r["expected"] for r in rows], dtype=float)
        simulated = np.array([r["simulated"] for r in rows], dtype=float)
        cond_n = simulated.sum()
        prob = expected / cond_n
        se_ratio = np.sqrt(cond_n * prob * (1.0 - prob)) / expected
        color = colors[fit["label"]]
        ps.grid(ax, axis="y")
        ax.axhline(1.0, color=ps.C_NEUTRAL, lw=0.9, zorder=1)
        ax.fill_between([-0.5, len(labels) - 0.5], 0.95, 1.05,
                        color=ps.C_SHADE, lw=0, zorder=0)
        ax.errorbar(x, ratio, yerr=se_ratio, fmt="o-", color=color,
                    ecolor=color, elinewidth=1.0, capsize=2.5, markersize=4,
                    lw=1.4, zorder=3)
        for xi, yi, zi in zip(x, ratio, [r["z"] for r in rows]):
            ax.text(xi, yi + (0.055 if yi >= 1 else -0.065), f"z={zi:+.1f}",
                    ha="center", va="center", fontsize=6, color=ps.GREY)
        ax.set_title(
            f"{titles[fit['label']]}\n"
            f"ratio range {ratio.min():.3f}-{ratio.max():.3f}", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0.88, 1.12)
        ax.set_xlabel(r"family size bin, conditioned on $\Omega \geq 2$")
        ps.panel_label(ax, letter, dy=1.18)
    axes[0].set_ylabel(r"simulated / $L(0)$-predicted count")
    fig.tight_layout()
    ps.save_fig(fig, out.with_suffix(""))
    plt.close(fig)


def _plot_loglog(results: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"omega_min_1": ps.C_OMIN1, "omega_min_4": ps.C_OMIN4}
    titles = {
        "omega_min_1": r"unfiltered fit ($\Omega_{\min}=1$)",
        "omega_min_4": r"filtered fit ($\Omega_{\min}=4$)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_2COL, 3.3), sharey=False)
    for ax, fit, letter in zip(axes, results["fits"], "AB"):
        h = fit["histogram"]
        k = np.array(h["conditional_ge2_k"], dtype=float)
        simulated = np.array(h["conditional_ge2_simulated"], dtype=float)
        expected = np.array(h["conditional_ge2_expected"], dtype=float)
        color = colors[fit["label"]]
        m = expected > 0
        ax.plot(k[m], expected[m], "-", color=color, lw=1.9,
                label=r"$L(0)$ predicted")
        ms = simulated > 0
        ax.plot(k[ms], simulated[ms], "o", color=ps.C_DATA, ms=3.0,
                markeredgecolor="white", markeredgewidth=0.35,
                label="forward simulation")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.8, results["max_k"] * 1.08)
        ymax = max(float(simulated.max()), float(expected.max()))
        ymin = max(0.7, min(float(expected[m].min()), float(simulated[ms].min())))
        ax.set_ylim(ymin * 0.65, ymax * 1.75)
        ax.grid(True, which="major", lw=0.4, alpha=0.30, linestyle="--")
        ax.grid(True, which="minor", lw=0.3, alpha=0.12, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_title(
            f"{titles[fit['label']]}\n"
            f"{h['conditional_ge2_sample_size']:,} simulated families with "
            r"$\Omega \geq 2$", pad=6)
        ax.set_xlabel(r"family size $\Omega$  (conditioned on $\Omega \geq 2$)")
        ax.legend(loc="upper right")
        ps.panel_label(ax, letter, dy=1.18)
    axes[0].set_ylabel("family count")
    fig.tight_layout()
    ps.save_fig(fig, out.with_suffix(""))
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", type=int, default=250_000,
                    help="raw families to simulate per fit")
    ap.add_argument("--max-k", type=int, default=60,
                    help="largest exact family-size probability to compute")
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--out", type=Path,
                    default=RESULTS / "pnas_l0_family_size_sim.json")
    ap.add_argument("--ratio-plot", type=Path,
                    default=RESULTS / "pnas_l0_family_size_sim.png")
    ap.add_argument("--loglog-plot", type=Path,
                    default=RESULTS / "pnas_l0_family_size_sim_loglog.png")
    args = ap.parse_args(argv)

    tree, *_ = load_dataset("arc269")
    results = {
        "dataset": "arc269",
        "families_per_fit": args.families,
        "seed": args.seed,
        "max_k": args.max_k,
        "fits": [],
    }
    for i, (label, dirname) in enumerate(FITS):
        results["fits"].append(_summarize_fit(
            label, dirname, tree, args.families, args.max_k, args.seed + i,
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"# wrote {args.out}")
    ps.apply()                          # install shared rcParams — once
    _plot_ratio(results, args.ratio_plot)
    print(f"# wrote {args.ratio_plot}")
    _plot_loglog(results, args.loglog_plot)
    print(f"# wrote {args.loglog_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
