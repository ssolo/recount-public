"""Parametric-bootstrap forward simulation on arc269 with composite rates.

Pipeline:

  1. Build composite arc269 rates by stitching together the 4
     brownian-extended subtree fits (dpann80, proteo75, eury114, ed194)
     onto the corresponding arc269 nodes via
     ``subclade_ancestor_comparison.map_subset_to_arc269``. For each
     arc269 node v, the rates come from the SMALLEST subset whose leaf
     set contains v's descendants — i.e. the most specific subtree fit
     wins (dpann80 / eury114 beat ed194 inside their clades; proteo75
     wins for the TACK clade; ed194 fills the connecting Euryarchaeota
     internal branches).
  2. The 2 "glue" nodes (arc269 root LACA + the inter-clade ancestor
     connecting ed194 to proteo75) get rates drawn from a log-Normal
     prior whose mean and sd are estimated empirically from the log-
     rates of all non-root subset nodes (per axis: γ, λ, τ).
  3. Apply the composite rates to the real arc269 profile table → root
     posterior families/copies (``reconstruct``).
  4. Forward-simulate F=90,243 families with the composite rates,
     conditioning on Ωmin=1 (matching the arc269 dataset convention).
  5. Compute summary stats on the simulated profiles and compare to
     real arc269 stats.

The verification pass first runs the simulator on each subtree alone
(see validation/simulator.py) to confirm the kernel is correct; this
script assumes that's already been done.

Usage:
  PYTHONPATH=. python3 validation/simulate_arc269_composite.py
  PYTHONPATH=. python3 validation/simulate_arc269_composite.py --seed 2026 --F 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from recount.rates import GLDRates
from recount.tree import Tree
from validation._shared import load_dataset, reconstruct
from validation.simulator import simulate_gld
from validation.subclade_ancestor_comparison import map_subset_to_arc269


SUBSETS = ["dpann80", "proteo75", "eury114", "ed194"]


# ----------------------------------------------------------------------------
# Composite rate construction
# ----------------------------------------------------------------------------


def _leaves_below(tree: Tree) -> Dict[int, set]:
    """Returns dict {node_idx: set of leaf NAMES descended from node_idx}."""
    out: Dict[int, set] = {}
    for v in range(tree.num_nodes):
        if tree.is_leaf[v]:
            out[v] = {tree.leaf_names[v]}
        else:
            kids = tree.children[v]
            out[v] = set().union(*(out[int(c)] for c in kids))
    return out


@dataclass
class CompositeRatesInfo:
    rates: GLDRates
    provenance: list      # length N, each entry a string label
    glue_nodes: list      # indices of arc269 nodes that got drawn from glue prior
    glue_prior: dict      # {axis: (mu, sd)} on log-rate
    subset_node_counts: dict  # {subset: # arc269 nodes that took rates from this subset}


def _resolve_subset_rates(subset: str) -> tuple[str, np.lib.npyio.NpzFile]:
    """Pick the best-available rates file for a subset, in priority order.

    Priority: brownian-extend ML (the canonical pipeline output) → cold
    brownian MAP → cold MAP σ=1 → bounded-csuros cold ML → sota_ml. The
    first existing one wins.
    """
    candidates = [
        f"validation/outputs/brownian_extend_{subset}/{subset}_final_rates.npz",
        f"validation/outputs/brownian_{subset}/{subset}_sigma1.0_final_rates.npz",
        f"validation/outputs/map_sigma1/{subset}_sigma1.0_final_rates.npz",
        f"validation/outputs/bounded_csuros/{subset}_final_rates.npz",
        f"validation/outputs/sota_ml/{subset}_final_rates.npz",
    ]
    for p in candidates:
        if Path(p).exists():
            return p, np.load(p)
    raise FileNotFoundError(
        f"no rates file found for subset {subset!r}; tried: {candidates}"
    )


def build_composite_arc269_rates(
    arc_tree: Tree,
    *,
    seed: int = 2025,
    glue_sigma_override: float | None = None,
) -> CompositeRatesInfo:
    """Stitch composite arc269 rates from the 4 subtree fits.

    Per subset the best-available rates file is picked (preferring brownian-
    extend ML, falling back through brownian MAP / MAP σ=1 / cold ML).

    Args:
        arc_tree: arc269 Tree.
        seed: RNG seed for sampling glue-node rates.
        glue_sigma_override: if set, use this sd (in log-rate space) for ALL
            glue draws; otherwise use the empirical sd across the 4 subsets'
            non-root per-node log-rates.

    Returns:
        CompositeRatesInfo with the composite GLDRates plus diagnostics.
    """
    rng = np.random.default_rng(seed)

    # Load each subset (tree + brownian-extended rates) + compute mapping to arc269
    subset_data: Dict[str, dict] = {}
    subset_rate_source: Dict[str, str] = {}
    for s in SUBSETS:
        tree_s, _, _, _, _ = load_dataset(s)
        path, d = _resolve_subset_rates(s)
        subset_rate_source[s] = path
        rates_s = GLDRates(tree=tree_s,
                           gain=d["gain"].copy(),
                           loss=d["loss"].copy(),
                           dup=d["dup"].copy(),
                           length=d["length"].copy())
        s_to_arc = map_subset_to_arc269(tree_s, arc_tree)
        arc_to_s = {a: u for u, a in s_to_arc.items()}
        subset_data[s] = {
            "tree": tree_s,
            "rates": rates_s,
            "leaves_set": set(tree_s.leaf_names),
            "arc_to_subset": arc_to_s,
            "size": len(tree_s.leaf_names),
        }

    # Glue prior: pool log-rates from non-root nodes across all 4 subsets.
    # Root nodes are excluded (τ=∞ at root; γ/λ at root encode the gain PMF,
    # which we'll handle separately for the arc269 root).
    def _pool_log(field):
        chunks = []
        for s, sd in subset_data.items():
            tr = sd["tree"]; r = sd["rates"]
            mask = np.array([v != tr.root for v in range(tr.num_nodes)])
            vals = getattr(r, field)[mask]
            vals = vals[(vals > 0) & np.isfinite(vals)]
            if len(vals):
                chunks.append(np.log(vals))
        return np.concatenate(chunks) if chunks else np.array([0.0])
    log_gain_pool   = _pool_log("gain")
    log_loss_pool   = _pool_log("loss")
    log_dup_pool    = _pool_log("dup")
    log_length_pool = _pool_log("length")

    def _stats(pool):
        return float(pool.mean()), (glue_sigma_override if glue_sigma_override is not None
                                     else float(pool.std(ddof=1)))
    mu_sd_gain   = _stats(log_gain_pool)
    mu_sd_loss   = _stats(log_loss_pool)
    mu_sd_dup    = _stats(log_dup_pool)
    mu_sd_length = _stats(log_length_pool)

    # Pre-compute arc269 leaf-name → arc269 leaf-idx for quick lookup of
    # "smallest containing subset" via leaves_below.
    leaves_below = _leaves_below(arc_tree)

    N = arc_tree.num_nodes
    composite_gain   = np.zeros(N)
    composite_loss   = np.zeros(N)
    composite_dup    = np.zeros(N)
    composite_length = np.zeros(N)
    provenance: list = ["" for _ in range(N)]
    glue_nodes: list = []
    subset_counts = {s: 0 for s in SUBSETS}

    for v in range(N):
        L_v = leaves_below[v]
        # Smallest subset s with L_v ⊆ leaves(s)
        smallest = None
        for s, sd in subset_data.items():
            if L_v <= sd["leaves_set"]:
                if smallest is None or subset_data[smallest]["size"] > sd["size"]:
                    smallest = s

        if smallest is None:
            # Glue: v is above all subset LCAs (the arc269 root + inter-clade ancestor)
            glue_nodes.append(v)
            # Draw rates from log-Normal pools. Root has τ=∞, μ=1.0 by convention.
            if v == arc_tree.root:
                composite_length[v] = np.inf
                composite_loss[v]   = 1.0
                provenance[v] = "glue (root: τ=∞, μ=1.0; γ/λ drawn)"
            else:
                composite_length[v] = float(np.exp(rng.normal(*mu_sd_length)))
                composite_loss[v]   = float(np.exp(rng.normal(*mu_sd_loss)))
                provenance[v] = "glue (inter-clade ancestor; all 4 axes drawn)"
            composite_gain[v] = float(np.exp(rng.normal(*mu_sd_gain)))
            composite_dup[v]  = float(np.exp(rng.normal(*mu_sd_dup)))
        else:
            sd = subset_data[smallest]
            u = sd["arc_to_subset"].get(v)
            if u is None:
                # Defensive: L_v ⊆ leaves(s) but v isn't a subset node image.
                # Shouldn't happen because subsets are full clade pruned trees, but
                # fall back to glue with the same prior.
                glue_nodes.append(v)
                composite_gain[v]   = float(np.exp(rng.normal(*mu_sd_gain)))
                composite_loss[v]   = float(np.exp(rng.normal(*mu_sd_loss)))
                composite_dup[v]    = float(np.exp(rng.normal(*mu_sd_dup)))
                composite_length[v] = float(np.exp(rng.normal(*mu_sd_length)))
                provenance[v] = f"glue (fallback: in {smallest} but no node mapping)"
            else:
                rates_s = sd["rates"]
                composite_gain[v]   = float(rates_s.gain[u])
                composite_loss[v]   = float(rates_s.loss[u])
                composite_dup[v]    = float(rates_s.dup[u])
                composite_length[v] = float(rates_s.length[u])
                provenance[v] = f"{smallest}[{u}]"
                subset_counts[smallest] += 1

    composite_rates = GLDRates(tree=arc_tree,
                               gain=composite_gain, loss=composite_loss,
                               dup=composite_dup,  length=composite_length)

    glue_prior = {
        "log_gain":   {"mu": mu_sd_gain[0],   "sd": mu_sd_gain[1]},
        "log_loss":   {"mu": mu_sd_loss[0],   "sd": mu_sd_loss[1]},
        "log_dup":    {"mu": mu_sd_dup[0],    "sd": mu_sd_dup[1]},
        "log_length": {"mu": mu_sd_length[0], "sd": mu_sd_length[1]},
    }
    info = CompositeRatesInfo(
        rates=composite_rates,
        provenance=provenance,
        glue_nodes=glue_nodes,
        glue_prior=glue_prior,
        subset_node_counts=subset_counts,
    )
    info._subset_rate_source = subset_rate_source  # for the summary JSON
    return info


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------


def _summarise_profiles(profiles: np.ndarray) -> dict:
    leaf_sums = profiles.sum(axis=1)
    leaf_mean = profiles.astype(np.float64).mean(axis=0)
    leaf_pres = (profiles > 0).astype(np.float64).mean(axis=0)
    return {
        "F": int(profiles.shape[0]),
        "leaf_sum_mean": float(leaf_sums.mean()),
        "leaf_sum_median": float(np.median(leaf_sums)),
        "leaf_sum_max": int(leaf_sums.max()),
        "leaf_sum_q90": float(np.quantile(leaf_sums, 0.9)),
        "leaf_mean_min": float(leaf_mean.min()),
        "leaf_mean_max": float(leaf_mean.max()),
        "leaf_mean_mean": float(leaf_mean.mean()),
        "leaf_pres_mean": float(leaf_pres.mean()),
        "frac_singletons": float(((profiles > 0).sum(axis=1) == 1).mean()),
        "frac_universal":  float(((profiles > 0).all(axis=1)).mean()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--F", type=int, default=None,
                    help="number of families to simulate "
                         "(default: F_real = 90,243)")
    ap.add_argument("--seed", type=int, default=2025,
                    help="RNG seed for composite glue draws AND simulator")
    ap.add_argument("--glue-sigma", type=float, default=None,
                    help="override the glue prior's log-rate sd (default: empirical)")
    ap.add_argument("--min-copies", type=int, default=None,
                    help="Ωmin for simulation (default: dataset default = 1 for arc269)")
    ap.add_argument("--out-json", type=Path,
                    default=Path("validation/outputs/arc269_composite_sim/summary.json"))
    ap.add_argument("--save-profiles", action="store_true",
                    help="also write the simulated profile table as a compressed npz")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("=" * 70, flush=True)
    print("[1/4] Loading arc269 + 4 subtree brownian-extended rates", flush=True)
    print("=" * 70, flush=True)
    arc_tree, real_profiles, _, mc, _ = load_dataset("arc269")
    if args.min_copies is not None:
        mc = args.min_copies
    F_real = int(real_profiles.shape[0])
    print(f"  arc269: N={arc_tree.num_nodes}, leaves={arc_tree.num_leaves}, "
          f"F_real={F_real}, Ωmin={mc}", flush=True)

    info = build_composite_arc269_rates(
        arc_tree, seed=args.seed, glue_sigma_override=args.glue_sigma,
    )
    print(f"  composite rates built; glue nodes = {info.glue_nodes}", flush=True)
    print(f"  subset → arc269 node counts:  {info.subset_node_counts}", flush=True)
    print(f"  rate source per subset:", flush=True)
    for s, p in info._subset_rate_source.items():
        print(f"     {s:9s} ← {p}", flush=True)
    print(f"  glue log-rate prior:  ", flush=True)
    for k, v in info.glue_prior.items():
        print(f"     {k:12s}  mu={v['mu']:+.3f}  sd={v['sd']:.3f}",
              flush=True)

    rates = info.rates
    print(f"  max gain={rates.gain.max():.3g}  max dup={rates.dup.max():.3g}  "
          f"min length (non-root)={rates.length[np.isfinite(rates.length)].min():.3g}",
          flush=True)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("[2/4] Apply composite rates to REAL arc269 profiles → root posterior", flush=True)
    print("=" * 70, flush=True)
    real_recon = reconstruct(arc_tree, rates, real_profiles, mc)
    print(f"  LL(real | composite rates)  = {real_recon['ll']:.1f}", flush=True)
    print(f"  root fm (corr)              = {real_recon['root_families_corr']:.1f}", flush=True)
    print(f"  root cp (corr)              = {real_recon['root_copies_corr']:.1f}", flush=True)
    print(f"  cp/fm                       = {real_recon['copies_per_family']:.3f}", flush=True)
    print(f"  L(0)                        = {real_recon['L0']:.4f}", flush=True)

    F_sim = args.F if args.F is not None else F_real
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"[3/4] Forward-simulate F={F_sim} families under composite rates  (Ωmin={mc})", flush=True)
    print("=" * 70, flush=True)
    t_sim = time.time()
    sim_profiles, root_counts = simulate_gld(
        arc_tree, rates, F=F_sim, min_copies=mc, seed=args.seed,
    )
    print(f"  done in {time.time() - t_sim:.1f} s", flush=True)
    print(f"  simulated root_count distribution:  mean={root_counts.mean():.2f}  "
          f"sd={root_counts.std():.2f}  max={root_counts.max()}", flush=True)
    print(f"  fraction of sim families with root_count=0:  "
          f"{(root_counts == 0).mean():.3f}", flush=True)

    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"[4/4] Apply composite rates to SIMULATED profiles → root posterior + summary", flush=True)
    print("=" * 70, flush=True)
    sim_recon = reconstruct(arc_tree, rates, sim_profiles, mc)
    print(f"  LL(sim | composite rates)   = {sim_recon['ll']:.1f}", flush=True)
    print(f"  root fm (corr)              = {sim_recon['root_families_corr']:.1f}", flush=True)
    print(f"  root cp (corr)              = {sim_recon['root_copies_corr']:.1f}", flush=True)
    print(f"  cp/fm                       = {sim_recon['copies_per_family']:.3f}", flush=True)
    print(f"  L(0)                        = {sim_recon['L0']:.4f}", flush=True)

    real_summary = _summarise_profiles(real_profiles)
    sim_summary  = _summarise_profiles(sim_profiles)

    print("", flush=True)
    print(f"  Leaf-level marginals (real vs sim):", flush=True)
    keys = ["leaf_sum_mean", "leaf_sum_median", "leaf_sum_q90",
            "leaf_mean_mean", "leaf_pres_mean", "frac_singletons", "frac_universal"]
    for k in keys:
        r = real_summary[k]; s = sim_summary[k]
        rel = (s - r) / abs(r) if abs(r) > 1e-9 else 0.0
        print(f"    {k:18s}  real={r:>10.3f}  sim={s:>10.3f}  Δ rel={rel:+.2%}",
              flush=True)

    def _rel(s, r):
        return f"{100*(s - r)/abs(r):+.2f} %" if abs(r) > 1e-9 else "n/a (denom≈0)"
    print("", flush=True)
    print(f"  Δ vs real LL: {sim_recon['ll'] - real_recon['ll']:+.1f} nat "
          f"({_rel(sim_recon['ll'], real_recon['ll'])})", flush=True)
    print(f"  Δ vs real fm: {sim_recon['root_families_corr'] - real_recon['root_families_corr']:+.1f}  "
          f"({_rel(sim_recon['root_families_corr'], real_recon['root_families_corr'])})", flush=True)
    print(f"  Δ vs real cp: {sim_recon['root_copies_corr'] - real_recon['root_copies_corr']:+.1f}  "
          f"({_rel(sim_recon['root_copies_corr'], real_recon['root_copies_corr'])})", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "seed": args.seed,
        "F_real": F_real,
        "F_sim": F_sim,
        "min_copies": mc,
        "glue_nodes": info.glue_nodes,
        "glue_prior": info.glue_prior,
        "subset_node_counts": info.subset_node_counts,
        "subset_rate_source": info._subset_rate_source,
        "composite_rates_summary": {
            "max_gain": float(rates.gain.max()),
            "max_dup":  float(rates.dup.max()),
            "min_length_nonroot": float(rates.length[np.isfinite(rates.length)].min()),
        },
        "real_recon":    {k: v for k, v in real_recon.items() if k != "stats"},
        "sim_recon":     {k: v for k, v in sim_recon.items()  if k != "stats"},
        "real_summary":  real_summary,
        "sim_summary":   sim_summary,
        "sim_root_counts": {
            "mean": float(root_counts.mean()),
            "sd":   float(root_counts.std(ddof=1)),
            "max":  int(root_counts.max()),
            "frac_zero": float((root_counts == 0).mean()),
        },
        "wall_seconds": float(time.time() - t0),
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: float(x))
    print(f"\n  wrote {args.out_json}", flush=True)

    if args.save_profiles:
        npz_path = args.out_json.with_name("simulated_profiles.npz")
        np.savez_compressed(npz_path,
                            profiles=sim_profiles,
                            root_counts=root_counts)
        print(f"  wrote {npz_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
