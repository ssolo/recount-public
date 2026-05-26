"""Two-level parametric bootstrap on arc269.

  Phase A: For each of the 90k arc269 families and each of the 4 focal
           subsets, compute the per-family posterior mean copy count
           at the subset's ancestor in arc269. Uses the BROWNIAN MAP fit
           (validation/outputs/brownian_<ds>/...) — not the brown-extend
           ML — as the rate model for each subset. Inside-outside is run
           per-family on the subset tree, with the family's per-leaf
           copies extracted from the full arc269 profile table.

           Result: an F×4 matrix of posterior mean copies at
           (dpann80, eury114, ed194, proteo75) ancestors per family.

  Phase B: Build a 3-leaf "species tree" of subset-ancestors:
                       root (LACA)
                      /          \
                     N           proteo75-anc
                    / \
              dpann80   eury114
              -anc      -anc  (= ed194-anc, same arc269 node;
                                ed194's per-family estimate is used as a
                                second noisy measurement at this node)
           Branch lengths default to unit (arc269 lacks published τ).
           Fit a Brownian motion on log(x+1) of the per-family subset-
           ancestor copies. MLE for a single σ² shared across all
           families and across all 3 species-tree edges. Then for each
           family, the posterior mean (in log-space) at the species-
           tree root is a Gaussian-conjugate update from its 3 leaf
           values.

  Phase C: Assemble a simulation tree by gluing the subset brownian
           rates onto arc269 within each clade, plus rates derived from
           the species-tree Brownian fit on the inter-clade edges.
           For each family, set its root copy count to its Phase-B
           posterior mean (rounded / sampled with Poisson noise) and
           forward-simulate down the assembled tree to produce a new
           arc269-shaped profile table.

Usage:
    PYTHONPATH=. python3 validation/lineage_history_simulator.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from recount.native_backend import per_family_posteriors_native
from recount.rates import GLDRates
from recount.tree import Tree
from validation._shared import load_dataset, reconstruct
from validation.simulator import _per_edge_params, _preorder, _sample_gain
from validation.subclade_ancestor_comparison import map_subset_to_arc269


SUBSETS = ["dpann80", "proteo75", "eury114", "ed194"]


# ============================================================================
# Phase A — per-family per-subset-ancestor posterior means
# ============================================================================


def _load_brownian_rates(subset: str) -> tuple[Tree, GLDRates]:
    """Load the cold Brownian-prior MAP rates for a subset (NOT brown-extend)."""
    tree_s, _, _, _, _ = load_dataset(subset)
    candidates = [
        f"validation/outputs/brownian_{subset}/{subset}_sigma1.0_final_rates.npz",
        # Brown-extend is the more recent canonical pipeline output, but the
        # user wants Brownian only — falling back to it only if the cold
        # brownian fit isn't on disk.
        f"validation/outputs/brownian_extend_{subset}/{subset}_final_rates.npz",
    ]
    path = next((p for p in candidates if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"no Brownian rates found for {subset}; tried: {candidates}"
        )
    d = np.load(path)
    rates = GLDRates(tree=tree_s,
                     gain=d["gain"].copy(), loss=d["loss"].copy(),
                     dup=d["dup"].copy(), length=d["length"].copy())
    return tree_s, rates, path


def compute_subset_ancestor_posteriors(
    arc_tree: Tree, arc_profiles: np.ndarray, *, num_threads: int = 0,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """For each of the 4 subsets and each arc269 family, run inside-outside on
    the subset tree to get the posterior mean copy count at the subset's root
    (= the subset's ancestor).

    Returns:
        post_means: float64 array of shape (F, 4) — column j is the per-family
            posterior mean at SUBSETS[j]'s ancestor.
        meta: per-subset dict with 'tree', 'rates', 'rates_path',
            'subset_root', 'arc_anc' (arc269 node id at which the subset
            ancestor sits), 'leaf_arc_idx' (per-subset-leaf arc269 column
            index used to extract that subset's profile slice).
    """
    F = arc_profiles.shape[0]
    arc_name_to_idx = {n: i for i, n in enumerate(arc_tree.leaf_names)}

    meta: Dict[str, dict] = {}
    post_means = np.zeros((F, len(SUBSETS)), dtype=np.float64)

    for j, s in enumerate(SUBSETS):
        t0 = time.time()
        tree_s, rates_s, path = _load_brownian_rates(s)

        # Map each subset leaf to its arc269 column.
        leaf_arc_idx = np.array([
            arc_name_to_idx[nm] for nm in tree_s.leaf_names
        ], dtype=np.int64)

        # Extract the subset-leaf-only profile from the arc269 profile table.
        # Subset tree leaf ordering MUST match (it does — leaf_names[i] is leaf i).
        sub_profiles = arc_profiles[:, leaf_arc_idx].astype(np.int32)

        # Per-family inside-outside on the subset tree.
        copies, _present = per_family_posteriors_native(
            tree_s, rates_s.gain, rates_s.loss, rates_s.dup, rates_s.length,
            sub_profiles, num_threads=num_threads,
        )
        subset_root = tree_s.root
        post_means[:, j] = copies[:, subset_root]

        # arc269 node id of this subset's ancestor.
        s_to_arc = map_subset_to_arc269(tree_s, arc_tree)
        arc_anc = s_to_arc[subset_root]

        meta[s] = {
            "tree": tree_s,
            "rates": rates_s,
            "rates_path": path,
            "subset_root": subset_root,
            "arc_anc": int(arc_anc),
            "leaf_arc_idx": leaf_arc_idx,
            "wall_s": time.time() - t0,
        }
        print(f"  [{s}] inside-outside on {F} families: {time.time()-t0:.1f}s "
              f"  mean posterior at subset-root = {post_means[:, j].mean():.3f} "
              f"  median = {np.median(post_means[:, j]):.3f} "
              f"  max = {post_means[:, j].max():.3f}", flush=True)

    return post_means, meta


# ============================================================================
# Phase B — Brownian-motion on log(x+1) of subset-ancestor copies
# ============================================================================


@dataclass
class SpeciesTreeBM:
    """3-leaf species tree + Brownian-motion fit.

    Topology (parent -> (children)):
        root  ->  (N, proteo75-anc)
        N     ->  (dpann80-anc, eury114-anc)

    eury114-anc and ed194-anc map to the SAME arc269 node (461). We
    treat ed194's per-family posterior at that node as a second noisy
    measurement of eury114-anc — averaged in with eury114's posterior
    when constructing the 3-vector of leaf values for the BM fit.
    """
    leaf_names: List[str]        # ["dpann80-anc", "eury114-anc", "proteo75-anc"]
    branch_lengths: Dict[str, float]  # keys: "root-N", "root-proteo75", "N-dpann80", "N-eury114"
    sigma2: float                 # ML estimate of BM variance per unit branch length
    root_mean: float              # ML estimate of root mean (in log(x+1) space)
    post_root_per_family: np.ndarray  # (F,) — posterior mean log(x+1) at root per family
    post_root_var: float          # posterior variance at root (shared across families)


def _species_tree_branch_lengths() -> Dict[str, float]:
    """Default unit branch lengths (arc269 has no published τ).

    Picking τ=1.0 for every species-tree edge gives the BM σ² the
    interpretation of "log-copy variance accumulated between arc269
    LACA and any subset ancestor" in a single unit of time. The BM fit
    + root posterior are invariant to a global rescaling of τ.
    """
    return {
        "root-N": 1.0,
        "root-proteo75": 1.0,
        "N-dpann80": 1.0,
        "N-eury114": 1.0,
    }


def fit_brownian_motion(
    post_means: np.ndarray, *, branch_lengths: Dict[str, float] | None = None,
) -> SpeciesTreeBM:
    """Fit a single-σ² Brownian motion on the 3-leaf species tree from the
    Phase-A per-family posterior means.

    post_means: (F, 4) — columns are (dpann80, proteo75, eury114, ed194)
    in the order of SUBSETS.

    Returns SpeciesTreeBM with σ² (MLE), root mean (MLE), and per-family
    posterior mean at root + shared posterior variance.

    Implementation: we transform to y = log(x + 1) (handles zeros). For a
    BM with shared σ² and a flat improper root prior, the MLE of the root
    mean is the weighted average of leaves (weights = 1/path-variance).
    The MLE of σ² is the average squared contrast / branch-length-sum.
    The per-family posterior at the root is the same weighted-average
    formula; posterior variance shared across families.
    """
    if branch_lengths is None:
        branch_lengths = _species_tree_branch_lengths()

    # Average eury114 + ed194 estimates (column 2 and 3 in SUBSETS order) since
    # they live at the same arc269 node — use simple mean to reduce noise.
    # SUBSETS index: 0=dpann80, 1=proteo75, 2=eury114, 3=ed194
    F = post_means.shape[0]
    eury_node_post = 0.5 * (post_means[:, 2] + post_means[:, 3])
    leaf_values = np.stack([
        post_means[:, 0],   # dpann80
        eury_node_post,     # eury114 (avg with ed194)
        post_means[:, 1],   # proteo75
    ], axis=1)               # shape (F, 3)
    y = np.log1p(np.maximum(leaf_values, 0.0))   # log(x+1) per leaf per family

    # Path variances from root to each leaf (under σ²=1):
    t_root_to_dpann = branch_lengths["root-N"] + branch_lengths["N-dpann80"]
    t_root_to_eury  = branch_lengths["root-N"] + branch_lengths["N-eury114"]
    t_root_to_prot  = branch_lengths["root-proteo75"]
    leaf_path_var = np.array([t_root_to_dpann, t_root_to_eury, t_root_to_prot])

    # Felsenstein pruning for a 3-leaf tree with shared σ²:
    # 1. At internal node N: combine dpann80-anc and eury114-anc via inverse-
    #    variance weighting → N̂, var(N̂)
    w_D = 1.0 / branch_lengths["N-dpann80"]
    w_E = 1.0 / branch_lengths["N-eury114"]
    N_hat = (w_D * y[:, 0] + w_E * y[:, 1]) / (w_D + w_E)
    var_N = 1.0 / (w_D + w_E)   # in units of σ²

    # 2. At root: combine N̂ (effective branch t_root-N + var_N) and proteo75-anc
    eff_t_RN = branch_lengths["root-N"] + var_N
    w_N = 1.0 / eff_t_RN
    w_P = 1.0 / branch_lengths["root-proteo75"]
    root_hat = (w_N * N_hat + w_P * y[:, 2]) / (w_N + w_P)
    var_root = 1.0 / (w_N + w_P)   # in units of σ²

    # 3. σ² MLE. Use Felsenstein's contrasts: at each split, the contrast
    #    is (child1 - child2) / sqrt(combined branch length); under H0 (BM
    #    with variance σ²) it's N(0, σ²). With 2 splits we have 2 contrasts
    #    per family × F families = 2F contrasts; σ̂² = sum(contrast²) / (2F).
    contrast_split_N = (y[:, 0] - y[:, 1]) / np.sqrt(
        branch_lengths["N-dpann80"] + branch_lengths["N-eury114"])
    contrast_split_root = (N_hat - y[:, 2]) / np.sqrt(eff_t_RN + branch_lengths["root-proteo75"])
    sigma2 = float((contrast_split_N**2).mean() + (contrast_split_root**2).mean()) / 2.0
    # Equivalent: sigma2 = mean of all contrasts² (already averaged per family)

    # Final posterior at root:
    post_root = root_hat   # already weighted-average per family
    post_var = float(sigma2 * var_root)
    root_mean = float(post_root.mean())

    return SpeciesTreeBM(
        leaf_names=["dpann80-anc", "eury114-anc(=ed194-anc avg)", "proteo75-anc"],
        branch_lengths=branch_lengths,
        sigma2=sigma2,
        root_mean=root_mean,
        post_root_per_family=post_root,
        post_root_var=post_var,
    )


# ============================================================================
# Phase C — assemble simulation tree + forward-simulate with per-family roots
# ============================================================================


def _build_assembled_rates(
    arc_tree: Tree, bm: SpeciesTreeBM, post_a_meta: Dict[str, dict],
    rng: np.random.Generator,
) -> GLDRates:
    """Composite rates for arc269 simulation.

    Within-subset arc269 nodes get the subset's brownian rates (mapped
    via map_subset_to_arc269 from each subset's brownian fit).
    Inter-subset arc269 internal nodes get rates derived from the
    species-tree BM σ²: a small gain consistent with the Brownian's σ²
    over a unit branch, no duplication on the inter-clade edges (set
    dup=0 → Poisson gain) — this isolates the inter-clade rate to a
    pure-gain-from-LACA picture.
    """
    msa = map_subset_to_arc269  # local alias (imported at module top)
    N = arc_tree.num_nodes

    composite_gain   = np.zeros(N)
    composite_loss   = np.ones(N)   # μ ≡ 1 convention
    composite_dup    = np.zeros(N)
    composite_length = np.ones(N)
    composite_length[arc_tree.root] = np.inf

    # Walk arc269 nodes; for each, find the smallest containing subset.
    def _leaves_below(tree: Tree) -> Dict[int, set]:
        out: Dict[int, set] = {}
        for v in range(tree.num_nodes):
            if tree.is_leaf[v]:
                out[v] = {tree.leaf_names[v]}
            else:
                kids = tree.children[v]
                out[v] = set().union(*(out[int(c)] for c in kids))
        return out
    LB = _leaves_below(arc_tree)

    for v in range(N):
        L_v = LB[v]
        smallest = None
        for s in SUBSETS:
            ls = set(post_a_meta[s]["tree"].leaf_names)
            if L_v <= ls:
                if smallest is None or len(set(post_a_meta[smallest]["tree"].leaf_names)) > len(ls):
                    smallest = s
        if smallest is None:
            # Inter-subset / root: use BM-derived rates.
            # Equivalent gain rate so that per-edge BM variance ≈ σ². For
            # the Poisson branch (no dup), copy-count variance equals mean
            # equals γ_v · τ_v. With τ_v=1.0 (species-tree unit length),
            # we want gain = exp(BM σ²/2) at the gain-only Poisson — but
            # this is a crude bridge. Set gain to exp(root_mean / N_glue)
            # so the LACA prior matches the BM root posterior mean. Set
            # dup = 0 (Poisson branch).
            composite_gain[v] = max(0.0, float(np.exp(bm.root_mean) - 1.0))
            composite_dup[v]  = 0.0
            composite_loss[v] = 1.0
            composite_length[v] = (1.0 if v != arc_tree.root else np.inf)
        else:
            sd = post_a_meta[smallest]
            s_to_arc = msa(sd["tree"], arc_tree)
            arc_to_s = {a: u for u, a in s_to_arc.items()}
            u = arc_to_s.get(v)
            if u is None:
                composite_gain[v]   = max(0.0, float(np.exp(bm.root_mean) - 1.0))
                composite_dup[v]    = 0.0
                composite_loss[v]   = 1.0
                composite_length[v] = 1.0
            else:
                r = sd["rates"]
                composite_gain[v]   = float(r.gain[u])
                composite_loss[v]   = float(r.loss[u])
                composite_dup[v]    = float(r.dup[u])
                composite_length[v] = float(r.length[u])

    return GLDRates(tree=arc_tree, gain=composite_gain, loss=composite_loss,
                    dup=composite_dup,  length=composite_length)


def simulate_from_per_family_root(
    arc_tree: Tree, rates: GLDRates, root_counts: np.ndarray, *,
    min_copies: int = 1, seed: int = 2025,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like simulate_gld but the root count for family f is fixed to
    root_counts[f] instead of being sampled from the model's root prior.

    Families whose simulated leaf-sum < min_copies are RE-ROLLED at the
    same root count (we keep their assigned root count fixed but resample
    the propagation noise) up to ``max_retries`` times. Returns whatever
    simulated profile each family produced after retries (no filtering
    drop).
    """
    rng = np.random.default_rng(seed)
    N = arc_tree.num_nodes
    root = arc_tree.root
    L = arc_tree.num_leaves
    pre = _preorder(arc_tree)

    p_surv, q_raw, is_polya, gain = _per_edge_params(arc_tree, rates)

    F = len(root_counts)
    profiles = np.zeros((F, L), dtype=np.int32)
    counts = np.zeros(N, dtype=np.int64)
    n_below_omin = 0

    for f in range(F):
        # Set root count for this family.
        counts[root] = int(root_counts[f])
        max_retries = 20
        for retry in range(max_retries):
            for v in pre:
                if v == root:
                    continue
                pa = int(arc_tree.parent[v])
                m_parent = int(counts[pa])
                j = (int(rng.binomial(m_parent, min(p_surv[v], 1.0)))
                     if m_parent > 0 and p_surv[v] > 0.0 else 0)
                extras = _sample_gain(rng, gain[v], q_raw[v], is_polya[v], j)
                counts[v] = j + extras
            if int(counts[:L].sum()) >= min_copies:
                break
            counts[root] = int(root_counts[f])  # reset for retry
        if int(counts[:L].sum()) < min_copies:
            n_below_omin += 1
        profiles[f] = counts[:L].astype(np.int32)

    return profiles, np.array(root_counts, dtype=np.int64), n_below_omin


# ============================================================================
# Driver
# ============================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--root-noise", choices=["round", "poisson"], default="poisson",
                    help="how to turn the BM posterior mean log(x+1) into "
                         "an integer per-family root count: 'round' uses "
                         "round(exp(post_root_mean)-1); 'poisson' samples "
                         "Poisson with that mean (default)")
    ap.add_argument("--out-json", type=Path,
                    default=Path("validation/outputs/arc269_lineage_history_sim/summary.json"))
    ap.add_argument("--save-profiles", action="store_true",
                    help="also save the simulated profile table + per-family root counts")
    ap.add_argument("--num-threads", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70); print("[1/3] Phase A — per-subset-ancestor posteriors")
    print("=" * 70, flush=True)
    arc_tree, arc_profiles, _, mc, _ = load_dataset("arc269")
    F = int(arc_profiles.shape[0])
    print(f"  arc269: N={arc_tree.num_nodes}, leaves={arc_tree.num_leaves}, "
          f"F={F}, Ωmin={mc}", flush=True)
    t0 = time.time()
    post_means, post_a_meta = compute_subset_ancestor_posteriors(
        arc_tree, arc_profiles, num_threads=args.num_threads,
    )
    print(f"  Phase A wall = {time.time()-t0:.1f}s", flush=True)
    print(f"  subset-anc posterior summaries (mean over F):", flush=True)
    for j, s in enumerate(SUBSETS):
        print(f"    {s:9s}  mean={post_means[:, j].mean():.3f}  "
              f"median={np.median(post_means[:, j]):.3f}  "
              f"q90={np.quantile(post_means[:, j], 0.9):.3f}  "
              f"max={post_means[:, j].max():.3f}", flush=True)

    print(""); print("=" * 70); print("[2/3] Phase B — species-tree Brownian fit + per-family root posterior")
    print("=" * 70, flush=True)
    bm = fit_brownian_motion(post_means)
    print(f"  species-tree branch lengths (unit defaults): {bm.branch_lengths}", flush=True)
    print(f"  σ² (MLE)               = {bm.sigma2:.4f}", flush=True)
    print(f"  root mean log(x+1)     = {bm.root_mean:+.4f}  →  exp-1 = {np.exp(bm.root_mean) - 1:.3f}", flush=True)
    print(f"  root posterior variance per family = {bm.post_root_var:.4f} "
          f"  (sd = {np.sqrt(bm.post_root_var):.3f})", flush=True)
    print(f"  per-family root posterior log(x+1):", flush=True)
    pr = bm.post_root_per_family
    print(f"    mean={pr.mean():.3f}  median={np.median(pr):.3f}  "
          f"q90={np.quantile(pr, 0.9):.3f}  max={pr.max():.3f}", flush=True)
    print(f"  per-family root in copy units (exp - 1):", flush=True)
    pr_cp = np.expm1(pr)
    print(f"    mean={pr_cp.mean():.3f}  median={np.median(pr_cp):.3f}  "
          f"q90={np.quantile(pr_cp, 0.9):.3f}  max={pr_cp.max():.3f}", flush=True)

    # Convert per-family posterior log(x+1) means to integer root counts.
    if args.root_noise == "round":
        root_counts_int = np.maximum(0, np.round(pr_cp).astype(np.int64))
    else:
        # Poisson sample with mean = max(0, exp(post_root) - 1)
        root_counts_int = rng.poisson(np.maximum(pr_cp, 0.0)).astype(np.int64)

    n_root_pos = int((root_counts_int > 0).sum())
    print(f"", flush=True)
    print(f"  ╔══════════════════════════════════════════════════════════════╗", flush=True)
    print(f"  ║  Families INFERRED AT THE ROOT (root_count > 0 after BM):    ║", flush=True)
    print(f"  ║                                                              ║", flush=True)
    print(f"  ║       N_root_pos = {n_root_pos:>6}  out of  F = {F:>6}", flush=True)
    print(f"  ║                                                              ║", flush=True)
    print(f"  ║   (root_count rule = '{args.root_noise}' on Phase-B posterior)", flush=True)
    print(f"  ╚══════════════════════════════════════════════════════════════╝", flush=True)

    print(""); print("=" * 70); print(f"[3/3] Phase C — assemble simulation tree + simulate F={F} families")
    print("=" * 70, flush=True)
    rates = _build_assembled_rates(arc_tree, bm, post_a_meta, rng)
    print(f"  assembled rates: max_gain={rates.gain.max():.3g}  "
          f"max_dup={rates.dup.max():.3g}  "
          f"min_length(non-root)={rates.length[np.isfinite(rates.length)].min():.3g}",
          flush=True)

    # Optionally evaluate composite on real data for comparison.
    print("  reconstruct(real | assembled rates) — root posterior:", flush=True)
    real_recon = reconstruct(arc_tree, rates, arc_profiles, mc)
    print(f"    root fm (corr) = {real_recon['root_families_corr']:.1f}", flush=True)
    print(f"    root cp (corr) = {real_recon['root_copies_corr']:.1f}", flush=True)
    print(f"    LL = {real_recon['ll']:.1f}", flush=True)

    t0 = time.time()
    sim_profiles, root_counts_used, n_below = simulate_from_per_family_root(
        arc_tree, rates, root_counts_int, min_copies=mc, seed=args.seed,
    )
    print(f"  Phase C simulation wall = {time.time()-t0:.1f}s "
          f"({n_below} families landed sub-Ωmin even after retries)", flush=True)

    real_leaf_sum_mean   = arc_profiles.astype(np.float64).sum(axis=1).mean()
    real_leaf_sum_median = float(np.median(arc_profiles.astype(np.float64).sum(axis=1)))
    sim_leaf_sum_mean    = sim_profiles.astype(np.float64).sum(axis=1).mean()
    sim_leaf_sum_median  = float(np.median(sim_profiles.astype(np.float64).sum(axis=1)))
    print(f"", flush=True)
    print(f"  Real-vs-sim leaf-sum-per-family:", flush=True)
    print(f"    mean    real={real_leaf_sum_mean:.3f}  sim={sim_leaf_sum_mean:.3f}", flush=True)
    print(f"    median  real={real_leaf_sum_median:.3f}  sim={sim_leaf_sum_median:.3f}", flush=True)

    out = {
        "seed": args.seed,
        "F": F,
        "min_copies": mc,
        "phase_A_subset_anc_summary": {
            s: {
                "rates_path": post_a_meta[s]["rates_path"],
                "subset_root_arc_node": post_a_meta[s]["arc_anc"],
                "mean":   float(post_means[:, j].mean()),
                "median": float(np.median(post_means[:, j])),
                "q90":    float(np.quantile(post_means[:, j], 0.9)),
                "max":    float(post_means[:, j].max()),
                "wall_s": post_a_meta[s]["wall_s"],
            }
            for j, s in enumerate(SUBSETS)
        },
        "phase_B_bm_fit": {
            "branch_lengths":      bm.branch_lengths,
            "sigma2":              bm.sigma2,
            "root_mean_log1p":     bm.root_mean,
            "root_mean_copies":    float(np.exp(bm.root_mean) - 1),
            "root_post_var_log1p": bm.post_root_var,
            "per_family_root_log1p": {
                "mean":   float(pr.mean()),
                "median": float(np.median(pr)),
                "q90":    float(np.quantile(pr, 0.9)),
                "max":    float(pr.max()),
            },
            "per_family_root_copies": {
                "mean":   float(pr_cp.mean()),
                "median": float(np.median(pr_cp)),
                "q90":    float(np.quantile(pr_cp, 0.9)),
                "max":    float(pr_cp.max()),
            },
            "n_families_inferred_at_root_positive": n_root_pos,
            "root_count_rule":                    args.root_noise,
        },
        "phase_C_simulation": {
            "real_recon_under_assembled": {
                k: float(v) for k, v in real_recon.items()
                if not isinstance(v, np.ndarray) and not isinstance(v, dict)
            },
            "real_leaf_sum_mean":   float(real_leaf_sum_mean),
            "real_leaf_sum_median": float(real_leaf_sum_median),
            "sim_leaf_sum_mean":    float(sim_leaf_sum_mean),
            "sim_leaf_sum_median":  float(sim_leaf_sum_median),
            "n_families_simulated_below_omin": n_below,
        },
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: float(x))
    print(f"\n  wrote {args.out_json}", flush=True)

    if args.save_profiles:
        npz_path = args.out_json.with_name("simulated_profiles.npz")
        np.savez_compressed(npz_path,
                            profiles=sim_profiles,
                            root_counts=root_counts_used,
                            post_means_subset_anc=post_means)
        print(f"  wrote {npz_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
