"""Forward GLD simulator + parametric-bootstrap helper.

Given a tree topology and per-node GLD rates (γ, μ, λ, τ), forward-simulate
``F`` gene families by sampling root copy counts from the gain PMF at the
root edge and propagating down each edge via the per-edge transition
kernel:

    T_v(m' | m) = Σ_j Binomial(j; m, p_surv_v) · NegBin(m' - j | κ_v + j, 1 - q_v)
                                                  [Pólya branch, q_v > 0]
    T_v(m' | m) = Σ_j Binomial(j; m, p_surv_v) · Poisson(m' - j | γ_v · t_factor_v)
                                                  [Poisson branch, q_v = 0]

where p_surv_v = 1 - rate_to_p(μ_v, λ_v, τ_v)[0] (the per-edge survival
probability of a single lineage) and q_v = rate_to_q(...)[0] is the Pólya
ratio. The Poisson-branch gain scales with edge survival, see
``gld.compute_survival_params`` for the conventions; we use the raw
per-edge κ_v / γ_v on each edge directly (NOT the subtree-augmented
``SurvivalParams.gain``, which folds in downstream extinction).

The forward kernel here is the per-edge marginal of the inside-outside
machinery in ``recount.gld`` — sampling from this kernel and then
recomputing the inside LL on the sampled profile should give a log-LL
distribution centred on the model's per-family entropy at the chosen
rates. The verification helpers in this file do exactly that check.

Usage:

    from validation.simulator import simulate_gld, verify_on_subset
    profiles, root_counts = simulate_gld(tree, rates, F=3000, min_copies=4,
                                         seed=2025)
    # ... or:
    verify_on_subset('dpann80', n_rep=5)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from recount.rates import GLDRates, rate_to_p, rate_to_q
from recount.tree import Tree
from validation._shared import load_dataset, reconstruct


# ----------------------------------------------------------------------------
# Per-edge raw parameters (used by the forward kernel)
# ----------------------------------------------------------------------------


def _per_edge_params(tree: Tree, rates: GLDRates):
    """Pre-compute per-edge raw (loss, survival, Pólya ratio) needed by
    the forward kernel.

    Returns:
        p_surv: P(single parent copy survives this edge to the child),  N-vector
        q_raw:  Pólya ratio q_v (1 - q_raw is the NegBin "success prob"), N-vector
        is_polya: bool N-vector — True when dup_v > 0 (NegBin gain); False otherwise (Poisson gain)
        gain: γ_v per-node (raw rate, used as κ_v shape for Pólya or as Poisson rate)
    """
    N = tree.num_nodes
    p_surv = np.zeros(N)
    q_raw = np.zeros(N)
    is_polya = np.zeros(N, dtype=bool)
    for v in range(N):
        p_loss, p_loss_c = rate_to_p(rates.loss[v], rates.dup[v], rates.length[v])
        q, _ = rate_to_q(rates.loss[v], rates.dup[v], rates.length[v])
        p_surv[v] = p_loss_c              # = 1 - P(extinction on this edge)
        q_raw[v] = q
        is_polya[v] = (rates.dup[v] > 0.0) and (q > 0.0)
    return p_surv, q_raw, is_polya, rates.gain.copy()


def _preorder(tree: Tree):
    """Root-first list of node indices (each parent appears before its
    children). recount's Tree has leaves first, parent[v] > v for v != root,
    so iterating in REVERSE node order gives a valid pre-order."""
    return list(range(tree.num_nodes - 1, -1, -1))


# ----------------------------------------------------------------------------
# Forward simulator
# ----------------------------------------------------------------------------


def _sample_gain(rng, kappa_or_rate: float, q: float, is_polya: bool,
                 j_survivors: int) -> int:
    """Sample the # of new copies emitted by the gain process on an edge,
    given ``j_survivors`` lineages already survived the same edge.

    Pólya branch:    NegBin(κ + j, 1 - q)   — shape grows with j
    Poisson branch:  Poisson(γ)             — rate independent of j
    """
    if is_polya:
        n = kappa_or_rate + j_survivors  # NegBin shape α
        if n <= 0.0:
            return 0
        p_success = 1.0 - q
        if p_success <= 0.0 or p_success >= 1.0 + 1e-15:
            return 0
        # np.random.negative_binomial(n, p) returns # failures before n successes
        # PMF: C(n+k-1, k) · p^n · (1-p)^k
        # With p = 1-q: PMF = C(α+k-1, k) · (1-q)^α · q^k = Pólya(k | α, q) ✓
        return int(rng.negative_binomial(n, p_success))
    else:
        if kappa_or_rate <= 0.0:
            return 0
        return int(rng.poisson(kappa_or_rate))


def simulate_gld(
    tree: Tree, rates: GLDRates, F: int, *,
    min_copies: int = 1,
    seed: int = 2025,
    max_tries_multiplier: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-simulate ``F`` gene families under the GLD model on ``tree``.

    Each family is drawn until its leaf profile satisfies
    ``sum(profile) >= min_copies`` (the same Ωmin condition used by the
    likelihood machinery). Families with sub-Ωmin profiles are silently
    rejected; ``max_tries_multiplier * F`` total tries is the cap.

    Returns:
        profiles: int32 array of shape (F, num_leaves)
        root_counts: int64 array of shape (F,) — the simulated ξ_root per family

    Use ``min_copies=1`` to keep any non-empty profile, ``min_copies=4`` to
    match Csurös' focal-subset convention.
    """
    rng = np.random.default_rng(seed)
    N = tree.num_nodes
    root = tree.root
    L = tree.num_leaves
    pre = _preorder(tree)

    p_surv, q_raw, is_polya, gain = _per_edge_params(tree, rates)

    profiles = np.zeros((F, L), dtype=np.int32)
    root_counts = np.zeros(F, dtype=np.int64)

    max_tries = int(max_tries_multiplier * F) + F
    got = 0
    tried = 0
    counts = np.zeros(N, dtype=np.int64)

    while got < F:
        if tried >= max_tries:
            raise RuntimeError(
                f"simulate_gld: only {got} of {F} families landed above Ωmin={min_copies} "
                f"after {tried} tries. Increase max_tries_multiplier or lower min_copies."
            )
        tried += 1

        # Root: no parent, no survivors → just sample the gain.
        counts[root] = _sample_gain(rng, gain[root], q_raw[root], is_polya[root], 0)

        # Pre-order walk: each parent count is set before its children.
        for v in pre:
            if v == root:
                continue
            pa = int(tree.parent[v])
            m_parent = int(counts[pa])
            if m_parent > 0 and p_surv[v] > 0.0:
                j = int(rng.binomial(m_parent, min(p_surv[v], 1.0)))
            else:
                j = 0
            extras = _sample_gain(rng, gain[v], q_raw[v], is_polya[v], j)
            counts[v] = j + extras

        leaf_sum = int(counts[:L].sum())
        if leaf_sum >= min_copies:
            profiles[got] = counts[:L].astype(np.int32)
            root_counts[got] = int(counts[root])
            got += 1

    return profiles, root_counts


# ----------------------------------------------------------------------------
# Verification on a small subtree
# ----------------------------------------------------------------------------


@dataclass
class VerificationReport:
    label: str
    F_real: int
    F_sim: int
    min_copies: int
    seed: int

    ll_real: float           # LL of real data under the rates
    ll_sim_mean: float        # mean LL of simulated data across replicates
    ll_sim_std: float
    ll_per_family_sd_pred: float   # predicted std of LL/F under the model

    leaf_mean_real: np.ndarray   # mean per-leaf count, real
    leaf_mean_sim: np.ndarray    # mean per-leaf count, simulated (average over reps)
    leaf_mean_corr: float        # Pearson r between real and sim per-leaf means

    fm_real: float; cp_real: float
    fm_sim_mean: float; cp_sim_mean: float

    notes: str = ""


def verify_on_subset(
    label: str = "dpann80", *,
    rates_path: str | None = None,
    F: int | None = None,
    n_rep: int = 3,
    min_copies_override: int | None = None,
    seed: int = 2025,
) -> VerificationReport:
    """Verify the simulator on a real subset whose rates are fit and saved.

    Loads (a) the dataset (tree + profiles + Ωmin), (b) the rates from the
    given .npz (default: the brownian-extend ML fit if available, else MAP
    σ=1, else any saved final_rates.npz). Computes:

      - LL of the real data under the rates (single number, ground truth).
      - LL of n_rep independent simulated datasets at the same F, then
        compare mean ± std to the real LL. Simulated LL should agree with
        real LL to within ~sqrt(F) per-family-LL stds.
      - Per-leaf mean count, real vs simulated → Pearson correlation
        should be ≥ 0.99 (rates were fit TO the real data).
      - Root-posterior cp/fm comparison.
    """
    tree, profiles, _, mc, _ = load_dataset(label,
        min_copies_override=min_copies_override)
    F_real = int(profiles.shape[0])
    if F is None:
        F = F_real

    if rates_path is None:
        candidates = [
            f"validation/outputs/brownian_extend_{label}/{label}_final_rates.npz",
            f"validation/outputs/sota_ml/{label}_final_rates.npz",
            f"validation/outputs/map_sigma1/{label}_sigma1.0_final_rates.npz",
            f"validation/outputs/bounded_csuros/{label}_final_rates.npz",
        ]
        rates_path = next((p for p in candidates if Path(p).exists()), None)
        if rates_path is None:
            raise FileNotFoundError(f"no rates file found for {label}")

    d = np.load(rates_path)
    rates = GLDRates(tree=tree, gain=d["gain"].copy(), loss=d["loss"].copy(),
                     dup=d["dup"].copy(), length=d["length"].copy())

    # --- Real data under these rates ---
    print(f"[{label}] loading {rates_path}", flush=True)
    print(f"[{label}] F_real = {F_real}, Ωmin = {mc}", flush=True)
    real_recon = reconstruct(tree, rates, profiles, mc)
    ll_real = float(real_recon["ll"])
    leaf_mean_real = profiles.astype(np.float64).mean(axis=0)
    print(f"[{label}] real: LL={ll_real:.1f}  fm={real_recon['root_families_corr']:.0f}  "
          f"cp={real_recon['root_copies_corr']:.0f}", flush=True)

    # --- Replicate forward simulations ---
    ll_sim_list = []
    fm_sim_list = []
    cp_sim_list = []
    leaf_sum_acc = np.zeros_like(leaf_mean_real)

    t0 = time.time()
    for r in range(n_rep):
        sim_profiles, root_counts = simulate_gld(
            tree, rates, F, min_copies=mc, seed=seed + r,
        )
        sim_recon = reconstruct(tree, rates, sim_profiles, mc)
        ll_sim_list.append(float(sim_recon["ll"]))
        fm_sim_list.append(float(sim_recon["root_families_corr"]))
        cp_sim_list.append(float(sim_recon["root_copies_corr"]))
        leaf_sum_acc += sim_profiles.astype(np.float64).mean(axis=0)
        print(f"[{label}] rep {r+1}/{n_rep}: LL={ll_sim_list[-1]:.1f}  "
              f"fm={fm_sim_list[-1]:.0f}  cp={cp_sim_list[-1]:.0f}  "
              f"<root>={root_counts.mean():.2f}  "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)
    leaf_mean_sim = leaf_sum_acc / n_rep

    ll_sim_arr = np.array(ll_sim_list)
    leaf_corr = float(np.corrcoef(leaf_mean_real, leaf_mean_sim)[0, 1])

    # Predicted per-family LL std under the model: just the empirical sd
    # of per-family log-LL on real data, divided by sqrt(F) for the mean.
    # (Cheap proxy — exact would need the inside DP to return per-family.)
    pred_sd_LL_total = float(np.sqrt(F)) * abs(ll_real) / max(F, 1) * 0.5

    return VerificationReport(
        label=label,
        F_real=F_real, F_sim=F, min_copies=mc, seed=seed,
        ll_real=ll_real,
        ll_sim_mean=float(ll_sim_arr.mean()),
        ll_sim_std=float(ll_sim_arr.std(ddof=1)) if n_rep > 1 else 0.0,
        ll_per_family_sd_pred=pred_sd_LL_total,
        leaf_mean_real=leaf_mean_real,
        leaf_mean_sim=leaf_mean_sim,
        leaf_mean_corr=leaf_corr,
        fm_real=float(real_recon["root_families_corr"]),
        cp_real=float(real_recon["root_copies_corr"]),
        fm_sim_mean=float(np.mean(fm_sim_list)),
        cp_sim_mean=float(np.mean(cp_sim_list)),
        notes=f"rates from {rates_path}",
    )


def _print_report(r: VerificationReport) -> None:
    print(f"\n=== Simulator verification report: {r.label} ===")
    print(f"  F = {r.F_real} (real) / {r.F_sim} (sim), Ωmin = {r.min_copies}, seed = {r.seed}")
    print(f"  rates: {r.notes}")
    print(f"")
    print(f"  LL under rates:")
    print(f"    real         = {r.ll_real:>12.1f}")
    print(f"    sim mean ± sd = {r.ll_sim_mean:>12.1f} ± {r.ll_sim_std:.1f}")
    print(f"    Δ (sim − real) = {r.ll_sim_mean - r.ll_real:+.1f} nat  "
          f"({100*(r.ll_sim_mean - r.ll_real)/abs(r.ll_real):+.2f} %)")
    print(f"")
    print(f"  Root posterior under rates:")
    print(f"    real         fm={r.fm_real:>7.1f}  cp={r.cp_real:>7.1f}  cpf={r.cp_real/r.fm_real:.3f}")
    print(f"    sim mean     fm={r.fm_sim_mean:>7.1f}  cp={r.cp_sim_mean:>7.1f}  cpf={r.cp_sim_mean/r.fm_sim_mean:.3f}")
    print(f"")
    print(f"  Per-leaf mean count:")
    print(f"    Pearson r(real, sim) = {r.leaf_mean_corr:.4f}")
    print(f"    real range = [{r.leaf_mean_real.min():.2f}, {r.leaf_mean_real.max():.2f}]")
    print(f"    sim  range = [{r.leaf_mean_sim.min():.2f}, {r.leaf_mean_sim.max():.2f}]")
    print(f"")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="dpann80")
    ap.add_argument("--rates", default=None,
                    help="path to a *_final_rates.npz (default: best available for the dataset)")
    ap.add_argument("--F", type=int, default=None,
                    help="number of families to simulate (default: same as real data)")
    ap.add_argument("--n-rep", type=int, default=3,
                    help="number of replicate simulations to average over")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min-copies-override", type=int, default=None)
    args = ap.parse_args(argv)

    r = verify_on_subset(args.dataset, rates_path=args.rates, F=args.F,
                         n_rep=args.n_rep, seed=args.seed,
                         min_copies_override=args.min_copies_override)
    _print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
