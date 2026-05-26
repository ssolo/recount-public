"""Shared helpers for the reproduce/ml/map driver scripts.

The three top-level drivers — `validation/reproduce.py`, `validation/ml.py`,
`validation/map.py` — all share dataset loading, the BFGS log-space
parameterization, the reconstruction logic, and the warm-restart cycling
loop. This module centralises that.

Datasets
--------
Five datasets are supported via `DATASETS[label]`:
  - `dpann80`, `proteo75`, `eury114`, `ed194`: the four ML-fitted focal
    subsets from Csurös 2026 PNAS (Ωmin=4), bundled with his published
    rates in `arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz`.
  - `arc269`: the full 269-leaf phylogeny (Ωmin=1), loaded from the
    standalone newick + csv files. No published rates exist for this
    one — the paper does not ML-fit it (SI B.1 explicit).

Conventions
-----------
- `loss` is fixed at 1.0 everywhere (Csurös convention).
- `length[root] = +inf` (root has no edge).
- `dup[root]` is NOT optimized (held at default 0.5). This avoids the
  "ROOTLOSS not supported" NotImplementedError that the analytical
  L(0) gradient throws when dup[root] > ~0.5 at Ωmin ≥ 2.
- Optimization is in log-space: x = [log(gain) for all v,
  log(dup) for non-root v, log(length) for non-root v], size 3N-2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from recount.io.countxml import load_countxml
from recount.io.newick import read_newick
from recount.io.table import read_profile_table
from recount.ml import default_initial_rates, _gradient_raw_native
from recount.native_backend import (
    corrected_log_likelihood_native,
    per_branch_stats_native,
    unobserved_inside_tensors_native,
    unobserved_logL0_native,
)
from recount.rates import GLDRates
from recount.unobserved_outside import (
    _get_survival_arrays,
    compute_unobserved_outside,
    compute_unobserved_posteriors,
)


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------


# The Csurös data bundle is mirrored in the repo under docs/csuros_data/.
# Set RECOUNT_CSUROS_DATA to point at an external copy if you have one.
import os as _os
_DATA_ROOT = Path(_os.environ.get("RECOUNT_CSUROS_DATA", "docs/csuros_data"))
_SUBSETS_XML = _DATA_ROOT / "arc269" / "datasets-sims-ED194-E114-D80-P75.countxml.gz"


@dataclass
class DatasetSpec:
    label: str
    biology: str
    leaves: int
    min_copies: int
    # Either (source_xml, session_id, table_name) for subsets, or
    # (newick_path, table_path) for arc269.
    source: tuple


DATASETS: dict[str, DatasetSpec] = {
    "dpann80":  DatasetSpec("dpann80",  "DPANN superphylum (Nanobdellati)",       80, 4,
                            (_SUBSETS_XML, "dpann80-alti-gtdb", "dpann80-arcogm-min4.txt")),
    "proteo75": DatasetSpec("proteo75", "Proteoarchaea (TACK)",                   75, 4,
                            (_SUBSETS_XML, "proteo75-gtdb", "proteo75-arcogm-min4.txt")),
    "eury114":  DatasetSpec("eury114",  "Methanobacteriati / basal archaea",     114, 4,
                            (_SUBSETS_XML, "eury114-ba-gtdb", "eury114-arcogm-min4.txt")),
    "ed194":    DatasetSpec("ed194",    "Euryarchaeota (LECA)",                  194, 4,
                            (_SUBSETS_XML, "ed194-gtdb", "ed194-arcogm-min4.txt")),
    "arc269":   DatasetSpec("arc269",   "Full Archaea (LACA)",                   269, 1,
                            (Path("examples/data/arc269_tree.nwk"),
                             Path("examples/data/arc269_table.csv.gz"))),
    # Williams2017: bacterial/archaeal common ancestor dataset (LUCA),
    # 60-leaf tree with Ωmin=4 ortholog families.
    "williams": DatasetSpec("williams", "Williams2017 (LUCA, all 3 domains)",     60, 4,
                            (Path("validation/Williams2017.countxml.gz"),
                             "wsz60-codes-edit", "wsz60-aletrim-min4.txt")),
    # Coleman2021: 265-leaf bacterial tree (LBCA), Ωmin=1 (raw count tables
    # already filtered to families present at >= 1 leaf). Loaded from the
    # bare-tree + count-table pair in examples/data/, same format as arc269.
    "coleman":  DatasetSpec("coleman",  "Coleman2021 (LBCA, bacteria, 265 leaves)", 265, 1,
                            (Path("examples/data/coleman_tree.nwk"),
                             Path("examples/data/coleman_table.csv.gz"))),
}


def load_dataset(label: str, min_copies_override: int | None = None):
    """Returns (tree, profiles, csuros_rates_or_None, min_copies, biology).

    min_copies_override: if not None, overrides the dataset's default Ωmin.
    Used for sensitivity analyses (e.g. arc269 with Ωmin=4 to match Csurös'
    subset convention). Profiles themselves are NOT filtered — this
    affects the L(0) correction, which is what Ωmin actually parameterises
    in Csurös' framework (probability of observing ≥ Ωmin copies somewhere
    in the tree).
    """
    if label not in DATASETS:
        raise SystemExit(f"Unknown dataset {label!r}. Choose from {sorted(DATASETS)}.")
    spec = DATASETS[label]
    mc = min_copies_override if min_copies_override is not None else spec.min_copies
    # arc269 + coleman ship as (Newick, CSV table) pairs in examples/data/
    # rather than as a Csurös countxml session; they share the same loader.
    if label in ("arc269", "coleman"):
        nwk, tbl = spec.source
        tree, _, _ = read_newick(str(nwk))
        _, profs = read_profile_table(str(tbl), list(tree.leaf_names))
        return tree, profs.astype(np.int32), None, mc, spec.biology
    src, sid, tname = spec.source
    sess = load_countxml(str(src))[sid]
    profs = sess.tables[tname].profiles.astype(np.int32)
    csuros_rates = GLDRates(tree=sess.tree, gain=sess.rates.gain,
                            loss=sess.rates.loss, dup=sess.rates.dup,
                            length=sess.rates.length)
    return sess.tree, profs, csuros_rates, mc, spec.biology


# ----------------------------------------------------------------------------
# Reconstruction (LL + L(0) correction at root)
# ----------------------------------------------------------------------------


def reconstruct(tree, rates, profiles, min_copies) -> dict:
    """Compute LL, L(0), and L(0)-corrected root copies/families."""
    g, l, d, t = rates.gain, rates.loss, rates.dup, rates.length
    ll = corrected_log_likelihood_native(tree, g, l, d, t, profiles, min_copies=min_copies)
    L0 = float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies)))
    stats = per_branch_stats_native(tree, g, l, d, t, profiles)
    root = tree.root
    F = profiles.shape[0]
    # L(0) correction: only meaningful at Ωmin ≥ 2 (at Ωmin=1 the empty profile has ξ=0 at every node).
    if min_copies > 1 and L0 < 1.0:
        C, K, _ = unobserved_inside_tensors_native(tree, g, l, d, t, min_copies)
        sp = _get_survival_arrays(tree, g, l, d, t)
        B_all, J_all, _, _, _ = compute_unobserved_outside(tree, sp, K, min_copies)
        log_node_post, _ = compute_unobserved_posteriors(
            C, K, B_all, J_all, np.log(max(L0, 1e-300)), min_copies - 1)
        factor = F * L0 / (1.0 - L0)
        n_grid = np.arange(min_copies, dtype=np.float64)
        Pn_root = np.exp(log_node_post[root])
        root_copies_corr = float(stats["copies_node"][root] + factor * (Pn_root * n_grid).sum())
        root_families_corr = float(stats["families_present"][root] + factor * (1.0 - Pn_root[0]))
    else:
        root_copies_corr = float(stats["copies_node"][root])
        root_families_corr = float(stats["families_present"][root])
    return {
        "ll": float(ll), "L0": float(L0), "F": int(F), "min_copies": int(min_copies),
        "root_copies_obs": float(stats["copies_node"][root]),
        "root_copies_corr": root_copies_corr,
        "root_families_obs": float(stats["families_present"][root]),
        "root_families_corr": root_families_corr,
        "copies_per_family": root_copies_corr / max(root_families_corr, 1e-12),
    }


# ----------------------------------------------------------------------------
# Log-space parameterization
# ----------------------------------------------------------------------------
# x layout (size 3N-2):
#   [0:N]      = log(gain) at every node
#   [N:2N-1]   = log(dup) at non-root nodes
#   [2N-1:]    = log(length) at non-root nodes
# dup[root] is held at base; loss is held at 1.0 everywhere.


# Log-space rate clip: keeps rate in [exp(-50), exp(50)] ≈ [2e-22, 5e21].
# Wide enough to reach Csurös' published basins (his ML rates go up to
# 10^14 on some nodes, i.e. log ≈ 32, on the four focal subsets — clipping
# tighter than that would block us from finding the same ML basin as him).
# Still narrow enough to prevent the survival math from overflowing to inf
# when BFGS line search or a perturbed init pushes rates to extreme values.
LOG_RATE_CLIP = 50.0
# Sub-critical Yule constraint via Csurös' Java logit transform:
#     dup_v = MAX_PROB_NOT1 · loss_v · sigmoid(θ_v)
# with MAX_PROB_NOT1 = 1 - 2^-30 ≈ 0.99999999907 (his `ML.java:72`).
# Optimiser sees θ_v ∈ ℝ unbounded; dup_v asymptotes to MAX_PROB_NOT1
# without ever reaching it. Matches `MLDistribution.java:329`:
#     newLogistic(new DuplicationRate(node), MAX_PROB_NOT1 * loss_rate)
# Replaces the earlier hard-clip approach (`log(dup) ≤ 0`) which produced
# a kinked objective and required the `_zero_grad_at_dup_cap` workaround.
MAX_PROB_NOT1 = 1.0 - 2.0 ** (-30)
# Soft inner clip for the logit's optimiser variable θ. With MAX = 1 - 2^-30
# and θ_safe = 30, dup = MAX · sigmoid(30) ≈ MAX · (1 - 9.4e-14) — i.e. the
# logit is effectively saturated at θ = ±LOGIT_THETA_CLIP. We need this
# only because scipy BFGS line search can otherwise propose huge θ values
# that produce subnormal sigmoid evaluations.
LOGIT_THETA_CLIP = 30.0
# Gain rate cap. Csurös' `MLDistribution.java:71` sets
#     MAX_GAIN_RATE = 33.0
# nominally as a Logistic(GainParameter, 33) max-value, but his empirical
# Gain parameter cap — Csurös' ML.java `Logistic(GainParameter, MAX_GAIN_RATE=33)`
# wraps the gain DISTRIBUTION PARAMETER (κ for Pólya, r for Poisson) in a
# bounded logit transform so the optimiser sees θ ∈ ℝ with
#     gain_v = MAX_GAIN_PARAMETER · sigmoid(θ_v)
# making gain_v ∈ (0, MAX_GAIN_PARAMETER). Our `gain[v]` IS this same κ
# (confirmed by bit-perfect LL match at Csurös' published rates), so the
# constraint maps directly onto our `gain` block. This matches Java's
# `Logistic.set(x)` which clips t = sigmoid(x) to [EPS, 1-EPS] before
# scaling by max_value.
MAX_GAIN_PARAMETER = 33.0


def _logit_to_dup(theta: np.ndarray, max_prob: float = MAX_PROB_NOT1) -> np.ndarray:
    """Csurös' Java Logistic.value: θ ∈ ℝ → dup ∈ (0, max_prob)."""
    th = np.clip(theta, -LOGIT_THETA_CLIP, LOGIT_THETA_CLIP)
    return max_prob / (1.0 + np.exp(-th))


def _dup_to_logit(dup: np.ndarray, max_prob: float = MAX_PROB_NOT1) -> np.ndarray:
    """Inverse of ``_logit_to_dup``. Maps dup ∈ (0, max_prob) → θ ∈ ℝ.

    Inputs at or above ``max_prob`` are clipped just inside the open
    interval before taking the log so θ stays finite.
    """
    eps = 1e-15  # keep dup strictly inside (0, max_prob)
    d_safe = np.clip(dup, eps * max_prob, max_prob * (1.0 - eps))
    return np.log(d_safe / (max_prob - d_safe))


def _dup_jacobian(dup: np.ndarray, max_prob: float = MAX_PROB_NOT1) -> np.ndarray:
    """∂ dup_v / ∂ θ_v for the logit transform.

    With dup = max · σ(θ): ∂dup/∂θ = max · σ(θ)(1 - σ(θ))
                                    = dup · (max - dup) / max
                                    = dup · (1 - dup/max).
    """
    return dup * (1.0 - dup / max_prob)


def _logit_to_gain(theta: np.ndarray, max_gain: float | None = None) -> np.ndarray:
    """Java Logistic.value for gain: θ ∈ ℝ → gain ∈ (0, max_gain).

    Bit-faithful to ML.java Logistic.set: t = sigmoid(x) clipped to
    [EPS, 1-EPS], gain = max_gain·t. ``max_gain=None`` reads the current
    module-level ``MAX_GAIN_PARAMETER`` at call time (so runtime overrides
    propagate).
    """
    if max_gain is None:
        max_gain = MAX_GAIN_PARAMETER
    th = np.clip(theta, -LOGIT_THETA_CLIP, LOGIT_THETA_CLIP)
    return max_gain / (1.0 + np.exp(-th))


def _gain_to_logit(gain: np.ndarray, max_gain: float | None = None) -> np.ndarray:
    """Inverse: gain ∈ (0, max_gain) → θ ∈ ℝ."""
    if max_gain is None:
        max_gain = MAX_GAIN_PARAMETER
    eps = 1e-15
    g_safe = np.clip(gain, eps * max_gain, max_gain * (1.0 - eps))
    return np.log(g_safe / (max_gain - g_safe))


def _gain_jacobian(gain: np.ndarray, max_gain: float | None = None) -> np.ndarray:
    """∂ gain_v / ∂ θ_v for the gain Logistic transform.

    Same form as Java Logistic.dL: gain · (1 - gain/max_gain).
    """
    if max_gain is None:
        max_gain = MAX_GAIN_PARAMETER
    return gain * (1.0 - gain / max_gain)


# Backwards-compat alias (older code paths still reference this); the
# value is no longer used now that gain has a proper Logistic transform.
LOG_GAIN_CLIP_SUBCRITICAL = np.log(MAX_GAIN_PARAMETER)


def rates_to_x(rates: GLDRates, root: int, subcritical: bool = True) -> np.ndarray:
    """Encode rates as the optimiser's unconstrained vector.

    With ``subcritical=True`` (default; matches Csurös' publication behaviour):
        * dup block uses the Java Logistic transform — dup ∈ (0, MAX_PROB_NOT1),
          matching the SI's sub-critical Yule constraint.
        * gain block is in plain log-space (no Logistic cap). The Java code
          declares ``Logistic(GainParameter, MAX_GAIN_RATE=33)`` but Csurös'
          published rates routinely violate this cap (κ up to 4.9×10¹³ on
          dpann80; 34 of 158 non-root nodes). See GAIN_CONSTRAINT.md for
          evidence and discussion. To match his publication regime we leave
          gain unconstrained on the upper side; only an outer LOG_RATE_CLIP
          guards against floating-point overflow.
    With ``subcritical=False`` BOTH dup and gain are in plain log-space.
    """
    N = rates.gain.shape[0]
    mask = np.ones(N, dtype=bool); mask[root] = False
    gain_block = np.clip(np.log(np.clip(rates.gain, 1e-300, np.inf)),
                         -LOG_RATE_CLIP, LOG_RATE_CLIP)
    if subcritical:
        dup_block = _dup_to_logit(rates.dup[mask])
    else:
        dup_block = np.clip(np.log(np.clip(rates.dup[mask], 1e-300, np.inf)),
                            -LOG_RATE_CLIP, LOG_RATE_CLIP)
    return np.concatenate([
        gain_block,
        dup_block,
        np.clip(np.log(np.clip(rates.length[mask], 1e-300, np.inf)), -LOG_RATE_CLIP, LOG_RATE_CLIP),
    ])


def x_to_rates(x: np.ndarray, base: GLDRates, root: int, subcritical: bool = True) -> GLDRates:
    """Unpack x into a GLDRates.

    With ``subcritical=True`` (default, matching Csurös' publication regime):
        * dup block is in logit space:
              ``dup_v = MAX_PROB_NOT1 · sigmoid(θ_v)`` → dup ∈ (0, 1 - 2⁻³⁰).
        * gain block is in plain log-space (no upper cap; see GAIN_CONSTRAINT.md).
    With ``subcritical=False`` both dup and gain are plain ``log(rate)``.
    """
    N = base.gain.shape[0]
    mask = np.ones(N, dtype=bool); mask[root] = False
    x_safe_gain   = np.clip(x[:N], -LOG_RATE_CLIP, LOG_RATE_CLIP)
    x_dup_block   = x[N:2*N - 1]
    x_safe_length = np.clip(x[2*N - 1:], -LOG_RATE_CLIP, LOG_RATE_CLIP)
    dup = base.dup.copy()
    length = base.length.copy()
    gain = np.exp(x_safe_gain)
    if subcritical:
        dup[mask] = _logit_to_dup(x_dup_block)
    else:
        dup[mask] = np.exp(np.clip(x_dup_block, -LOG_RATE_CLIP, LOG_RATE_CLIP))
    length[mask] = np.exp(x_safe_length)
    return GLDRates(tree=base.tree, gain=gain, loss=base.loss.copy(),
                    dup=dup, length=length)


def make_bounds(N: int, subcritical: bool = True) -> list[tuple[float, float]]:
    """Per-parameter bounds aligned with ``rates_to_x`` for L-BFGS-B users.

    With ``subcritical=True``: dup in logit space (±LOGIT_THETA_CLIP), gain
    and length in plain log space (±LOG_RATE_CLIP).
    With ``subcritical=False``: all three blocks are ±LOG_RATE_CLIP.
    """
    bounds: list[tuple[float, float]] = []
    bounds += [(-LOG_RATE_CLIP, LOG_RATE_CLIP)] * N                   # gain (log)
    if subcritical:
        bounds += [(-LOGIT_THETA_CLIP, LOGIT_THETA_CLIP)] * (N - 1)    # dup (logit)
    else:
        bounds += [(-LOG_RATE_CLIP, LOG_RATE_CLIP)] * (N - 1)
    bounds += [(-LOG_RATE_CLIP, LOG_RATE_CLIP)] * (N - 1)
    return bounds


# ----------------------------------------------------------------------------
# Objective + gradient
# ----------------------------------------------------------------------------


def make_objgrad_ml(tree, base, profiles, min_copies,
                    subcritical: bool = True) -> Callable[[np.ndarray], tuple]:
    """Negative log-likelihood + analytical gradient (no prior).

    With ``subcritical=True`` (default): dup is in Java-Logistic space
    (dup ≤ MAX_PROB_NOT1), gain is in plain log-space (no upper cap; matches
    Csurös' publication regime — see GAIN_CONSTRAINT.md). Chain rules:
        ∂(-LL)/∂(log gain) = -gg · gain
        ∂(-LL)/∂θ_dup      = -gd · dup · (1 - dup/MAX_PROB_NOT1)
    Length stays in log-space.
    """
    root = tree.root; N = tree.num_nodes
    nonroot = np.ones(N, dtype=bool); nonroot[root] = False
    _SAFE_LARGE = 1.0e30   # sentinel for line search to backtrack
    def f_and_g(x):
        rates = x_to_rates(x, base, root, subcritical=subcritical)
        try:
            LL, gg, _, gd, gt = _gradient_raw_native(tree, rates, profiles, min_copies=min_copies)
        except Exception:
            return _SAFE_LARGE, np.zeros_like(x)
        if not np.isfinite(LL):
            return _SAFE_LARGE, np.zeros_like(x)
        out = np.empty_like(x)
        out[:N]      = -(rates.gain * gg)
        if subcritical:
            dup_jac = _dup_jacobian(rates.dup[nonroot])
            out[N:2*N-1] = -(dup_jac * gd[nonroot])
        else:
            out[N:2*N-1] = -(rates.dup[nonroot] * gd[nonroot])
        out[2*N-1:]  = -(rates.length[nonroot] * gt[nonroot])
        if not np.all(np.isfinite(out)):
            # Non-finite gradient: zero it so lnsrch backtracks (rather than
            # pushing the optimizer further into the degenerate region).
            out[~np.isfinite(out)] = 0.0
        return -LL, out
    return f_and_g


def make_objgrad_map(tree, base, profiles, min_copies,
                     mu_gain: float, mu_dup: float, mu_length: float, sigma: float,
                     subcritical: bool = True) -> Callable[[np.ndarray], tuple]:
    """Negative log-posterior (-LL - log_prior) + gradient.

    With ``subcritical=True``: dup is θ ∈ ℝ via logit (prior on θ),
    gain is log γ ∈ ℝ (prior on log γ). Length is log t (prior on log t).
    """
    root = tree.root; N = tree.num_nodes
    nonroot = np.ones(N, dtype=bool); nonroot[root] = False
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    inv_sig2 = 1.0 / (sigma * sigma)
    _SAFE_LARGE = 1.0e30
    def f_and_g(x):
        rates = x_to_rates(x, base, root, subcritical=subcritical)
        try:
            LL, gg, _, gd, gt = _gradient_raw_native(tree, rates, profiles, min_copies=min_copies)
        except Exception:
            return _SAFE_LARGE, np.zeros_like(x)
        if not np.isfinite(LL):
            return _SAFE_LARGE, np.zeros_like(x)
        log_g = x[:N]; theta_d = x[N:2*N-1]; log_t = x[2*N-1:]
        dev_g = log_g - mu_gain
        dev_d = theta_d - mu_dup       # in logit-space when subcritical=True
        dev_t = log_t - mu_length
        logp = - inv2sig2 * (dev_g @ dev_g + dev_d @ dev_d + dev_t @ dev_t)
        prior_g_g = - dev_g * inv_sig2
        prior_g_d = - dev_d * inv_sig2
        prior_g_t = - dev_t * inv_sig2
        out = np.empty_like(x)
        out[:N]      = -(rates.gain * gg + prior_g_g)
        if subcritical:
            dup_jac = _dup_jacobian(rates.dup[nonroot])
            out[N:2*N-1] = -(dup_jac * gd[nonroot] + prior_g_d)
        else:
            out[N:2*N-1] = -(rates.dup[nonroot] * gd[nonroot] + prior_g_d)
        out[2*N-1:]  = -(rates.length[nonroot] * gt[nonroot] + prior_g_t)
        if not np.all(np.isfinite(out)):
            out[~np.isfinite(out)] = 0.0
        return -(LL + logp), out
    return f_and_g


# ----------------------------------------------------------------------------
# Tree-Brownian (autocorrelated log-rate) prior
# ----------------------------------------------------------------------------
#
# Replaces the per-node independent log-Normal prior with a per-EDGE
# Gaussian on the log-rate increment, adapted from Thorne-Kishino-Painter
# 1998. For each non-root node v with parent pa(v) and each rate axis:
#
#     x_block[v] − x_block[pa(v)] ~ N(0, σ²_block)
#
# v1 uses time-uniform variance — σ² alone, no per-edge t_v weighting.
# The TKP-classic σ²·t_v form is a v2 enhancement (introduces non-trivial
# gradient w.r.t. log_length since t_v depends on it).
#
# For the root, each axis gets a wide independent N(mu_root, sigma_root²)
# anchor to keep the absolute scale identifiable. With dup/length whose
# root values are NOT in the optimiser, the root "value" used for
# children's Brownian increment is just the prior centre mu_root_*
# (a fixed reference, not a free parameter).


def _brownian_prior_and_grad_python(
    x: np.ndarray,
    tree,
    sigma_brownian_gain: float,
    sigma_brownian_dup: float,
    sigma_brownian_length: float,
    mu_root_gain: float,
    mu_root_dup: float,
    mu_root_length: float,
    sigma_root: float = 5.0,
    subcritical: bool = True,
) -> tuple[float, np.ndarray]:
    """Pure-Python reference implementation. Kept for FD-validation and
    fallback if the native backend is unavailable. See `_brownian_prior_and_grad`
    for the public dispatcher (native by default, ~30× faster).
    """
    N = tree.num_nodes
    root = tree.root
    parent = tree.parent

    # Non-root index lookup: each non-root v has position nonroot_to_pos[v]
    # in the dup/length blocks of x.
    nonroot_mask = np.ones(N, dtype=bool); nonroot_mask[root] = False
    nonroot_to_pos = np.full(N, -1, dtype=np.int64)
    nonroot_to_pos[nonroot_mask] = np.arange(N - 1)
    nonroot_idx = np.where(nonroot_mask)[0]            # length N-1 (node indices)
    pa_idx = parent[nonroot_idx]                       # length N-1 (parent node indices)
    pa_is_nonroot = pa_idx != root                     # True for "internal" edges

    # Build per-axis full-tree value arrays (indexed by node)
    val_gain_full = x[:N]                              # root IS in x for gain
    val_dup_full = np.empty(N)
    val_dup_full[nonroot_mask] = x[N:2 * N - 1]
    val_dup_full[root] = mu_root_dup                   # root dup centred at prior mean (not in x)
    val_length_full = np.empty(N)
    val_length_full[nonroot_mask] = x[2 * N - 1:]
    val_length_full[root] = mu_root_length             # root length centred at prior mean

    grad = np.zeros_like(x)
    log_prior = 0.0

    # --- Edge terms (Brownian: child - parent) per axis ---
    for axis, val, sigma, x_offset, x_in_root in [
        ("gain",   val_gain_full,   sigma_brownian_gain,   0,           True),
        ("dup",    val_dup_full,    sigma_brownian_dup,    N,           False),
        ("length", val_length_full, sigma_brownian_length, 2 * N - 1,   False),
    ]:
        inv_sig2 = 1.0 / (sigma * sigma)
        diff = val[nonroot_idx] - val[pa_idx]          # length N-1
        log_prior -= 0.5 * inv_sig2 * float(np.dot(diff, diff))

        # ∂L/∂val[v]  = -inv_sig2 · diff[i]   for each child v = nonroot_idx[i]
        # ∂L/∂val[pa] = +inv_sig2 · diff[i]   for each parent pa = pa_idx[i]
        if x_in_root:
            # gain block: positions are nonroot_idx (child) and pa_idx (parent),
            # both directly in the gain block since root is also in there.
            np.add.at(grad, x_offset + nonroot_idx, -inv_sig2 * diff)
            np.add.at(grad, x_offset + pa_idx,      +inv_sig2 * diff)
        else:
            # dup / length blocks: only non-root nodes have positions.
            # Child position always exists; parent position only if pa is non-root.
            np.add.at(grad, x_offset + nonroot_to_pos[nonroot_idx], -inv_sig2 * diff)
            if pa_is_nonroot.any():
                np.add.at(grad, x_offset + nonroot_to_pos[pa_idx[pa_is_nonroot]],
                          +inv_sig2 * diff[pa_is_nonroot])

    # --- Root anchor: independent N(mu_root_gain, sigma_root²) on gain root ---
    # (dup/length root values are constants — no anchor needed in x)
    inv_sig2_root = 1.0 / (sigma_root * sigma_root)
    dev_g_root = float(val_gain_full[root] - mu_root_gain)
    log_prior -= 0.5 * inv_sig2_root * (dev_g_root ** 2)
    grad[root] += -inv_sig2_root * dev_g_root          # gain root position

    return log_prior, grad


def _brownian_prior_and_grad(
    x: np.ndarray,
    tree,
    sigma_brownian_gain: float,
    sigma_brownian_dup: float,
    sigma_brownian_length: float,
    mu_root_gain: float,
    mu_root_dup: float,
    mu_root_length: float,
    sigma_root: float = 5.0,
    subcritical: bool = True,
) -> tuple[float, np.ndarray]:
    """Tree-Brownian autocorrelated log-rate prior, dispatcher.

    Native C O(N) implementation by default; falls back to the pure-Python
    reference if the native backend isn't built. See docs/brownian_prior.tex.

    Returns (log_prior, grad) with grad in the packed shape of
    ``rates_to_x``. Sign convention: ``log_prior`` is the log-density;
    ``grad`` is ∂(log_prior)/∂x.
    """
    try:
        from recount.native_backend import brownian_prior_native
        parent = np.ascontiguousarray(tree.parent, dtype=np.int32)
        return brownian_prior_native(
            parent=parent,
            root=int(tree.root),
            x=x,
            sigma_brownian_gain=sigma_brownian_gain,
            sigma_brownian_dup=sigma_brownian_dup,
            sigma_brownian_length=sigma_brownian_length,
            mu_root_gain=mu_root_gain,
            mu_root_dup=mu_root_dup,
            mu_root_length=mu_root_length,
            sigma_root=sigma_root,
        )
    except (ImportError, AttributeError, OSError):
        # Native unavailable — use the pure-Python reference.
        return _brownian_prior_and_grad_python(
            x, tree,
            sigma_brownian_gain, sigma_brownian_dup, sigma_brownian_length,
            mu_root_gain, mu_root_dup, mu_root_length,
            sigma_root=sigma_root, subcritical=subcritical,
        )


def make_objgrad_map_brownian(
    tree, base, profiles, min_copies,
    sigma_brownian_gain: float,
    sigma_brownian_dup: float,
    sigma_brownian_length: float,
    mu_root_gain: float = float(np.log(0.1)),
    mu_root_dup: float = 0.0,            # logit(0.5) when subcritical=True
    mu_root_length: float = 0.0,         # log(1.0)
    sigma_root: float = 5.0,
    subcritical: bool = True,
) -> Callable[[np.ndarray], tuple]:
    """Negative log-posterior with tree-Brownian autocorrelated log-rate prior.

    Replacement for ``make_objgrad_map`` that uses tree topology to share
    statistical strength across neighbouring branches. For each non-root v
    and each axis (gain, dup-in-logit-space-when-subcritical, length):

        log_rate_v − log_rate_pa(v) ~ N(0, σ²_axis)

    Per-edge Gaussian. Small σ → strong shrinkage (child pulled toward
    parent). σ → ∞ → independent (equivalent to pure ML at the limit).

    Root values: gain root is in the optimiser and gets an independent
    N(mu_root_gain, sigma_root²) anchor. dup/length root values are fixed
    (treated as the prior centres mu_root_dup / mu_root_length), so the
    children of root use those constants as the parent value in the
    Brownian increment.
    """
    root = tree.root; N = tree.num_nodes
    nonroot = np.ones(N, dtype=bool); nonroot[root] = False
    _SAFE_LARGE = 1.0e30

    def f_and_g(x):
        rates = x_to_rates(x, base, root, subcritical=subcritical)
        try:
            LL, gg, _, gd, gt = _gradient_raw_native(tree, rates, profiles, min_copies=min_copies)
        except Exception:
            return _SAFE_LARGE, np.zeros_like(x)
        if not np.isfinite(LL):
            return _SAFE_LARGE, np.zeros_like(x)

        log_prior, prior_grad = _brownian_prior_and_grad(
            x, tree,
            sigma_brownian_gain, sigma_brownian_dup, sigma_brownian_length,
            mu_root_gain, mu_root_dup, mu_root_length,
            sigma_root=sigma_root, subcritical=subcritical,
        )

        # Likelihood gradient (same chain rule as make_objgrad_ml)
        out = np.empty_like(x)
        out[:N] = -(rates.gain * gg)
        if subcritical:
            dup_jac = _dup_jacobian(rates.dup[nonroot])
            out[N:2 * N - 1] = -(dup_jac * gd[nonroot])
        else:
            out[N:2 * N - 1] = -(rates.dup[nonroot] * gd[nonroot])
        out[2 * N - 1:] = -(rates.length[nonroot] * gt[nonroot])

        # Subtract prior gradient: minimise -(LL + log_prior)
        # out is already -∂LL/∂x; -∂(LL+log_prior)/∂x = -∂LL/∂x - ∂log_prior/∂x
        out -= prior_grad
        if not np.all(np.isfinite(out)):
            out[~np.isfinite(out)] = 0.0
        return -(LL + log_prior), out
    return f_and_g


# ----------------------------------------------------------------------------
# Warm-restart BFGS cycling
# ----------------------------------------------------------------------------


def fit_bfgs_cycles(objgrad_factory: Callable, tree, init_rates: GLDRates,
                    cycle_iters: int = 100, max_cycles: int = 12,
                    gtol: float = 1e-7, label: str = "fit",
                    early_stop_iters_zero: int = 2,
                    optimizer: str = "BFGS",
                    subcritical: bool = True) -> tuple[GLDRates, float, list[dict]]:
    """Warm-restart optimization with re-launch.

    Each cycle runs the chosen scipy `method` for up to `cycle_iters` iters,
    then re-launches from current rates (clearing the Hessian estimate).
    Stops when `early_stop_iters_zero` consecutive cycles produce iters=0.

    When ``subcritical=True`` (default; matching Csurös' Java
    ``is_duprate_bounded=true``), the optimizer is switched to L-BFGS-B
    with per-parameter bounds enforcing ``log(dup) ≤ 0`` (i.e. dup ≤ 1).
    BFGS / trust-constr / csuros_dfpmin paths are only used when
    ``subcritical=False`` for backward compatibility.

    Supported optimizers (only when ``subcritical=False``):
        "BFGS" — scipy's dense-Hessian BFGS with Wolfe line search (default).
        "trust-constr" — scipy's trust-region method.
        "csuros_dfpmin" — Python port of Csurös' Java dfpmin.

    Returns: (best_rates, best_neg_obj, trajectory).
    """
    import time
    root = tree.root
    rates = init_rates
    objgrad = objgrad_factory(rates)
    best_obj = np.inf
    best_x = rates_to_x(rates, root, subcritical=subcritical)
    trajectory = []
    consecutive_zero = 0
    t0 = time.time()
    # Note on sub-critical handling: when subcritical=True, the dup
    # constraint dup ≤ 1 is enforced by an asymmetric clip in
    # `x_to_rates`/`rates_to_x` (log(dup) ≤ 0). BFGS still runs unbounded;
    # past the clip the objective is locally flat in those directions, so
    # the line search converges in the constrained subspace. (We tried
    # scipy's L-BFGS-B with explicit bounds — it gets stuck on the Cauchy
    # projection because the GLD objective's Hessian is too ill-conditioned
    # at the initial point. The asymmetric-clip-with-BFGS approach matches
    # what we already do for the symmetric ±LOG_RATE_CLIP outer clip.)
    for cyc in range(max_cycles):
        x0 = rates_to_x(rates, root, subcritical=subcritical)
        if optimizer == "trust-constr":
            r = minimize(objgrad, x0, jac=True, method="trust-constr",
                         options={"maxiter": cycle_iters, "gtol": gtol,
                                  "xtol": 1e-10, "disp": False, "verbose": 0})
            fun, x_final, nit, nfev = float(r.fun), r.x.copy(), int(r.nit), int(getattr(r, "nfev", 0))
            try:
                grad_inf = float(np.max(np.abs(r.jac))) if (r.jac is not None and len(r.jac) > 0) else float("nan")
            except (TypeError, ValueError):
                grad_inf = float("nan")
        elif optimizer == "csuros_dfpmin":
            from validation.csuros_dfpmin import dfpmin
            nfev_counter = [0]
            # When x[i] hits the clip boundary, the gradient component in
            # that direction is mathematically zero (the function is
            # constant outside [-LOG_RATE_CLIP, LOG_RATE_CLIP]). Returning
            # the kernel's gradient at the clipped rates would mislead
            # dfpmin into stepping toward the clip and stalling at alamin.
            CLIP_GUARD = 0.999 * LOG_RATE_CLIP  # small margin
            def f_only(x):
                nfev_counter[0] += 1
                return objgrad(x)[0]
            def g_only(x):
                g = objgrad(x)[1].copy()
                # Zero out gradient on any component sitting at/near the clip
                # (where the function is locally flat in that direction).
                mask = np.abs(x) >= CLIP_GUARD
                g[mask] = 0.0
                return g
            x_work = x0.copy()
            fmin, nit, status = dfpmin(x_work, gtol=gtol, func=f_only, gradient=g_only,
                                       dfp_itmax=cycle_iters)
            fun, x_final, nfev = fmin, x_work, nfev_counter[0]
            grad_inf = float(np.max(np.abs(g_only(x_final))))
        elif optimizer == "native_bfgs":
            # Native C BFGS — bit-faithful port of Csurös' Java dfpmin.
            # Pure C bookkeeping (Hessian update + line search) + libdispatch-
            # parallel gradient via objgrad. Much faster than pure-Python
            # csuros_dfpmin and scipy BFGS on the same workload.
            from recount.native_backend import bfgs_native
            nfev_counter = [0]
            def og_native(x):
                nfev_counter[0] += 1
                return objgrad(x)
            x_work, fmin, nit, status_code = bfgs_native(
                x0, og_native, gtol=gtol, max_iters=cycle_iters)
            fun, x_final, nfev = fmin, x_work, nfev_counter[0]
            grad_inf = float(np.max(np.abs(objgrad(x_final)[1])))
        else:
            r = minimize(objgrad, x0, jac=True, method="BFGS",
                         options={"maxiter": cycle_iters, "gtol": gtol, "disp": False})
            fun, x_final, nit, nfev = float(r.fun), r.x.copy(), int(r.nit), int(getattr(r, "nfev", 0))
            try:
                grad_inf = float(np.max(np.abs(r.jac))) if (r.jac is not None and len(r.jac) > 0) else float("nan")
            except (TypeError, ValueError):
                grad_inf = float("nan")
        wall = time.time() - t0
        print(f"    [{label}] cyc {cyc+1:2d}: obj={fun:.4f}  iters={nit}  "
              f"|g|={grad_inf:.2e}  wall={wall:.0f}s", flush=True)
        trajectory.append({"cycle": cyc+1, "obj": fun, "grad_inf": grad_inf,
                           "iters": nit, "nfev": nfev, "wall_s": wall})
        if fun < best_obj:
            best_obj = fun; best_x = x_final.copy()
        if nit == 0:
            consecutive_zero += 1
            if consecutive_zero >= early_stop_iters_zero:
                print(f"    [{label}] early-stop: {consecutive_zero} consecutive cycles "
                      f"with iters=0", flush=True)
                break
        else:
            consecutive_zero = 0
        # Always restart from the BEST-seen x so non-monotone single-cycle
        # behaviour (line search overshoots near boundary) doesn't degrade
        # the warm-restart sequence. Java's MLDistribution.optimize tracks
        # xbest the same way.
        rates = x_to_rates(best_x, init_rates, root, subcritical=subcritical)
        objgrad = objgrad_factory(rates)
    return x_to_rates(best_x, init_rates, root, subcritical=subcritical), best_obj, trajectory


def perturb_init(base: GLDRates, sigma: float, rng: np.random.Generator,
                 max_log_dev: float = 5.0,
                 subcritical: bool = True) -> GLDRates:
    """Multiplicative log-Normal perturbation, with per-rate log-deviation clipped.

    Without the clip a single tail-sample from lognormal(0, sigma=0.5) can push
    one node's rate by ×10⁵, which the survival math then NaNs on. Clipping the
    log-deviation to ±5 (≈ rate × {0.0067, 150}) keeps the perturbation bounded
    while still letting multistart explore many basins. Root length stays +inf.

    With ``subcritical=True`` (default), the perturbed dup rates are
    additionally clipped to ≤ 1 so the perturbation respects the
    sub-critical Yule constraint and the L-BFGS-B initial point lies
    within the bounds.
    """
    def lognorm_clipped(shape):
        dev = rng.normal(0, sigma, size=shape)
        return np.exp(np.clip(dev, -max_log_dev, max_log_dev))
    dup = base.dup * lognorm_clipped(base.dup.shape)
    if subcritical:
        dup = np.minimum(dup, 1.0)
    return GLDRates(
        tree=base.tree, loss=base.loss.copy(),
        gain=base.gain * lognorm_clipped(base.gain.shape),
        dup=dup,
        length=np.where(np.isinf(base.length), base.length,
                        base.length * lognorm_clipped(base.length.shape)),
    )


# ----------------------------------------------------------------------------
# Output helpers — countxml + branches.csv
# ----------------------------------------------------------------------------


def _get_names_for_dataset(label: str, tree) -> tuple[list[str], list[str]]:
    """Return (leaf_names, internal_names) for a dataset."""
    if label == "arc269":
        # arc269 was loaded from newick — newick provides internal names too.
        from recount.io.newick import read_newick
        spec = DATASETS[label]
        _, _, internal_names = read_newick(str(spec.source[0]))
        leaf_names = list(tree.leaf_names) if tree.leaf_names else []
        return leaf_names, internal_names
    # Subsets came from countxml — the session has names in tree.leaf_names and tree.internal_names
    leaf_names = list(tree.leaf_names) if tree.leaf_names else []
    internal_names = list(getattr(tree, "internal_names", []))
    if not internal_names:
        internal_names = [f"node{v}" if not tree.is_leaf[v] else
                          (leaf_names[v] if v < len(leaf_names) else f"leaf{v}")
                          for v in range(tree.num_nodes)]
    return leaf_names, internal_names


def write_outputs(label: str, tree, rates: GLDRates, profiles: np.ndarray,
                  min_copies: int, out_prefix: str | Path) -> None:
    """Write {out_prefix}.{countxml.gz,branches.csv} for a fitted/loaded result.

    - countxml.gz: Java-compatible session XML (loadable by CountXXV.jar).
    - branches.csv: per-node summary with rates, copies (observed + L(0)-
      corrected), gain/loss events, and families-present.
    """
    from recount.io.countxml_writer import write_countxml
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    leaf_names, internal_names = _get_names_for_dataset(label, tree)

    # Build family_names from index if not available
    F = profiles.shape[0]
    family_names = [f"fam{i:06d}" for i in range(F)]

    # 1) countxml
    countxml_path = Path(f"{out_prefix}.countxml.gz")
    write_countxml(
        countxml_path, tree, rates, family_names, profiles,
        internal_names=internal_names, session_id=f"recount-{label}",
        table_name=f"{label}-min{min_copies}.txt",
    )

    # 2) branches.csv  (per-node summary including L(0)-corrected root values)
    g, l, d, t = rates.gain, rates.loss, rates.dup, rates.length
    stats = per_branch_stats_native(tree, g, l, d, t, profiles)
    # L(0) correction at the root (per the same logic as reconstruct())
    L0 = float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies)))
    if min_copies > 1 and L0 < 1.0:
        C, K, _ = unobserved_inside_tensors_native(tree, g, l, d, t, min_copies)
        sp = _get_survival_arrays(tree, g, l, d, t)
        B_all, J_all, _, _, _ = compute_unobserved_outside(tree, sp, K, min_copies)
        log_node_post, _ = compute_unobserved_posteriors(
            C, K, B_all, J_all, np.log(max(L0, 1e-300)), min_copies - 1)
        factor = F * L0 / (1.0 - L0)
        n_grid = np.arange(min_copies, dtype=np.float64)
        Pn = np.exp(log_node_post)
        copies_corr = stats["copies_node"] + factor * (Pn @ n_grid)
        present_corr = stats["families_present"] + factor * (1.0 - Pn[:, 0])
    else:
        copies_corr = stats["copies_node"].copy()
        present_corr = stats["families_present"].copy()

    branches_path = Path(f"{out_prefix}.branches.csv")
    with open(branches_path, "w") as fh:
        fh.write("node_index,node_name,is_leaf,parent_index,parent_name,branch_length,"
                 "gain_rate,loss_rate,dup_rate,"
                 "copies_node_observed,copies_node_corrected,copies_edge_observed,"
                 "gain_events,loss_events,"
                 "families_present_observed,families_present_corrected,"
                 "families_active_thresh50\n")
        for v in range(tree.num_nodes):
            is_leaf = bool(tree.is_leaf[v])
            p = int(tree.parent[v])
            nm = (internal_names[v] if internal_names and internal_names[v]
                  else (leaf_names[v] if is_leaf and v < len(leaf_names) else f"node{v}"))
            pnm = (internal_names[p] if (p >= 0 and internal_names and internal_names[p])
                   else (f"node{p}" if p >= 0 else ""))
            length = rates.length[v]
            length_s = "inf" if np.isinf(length) else f"{length:.6g}"
            fh.write(f"{v},{nm},{is_leaf},{p if p >= 0 else ''},{pnm},{length_s},"
                     f"{rates.gain[v]:.6g},{rates.loss[v]:.6g},{rates.dup[v]:.6g},"
                     f"{stats['copies_node'][v]:.6f},{copies_corr[v]:.6f},"
                     f"{stats['copies_edge'][v]:.6f},"
                     f"{stats['gain_events'][v]:.6f},{stats['loss_events'][v]:.6f},"
                     f"{stats['families_present'][v]:.6f},{present_corr[v]:.6f},"
                     f"{int(stats['num_families_active'][v])}\n")
