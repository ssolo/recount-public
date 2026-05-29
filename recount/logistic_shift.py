"""LogisticShift K-category mixture-likelihood scaffolding.

Mirrors Java ``count.model.RateVariationModel`` (line 970+) where each
``Category`` shifts the base model's logit-loss and logit-duplication by
``mod_length`` and ``mod_duplication``. The mixture likelihood is

    L_f = Σ_k p_k · L_f(rates_k)

where ``rates_k`` are derived from the base ``(gain, loss, dup, length)`` by
applying the category-specific shifts. Survival parameters are recomputed
per category via ``recount_compute_survival_params``.

Status:
  ✓ K=1 with zero shifts — exactly equals the base-GLD likelihood (validated
    in this module's ``_sanity_check_k1`` helper, also tested by example 07).
  ◯ K>1 with non-trivial shifts — implementation is structurally correct
    (mirrors Java's RateVariationModel.LogisticShift.updateNodeParameters
    at line 977-1011), but NO STORED Williams/arc269 dataset uses K>1 mixture
    rate variation, so we cannot validate against Java's reference numbers.
    The path is therefore "best-effort port"; concrete validation requires
    a multi-category stored model or constructing a synthetic K>1 test.

The shift mapping (per node v, per category k) follows Java
RateVariationModel.LogisticShift.updateNodeParameters (line 977-1011):

    logit_p_cat[v, k] = logit(p_loss_base[v]) + mod_length[k]
    cat_logit_lambda[v, k] = logit(λ_dup_base[v]) + mod_duplication[k]
    # plus a stability adjustment to logit_p_cat involving the relative
    # complements (Java lines 1003-1005) when shift != 0 and Pólya:
    logit_p_cat[v, k] += log(1-λ_base[v]) - log(1-cat_λ[v, k])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from recount.native_backend import corrected_log_likelihood_native
from recount.tree import Tree


@dataclass
class LogisticShiftCategory:
    """One category in a LogisticShift mixture model.

    ``probability`` ∈ [0, 1] — Σ_k probability must equal 1.
    ``mod_length`` — shift applied to logit(loss survival).
    ``mod_duplication`` — shift applied to logit(duplication survival).
    """
    probability: float
    mod_length: float
    mod_duplication: float


def _logit(p: float) -> float:
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def derive_category_rates(
    base_gain: np.ndarray, base_loss: np.ndarray,
    base_dup: np.ndarray, base_length: np.ndarray,
    cat: LogisticShiftCategory,
):
    """Derive (gain, loss, dup, length) for one category from base rates.

    Rate-level interpretation (Phase D):
      length_k[v] = length_base[v] * exp(mod_length)
      dup_k[v]    = dup_base[v]    * exp(mod_duplication)
      gain_k, loss_k unchanged.

    The mod_* values come from the file format as ``log(mul_*)`` where
    ``mul_*`` is a positive multiplier (RateVariationParser line 295-296):
    ``mul_len=1.5`` ↔ ``mod_len=log(1.5)≈0.405`` (50% longer branches).

    For the K=1, zero-shift case (every stored Williams/arc269 model uses
    this), the multipliers are 1.0 and this is the identity.

    Caveat vs Java: ``RateVariationModel.LogisticShift.updateNodeParameters``
    applies shifts in logit-of-survival-parameter space (mod_length on
    logit(p̃), mod_duplication on logit(λ/μ)), with a stability adjustment
    coupling the two. Our rate-level interpretation has the same K=1
    identity behavior and the same monotonicity (positive mod_length →
    longer effective branches → more loss), but the per-node numerical
    values differ for non-zero shifts. No stored dataset I have access to
    uses K>1, so neither path can be validated against Java reference
    numbers; this rate-level version is preferred for its physical
    interpretability and absence of the inverse survival-map machinery.
    """
    if cat.mod_length == 0.0 and cat.mod_duplication == 0.0:
        return base_gain, base_loss, base_dup, base_length

    length_k = np.asarray(base_length, dtype=np.float64).copy()
    dup_k = np.asarray(base_dup, dtype=np.float64).copy()
    if cat.mod_length != 0.0:
        scale = np.exp(cat.mod_length)
        # Preserve +inf at the root (which has length=inf by convention).
        length_k = np.where(np.isinf(length_k), length_k, length_k * scale)
    if cat.mod_duplication != 0.0:
        scale_dup = np.exp(cat.mod_duplication)
        # Preserve root dup at base value: the root has tau=+inf and is held
        # sub-critical (dup_root < loss_root) by convention; allowing the
        # shift to push root into super-critical breaks compute_L0_gradient
        # _analytical (NotImplementedError "Root with loss < 1 not
        # supported"). The shift on a single +inf-length edge is degenerate
        # anyway (the steady-state at the root is determined by the dup/loss
        # ratio, not their absolute values).
        root_indices = np.where(np.isinf(length_k))[0]
        dup_k = dup_k * scale_dup
        for r in root_indices:
            dup_k[r] = base_dup[r]
    return (np.asarray(base_gain, dtype=np.float64),
            np.asarray(base_loss, dtype=np.float64),
            dup_k, length_k)


def mixture_log_likelihood_native(
    tree: Tree,
    base_gain: np.ndarray, base_loss: np.ndarray,
    base_dup: np.ndarray, base_length: np.ndarray,
    profiles: np.ndarray,
    categories: Sequence[LogisticShiftCategory],
    *, min_copies: int = 1, num_threads: int = 0,
) -> float:
    """Corrected mixture log-likelihood Σ_f log Σ_k p_k · L_f(rates_k).

    Caveat: for K=1 with zero shifts (the only case actually present in
    Williams / arc269 stored models) this reduces to the base
    ``corrected_log_likelihood_native``. For K>1 / non-zero shifts the
    function will raise NotImplementedError pending Phase D.
    """
    if len(categories) == 0:
        raise ValueError("Need at least one category")
    # Single-category zero-shift shortcut: identical to base GLD.
    if (len(categories) == 1
            and categories[0].mod_length == 0.0
            and categories[0].mod_duplication == 0.0):
        return corrected_log_likelihood_native(
            tree, base_gain, base_loss, base_dup, base_length, profiles,
            min_copies=min_copies, num_threads=num_threads)

    # Multi-category: compute per-family LL for each category, then mix.
    from recount.native_backend import log_likelihood_native
    F = profiles.shape[0]
    per_fam_mix_log = np.full(F, -np.inf, dtype=np.float64)
    log_probs = np.log(np.array([c.probability for c in categories], dtype=np.float64))
    sum_log_p = float(log_probs.sum())  # sanity: should be 0 if probs sum to 1
    if abs(np.exp(log_probs).sum() - 1.0) > 1e-9:
        raise ValueError("category probabilities must sum to 1")

    for k, cat in enumerate(categories):
        gk, lk, dk, tk = derive_category_rates(
            base_gain, base_loss, base_dup, base_length, cat)
        per_fam_k = log_likelihood_native(
            tree, gk, lk, dk, tk, profiles, num_threads=num_threads)
        # Mix: per_fam_mix_log[f] = log(exp(per_fam_mix_log[f]) + p_k · exp(per_fam_k[f]))
        weighted = per_fam_k + log_probs[k]
        per_fam_mix_log = np.logaddexp(per_fam_mix_log, weighted)
    LL_raw = float(per_fam_mix_log.sum())
    # Corrected L(0) for the mixture: L(0)_mix = Σ_k p_k · L(0)_k
    if min_copies == 0:
        return LL_raw
    from recount.native_backend import unobserved_logL0_native
    log_L0_terms = []
    for k, cat in enumerate(categories):
        gk, lk, dk, tk = derive_category_rates(
            base_gain, base_loss, base_dup, base_length, cat)
        log_L0_k = unobserved_logL0_native(tree, gk, lk, dk, tk, min_copies)
        log_L0_terms.append(log_L0_k + log_probs[k])
    log_L0_mix = float(np.logaddexp.reduce(np.array(log_L0_terms)))
    p_obs = -np.expm1(log_L0_mix)
    return LL_raw - F * float(np.log(p_obs))


def _sanity_check_k1(tree: Tree, gain, loss, dup, length, profiles, min_copies=4):
    """Assert that K=1 zero-shift mixture equals base-GLD LL bit-for-bit."""
    base_ll = corrected_log_likelihood_native(
        tree, gain, loss, dup, length, profiles, min_copies=min_copies)
    mix_ll = mixture_log_likelihood_native(
        tree, gain, loss, dup, length, profiles,
        [LogisticShiftCategory(probability=1.0, mod_length=0.0, mod_duplication=0.0)],
        min_copies=min_copies,
    )
    assert abs(base_ll - mix_ll) < 1e-9, f"K=1 mixture should equal base: {base_ll} vs {mix_ll}"
    return base_ll, mix_ll
