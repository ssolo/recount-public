"""LogisticShift K-category rate variation model.

This is Count's ``RateVariationModel.LogisticShift``: each family is drawn from
one of ``K`` categories, where category ``k`` shifts the per-edge logit
parameters by ``(mod_p[k], mod_q[k])``.  The per-family likelihood is a
mixture::

    L(profile_f) = sum_k  w_k  ·  L_k(profile_f)

where ``L_k`` is the standard GLD likelihood evaluated with category-k
parameters and ``w_k`` are non-negative weights that sum to 1.

The transformation (cf. ``RateVariationModel.LogisticShift.updateNodeParameters``
in Count's Java)::

    cat_logit(p_v)  =  logit(p_v_raw) + mod_p[k]
    cat_logit(λ_v)  =  logit(λ_v_raw) + mod_q[k]      (Pólya only)
    cat_logit(p_v) += log(1 − λ_v_raw) − log(1 − cat_λ_v)   (when mod_q ≠ 0)

with ``λ = q/p`` the per-edge "relative duplication rate" (≤ 1 by the
Count convention ``q ≤ p``).  Poisson nodes (``q_raw = 0``) only see the
``mod_p`` shift; ``q_cat`` stays 0.  The gain rate (γ or κ) is **not**
modified per category.

Public API:

  * :class:`LogisticShift` — a dataclass holding ``(weights, mod_p, mod_q)``.
  * :func:`mixture_log_likelihood` — corrected log-likelihood summed over
    families, autograd-friendly so L-BFGS-B can chain through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import softplus

from recount.rates import GLDRates
from recount.tree import Tree
from recount.torch_fast import (
    SurvivalParamsT,
    _rate_to_pq,
    _safe_log,
    _safe_log_pair,
    forward_fast,
    empty_log_likelihood_t,
    precompute_log_tables,
)


# ----------------------------------------------------------------------------
# data class
# ----------------------------------------------------------------------------


@dataclass
class LogisticShift:
    """K-category LogisticShift rate variation parameters.

    Attributes
    ----------
    weights : Tensor [K]
        Non-negative mixture weights summing to 1.  Stored directly (not as
        logits) so the dataclass is also a serializable record.
    mod_p : Tensor [K]
        Additive shift on logit(p_v_raw) per category.  ``mod_p[k] = 0``
        leaves the loss parameter unchanged.
    mod_q : Tensor [K]
        Additive shift on logit(λ_v) where ``λ = q/p``.  Ignored at Poisson
        nodes (``q_raw = 0``).  ``mod_q[k] = 0`` leaves the relative
        duplication unchanged.
    """
    weights: Tensor
    mod_p: Tensor
    mod_q: Tensor

    @property
    def K(self) -> int:
        return int(self.weights.shape[0])

    @classmethod
    def identity(cls, K: int = 1, *, dtype=torch.float64,
                 device: str = "cpu") -> "LogisticShift":
        """K identical no-shift categories with uniform weights."""
        return cls(
            weights=torch.full((K,), 1.0 / K, dtype=dtype, device=device),
            mod_p=torch.zeros(K, dtype=dtype, device=device),
            mod_q=torch.zeros(K, dtype=dtype, device=device),
        )

    def to(self, *, dtype=None, device=None) -> "LogisticShift":
        kw = {}
        if dtype is not None:
            kw["dtype"] = dtype
        if device is not None:
            kw["device"] = device
        return LogisticShift(
            weights=self.weights.to(**kw),
            mod_p=self.mod_p.to(**kw),
            mod_q=self.mod_q.to(**kw),
        )


# ----------------------------------------------------------------------------
# logit shift transform
# ----------------------------------------------------------------------------


def _shifted_pq(
    p_raw: Tensor, p_raw_c: Tensor, q_raw: Tensor, q_raw_c: Tensor,
    mod_p_k: Tensor, mod_q_k: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply LogisticShift for one category.

    Inputs are per-node raw (p, q) from ``_rate_to_pq`` and the category's
    scalar shifts.  Returns per-node ``(p_cat, p_cat_c, q_cat, q_cat_c)``
    after the logit transformation.

    For Poisson nodes (``q_raw = 0``) we only shift logit(p); for Pólya
    nodes both logit(p) and logit(q/p) shift, with the secondary
    ``log(1−λ) − log(1−cat_λ)`` adjustment on logit(p) that Count's code
    applies when ``mod_q ≠ 0`` to keep the (p, λ) Jacobian well-scaled.
    """
    is_polya = q_raw > 0
    # For autograd safety, substitute safe stand-ins for q_raw and (p−q) on
    # Poisson nodes (where they are 0) before taking logs.  `torch.where`
    # masks the output but its backward still multiplies through both
    # branches; if either branch produces a NaN/Inf gradient anywhere in
    # its sub-graph the mask cannot rescue it.  By making the inputs
    # numerically benign on the unselected side we get exact 0 contributions
    # in the gradient instead of 0·NaN.
    q_for_log = torch.where(is_polya, q_raw, torch.ones_like(q_raw))
    pq_gap = torch.where(is_polya, p_raw - q_raw, p_raw)

    # logit(p_raw) — use whichever of log(p) / log1p(-p_c) is more accurate.
    log_p = _safe_log_pair(p_raw, p_raw_c)
    log_p_c = _safe_log_pair(p_raw_c, p_raw)
    logit_p = log_p - log_p_c

    # λ = q/p ∈ [0, 1] (q ≤ p by Count convention); pq_gap = p − q
    log_q = torch.log(q_for_log)
    log_pq_gap = torch.log(pq_gap.clamp_min(1e-300))
    log_lam_c = log_pq_gap - log_p     # log((p−q)/p) — Pólya only
    logit_lam = log_q - log_pq_gap     # log(q/(p−q)) — Pólya only

    # Category-shifted logits
    cat_logit_p_base = logit_p + mod_p_k                  # before the λ adjustment
    cat_logit_lam = logit_lam + mod_q_k

    # log(1 − sigmoid(x)) = −softplus(x)  (numerically stable both ways)
    log_cat_lam_c = -softplus(cat_logit_lam)

    # Count's secondary adjustment: keeps logit(p) compatible with the
    # shifted relative duplication so the per-edge gradient remains
    # well-scaled even when mod_q is large.
    cat_logit_p_polya = cat_logit_p_base + (log_lam_c - log_cat_lam_c)

    # Final category-specific p and q
    p_cat_polya = torch.sigmoid(cat_logit_p_polya)
    p_cat_c_polya = torch.sigmoid(-cat_logit_p_polya)
    lam_cat = torch.sigmoid(cat_logit_lam)
    q_cat_polya = p_cat_polya * lam_cat

    p_cat_poisson = torch.sigmoid(cat_logit_p_base)
    p_cat_c_poisson = torch.sigmoid(-cat_logit_p_base)

    zero = torch.zeros_like(p_raw)
    one = torch.ones_like(p_raw)
    p_cat = torch.where(is_polya, p_cat_polya, p_cat_poisson)
    p_cat_c = torch.where(is_polya, p_cat_c_polya, p_cat_c_poisson)
    q_cat = torch.where(is_polya, q_cat_polya, zero)
    q_cat_c = torch.where(is_polya, one - q_cat, one)
    return p_cat, p_cat_c, q_cat, q_cat_c


# ----------------------------------------------------------------------------
# survival recursion from explicit (p, q)
# ----------------------------------------------------------------------------


def survival_params_from_pq(
    tree: Tree,
    p_raw: Tensor, p_raw_c: Tensor, q_raw: Tensor, q_raw_c: Tensor,
    gain: Tensor,
) -> SurvivalParamsT:
    """Run the rate→survival recursion starting from per-node raw (p, q).

    Mirrors :func:`recount.torch_fast.compute_survival_params_t` but skips
    the ``_rate_to_pq`` step at the front, so rate-variation modifiers that
    inject shifted (p, q) per category do not need to round-trip through
    (μ, λ, t).

    Parameters
    ----------
    tree : Tree
    p_raw, p_raw_c : Tensor [N]
        Per-node raw loss survival probability and its complement.
    q_raw, q_raw_c : Tensor [N]
        Per-node raw duplication survival probability and its complement
        (``q_raw = 0`` selects the Poisson branch).
    gain : Tensor [N]
        Per-node raw gain rate.  Treated as κ for Pólya nodes (Pólya
        concentration) and as γ for Poisson nodes (per-unit-time gain
        rate); the recursion converts γ → r = γ·(1−ε) internally.
    """
    n = tree.num_nodes
    dtype = p_raw.dtype
    device = p_raw.device
    is_polya = q_raw > 0

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    p_t_list: List[Tensor] = [None] * n  # type: ignore
    p_t_c_list: List[Tensor] = [None] * n  # type: ignore
    q_t_list: List[Tensor] = [None] * n  # type: ignore
    q_t_c_list: List[Tensor] = [None] * n  # type: ignore
    gain_list: List[Tensor] = [None] * n  # type: ignore
    eps_list: List[Tensor] = [None] * n  # type: ignore
    eps_c_list: List[Tensor] = [None] * n  # type: ignore

    for v in range(n):
        if tree.is_leaf[v]:
            e, e_c = zero, one
        else:
            e0, e1 = one, zero
            for c in tree.children[v]:
                e1 = e1 + e0 * p_t_c_list[c]
                e0 = e0 * p_t_list[c]
            e = e0
            e_c = torch.where(e == 1.0, e1, 1.0 - e)
        eps_list[v] = e
        eps_c_list[v] = e_c

        p = p_raw[v]; p_c = p_raw_c[v]
        q = q_raw[v]; q_c = q_raw_c[v]

        gain_list[v] = torch.where(is_polya[v], gain[v], gain[v] * e_c)

        a = q_c * e + e_c    # = 1 − q·ε
        p_t_list[v] = (p * e_c + e * q_c) / a
        q_t_list[v] = q * e_c / a
        b = e_c / a
        p_t_c_list[v] = p_c * b
        q_t_c_list[v] = q_c / a

    p_t = torch.stack(p_t_list)
    p_t_c = torch.stack(p_t_c_list)
    q_t = torch.stack(q_t_list)
    q_t_c = torch.stack(q_t_c_list)
    gain_t = torch.stack(gain_list)
    eps_t = torch.stack(eps_list)
    eps_c_t = torch.stack(eps_c_list)

    log_p = _safe_log_pair(p_t, p_t_c)
    log_p_c = _safe_log_pair(p_t_c, p_t)
    log_q = _safe_log_pair(q_t, q_t_c)
    log_q_c = _safe_log_pair(q_t_c, q_t)
    log_gain = _safe_log(gain_t)

    return SurvivalParamsT(
        p=p_t, p_c=p_t_c, q=q_t, q_c=q_t_c, gain=gain_t,
        eps=eps_t, eps_c=eps_c_t, is_polya=is_polya,
        log_p=log_p, log_p_c=log_p_c, log_q=log_q, log_q_c=log_q_c,
        log_gain=log_gain,
    )


# ----------------------------------------------------------------------------
# per-category survival params from a base GLDRates + LogisticShift
# ----------------------------------------------------------------------------


def category_survival_params(
    tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    shift: LogisticShift,
) -> List[SurvivalParamsT]:
    """Compute one ``SurvivalParamsT`` per category.

    Convenience wrapper: starting from base rates ``(gain, loss, dup, length)``,
    convert to per-edge raw (p, q), apply each category's logit shift, and
    run the survival recursion separately for each.
    """
    p_raw, p_raw_c, q_raw, q_raw_c = _rate_to_pq(loss, dup, length)
    out: List[SurvivalParamsT] = []
    for k in range(shift.K):
        p_k, p_c_k, q_k, q_c_k = _shifted_pq(
            p_raw, p_raw_c, q_raw, q_raw_c, shift.mod_p[k], shift.mod_q[k],
        )
        sp_k = survival_params_from_pq(tree, p_k, p_c_k, q_k, q_c_k, gain)
        out.append(sp_k)
    return out


# ----------------------------------------------------------------------------
# mixture LL
# ----------------------------------------------------------------------------


def mixture_log_likelihood(
    tree: Tree,
    gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    shift: LogisticShift,
    profiles: Tensor,
    *,
    min_copies: int = 1,
    W: Optional[int] = None,
    chunk_F: int = 1024,
) -> Tensor:
    """K-category LogisticShift mixture log-likelihood, corrected.

    Per-family LL is ``log Σ_k w_k · L_k(family)``.  Total = sum over
    families minus ``F · log(1 − L_0_mix)`` for the ``min_copies = 1``
    bias correction.

    Uses :func:`recount.torch_fast.forward_fast` (the vectorized inside
    pass) for each category, so ``.backward()`` is fast — about 60× per
    iter over the previous per-family-loop fallback on Williams.
    ``.backward()`` on the result gives the analytical gradient w.r.t.
    base rates ``(gain, loss, dup, length)`` and LogisticShift parameters
    ``(weights, mod_p, mod_q)``.

    Parameters
    ----------
    tree, profiles
        As in :func:`recount.torch_fast.corrected_log_likelihood_fast`.
    gain, loss, dup, length : Tensor [N]
        Base GLD rates.
    shift : LogisticShift
        Per-category logit shifts and weights.
    min_copies : {0, 1}
        Observation-bias correction.  ``min_copies = 2`` is not yet
        supported under the mixture.
    W : optional int
        Max profile sum (forward-pass width).  Defaults to ``profiles.sum(1).max()``.
    chunk_F : int
        Family-batch chunk size for the combine step (memory control).
    """
    if min_copies == 2:
        raise NotImplementedError("min_copies=2 mixture correction TBD")

    F = profiles.shape[0]
    if W is None:
        W = int(profiles.sum(dim=1).max().item())

    sp_list = category_survival_params(tree, gain, loss, dup, length, shift)

    log_w = torch.log(shift.weights.clamp_min(1e-300))
    log_w_norm = log_w - torch.logsumexp(log_w, dim=0)  # [K]

    LL_per_cat: List[Tensor] = []
    L0_per_cat: List[Tensor] = []
    for sp in sp_list:
        LL_k_f = forward_fast(tree, sp, profiles, W, chunk_F=chunk_F)
        LL_per_cat.append(LL_k_f)
        if min_copies >= 1:
            L0_per_cat.append(empty_log_likelihood_t(sp))
    LL_stack = torch.stack(LL_per_cat, dim=0)  # [K, F]

    LL_mixed_f = torch.logsumexp(LL_stack + log_w_norm.unsqueeze(1), dim=0)
    LL_total = LL_mixed_f.sum()
    if min_copies == 0:
        return LL_total

    L0_stack = torch.stack(L0_per_cat, dim=0)
    L0_mixed = torch.logsumexp(L0_stack + log_w_norm, dim=0)
    log_p_obs = torch.log(-torch.expm1(L0_mixed))
    return LL_total - F * log_p_obs


# ----------------------------------------------------------------------------
# NumPy-side convenience: corrected LL with a fixed LogisticShift
# ----------------------------------------------------------------------------


def corrected_mixture_log_likelihood(
    tree: Tree, rates: GLDRates, shift: LogisticShift,
    profiles: np.ndarray, *,
    min_copies: int = 1, W: Optional[int] = None, chunk_F: int = 1024,
) -> float:
    """Float wrapper around :func:`mixture_log_likelihood` for evaluation
    (no gradient needed).  Convenient for verifying that K=1 + identity
    LogisticShift reproduces the bare GLD likelihood."""
    dtype = torch.float64
    gain_t = torch.tensor(rates.gain, dtype=dtype)
    loss_t = torch.tensor(rates.loss, dtype=dtype)
    dup_t = torch.tensor(rates.dup, dtype=dtype)
    length_t = torch.tensor(rates.length, dtype=dtype)
    prof_t = torch.tensor(profiles, dtype=torch.long)
    shift_t = shift.to(dtype=dtype)
    with torch.no_grad():
        LL = mixture_log_likelihood(
            tree, gain_t, loss_t, dup_t, length_t, shift_t,
            prof_t, min_copies=min_copies, W=W, chunk_F=chunk_F,
        )
    return float(LL)
