"""K>1 LogisticShift mixture: corrected LL + analytical gradient.

Implements the responsibility-weighted mixture gradient derived in
docs/logistic_shift_gradient.tex (Prop 1):

    ∂(LL_corr_mix)/∂φ = Σ_k Σ_f γ_{f,k} · ∂log L_{f,k}/∂φ
                       + F/(1 - L(0)_mix) · Σ_k p_k · ∂L(0)_k/∂φ

where γ_{f,k} = p_k · L_{f,k} / L_f and L(0)_mix = Σ_k p_k · L(0)_k.

For our rate-level interpretation of the LogisticShift (see
recount/logistic_shift.py docstring): each category k has rates
``dup_k = dup_base · exp(Δ_dup^{(k)})`` and
``length_k = length_base · exp(Δ_length^{(k)})``; ``gain_k = gain_base``
and ``loss_k = loss_base`` are unshifted. So the chain-rule from base
parameters to per-category gradients is trivial multiplicative
re-scaling.

The softmax mixing weights ``p_k = exp(α_k) / Σ_k' exp(α_k')`` have
gradient ``∂LL/∂α_{k'} = N_{k'} - F · p_{k'}`` where
``N_{k'} = Σ_f γ_{f, k'}`` (also derived in the .tex, Prop 3).

This module provides ``mixture_gradient_native`` which returns the
full gradient vector w.r.t. (base_gain, base_dup, base_length,
Δ_dup, Δ_length, α). Useful for plugging into scipy.optimize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from recount.native_backend import (
    gradient_native_weighted,
    log_likelihood_native,
    unobserved_logL0_native,
)
from recount.logistic_shift import (
    LogisticShiftCategory,
    derive_category_rates,
)
from recount.tree import Tree


@dataclass
class MixtureGradient:
    """Analytical gradient of LL_corr_mix w.r.t. all free parameters.

    All arrays are float64.
    """
    LL_corr_mix: float                  # corrected mixture log-likelihood
    L0_mix: float                       # mixture L(0) at current params
    g_base_gain: np.ndarray             # [N]
    g_base_dup: np.ndarray              # [N]
    g_base_length: np.ndarray           # [N]
    g_delta_dup: np.ndarray             # [K] (the [0] entry is always 0 by id.)
    g_delta_length: np.ndarray          # [K]
    g_alpha: np.ndarray                 # [K] (the [0] entry is always 0 by id.)
    responsibilities: np.ndarray        # [F, K] for diagnostic use


def _softmax(alphas: np.ndarray) -> np.ndarray:
    a_max = float(np.max(alphas))
    e = np.exp(alphas - a_max)
    return e / e.sum()


def mixture_log_likelihood_and_responsibilities(
    tree: Tree,
    base_gain, base_loss, base_dup, base_length,
    profiles: np.ndarray,
    categories: Sequence[LogisticShiftCategory],
    *, min_copies: int = 1, num_threads: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Helper: forward only.

    Returns ``(LL_corr_mix, L0_mix, gamma)`` where gamma has shape [F, K].
    Does NOT compute gradients.
    """
    F = profiles.shape[0]
    K = len(categories)
    if K == 0:
        raise ValueError("Need at least one category")

    log_p = np.log(np.array([c.probability for c in categories], dtype=np.float64))
    if abs(np.exp(log_p).sum() - 1.0) > 1e-9:
        raise ValueError("category probabilities must sum to 1")

    log_L_fk = np.empty((F, K), dtype=np.float64)
    log_L0_k = np.empty(K, dtype=np.float64)
    for k, cat in enumerate(categories):
        gk, lk, dk, tk = derive_category_rates(
            base_gain, base_loss, base_dup, base_length, cat)
        log_L_fk[:, k] = log_likelihood_native(tree, gk, lk, dk, tk, profiles,
                                                num_threads=num_threads)
        log_L0_k[k] = unobserved_logL0_native(tree, gk, lk, dk, tk, min_copies)

    # log L_f = logsumexp_k (log_p[k] + log L_{f,k})
    log_L_f = np.logaddexp.reduce(log_p[np.newaxis, :] + log_L_fk, axis=1)
    LL_raw = float(log_L_f.sum())

    # L(0)_mix = Σ_k p_k * exp(log_L0_k)  (linear in L(0), not log)
    log_L0_mix = float(np.logaddexp.reduce(log_p + log_L0_k))
    L0_mix = float(np.exp(log_L0_mix))

    LL_corr_mix = LL_raw - F * float(np.log1p(-L0_mix))

    # Responsibilities: γ_{f,k} = p_k L_{f,k} / L_f
    #   log γ_{f,k} = log_p[k] + log L_{f,k} - log L_f
    log_gamma = log_p[np.newaxis, :] + log_L_fk - log_L_f[:, np.newaxis]
    gamma = np.exp(log_gamma)

    return LL_corr_mix, L0_mix, gamma


def _chain_rule_survival_to_rates(tree, gain, loss, dup, length, g_surv_flat):
    """Convert ∂(...)/∂(p̃, q̃, κ̃) gradient at given rates to ∂/∂(gain, loss, dup, length).

    Mirrors the chain-rule block in recount.ml._gradient_raw_native.
    """
    import torch
    from recount.torch_fast import compute_survival_params_t

    N = tree.num_nodes
    g_surv = g_surv_flat.reshape(N, 3)
    dt = torch.float64
    g_t = torch.tensor(gain, dtype=dt, requires_grad=True)
    l_t = torch.tensor(loss, dtype=dt, requires_grad=True)
    d_t = torch.tensor(dup, dtype=dt, requires_grad=True)
    t_t = torch.tensor(length, dtype=dt, requires_grad=True)
    sp = compute_survival_params_t(tree, g_t, l_t, d_t, t_t)
    surv_outputs = torch.stack([sp.gain, sp.p, sp.q])           # (3, N)
    surv_grad_out = torch.tensor(g_surv.T, dtype=dt)            # (3, N)
    grads = torch.autograd.grad(
        outputs=surv_outputs, inputs=(g_t, l_t, d_t, t_t),
        grad_outputs=surv_grad_out, allow_unused=True,
    )
    g_gain = grads[0].detach().numpy() if grads[0] is not None else np.zeros(N)
    g_loss = grads[1].detach().numpy() if grads[1] is not None else np.zeros(N)
    g_dup  = grads[2].detach().numpy() if grads[2] is not None else np.zeros(N)
    g_len  = grads[3].detach().numpy() if grads[3] is not None else np.zeros(N)
    return g_gain, g_loss, g_dup, g_len


def mixture_gradient_native(
    tree: Tree,
    base_gain, base_loss, base_dup, base_length,
    delta_dup: np.ndarray,        # [K] log-shift on dup (delta_dup[0] = 0)
    delta_length: np.ndarray,     # [K] log-shift on length (delta_length[0] = 0)
    alphas: np.ndarray,           # [K] softmax pre-images (alphas[0] = 0)
    profiles: np.ndarray,
    *, min_copies: int = 1, num_threads: int = 0,
) -> MixtureGradient:
    """Full K>1 LogisticShift mixture gradient.

    Inputs (free parameters):
      - base_gain, base_loss, base_dup, base_length: per-node base rates.
      - delta_dup[k], delta_length[k]: log-shifts for category k.
      - alphas[k]: pre-softmax weights. p_k = exp(alpha_k) / Σ exp(alpha_k').
      - We enforce delta_dup[0] = delta_length[0] = alphas[0] = 0 (identifiability).
        Gradient entries for [0] are returned as 0 and should not be optimized.

    Output: MixtureGradient with all gradient components + diagnostics.
    """
    F = profiles.shape[0]
    N = int(tree.num_nodes)
    K = len(delta_dup)
    if len(delta_length) != K or len(alphas) != K:
        raise ValueError("delta_dup, delta_length, alphas must have same length K")

    base_gain = np.asarray(base_gain, dtype=np.float64)
    base_loss = np.asarray(base_loss, dtype=np.float64)
    base_dup = np.asarray(base_dup, dtype=np.float64)
    base_length = np.asarray(base_length, dtype=np.float64)

    # Mixing weights from softmax.
    p_mix = _softmax(alphas)

    # Build per-category derived rates.
    rates_per_k = []
    for k in range(K):
        cat = LogisticShiftCategory(
            probability=float(p_mix[k]),
            mod_length=float(delta_length[k]),
            mod_duplication=float(delta_dup[k]),
        )
        gk, lk, dk, tk = derive_category_rates(
            base_gain, base_loss, base_dup, base_length, cat)
        rates_per_k.append((gk, lk, dk, tk))

    # Forward LL per category + L(0) per category.
    log_L_fk = np.empty((F, K), dtype=np.float64)
    log_L0_k = np.empty(K, dtype=np.float64)
    for k, (gk, lk, dk, tk) in enumerate(rates_per_k):
        log_L_fk[:, k] = log_likelihood_native(tree, gk, lk, dk, tk, profiles,
                                                num_threads=num_threads)
        log_L0_k[k] = unobserved_logL0_native(tree, gk, lk, dk, tk, min_copies)

    log_p = np.log(p_mix)
    # log L_f = log Σ_k p_k L_{f,k}
    log_L_f = np.logaddexp.reduce(log_p[np.newaxis, :] + log_L_fk, axis=1)
    LL_raw = float(log_L_f.sum())
    log_L0_mix = float(np.logaddexp.reduce(log_p + log_L0_k))
    L0_mix = float(np.exp(log_L0_mix))
    # Guard against L0_mix hitting 1.0 exactly (would cause 1/(1-L0_mix)
    # divide-by-zero downstream). Happens when the mixture-derived rates
    # push the unobserved-profile mass to all of (0, 0, ..., 0).
    L0_mix_safe = min(L0_mix, 1.0 - 1e-12)
    LL_corr_mix = LL_raw - F * float(np.log1p(-L0_mix_safe))

    # Responsibilities γ_{f,k} = p_k L_{f,k} / L_f.
    log_gamma = log_p[np.newaxis, :] + log_L_fk - log_L_f[:, np.newaxis]
    gamma = np.exp(log_gamma)

    # Initialise gradients.
    g_base_gain = np.zeros(N)
    g_base_loss = np.zeros(N)  # loss is fixed at 1.0 in our problems but we compute it anyway
    g_base_dup  = np.zeros(N)
    g_base_length = np.zeros(N)
    g_delta_dup = np.zeros(K)
    g_delta_length = np.zeros(K)
    g_alpha = np.zeros(K)

    # Per-category responsibility-weighted gradient (raw, no L(0) correction).
    # We get gradient at category-k rates (∂/∂gain_k, ∂/∂dup_k, ∂/∂length_k),
    # then chain-rule to base rates and shifts.
    coef = F / max(1.0 - L0_mix, 1e-12)  # for L(0) correction term (guarded)

    from recount.unobserved_outside import compute_L0_gradient_analytical

    for k, (gk, lk, dk, tk) in enumerate(rates_per_k):
        # Weighted raw gradient: Σ_f γ_{f,k} · ∂log L_{f,k} / ∂(p̃, q̃, κ̃)_v at rates_k.
        weights_k = np.ascontiguousarray(gamma[:, k], dtype=np.float64)
        _, g_surv_raw = gradient_native_weighted(
            tree, gk, lk, dk, tk, profiles, weights_k, num_threads=num_threads)

        # Chain-rule survival → category-k rates.
        gk_g_gain, gk_g_loss, gk_g_dup, gk_g_len = _chain_rule_survival_to_rates(
            tree, gk, lk, dk, tk, g_surv_raw)

        # L(0) correction at category k: ∂L(0)_k/∂(rates_k) analytical.
        _, gL0_g_k, gL0_l_k, gL0_d_k, gL0_t_k = compute_L0_gradient_analytical(
            tree, gk, lk, dk, tk, min_copies=min_copies)
        # Contribution to corrected gradient: + coef * p_k * exp(log_L0_k) * (1/L0_k) * ∂L0_k/∂rate
        #                                   = + coef * p_k * ∂L0_k/∂rate
        # (since the L(0) gradient functions return ∂L0/∂rate already)
        L0_correction_factor = coef * p_mix[k]
        gk_g_gain += L0_correction_factor * gL0_g_k
        gk_g_loss += L0_correction_factor * gL0_l_k
        gk_g_dup  += L0_correction_factor * gL0_d_k
        gk_g_len  += L0_correction_factor * gL0_t_k

        # Chain-rule category-k rates → base + shifts.
        # gain_k = gain_base, loss_k = loss_base → identity.
        # Convention from derive_category_rates: dup and length at the
        # root (length=+inf) are preserved at base values (shift does not
        # apply), so the chain rule treats root differently from non-root.
        g_base_gain += gk_g_gain
        g_base_loss += gk_g_loss
        scale_d = float(np.exp(delta_dup[k]))
        scale_t = float(np.exp(delta_length[k]))
        finite_mask = np.isfinite(base_length)  # True everywhere except root
        # base_dup chain rule: ∂dup_k/∂dup_base = scale_d on non-root, 1 on root
        g_base_dup[finite_mask] += gk_g_dup[finite_mask] * scale_d
        g_base_dup[~finite_mask] += gk_g_dup[~finite_mask]  # root: unchanged
        # delta_dup chain rule: sum over non-root only (root doesn't see shift)
        g_delta_dup[k] = float(np.sum(gk_g_dup[finite_mask] * dk[finite_mask]))
        # base_length chain rule: ∂length_k/∂length_base = scale_t on non-root
        # Root length stays +inf, so no contribution.
        g_base_length[finite_mask] += gk_g_len[finite_mask] * scale_t
        # delta_length chain rule: sum over non-root only.
        g_delta_length[k] = float(np.sum(gk_g_len[finite_mask] * tk[finite_mask]))

    # Softmax weight gradient: full chain rule including the L(0) correction.
    # ∂LL_raw/∂α_{k'} = N_{k'} - F · p_{k'}     where N_{k'} = Σ_f γ_{f,k'}
    # ∂L(0)_mix/∂α_{k'} = p_{k'} · (L(0)_{k'} - L(0)_mix)
    # ∂(-F · log(1 - L(0)_mix))/∂α_{k'} = +F · p_{k'} · (L(0)_{k'} - L(0)_mix) / (1 - L(0)_mix)
    # Total: ∂LL_corr_mix/∂α_{k'} = N_{k'} - F · p_{k'}
    #                              + F · p_{k'} · (L(0)_{k'} - L(0)_mix) / (1 - L(0)_mix)
    N_k = gamma.sum(axis=0)
    L0_k = np.exp(log_L0_k)
    g_alpha = N_k - F * p_mix + F * p_mix * (L0_k - L0_mix) / max(1.0 - L0_mix, 1e-12)

    # By identifiability convention, α[0] = 0 (and delta[0] = 0). We zero out
    # gradient on those slots so the caller doesn't accidentally optimize them.
    g_alpha[0] = 0.0
    g_delta_dup[0] = 0.0
    g_delta_length[0] = 0.0

    return MixtureGradient(
        LL_corr_mix=LL_corr_mix, L0_mix=L0_mix,
        g_base_gain=g_base_gain, g_base_dup=g_base_dup, g_base_length=g_base_length,
        g_delta_dup=g_delta_dup, g_delta_length=g_delta_length, g_alpha=g_alpha,
        responsibilities=gamma,
    )
