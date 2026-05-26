"""GPU-friendly vectorized PyTorch backend for the GLD likelihood.

Same model and conventions as ``recount.gld`` but rebuilt for **batch
parallelism** over families and **tensor parallelism** over copy-count axes.
The forward pass is purely functional torch ops (no in-place writes) so
autograd recovers the analytical gradient by reverse-mode AD.

Key design choices:

  * **Uniform width.** All per-node likelihood vectors are padded to a
    fixed width ``W+1`` (the max ancestral surviving-copy count plus one).
    Invalid entries are ``-inf``; ``logsumexp`` ignores them.

  * **Family batch dim.** The leading axis of every tensor is the family
    index ``F``. The whole 5378-family Williams2017 dataset runs as one
    batched forward.

  * **No destructive update.** ``recount.gld._compute_sibling`` uses an
    in-place log-space recurrence to marginalize over copies that survive
    in both children of a binary parent. Here we replace that with a
    closed-form 3-cell multinomial split

        log_split[v, ell, a, b]
            = log Mult(ell; ell-b, a+b-ell, ell-a)
              + (ell-b)·log π₁ + (a+b-ell)·log π_both + (ell-a)·log π₂

    where π₁ = (1-p̃_{j1})·p̃_{j2}/(1-ε), π₂ = p̃_{j1}·(1-p̃_{j2})/(1-ε),
    π_both = (1-p̃_{j1})·(1-p̃_{j2})/(1-ε), and ε = p̃_{j1}·p̃_{j2}. The
    parent's inside likelihood is then a single 2-D logsumexp:

        C[v][f, ell] = LSE_{a,b} (K[j1][f, a] + K[j2][f, b]
                                  + log_split[v, ell, a, b])

  * **Edge / gain step.** ``K[v][f, s] = LSE_t (C[v][f, s+t] + log_pmf[v, s, t])``
    where log_pmf encodes the per-edge Poisson (q=0) or Pólya (q>0) gain.
    Both s and t are vectorized; we ``gather`` C at offset s+t.

Limitations:

  * **Binary trees only** for now. Multifurcations need a chained binary
    combine (straightforward extension, not yet implemented).
  * Root must satisfy ``p̃_root == 1`` (the ``length=+∞`` convention).
    The ROOTLOSS configuration (root with finite edge) raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from recount.destructive_combine import destructive_combine
from recount.tree import Tree


NEG_INF = float("-inf")


# ----------------------------------------------------------------------------
# survival-parameter precomputation (torch version)
# ----------------------------------------------------------------------------


@dataclass
class SurvivalParamsT:
    p: Tensor              # [N] survival p̃
    p_c: Tensor            # [N] 1 - p̃
    q: Tensor              # [N] survival q̃
    q_c: Tensor            # [N] 1 - q̃
    gain: Tensor           # [N] r̃ (Poisson) or κ̃=κ (Pólya)
    eps: Tensor            # [N] extinction
    eps_c: Tensor          # [N] 1 - eps
    log_p: Tensor
    log_p_c: Tensor
    log_q: Tensor
    log_q_c: Tensor
    log_gain: Tensor
    is_polya: Tensor       # [N] bool


def _safe_log(x: Tensor) -> Tensor:
    """log(x) with -inf at x=0 (clamped to a tiny positive for safety)."""
    return torch.log(x.clamp_min(1e-300))


def _safe_logsumexp(x: Tensor, dim) -> Tensor:
    """``torch.logsumexp`` whose backward is 0 (not NaN) on all-``-inf`` rows.

    ``torch.logsumexp`` divides 0 by 0 internally when every input along the
    reduction axis is ``-inf`` (the softmax-style backward needs
    ``exp(x − logsumexp)`` and gets ``exp(-inf − -inf) = NaN``).  We swap
    ``-inf`` rows for zeros on the way in (so backward sees a benign input
    with gradient 0) and re-mask the output to ``-inf`` on the way out.
    """
    m = x.amax(dim=dim, keepdim=True)
    valid = (m > NEG_INF)
    safe_x = torch.where(valid.expand_as(x), x, torch.zeros_like(x))
    result_safe = torch.logsumexp(safe_x, dim=dim)
    # `dim` may be int or tuple; squeeze the same axes we kept.
    if isinstance(dim, int):
        valid_squeezed = valid.squeeze(dim)
    else:
        valid_squeezed = valid
        for d in sorted(dim, reverse=True):
            valid_squeezed = valid_squeezed.squeeze(d)
    return torch.where(valid_squeezed, result_safe,
                       torch.full_like(result_safe, NEG_INF))


def _safe_log_pair(x: Tensor, x_complement: Tensor) -> Tensor:
    """log(x) using whichever of log(x) or log1p(-x_complement) is more
    accurate. For x near 1, the second form preserves the ULPs that ``log``
    would round away.

    Autograd-safe: the unselected branch evaluates on a benign 0.5, so its
    backward never produces NaN that would leak through ``torch.where``.
    ``clamp_max(1 − 2e-16)`` (not ``1 − 1e-300``) is needed because below
    ULP the clamp is a no-op in fp64 and ``log1p(-1) = -inf`` whose
    backward 1/(1+x) explodes.
    """
    use_direct = x < x_complement
    half = torch.full_like(x, 0.5)
    x_safe = torch.where(use_direct, x.clamp_min(1e-300), half)
    xc_safe = torch.where(use_direct, half, x_complement.clamp_max(1.0 - 2e-16))
    direct = torch.log(x_safe)
    via_complement = torch.log1p(-xc_safe)
    return torch.where(use_direct, direct, via_complement)


def _rate_to_pq(mu: Tensor, lam: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-edge (p, p_c, q, q_c) — vectorized & differentiable.

    Branches on (μ vs λ) using ``torch.where`` so gradients propagate
    through every case.  Handles t=+∞ for the root.
    """
    one = torch.ones_like(mu)
    eps_t = torch.tensor(1e-300, dtype=mu.dtype, device=mu.device)
    mask_inf = torch.isinf(t)
    t_fin = torch.where(mask_inf, torch.zeros_like(t), t)

    mask_mu0 = mu == 0
    mask_eq = (~mask_mu0) & (mu == lam)
    mask_lam0 = (~mask_mu0) & (lam == 0)
    mask_lt = (~mask_mu0) & (~mask_eq) & (lam < mu) & (~mask_lam0)
    mask_gt = (~mask_mu0) & (~mask_eq) & (lam > mu)

    # μ == λ : p = q = μt/(1+μt)
    mu_t = mu * t_fin
    p_eq = mu_t / (1.0 + mu_t)
    p_c_eq = 1.0 / (1.0 + mu_t)
    p_eq = torch.where(mask_inf, one, p_eq)
    p_c_eq = torch.where(mask_inf, torch.zeros_like(p_c_eq), p_c_eq)
    q_eq = p_eq
    q_c_eq = p_c_eq

    # λ == 0 (Poisson)
    p_lam0 = -torch.expm1(-mu * t_fin)
    p_c_lam0 = torch.exp(-mu * t_fin)
    p_lam0 = torch.where(mask_inf, one, p_lam0)
    p_c_lam0 = torch.where(mask_inf, torch.zeros_like(p_c_lam0), p_c_lam0)
    q_lam0 = torch.zeros_like(mu)
    q_c_lam0 = torch.ones_like(mu)

    # λ < μ — use the stable `−expm1(−d + log1p(−δ))` form when δ = gap/μ
    # is small (avoids catastrophic cancellation in `μ − λ·E` when λ ≈ μ).
    gap_lt = mu - lam
    d_lt = gap_lt * t_fin
    E_lt = torch.exp(-d_lt)
    E1_lt = -torch.expm1(-d_lt)
    delta_lt = gap_lt / torch.clamp(mu, min=eps_t)
    denom_stable_lt = mu * (-torch.expm1(-d_lt + torch.log1p(-delta_lt.clamp_max(1 - 1e-15))))
    denom_naive_lt = mu - lam * E_lt
    denom_lt = torch.where(delta_lt < 0.5, denom_stable_lt, denom_naive_lt)
    denom_lt = torch.clamp(denom_lt, min=eps_t)
    p_lt = mu * E1_lt / denom_lt
    p_c_lt = gap_lt * E_lt / denom_lt
    q_lt = lam * E1_lt / denom_lt
    q_c_lt = gap_lt / denom_lt
    # t = inf
    p_lt = torch.where(mask_inf, one, p_lt)
    p_c_lt = torch.where(mask_inf, torch.zeros_like(p_c_lt), p_c_lt)
    q_lt = torch.where(mask_inf, lam / torch.clamp(mu, min=eps_t), q_lt)
    q_c_lt = torch.where(mask_inf, 1.0 - lam / torch.clamp(mu, min=eps_t), q_c_lt)

    # λ > μ — symmetric stable form (now δ = gap/λ).
    gap_gt = lam - mu
    d_gt = gap_gt * t_fin
    E_gt = torch.exp(-d_gt)
    E1_gt = -torch.expm1(-d_gt)
    delta_gt = gap_gt / torch.clamp(lam, min=eps_t)
    denom_stable_gt = lam * (-torch.expm1(-d_gt + torch.log1p(-delta_gt.clamp_max(1 - 1e-15))))
    denom_naive_gt = lam - mu * E_gt
    denom_gt = torch.where(delta_gt < 0.5, denom_stable_gt, denom_naive_gt)
    denom_gt = torch.clamp(denom_gt, min=eps_t)
    p_gt = mu * E1_gt / denom_gt
    p_c_gt = gap_gt / denom_gt
    q_gt = lam * E1_gt / denom_gt
    q_c_gt = gap_gt * E_gt / denom_gt
    p_gt = torch.where(mask_inf, mu / torch.clamp(lam, min=eps_t), p_gt)
    p_c_gt = torch.where(mask_inf, 1.0 - mu / torch.clamp(lam, min=eps_t), p_c_gt)
    q_gt = torch.where(mask_inf, one, q_gt)
    q_c_gt = torch.where(mask_inf, torch.zeros_like(q_c_gt), q_c_gt)

    # Assemble (μ=0 case is the catch-all)
    zero = torch.zeros_like(mu)
    p = torch.where(mask_mu0, zero,
        torch.where(mask_eq, p_eq,
        torch.where(mask_lam0, p_lam0,
        torch.where(mask_lt, p_lt, p_gt))))
    p_c = torch.where(mask_mu0, one,
        torch.where(mask_eq, p_c_eq,
        torch.where(mask_lam0, p_c_lam0,
        torch.where(mask_lt, p_c_lt, p_c_gt))))
    q = torch.where(mask_mu0, zero,
        torch.where(mask_eq, q_eq,
        torch.where(mask_lam0, q_lam0,
        torch.where(mask_lt, q_lt, q_gt))))
    q_c = torch.where(mask_mu0, one,
        torch.where(mask_eq, q_c_eq,
        torch.where(mask_lam0, q_c_lam0,
        torch.where(mask_lt, q_c_lt, q_c_gt))))
    return p, p_c, q, q_c


def compute_survival_params_t(
    tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
) -> SurvivalParamsT:
    """Bottom-up: raw rates → survival (p̃, q̃, r̃/κ̃) + ε per node.

    The same recursion as `recount.gld.compute_survival_params`, written
    with tensor ops so it's differentiable end-to-end.
    """
    n = tree.num_nodes
    dtype = gain.dtype
    device = gain.device

    p_raw, p_raw_c, q_raw, q_raw_c = _rate_to_pq(loss, dup, length)
    is_polya = q_raw > 0

    # Build per-node tensors via Python loop (the loop is over ~120 nodes,
    # negligible compared to the per-family inner work).
    p_t_list: List[Tensor] = [None] * n  # type: ignore
    p_t_c_list: List[Tensor] = [None] * n  # type: ignore
    q_t_list: List[Tensor] = [None] * n  # type: ignore
    q_t_c_list: List[Tensor] = [None] * n  # type: ignore
    gain_list: List[Tensor] = [None] * n  # type: ignore
    eps_list: List[Tensor] = [None] * n  # type: ignore
    eps_c_list: List[Tensor] = [None] * n  # type: ignore

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

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

        a = q_c * e + e_c   # = 1 − q·ε (stable form)
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

    # Use the dual log/log1p form so values near 1 keep full precision.
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
# precomputed log_split (combinatorial weights for binary internal nodes)
# ----------------------------------------------------------------------------


def _log_fact_table(W: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """``log(n!)`` for n in [0, W] — single source of truth for the factorial LUT."""
    return torch.lgamma(torch.arange(W + 1, dtype=dtype, device=device) + 1.0)


def _build_sibling_params(
    p_j1: Tensor, p_c_j1: Tensor, p_j2: Tensor, p_c_j2: Tensor,
    eps_c_parent: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-node scalars for the destructive O(W²) sibling combine.

    Maps to ``recount.gld._compute_sibling``'s (log_e, log_e_c, logp1, logp2)
    in the binary case where ``eps_sib`` collapses to a single sibling's
    ``p̃_{j1}``:

        log_e   = log(p̃_{j1})
        log_e_c = log(1 − p̃_{j1})
        log_a   = log(1 − p̃_{j2}·p̃_{j1}) = log(ε_c_parent)
        logp1   = log(p̃_c_{j2}) − log_a
        logp2   = log(p̃_{j2}) + log_e_c − log_a

    The ``log_a`` form uses the parent's precomputed ``eps_c`` (built with a
    stable accumulator) instead of ``log(1 − p_j1·p_j2)`` directly — the
    latter would catastrophically cancel when ε is close to 1.
    """
    log_e = _safe_log_pair(p_j1, p_c_j1)
    log_e_c = _safe_log_pair(p_c_j1, p_j1)
    log_a = _safe_log(eps_c_parent)
    log_pj2 = _safe_log_pair(p_j2, p_c_j2)
    log_pj2_c = _safe_log_pair(p_c_j2, p_j2)
    logp1 = log_pj2_c - log_a
    logp2 = log_pj2 + log_e_c - log_a
    return log_e, log_e_c, logp1, logp2


def _build_log_split(
    p_j1: Tensor, p_c_j1: Tensor, p_j2: Tensor, p_c_j2: Tensor,
    eps_c_parent: Tensor, W: int, log_fact: Tensor,
) -> Tensor:
    """log_split[ell, a, b] for one binary internal node.

    Encodes the multinomial split (X₁, X_both, X₂) ~ Mult(ell; π₁, π_both, π₂)
    with a = X₁ + X_both, b = X₂ + X_both. Invalid (ell, a, b) entries return
    ``-inf``; ``logsumexp`` ignores them naturally.
    """
    # Use the precomputed eps_c (= 1 − ε) from the parent's survival-params
    # pass — that one was built with a numerically stable accumulator, while
    # `1.0 - p_j1*p_j2` here would catastrophically cancel when ε is close to 1.
    eps_c = eps_c_parent.clamp_min(1e-300)
    pi_1 = (p_c_j1 * p_j2) / eps_c
    pi_2 = (p_j1 * p_c_j2) / eps_c
    pi_b = (p_c_j1 * p_c_j2) / eps_c
    log_pi_1 = _safe_log(pi_1)
    log_pi_2 = _safe_log(pi_2)
    log_pi_b = _safe_log(pi_b)

    device = p_j1.device
    dtype = p_j1.dtype
    idx = torch.arange(W + 1, device=device)
    ell = idx.view(W + 1, 1, 1)
    a = idx.view(1, W + 1, 1)
    b = idx.view(1, 1, W + 1)

    X1 = ell - b
    X2 = ell - a
    Xb = a + b - ell
    valid = (X1 >= 0) & (X2 >= 0) & (Xb >= 0)

    # Clamp to [0, W] to keep all factorial lookups in-bounds for the invalid
    # entries; valid entries always satisfy X1, X2, Xb ≤ ell ≤ W.
    X1_safe = X1.clamp(0, W)
    X2_safe = X2.clamp(0, W)
    Xb_safe = Xb.clamp(0, W)

    log_split = (
        log_fact[ell.expand_as(X1_safe)]
        - log_fact[X1_safe]
        - log_fact[X2_safe]
        - log_fact[Xb_safe]
        + X1_safe.to(dtype) * log_pi_1
        + X2_safe.to(dtype) * log_pi_2
        + Xb_safe.to(dtype) * log_pi_b
    )
    return torch.where(valid, log_split, torch.full_like(log_split, NEG_INF))


def _build_log_pmf(
    is_polya_v: bool, gain_v: Tensor, log_gain_v: Tensor,
    log_q_v: Tensor, log_q_c_v: Tensor,
    W: int, log_fact: Tensor,
) -> Tensor:
    """log_pmf[s, t] for one node: probability of gaining t copies on the edge
    entering v that brought s incoming surviving copies.

    Poisson (q=0): exp(-r) · r^t / t!  →  log = -r + t·log(r) - log(t!)
    Pólya (q>0):   NegBin(t; κ+s, 1-q) = Γ(κ+s+t)/(Γ(κ+s)·t!) · (1-q)^(κ+s) · q^t
                   →  log_rf(κ, s+t) - log_rf(κ, s) - log(t!) + (κ+s)·log(1-q) + t·log(q)

    Returns [W+1, W+1] with -inf for s+t > W.
    """
    device = gain_v.device
    dtype = gain_v.dtype
    s = torch.arange(W + 1, device=device).view(W + 1, 1)
    t = torch.arange(W + 1, device=device).view(1, W + 1)
    st = s + t
    valid = st <= W

    if not bool(is_polya_v):
        # Poisson
        if log_gain_v.item() == NEG_INF:
            log_pmf = torch.full((W + 1, W + 1), NEG_INF, device=device, dtype=dtype)
            log_pmf[:, 0] = 0.0
            return log_pmf
        r = gain_v
        t_logr = t.to(dtype) * log_gain_v
        # t=0 gives 0·log_r = 0 (handled correctly since log_r is finite here)
        t_logr = torch.where(t == 0, torch.zeros_like(t_logr), t_logr)
        log_pmf = -r + t_logr - log_fact[t.expand(W + 1, W + 1)]
        return torch.where(valid, log_pmf, torch.full_like(log_pmf, NEG_INF))

    # Pólya
    if log_gain_v.item() == NEG_INF:
        log_pmf = torch.full((W + 1, W + 1), NEG_INF, device=device, dtype=dtype)
        log_pmf[:, 0] = 0.0
        return log_pmf
    kappa = gain_v
    log_rf_st = torch.lgamma(kappa + st.to(dtype)) - torch.lgamma(kappa)
    log_rf_s = torch.lgamma(kappa + s.to(dtype)) - torch.lgamma(kappa)
    ks_log1_q = (kappa + s.to(dtype)) * log_q_c_v
    t_logq = t.to(dtype) * log_q_v
    t_logq = torch.where(t == 0, torch.zeros_like(t_logq), t_logq)
    log_pmf = log_rf_st - log_rf_s - log_fact[t.expand(W + 1, W + 1)] + ks_log1_q + t_logq
    return torch.where(valid, log_pmf, torch.full_like(log_pmf, NEG_INF))


# ----------------------------------------------------------------------------
# vectorized inside (forward) pass
# ----------------------------------------------------------------------------


def _vectorized_combine(
    k1: Tensor, k2: Tensor, log_split: Tensor, chunk_F: int,
) -> Tensor:
    """Combine two children's edge log-likelihoods into the parent's node LL.

    k1, k2 : [F, W+1]   — child edge log-likelihoods, indexed by `a`, `b`.
    log_split : [W+1, W+1, W+1] — combinatorial weights, indexed by [ell, a, b].
    Returns C : [F, W+1].

    Builds the 4-D tensor terms[f, ell, a, b] = k1[f,a] + k2[f,b] + log_split
    and reduces over (a, b) via logsumexp. Chunked along F to control memory.
    """
    F, Wp1 = k1.shape
    out_chunks: List[Tensor] = []
    for s in range(0, F, chunk_F):
        e = min(s + chunk_F, F)
        # broadcast: [c, 1, W+1, 1] + [c, 1, 1, W+1] + [1, W+1, W+1, W+1]
        term = (
            k1[s:e].view(-1, 1, Wp1, 1)
            + k2[s:e].view(-1, 1, 1, Wp1)
            + log_split.view(1, Wp1, Wp1, Wp1)
        )
        C_chunk = _safe_logsumexp(term, dim=(2, 3))  # [c, W+1]
        out_chunks.append(C_chunk)
    return torch.cat(out_chunks, dim=0) if len(out_chunks) > 1 else out_chunks[0]


def _vectorized_edge(C_v: Tensor, log_pmf_v: Tensor) -> Tensor:
    """K[v][f, s] = LSE_t (C[v][f, s+t] + log_pmf_v[s, t]).

    C_v : [F, W+1]
    log_pmf_v : [W+1, W+1]  (-inf where s+t > W)
    Returns K_v : [F, W+1].
    """
    F, Wp1 = C_v.shape
    W = Wp1 - 1
    device = C_v.device
    # idx[s, t] = clamp(s + t, max=W)
    s_idx = torch.arange(Wp1, device=device).view(Wp1, 1)
    t_idx = torch.arange(Wp1, device=device).view(1, Wp1)
    sum_idx = (s_idx + t_idx).clamp_max(W)  # [W+1, W+1]
    # Gather C_v at positions sum_idx: shape [F, W+1, W+1]
    C_gather = C_v[:, sum_idx]
    terms = C_gather + log_pmf_v.unsqueeze(0)  # [F, W+1, W+1]
    return _safe_logsumexp(terms, dim=2)        # [F, W+1]


def precompute_log_tables(
    tree: Tree, sp: SurvivalParamsT, W: int, *, use_destructive: bool = False,
) -> Tuple[
    List[Optional[Tensor]],
    List[Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]],
    List[Tensor],
]:
    """Build per-node combine tables: ``log_split`` *or* ``sibling_params``, plus ``log_pmf``.

    Returns ``(log_splits, sibling_params, log_pmfs)`` — three lists indexed
    by node id. Doing this once on CPU in fp64 keeps the precision-sensitive
    ``log1p(-tiny)`` evaluations stable; the returned tensors can then be
    moved to a different device/dtype (e.g. MPS fp32) for the heavy inside
    pass.

    When ``use_destructive=False`` (default): only ``log_splits`` is
    populated for internal nodes (closed-form O(W³) combine);
    ``sibling_params`` is all ``None``.

    When ``use_destructive=True``: only ``sibling_params`` is populated
    (O(W²) destructive combine); ``log_splits`` is all ``None``.
    """
    n = tree.num_nodes
    dtype = sp.p.dtype
    device = sp.p.device
    log_fact = _log_fact_table(W, dtype, device)

    log_splits: List[Optional[Tensor]] = [None] * n
    sibling_params: List[Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]] = [None] * n
    log_pmfs: List[Tensor] = [None] * n  # type: ignore
    for v in range(n):
        log_pmfs[v] = _build_log_pmf(
            bool(sp.is_polya[v]), sp.gain[v], sp.log_gain[v],
            sp.log_q[v], sp.log_q_c[v], W, log_fact,
        )
        if tree.is_leaf[v]:
            continue
        kids = tree.children[v]
        if len(kids) != 2:
            raise NotImplementedError(
                f"node {v} has {len(kids)} children; this backend supports "
                "binary trees only (multifurcations need a chained combine)"
            )
        j1, j2 = int(kids[0]), int(kids[1])
        if use_destructive:
            sibling_params[v] = _build_sibling_params(
                sp.p[j1], sp.p_c[j1], sp.p[j2], sp.p_c[j2], sp.eps_c[v],
            )
        else:
            log_splits[v] = _build_log_split(
                sp.p[j1], sp.p_c[j1], sp.p[j2], sp.p_c[j2],
                sp.eps_c[v], W, log_fact,
            )
    return log_splits, sibling_params, log_pmfs


def forward_fast(
    tree: Tree, sp: SurvivalParamsT, profiles: Tensor, W: int,
    chunk_F: int = 1024,
    log_splits: Optional[List[Optional[Tensor]]] = None,
    sibling_params: Optional[List[Optional[Tuple[Tensor, ...]]]] = None,
    log_pmfs: Optional[List[Tensor]] = None,
    *, use_destructive: bool = False,
) -> Tensor:
    """Vectorized Felsenstein peeling: returns per-family root edge LL.

    Parameters
    ----------
    tree : Tree
        Binary rooted tree (multifurcations not yet supported here).
    sp : SurvivalParamsT
        Survival params computed by ``compute_survival_params_t``.
    profiles : LongTensor [F, num_leaves]
        Per-family observed copy counts at each leaf.
    W : int
        Uniform max copy width (must cover the largest profile sum).
    chunk_F : int
        Family-batch chunk size for the combine step (memory control).
        Ignored when ``use_destructive=True``.
    use_destructive : bool, default False
        If True, use the O(W²) destructive sibling combine (Triton on CUDA,
        PyTorch elsewhere) instead of the closed-form O(W³) combine.
    """
    n = tree.num_nodes
    device = profiles.device
    dtype = sp.p.dtype
    F = profiles.shape[0]

    # Precompute log_split / sibling_params / log_pmf if not supplied. When
    # supplied (as produced by ``precompute_log_tables``), they're trusted
    # to be in the right shape and on a compatible device/dtype.
    if log_pmfs is None or (
        use_destructive and sibling_params is None
    ) or (not use_destructive and log_splits is None):
        log_splits, sibling_params, log_pmfs = precompute_log_tables(
            tree, sp, W, use_destructive=use_destructive,
        )
    # Cast to profiles' device/dtype so all ops live on the same device.
    log_pmfs = [lp.to(device=device, dtype=dtype) for lp in log_pmfs]
    if use_destructive:
        sibling_params = [
            tuple(p.to(device=device, dtype=dtype) for p in sp_v)
            if sp_v is not None else None
            for sp_v in sibling_params
        ]
        log_fact = _log_fact_table(W, dtype, device)
    else:
        log_splits = [
            ls.to(device=device, dtype=dtype) if ls is not None else None
            for ls in log_splits
        ]

    # Helpful constants
    arange_W = torch.arange(W + 1, device=device).view(1, W + 1)

    # ---- Felsenstein up-pass --------------------------------------------
    Cs: List[Tensor] = [None] * n  # type: ignore
    Ks: List[Tensor] = [None] * n  # type: ignore
    for v in range(n):
        if tree.is_leaf[v]:
            counts = profiles[:, v].view(F, 1)
            mask = arange_W == counts
            C_v = torch.where(mask, torch.zeros((F, W + 1), device=device, dtype=dtype),
                              torch.full((F, W + 1), NEG_INF, device=device, dtype=dtype))
        else:
            j1, j2 = (int(c) for c in tree.children[v])
            if use_destructive:
                le, lec, lp1, lp2 = sibling_params[v]
                C_v = destructive_combine(
                    Ks[j1], Ks[j2], le, lec, lp1, lp2, log_fact,
                )
            else:
                C_v = _vectorized_combine(Ks[j1], Ks[j2], log_splits[v], chunk_F)
        Cs[v] = C_v
        Ks[v] = _vectorized_edge(C_v, log_pmfs[v])

    # ---- Final LL at root ------------------------------------------------
    # When p̃_root == 1 (the typical length=+∞ root convention), the LL is
    # just K[root][0]. Otherwise the "ROOTLOSS" variant adds a K[root][1]
    # term — supported here for completeness.
    root = tree.root
    p_root = sp.p[root].to(device=device, dtype=dtype)
    p_root_c = sp.p_c[root].to(device=device, dtype=dtype)
    LL = Ks[root][:, 0]
    if float(p_root_c) > 1e-15:
        # ROOTLOSS: LL = log(K[root][0]·p̃_root + K[root][1]·(1-p̃_root))
        LL = LL + torch.log(p_root)
        if Ks[root].shape[1] >= 2:
            LL = torch.logaddexp(LL, Ks[root][:, 1] + torch.log(p_root_c))
    return LL


# ----------------------------------------------------------------------------
# top-level API
# ----------------------------------------------------------------------------


def empty_log_likelihood_t(sp: SurvivalParamsT) -> Tensor:
    """log P(empty profile) — closed-form, autograd-friendly."""
    LL = torch.zeros((), dtype=sp.p.dtype, device=sp.p.device)
    for v in range(sp.p.shape[0]):
        if bool(sp.is_polya[v]):
            log1_q = sp.log_q_c[v]
            log_k = sp.log_gain[v]
            if log_k.item() == NEG_INF:
                continue
            LL = LL - torch.exp(log_k + torch.log(-log1_q))
        else:
            LL = LL - sp.gain[v]
    LL = LL + sp.log_p[-1]  # log p̃_root; root is index N-1
    return LL


def corrected_log_likelihood_fast(
    tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    profiles: Tensor, *, min_copies: int = 1, W: int,
    chunk_F: int = 1024, use_destructive: bool = False,
) -> Tensor:
    """Top-level: corrected log-likelihood (scalar, summed over families).

    All operations are torch ops, so ``.backward()`` on the result gives the
    analytical gradient w.r.t. (gain, loss, dup, length).
    """
    sp = compute_survival_params_t(tree, gain, loss, dup, length)
    per_family = forward_fast(
        tree, sp, profiles, W, chunk_F=chunk_F, use_destructive=use_destructive,
    )
    LL = per_family.sum()
    if min_copies == 0:
        return LL
    if min_copies >= 2:
        # The Ωmin ≥ 2 correction needs the unobserved-profile probability
        # P(profile_sum < min_copies) which the torch_fast path does NOT
        # compute — only the empty-profile L(0) is available here. Silently
        # using the empty-profile L(0) for Ωmin ≥ 2 under-corrects by the
        # singleton/pair/etc. mass and yields a wrong LL. The native backend
        # has the full Ωmin-aware correction via
        # recount.unobserved_outside.compute_L0_gradient_analytical.
        raise NotImplementedError(
            f"min_copies={min_copies}: the Ωmin ≥ 2 unobserved-profile "
            "correction is not implemented in torch_fast. Use the native "
            "backend instead — `recount.native_backend.corrected_log_likelihood_native(...)` "
            "or `recount.ml.fit_rates(..., backend='native')`. (Both are "
            "also faster than the torch path.)"
        )
    L0 = empty_log_likelihood_t(sp)
    F = profiles.shape[0]
    p_obs = -torch.expm1(L0)  # = 1 - exp(L0)
    return LL - F * torch.log(p_obs)


def gradient_fast(
    tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    profiles: Tensor, *, min_copies: int = 1, W: int, chunk_F: int = 1024,
    use_destructive: bool = False,
) -> Tuple[Tensor, dict]:
    """Compute the corrected log-likelihood AND its gradient w.r.t. all four
    rate-parameter tensors via autograd.

    Returns (LL_scalar, dict of per-rate gradient tensors).
    """
    gain = gain.detach().clone().requires_grad_(True)
    loss = loss.detach().clone().requires_grad_(True)
    dup = dup.detach().clone().requires_grad_(True)
    length = length.detach().clone().requires_grad_(True)
    LL = corrected_log_likelihood_fast(
        tree, gain, loss, dup, length, profiles,
        min_copies=min_copies, W=W, chunk_F=chunk_F,
        use_destructive=use_destructive,
    )
    LL.backward()
    return LL.detach(), {
        "gain": gain.grad, "loss": loss.grad, "dup": dup.grad, "length": length.grad,
    }
