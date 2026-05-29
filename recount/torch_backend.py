"""PyTorch backend: same forward pass, gradient via autograd.

Same Felsenstein peeling as `recount.gld`, but with `torch` tensors throughout
the forward pass so that `tensor.backward()` yields the gradient by reverse-
mode AD. Matches the analytical NumPy gradient to machine precision on the
canonical 4-leaf test, and remains correct at model boundaries (p_raw=1,
q_raw=0) where finite differences become unstable.

Public API:

    log_likelihood_t(tree, gain, loss, dup, length, profiles)
    corrected_log_likelihood_t(tree, gain, loss, dup, length, profiles, min_copies=1)
    gradient_autograd(tree, gain, loss, dup, length, profiles, min_copies=1)
        → dict {gain, loss, dup, length} of per-node ∂(ln L*)/∂rate.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from recount.gld import GAIN, LOSS, DUP  # noqa: F401  (re-exported for convenience)
from recount.tree import Tree


NEG_INF = float("-inf")


# ---- helpers ----------------------------------------------------------------

def logaddexp(a: Tensor, b: Tensor) -> Tensor:
    return torch.logaddexp(a, b)


def logsumexp(xs: Tensor) -> Tensor:
    return torch.logsumexp(xs, dim=0) if xs.numel() else torch.tensor(NEG_INF, dtype=xs.dtype)


# ---- rate → (p, q) per edge -------------------------------------------------

def _rate_to_pq(mu: Tensor, lam: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-edge (p, p_c, q, q_c) — differentiable.

    Handles the {μ==λ, λ<μ, λ>μ, λ==0, μ==0, t==inf} cases through
    `torch.where` masks so gradients propagate.
    """
    one = torch.tensor(1.0, dtype=mu.dtype, device=mu.device)
    eps_t = torch.tensor(1e-300, dtype=mu.dtype, device=mu.device)

    # μ == 0
    mask_mu0 = mu == 0
    # μ == λ
    mask_eq = (~mask_mu0) & (mu == lam)
    # λ == 0
    mask_lam0 = (~mask_mu0) & (lam == 0)
    # λ < μ
    mask_lt = (~mask_mu0) & (~mask_eq) & (lam < mu) & (~mask_lam0)
    # λ > μ
    mask_gt = (~mask_mu0) & (~mask_eq) & (lam > mu)

    # Common subexpressions for the rate-difference branches
    gap_lt = mu - lam        # positive in mask_lt
    gap_gt = lam - mu        # positive in mask_gt

    # for finite t cases we can compute; for t=inf, we handle via masks
    t_finite = torch.where(torch.isinf(t), torch.zeros_like(t), t)

    # μ == λ
    mu_t = mu * t_finite
    p_eq = mu_t / (1.0 + mu_t)
    p_c_eq = 1.0 / (1.0 + mu_t)
    # If t == inf in mask_eq → p=1, p_c=0
    inf_mask = torch.isinf(t)
    p_eq = torch.where(inf_mask, torch.ones_like(p_eq), p_eq)
    p_c_eq = torch.where(inf_mask, torch.zeros_like(p_c_eq), p_c_eq)
    q_eq = p_eq.clone(); q_c_eq = p_c_eq.clone()  # same formula for q

    # λ == 0 (and μ > 0)
    p_lam0 = -torch.expm1(-mu * t_finite)
    p_c_lam0 = torch.exp(-mu * t_finite)
    p_lam0 = torch.where(inf_mask, torch.ones_like(p_lam0), p_lam0)
    p_c_lam0 = torch.where(inf_mask, torch.zeros_like(p_c_lam0), p_c_lam0)
    q_lam0 = torch.zeros_like(p_lam0)
    q_c_lam0 = torch.ones_like(p_lam0)

    # λ < μ
    d_lt = gap_lt * t_finite
    E_lt = torch.exp(-d_lt)
    E1_lt = -torch.expm1(-d_lt)
    denom_lt = torch.clamp(mu - lam * E_lt, min=eps_t)
    p_lt = mu * E1_lt / denom_lt
    p_c_lt = gap_lt * E_lt / denom_lt
    q_lt = lam * E1_lt / denom_lt
    q_c_lt = gap_lt / denom_lt
    # t == inf in mask_lt: p → 1, q → λ/μ (steady state ratio)
    p_lt_inf = torch.ones_like(p_lt)
    p_c_lt_inf = torch.zeros_like(p_lt)
    q_lt_inf = lam / torch.clamp(mu, min=eps_t)
    q_c_lt_inf = 1.0 - q_lt_inf
    p_lt = torch.where(inf_mask, p_lt_inf, p_lt)
    p_c_lt = torch.where(inf_mask, p_c_lt_inf, p_c_lt)
    q_lt = torch.where(inf_mask, q_lt_inf, q_lt)
    q_c_lt = torch.where(inf_mask, q_c_lt_inf, q_c_lt)

    # λ > μ
    d_gt = gap_gt * t_finite
    E_gt = torch.exp(-d_gt)
    E1_gt = -torch.expm1(-d_gt)
    denom_gt = torch.clamp(lam - mu * E_gt, min=eps_t)
    p_gt = mu * E1_gt / denom_gt
    p_c_gt = gap_gt / denom_gt
    q_gt = lam * E1_gt / denom_gt
    q_c_gt = gap_gt * E_gt / denom_gt
    # t == inf in mask_gt: p → μ/λ, q → 1
    p_gt_inf = mu / torch.clamp(lam, min=eps_t)
    p_c_gt_inf = 1.0 - p_gt_inf
    q_gt_inf = torch.ones_like(q_gt)
    q_c_gt_inf = torch.zeros_like(q_gt)
    p_gt = torch.where(inf_mask, p_gt_inf, p_gt)
    p_c_gt = torch.where(inf_mask, p_c_gt_inf, p_c_gt)
    q_gt = torch.where(inf_mask, q_gt_inf, q_gt)
    q_c_gt = torch.where(inf_mask, q_c_gt_inf, q_c_gt)

    # μ == 0
    p_mu0 = torch.zeros_like(mu)
    p_c_mu0 = torch.ones_like(mu)
    q_mu0 = torch.zeros_like(mu)  # if μ=0 and λ>0, technically not well-defined
    q_c_mu0 = torch.ones_like(mu)

    # Assemble
    p = torch.where(mask_mu0, p_mu0,
        torch.where(mask_eq, p_eq,
        torch.where(mask_lam0, p_lam0,
        torch.where(mask_lt, p_lt, p_gt))))
    p_c = torch.where(mask_mu0, p_c_mu0,
        torch.where(mask_eq, p_c_eq,
        torch.where(mask_lam0, p_c_lam0,
        torch.where(mask_lt, p_c_lt, p_c_gt))))
    q = torch.where(mask_mu0, q_mu0,
        torch.where(mask_eq, q_eq,
        torch.where(mask_lam0, q_lam0,
        torch.where(mask_lt, q_lt, q_gt))))
    q_c = torch.where(mask_mu0, q_c_mu0,
        torch.where(mask_eq, q_c_eq,
        torch.where(mask_lam0, q_c_lam0,
        torch.where(mask_lt, q_c_lt, q_c_gt))))
    return p, p_c, q, q_c


# ---- survival params (bottom-up) -------------------------------------------

def _compute_survival(tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor):
    """Returns dict of per-node tensors: p_raw, p_raw_c, q_raw, q_raw_c, p, p_c, q, q_c,
    gain (= r~ for Poisson, κ for Polya), eps, eps_c, is_polya (bool), log_*.
    """
    p_raw, p_raw_c, q_raw, q_raw_c = _rate_to_pq(loss, dup, length)
    is_polya = q_raw > 0

    n = tree.num_nodes
    # Bottom-up: compute eps, eps_c, then the survival params per node.
    eps_list = []
    eps_c_list = []
    p_tilde_list = []
    p_tilde_c_list = []
    q_tilde_list = []
    q_tilde_c_list = []
    gain_tilde_list = []

    # Pre-fill placeholders to enable indexed reads
    p_tilde_arr = [None] * n
    p_tilde_c_arr = [None] * n
    q_tilde_arr = [None] * n
    q_tilde_c_arr = [None] * n
    gain_tilde_arr = [None] * n
    eps_arr = [None] * n
    eps_c_arr = [None] * n

    zero = torch.tensor(0.0, dtype=gain.dtype, device=gain.device)
    one = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)

    for v in range(n):
        if tree.is_leaf[v]:
            e = zero
            e_c = one
        else:
            e = one
            e_c = zero
            e0 = one
            e1 = zero
            for c in tree.children[v]:
                e1 = e1 + e0 * p_tilde_c_arr[c]
                e0 = e0 * p_tilde_arr[c]
            e = e0
            # If e == 1 exactly, fall back to e1 for numeric stability of (1-e)
            e_c = torch.where(e == 1.0, e1, 1.0 - e)
        eps_arr[v] = e
        eps_c_arr[v] = e_c

        p = p_raw[v]; p_c = p_raw_c[v]
        q = q_raw[v]; q_c = q_raw_c[v]

        # r~ (Poisson) or κ~=κ (Polya)
        gain_tilde_arr[v] = torch.where(is_polya[v], gain[v], gain[v] * e_c)

        a = q_c * e + e_c              # = 1 - q·e (numerically stable form)
        p_tilde_arr[v] = (p * e_c + e * q_c) / a
        q_tilde_arr[v] = q * e_c / a
        b = e_c / a
        p_tilde_c_arr[v] = p_c * b
        q_tilde_c_arr[v] = q_c / a

    p_tilde = torch.stack(p_tilde_arr)
    p_tilde_c = torch.stack(p_tilde_c_arr)
    q_tilde = torch.stack(q_tilde_arr)
    q_tilde_c = torch.stack(q_tilde_c_arr)
    gain_tilde = torch.stack(gain_tilde_arr)
    eps = torch.stack(eps_arr)
    eps_c = torch.stack(eps_c_arr)

    # log-space (clamp to avoid log(0))
    eps_t = torch.tensor(1e-300, dtype=p_tilde.dtype, device=p_tilde.device)
    log_p = torch.log(torch.clamp(p_tilde, min=eps_t))
    log_p_c = torch.log(torch.clamp(p_tilde_c, min=eps_t))
    log_q = torch.log(torch.clamp(q_tilde, min=eps_t))
    log_q_c = torch.log(torch.clamp(q_tilde_c, min=eps_t))
    log_gain = torch.log(torch.clamp(gain_tilde, min=eps_t))

    return dict(
        p_raw=p_raw, p_raw_c=p_raw_c, q_raw=q_raw, q_raw_c=q_raw_c,
        p=p_tilde, p_c=p_tilde_c, q=q_tilde, q_c=q_tilde_c,
        gain=gain_tilde, eps=eps, eps_c=eps_c, is_polya=is_polya,
        log_p=log_p, log_p_c=log_p_c, log_q=log_q, log_q_c=log_q_c,
        log_gain=log_gain,
    )


# ---- factorial helpers ------------------------------------------------------

def _log_factorial(n: int, dtype, device) -> Tensor:
    """log n! via lgamma — autograd-safe (no gradient through n)."""
    return torch.lgamma(torch.tensor(n + 1.0, dtype=dtype, device=device))


def _log_rising_factorial(kappa: Tensor, n: int) -> Tensor:
    """log Γ(κ+n)/Γ(κ) = log((κ)(κ+1)...(κ+n-1))."""
    if n == 0:
        return torch.zeros_like(kappa)
    return torch.lgamma(kappa + n) - torch.lgamma(kappa)


# ---- inside (Felsenstein) pass for one profile -----------------------------

def _calc_widths_t(tree: Tree, profile_row) -> List[int]:
    n = tree.num_nodes
    w = [0] * n
    for v in range(n):
        if tree.is_leaf[v]:
            c = int(profile_row[v])
            w[v] = 0 if c < 0 else c + 1
        else:
            kids = tree.children[v]
            ambi = 0
            total = 0
            for c in kids:
                cn = w[c] - 1
                if cn < 0:
                    ambi += 1
                else:
                    total += cn
            w[v] = 0 if ambi == len(kids) else total + 1
    return w


def _edge_pmf_K(C: Tensor, sp, v: int, K_len: int) -> Tensor:
    """K[v][s] = log Σ_{t : s+t<C_len} C[s+t] + gain_log_pmf(t; κ+s for Polya, r for Poisson)."""
    if C.numel() == 0 or K_len == 0:
        return torch.full((K_len,), NEG_INF, dtype=C.dtype, device=C.device)
    C_len = C.numel()
    dtype = C.dtype; device = C.device
    K = [torch.tensor(NEG_INF, dtype=dtype, device=device)] * K_len

    if not bool(sp["is_polya"][v]):
        log_gain = sp["log_gain"][v]
        r = sp["gain"][v]
        if log_gain == NEG_INF:
            for s in range(min(K_len, C_len)):
                K[s] = C[s]
        else:
            for s in range(K_len):
                terms = []
                ell = s
                t = 0
                while ell < C_len:
                    t_logr = t * log_gain if t > 0 else torch.zeros_like(log_gain)
                    terms.append(C[ell] - r + t_logr - _log_factorial(t, dtype, device))
                    t += 1
                    ell += 1
                if terms:
                    K[s] = torch.logsumexp(torch.stack(terms), dim=0)
        return torch.stack(K)

    # Polya
    log_q = sp["log_q"][v]
    log_q_c = sp["log_q_c"][v]
    log_kappa = sp["log_gain"][v]
    kappa = sp["gain"][v]
    if log_kappa == NEG_INF:
        for s in range(min(K_len, C_len)):
            K[s] = C[s]
        return torch.stack(K)
    # loglog1_q = log(-log(1-q)) — exists for 0<q<1
    loglog1_q = torch.log(-log_q_c)
    kappa_log1_q = torch.exp(log_kappa + loglog1_q)  # = κ · |log(1-q)|

    for s in range(K_len):
        terms = []
        ell = s
        t = 0
        # rising factorial at index ell vs s: log Γ(κ+ell)/Γ(κ+s)
        log_rf_s = _log_rising_factorial(kappa, s)
        while ell < C_len:
            log_rf_ell = _log_rising_factorial(kappa, ell)
            binom = log_rf_ell - log_rf_s - _log_factorial(t, dtype, device)
            t_logq = t * log_q if t > 0 else torch.zeros_like(log_q)
            ks_log1_q = s * log_q_c - kappa_log1_q
            terms.append(C[ell] + binom + ks_log1_q + t_logq)
            t += 1
            ell += 1
        if terms:
            K[s] = torch.logsumexp(torch.stack(terms), dim=0)
    return torch.stack(K)


def _compute_sibling_t(C: Tensor, K_junior: Tensor, p_junior: Tensor, p_junior_c: Tensor,
                       eps_sib: Tensor, dtype, device) -> Tensor:
    """PyTorch port of count_gld._compute_sibling (forward sibling combination)."""
    if (isinstance(eps_sib, Tensor) and eps_sib.item() == 1.0):
        return K_junior.clone() if K_junior.numel() > 0 else C.clone()
    if C.numel() == 0:
        return K_junior.clone()
    if K_junior.numel() == 0:
        return C

    K = K_junior
    K_len = K.numel()
    C_len = C.numel()
    combined = (K_len - 1) + (C_len - 1)

    p = p_junior; p_c = p_junior_c
    log_e = torch.log(eps_sib)
    log_e_c = torch.log1p(-eps_sib)
    log_a = torch.log1p(-p * eps_sib)
    logp1 = torch.log(p_c) - log_a
    logp2 = torch.log(p) + log_e_c - log_a

    # work on a copy of C since we mutate
    Cw = [C[i] for i in range(C_len)]
    C2 = [torch.tensor(NEG_INF, dtype=dtype, device=device)] * (combined + 1)
    neg_inf_t = torch.tensor(NEG_INF, dtype=dtype, device=device)

    for ell in range(combined + 1):
        # destructive update of Cw[t]
        t = min(ell, C_len)
        x = Cw[t] if t < C_len else neg_inf_t
        s = ell - t
        while t > 0 and s < K_len - 1:
            t -= 1
            s += 1
            x = x + log_e_c
            y = Cw[t] + log_e
            x = torch.logaddexp(x, y)
            Cw[t] = x
        # gather terms
        log_ellfact = _log_factorial(ell, dtype, device)
        tmin = t
        terms = []
        while s >= 0 and t < C_len:
            log_sfact = _log_factorial(s, dtype, device)
            log_tfact = _log_factorial(t, dtype, device)
            binom = log_ellfact - log_sfact - log_tfact
            slogp1 = (s * logp1) if s > 0 else torch.zeros_like(logp1)
            tlogp2 = (t * logp2) if t > 0 else torch.zeros_like(logp2)
            terms.append(K[s] + Cw[t] + binom + slogp1 + tlogp2)
            s -= 1
            t += 1
        if terms:
            C2[ell] = torch.logsumexp(torch.stack(terms), dim=0)
    return torch.stack(C2)


def _profile_LL_t(tree: Tree, sp, profile_row, dtype, device) -> Tensor:
    """Felsenstein peeling for one profile → log P(profile), autograd-tracked."""
    n = tree.num_nodes
    width = _calc_widths_t(tree, profile_row)
    C_list: List[Tensor] = [None] * n  # type: ignore[list-item]
    K_list: List[Tensor] = [None] * n  # type: ignore[list-item]
    neg_inf_t = torch.tensor(NEG_INF, dtype=dtype, device=device)

    for v in range(n):
        w = width[v]
        if tree.is_leaf[v]:
            count = int(profile_row[v])
            Cv = [neg_inf_t.clone() for _ in range(w)]
            if w > 0:
                Cv[w - 1] = torch.zeros((), dtype=dtype, device=device)
            C_list[v] = torch.stack(Cv) if Cv else torch.empty(0, dtype=dtype, device=device)
        else:
            if sp["eps"][v].item() == 1.0:
                acc = torch.zeros((), dtype=dtype, device=device)
                for c in tree.children[v]:
                    if K_list[c].numel() > 0:
                        acc = acc + K_list[c][0]
                Cv_t = torch.stack([acc])
            else:
                Cv_t = torch.empty(0, dtype=dtype, device=device)
                sib_extinct = torch.ones((), dtype=dtype, device=device)
                for c in tree.children[v]:
                    Cv_t = _compute_sibling_t(
                        Cv_t, K_list[c], sp["p"][c], sp["p_c"][c], sib_extinct, dtype, device
                    )
                    sib_extinct = sib_extinct * sp["p"][c]
            # Truncate / pad to w
            if Cv_t.numel() > w:
                Cv_t = Cv_t[:w]
            elif Cv_t.numel() < w:
                pad = torch.full((w - Cv_t.numel(),), NEG_INF, dtype=dtype, device=device)
                Cv_t = torch.cat([Cv_t, pad])
            C_list[v] = Cv_t

        if v == tree.root:
            K_len = 1 if sp["log_p_c"][v].item() == NEG_INF else 2
        else:
            K_len = width[v]
        K_list[v] = _edge_pmf_K(C_list[v], sp, v, K_len)

    # Final LL
    root = tree.root
    Kr = K_list[root]
    LL = Kr[0]
    p_root = sp["p"][root]
    if p_root.item() != 1.0 and Kr.numel() == 2:
        LL = LL + torch.log(p_root)
        LL = torch.logaddexp(LL, Kr[1] + torch.log(sp["p_c"][root]))
    return LL


def _empty_LL_t(tree: Tree, sp) -> Tensor:
    """log P(empty profile) — closed-form, autograd-tracked."""
    LL = torch.zeros((), dtype=sp["p"].dtype, device=sp["p"].device)
    for v in range(tree.num_nodes):
        if bool(sp["is_polya"][v]):
            log1_q = sp["log_q_c"][v]
            log_k = sp["log_gain"][v]
            if log_k == NEG_INF:
                continue
            loglog1_q = torch.log(-log1_q)
            LL = LL - torch.exp(log_k + loglog1_q)
        else:
            LL = LL - sp["gain"][v]
    LL = LL + sp["log_p"][tree.root]
    return LL


# ---- public API -------------------------------------------------------------

def log_likelihood_t(tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
                     profiles) -> Tensor:
    """Raw (uncorrected) log-likelihood as a torch scalar."""
    sp = _compute_survival(tree, gain, loss, dup, length)
    LL = torch.zeros((), dtype=gain.dtype, device=gain.device)
    for f in range(profiles.shape[0]):
        LL = LL + _profile_LL_t(tree, sp, profiles[f], gain.dtype, gain.device)
    return LL


def corrected_log_likelihood_t(tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
                                profiles, min_copies: int = 1) -> Tensor:
    """Corrected log-likelihood as a torch scalar (autograd-tracked)."""
    sp = _compute_survival(tree, gain, loss, dup, length)
    LL = torch.zeros((), dtype=gain.dtype, device=gain.device)
    for f in range(profiles.shape[0]):
        LL = LL + _profile_LL_t(tree, sp, profiles[f], gain.dtype, gain.device)
    if min_copies == 0:
        return LL
    L0 = _empty_LL_t(tree, sp)
    if min_copies == 2:
        # singleton terms — sum log P over each single-copy-at-one-leaf profile
        log_p_singletons = []
        L = tree.num_leaves
        for ell in range(L):
            prof = torch.zeros(tree.num_leaves, dtype=torch.long)
            prof[ell] = 1
            log_p_singletons.append(_profile_LL_t(tree, sp, prof, gain.dtype, gain.device))
        L1 = torch.logsumexp(torch.stack(log_p_singletons), dim=0)
        L0 = torch.logaddexp(L0, L1)
    F = profiles.shape[0]
    p_obs = -torch.expm1(L0)  # 1 - exp(L0)
    return LL - F * torch.log(p_obs)


def gradient_autograd(
    tree: Tree, gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    profiles: Tensor, min_copies: int = 1,
) -> Dict[str, Tensor]:
    """Compute ∂(ln L*)/∂{gain, loss, dup, length} via torch.autograd.

    Returns a dict of per-node gradients, each a tensor of shape (num_nodes,).
    Equivalent in spirit to the NumPy analytical gradient but expressed w.r.t.
    the *raw* rate parameters (μ, λ, γ, t) — the chain rule from survival
    (p̃, q̃, r̃/κ̃) to raw is handled automatically by reverse-mode AD.
    """
    gain = gain.detach().clone().requires_grad_(True)
    loss = loss.detach().clone().requires_grad_(True)
    dup = dup.detach().clone().requires_grad_(True)
    length = length.detach().clone().requires_grad_(True)

    LL = corrected_log_likelihood_t(tree, gain, loss, dup, length, profiles, min_copies)
    LL.backward()

    return {"gain": gain.grad, "loss": loss.grad, "dup": dup.grad, "length": length.grad}
