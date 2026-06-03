"""Observation-bias correction L(0) for arbitrary Ωmin (Csurös SI Thms 3-5).

Two implementations:

1. ``unobserved_logL0_native(tree, gain, loss, dup, length, min_copies)``:
   wraps the C implementation in ``native/src/recount_unobserved.c`` (the
   Uinner algorithm from Fig S11 of the SI). Validated against Java's
   ``Likelihood.getEmptyLL() + getSingletonLL()`` to machine precision
   for Ωmin ∈ {1, 2}.

2. ``unobserved_logL0_torch(tree, gain, loss, dup, length, min_copies)``:
   pure-PyTorch port of the same algorithm — autograd-friendly so the
   corrected gradient ``∂log(1-L(0))/∂θ`` falls out for free. Useful for
   ML fitting at arbitrary Ωmin (the shipped CountXXV.jar caps at 2 via
   ``Integer.min(2, table.minCopies())`` in ``Gradient.java`` line 57).

The PyTorch version is much slower per call than the C version (~10× at
typical sizes), but only invoked once per ML iteration (not per family)
so the absolute cost is small.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from recount.tree import Tree


def unobserved_logL0_torch(
    tree: Tree, gain, loss, dup, length, min_copies: int,
):
    """Autograd-friendly log L(0) computation.

    All four rate arrays should be ``torch.Tensor`` of dtype float64.
    Returns a scalar 0-D tensor — call ``.backward()`` to get gradients.

    Implements Algorithm Uinner from Fig S11 of the Csurös 2026 SI: per
    node, maintain ``C[m, n]`` = log P{Ω_u = m | ξ̃_u = n} for
    m, n ∈ [0, M] (M = min_copies − 1), and ``L[m, n, t]`` for the
    parent-side L̃ tensor. K[m, s] derived from C via the per-node edge
    PMF (Poisson or Pólya).

    L(0) = Σ_m K_R[m, 0] (sum of "m surviving copies generate ≤ m total
    leaf copies, starting from 0 ancestral copies at the root").
    """
    import torch

    if min_copies <= 0:
        # No correction
        return torch.tensor(float("-inf"), dtype=gain.dtype)

    M = min_copies - 1
    W = M + 1
    N = tree.num_nodes
    NEG_INF = float("-inf")
    dtype = gain.dtype

    # Build survival params via the existing torch_fast helper (autograd-friendly)
    from recount.torch_fast import compute_survival_params_t
    sp = compute_survival_params_t(tree, gain, loss, dup, length)

    # Per-node C and K tensors, stored as lists of [W, W] tensors so we
    # can rebind slots without in-place mutation (keeps autograd clean).
    C_list = [None] * N
    K_list = [None] * N
    L_list = [None] * N  # [W, W, W]

    # log factorial table for n in [0, M+2]
    log_fact = torch.lgamma(torch.arange(M + 4, dtype=dtype) + 1.0)

    def _safe_log(x):
        return torch.log(x.clamp_min(1e-300))

    NEG_INF_T = torch.tensor(NEG_INF, dtype=dtype)

    def _mul_log_by_int(log_x, n):
        """Csurös' convention: 0 · (-∞) = 0 (the term carries zero weight).

        IEEE-correct multiplication gives NaN; this matches the C helper
        `mul_log_by_count` in native/src/recount_unobserved.c.
        """
        if n == 0:
            return torch.zeros((), dtype=dtype)
        return float(n) * log_x

    for u in range(N):
        # ---- C̃_u,m[n] ----
        if tree.is_leaf[u]:
            # Diagonal: C[m, n] = 0 if n==m else -inf
            mask = torch.eye(W, dtype=dtype)
            C_u = torch.where(mask > 0,
                              torch.zeros((W, W), dtype=dtype),
                              torch.full((W, W), NEG_INF, dtype=dtype))
        else:
            kids = tree.children[u]
            if len(kids) != 2:
                raise NotImplementedError("multifurcations not supported in unobserved-profiles path")
            v, w = int(kids[0]), int(kids[1])
            p_v, pc_v = sp.p[v], sp.p_c[v]
            p_w, pc_w = sp.p_c[w], None  # placeholder
            pc_w = sp.p_c[w]

            # p⁻_v = p_v · (1 - p_w) / (1 - p_v · p_w)
            denom_c = pc_v + p_v * pc_w  # 1 - p_v·p_w (stable)
            denom_c = denom_c.clamp_min(1e-300)
            pminus = (p_v * pc_w) / denom_c
            pminus_c = pc_v / denom_c
            z0 = _safe_log(pminus)
            z1 = _safe_log(pminus_c)

            K_v = K_list[v]   # [W, W] indexed by [m, s]
            L_w = L_list[w]   # [W, W, W] indexed by [m, n, t]
            # Build C_u[m, n] by the recurrence in Fig S11 lines I7-I13.
            # For each (m, n), iterate over (s, t) with s+t=n, s from n down to 0.
            C_u_rows = []
            for m in range(W):
                row = []
                for n in range(W):
                    if n > m:
                        row.append(NEG_INF_T)
                        continue
                    accum = NEG_INF_T
                    s = n
                    t = 0
                    while s >= 0:
                        # inner sum over k = t, t+1, ..., m-s
                        x = NEG_INF_T
                        for k in range(t, m - s + 1):
                            mk = m - k
                            if 0 <= mk < W:
                                Kv = K_v[mk, s]
                                Lwk = L_w[k, n, t]
                                x = torch.logaddexp(x, Kv + Lwk)
                        # binom(n, s) + z1·s + z0·t — guard 0·(-∞)
                        binom = log_fact[n] - log_fact[s] - log_fact[n - s]
                        term = x + binom + _mul_log_by_int(z1, s) + _mul_log_by_int(z0, t)
                        accum = torch.logaddexp(accum, term)
                        s -= 1
                        t += 1
                    row.append(accum)
                C_u_rows.append(torch.stack(row))
            C_u = torch.stack(C_u_rows)
        C_list[u] = C_u

        # ---- K̃_u,m[s] from C̃_u,m via the edge PMF ----
        is_polya = bool(sp.is_polya[u])
        log_q   = sp.log_q[u]
        log_q_c = sp.log_q_c[u]
        log_gain = sp.log_gain[u]
        kappa = sp.gain[u]
        is_root = (u == tree.root)
        s_max = 0 if is_root else M

        K_u_rows = []
        for m in range(W):
            srow = []
            for s in range(W):
                if s > min(m, s_max):
                    srow.append(NEG_INF_T)
                    continue
                Kval = NEG_INF_T
                t = 0
                n = s
                while n <= m:
                    Cv = C_list[u][m, n]
                    # PMF
                    if not is_polya:
                        if log_gain < -500:
                            logpmf = torch.tensor(0.0, dtype=dtype) if t == 0 else NEG_INF_T
                        else:
                            r_u = torch.exp(log_gain)
                            tlog = _mul_log_by_int(log_gain, t)
                            logpmf = -r_u + tlog - log_fact[t]
                    else:
                        if log_gain < -500:
                            logpmf = torch.tensor(0.0, dtype=dtype) if t == 0 else NEG_INF_T
                        else:
                            # log binom(κ+s+t-1, t) = lgamma(κ+s+t) - lgamma(κ+s) - lgamma(t+1)
                            binom = (torch.lgamma(kappa + float(s + t))
                                     - torch.lgamma(kappa + float(s))
                                     - log_fact[t])
                            # (s + κ) · log(1-q): in Polya with finite log_gain we
                            # always have κ > 0, so (s+κ) > 0 and 0·(-∞) doesn't fire.
                            # But guard log_q_c = -∞ explicitly to match Java's
                            # Likelihood.getEmptyLL early-out and the C edge guard
                            # in recount_edge.c line 60.
                            if log_q_c < -500:
                                ks_log1q = NEG_INF_T
                            else:
                                ks_log1q = (float(s) + kappa) * log_q_c
                            tlogq = _mul_log_by_int(log_q, t)
                            logpmf = binom + ks_log1q + tlogq
                    Kval = torch.logaddexp(Kval, Cv + logpmf)
                    t += 1
                    n += 1
                srow.append(Kval)
            K_u_rows.append(torch.stack(srow))
        K_list[u] = torch.stack(K_u_rows)

        # ---- L̃_u,m[n, t] for non-root (Fig S11 I26-I31) ----
        if not is_root:
            log_p_u  = _safe_log(sp.p[u])
            log_pc_u = _safe_log(sp.p_c[u])
            L_u_planes = []
            for m in range(W):
                # L_u[m, n, t] for n in [0..M], t in [0..n]
                plane = [[NEG_INF_T for _ in range(W)] for _ in range(W)]
                for n in range(W):
                    t = n
                    plane[n][t] = K_list[u][m, t]
                    x = plane[n][t]
                    while t > 0:
                        t -= 1
                        x = x + log_pc_u
                        y = plane[n - 1][t] + log_p_u if n >= 1 else NEG_INF_T
                        x = torch.logaddexp(x, y)
                        plane[n][t] = x
                L_u_planes.append(torch.stack([torch.stack(row) for row in plane]))
            L_list[u] = torch.stack(L_u_planes)

    # L(0) = Σ_m K_R[m, 0]
    R = tree.root
    K_R = K_list[R]
    logL0 = K_R[0, 0]
    for m in range(1, W):
        logL0 = torch.logaddexp(logL0, K_R[m, 0])
    return logL0


def unobserved_logL0_native(tree: Tree, gain, loss, dup, length, min_copies: int) -> float:
    """Thin re-export of the native C implementation. Use this for forward-only.

    For the analytical gradient of L(0) wrt (gain, loss, dup, length), use
    ``recount.unobserved_outside.compute_L0_gradient_analytical`` (Phase B
    port of Java FamilySizeLikelihood + LogGradient.getLogSurvivalGradient,
    validated to median 1e-7 rel error against FD on Williams2017).
    The ``unobserved_logL0_grad_fd`` and ``unobserved_logL0_torch`` paths
    below are retained for diagnostics only.
    """
    from recount.native_backend import unobserved_logL0_native as _native
    return _native(tree, gain, loss, dup, length, min_copies)


def unobserved_logL0_grad_fd(
    tree: Tree, gain, loss, dup, length, min_copies: int,
    *, eps_rel: float = 1e-6,
):
    """Central finite-difference gradient of log L(0) w.r.t. (gain, dup, length).

    Returns ``(L0, g_gain, g_dup, g_length)`` numpy arrays. Loss is held
    fixed at 1.0 by convention. Root edge length gradient is 0 (root has
    edge_length = +∞ by convention).

    Cost: 2·(2·N − 1) native L(0) evaluations per call. At ~5 ms each for
    Williams (N=119), that's ~2.4 s per gradient call — acceptable for a
    one-time validation; for production fitting at arbitrary Ωmin, an
    analytical port of SI Theorem 5 (unobservedNodePosteriors) would be
    faster.
    """
    gain = np.asarray(gain, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    dup  = np.asarray(dup,  dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)

    L0_base = unobserved_logL0_native(tree, gain, loss, dup, length, min_copies)
    N = tree.num_nodes
    g_gain = np.zeros(N)
    g_dup  = np.zeros(N)
    g_len  = np.zeros(N)

    def perturb(arr_idx, axis, delta):
        """Return a copy of (gain,dup,length) with one component perturbed."""
        g, d, t = gain.copy(), dup.copy(), length.copy()
        if axis == 0:   g[arr_idx] *= (1.0 + delta)
        elif axis == 1: d[arr_idx] *= (1.0 + delta)
        elif axis == 2: t[arr_idx] *= (1.0 + delta)
        return g, d, t

    for v in range(N):
        for axis, dest in [(0, g_gain), (1, g_dup), (2, g_len)]:
            x = (gain, dup, length)[axis][v]
            # Skip non-finite, zero, or sub-ULP parameters — the FD derivative
            # there is either ambiguous (boundary) or numerically unstable.
            # Treating it as 0 keeps the optimizer well-behaved (Csurös' SI
            # §A.4 explicitly notes the gradient needs L(0) bounded away
            # from 1 and rates bounded away from 0).
            if not np.isfinite(x) or abs(x) < 1e-30:
                continue
            g_plus,  d_plus,  t_plus  = perturb(v, axis, eps_rel)
            g_minus, d_minus, t_minus = perturb(v, axis, -eps_rel)
            L_plus  = unobserved_logL0_native(tree, g_plus,  loss, d_plus,  t_plus,  min_copies)
            L_minus = unobserved_logL0_native(tree, g_minus, loss, d_minus, t_minus, min_copies)
            if not (np.isfinite(L_plus) and np.isfinite(L_minus)):
                # Boundary regime — leave gradient at 0
                continue
            denom = 2.0 * x * eps_rel
            if abs(denom) < 1e-300:
                continue
            val = (L_plus - L_minus) / denom
            if np.isfinite(val):
                dest[v] = val
    return L0_base, g_gain, g_dup, g_len


__all__ = [
    "unobserved_logL0_native",
    "unobserved_logL0_torch",
    "unobserved_logL0_grad_fd",
]
