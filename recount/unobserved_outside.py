"""Outside-pass for the SI Theorems 3-5 unobserved-profile correction.

Port of Java count.model.FamilySizeLikelihood (computeAll + UnobservedProfile
+ MockPosteriors). Provides the building blocks for the analytical L(0)
gradient at arbitrary Ωmin — exactly what the shipped Count uses via
LogGradient.setMinimumObservedCopies(k) + FamilySizeLikelihood.getUnobservedProfile
when k ≥ 3 (LogGradient.java line 384).

Status:
  ✓ Inside pass — implemented in native C via recount_unobserved_inside_tensors;
                  matches Java FSL to 1.1e-16 on Williams2017 at Ωmin=4.
  ✓ Outside pass — implemented in numpy here (compute_unobserved_outside);
                  recovers L(0) to machine precision via the root combine.
  ✓ Posteriors — implemented here (compute_unobserved_posteriors); row sums
                 = 1.0 exactly at every node on Williams2017.
  ✓ Gradient — compute_L0_gradient_analytical at the bottom of this file
               ports LogGradient.PosteriorStatistics.getLogSurvivalGradient
               (Java lines 600-777) and chains to (gain, loss, dup, length)
               via torch autograd through compute_survival_params_t.
               Wired into recount.native_backend._gradient_raw_native for
               min_copies ≥ 2; FD path retired except for diagnostic use.

Algorithm:
  inside  — Csurös SI Algorithm Uinner (Fig S11): per-size tensors
            C̃_u,m[n] = log P{Ω_u = m | ξ̃_u = n}     (node)
            K̃_u,m[s] = log P{Ω_u = m | η̃_u = s}     (edge)
            Already implemented in native C via
            recount.native_backend.unobserved_inside_tensors_native.

  outside — Csurös SI Algorithm Uouter (Fig S12), in this code:
            B̃_u,m[n] = log P{ξ̃_u = n AND profile = m-from-elsewhere}
            J̃_v,m[s] = log P{η̃_v = s AND profile = m-from-elsewhere}
            L̃_w,m[ell, t] = pairing likelihoods (FamilySizeLikelihood
            line 195-225). Computed here in numpy.

  posteriors — log P{η̃_v, ξ̃_v | profile sum < min_copies}, summed over m.

  gradient — Theorem tm:Ld (Cor cor:loglik.d) in phylobd-gradient.tex:
             ∂(log L*)/∂q̃_v = (Ñ_v - S̃_v)/q̃_v
                              - (S̃_v + κ_v/(1-L(0)))/(1-q̃_v)
             ∂(log L*)/∂p̃_v = Ñ_v/p̃_v - (1-ε)·S̃_y/(1-p̃_v)
             ∂(log L*)/∂κ_v = F·log(1-q̃_v)/(1-L(0))
                              + Σ_i (Ñ_v^{>i} - S̃_v^{>i})/(κ_v+i)

The shipped Java code at LogGradient.java:382-433 dispatches to this
same FamilySizeLikelihood path when min_copies ≥ USE_FAMILY_SIZE_LIKELIHOOD
(= 3) — so this is the published reference algorithm for arbitrary Ωmin.

Cost: O(N·M³) per call for the outside (dominated by sibling-pair
combine inside computeOuterEdgeTransitions). For Williams (N=119, M=3)
this is ~7000 elementary log-add operations — well under 10ms in numpy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from recount.tree import Tree
from recount.native_backend import (
    _compute_survival,
    _lib,
    unobserved_inside_tensors_native,
)
import ctypes


NEG_INF = float("-inf")


def _logadd(a: float, b: float) -> float:
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _logsumexp(arr) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return NEG_INF
    mfin = arr[np.isfinite(arr)]
    if mfin.size == 0:
        return NEG_INF
    m = mfin.max()
    return float(m + np.log(np.exp(arr - m).sum()))


def _factln_table(W: int) -> np.ndarray:
    """log n! for n in [0, W-1]."""
    out = np.zeros(W + 4)
    s = 0.0
    for n in range(1, W + 4):
        s += np.log(n)
        out[n] = s
    return out


def _safe_log(x: float) -> float:
    return np.log(x) if x > 0 else NEG_INF


def _get_survival_arrays(tree: Tree, gain, loss, dup, length):
    """Pull C-computed survival params into numpy arrays (bit-exact w/ native)."""
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    N = int(tree.num_nodes)
    def _g(ptr):
        return np.frombuffer(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * N))[0],
            dtype=np.float64,
        ).copy()
    sp = {
        "p":     _g(sp_struct.p),
        "p_c":   _g(sp_struct.p_c),
        "q":     _g(sp_struct.q),
        "q_c":   _g(sp_struct.q_c),
        "gain":  _g(sp_struct.gain),
        "eps":   _g(sp_struct.eps),
        "eps_c": _g(sp_struct.eps_c),
        "log_p":     _g(sp_struct.log_p),
        "log_p_c":   _g(sp_struct.log_p_c),
        "log_q":     _g(sp_struct.log_q),
        "log_q_c":   _g(sp_struct.log_q_c),
        "log_gain":  _g(sp_struct.log_gain),
        "is_polya":  np.frombuffer(
            ctypes.cast(sp_struct.is_polya,
                        ctypes.POINTER(ctypes.c_uint8 * N))[0],
            dtype=np.uint8).copy(),
    }
    _lib.recount_survival_free(ctypes.byref(sp_struct))
    return sp


def compute_pairing_likelihoods(
    K_all: np.ndarray, log_pw: float, log_pw_c: float, w: int, M: int,
) -> np.ndarray:
    """L̃_w,m[ell, t] per FamilySizeLikelihood.getPairingLikelihoods (line 195).

    For each size m ∈ [0, M], computes a (M+1)×(m+1) tensor.
    Returns Lw[m, ell, t] padded with -inf where (ell, t) out of range.
    """
    W = M + 1
    Lw = np.full((W, W, W), NEG_INF, dtype=np.float64)
    for m in range(W):
        Kw = K_all[w, m, :m + 1]  # K̃_w,m[s] for s in [0..m]
        for ell in range(W):
            t = ell
            if t <= m:
                x = Kw[t]
                Lw[m, ell, t] = x
            else:
                x = NEG_INF
                t = m + 1  # signal: skip the inner loop
            while t > 0:
                t -= 1
                # x ← x + log(1-p̃_w);  y ← Lw[m, ell-1, t] + log(p̃_w)
                x = x + log_pw_c if x != NEG_INF and log_pw_c != NEG_INF else (
                    x if log_pw_c != NEG_INF else NEG_INF)
                if ell >= 1:
                    y = Lw[m, ell - 1, t] + log_pw if (
                        Lw[m, ell - 1, t] != NEG_INF and log_pw != NEG_INF
                    ) else NEG_INF
                    x = _logadd(x, y)
                Lw[m, ell, t] = x
    return Lw


def _children(tree: Tree, parent_of: np.ndarray):
    """Build child lists per node from parent array."""
    N = int(tree.num_nodes)
    kids = [[] for _ in range(N)]
    for v, p in enumerate(parent_of):
        if p >= 0:
            kids[p].append(v)
    return kids


def compute_unobserved_outside(
    tree: Tree, sp: dict, K_all: np.ndarray, min_copies: int,
):
    """Per-size outside tensors B̃[m, v, n] (node) and J̃[m, v, s] (edge).

    Mirrors FamilySizeLikelihood.computeOuterEdgeTransitions / NodeTransitions.
    For each m ∈ [0, M], the tensors give the joint log-probability of the
    OUTSIDE event ("subtree-complement profile sum = M-m") at value n or s
    at node v.

    Returns (B_all, J_all, Lw_all) where:
      B_all[m, v, n] = log P{ξ̃_v = n AND complement_sum = M - m}
      J_all[m, v, s] = log P{η̃_v = s AND complement_sum = M - m}
      Lw_all[w, m, ell, t] = pairing likelihoods for w (cached for posteriors)
    """
    if min_copies <= 0:
        raise ValueError("min_copies must be ≥ 1")
    M = min_copies - 1
    W = M + 1
    N = int(tree.num_nodes)
    R = int(tree.root)

    parent = np.asarray(tree.parent, dtype=np.int32)
    kids = _children(tree, parent)
    fact = _factln_table(W + 4)

    # Pairing likelihoods per node w (used by parent's outside)
    Lw_all = np.full((N, W, W, W), NEG_INF, dtype=np.float64)
    for w in range(N):
        if w == R:
            continue
        Lw_all[w] = compute_pairing_likelihoods(
            K_all, sp["log_p"][w], sp["log_p_c"][w], w, M)

    # Outside tensors
    B_all = np.full((W, N, W), NEG_INF, dtype=np.float64)
    J_all = np.full((W, N, W), NEG_INF, dtype=np.float64)
    # We also need the joint Jns/Bns (n,s) tensors for posteriors
    Jns_all = np.full((W, N, W, W), NEG_INF, dtype=np.float64)
    Bns_all = np.full((W, N, W, W), NEG_INF, dtype=np.float64)

    # Initialize root: J̃_R,m[s] is 1 at s=0 m=0, 0 elsewhere
    # (per FSL.computeOuterEdgeTransitions line 334-342: when root has
    # p_root=1, Jns has just [0][0] = 0 if m==0 else -inf)
    log_pc_root = sp["log_p_c"][R]
    if log_pc_root == NEG_INF:
        for m in range(W):
            if m == 0:
                Jns_all[m, R, 0, 0] = 0.0
                J_all[m, R, 0] = 0.0
    else:
        # ROOTLOSS — not supported (FSL throws too)
        raise NotImplementedError("Root with loss < 1 not supported")

    # Compute B[root] = node-outside at root via gain PMF (turn J → B)
    for m in range(W):
        Bns_all[m, R], B_all[m, R] = _edge_outside_to_node(
            J_all[m, R], sp, R, M, fact)

    # Pre-order: descend from root's children
    order = []
    visited = [False] * N
    visited[R] = True
    stack = [R]
    while stack:
        u = stack.pop()
        for c in kids[u]:
            if not visited[c]:
                visited[c] = True
                order.append(c)
                stack.append(c)
    # order is pre-order excluding root

    for v in order:
        u = int(parent[v])
        sibs = [c for c in kids[u] if c != v]
        if len(sibs) != 1:
            raise NotImplementedError("Only binary trees supported here")
        w = sibs[0]
        # computeOuterEdgeTransitions(v): Jns[m, v, n, s]
        logit_pv = sp["log_p"][v] - sp["log_p_c"][v]
        log1_e = sp["log_p_c"][w]
        logit_p = logit_pv + log1_e
        # log_p = logit_to_log_value, log1_p = logit_to_log_complement
        if logit_p >= 0:
            # log(p) where p large; use stable form
            log_p = -np.log1p(np.exp(-logit_p))
            log1_p = -logit_p - np.log1p(np.exp(-logit_p))
        else:
            log_p = logit_p - np.log1p(np.exp(logit_p))
            log1_p = -np.log1p(np.exp(logit_p))

        for m in range(W):
            for n in range(W):
                lognf = fact[n]
                for s in range(n + 1):
                    BuLw = NEG_INF
                    t = n - s
                    # Iterate over mw = size assigned to sibling subtree
                    for mw in range(m + 1):
                        if M - m + mw > M:
                            continue  # not enough room
                        # We compute size m for node v: complement has size M-m
                        # The split: this edge has "remaining" m - mw from
                        # v's subtree side and mw from w's side.
                        # Bu indexed by sizeProfiles[m - mw]
                        Bu_size = m - mw
                        if Bu_size < 0 or Bu_size >= W: continue
                        Bu_n = B_all[Bu_size, u, n] if n < W else NEG_INF
                        Lw_t = Lw_all[w, mw, n, t] if (
                            n < W and t < W and t <= mw) else NEG_INF
                        if Bu_n == NEG_INF or Lw_t == NEG_INF:
                            continue
                        BuLw = _logadd(BuLw, Bu_n + Lw_t)
                    binom = lognf - fact[s] - fact[t]
                    slog1_p = 0.0 if s == 0 else s * log1_p
                    tlog_p  = 0.0 if t == 0 else t * log_p
                    Jns_all[m, v, n, s] = BuLw + binom + slog1_p + tlog_p

            # J̃_v,m[s] = sum over n of Jns[m, v, n, s] (s ≤ n ≤ M)
            for s in range(W):
                acc = NEG_INF
                for n in range(s, W):
                    val = Jns_all[m, v, n, s]
                    if val != NEG_INF:
                        acc = _logadd(acc, val)
                J_all[m, v, s] = acc

        # B̃_v,m[n] via edge_outside_to_node
        for m in range(W):
            Bns_all[m, v], B_all[m, v] = _edge_outside_to_node(
                J_all[m, v], sp, v, M, fact)

    # Cumulate outside arrays in m (FSL.computeAll line 112-147):
    # after this transform,
    #   B_cumul[m, v, n] = log P{ξ̃_v_outside = n AND complement_size ≤ m}
    #   J_cumul[m, v, s] = log P{η̃_v_outside = s AND complement_size ≤ m}
    # which is what the posterior formulas in FSL.UnobservedProfile.getLog*
    # expect (Java mutates the arrays in place).
    for m in range(1, W):
        for v in range(N):
            for n in range(W):
                B_all[m, v, n] = _logadd(B_all[m, v, n], B_all[m - 1, v, n])
            for s in range(W):
                J_all[m, v, s] = _logadd(J_all[m, v, s], J_all[m - 1, v, s])
            # Joint tensors too — needed for transition posteriors
            for n in range(W):
                for s in range(W):
                    Bns_all[m, v, n, s] = _logadd(Bns_all[m, v, n, s],
                                                  Bns_all[m - 1, v, n, s])
                    Jns_all[m, v, n, s] = _logadd(Jns_all[m, v, n, s],
                                                  Jns_all[m - 1, v, n, s])

    return B_all, J_all, Bns_all, Jns_all, Lw_all


def _edge_outside_to_node(Jv: np.ndarray, sp: dict, v: int, M: int, fact: np.ndarray):
    """B̃_v,m[n] from J̃_v,m[s] via the gain PMF (per Java
    FamilySizeLikelihood.computeOuterNodeTransitions, line 398).

    Returns (Bns[n, s], B[n]) where Bns is the joint and B sums over s.
    """
    W = M + 1
    is_polya = bool(sp["is_polya"][v])
    log_q   = sp["log_q"][v]
    log_q_c = sp["log_q_c"][v]
    log_gain = sp["log_gain"][v]
    kappa = sp["gain"][v]

    Bns = np.full((W, W), NEG_INF, dtype=np.float64)

    if not is_polya:
        # Poisson
        if log_gain == NEG_INF:
            # No gain: B[n] = J[n] at s=n, 0 elsewhere
            for n in range(W):
                if n < Jv.size:
                    Bns[n, n] = Jv[n]  # t=0, s=n
        else:
            r = np.exp(log_gain)
            for n in range(W):
                t = n
                s = 0
                while s < W and s <= n:
                    if s < Jv.size and Jv[s] != NEG_INF:
                        tlogr = 0.0 if t == 0 else t * log_gain
                        Bns[n, s] = Jv[s] - r + tlogr - fact[t]
                    s += 1
                    t -= 1
    else:
        # Pólya
        if log_gain == NEG_INF:
            for n in range(W):
                if n < Jv.size:
                    Bns[n, n] = Jv[n]
        else:
            log_kappa = log_gain
            loglog1_q = np.log(-log_q_c) if log_q_c < 0 else log_q
            for n in range(W):
                t = n
                s = 0
                while s < W and s <= n:
                    if s < Jv.size and Jv[s] != NEG_INF:
                        # Pólya PMF with starting state s:
                        # log P{gain t | s starting} = binom(κ+s+t-1, t) ·
                        #                              (1-q)^{κ+s} · q^t
                        # lgamma(κ+s+t) − lgamma(κ+s) computed as direct sum
                        # log(κ+s) + log(κ+s+1) + ... + log(κ+s+t-1) to avoid
                        # the catastrophic cancellation lgamma() suffers
                        # when κ is huge (publication-regime κ up to 5×10^13
                        # violate Java's Logistic(33) — see GAIN_CONSTRAINT.md).
                        from math import log as math_log
                        rising = 0.0
                        for i in range(t):
                            rising += math_log(kappa + s + i)
                        binom = rising - fact[t]
                        if log_q_c == NEG_INF:
                            ks_l1q = NEG_INF
                        else:
                            ks_l1q = (kappa + s) * log_q_c
                        tlogq = 0.0 if t == 0 else t * log_q
                        Bns[n, s] = Jv[s] + binom + ks_l1q + tlogq
                    s += 1
                    t -= 1

    B = np.full(W, NEG_INF, dtype=np.float64)
    for n in range(W):
        # B[n] = logsumexp over s of Bns[n, s]
        terms = Bns[n, :n + 1]
        finite = terms[np.isfinite(terms)]
        if finite.size > 0:
            m = finite.max()
            B[n] = float(m + np.log(np.exp(terms - m).sum(where=np.isfinite(terms))))
    return Bns, B


def compute_unobserved_log_likelihood(
    K_all: np.ndarray, J_all: np.ndarray, root: int, M: int,
) -> float:
    """L(0) = log P{profile sum ≤ M} (the unobserved-profile mass).

    Equals Java FamilySizeLikelihood.UnobservedProfile.getLogLikelihood:
        LL = logsumexp over (m, s) of [J_cumul[M-m, root, s] + K_root,m[s]]
    """
    W = M + 1
    R = int(root)
    LL = NEG_INF
    for m in range(W):
        for s in range(W):
            Js = J_all[M - m, R, s] if (M - m) < W else NEG_INF
            Ks = K_all[R, m, s] if s <= m else NEG_INF
            if Js != NEG_INF and Ks != NEG_INF:
                LL = _logadd(LL, Js + Ks)
    return LL


def compute_unobserved_posteriors(
    C_all: np.ndarray, K_all: np.ndarray, B_all: np.ndarray, J_all: np.ndarray,
    log_L0: float, M: int,
):
    """Per-node posteriors under the unobserved profile distribution.

    Returns:
      log_node_post[v, n] = log P{ξ̃_v = n | profile sum ≤ M}
      log_edge_post[v, s] = log P{η̃_v = s | profile sum ≤ M}

    Mirrors Java FamilySizeLikelihood.UnobservedProfile.getLogNodePosteriors
    (line 666) and getLogEdgePosteriors (line 640).
    """
    W = M + 1
    N = J_all.shape[1]
    log_node_post = np.full((N, W), NEG_INF, dtype=np.float64)
    log_edge_post = np.full((N, W), NEG_INF, dtype=np.float64)

    for v in range(N):
        # Edge posteriors: P{η̃_v = s | unobs}
        for s in range(W):
            z = NEG_INF
            for m in range(s, W):
                Js = J_all[M - m, v, s] if (M - m) < W else NEG_INF
                Ks = K_all[v, m, s] if s <= m else NEG_INF
                if Js != NEG_INF and Ks != NEG_INF:
                    z = _logadd(z, Js + Ks)
            log_edge_post[v, s] = z - log_L0

        # Node posteriors: P{ξ̃_v = n | unobs}
        for n in range(W):
            z = NEG_INF
            for m in range(n, W):
                Bn = B_all[M - m, v, n] if (M - m) < W else NEG_INF
                Cn = C_all[v, m, n] if n <= m else NEG_INF
                if Bn != NEG_INF and Cn != NEG_INF:
                    z = _logadd(z, Bn + Cn)
            log_node_post[v, n] = z - log_L0

    return log_node_post, log_edge_post


def compute_transition_posteriors(
    C_all: np.ndarray, K_all: np.ndarray,
    Bns_all: np.ndarray, Jns_all: np.ndarray,
    log_L0: float, M: int,
):
    """Joint (ξ̃_v=n, η̃_v=s) posteriors at the node, and (ξ̃_u=n, η̃_v=s)
    posteriors across the edge entering v.

    Returns (log_node_trans, log_edge_trans) — both shaped [N, W, W] with
    -inf where s > n.

    Mirrors FamilySizeLikelihood.UnobservedProfile.getLogNodeTransitionPosteriors
    (line 611) and getLogEdgeTransitionPosteriors (line 552).
    """
    W = K_all.shape[2]
    N = K_all.shape[0]
    log_node_trans = np.full((N, W, W), NEG_INF, dtype=np.float64)
    log_edge_trans = np.full((N, W, W), NEG_INF, dtype=np.float64)

    for v in range(N):
        # Node transitions: P{ξ̃_v=n AND η̃_v=s | unobs}
        # = Σ_{m=n..M} Bns_all[M-m, v, n, s] · C_all[v, m, n] / L(0)
        for n in range(W):
            for s in range(n + 1):
                z = NEG_INF
                for m in range(n, W):
                    Bnsm = Bns_all[M - m, v, n, s] if (M - m) < W else NEG_INF
                    Cn   = C_all[v, m, n]
                    if Bnsm != NEG_INF and Cn != NEG_INF:
                        z = _logadd(z, Bnsm + Cn)
                log_node_trans[v, n, s] = z - log_L0

        # Edge transitions: P{ξ̃_u=n AND η̃_v=s | unobs}
        # = Σ_{m=s..M} Jns_all[M-m, v, n, s] · K_all[v, m, s] / L(0)
        for n in range(W):
            for s in range(n + 1):
                z = NEG_INF
                for m in range(s, W):
                    Jnsm = Jns_all[M - m, v, n, s] if (M - m) < W else NEG_INF
                    Ks = K_all[v, m, s] if s <= m else NEG_INF
                    if Jnsm != NEG_INF and Ks != NEG_INF:
                        z = _logadd(z, Jnsm + Ks)
                log_edge_trans[v, n, s] = z - log_L0
    return log_node_trans, log_edge_trans


def log_tail_difference(log_trans_v: np.ndarray) -> np.ndarray:
    """Port of Java Posteriors.logTailDifference (line 158).

    Input log_trans_v[n, s] = log P{N=n, S=s} (n in [0,W-1], s in [0,n]).
    Output N_S[ell] for ell in [0,W-1]:
        N_S[ell] = log Σ_s Σ_{n>ell} exp(log_trans_v[n, s])
                 = log (P{N > ell} - P{S > ell})
    (since S ≤ N implies P{S>ell} ⊆ P{N>ell} and N_S = P{N>ell, S≤ell}.)
    """
    W = log_trans_v.shape[0]
    N_S = np.full(W, NEG_INF, dtype=np.float64)
    for s in range(W):
        tail = NEG_INF
        for ell in range(W - 1, s - 1, -1):
            N_S[ell] = _logadd(N_S[ell], tail)
            if s <= ell:
                tail = _logadd(tail, log_trans_v[ell, s])
    return N_S


def compute_birth_death_tails(
    log_node_trans: np.ndarray, log_edge_trans: np.ndarray,
):
    """log_birth_tails[v, ell] = log (P{ξ̃_v > ell} - P{η̃_v > ell} | unobs)
    log_death_tails[v, ell] = log (P{ξ̃_u > ell} - P{η̃_v > ell} | unobs)
    using log_tail_difference applied to the joint posteriors.
    """
    N, W, _ = log_node_trans.shape
    birth = np.full((N, W), NEG_INF, dtype=np.float64)
    death = np.full((N, W), NEG_INF, dtype=np.float64)
    for v in range(N):
        birth[v] = log_tail_difference(log_node_trans[v])
        death[v] = log_tail_difference(log_edge_trans[v])
    return birth, death


def _log_sum(arr: np.ndarray) -> float:
    """logsumexp of a 1-D array, with -inf inputs handled."""
    if arr.size == 0:
        return NEG_INF
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return NEG_INF
    m = finite.max()
    return float(m + np.log(np.exp(arr - m).sum(where=np.isfinite(arr))))


def compute_log_survival_gradient_unobs(
    tree: Tree, sp: dict,
    log_edge_post: np.ndarray, log_birth_tails: np.ndarray,
    log_death_tails: np.ndarray, profile_count: float,
):
    """Port of Java LogGradient.PosteriorStatistics.getLogSurvivalGradient
    (lines 600-777) for the unobserved-profile statistics.

    Returns (d_logit_p, d_logit_q, d_log_kappa) each of shape [N] in linear
    space — the gradient of log L(<min_copies) with respect to the
    LOGIT-parameterized survival params (logit p̃, logit q̃, log κ̃).

    For the corrected ML, this gets combined as:
        ∂(log L*)/∂x = ∂(log L_obs)/∂x + F·L(0)/(1-L(0)) · ∂(log L_unobs)/∂x
    where F is the OBSERVED-DATA family count (not profile_count here, which
    is 1 for the unobserved profile).
    """
    N = int(tree.num_nodes)
    W = log_edge_post.shape[1]
    root = int(tree.root)
    parent = np.asarray(tree.parent, dtype=np.int32)
    kids = _children(tree, parent)

    p = sp["p"]; p_c = sp["p_c"]
    q = sp["q"]; q_c = sp["q_c"]
    kappa = sp["gain"]
    eps = sp["eps"]
    is_polya = sp["is_polya"].astype(bool)
    log_p = sp["log_p"]; log_p_c = sp["log_p_c"]
    log_q = sp["log_q"]; log_q_c = sp["log_q_c"]
    log_gain = sp["log_gain"]

    logF = np.log(profile_count) if profile_count > 0 else NEG_INF

    d_logit_p     = np.zeros(N, dtype=np.float64)
    d_logit_q     = np.zeros(N, dtype=np.float64)
    d_log_kappa   = np.zeros(N, dtype=np.float64)

    for v in range(N):
        # ----- Compute logSv (= log Σ_s s·P{η̃_v=s} = log E[η̃_v]) via tail trick
        # pSv[j] is destructively replaced by cumulative tail probabilities.
        pSv = log_edge_post[v].copy()
        log_tail = NEG_INF
        for j in range(W - 1, -1, -1):
            x = pSv[j]
            pSv[j] = log_tail
            log_tail = _logadd(log_tail, x)
        # logSv = logsumexp of pSv (tails) — that's log Σ_s P{η̃_v > s} = log E[η̃_v]
        logSv = _log_sum(pSv)

        # ----- Loss gradient w.r.t. logit p̃ (only non-root, finite log1_p)
        logp_v  = log_p[v]
        log1_pv = log_p_c[v]
        if log1_pv == NEG_INF:
            # p̃=1 (root or fully-extinct edge): no loss gradient
            d_logit_p[v] = 0.0
        else:
            tNu_Sv = log_death_tails[v]
            logNu_Sv = _log_sum(tNu_Sv)
            if v == root:
                # ROOTLOSS variant — Java treats this specially but
                # for our standard tree (root edge length = ∞ → p̃=1)
                # we never enter this branch. Keep for completeness.
                dpos = logNu_Sv + log1_pv
                dneg = logSv + logp_v
            else:
                u = int(parent[v])
                num_children_u = len(kids[u])
                if num_children_u == 2:
                    # log(1 - eps_v) where eps_v = sibling p̃
                    sibs = [c for c in kids[u] if c != v]
                    log1_e = log_p_c[sibs[0]]
                else:
                    # Multifurcation — use log(extinction_u / p̃_v)
                    # eps_excl_v = ε_u / p̃_v
                    log_eps_u = np.log(eps[u]) if eps[u] > 0 else NEG_INF
                    diff = log_eps_u - logp_v
                    # log(1 - exp(diff))
                    if diff >= 0:
                        log1_e = NEG_INF
                    elif diff < -1e-3:
                        log1_e = np.log1p(-np.exp(diff))
                    else:
                        log1_e = np.log(-np.expm1(diff))
                # log(1 - p̃·eps) = logadd(log(1-p̃), log(p̃·eps))
                # = logadd(log1_pv, logp_v + log1_e_negation)... actually
                # Java's log1_pe = logadd(log1_p, log(p)+log(1-e))
                log1_pe = _logadd(log1_pv, logp_v + log1_e)
                dpos = logNu_Sv + log1_pv - log1_pe
                dneg = logSv + logp_v + log1_e - log1_pe
            # d_logit_p = exp(dpos) - exp(dneg) (linear scale)
            d_logit_p[v] = (np.exp(dpos) if np.isfinite(dpos) else 0.0
                            ) - (np.exp(dneg) if np.isfinite(dneg) else 0.0)

        # ----- Gain & dup gradients
        log_q_v   = log_q[v]
        log1_q_v  = log_q_c[v]
        tNv_Sv = log_birth_tails[v]
        logNv_Sv = _log_sum(tNv_Sv)

        if log_q_v < -500:  # Poisson (q=0)
            r = kappa[v]
            log_r = log_gain[v]
            if log_r < -500:
                d_log_kappa[v] = 0.0
            else:
                dpos = logNv_Sv
                dneg = log_r + logF
                d_log_kappa[v] = (np.exp(dpos) if np.isfinite(dpos) else 0.0
                                  ) - (np.exp(dneg) if np.isfinite(dneg) else 0.0)
            d_logit_q[v] = 0.0  # no dup
        else:
            # Pólya
            log_kappa_v = log_gain[v]
            if log_kappa_v < -500:
                d_log_kappa[v] = 0.0
            else:
                # κ-gradient with harmonic-sum term
                dpos = tNv_Sv[0]
                for i in range(1, W):
                    if i < kappa[v]:
                        log_k_ki = -np.log1p(i / kappa[v])
                    else:
                        log_k_ki = log_kappa_v - np.log(i) - np.log1p(kappa[v] / i)
                    if np.isfinite(tNv_Sv[i]):
                        dpos = _logadd(dpos, tNv_Sv[i] + log_k_ki)
                # loglog1_q with Java's fallback to log_q when 1-q is sub-ULP
                if log1_q_v < 0:
                    loglog1_q = np.log(-log1_q_v)
                    if loglog1_q == NEG_INF:
                        loglog1_q = log_q_v
                else:
                    loglog1_q = log_q_v
                log_kappa_log1_q = log_kappa_v + loglog1_q
                dneg = log_kappa_log1_q + logF
                d_log_kappa[v] = (np.exp(dpos) if np.isfinite(dpos) else 0.0
                                  ) - (np.exp(dneg) if np.isfinite(dneg) else 0.0)
            # dup: dpos = (N-S)·(1-q), dneg = (S + F·κ)·q
            dpos_q = logNv_Sv + log1_q_v
            dneg_q = _logadd(logSv, log_gain[v] + logF) + log_q_v
            d_logit_q[v] = (np.exp(dpos_q) if np.isfinite(dpos_q) else 0.0
                            ) - (np.exp(dneg_q) if np.isfinite(dneg_q) else 0.0)

    return d_logit_p, d_logit_q, d_log_kappa


def compute_L0_gradient_analytical(
    tree: Tree, gain, loss, dup, length, min_copies: int,
):
    """End-to-end analytical ∂L(0)/∂(gain, loss, dup, length).

    Pipeline (entirely native C as of 2026-05-18 evening — torch retired):
      1. native Uinner → C, K per-size tensors (Csurös SI Thm 3, Fig S11)
      2. native outside → B, J, Bns, Jns per-size tensors (Thm 4, Fig S12)
      3. native posteriors → unobserved-profile marginals + joint transitions
      4. native log_tail_difference → birth/death tails
      5. native getLogSurvivalGradient → ∂log L(0)/∂(logit p̃, logit q̃, log κ)
      6. native chain rule → ∂L(0)/∂(g, l, d, t)
         (was torch autograd; replaced by recount_chain_rule_logit_to_rates,
         which walks the survival recurrence in reverse using analytical
         rate_to_p_jac / rate_to_q_jac.)

    Replaces ``recount.unobserved.unobserved_logL0_grad_fd`` for production use:
    same output signature but bit-exact analytical (matches Java's
    LogGradient.PosteriorStatistics.getLogSurvivalGradient applied to the
    FamilySizeLikelihood unobserved-profile statistics).

    Returns (L0, g_gain[N], g_loss[N], g_dup[N], g_length[N]) — all derivatives
    are of L(0) itself (not log L(0)), matching the FD function's convention.

    The caller wires this into the corrected gradient:
        ∂(corr LL)/∂θ = ∂(raw LL)/∂θ + F · ∂L(0)/∂θ / (1 - L(0))
    """
    from recount.native_backend import (
        unobserved_outside_native, unobserved_posteriors_native,
        unobserved_transitions_native, unobserved_bd_tails_native,
        unobserved_logsurv_grad_native, chain_rule_logit_to_rates_native,
    )

    M = min_copies - 1
    if min_copies <= 0:
        gain_a = np.asarray(gain, dtype=np.float64)
        zeros = np.zeros_like(gain_a)
        return NEG_INF, zeros, zeros, zeros, zeros

    # Step 1: inside tensors from native
    C_all, K_all, _ = unobserved_inside_tensors_native(
        tree, gain, loss, dup, length, min_copies)

    # Step 2: outside via native C
    B_all, J_all, Bns_all, Jns_all, Lw_all = unobserved_outside_native(
        tree, gain, loss, dup, length, K_all, min_copies)
    log_L0 = compute_unobserved_log_likelihood(K_all, J_all, tree.root, M)
    L0 = float(np.exp(log_L0))

    if not np.isfinite(log_L0):
        gain_a = np.asarray(gain, dtype=np.float64)
        zeros = np.zeros_like(gain_a)
        return log_L0, zeros, zeros, zeros, zeros

    # Steps 3-5 native (~0.1 ms total): posteriors, transitions, BD tails,
    # log-survival gradient w.r.t. (logit p̃, logit q̃, log κ).
    _, log_edge_post = unobserved_posteriors_native(
        C_all, K_all, B_all, J_all, log_L0, min_copies)
    log_node_trans, log_edge_trans = unobserved_transitions_native(
        C_all, K_all, Bns_all, Jns_all, log_L0, min_copies)
    log_birth_tails, log_death_tails = unobserved_bd_tails_native(
        log_node_trans, log_edge_trans, min_copies)
    d_logit_p, d_logit_q, d_log_kappa = unobserved_logsurv_grad_native(
        tree, gain, loss, dup, length,
        log_edge_post, log_birth_tails, log_death_tails,
        profile_count=1.0, min_copies=min_copies)
    # d_* are the gradient of log L(0) w.r.t. (logit p̃, logit q̃, log κ).

    # Step 6: chain to (gain, loss, dup, length) via native C reverse-mode
    # chain rule (recount_chain_rule_logit_to_rates). Walks the survival
    # recurrence in reverse using analytical rate_to_p_jac / rate_to_q_jac
    # — replaces the torch autograd dependency. ~0.06-0.2 ms / call,
    # vs ~25 ms for the old torch path.
    cdp = np.nan_to_num(d_logit_p,   nan=0.0, posinf=0.0, neginf=0.0)
    cdq = np.nan_to_num(d_logit_q,   nan=0.0, posinf=0.0, neginf=0.0)
    cdk = np.nan_to_num(d_log_kappa, nan=0.0, posinf=0.0, neginf=0.0)
    g_g_log, g_l_log, g_d_log, g_len_log = chain_rule_logit_to_rates_native(
        tree, gain, loss, dup, length, cdp, cdq, cdk)
    # Returned gradients are of log L(0). Convert to L(0) gradient via L(0).
    g_g   = g_g_log   * L0
    g_l   = g_l_log   * L0
    g_d   = g_d_log   * L0
    g_len = g_len_log * L0
    return log_L0, g_g, g_l, g_d, g_len


# The analytical L(0) gradient is implemented above (Phase B), validated
# against finite differences to median 1e-7 rel error on Williams2017 and
# wired into recount.native_backend / recount.ml as the production gradient
# path for min_copies ≥ 2. See VALIDATION.md for the agreement table.
