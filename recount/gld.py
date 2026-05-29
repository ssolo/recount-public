"""NumPy implementation of the GLD likelihood and analytical gradient.

The forward (Felsenstein peeling) pass computes for each profile and each node v
the log-likelihood vectors

    C[v][ℓ] = log P(profile in subtree below v | v has ℓ surviving copies)
    K[v][s] = log P(profile in subtree below v | parent sends s surviving copies)

operating in the "survival" parameterization where each ε-corrected
(p̃, q̃, r̃/κ̃) per node implicitly marginalizes out copies whose lineage
goes extinct before reaching any leaf.

The outside (down-pass) gives the matching outside log-likelihoods

    B[v][ℓ] = log P(profile outside v's subtree | v has ℓ surviving copies)
    J[v][s] = log P(profile outside v's edge   | parent sends s copies in)

so that the posterior P(ξ̃_v = ℓ | Ξ) ∝ exp(B[v][ℓ] + C[v][ℓ]) and likewise
for the edge posterior.  Per-family posterior means/tails of (ξ̃_v, η̃_v)
plug into the closed-form gradient from Csűrös (2021) Corollary 9.

Reference: M. Csűrös. "Gain-loss-duplication models on a phylogeny: exact
algorithms for computing the likelihood and its gradient." arXiv:2107.11440.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from recount._math import (
    NEG_INF,
    LogFactorial,
    LogRisingFactorial,
    logadd,
    logsumexp,
    safelog_pair,
)
from recount.rates import GLDRates, rate_to_p, rate_to_q
from recount.tree import Tree


# Parameter indices in the flat gradient array (matches Java's GLDParameters.Type ordinal).
GAIN = 0
LOSS = 1
DUP = 2


# ----------------------------------------------------------------------------
# survival-parameter precomputation
# ----------------------------------------------------------------------------


@dataclass
class SurvivalParams:
    """Per-node survival-parameterized rates.

    These are the (p̃, q̃, r̃/κ̃) of the inside-outside recursion, derived
    bottom-up from raw (p, q, r/κ) with the subtree-extinction probability

        ε[v] = P(a single lineage at v has no leaf descendant) = Π_{c child of v} p̃_c.
    """
    p_raw: np.ndarray
    q_raw: np.ndarray
    p_raw_c: np.ndarray
    q_raw_c: np.ndarray
    p: np.ndarray
    q: np.ndarray
    p_c: np.ndarray
    q_c: np.ndarray
    gain: np.ndarray
    eps: np.ndarray
    eps_c: np.ndarray
    is_polya: np.ndarray
    log_p: np.ndarray
    log_q: np.ndarray
    log_p_c: np.ndarray
    log_q_c: np.ndarray
    log_gain: np.ndarray
    log_eps: np.ndarray
    log_eps_c: np.ndarray


def compute_survival_params(tree: Tree, rates: GLDRates) -> SurvivalParams:
    """Bottom-up pass: raw (p, q, r/κ) per edge → survival (p̃, q̃, r̃/κ̃)."""
    n = tree.num_nodes

    p_raw = np.zeros(n); q_raw = np.zeros(n)
    p_raw_c = np.ones(n); q_raw_c = np.ones(n)
    for v in range(n):
        p_v, p_v_c = rate_to_p(rates.loss[v], rates.dup[v], rates.length[v])
        q_v, q_v_c = rate_to_q(rates.loss[v], rates.dup[v], rates.length[v])
        p_raw[v], p_raw_c[v] = p_v, p_v_c
        q_raw[v], q_raw_c[v] = q_v, q_v_c

    eps = np.zeros(n); eps_c = np.ones(n)
    p_t = np.zeros(n); q_t = np.zeros(n)
    p_t_c = np.ones(n); q_t_c = np.ones(n)
    g_t = np.zeros(n)
    is_polya = np.zeros(n, dtype=bool)

    for v in range(n):  # post-order (leaves first)
        if tree.is_leaf[v]:
            eps[v] = 0.0
            eps_c[v] = 1.0
        else:
            # ε[v] = ∏_c p̃_c, computed with a parallel (1-ε) accumulator to
            # remain accurate when ε approaches 1.
            e0 = 1.0
            e1 = 0.0
            for c in tree.children[v]:
                pc = p_t[c]; pc_c = p_t_c[c]
                e1 = e1 + e0 * pc_c
                e0 = e0 * pc
            eps[v] = e0
            eps_c[v] = e1 if e0 == 1.0 else (1.0 - e0)

        e = eps[v]; e_c = eps_c[v]
        p = p_raw[v]; p_c = p_raw_c[v]
        q = q_raw[v]; q_c = q_raw_c[v]
        is_polya[v] = q > 0.0

        # r̃ (Poisson) or κ̃=κ (Pólya): the gain rate is ε-corrected only when
        # the gain process is Poisson (the Pólya shape carries through unchanged).
        g_t[v] = rates.gain[v] if is_polya[v] else rates.gain[v] * e_c

        # a = (1-q)·ε + (1-ε) = 1 - q·ε  (no-subtraction form for stability)
        a = q_c * e + e_c
        p_t[v] = (p * e_c + e * q_c) / a
        q_t[v] = q * e_c / a
        b = e_c / a
        p_t_c[v] = p_c * b
        q_t_c[v] = q_c / a

    log_p = safelog_pair(p_t, p_t_c)
    log_q = safelog_pair(q_t, q_t_c)
    log_p_c = safelog_pair(p_t_c, p_t)
    log_q_c = safelog_pair(q_t_c, q_t)
    log_gain = np.where(g_t > 0, np.log(np.where(g_t > 0, g_t, 1.0)), NEG_INF)
    log_eps = safelog_pair(eps, eps_c)
    log_eps_c = safelog_pair(eps_c, eps)

    return SurvivalParams(
        p_raw=p_raw, q_raw=q_raw, p_raw_c=p_raw_c, q_raw_c=q_raw_c,
        p=p_t, q=q_t, p_c=p_t_c, q_c=q_t_c,
        gain=g_t, eps=eps, eps_c=eps_c, is_polya=is_polya,
        log_p=log_p, log_q=log_q, log_p_c=log_p_c, log_q_c=log_q_c,
        log_gain=log_gain, log_eps=log_eps, log_eps_c=log_eps_c,
    )


# ----------------------------------------------------------------------------
# inside (forward) pass
# ----------------------------------------------------------------------------


def _calc_widths(tree: Tree, profile: np.ndarray) -> np.ndarray:
    """Per-node array of `max(surviving copies)+1` for one profile.

    Leaf widths come from the observed counts; an internal node's width is
    one more than the sum of children's observable copies, covering the full
    support of S_v.  An "ambiguous" leaf (negative count) collapses to 0.
    """
    n = tree.num_nodes
    w = np.zeros(n, dtype=np.int64)
    for v in range(n):
        if tree.is_leaf[v]:
            c = int(profile[v])
            w[v] = 0 if c < 0 else c + 1
        else:
            kids = tree.children[v]
            ambi = 0; total = 0
            for c in kids:
                cn = w[c] - 1
                if cn < 0:
                    ambi += 1
                else:
                    total += cn
            w[v] = 0 if ambi == len(kids) else total + 1
    return w


def _compute_edge(
    C: np.ndarray, log_q: float, log_q_c: float, log_gain: float,
    is_polya: bool, fact: LogFactorial, rfact: Optional[LogRisingFactorial],
    K_len: int,
) -> np.ndarray:
    """K[s] = log Σ_{t : s+t<len(C)} C[s+t] · gain_pmf(t | s).

    Gain distribution on the edge entering v: Poisson(r̃) when q=0 or
    NegBinomial(κ+s, 1-q) when q>0 (so the gain shape grows with s, the
    incoming surviving count).
    """
    K = np.full(K_len, NEG_INF)
    if C.size == 0 or K_len == 0:
        return K
    C_len = C.size

    if not is_polya:
        if log_gain == NEG_INF:
            # No gain → identity: K[s] = C[s]
            for s in range(min(K_len, C_len)):
                K[s] = C[s]
            return K
        r = float(np.exp(log_gain))
        terms = np.empty(C_len)
        for s in range(K_len):
            t = 0; ell = s; count = 0
            while ell < C_len:
                t_logr = 0.0 if t == 0 else t * log_gain
                terms[count] = C[ell] - r + t_logr - fact.factln(t)
                t += 1; ell += 1; count += 1
            K[s] = logsumexp(terms[:count]) if count > 0 else NEG_INF
        return K

    # Pólya
    if log_gain == NEG_INF:
        for s in range(min(K_len, C_len)):
            K[s] = C[s]
        return K
    assert rfact is not None
    loglog1_q = float(np.log(-log_q_c)) if log_q_c < 0 else float(log_q)
    kappa_log1_q = float(np.exp(log_gain + loglog1_q))  # = κ · |log(1-q)|
    terms = np.empty(C_len)
    for s in range(K_len):
        factln_s = rfact.factln(s)
        # (κ+s)·log(1-q) = s·log(1-q) − κ·|log(1-q)|.  When q → 1 (log_q_c
        # = −∞, e.g. the t = ∞ root edge) the s·log(1-q) term is 0 at s = 0
        # and −∞ for s > 0; guard the 0·(−∞) = NaN.  Mirrors the native
        # port (native/src/recount_edge.c).  s-invariant → hoisted here.
        if log_q_c == NEG_INF:
            ks_log1_q = -kappa_log1_q if s == 0 else NEG_INF
        else:
            ks_log1_q = s * log_q_c - kappa_log1_q
        t = 0; ell = s; count = 0
        while ell < C_len:
            binom = rfact.factln(ell) - factln_s - fact.factln(t)
            t_logq = 0.0 if t == 0 else t * log_q
            terms[count] = C[ell] + binom + ks_log1_q + t_logq
            t += 1; ell += 1; count += 1
        K[s] = logsumexp(terms[:count]) if count > 0 else NEG_INF
    return K


def _compute_sibling(
    C: np.ndarray, K_junior: np.ndarray,
    p_junior: float, p_junior_c: float, eps_sib: float,
    fact: LogFactorial,
) -> np.ndarray:
    """Combine accumulated siblings' inside (`C`) with junior child's edge (`K_junior`).

    Adds junior to the set of already-processed siblings of a parent node.
    Each of the parent's surviving copies is independently classified as
    (survives in junior, survives in already-processed siblings), with the
    "both survive" event folded back into the recursion via the in-place
    update on `C`.

        ε_sib  = product of already-processed siblings' p̃
        p1     = (1-p̃_junior) / (1 - p̃_junior · ε_sib)
        p2     = p̃_junior · (1 - ε_sib) / (1 - p̃_junior · ε_sib)

        C'[ℓ] = log Σ_{s+t=ℓ}  K_junior[s] + C[t] + log binom(ℓ,s)
                              + s·log(p1) + t·log(p2)

    with the destructive update C[t] ← (1-ε)·C[t+1] + ε·C[t] (in linear space)
    marginalizing copies surviving in both junior and the prior siblings.

    Mirrors count.model.Likelihood.Profile.computeSibling in the Java source.
    """
    if eps_sib == 1.0:
        return K_junior.copy() if K_junior.size > 0 else C.copy()
    if C.size == 0:
        return K_junior.copy()
    if K_junior.size == 0:
        return C  # junior ambiguous

    C = C.copy()
    K = K_junior
    K_len = K.size; C_len = C.size
    combined = (K_len - 1) + (C_len - 1)

    log_e = float(np.log(eps_sib))
    log_e_c = float(np.log1p(-eps_sib))
    log_a = float(np.log1p(-p_junior * eps_sib))     # log(1 − p̃·ε)
    logp1 = float(np.log(p_junior_c)) - log_a
    logp2 = float(np.log(p_junior)) + log_e_c - log_a

    C2 = np.full(combined + 1, NEG_INF)
    terms_buf = np.empty(K_len)
    for ell in range(combined + 1):
        # In-place "destructive update" of C[t]: each step folds the
        # parent-copy-shared-between-junior-and-siblings outcome into the
        # sibling tail by mixing C[t+1] (weight 1-ε) and C[t] (weight ε).
        t = min(ell, C_len)
        x = C[t] if t < C_len else NEG_INF
        s = ell - t
        while t > 0 and s < K_len - 1:
            t -= 1; s += 1
            x = x + log_e_c
            y = C[t] + log_e
            x = logadd(x, y)
            C[t] = x

        log_ellfact = fact.factln(ell)
        tmin = t
        count = 0
        while s >= 0 and t < C_len:
            binom = log_ellfact - fact.factln(s) - fact.factln(t)
            slogp1 = 0.0 if s == 0 else s * logp1
            tlogp2 = 0.0 if t == 0 else t * logp2
            terms_buf[count] = K[s] + C[t] + binom + slogp1 + tlogp2
            s -= 1; t += 1; count += 1
        C2[ell] = logsumexp(terms_buf[:count]) if count > 0 else NEG_INF
    return C2


@dataclass
class _ProfileCache:
    C: List[np.ndarray]
    K: List[np.ndarray]
    LL: float


def _forward(
    tree: Tree, sp: SurvivalParams, profile: np.ndarray,
    fact: LogFactorial, rfacts: List[Optional[LogRisingFactorial]],
) -> _ProfileCache:
    """Felsenstein peeling for a single profile."""
    n = tree.num_nodes
    width = _calc_widths(tree, profile)
    C: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    K: List[np.ndarray] = [None] * n  # type: ignore[list-item]

    for v in range(n):
        w = int(width[v])
        if tree.is_leaf[v]:
            Cv = np.full(w, NEG_INF)
            if w > 0:
                Cv[w - 1] = 0.0
            C[v] = Cv
        else:
            if sp.eps[v] == 1.0:
                # Whole subtree extinct: only the "0 surviving" cell carries weight.
                acc = 0.0
                for c in tree.children[v]:
                    Kc = K[c]
                    if Kc.size > 0:
                        acc += Kc[0]
                Cv = np.array([acc])
            else:
                Cv = np.empty(0)
                sib_extinct = 1.0
                for c in tree.children[v]:
                    Cv = _compute_sibling(Cv, K[c], sp.p[c], sp.p_c[c], sib_extinct, fact)
                    sib_extinct = sib_extinct * sp.p[c]
            if Cv.size > w:
                Cv = Cv[:w]
            elif Cv.size < w:
                tmp = np.full(w, NEG_INF)
                tmp[: Cv.size] = Cv
                Cv = tmp
            C[v] = Cv

        if v == tree.root:
            K_len = 1 if sp.log_p_c[v] == NEG_INF else 2
        else:
            K_len = int(width[v])
        K[v] = _compute_edge(C[v], sp.log_q[v], sp.log_q_c[v], sp.log_gain[v],
                             bool(sp.is_polya[v]), fact, rfacts[v], K_len)

    root = tree.root
    Kr = K[root]
    LL = Kr[0]
    p_root = sp.p[root]
    if p_root != 1.0 and Kr.size == 2:
        LL = LL + float(np.log(p_root))
        LL = logadd(LL, Kr[1] + float(np.log(sp.p_c[root])))
    return _ProfileCache(C=C, K=K, LL=float(LL))


def _build_rfacts(sp: SurvivalParams, max_n: int) -> List[Optional[LogRisingFactorial]]:
    out: List[Optional[LogRisingFactorial]] = []
    for v in range(len(sp.gain)):
        if sp.is_polya[v] and sp.gain[v] > 0:
            out.append(LogRisingFactorial(float(sp.gain[v]), max_n=max_n + 2))
        else:
            out.append(None)
    return out


def _make_caches(tree: Tree, sp: SurvivalParams, profiles: np.ndarray):
    """Allocate factorial caches large enough for the largest profile."""
    max_count = int(profiles.max(initial=0))
    total = int(profiles.sum(axis=1).max(initial=0)) + 8
    fact = LogFactorial(max_n=max(total, max_count) + 4)
    rfacts = _build_rfacts(sp, max_n=total + 4)
    return fact, rfacts


# ----------------------------------------------------------------------------
# top-level likelihood functions
# ----------------------------------------------------------------------------


def log_likelihood(tree: Tree, rates: GLDRates, profiles: np.ndarray) -> float:
    """Raw log-likelihood Σ_f log P(Ξ_f) across all profiles (no observation-bias correction)."""
    sp = compute_survival_params(tree, rates)
    fact, rfacts = _make_caches(tree, sp, profiles)
    LL = 0.0
    for f in range(profiles.shape[0]):
        LL += _forward(tree, sp, profiles[f], fact, rfacts).LL
    return LL


def empty_log_likelihood(tree: Tree, rates: GLDRates) -> float:
    """log P(empty profile = all zeros) — closed-form, no peeling needed."""
    sp = compute_survival_params(tree, rates)
    LL = 0.0
    for v in range(tree.num_nodes):
        if sp.is_polya[v]:
            log1_q = sp.log_q_c[v]
            if log1_q == NEG_INF:
                return NEG_INF
            log_k = sp.log_gain[v]
            if log_k == NEG_INF:
                continue
            loglog1_q = np.log(-log1_q)
            LL -= float(np.exp(log_k + loglog1_q))
        else:
            LL -= float(sp.gain[v])
    LL += float(sp.log_p[tree.root])
    return LL


def singleton_log_likelihood(tree: Tree, rates: GLDRates) -> float:
    """log Σ_leaf P(profile = 1 copy at that single leaf, 0 elsewhere)."""
    sp = compute_survival_params(tree, rates)
    log_p = []
    for ell in range(tree.num_leaves):
        prof = np.zeros(tree.num_leaves, dtype=np.int64)
        prof[ell] = 1
        fact, rfacts = _make_caches(tree, sp, prof.reshape(1, -1))
        log_p.append(_forward(tree, sp, prof, fact, rfacts).LL)
    return float(logsumexp(np.asarray(log_p)))


def corrected_log_likelihood(
    tree: Tree, rates: GLDRates, profiles: np.ndarray, min_copies: int = 1,
) -> float:
    """log-likelihood conditioned on observing at least `min_copies` copies.

    min_copies = 0 disables the correction (returns the raw LL).
    min_copies = 1 corrects for unobserved empty profiles (the typical setting).
    min_copies = 2 also corrects for unobserved singletons.
    """
    if min_copies not in (0, 1, 2):
        raise ValueError("min_copies must be 0, 1, or 2")
    LL = log_likelihood(tree, rates, profiles)
    if min_copies == 0:
        return LL
    L0 = empty_log_likelihood(tree, rates)
    if min_copies == 2:
        L0 = float(np.logaddexp(L0, singleton_log_likelihood(tree, rates)))
    F = profiles.shape[0]
    p_obs = -np.expm1(L0)
    return float(LL - F * np.log(p_obs))


# ----------------------------------------------------------------------------
# outside (backward) pass — inside-outside marginals
# ----------------------------------------------------------------------------


def _combine_siblings_excluding(
    tree: Tree, K: List[np.ndarray], sp: SurvivalParams,
    parent: int, exclude_child: int, fact: LogFactorial,
) -> np.ndarray:
    """Combine the inside K's of `parent`'s children except `exclude_child`.

    Used by the outside pass to "remove" v from its parent's combined inside,
    leaving an inside log-likelihood for the siblings' subtree only.
    """
    sibs = [c for c in tree.children[parent] if c != exclude_child]
    if not sibs:
        return np.empty(0)
    K2 = np.empty(0)
    sib_extinct = 1.0
    for c in sibs:
        K2 = _compute_sibling(K2, K[c], sp.p[c], sp.p_c[c], sib_extinct, fact)
        sib_extinct = sib_extinct * sp.p[c]
    return K2


def _compute_outside(
    tree: Tree, sp: SurvivalParams,
    C: List[np.ndarray], K: List[np.ndarray],
    fact: LogFactorial, rfacts: List[Optional[LogRisingFactorial]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Down-pass: outside log-likelihoods (J, B) for one profile.

    J[v][s]   = log P(profile outside v's edge | s copies enter v's edge)
    B[v][ℓ]   = log P(profile outside v's subtree | v has ℓ surviving copies)

    Invariant: for any node v,
        log Σ_ℓ exp(B[v][ℓ] + C[v][ℓ])  ==  LL(profile)
    (and same with J ⊗ K on the edge), verified to machine precision in tests.
    """
    n = tree.num_nodes
    J: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    B: List[np.ndarray] = [None] * n  # type: ignore[list-item]

    root = tree.root
    if sp.log_p_c[root] == NEG_INF:
        J[root] = np.array([0.0])  # p̃_root == 1 case
    else:
        J[root] = np.array([sp.log_p[root], sp.log_p_c[root]])  # ROOTLOSS

    def edge_outside_to_node(v: int, Jv: np.ndarray) -> np.ndarray:
        """B[v][ℓ] = log Σ_s J[v][s] · gain_pmf(ℓ-s | s) — same structure
        as the forward edge step but going outside→inside."""
        Bv_len = C[v].size
        if Bv_len == 0:
            return np.empty(0)
        Bv = np.full(Bv_len, NEG_INF)
        if not sp.is_polya[v]:
            log_gain = sp.log_gain[v]
            if log_gain == NEG_INF:
                for ell in range(min(Bv_len, Jv.size)):
                    Bv[ell] = Jv[ell]
                return Bv
            r = float(np.exp(log_gain))
            for ell in range(Bv_len):
                terms = []
                for s in range(min(ell + 1, Jv.size)):
                    t = ell - s
                    log_pmf = -r + (t * log_gain if t > 0 else 0.0) - fact.factln(t)
                    terms.append(Jv[s] + log_pmf)
                Bv[ell] = logsumexp(np.asarray(terms))
            return Bv
        # Pólya
        log_q = sp.log_q[v]; log_q_c = sp.log_q_c[v]
        log_kappa = sp.log_gain[v]
        if log_kappa == NEG_INF:
            for ell in range(min(Bv_len, Jv.size)):
                Bv[ell] = Jv[ell]
            return Bv
        rfact = rfacts[v]
        assert rfact is not None
        loglog1_q = float(np.log(-log_q_c)) if log_q_c < 0 else float(log_q)
        kappa_log1_q = float(np.exp(log_kappa + loglog1_q))
        for ell in range(Bv_len):
            terms = []
            for s in range(min(ell + 1, Jv.size)):
                t = ell - s
                binom = rfact.factln(ell) - rfact.factln(s) - fact.factln(t)
                # guard 0·(−∞) when q → 1 (log_q_c = −∞); see _compute_edge
                if log_q_c == NEG_INF:
                    ks_log1_q = -kappa_log1_q if s == 0 else NEG_INF
                else:
                    ks_log1_q = s * log_q_c - kappa_log1_q
                t_logq = 0.0 if t == 0 else t * log_q
                terms.append(Jv[s] + binom + ks_log1_q + t_logq)
            Bv[ell] = logsumexp(np.asarray(terms))
        return Bv

    B[root] = edge_outside_to_node(root, J[root])

    # Pre-order descent: iterate v from root-1 down to 0 so each node's
    # parent has already been processed.
    for v in range(n - 1, -1, -1):
        if v == root:
            continue
        parent = int(tree.parent[v])
        Bu = B[parent]
        K_sib = _combine_siblings_excluding(tree, K, sp, parent, exclude_child=v, fact=fact)

        p = float(sp.p[v]); p_c = float(sp.p_c[v])
        if K_sib.size == 0:
            K2 = np.array([0.0])
            eps_sib = 0.0
        else:
            K2 = K_sib
            eps_sib = 1.0
            for c in tree.children[parent]:
                if c != v:
                    eps_sib *= float(sp.p[c])

        Jv_len = K[v].size
        Jv = np.full(Jv_len, NEG_INF)
        if Bu.size == 0:
            J[v] = Jv
            B[v] = edge_outside_to_node(v, Jv)
            continue

        log_e = np.log(eps_sib) if eps_sib > 0 else NEG_INF
        log_e_c = np.log1p(-eps_sib) if eps_sib < 1 else NEG_INF
        log_a = np.log1p(-p * eps_sib)
        logp1 = float(np.log(p_c) - log_a)
        logp2 = float(np.log(p) + log_e_c - log_a)

        K2_work = K2.copy()
        for s in range(Jv_len):
            if s > 0 and K2_work.size > 0:
                # Symmetric destructive update on K2 (analogue of the forward
                # update on C): K2(s+t,t) = (1-ε)·K2_prev(s-1+t+1,t+1) + ε·K2_prev(s-1+t,t)
                t = K2_work.size - 1
                y = K2_work[t]
                K2_work[t] = K2_work[t] + log_e
                while t > 0:
                    t -= 1
                    x = y + log_e_c
                    y = K2_work[t]
                    z = y + log_e
                    K2_work[t] = logadd(x, z)

            terms = []
            t = 0; ell = s + t
            log_sfact = fact.factln(s)
            while t < K2_work.size and ell < Bu.size:
                log_ellfact = fact.factln(ell)
                binom = log_ellfact - fact.factln(t) - log_sfact
                slogp1 = 0.0 if s == 0 else s * logp1
                tlogp2 = 0.0 if t == 0 else t * logp2
                terms.append(Bu[ell] + K2_work[t] + binom + slogp1 + tlogp2)
                t += 1; ell += 1
            Jv[s] = logsumexp(np.asarray(terms)) if terms else NEG_INF
        J[v] = Jv
        B[v] = edge_outside_to_node(v, Jv)
    return J, B


# ----------------------------------------------------------------------------
# analytical gradient (Csűrös 2021 Corollary 9, with corrected loss formula)
# ----------------------------------------------------------------------------


def _posterior_marginal_stats(
    log_outside: np.ndarray, log_inside: np.ndarray, LL_f: float, max_len: int,
) -> Tuple[float, np.ndarray]:
    """Posterior E[X | Ξ] and tail P(X > i | Ξ) up to length `max_len`."""
    L = min(log_outside.size, log_inside.size)
    if L == 0:
        return 0.0, np.zeros(max_len)
    log_p = log_outside[:L] + log_inside[:L] - LL_f
    finite = np.isfinite(log_p)
    if not finite.any():
        return 0.0, np.zeros(max_len)
    post = np.zeros(L)
    post[finite] = np.exp(log_p[finite])
    mean = float(np.sum(np.arange(L) * post))
    cum_right = np.cumsum(post[::-1])[::-1]
    tail_post = np.zeros(L)
    if L > 1:
        tail_post[:-1] = cum_right[1:]
    out_tail = np.zeros(max_len)
    n = min(max_len, L)
    out_tail[:n] = tail_post[:n]
    return mean, out_tail


def gradient_survival(
    tree: Tree, rates: GLDRates, profiles: np.ndarray, min_copies: int = 1,
) -> np.ndarray:
    """Analytical gradient ∂(ln L*)/∂(p̃, q̃, r̃/κ̃) of the corrected log-likelihood.

    Implements the closed-form expressions from Csűrös (2021), Corollary 9:

        ∂(ln L*)/∂κ_v   = F · ln(1−q̃)/(1−L(0))
                          + Σ_i (Ñ_v^{>i} − S̃_v^{>i})/(κ + i)              (Pólya)
                        = (Ñ_v − S̃_v)/r̃ − F/(1−L(0))                       (Poisson)

        ∂(ln L*)/∂q̃_v   = (Ñ_v − S̃_v)/q̃ − (S̃_v + F·κ/(1−L(0)))/(1−q̃)   (Pólya;
                                                                            0 for Poisson)

        ∂(ln L*)/∂p̃_y   = [(Ñ_x − S̃_y)/p̃_y − (1−ε)·S̃_y/(1−p̃_y)] / (1 − p̃_y·ε)
                          (non-root; x = parent(y), ε = Π siblings(y) p̃_z)

    Posterior expectations Ñ_v, S̃_v, Ñ^{>i}, S̃^{>i} are summed over families
    from the inside-outside marginals (B[v]·C[v] for the node, J[v]·K[v] for
    the edge).

    Returns a flat array indexed `3*v + {GAIN=0, LOSS=1, DUP=2}`, matching
    the orientation of Java's `Gradient.getCorrectedGradient()`.

    Note: the loss formula's (1−p̃·ε) divisor is missing from the paper's
    Corollary 9 (apparent typo); the proof's intermediate expression and
    the Java source both include it, as does the derivation here.
    """
    if min_copies not in (0, 1, 2):
        raise ValueError("min_copies must be 0, 1, or 2")
    sp = compute_survival_params(tree, rates)
    fact, rfacts = _make_caches(tree, sp, profiles)
    n = tree.num_nodes
    F = profiles.shape[0]

    N_v = np.zeros(n); S_v = np.zeros(n)
    N_tail: List[np.ndarray] = [np.zeros(0) for _ in range(n)]
    S_tail: List[np.ndarray] = [np.zeros(0) for _ in range(n)]

    def acc(target: List[np.ndarray], v: int, add: np.ndarray) -> None:
        if add.size == 0:
            return
        if target[v].size < add.size:
            grown = np.zeros(add.size)
            grown[: target[v].size] = target[v]
            target[v] = grown
        target[v][: add.size] += add

    for f in range(F):
        pc = _forward(tree, sp, profiles[f], fact, rfacts)
        J, B = _compute_outside(tree, sp, pc.C, pc.K, fact, rfacts)
        LL_f = pc.LL
        for v in range(n):
            mean_node, tail_node = _posterior_marginal_stats(B[v], pc.C[v], LL_f, pc.C[v].size)
            mean_edge, tail_edge = _posterior_marginal_stats(J[v], pc.K[v], LL_f, pc.K[v].size)
            N_v[v] += mean_node; S_v[v] += mean_edge
            acc(N_tail, v, tail_node); acc(S_tail, v, tail_edge)

    if min_copies == 0:
        one_minus_L0 = 1.0
        use_correction = False
    else:
        L0 = empty_log_likelihood(tree, rates)
        if min_copies == 2:
            L0 = float(np.logaddexp(L0, singleton_log_likelihood(tree, rates)))
        one_minus_L0 = -np.expm1(L0)
        use_correction = True

    grad = np.zeros(3 * n)
    for v in range(n):
        is_polya = bool(sp.is_polya[v])
        kappa_or_r = float(sp.gain[v])
        q = float(sp.q[v]); q_c = float(sp.q_c[v])

        # ---- GAIN derivative ----
        if is_polya:
            log1_q = float(sp.log_q_c[v])
            emp = F * log1_q / one_minus_L0 if use_correction else F * log1_q
            Nt = N_tail[v]; St = S_tail[v]
            for i in range(max(Nt.size, St.size)):
                n_t = Nt[i] if i < Nt.size else 0.0
                s_t = St[i] if i < St.size else 0.0
                emp += (n_t - s_t) / (kappa_or_r + i)
            grad[3 * v + GAIN] = emp
        else:
            if kappa_or_r > 0:
                if use_correction:
                    grad[3 * v + GAIN] = (N_v[v] - S_v[v]) / kappa_or_r - F / one_minus_L0
                else:
                    grad[3 * v + GAIN] = (N_v[v] - S_v[v]) / kappa_or_r - F

        # ---- DUP derivative ----
        if is_polya:
            if use_correction:
                grad[3 * v + DUP] = (N_v[v] - S_v[v]) / q - (S_v[v] + F * kappa_or_r / one_minus_L0) / q_c
            else:
                grad[3 * v + DUP] = (N_v[v] - S_v[v]) / q - (S_v[v] + F * kappa_or_r) / q_c

        # ---- LOSS derivative ----
        if v == tree.root:
            # p̃_root == 1 → derivative is undefined at the boundary; convention 0.
            # ROOTLOSS (p̃_root < 1) configuration is not implemented here.
            continue
        p = float(sp.p[v]); p_c = float(sp.p_c[v])
        parent = int(tree.parent[v])
        eps_sib = 1.0
        for c in tree.children[parent]:
            if c != v:
                eps_sib *= float(sp.p[c])
        one_minus_pe = 1.0 - p * eps_sib
        grad[3 * v + LOSS] = (
            (N_v[parent] - S_v[v]) / p - (1.0 - eps_sib) * S_v[v] / p_c
        ) / one_minus_pe

    return grad
