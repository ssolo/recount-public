"""Per-edge rate parameters for the GLD birth-death process.

Each edge of the tree carries:
    μ  — loss rate (per-copy)
    λ  — duplication rate (per-copy)
    γ  — gain rate (copy-independent)  — stored as r if λ=0, κ if λ>0
    t  — edge length

The closed-form solution of the linear birth-death process on an edge of
length t yields the per-copy probability parameters:
    p = P(a single ancestral copy produces 0 surviving descendants on the edge)
    q = the duplication parameter of the resulting Shifted Geometric / NegBin

At the root, edge length is +∞ and p_root = 1 by convention — i.e. no copies
survive from "before the root", making the root marginal equal to the gain
distribution alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from recount.tree import Tree


@dataclass
class GLDRates:
    """Per-node GLD rate parameters.

    Arrays have shape (num_nodes,).  At the root, length is +∞ by convention.
    The `gain` field is interpreted as the Poisson rate r when dup==0, or as
    the Pólya/NegBinomial shape κ when dup>0.
    """
    tree: Tree
    gain: np.ndarray
    loss: np.ndarray
    dup: np.ndarray
    length: np.ndarray

    def __post_init__(self) -> None:
        n = self.tree.num_nodes
        self.gain = np.asarray(self.gain, dtype=np.float64).copy()
        self.loss = np.asarray(self.loss, dtype=np.float64).copy()
        self.dup = np.asarray(self.dup, dtype=np.float64).copy()
        self.length = np.asarray(self.length, dtype=np.float64).copy()
        for arr, name in [
            (self.gain, "gain"), (self.loss, "loss"),
            (self.dup, "dup"), (self.length, "length"),
        ]:
            if arr.shape != (n,):
                raise ValueError(f"rates.{name} must have shape ({n},); got {arr.shape}")


def rate_to_p(mu: float, lam: float, t: float) -> Tuple[float, float]:
    """Per-edge loss probability p and its complement (1-p).

    Returns the closed-form solution of the linear birth-death process on
    an edge of length t with rates (μ, λ). Both p and 1-p are returned —
    each is computed via whichever branch is numerically safer:

        * λ == μ : p = μt/(1+μt)
        * λ <  μ : p = μ(1-E)/(μ - λE),  E = exp(-(μ-λ)t)
        * λ >  μ : p = μ(1-E)/(λ - μE),  E = exp(-(λ-μ)t)
        * λ == 0 : p = 1 - exp(-μt)
        * μ == 0 : p = 0
    """
    if mu == 0.0:
        return 0.0, 1.0
    if mu == lam:
        mu_t = mu * t
        if np.isinf(mu_t):
            return 1.0, 0.0
        return mu_t / (1.0 + mu_t), 1.0 / (1.0 + mu_t)
    if lam == 0.0:
        return -np.expm1(-mu * t), np.exp(-mu * t)
    if lam < mu:
        gap = mu - lam
        d = gap * t
        E = np.exp(-d)
        E1 = -np.expm1(-d)
        delta = gap / mu
        if delta < 0.5:
            denom = mu * (-np.expm1(-d + np.log1p(-delta)))
        else:
            denom = mu - lam * E
        p = mu * E1 / denom
        p_c = gap * E / denom
        return p, p_c
    # lam > mu
    gap = lam - mu
    d = gap * t
    E = np.exp(-d)
    E1 = -np.expm1(-d)
    delta = gap / lam
    if delta < 0.5:
        denom = lam * (-np.expm1(-d + np.log1p(-delta)))
    else:
        denom = lam - mu * E
    p = mu * E1 / denom
    p_c = gap / denom
    return p, p_c


def rate_to_q(mu: float, lam: float, t: float) -> Tuple[float, float]:
    """Per-edge duplication parameter q and its complement (1-q)."""
    if lam == 0.0:
        return 0.0, 1.0
    if mu == lam:
        lam_t = lam * t
        if np.isinf(lam_t):
            return 1.0, 0.0
        return lam_t / (1.0 + lam_t), 1.0 / (1.0 + lam_t)
    if lam < mu:
        gap = mu - lam
        d = gap * t
        E = np.exp(-d)
        E1 = -np.expm1(-d)
        delta = gap / mu
        if delta < 0.5:
            denom = mu * (-np.expm1(-d + np.log1p(-delta)))
        else:
            denom = mu - lam * E
        return lam * E1 / denom, gap / denom
    # lam > mu
    gap = lam - mu
    d = gap * t
    E = np.exp(-d)
    E1 = -np.expm1(-d)
    delta = gap / lam
    if delta < 0.5:
        denom = lam * (-np.expm1(-d + np.log1p(-delta)))
    else:
        denom = lam - mu * E
    return lam * E1 / denom, gap * E / denom


def rate_to_p_jac(mu: float, lam: float, t: float) -> Tuple[float, float, float]:
    """Analytical Jacobian (∂p/∂μ, ∂p/∂λ, ∂p/∂t) of rate_to_p.

    Validated to 1e-8 vs centered finite differences on the dominant
    (μ ≠ λ, both > 0) branch. The μ = λ branch returns the one-sided
    limit (∂p/∂λ = 0 from the λ < μ side); the optimizer never lands
    exactly at μ = λ in practice (measure zero).

    The p·(1/μ - X) form is rewritten as E1/D + p·(... no 1/μ term ...)
    to stay finite at μ → 0+ (where p = 0 but ∂p/∂μ = E1/D ≠ 0).

    Used by the reverse-mode chain rule that converts the analytical
    log-survival gradient (logit p̃, logit q̃, log κ) to (gain, loss,
    dup, length) — replacing torch autograd in
    compute_L0_gradient_analytical.
    """
    if np.isinf(t):
        # t=∞ root edge: p reaches its stationary limit. ∂p/∂t vanishes, but
        # the limiting p still depends on (μ,λ) when λ > μ (p = μ/λ); for
        # λ ≤ μ (or μ = 0) p is the constant 0 or 1 and all derivatives are
        # 0. Returning 0,0,0 unconditionally — the old behaviour — silently
        # pinned a super-critical root edge's loss/dup rates during fits.
        if mu != 0.0 and lam > mu:
            return 1.0 / lam, -mu / (lam * lam), 0.0
        return 0.0, 0.0, 0.0
    if mu == 0.0:
        # p = 0; ∂p/∂μ from the limit: μ=0 means lam ≥ 0 dominates.
        if lam == 0.0:
            return t, 0.0, 0.0  # μ=λ=0: p = μt/(1+μt) so dp/dμ|₀ = t
        # General lam > 0 case: dp/dμ at μ=0 = E1/D|_{μ=0} = (1-exp(-λt))/λ
        return (1.0 - np.exp(-lam * t)) / lam, 0.0, 0.0
    # Near-Yule: when |mu-lam|/max(mu,lam) is below ~1e-8 the general
    # formula `p*(t*E/E1 - (1+lam*t*E)/D)` catastrophically cancels
    # because t*E/E1 and (1+lam*t*E)/D both blow up like 1/gap and
    # their difference loses all ULPs. Use the Yule formula in that
    # neighborhood. (When mu == lam exactly we'd also take this branch.)
    YULE_TOL = 1e-7
    if mu == lam or abs(mu - lam) <= YULE_TOL * max(mu, lam):
        # Yule limit p = x·t/(1+x·t). The derivatives w.r.t. μ and λ
        # individually are NOT the one-sided naive limits — they come from
        # Taylor-expanding the general formula around μ=λ. Validated to FD:
        #   ∂p/∂μ = t·(2+x·t) / (2·(1+x·t)²)
        #   ∂p/∂λ = -x·t² / (2·(1+x·t)²)
        #   ∂p/∂t = x / (1+x·t)² = x·pc²
        x = 0.5 * (mu + lam)         # tiny averaging — both are essentially x
        u = x * t
        if np.isinf(u):
            return 0.0, 0.0, 0.0
        one_plus_u_sq = (1.0 + u) ** 2
        pc2 = 1.0 / one_plus_u_sq
        dp_dmu  = t * (2.0 + u) / (2.0 * one_plus_u_sq)
        dp_dlam = -x * t * t / (2.0 * one_plus_u_sq)
        dp_dt   = x * pc2
        return dp_dmu, dp_dlam, dp_dt
    if lam == 0.0:
        p_c = np.exp(-mu * t)
        return t * p_c, 0.0, mu * p_c
    if lam < mu:
        gap = mu - lam
        d = gap * t
        E = np.exp(-d)
        E1 = -np.expm1(-d)
        D = mu - lam * E
        p = mu * E1 / D
        # p/mu = E1/D (algebraic identity); rewrite p*(1/mu - X) as E1/D - p*X.
        dp_dmu  = E1/D + p * (t*E/E1 - (1.0 + lam*t*E)/D)
        dp_dlam = p * (-t*E/E1 + E*(1.0 + lam*t)/D)
        dp_dt   = p * gap * E * (1.0/E1 - lam/D)
        return dp_dmu, dp_dlam, dp_dt
    # lam > mu
    gap = lam - mu
    d = gap * t
    E = np.exp(-d)
    E1 = -np.expm1(-d)
    D = lam - mu * E
    p = mu * E1 / D
    dp_dmu  = E1/D + p * (-t*E/E1 + E*(1.0 + mu*t)/D)
    dp_dlam = p * (t*E/E1 - (1.0 + mu*t*E)/D)
    dp_dt   = p * gap * E * (1.0/E1 - mu/D)
    return dp_dmu, dp_dlam, dp_dt


def rate_to_q_jac(mu: float, lam: float, t: float) -> Tuple[float, float, float]:
    """Analytical Jacobian (∂q/∂μ, ∂q/∂λ, ∂q/∂t) of rate_to_q.

    See rate_to_p_jac docstring; same structure and validation.
    Care is taken at λ=0 — q=0 but ∂q/∂λ = (1-exp(-μt))/μ from the limit,
    not zero. For the general branches the q·(1/λ - ...) form is
    rewritten as E1/D + λ·(...) to keep finite as λ → 0+.
    """
    if np.isinf(t):
        # t=∞ root edge: q reaches its stationary limit. ∂q/∂t vanishes, but
        # the limiting q still depends on (μ,λ) when λ < μ (q = λ/μ); for
        # λ ≥ μ (or λ = 0) q is the constant 0 or 1 and all derivatives are
        # 0. Returning 0,0,0 unconditionally — the old behaviour — silently
        # pinned the root edge's dup rate during ML/MAP fits.
        if lam != 0.0 and lam < mu:
            return -lam / (mu * mu), 1.0 / mu, 0.0
        return 0.0, 0.0, 0.0
    if lam == 0.0:
        # q = 0 (so ∂q/∂μ and ∂q/∂t both vanish); ∂q/∂λ from the limit
        # of λ·E1/D as λ→0+: dq/dλ|₀ = E1/D|_{λ=0} = (1 - exp(-μt))/μ.
        if mu == 0.0:
            return 0.0, t, 0.0      # μ=λ=0 limit: dq/dλ = t
        p_c = np.exp(-mu * t)
        return 0.0, (1.0 - p_c) / mu, 0.0
    YULE_TOL = 1e-7
    if mu == lam or abs(mu - lam) <= YULE_TOL * max(mu, lam):
        # Yule limit q = x·t/(1+x·t). By symmetry with rate_to_p_jac:
        #   ∂q/∂μ = -x·t² / (2·(1+x·t)²)      [was wrongly 0]
        #   ∂q/∂λ = t·(2+x·t) / (2·(1+x·t)²)  [was wrongly t·qc²]
        #   ∂q/∂t = x · qc²
        x = 0.5 * (mu + lam)
        u = x * t
        if np.isinf(u):
            return 0.0, 0.0, 0.0
        one_plus_u_sq = (1.0 + u) ** 2
        qc2 = 1.0 / one_plus_u_sq
        dq_dmu  = -x * t * t / (2.0 * one_plus_u_sq)
        dq_dlam = t * (2.0 + u) / (2.0 * one_plus_u_sq)
        dq_dt   = x * qc2
        return dq_dmu, dq_dlam, dq_dt
    if lam < mu:
        gap = mu - lam
        d = gap * t
        E = np.exp(-d)
        E1 = -np.expm1(-d)
        D = mu - lam * E
        q = lam * E1 / D
        # Rewrite q*(1/lam - X) as E1/D - q*X to avoid 0*inf at lam=0:
        #   q/lam = E1/D
        # Remaining q*X is finite at any lam.
        dq_dmu  = q * (t*E/E1 - (1.0 + lam*t*E)/D)
        dq_dlam = E1/D + q * (-t*E/E1 + E*(1.0 + lam*t)/D)
        dq_dt   = q * gap * E * (1.0/E1 - lam/D)
        return dq_dmu, dq_dlam, dq_dt
    # lam > mu
    gap = lam - mu
    d = gap * t
    E = np.exp(-d)
    E1 = -np.expm1(-d)
    D = lam - mu * E
    q = lam * E1 / D
    dq_dmu  = q * (-t*E/E1 + E*(1.0 + mu*t)/D)
    # Same q/lam = E1/D trick on this branch.
    dq_dlam = E1/D + q * (t*E/E1 - (1.0 + mu*t*E)/D)
    dq_dt   = q * gap * E * (1.0/E1 - mu/D)
    return dq_dmu, dq_dlam, dq_dt
