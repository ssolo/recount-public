/* Port of recount/rates.py — rate_to_p, rate_to_q.
 *
 * Per-edge linear birth-death (μ loss, λ duplication) on interval t:
 *   p̃ = P(lineage extinct on edge) before gain
 *   q̃ = P(dup happens)
 *
 * Branches:
 *   μ == 0:    p = 0, q = 0 (no loss, no dup — Poisson gain only)
 *   μ == λ:   p = q = μt / (1 + μt)
 *   λ == 0:    p = 1 - exp(-μt) (pure death)
 *   λ < μ:    stable form using gap = μ - λ
 *   λ > μ:    stable form using gap = λ - μ
 *   t == inf: limit cases
 *
 * Mirrors recount/torch_fast.py::_rate_to_pq lines 130-222 (which is the
 * vectorized torch port of the scalar formulas in recount/rates.py).
 */
#include <math.h>
#include "recount_native.h"

/* All formulas use a small epsilon to avoid divides by 0 at exact boundaries. */
#define EPS_DENOM 1e-300

typedef struct { double p, p_c, q, q_c; } pq_t;

static pq_t rate_to_pq(double mu, double lam, double t)
{
    pq_t out;

    if (mu == 0.0) {
        out.p = 0.0; out.p_c = 1.0;
        out.q = 0.0; out.q_c = 1.0;
        return out;
    }

    if (isinf(t)) {
        /* t = +inf: limit cases */
        if (mu == lam) {
            out.p = 1.0; out.p_c = 0.0;
            out.q = 1.0; out.q_c = 0.0;
        } else if (lam == 0.0) {
            out.p = 1.0; out.p_c = 0.0;
            out.q = 0.0; out.q_c = 1.0;
        } else if (lam < mu) {
            out.p = 1.0; out.p_c = 0.0;
            out.q = lam / mu;  out.q_c = 1.0 - lam / mu;
        } else { /* lam > mu */
            out.p = mu / lam;  out.p_c = 1.0 - mu / lam;
            out.q = 1.0; out.q_c = 0.0;
        }
        return out;
    }

    if (mu == lam) {
        double mt = mu * t;
        out.p   = mt / (1.0 + mt);
        out.p_c = 1.0 / (1.0 + mt);
        out.q   = out.p;
        out.q_c = out.p_c;
        return out;
    }

    if (lam == 0.0) {
        /* Poisson loss: p = 1 - exp(-μt), q = 0 */
        double E = exp(-mu * t);
        out.p   = -expm1(-mu * t);   /* numerically stable for small μt */
        out.p_c = E;
        out.q   = 0.0;
        out.q_c = 1.0;
        return out;
    }

    /* General case: μ != λ, both > 0. Use the stable form when the gap is
     * small relative to max(μ, λ): -expm1(-d + log1p(-δ)) where d = gap·t,
     * δ = gap/max(μ,λ) — avoids catastrophic cancellation in μ - λ·E. */
    if (lam < mu) {
        double gap = mu - lam;
        double d = gap * t;
        double E = exp(-d);
        double E1 = -expm1(-d);
        double delta = gap / (mu > EPS_DENOM ? mu : EPS_DENOM);
        double denom;
        if (delta < 0.5) {
            double clamped = delta > (1.0 - 1e-15) ? (1.0 - 1e-15) : delta;
            denom = mu * (-expm1(-d + log1p(-clamped)));
        } else {
            denom = mu - lam * E;
        }
        if (denom < EPS_DENOM) denom = EPS_DENOM;
        out.p   = mu  * E1 / denom;
        out.p_c = gap * E  / denom;
        out.q   = lam * E1 / denom;
        out.q_c = gap      / denom;
    } else { /* lam > mu */
        double gap = lam - mu;
        double d = gap * t;
        double E = exp(-d);
        double E1 = -expm1(-d);
        double delta = gap / (lam > EPS_DENOM ? lam : EPS_DENOM);
        double denom;
        if (delta < 0.5) {
            double clamped = delta > (1.0 - 1e-15) ? (1.0 - 1e-15) : delta;
            denom = lam * (-expm1(-d + log1p(-clamped)));
        } else {
            denom = lam - mu * E;
        }
        if (denom < EPS_DENOM) denom = EPS_DENOM;
        out.p   = mu  * E1 / denom;
        out.p_c = gap      / denom;
        out.q   = lam * E1 / denom;
        out.q_c = gap * E  / denom;
    }
    return out;
}

/* Out-of-line entry points used by survival_params. */
void recount_rate_to_pq(double mu, double lam, double t,
                        double *p, double *p_c, double *q, double *q_c)
{
    pq_t r = rate_to_pq(mu, lam, t);
    *p = r.p; *p_c = r.p_c;
    *q = r.q; *q_c = r.q_c;
}

/*
 * Analytical Jacobians of (p, q) w.r.t. (μ, λ, t).
 *
 * Mirrors recount/rates.py::rate_to_p_jac / rate_to_q_jac. Uses the same
 * Yule-tolerance branch (|μ-λ|/max(μ,λ) ≤ 1e-7) to dodge catastrophic
 * cancellation in `t·E/E1` vs `(1+λ·t·E)/D` when μ ≈ λ.
 *
 * Validated against centered FD to 1e-10 on the dominant branches; at the
 * exact Yule point and the t=∞ root the analytical limits are exact.
 *
 * Outputs (each of these is a single double):
 *   *dp_dmu  *dp_dlam  *dp_dt   ← ∂p/∂(μ,λ,t)
 *   *dq_dmu  *dq_dlam  *dq_dt   ← ∂q/∂(μ,λ,t)
 *
 * (∂p_c/∂x = -∂p/∂x and ∂q_c/∂x = -∂q/∂x by the p + p_c = 1 identity.)
 */
#define RECOUNT_YULE_TOL 1e-7

void recount_rate_to_pq_jac(double mu, double lam, double t,
                            double *dp_dmu, double *dp_dlam, double *dp_dt,
                            double *dq_dmu, double *dq_dlam, double *dq_dt)
{
    /* Defaults */
    *dp_dmu = 0.0; *dp_dlam = 0.0; *dp_dt = 0.0;
    *dq_dmu = 0.0; *dq_dlam = 0.0; *dq_dt = 0.0;

    if (isinf(t)) {
        /* t=∞ root edge: p, q reach their stationary limits (see the
         * matching isinf(t) branch of rate_to_pq above). The ∂/∂t
         * derivatives vanish, but the limiting p, q still depend on μ, λ:
         *   μ == 0:   p ≡ 0,   q ≡ 0    → all derivs 0
         *   μ == λ:   p ≡ 1,   q ≡ 1    → all derivs 0
         *   λ == 0:   p ≡ 1,   q ≡ 0    → all derivs 0
         *   λ <  μ:   p ≡ 1,   q  = λ/μ → ∂q/∂μ = -λ/μ², ∂q/∂λ = 1/μ
         *   λ >  μ:   p  = μ/λ, q ≡ 1   → ∂p/∂μ = 1/λ,   ∂p/∂λ = -μ/λ²
         * Zeroing every entry here (the old behaviour) silently pinned the
         * root edge's dup rate during ML/MAP fits: ∂q̃/∂λ at the root was
         * dropped, so the optimizer never moved dup[root]. */
        if (mu != 0.0 && lam != 0.0 && mu != lam) {
            if (lam < mu) {
                *dq_dmu  = -lam / (mu * mu);
                *dq_dlam = 1.0 / mu;
            } else { /* lam > mu */
                *dp_dmu  = 1.0 / lam;
                *dp_dlam = -mu / (lam * lam);
            }
        }
        return;
    }

    /* rate_to_p part: branch handling matches rate_to_p_jac in rates.py. */
    if (mu == 0.0) {
        /* p = 0; ∂p/∂μ from the limit. */
        if (lam == 0.0) {
            *dp_dmu = t;       /* p = μt/(1+μt), dp/dμ|₀ = t */
        } else {
            *dp_dmu = (1.0 - exp(-lam * t)) / lam;
        }
        /* dp_dlam, dp_dt stay 0 */
    } else {
        double max_ml = mu > lam ? mu : lam;
        double gap_abs = mu > lam ? (mu - lam) : (lam - mu);
        int near_yule = (mu == lam) || (gap_abs <= RECOUNT_YULE_TOL * max_ml);

        if (near_yule) {
            double x = 0.5 * (mu + lam);
            double u = x * t;
            if (isinf(u)) {
                /* leave as 0 */
            } else {
                double one_plus_u_sq = (1.0 + u) * (1.0 + u);
                double pc2 = 1.0 / one_plus_u_sq;
                *dp_dmu  = t * (2.0 + u) / (2.0 * one_plus_u_sq);
                *dp_dlam = -x * t * t / (2.0 * one_plus_u_sq);
                *dp_dt   = x * pc2;
            }
        } else if (lam == 0.0) {
            double p_c = exp(-mu * t);
            *dp_dmu = t * p_c;
            *dp_dt  = mu * p_c;
        } else if (lam < mu) {
            double gap = mu - lam;
            double d = gap * t;
            double E = exp(-d);
            double E1 = -expm1(-d);
            double D = mu - lam * E;
            double p = mu * E1 / D;
            /* p/μ = E1/D used to dodge 0*∞ at μ → 0; here μ > 0. */
            *dp_dmu  = E1/D + p * (t*E/E1 - (1.0 + lam*t*E)/D);
            *dp_dlam = p * (-t*E/E1 + E * (1.0 + lam*t) / D);
            *dp_dt   = p * gap * E * (1.0/E1 - lam/D);
        } else { /* lam > mu */
            double gap = lam - mu;
            double d = gap * t;
            double E = exp(-d);
            double E1 = -expm1(-d);
            double D = lam - mu * E;
            double p = mu * E1 / D;
            *dp_dmu  = E1/D + p * (-t*E/E1 + E * (1.0 + mu*t) / D);
            *dp_dlam = p * (t*E/E1 - (1.0 + mu*t*E)/D);
            *dp_dt   = p * gap * E * (1.0/E1 - mu/D);
        }
    }

    /* rate_to_q part: similar structure. */
    if (lam == 0.0) {
        if (mu == 0.0) {
            *dq_dlam = t;       /* q = λt/(1+λt) at limit */
        } else {
            *dq_dlam = (1.0 - exp(-mu * t)) / mu;
        }
        return;
    }
    {
        double max_ml = mu > lam ? mu : lam;
        double gap_abs = mu > lam ? (mu - lam) : (lam - mu);
        int near_yule = (mu == lam) || (gap_abs <= RECOUNT_YULE_TOL * max_ml);

        if (near_yule) {
            double x = 0.5 * (mu + lam);
            double u = x * t;
            if (isinf(u)) {
                /* leave as 0 */
            } else {
                double one_plus_u_sq = (1.0 + u) * (1.0 + u);
                double qc2 = 1.0 / one_plus_u_sq;
                *dq_dmu  = -x * t * t / (2.0 * one_plus_u_sq);
                *dq_dlam = t * (2.0 + u) / (2.0 * one_plus_u_sq);
                *dq_dt   = x * qc2;
            }
        } else if (lam < mu) {
            double gap = mu - lam;
            double d = gap * t;
            double E = exp(-d);
            double E1 = -expm1(-d);
            double D = mu - lam * E;
            double q = lam * E1 / D;
            *dq_dmu  = q * (t*E/E1 - (1.0 + lam*t*E)/D);
            *dq_dlam = E1/D + q * (-t*E/E1 + E * (1.0 + lam*t) / D);
            *dq_dt   = q * gap * E * (1.0/E1 - lam/D);
        } else { /* lam > mu */
            double gap = lam - mu;
            double d = gap * t;
            double E = exp(-d);
            double E1 = -expm1(-d);
            double D = lam - mu * E;
            double q = lam * E1 / D;
            *dq_dmu  = q * (-t*E/E1 + E * (1.0 + mu*t) / D);
            *dq_dlam = E1/D + q * (t*E/E1 - (1.0 + mu*t*E)/D);
            *dq_dt   = q * gap * E * (1.0/E1 - mu/D);
        }
    }
}
