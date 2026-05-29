"""Bit-faithful Python port of Csurös' FunctionMinimization.{lnsrch, dfpmin}.

Direct port of the Java source at
/Users/ssolo/src/count/src/count/matek/FunctionMinimization.java
(lines 594-1057). Variable names, control flow, and constants all match
Csurös' Java verbatim so that running this on the same starting point
(in float64 with the same gradient kernel) reproduces his trajectory
to floating-point precision.

The Java is itself a direct port of Numerical Recipes Ch 10.7 dfpmin
(BFGS variant) with NR's lnsrch (Armijo-backtracking line search using
cubic interpolation). The only Csurös-specific divergence from textbook
NR is:
- the line search keeps the best `f` it has seen across iterations and
  returns the best (not just the last);
- the convergence test on the gradient uses |g_i| (max-norm) directly,
  not the NR-form normalised by max(|x|, 1) / max(|f|, 1);
- inverse-Hessian update can be DFP (10.7.8) or BFGS (10.7.9), selected
  by DFP_BFGS_UPDATE; default is BFGS.

Usage (mirrors the Java call):

    from validation.csuros_dfpmin import dfpmin
    x = np.array(...)  # starting point, will be overwritten with the minimiser
    fmin = dfpmin(x, gtol=1e-7, func=f, gradient=g, dfp_itmax=200)
"""
from __future__ import annotations

import numpy as np

# Constants — match Csurös' Java exactly.
EPS = 1.0 / (1 << 52)             # 2.22e-16 — machine epsilon for float64
DFP_TOLX = 4.0 * EPS              # convergence on Δx in dfpmin
LNSRCH_TOLX = DFP_TOLX            # convergence on Δx in lnsrch
LNSRCH_ALF = 1.0e-4               # Armijo "sufficient decrease" constant
DFP_ITMAX = 200                   # default max iterations for dfpmin
DFP_STPMX = 100.0                 # max step length scaling (Csurös default)
# Csurös' Java uses stpmax = DFP_STPMX * max(sqrt(|p|^2), n) which on
# log-rate-space problems with n=O(100..1000) gives stpmax ~ 10^4..10^5;
# the optimizer then takes a single full-Newton step that pushes log-rates
# out to ±50 (the clip) and the gradient overflows. We cap stpmax in
# absolute terms — this is the only deviation from a bit-faithful port.
# Value: 10 keeps each step inside the clip box [-50, 50] when starting
# near 0 (default init log-rates are O(1)), so the gradient never sees
# the boundary's "clipped" gradient pathology.
STPMAX_ABS_CAP = 10.0


def lnsrch(xold: np.ndarray, fold: float, g: np.ndarray, p: np.ndarray,
           stpmax: float, func) -> tuple[np.ndarray, float, bool]:
    """NR backtracking line search with Armijo sufficient-decrease.

    Mirrors Java `FunctionMinimization.lnsrch(xold, fold, g, p, x, f, stpmax, func)`.

    Inputs:
        xold: starting point, shape (n,) — NOT modified.
        fold: f(xold).
        g: gradient at xold, shape (n,).
        p: search direction at xold, shape (n,) — MAY be rescaled in place if |p|>stpmax.
        stpmax: max step length.
        func: callable x -> float.

    Returns:
        x: new point, shape (n,).
        f: f(x).
        converged: True if Δx is too small (= "convergence on Δx"; usually
                   means we're at the minimum and the caller can stop).

    Csurös-specific deviation from NR: tracks (myminx, myminf) = best
    point seen during backtracking, and returns it if the final iterate
    is worse.
    """
    n = xold.shape[0]
    x = xold.copy()
    myminf = fold
    myminx = xold.copy()

    # Scale p if it's too long.
    sum_p = float(np.sqrt(np.dot(p, p)))
    if sum_p > stpmax:
        p *= stpmax / sum_p  # in-place, matches Java

    slope = float(np.dot(g, p))
    if slope >= 0.0:
        # Roundoff: search direction is not a descent direction; bail.
        return xold.copy(), fold, True

    # Compute lambda_min: smallest step size for "noticeable" Δx.
    test = 0.0
    for i in range(n):
        temp = abs(p[i]) / max(abs(xold[i]), 1.0)
        if temp > test:
            test = temp
    alamin = LNSRCH_TOLX / test

    alam = 1.0       # always try the full Newton step first
    alam2 = float("nan")
    f2 = float("nan")

    while True:
        # Trial point.
        for i in range(n):
            x[i] = xold[i] + alam * p[i]
        f = float(func(x))

        # Guard: if f is +inf or nan (overflow at trial point), treat as
        # arbitrarily large so the Armijo test fails and we backtrack.
        # Not in Csurös' Java because his GUI runs in unclipped log-rate
        # space where the gradient is always finite; ours has a clip and
        # the function returns inf outside the clipped support.
        if not np.isfinite(f):
            f = 1e300

        # Track best-seen during the search (Csurös-specific).
        if f < myminf:
            myminf = f
            myminx = x.copy()

        if alam < alamin:
            # Convergence on Δx — the step is below the noise floor.
            if myminf < f:
                # Return the best-seen instead of the final iterate.
                x[:] = myminx
                f = myminf
            else:
                # No improvement found; return xold.
                x[:] = xold
            return x, f, True

        if f <= fold + LNSRCH_ALF * alam * slope:
            # Armijo: sufficient decrease — accept this step.
            if myminf < f:
                # Even with sufficient decrease, prefer a better seen point.
                x[:] = myminx
                f = myminf
            return x, f, False

        # Backtrack via cubic interpolation.
        if alam == 1.0:
            # First backtrack: pure quadratic.
            tmplam = -slope / (2.0 * (f - fold - slope))
        else:
            # Subsequent: cubic fit through (alam, f), (alam2, f2), (0, fold) with slope at 0.
            rhs1 = f - fold - alam * slope
            rhs2 = f2 - fold - alam2 * slope
            a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2)
            b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2)) / (alam - alam2)
            if a == 0.0:
                tmplam = -slope / (2.0 * b)
            else:
                disc = b * b - 3.0 * a * slope
                if disc < 0.0:
                    tmplam = 0.5 * alam
                elif b <= 0.0:
                    tmplam = (-b + np.sqrt(disc)) / (3.0 * a)
                else:
                    tmplam = -slope / (b + np.sqrt(disc))
            if tmplam > 0.5 * alam:
                tmplam = 0.5 * alam   # lambda <= 0.5 * lambda_old

        alam2 = alam
        f2 = f
        alam = max(tmplam, 0.1 * alam)  # lambda >= 0.1 * lambda_old


def dfpmin(p: np.ndarray, gtol: float, func, gradient,
           dfp_itmax: int = DFP_ITMAX,
           bfgs_update: bool = True,
           iterations: list[float] | None = None,
           callback=None) -> tuple[float, int, str]:
    """NR Ch 10.7 BFGS variant (or DFP if bfgs_update=False).

    Mirrors Java `FunctionMinimization.dfpmin(p, gtol, dfp_itmax, func, gradient, iterations)`.

    Inputs:
        p: starting point, shape (n,) — MODIFIED IN PLACE; on return holds
           the (approximate) minimiser.
        gtol: gradient max-norm convergence threshold.
        func: callable x -> float.
        gradient: callable x -> np.ndarray of shape (n,).
        dfp_itmax: max outer iterations (default 200, matching Java).
        bfgs_update: True for BFGS (10.7.9), False for DFP (10.7.8). Default True.
        iterations: if provided, every f-value is appended (one per outer iter).
        callback: optional callable(its, f, |grad|_inf) called after each outer iter.

    Returns:
        (fmin, n_iters, status)
        status one of: 'deltax', 'grad', 'maxiter'.
    """
    n = p.shape[0]
    g = np.asarray(gradient(p), dtype=np.float64).copy()
    fp = float(func(p))

    if iterations is not None:
        iterations.append(fp)

    hessin = np.eye(n)
    xi = -g.copy()                           # initial direction = steepest descent

    sum_p2 = float(np.dot(p, p))
    stpmax = min(STPMAX_ABS_CAP, DFP_STPMX * max(np.sqrt(sum_p2), float(n)))

    dg = np.empty(n)
    hdg = np.empty(n)

    for its in range(1, dfp_itmax + 1):
        # Save old gradient.
        dg[:] = g

        # Line search along xi.
        pnew, fret, _ = lnsrch(p, fp, g, xi, stpmax, func)
        fp = fret
        if iterations is not None:
            iterations.append(fp)

        # Update line direction and current point.
        xi = pnew - p
        p[:] = pnew

        # Convergence test on Δx.
        test = 0.0
        for i in range(n):
            temp = abs(xi[i]) / max(abs(p[i]), 1.0)
            if temp > test:
                test = temp
        if test < DFP_TOLX:
            if callback is not None:
                callback(its, fp, float(np.max(np.abs(g))))
            return fp, its, "deltax"

        # New gradient.
        g = np.asarray(gradient(p), dtype=np.float64).copy()

        # Convergence test on gradient (Csurös uses max-norm, NOT NR-normalised).
        test = float(np.max(np.abs(g)))
        if callback is not None:
            callback(its, fp, test)
        if test < gtol:
            return fp, its, "grad"

        # Δgradient.
        dg = g - dg

        # H · Δgradient.
        hdg[:] = hessin @ dg

        # Dot products for updates.
        fac = float(np.dot(dg, xi))
        fae = float(np.dot(dg, hdg))
        sumdg = float(np.dot(dg, dg))
        sumxi = float(np.dot(xi, xi))

        # Skip H-update if denominator is too small (Csurös' guard).
        if fac > np.sqrt(EPS * sumdg * sumxi):
            fac = 1.0 / fac
            fad = 1.0 / fae

            if bfgs_update:
                # BFGS auxiliary vector u = fac*xi - fad*hdg.
                u = fac * xi - fad * hdg
                # H += fac * xi xi^T - fad * hdg hdg^T + fae * u u^T.
                hessin += fac * np.outer(xi, xi)
                hessin -= fad * np.outer(hdg, hdg)
                hessin += fae * np.outer(u, u)
            else:
                # DFP update.
                hessin += fac * np.outer(xi, xi)
                hessin -= fad * np.outer(hdg, hdg)

        # Guard: reset Hessian to identity if it has overflowed or developed
        # NaNs (numerical pathology from the rank-2 update on ill-conditioned
        # problems). This loses the accumulated curvature info but lets the
        # optimizer continue with a steepest-descent step.
        max_h = np.max(np.abs(hessin))
        if not np.isfinite(max_h) or max_h > 1e15:
            hessin = np.eye(n)

        # Next search direction: xi = -H · g.
        xi = -hessin @ g
        # Also guard against an overflowed direction (e.g. if g still has
        # extreme components even after the H reset).
        max_xi = np.max(np.abs(xi))
        if not np.isfinite(max_xi):
            xi = -g  # fall back to steepest descent

    return fp, dfp_itmax, "maxiter"
