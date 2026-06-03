/*
 * recount_bfgs.c — bit-faithful C port of Csuros' Java dfpmin + lnsrch.
 *
 * Source: count/src/count/matek/FunctionMinimization.java (BSD-style,
 * derived from NumRecipes Ch 10.7 with BFGS update + Armijo lnsrch).
 *
 * The optimizer mutates x[] in place; on return x[] holds the best-seen
 * iterate (smallest f) — matches Java's xbest tracking.
 */

#include "recount_bfgs.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Constants matching Java FunctionMinimization. */
static const double DFP_EPS         = 2.220446049250313e-16;  /* 1.0/(1<<52) */
static const double DFP_TOLX        = 4.0 * 2.220446049250313e-16;
static const double LNSRCH_TOLX     = 4.0 * 2.220446049250313e-16;
static const double LNSRCH_ALF      = 1.0e-4;
static const double DFP_STPMX       = 100.0;

/* Same as csuros_dfpmin.py STPMAX_ABS_CAP: the Java default of
 *   stpmax = DFP_STPMX * max(sqrt(|p|^2), n)
 * gives stpmax ~10^4-10^5 on n=O(100..1000) problems, which lets the very
 * first full-Newton step push log-rates out to ±50 (the clip box) where the
 * GLD likelihood's gradient is degenerate (L(0)→1 region: -F·log(1-L(0))
 * inflates the LL artificially). Capping the first-iter step keeps the
 * walker inside a well-behaved region. The cap only binds when the natural
 * stpmax would exceed 10, so well-scaled problems (small Rosenbrock etc.)
 * are unaffected. */
static const double STPMAX_ABS_CAP  = 10.0;

/* Armijo backtracking line search — NumRecipes lnsrch / Java
 * FunctionMinimization.lnsrch.
 *
 * Inputs:
 *   xold[n] : starting point
 *   fold    : f(xold)
 *   g[n]    : grad(xold)
 *   p[n]    : search direction (MAY BE RESCALED INTERNALLY if |p| > stpmax)
 *   stpmax  : max scaled step
 *   n       : dimension
 *   objgrad : f+g callback
 *   user    : opaque user pointer
 *
 * Outputs:
 *   xnew[n]    : accepted point on success; copy of xold on failure
 *   *fnew      : f(xnew) on success
 *   gnew_out[n]: grad(xnew) on success (caller-allocated; computed inside)
 *
 * Returns: 1 on success, 0 if line search failed (alam < alamin).
 */
static int lnsrch(const double *xold, double fold, const double *g,
                  double *p, double stpmax, int n,
                  recount_objgrad_fn objgrad, void *user,
                  double *xnew, double *fnew, double *gnew_out)
{
    int i;

    /* Scale p if too large. */
    double sum = 0.0;
    for (i = 0; i < n; i++) sum += p[i] * p[i];
    sum = sqrt(sum);
    if (sum > stpmax) {
        double scale = stpmax / sum;
        for (i = 0; i < n; i++) p[i] *= scale;
    }
    /* Slope at starting point along direction p. */
    double slope = 0.0;
    for (i = 0; i < n; i++) slope += g[i] * p[i];
    if (slope >= 0.0) {
        /* Not a descent direction. */
        memcpy(xnew, xold, n * sizeof(double));
        *fnew = fold;
        return 0;
    }

    /* Minimum step size. */
    double test = 0.0;
    for (i = 0; i < n; i++) {
        double denom = fmax(fabs(xold[i]), 1.0);
        double tmp = fabs(p[i]) / denom;
        if (tmp > test) test = tmp;
    }
    double alamin = LNSRCH_TOLX / fmax(test, 1e-300);

    double alam = 1.0, alam2 = 0.0, f2 = 0.0;

    while (1) {
        for (i = 0; i < n; i++) xnew[i] = xold[i] + alam * p[i];
        *fnew = objgrad(xnew, n, gnew_out, user);

        if (alam < alamin) {
            /* Step too small — restore xold. */
            memcpy(xnew, xold, n * sizeof(double));
            *fnew = fold;
            return 0;
        }
        /* Reject any non-finite f (NaN/Inf from degenerate regions like
         * L(0)→1 or extreme rates). Forces lnsrch to backtrack rather
         * than wedge the optimizer with a NaN gradient. */
        int fnew_ok = isfinite(*fnew);
        if (fnew_ok && *fnew <= fold + LNSRCH_ALF * alam * slope) {
            /* Sufficient decrease — done. */
            return 1;
        }

        /* Backtrack via cubic / quadratic model. */
        double tmplam;
        if (alam == 1.0) {
            tmplam = -slope / (2.0 * (*fnew - fold - slope));
        } else {
            double rhs1 = *fnew - fold - alam * slope;
            double rhs2 = f2    - fold - alam2 * slope;
            double a = (rhs1/(alam*alam)   - rhs2/(alam2*alam2)) / (alam - alam2);
            double b = (-alam2*rhs1/(alam*alam) + alam*rhs2/(alam2*alam2))
                       / (alam - alam2);
            if (a == 0.0) {
                tmplam = -slope / (2.0 * b);
            } else {
                double disc = b*b - 3.0*a*slope;
                if (disc < 0.0)        tmplam = 0.5 * alam;
                else if (b <= 0.0)     tmplam = (-b + sqrt(disc)) / (3.0 * a);
                else                    tmplam = -slope / (b + sqrt(disc));
            }
            if (tmplam > 0.5 * alam) tmplam = 0.5 * alam;
        }
        alam2 = alam;
        f2 = *fnew;
        alam = fmax(tmplam, 0.1 * alam);
    }
}

int recount_bfgs(double *x, int n,
                 recount_objgrad_fn objgrad, void *user,
                 double gtol, int max_iters,
                 double *final_obj, int *iters_out, int *status_out)
{
    /* Allocate scratch. */
    double *g           = (double *)malloc(n * sizeof(double));
    double *dg          = (double *)malloc(n * sizeof(double));
    double *xnew        = (double *)malloc(n * sizeof(double));
    double *p           = (double *)malloc(n * sizeof(double));
    double *hdg         = (double *)malloc(n * sizeof(double));
    double *xi          = (double *)malloc(n * sizeof(double));
    double *grad_scratch= (double *)malloc(n * sizeof(double));
    double *xbest       = (double *)malloc(n * sizeof(double));
    /* Inverse-Hessian dense n×n (matches Java's hessin[][]). */
    double *hessin      = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!g || !dg || !xnew || !p || !hdg || !xi || !grad_scratch || !xbest || !hessin) {
        free(g); free(dg); free(xnew); free(p); free(hdg); free(xi);
        free(grad_scratch); free(xbest); free(hessin);
        *status_out = RECOUNT_BFGS_ALLOC_FAIL;
        *iters_out = 0;
        *final_obj = 0.0;
        return -1;
    }

    int i, j, its;
    double fp = objgrad(x, n, g, user);
    double fbest = fp;
    memcpy(xbest, x, n * sizeof(double));

    /* hessin = I, xi = -g. */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) hessin[(size_t)i * n + j] = 0.0;
        hessin[(size_t)i * n + i] = 1.0;
        xi[i] = -g[i];
    }
    double sumx = 0.0;
    for (i = 0; i < n; i++) sumx += x[i] * x[i];
    /* Java: stpmax = DFP_STPMX * max(sqrt(sum), n)
     * Plus the csuros_dfpmin.py-style absolute cap to prevent the very
     * first full-Newton step from blowing past the rate-clip boundary
     * into the L(0)→1 degenerate region. */
    double stpmax = DFP_STPMX * fmax(sqrt(sumx), (double)n);
    if (stpmax > STPMAX_ABS_CAP) stpmax = STPMAX_ABS_CAP;

    int final_status = RECOUNT_BFGS_MAX_ITERS;
    for (its = 1; its <= max_iters; its++) {
        memcpy(dg, g, n * sizeof(double));     /* save old grad */
        memcpy(p,  xi, n * sizeof(double));    /* copy direction (lnsrch may scale) */

        double fnew;
        int ok = lnsrch(x, fp, g, p, stpmax, n, objgrad, user,
                        xnew, &fnew, grad_scratch);
        if (!ok) {
            final_status = RECOUNT_BFGS_LNSRCH_FAIL;
            break;
        }
        fp = fnew;
        /* Track best-seen iterate (Java does this too). */
        if (fp < fbest) {
            fbest = fp;
            memcpy(xbest, xnew, n * sizeof(double));
        }
        /* xi = step, x ← xnew */
        for (i = 0; i < n; i++) {
            xi[i] = xnew[i] - x[i];
            x[i]  = xnew[i];
        }

        /* Convergence on |Δx| (relative). */
        double test = 0.0;
        for (i = 0; i < n; i++) {
            double tmp = fabs(xi[i]) / fmax(fabs(x[i]), 1.0);
            if (tmp > test) test = tmp;
        }
        if (test < DFP_TOLX) {
            final_status = RECOUNT_BFGS_OK_DX;
            break;
        }

        /* New gradient (lnsrch wrote it into grad_scratch on each f-eval, so the
         * final value there is grad(xnew)). */
        memcpy(g, grad_scratch, n * sizeof(double));

        /* Convergence on |g|. */
        test = 0.0;
        for (i = 0; i < n; i++) {
            double a = fabs(g[i]);
            if (a > test) test = a;
        }
        if (test < gtol) {
            final_status = RECOUNT_BFGS_OK_GRAD;
            break;
        }

        /* BFGS update of inverse Hessian. */
        for (i = 0; i < n; i++) dg[i] = g[i] - dg[i];   /* gradient diff */
        for (i = 0; i < n; i++) {
            double s = 0.0;
            for (j = 0; j < n; j++) s += hessin[(size_t)i * n + j] * dg[j];
            hdg[i] = s;
        }
        double fac = 0.0, fae = 0.0, sumdg = 0.0, sumxi = 0.0;
        for (i = 0; i < n; i++) {
            fac   += dg[i] * xi[i];
            fae   += dg[i] * hdg[i];
            sumdg += dg[i] * dg[i];
            sumxi += xi[i] * xi[i];
        }
        if (fac > sqrt(DFP_EPS * sumdg * sumxi)) {
            double inv_fac = 1.0 / fac;
            double inv_fae = 1.0 / fae;
            /* dg becomes the BFGS auxiliary vector */
            for (i = 0; i < n; i++)
                dg[i] = inv_fac * xi[i] - inv_fae * hdg[i];
            /* Symmetric rank-2 update — exploit symmetry. */
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    double upd = inv_fac * xi[i] * xi[j]
                               - inv_fae * hdg[i] * hdg[j]
                               + fae * dg[i] * dg[j];
                    hessin[(size_t)i * n + j] += upd;
                    hessin[(size_t)j * n + i]  = hessin[(size_t)i * n + j];
                }
            }
        }
        /* New direction: xi = -hessin · g */
        for (i = 0; i < n; i++) {
            double s = 0.0;
            for (j = 0; j < n; j++) s += hessin[(size_t)i * n + j] * g[j];
            xi[i] = -s;
        }
    }

    /* Return best-seen x. */
    memcpy(x, xbest, n * sizeof(double));
    *final_obj  = fbest;
    *iters_out  = (its > max_iters) ? max_iters : its;
    *status_out = final_status;

    free(g); free(dg); free(xnew); free(p); free(hdg); free(xi);
    free(grad_scratch); free(xbest); free(hessin);
    return 0;
}
