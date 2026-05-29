/*
 * recount_bfgs.h — native C BFGS optimizer matching Csuros' Java dfpmin
 *
 * Bit-faithful port of count/src/count/matek/FunctionMinimization.java
 * dfpmin (BFGS variant of Davidon-Fletcher-Powell, with NumRecipes lnsrch).
 *
 * The optimizer accepts an opaque user-supplied callback that computes the
 * objective value f(x) and writes the gradient into a caller-allocated array.
 * This decouples the BFGS bookkeeping (pure C, single-threaded but cheap) from
 * the LL+gradient evaluation (which is native libdispatch-parallelised, the
 * heavy work).
 *
 * Constants match the Java verbatim:
 *   EPS         = 1/2^52        ≈ 2.22e-16
 *   DFP_TOLX    = 4 * EPS       ≈ 8.88e-16     (convergence on |Δx|)
 *   LNSRCH_TOLX = 4 * EPS                       (line-search minimum step)
 *   LNSRCH_ALF  = 1e-4                          (Armijo sufficient-decrease)
 *   DFP_STPMX   = 100                           (max step length scaling)
 */

#ifndef RECOUNT_BFGS_H
#define RECOUNT_BFGS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Callback computing f(x) and its gradient.
 *   x[n]   : parameter vector (read-only)
 *   n      : dimension
 *   grad[n]: caller-allocated, callee writes the gradient ∂f/∂x
 *   user   : opaque user pointer
 * Returns f(x). The callee MUST NOT mutate x.
 */
typedef double (*recount_objgrad_fn)(const double *x, int n,
                                     double *grad, void *user);

/*
 * Status codes returned via *status_out.
 */
enum recount_bfgs_status {
    RECOUNT_BFGS_OK_GRAD    = 0,  /* max(|g_i|) < gtol */
    RECOUNT_BFGS_OK_DX      = 1,  /* max(|Δx_i|/max(|x_i|,1)) < DFP_TOLX */
    RECOUNT_BFGS_MAX_ITERS  = 2,  /* hit max_iters without convergence */
    RECOUNT_BFGS_LNSRCH_FAIL= 3,  /* line search failed (alam < alamin) */
    RECOUNT_BFGS_ALLOC_FAIL = 4,  /* malloc failed */
};

/*
 * BFGS optimizer.
 *
 * INPUTS:
 *   x[n]        : starting point; overwritten with best minimiser found
 *   n           : problem dimension
 *   objgrad     : callback to compute f, grad
 *   user        : passed to objgrad each call
 *   gtol        : convergence on max(|g_i|) — matches Java dfpmin's gtol arg
 *   max_iters   : max BFGS iterations (Java default = 200 per call,
 *                 typical total budget = 1024 across warm restarts)
 *
 * OUTPUTS:
 *   *final_obj  : f(x) at returned x
 *   *iters_out  : iterations completed
 *   *status_out : enum value (see above)
 *
 * RETURN: 0 on success (including all converged/max-iters cases),
 *         nonzero only on malloc failure.
 *
 * Tracking the best-seen f (Java keeps a `best` snapshot too): on return,
 * x[] holds the iterate with the smallest f observed during the run.
 */
int recount_bfgs(double *x, int n,
                 recount_objgrad_fn objgrad, void *user,
                 double gtol, int max_iters,
                 double *final_obj, int *iters_out, int *status_out);

#ifdef __cplusplus
}
#endif

#endif /* RECOUNT_BFGS_H */
