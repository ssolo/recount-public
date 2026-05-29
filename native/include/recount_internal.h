/* recount_internal.h — shared inlines + constants for the native backend.
 * Not part of the public API.
 */
#ifndef RECOUNT_INTERNAL_H
#define RECOUNT_INTERNAL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef NEG_INF
#define NEG_INF (-INFINITY)
#endif

/* logaddexp(a, b) — branch-free except for the -inf guards. */
static inline double recount_logaddexp(double a, double b) {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    double m = a > b ? a : b;
    double d = a > b ? (b - a) : (a - b);  /* always <= 0 */
    return m + log1p(exp(d));
}

/* Numerically stable log over a (possibly tiny) probability. */
static inline double recount_safe_log(double x) {
    return x > 0.0 ? log(x) : NEG_INF;
}

/* log(x) using whichever of log(x) or log1p(-x_complement) is more
 * accurate. x and x_complement should sum to ~1. */
static inline double recount_safelog_pair(double x, double x_complement) {
    if (x < x_complement) {
        return x > 0.0 ? log(x) : NEG_INF;
    } else {
        return x_complement < 1.0 ? log1p(-x_complement) : NEG_INF;
    }
}

/* logsumexp over an array of length n (handles all--inf rows).
 *
 * Hot path (called once per node per family in the inside/outside pipeline).
 * For n >= RECOUNT_VVEXP_THRESHOLD we route the inner exp() loop through
 * Apple Accelerate's vForce (vvexp) — 1.3-1.5× faster per call on M-series
 * (measured: n=64 142→94 ns, n=256 573→397 ns, n=1024 2291→1551 ns).
 * For tiny n the vForce setup overhead loses, so we keep the scalar path. */
#define RECOUNT_VVEXP_THRESHOLD 8
#define RECOUNT_VVEXP_STACK_MAX 2048  /* stack fast-path cap; wider gene families spill to the heap (recount_width_scratch) */

double recount_logsumexp_vforce(const double *xs, int n);

static inline double recount_logsumexp(const double *xs, int n) {
    if (n <= 0) return NEG_INF;
    /* The inside/outside antidiagonal gather hits n=1 on the (ell=0) end
     * of every node's combine, and n=2 on (ell=1). At n=1 the generic
     * path costs an exp+log pair to return xs[0]; at n=2 a logaddexp call
     * is identical (and uses log1p, more accurate near 0). */
    if (n == 1) return xs[0];
    if (n == 2) return recount_logaddexp(xs[0], xs[1]);
    if (n >= RECOUNT_VVEXP_THRESHOLD && n <= RECOUNT_VVEXP_STACK_MAX) {
        return recount_logsumexp_vforce(xs, n);
    }
    double m = xs[0];
    for (int i = 1; i < n; i++) {
        if (xs[i] > m) m = xs[i];
    }
    if (m == NEG_INF) return NEG_INF;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += exp(xs[i] - m);
    }
    return m + log(s);
}

/* Inside/outside kernel scratch is sized by the per-node DP width — a gene
 * family's total surviving copy count + 1.  RECOUNT_VVEXP_STACK_MAX covers
 * typical data, but COG / ortholog tables with very abundant families can
 * exceed it.  recount_width_scratch() returns the caller's stack buffer on
 * the fast path and a heap block for the rare wide families; pair every
 * call with recount_width_scratch_free(). */
static inline double *recount_width_scratch(double *stackbuf, int stack_cap,
                                            int need) {
    if (need <= stack_cap) return stackbuf;
    return (double *)malloc((size_t)need * sizeof(double));
}
static inline void recount_width_scratch_free(double *buf,
                                              const double *stackbuf) {
    if (buf != stackbuf) free(buf);
}

/* Log-factorial cache (one-shot built at recount_factln_init). */
typedef struct {
    int max_n;
    double *table;  /* table[i] = lgamma(i + 1) for i in [0, max_n] */
} recount_factln_t;

int recount_factln_init(recount_factln_t *f, int max_n);
void recount_factln_free(recount_factln_t *f);
double recount_factln_get(const recount_factln_t *f, int n);

/* Log-rising-factorial cache for Pólya κ:
 *   rfact[n] = lgamma(kappa + n) - lgamma(kappa) = sum_{i=0..n-1} log(kappa + i)
 */
typedef struct {
    int max_n;
    double kappa;
    double *table;  /* table[i] = log(rising factorial) for i in [0, max_n] */
} recount_rfactln_t;

int recount_rfactln_init(recount_rfactln_t *rf, double kappa, int max_n);
void recount_rfactln_free(recount_rfactln_t *rf);
double recount_rfactln_get(const recount_rfactln_t *rf, int n);

#endif /* RECOUNT_INTERNAL_H */
