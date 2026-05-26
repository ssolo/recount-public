/* Math primitives — most are inlined via recount_internal.h.
 * This file holds the out-of-line / Accelerate-batched variants.
 */
#include "recount_native.h"
#include "recount_internal.h"
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

/* SIMD logsumexp via Accelerate vForce.
 *
 * Replaces the scalar inner exp() loop with a batched vvexp() call. The
 * vDSP max-reduction picks up the same NEON unit. Worth it for
 * n >= RECOUNT_VVEXP_THRESHOLD (= 8 from microbench: n=64 142→94 ns,
 * n=256 573→397 ns, n=1024 2291→1551 ns; ~1.3-1.5× speedup).
 *
 * Numerical equivalence to the scalar path is exact (vvexp returns the
 * same per-element result as libm exp at fp64; the max-reduction is
 * order-independent because we extract the unique max).
 *
 * Stack-allocated diffs buffer: caller enforces n <= RECOUNT_VVEXP_STACK_MAX
 * before dispatching to us (the inline wrapper in recount_internal.h
 * falls back to scalar otherwise). */
double recount_logsumexp_vforce(const double *xs, int n) {
#ifdef __APPLE__
    double m;
    vDSP_maxvD(xs, 1, &m, (vDSP_Length)n);
    if (m == NEG_INF) return NEG_INF;
    double diffs[RECOUNT_VVEXP_STACK_MAX];
    double neg_m = -m;
    vDSP_vsaddD(xs, 1, &neg_m, diffs, 1, (vDSP_Length)n);
    int nn = n;
    vvexp(diffs, diffs, &nn);
    double s;
    vDSP_sveD(diffs, 1, &s, (vDSP_Length)n);
    return m + log(s);
#else
    double m = xs[0];
    for (int i = 1; i < n; i++) if (xs[i] > m) m = xs[i];
    if (m == NEG_INF) return NEG_INF;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(xs[i] - m);
    return m + log(s);
#endif
}

/* Out-of-line versions so they can be unit-tested via dlsym. */
double recount_logaddexp_extern(double a, double b) {
    return recount_logaddexp(a, b);
}

double recount_logsumexp_extern(const double *xs, int n) {
    return recount_logsumexp(xs, n);
}

double recount_safelog_pair_extern(double x, double xc) {
    return recount_safelog_pair(x, xc);
}
