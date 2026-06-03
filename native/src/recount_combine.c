/* Port of recount.gld._compute_sibling (gld.py:241-310).
 * The destructive O(W²) sibling combine — the hot inner loop.
 *
 * Combines accumulator C (already-processed siblings' inside) with the
 * junior child's edge LL K, producing the parent's inside.
 *
 * Mutates `C` in place (caller provides a buffer to mutate).
 * Output written to `C2[0..combined]` where combined = (K_len-1) + (C_len-1).
 */
#include <math.h>
#include "recount_native.h"
#include "recount_internal.h"

void recount_compute_sibling(
    double *C, int C_len,          /* in/out: mutated by destructive sweep */
    const double *K, int K_len,
    double p_junior, double p_junior_c, double eps_sib,
    const recount_factln_t *fact,
    double *C2, int *out_combined, /* output buffer (size >= K_len + C_len - 1) */
    double *terms_scratch)         /* size >= K_len */
{
    /* Boundary cases mirroring gld.py:266-271 */
    if (eps_sib == 1.0) {
        /* siblings extinct: return K_junior (copy) */
        int n = K_len > 0 ? K_len : C_len;
        if (K_len > 0) {
            for (int i = 0; i < K_len; i++) C2[i] = K[i];
        } else {
            for (int i = 0; i < C_len; i++) C2[i] = C[i];
        }
        *out_combined = n - 1;
        return;
    }
    if (C_len == 0) {
        for (int i = 0; i < K_len; i++) C2[i] = K[i];
        *out_combined = K_len - 1;
        return;
    }
    if (K_len == 0) {
        for (int i = 0; i < C_len; i++) C2[i] = C[i];
        *out_combined = C_len - 1;
        return;
    }

    int combined = (K_len - 1) + (C_len - 1);
    *out_combined = combined;

    double log_e   = log(eps_sib);
    double log_e_c = log1p(-eps_sib);
    double log_a   = log1p(-p_junior * eps_sib);     /* log(1 − p̃·ε) */
    double logp1   = log(p_junior_c) - log_a;
    double logp2   = log(p_junior) + log_e_c - log_a;

    /* Precompute per-axis lookup tables so the antidiagonal gather is
     * branchless and uses one factorial-fold lookup per (s, t) instead of
     * two factln calls + two cmov-style zero-guards. Both tables fold the
     * (·==0 ? 0 : · * log_·) guard via direct assignment at index 0
     * (since factln(0) = 0).
     *
     *   sterm[s] = s * logp1 - factln(s)
     *   tterm[t] = t * logp2 - factln(t)
     *
     * Then gather inner: K[s] + C[t] + log_ellfact + sterm[s] + tterm[t].
     * sterm spans K_len and tterm spans C_len; one heap block backs both
     * when K_len + C_len exceeds the stack cap (very abundant families). */
    double sib_stack[2 * RECOUNT_VVEXP_STACK_MAX];
    double *sib_buf = recount_width_scratch(
        sib_stack, 2 * RECOUNT_VVEXP_STACK_MAX, K_len + C_len);
    if (!sib_buf) {   /* OOM — emit a -inf result rather than crash */
        for (int i = 0; i <= combined; i++) C2[i] = NEG_INF;
        return;
    }
    double *sterm = sib_buf;
    double *tterm = sib_buf + K_len;
    sterm[0] = 0.0;
    for (int s = 1; s < K_len; s++) {
        sterm[s] = (double)s * logp1 - recount_factln_get(fact, s);
    }
    tterm[0] = 0.0;
    for (int t = 1; t < C_len; t++) {
        tterm[t] = (double)t * logp2 - recount_factln_get(fact, t);
    }

    for (int ell = 0; ell <= combined; ell++) {
        /* Destructive update of C[t] */
        int t = (ell < C_len) ? ell : C_len;
        double x = (t < C_len) ? C[t] : NEG_INF;
        int s = ell - t;
        while (t > 0 && s < K_len - 1) {
            t--; s++;
            x = x + log_e_c;
            double y = C[t] + log_e;
            x = recount_logaddexp(x, y);
            C[t] = x;
        }

        /* Gather: walk (s, t) on the s+t=ell diagonal */
        double log_ellfact = recount_factln_get(fact, ell);
        int cnt = 0;
        while (s >= 0 && t < C_len) {
            terms_scratch[cnt++] = K[s] + C[t] + log_ellfact + sterm[s] + tterm[t];
            s--; t++;
        }
        C2[ell] = recount_logsumexp(terms_scratch, cnt);
    }
    recount_width_scratch_free(sib_buf, sib_stack);
}
