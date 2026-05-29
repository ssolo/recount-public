/* Port of recount.gld._compute_edge (gld.py:184-238).
 * K[s] = log Σ_t C[s+t] · gain_pmf(t | s)
 *   - Poisson (q=0):     gain_pmf(t) = exp(-r) · r^t / t!
 *   - Pólya (q>0):       gain_pmf(t | s) = NegBin(t; κ+s, 1-q)
 */
#include <math.h>
#include "recount_native.h"
#include "recount_internal.h"

/* terms_scratch must hold up to C_len doubles (caller-sized).
 *
 * The per-node precompute tables (tterm, Cre) are sized by C_len — the DP
 * width, equal to a family's total surviving copy count + 1.  That is
 * usually <= RECOUNT_VVEXP_STACK_MAX, but abundant gene families exceed it,
 * so the tables fall back to the heap (recount_width_scratch). */
void recount_compute_edge(
    const double *C, int C_len, double *K, int K_len,
    double log_q, double log_q_c, double log_gain, int is_polya,
    const recount_factln_t *fact, const recount_rfactln_t *rfact,
    double *terms_scratch)
{
    for (int s = 0; s < K_len; s++) K[s] = NEG_INF;
    if (C_len == 0 || K_len == 0) return;

    if (!is_polya) {
        if (log_gain == NEG_INF) {
            /* No gain: K[s] = C[s] */
            int n = K_len < C_len ? K_len : C_len;
            for (int s = 0; s < n; s++) K[s] = C[s];
            return;
        }
        double r = exp(log_gain);
        /* Precompute tterm[t] = t·log_gain - factln(t) - r once per node.
         * Folds the (-r) constant and the (t==0)?0.0 guard via direct
         * assignment at t=0 (since factln(0)=0). */
        double tterm_stack[RECOUNT_VVEXP_STACK_MAX];
        double *tterm = recount_width_scratch(
            tterm_stack, RECOUNT_VVEXP_STACK_MAX, C_len);
        if (!tterm) return;   /* OOM — K already NEG_INF-initialised */
        int max_t = C_len;
        tterm[0] = -r;
        for (int t = 1; t < max_t; t++) {
            tterm[t] = (double)t * log_gain - recount_factln_get(fact, t) - r;
        }
        for (int s = 0; s < K_len; s++) {
            int t = 0, ell = s, cnt = 0;
            while (ell < C_len) {
                terms_scratch[cnt++] = C[ell] + tterm[t];
                t++; ell++;
            }
            K[s] = recount_logsumexp(terms_scratch, cnt);
        }
        recount_width_scratch_free(tterm, tterm_stack);
        return;
    }

    /* Pólya */
    if (log_gain == NEG_INF) {
        int n = K_len < C_len ? K_len : C_len;
        for (int s = 0; s < n; s++) K[s] = C[s];
        return;
    }
    /* loglog1_q = log(-log_q_c) (works because log_q_c < 0) */
    double loglog1_q = (log_q_c < 0.0) ? log(-log_q_c) : log_q;
    double kappa_log1_q = exp(log_gain + loglog1_q);   /* = κ · |log(1-q)| */

    /* Per-node precomputes — invariant across the outer s loop.
     *   tterm[t]    = t·log_q - factln(t)        (folds (t==0)?0.0 guard)
     *   Cre[ell]    = C[ell] + rfact(ell)        (rfact contribution to binom)
     * Then inner: terms[cnt++] = Cre[ell] + tterm[t] + (ks_log1_q - rfact(s)).
     * One heap block backs both tables when C_len exceeds the stack cap. */
    double edge_stack[2 * RECOUNT_VVEXP_STACK_MAX];
    double *edge_buf = recount_width_scratch(
        edge_stack, 2 * RECOUNT_VVEXP_STACK_MAX, 2 * C_len);
    if (!edge_buf) return;   /* OOM — K already NEG_INF-initialised */
    double *tterm = edge_buf;
    double *Cre = edge_buf + C_len;
    tterm[0] = 0.0;
    for (int t = 1; t < C_len; t++) {
        tterm[t] = (double)t * log_q - recount_factln_get(fact, t);
    }
    for (int ell = 0; ell < C_len; ell++) {
        Cre[ell] = C[ell] + recount_rfactln_get(rfact, ell);
    }

    for (int s = 0; s < K_len; s++) {
        /* (s + κ) · log(1-q): guard 0·(-∞) when s=0 and q→1. κ > 0 in
         * this branch (Pólya with log_gain finite). */
        double ks_log1_q;
        if (log_q_c == NEG_INF) {
            ks_log1_q = -kappa_log1_q;  /* s·(-∞) treated as 0 when s=0 */
            if (s != 0) ks_log1_q = NEG_INF;  /* s>0 dominates */
        } else {
            ks_log1_q = (double)s * log_q_c - kappa_log1_q;
        }
        double s_invariant = ks_log1_q - recount_rfactln_get(rfact, s);
        int t = 0, ell = s, cnt = 0;
        while (ell < C_len) {
            terms_scratch[cnt++] = Cre[ell] + tterm[t] + s_invariant;
            t++; ell++;
        }
        K[s] = recount_logsumexp(terms_scratch, cnt);
    }
    recount_width_scratch_free(edge_buf, edge_stack);
}
