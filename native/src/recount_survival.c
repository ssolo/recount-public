/* Port of recount.gld.compute_survival_params (gld.py:85-150).
 * Bottom-up pass: raw rates (μ, λ, γ, t) per node -> survival
 * parameterization (p̃, q̃, r̃/κ̃, ε), with stable accumulators for ε.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "recount_native.h"
#include "recount_internal.h"

extern void recount_rate_to_pq(double mu, double lam, double t,
                               double *p, double *p_c, double *q, double *q_c);

int recount_survival_alloc(recount_survival_t *sp, int32_t num_nodes) {
    sp->num_nodes = num_nodes;
    /* one big alloc for all the [num_nodes] arrays */
    size_t n = (size_t)num_nodes;
    size_t n_arrays = 14;  /* p..log_eps_c */
    sp->p = (double *)calloc(n_arrays * n, sizeof(double));
    if (!sp->p) return RECOUNT_ENOMEM;
    sp->p_c       = sp->p + 1 * n;
    sp->q         = sp->p + 2 * n;
    sp->q_c       = sp->p + 3 * n;
    sp->gain      = sp->p + 4 * n;
    sp->eps       = sp->p + 5 * n;
    sp->eps_c     = sp->p + 6 * n;
    sp->log_p     = sp->p + 7 * n;
    sp->log_p_c   = sp->p + 8 * n;
    sp->log_q     = sp->p + 9 * n;
    sp->log_q_c   = sp->p + 10 * n;
    sp->log_gain  = sp->p + 11 * n;
    sp->log_eps   = sp->p + 12 * n;
    sp->log_eps_c = sp->p + 13 * n;
    sp->is_polya  = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!sp->is_polya) { free(sp->p); sp->p = NULL; return RECOUNT_ENOMEM; }
    return RECOUNT_OK;
}

void recount_survival_free(recount_survival_t *sp) {
    free(sp->p);
    free(sp->is_polya);
    memset(sp, 0, sizeof(*sp));
}

int recount_compute_survival_params(
    const recount_tree_t *tree,
    const double *gain, const double *loss,
    const double *dup,  const double *length,
    recount_survival_t *sp)
{
    int32_t n = tree->num_nodes;
    if (sp->num_nodes != n) return RECOUNT_EINVAL;

    /* per-edge raw (p, p_c, q, q_c) */
    double *p_raw   = (double *)calloc((size_t)n * 4, sizeof(double));
    if (!p_raw) return RECOUNT_ENOMEM;
    double *p_raw_c = p_raw + n;
    double *q_raw   = p_raw + 2 * n;
    double *q_raw_c = p_raw + 3 * n;
    for (int32_t v = 0; v < n; v++) {
        recount_rate_to_pq(loss[v], dup[v], length[v],
                           &p_raw[v], &p_raw_c[v], &q_raw[v], &q_raw_c[v]);
    }

    /* Bottom-up: ε[v], (p̃, q̃, r̃/κ̃) */
    for (int32_t v = 0; v < n; v++) {
        double e, e_c;
        if (tree->is_leaf[v]) {
            e = 0.0; e_c = 1.0;
        } else {
            /* ε[v] = ∏_c p̃_c via parallel (1-ε) accumulator (stable when ε→1) */
            double e0 = 1.0, e1 = 0.0;
            int32_t cs = tree->first_child[v], ce = tree->first_child[v + 1];
            for (int32_t ci = cs; ci < ce; ci++) {
                int32_t c = tree->child_list[ci];
                double pc   = sp->p[c];
                double pc_c = sp->p_c[c];
                e1 = e1 + e0 * pc_c;
                e0 = e0 * pc;
            }
            e   = e0;
            e_c = (e0 == 1.0) ? e1 : (1.0 - e0);
        }
        sp->eps[v]   = e;
        sp->eps_c[v] = e_c;

        double p  = p_raw[v],   p_c = p_raw_c[v];
        double q  = q_raw[v],   q_c = q_raw_c[v];
        sp->is_polya[v] = (q > 0.0) ? 1 : 0;

        /* r̃ (Poisson) or κ̃ = κ (Pólya): gain ε-corrected only for Poisson */
        sp->gain[v] = sp->is_polya[v] ? gain[v] : gain[v] * e_c;

        /* a = (1-q)·ε + (1-ε) = 1 - q·ε  (stable, no subtraction) */
        double a = q_c * e + e_c;
        sp->p[v]   = (p * e_c + e * q_c) / a;
        sp->q[v]   = q * e_c / a;
        double b = e_c / a;
        sp->p_c[v] = p_c * b;
        sp->q_c[v] = q_c / a;
    }

    /* Log-space companions */
    for (int32_t v = 0; v < n; v++) {
        sp->log_p[v]     = recount_safelog_pair(sp->p[v],   sp->p_c[v]);
        sp->log_p_c[v]   = recount_safelog_pair(sp->p_c[v], sp->p[v]);
        sp->log_q[v]     = recount_safelog_pair(sp->q[v],   sp->q_c[v]);
        sp->log_q_c[v]   = recount_safelog_pair(sp->q_c[v], sp->q[v]);
        sp->log_gain[v]  = sp->gain[v] > 0.0 ? log(sp->gain[v]) : NEG_INF;
        sp->log_eps[v]   = recount_safelog_pair(sp->eps[v],   sp->eps_c[v]);
        sp->log_eps_c[v] = recount_safelog_pair(sp->eps_c[v], sp->eps[v]);
    }

    free(p_raw);
    return RECOUNT_OK;
}
