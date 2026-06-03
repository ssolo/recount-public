/* recount_brownian_prior.c — native O(N) computation of the tree-Brownian
 * autocorrelated log-rate prior and its gradient.
 *
 * Spec: see docs/brownian_prior.tex and validation/_shared.py
 * `_brownian_prior_and_grad` (the Python reference). This C implementation
 * is bit-for-bit equivalent to that Python function under
 *   - time-uniform variance (no per-edge t_v weighting; v1 of the plan)
 *   - subcritical = True (dup in logit space; same packing as rates_to_x)
 *
 * Single-call O(N) loop over edges. No threading needed (N ≤ ~10⁴ on any
 * realistic phylogeny → microseconds even in serial).
 *
 * Packing layout (matches validation/_shared.py:rates_to_x):
 *     x[0 .. N-1]           : log_gain at every node (root included)
 *     x[N .. 2*N-2]         : dup block (logit space; one entry per non-root)
 *     x[2*N-1 .. 3*N-3]     : log_length (one entry per non-root)
 * Non-root node v has dup/length position p = (v < root) ? v : v - 1.
 */
#include "recount_native.h"
#include <stddef.h>
#include <string.h>


/* Compute log_prior + gradient. Output `out_grad` must be size 3*N - 2
 * doubles; it is ZERO-INITIALISED inside this function. Returns RECOUNT_OK.
 */
int recount_brownian_prior_and_grad(
    int32_t N,
    int32_t root,
    const int32_t *parent,          /* [N], parent[root] == -1 */
    const double *x,                /* [3*N - 2] */
    double sigma_brownian_gain,
    double sigma_brownian_dup,
    double sigma_brownian_length,
    double mu_root_gain,
    double mu_root_dup,
    double mu_root_length,
    double sigma_root,
    double *out_log_prior,
    double *out_grad)               /* [3*N - 2] */
{
    if (N < 2 || parent == NULL || x == NULL || out_log_prior == NULL || out_grad == NULL) {
        return RECOUNT_EINVAL;
    }
    if (sigma_brownian_gain <= 0.0 || sigma_brownian_dup <= 0.0 ||
        sigma_brownian_length <= 0.0 || sigma_root <= 0.0) {
        return RECOUNT_EINVAL;
    }

    const size_t total = (size_t)(3 * N - 2);
    memset(out_grad, 0, total * sizeof(double));

    const double *x_gain = x;                 /* [N]   */
    const double *x_dup  = x + N;             /* [N-1] */
    const double *x_len  = x + (size_t)(2 * N - 1);  /* [N-1] */
    double *g_gain = out_grad;
    double *g_dup  = out_grad + N;
    double *g_len  = out_grad + (size_t)(2 * N - 1);

    const double inv_s2_gain = 1.0 / (sigma_brownian_gain * sigma_brownian_gain);
    const double inv_s2_dup  = 1.0 / (sigma_brownian_dup  * sigma_brownian_dup);
    const double inv_s2_len  = 1.0 / (sigma_brownian_length * sigma_brownian_length);

    double log_prior = 0.0;

    /* --- edge terms (Brownian increments) --- */
    for (int32_t v = 0; v < N; v++) {
        if (v == root) continue;
        const int32_t pa = parent[v];
        const int32_t pos_v  = (v  < root) ? v  : v  - 1;
        const int32_t pos_pa = (pa < root) ? pa : pa - 1;

        /* gain: root is in x_gain, so parent always has a position */
        {
            const double diff = x_gain[v] - x_gain[pa];
            log_prior -= 0.5 * inv_s2_gain * diff * diff;
            const double scaled = inv_s2_gain * diff;
            g_gain[v]  -= scaled;
            g_gain[pa] += scaled;
        }

        /* dup: parent may be root → use mu_root_dup as the parent value */
        {
            const double parent_val = (pa == root) ? mu_root_dup : x_dup[pos_pa];
            const double diff = x_dup[pos_v] - parent_val;
            log_prior -= 0.5 * inv_s2_dup * diff * diff;
            const double scaled = inv_s2_dup * diff;
            g_dup[pos_v] -= scaled;
            if (pa != root) g_dup[pos_pa] += scaled;
        }

        /* length: same convention as dup */
        {
            const double parent_val = (pa == root) ? mu_root_length : x_len[pos_pa];
            const double diff = x_len[pos_v] - parent_val;
            log_prior -= 0.5 * inv_s2_len * diff * diff;
            const double scaled = inv_s2_len * diff;
            g_len[pos_v] -= scaled;
            if (pa != root) g_len[pos_pa] += scaled;
        }
    }

    /* --- root anchor: independent N(mu_root_gain, sigma_root²) on gain root --- */
    {
        const double inv_s2_root = 1.0 / (sigma_root * sigma_root);
        const double dev = x_gain[root] - mu_root_gain;
        log_prior -= 0.5 * inv_s2_root * dev * dev;
        g_gain[root] -= inv_s2_root * dev;
    }

    *out_log_prior = log_prior;
    return RECOUNT_OK;
}
