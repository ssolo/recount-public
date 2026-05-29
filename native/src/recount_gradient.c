/* Analytical gradient — port of recount.gld.gradient_survival (gld.py:653-767).
 *
 * Per-family workflow: forward → outside → posterior expectations of (ξ̃_v, η̃_v).
 * Across families: accumulate sums N_v[v], S_v[v] and tail sums N_tail[v][i],
 * S_tail[v][i]. After all families: apply Csurös 2021 Corollary 9 closed forms.
 *
 * Note: returns the *survival-parameterization* gradient ∂LL_raw/∂(p̃, q̃, r̃/κ̃)
 * indexed as 3*v + {GAIN=0, LOSS=1, DUP=2}. The corrected gradient
 * (min_copies=1) adds the closed-form ∂(F·log(1-L(0)))/∂(...) term, handled
 * inside the GAIN/DUP/LOSS formulas via the one_minus_L0 factor.
 *
 * Threading: libdispatch dispatches families across workers. Each worker has
 * its own scratchpad + thread-local accumulators; after the dispatch, a
 * serial reduction sums thread-locals into the final arrays.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include "recount_native.h"
#include "recount_internal.h"
#include "recount_parallel.h"

#define RECOUNT_GAIN 0
#define RECOUNT_LOSS 1
#define RECOUNT_DUP  2

extern int recount_forward_family_with_scratch(
    const recount_tree_t *tree, const recount_survival_t *sp,
    const int32_t *profile,
    const recount_factln_t *fact, const recount_rfactln_t * const *rfacts,
    int32_t *widths, double **C_slots, double **K_slots,
    double *bigpool, size_t bigpool_n, double *out_ll);

extern int recount_compute_outside(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int32_t *widths,
    int *K_lens,
    double **C_slots, double **K_slots,
    double **B_slots, double **J_slots,
    const recount_factln_t *fact,
    const recount_rfactln_t * const *rfacts,
    double *scratch_work, int max_K_len);

/* Posterior marginal stats: given log_outside[L] + log_inside[L] - LL,
 * compute the posterior mean E[X] and tail P(X > i) up to max_tail entries.
 * out_tail must be zeroed by caller and have length >= max_tail. */
static void posterior_marginal_stats(
    const double *log_outside, int L_out,
    const double *log_inside, int L_in,
    double LL,
    double *out_mean,
    double *out_tail, int max_tail)
{
    int L = L_out < L_in ? L_out : L_in;
    *out_mean = 0.0;
    if (L <= 0) return;

    /* Compute posterior p[i] = exp(out[i] + in[i] - LL) for i in [0, L) */
    /* For numerical stability we just use exp(diff) since LL is the true norm
     * constant; finite checks handle the -inf entries. */
    double mean = 0.0;
    /* Accumulate post[i] in tail-suffix order so tail_post[i] = Σ_{j>i} p[j] */
    /* tail_post[i] for i in [0, L-1]; tail_post[L-1] = 0 by definition
     * (P(X > L-1) given L-padded support).
     * We sweep right-to-left: cum = p[L-1] + p[L-2] + ...
     * tail_post[i] = cum at one position right of i, i.e. tail_post[i] = sum_{j=i+1..L-1} p[j]
     */
    /* Allocate temporary in caller's tail buffer's first L slots — we'll
     * shift to the right place after. Actually it's easier to compute
     * directly: */

    /* First pass: compute post[i] = exp(out + in - LL).
     * For L >= RECOUNT_VVEXP_THRESHOLD we batch via Accelerate vForce;
     * otherwise stay scalar (vForce setup cost > work for tiny L).
     * For Williams' heavy tail (L up to ~700) this is the dominant per-node
     * cost in the gradient/branch-stats pipeline. */
    double *post = out_tail;  /* max_tail >= L always */
#ifdef __APPLE__
    if (L >= RECOUNT_VVEXP_THRESHOLD) {
        /* post[i] = log_outside[i] + log_inside[i] */
        vDSP_vaddD(log_outside, 1, log_inside, 1, post, 1, (vDSP_Length)L);
        /* post[i] -= LL */
        double neg_LL = -LL;
        vDSP_vsaddD(post, 1, &neg_LL, post, 1, (vDSP_Length)L);
        /* post[i] = exp(post[i]) — NaN propagates from -inf+-inf-LL=NaN */
        int LL_int = L;
        vvexp(post, post, &LL_int);
        /* Sanitize non-finite entries to 0 + accumulate Σ i·post[i]. The
         * scalar pass here is short and avoids a separate vDSP_dotpr +
         * ramp allocation that'd lose the NaN/inf handling. */
        for (int i = 0; i < L; i++) {
            if (!isfinite(post[i])) post[i] = 0.0;
            mean += (double)i * post[i];
        }
    } else
#endif
    {
        for (int i = 0; i < L; i++) {
            double v = log_outside[i] + log_inside[i] - LL;
            post[i] = (isfinite(v)) ? exp(v) : 0.0;
            mean += (double)i * post[i];
        }
    }
    *out_mean = mean;

    /* Now compute tail_post[i] = sum_{j>i} post[j] in-place from the right. */
    double cum = 0.0;
    /* tail_post[L-1] = 0; we want out_tail[i] (for i < max_tail) */
    /* Walk right-to-left filling in tail_post over the post[] storage. */
    /* tail_post[i] = cum AFTER adding post[i+1..L-1] */
    /* Easiest: build tail_post values by overwriting post from right to left. */
    /* Save the last one first since we need its post[L-1] in cum. */
    /* Loop: for i from L-1 down to 0:
            new_tail = cum
            cum += post[i]
            tail_post[i] = new_tail
       But we're writing to the same buffer as post[]. Since we go right-to-left
       and we read post[i] before writing tail_post[i], it's safe. */
    for (int i = L - 1; i >= 0; i--) {
        double pi = post[i];
        post[i] = cum;        /* = tail_post[i] */
        cum += pi;
    }
    /* Now post[0..L-1] contains tail_post[0..L-1]. Zero beyond L. */
    for (int i = L; i < max_tail; i++) out_tail[i] = 0.0;
}

/* Compute L0 = empty profile log-likelihood (closed form). */
static double empty_log_likelihood(const recount_survival_t *sp, int root) {
    double LL = 0.0;
    for (int v = 0; v < sp->num_nodes; v++) {
        if (sp->is_polya[v]) {
            double log1_q = sp->log_q_c[v];
            if (log1_q == NEG_INF) return NEG_INF;
            double log_k = sp->log_gain[v];
            if (log_k == NEG_INF) continue;
            LL -= exp(log_k + log(-log1_q));
        } else {
            LL -= sp->gain[v];
        }
    }
    LL += sp->log_p[root];
    return LL;
}

/* Thread-local accumulator. */
typedef struct {
    double *N_v;          /* [num_nodes] */
    double *S_v;          /* [num_nodes] */
    double *N_tail;       /* [num_nodes * tail_size] row-major */
    double *S_tail;       /* [num_nodes * tail_size] row-major */
    int32_t *n_active;    /* [num_nodes] - # families with P(ξ_v > 0) > active_threshold */
    double *fp_v;         /* [num_nodes] - Σ_f (1 - P{ξ_v=0|Ξ_f}) (continuous families_present) */
    double active_threshold;
    /* Pool for one family's forward + outside scratch. */
    uint8_t *scratch;
    int32_t *widths;
    int *K_lens;
    double **C_slots;
    double **K_slots;
    double **B_slots;
    double **J_slots;
    double *bigpool;
    size_t bigpool_doubles;
    double *outside_work;     /* extra work buffer for outside pass */
    int max_K_len;
} grad_worker_t;

static int alloc_worker(grad_worker_t *w, int n, int max_w, int tail_size) {
    memset(w, 0, sizeof(*w));
    w->N_v = (double *)calloc((size_t)n, sizeof(double));
    w->S_v = (double *)calloc((size_t)n, sizeof(double));
    w->N_tail = (double *)calloc((size_t)n * (size_t)tail_size, sizeof(double));
    w->S_tail = (double *)calloc((size_t)n * (size_t)tail_size, sizeof(double));
    w->n_active = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    w->fp_v = (double *)calloc((size_t)n, sizeof(double));
    w->active_threshold = 0.5;
    if (!w->N_v || !w->S_v || !w->N_tail || !w->S_tail || !w->n_active || !w->fp_v)
        return RECOUNT_ENOMEM;

    /* Scratch: widths[N] + K_lens[N] + slots (C, K, B, J)[N] + bigpool */
    size_t header = (size_t)n * sizeof(int32_t)
                  + (size_t)n * sizeof(int)
                  + (size_t)n * 4 * sizeof(double *);
    header = (header + 15) & ~(size_t)15;

    /* Bigpool sizing: for each node, C[v], K[v], B[v], J[v] each up to (max_w + 2).
     * Plus combine scratch: terms (max_w+4) + C_work (2*max_w+4) + C2_buf (2*max_w+4)
     * Plus outside scratch: K_sib_buf + K_sib_work + terms_buf (~6 * max_K_len). */
    size_t per_node = (size_t)4 * (size_t)(max_w + 2);
    size_t global_extra = (size_t)(5 * (max_w + 4));
    size_t outside_extra = (size_t)(6 * (max_w + 4));
    size_t pool_doubles = per_node * (size_t)n + global_extra + outside_extra + 64;

    size_t scratch_n = header + pool_doubles * sizeof(double) + 1024;
    w->scratch = (uint8_t *)malloc(scratch_n);
    if (!w->scratch) return RECOUNT_ENOMEM;

    uint8_t *p = w->scratch;
    w->widths = (int32_t *)p; p += (size_t)n * sizeof(int32_t);
    w->K_lens = (int *)p;     p += (size_t)n * sizeof(int);
    w->C_slots = (double **)p; p += (size_t)n * sizeof(double *);
    w->K_slots = (double **)p; p += (size_t)n * sizeof(double *);
    w->B_slots = (double **)p; p += (size_t)n * sizeof(double *);
    w->J_slots = (double **)p; p += (size_t)n * sizeof(double *);
    size_t used = (size_t)(p - w->scratch);
    used = (used + 15) & ~(size_t)15;
    p = w->scratch + used;
    w->bigpool = (double *)p;
    w->bigpool_doubles = (scratch_n - used - outside_extra * sizeof(double)) / sizeof(double);
    /* outside_work is a separate region at the end */
    w->outside_work = w->bigpool + w->bigpool_doubles;
    w->max_K_len = max_w + 2;
    return RECOUNT_OK;
}

static void free_worker(grad_worker_t *w) {
    free(w->N_v); free(w->S_v);
    free(w->N_tail); free(w->S_tail);
    free(w->fp_v);
    free(w->n_active);
    free(w->scratch);
    memset(w, 0, sizeof(*w));
}

/* Build rfacts for all polya nodes — shared across threads (read-only). */
static int build_rfacts_shared(const recount_survival_t *sp, int max_n,
                               recount_rfactln_t ***out_rfacts) {
    int n = sp->num_nodes;
    recount_rfactln_t **rfacts = (recount_rfactln_t **)calloc((size_t)n, sizeof(*rfacts));
    if (!rfacts) return RECOUNT_ENOMEM;
    for (int v = 0; v < n; v++) {
        if (sp->is_polya[v] && sp->gain[v] > 0.0) {
            rfacts[v] = (recount_rfactln_t *)malloc(sizeof(recount_rfactln_t));
            if (!rfacts[v] || recount_rfactln_init(rfacts[v], sp->gain[v], max_n) != RECOUNT_OK) {
                for (int u = 0; u <= v; u++) {
                    if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
                }
                free(rfacts);
                return RECOUNT_ENOMEM;
            }
        }
    }
    *out_rfacts = rfacts;
    return RECOUNT_OK;
}

static void free_rfacts(recount_rfactln_t **rfacts, int n) {
    if (!rfacts) return;
    for (int v = 0; v < n; v++) {
        if (rfacts[v]) { recount_rfactln_free(rfacts[v]); free(rfacts[v]); }
    }
    free(rfacts);
}

/* Per-family kernel: forward + outside + posterior accumulation.
 *
 * If pf_copies / pf_present are non-NULL, the per-node values for THIS
 * family are also written into them (length: num_nodes each). Used by
 * recount_per_family_posteriors_batch to expose family-level posteriors
 * without re-running the whole inside/outside pipeline. */
static int process_family(
    const recount_tree_t *tree, const recount_survival_t *sp,
    const int32_t *profile,
    const recount_factln_t *fact,
    const recount_rfactln_t * const *rfacts,
    grad_worker_t *w, int tail_size,
    double *out_ll,
    double *pf_copies,    /* nullable [num_nodes] */
    double *pf_present,   /* nullable [num_nodes] */
    double family_weight) /* multiplier applied to all `+=` accumulations;
                           * 1.0 for unweighted (identity behavior).
                           * Used by recount_gradient_batch_weighted to
                           * support K>1 mixture gradients via per-family
                           * posterior responsibilities γ_{f,k}. */
{
    int n = tree->num_nodes;
    int root = tree->root;

    /* Forward */
    double LL;
    int rc = recount_forward_family_with_scratch(
        tree, sp, profile, fact, rfacts,
        w->widths, w->C_slots, w->K_slots,
        w->bigpool, w->bigpool_doubles, &LL);
    if (rc != RECOUNT_OK) return rc;
    *out_ll = LL;

    /* Derive K_lens from widths */
    for (int v = 0; v < n; v++) {
        if (v == root) {
            w->K_lens[v] = (sp->log_p_c[v] == NEG_INF) ? 1 : 2;
        } else {
            w->K_lens[v] = w->widths[v];
        }
    }

    /* Allocate B_slots and J_slots from bigpool — bump past what forward used.
     * recount_forward_family_with_scratch consumed roughly:
     *   sum_v (widths[v] + K_lens[v]) + (max_w + 4) + (2*max_w+4) + (2*max_w+4)
     * but we can't easily know exactly. Simpler: allocate B/J at the END of
     * bigpool and rely on bigpool_doubles being sized to fit BOTH forward AND
     * outside (alloc_worker computed pool_doubles for that case). */
    /* Easiest: use forward's outputs for C/K, but B/J need their own slots.
     * We'll lay them out at the FAR END of bigpool. */
    int max_w = 0;
    for (int v = 0; v < n; v++) {
        if (w->widths[v] > max_w) max_w = w->widths[v];
        if (w->K_lens[v] > max_w) max_w = w->K_lens[v];
    }
    size_t per_node = (size_t)(max_w + 2) * 2;  /* B[v] + J[v] each up to max_w */
    size_t bj_off = w->bigpool_doubles - per_node * (size_t)n - 64;
    double *bj_pool = w->bigpool + bj_off;
    size_t off = 0;
    for (int v = 0; v < n; v++) {
        w->B_slots[v] = &bj_pool[off]; off += (size_t)w->widths[v];
        w->J_slots[v] = &bj_pool[off]; off += (size_t)w->K_lens[v];
    }

    /* Outside */
    rc = recount_compute_outside(
        tree, sp, w->widths, w->K_lens,
        w->C_slots, w->K_slots, w->B_slots, w->J_slots,
        fact, rfacts, w->outside_work, w->max_K_len);
    if (rc != RECOUNT_OK) return rc;

    /* Posterior marginal stats per node — accumulate into thread-local */
    /* Reuse w->outside_work as a temporary tail buffer (it has room) */
    double *tail_scratch = w->outside_work;
    for (int v = 0; v < n; v++) {
        double mean_node = 0.0, mean_edge = 0.0;
        /* Node marginals from B[v] + C[v] */
        int Cv_len = w->widths[v];
        if (Cv_len > 0) {
            /* Zero the tail scratch up to widths[v] */
            for (int i = 0; i < Cv_len; i++) tail_scratch[i] = 0.0;
            posterior_marginal_stats(
                w->B_slots[v], Cv_len, w->C_slots[v], Cv_len, LL,
                &mean_node, tail_scratch, Cv_len);
            w->N_v[v] += family_weight * mean_node;
            int n_acc = Cv_len < tail_size ? Cv_len : tail_size;
            double *dst = w->N_tail + (size_t)v * (size_t)tail_size;
            for (int i = 0; i < n_acc; i++) dst[i] += family_weight * tail_scratch[i];
            /* num_families_active: P(ξ_v > 0) = 1 - exp(B[v][0] + C[v][0] - LL).
             * Non-finite log_p (-inf or NaN) → posterior P(ξ_v=0) = 0
             * (vanishing mass at 0) → treat as active. Matches the NumPy
             * reference which filters non-finite log_p to post=0. */
            double lp0 = w->B_slots[v][0] + w->C_slots[v][0] - LL;
            double p0 = isfinite(lp0) ? exp(lp0) : 0.0;
            /* n_active is an unweighted integer count — leave as-is so the
             * unweighted call path is bit-identical. */
            if (1.0 - p0 > w->active_threshold) w->n_active[v]++;
            w->fp_v[v] += family_weight * (1.0 - p0);
            if (pf_copies)  pf_copies[v]  = mean_node;
            if (pf_present) pf_present[v] = 1.0 - p0;
        } else {
            if (pf_copies)  pf_copies[v]  = 0.0;
            if (pf_present) pf_present[v] = 0.0;
        }
        /* Edge marginals from J[v] + K[v] */
        int Kv_len = w->K_lens[v];
        if (Kv_len > 0) {
            for (int i = 0; i < Kv_len; i++) tail_scratch[i] = 0.0;
            posterior_marginal_stats(
                w->J_slots[v], Kv_len, w->K_slots[v], Kv_len, LL,
                &mean_edge, tail_scratch, Kv_len);
            w->S_v[v] += family_weight * mean_edge;
            int n_acc = Kv_len < tail_size ? Kv_len : tail_size;
            double *dst = w->S_tail + (size_t)v * (size_t)tail_size;
            for (int i = 0; i < n_acc; i++) dst[i] += family_weight * tail_scratch[i];
        }
    }
    return RECOUNT_OK;
}

/* ===================================================================
 * Per-branch events
 * =================================================================== */

int recount_branch_stats_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double active_threshold,
    recount_branch_stats_t *out)
{
    int n = tree->num_nodes;
    int num_leaves = tree->num_leaves;
    int root = tree->root;

    /* Max profile sum for factorial caches */
    int max_sum = 0, max_count = 0;
    for (int32_t f = 0; f < F; f++) {
        int s = 0;
        const int32_t *prof = profiles + (size_t)f * num_leaves;
        for (int32_t v = 0; v < num_leaves; v++) {
            int32_t c = prof[v];
            if (c > 0) { s += c; if (c > max_count) max_count = c; }
        }
        if (s > max_sum) max_sum = s;
    }
    int max_n_fact = (max_count > max_sum ? max_count : max_sum) + 8;
    int tail_size = max_sum + 2;

    recount_factln_t fact;
    if (recount_factln_init(&fact, max_n_fact) != RECOUNT_OK) return RECOUNT_ENOMEM;
    recount_rfactln_t **rfacts = NULL;
    int rc = build_rfacts_shared(sp, max_n_fact, &rfacts);
    if (rc != RECOUNT_OK) { recount_factln_free(&fact); return rc; }

    int nworkers = (num_threads > 0) ? num_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nworkers < 1) nworkers = 1;
    if (nworkers > 32) nworkers = 32;

    grad_worker_t *workers = (grad_worker_t *)calloc((size_t)nworkers, sizeof(grad_worker_t));
    if (!workers) { free_rfacts(rfacts, n); recount_factln_free(&fact); return RECOUNT_ENOMEM; }
    int alloc_failed = 0;
    for (int w = 0; w < nworkers; w++) {
        if (alloc_worker(&workers[w], n, max_sum + 1, tail_size) != RECOUNT_OK) {
            alloc_failed = 1;
        } else {
            workers[w].active_threshold = active_threshold;
        }
    }
    if (alloc_failed) {
        for (int w = 0; w < nworkers; w++) free_worker(&workers[w]);
        free(workers); free_rfacts(rfacts, n); recount_factln_free(&fact);
        return RECOUNT_ENOMEM;
    }

    /* next_idx is shared across workers; needs __block on Apple for
     * the dispatch closure to see writes, plain shared on OpenMP. */
#ifdef __APPLE__
    __block int32_t next_idx = 0;
#else
    int32_t next_idx = 0;
#endif
    RECOUNT_PARALLEL_APPLY(nworkers, {
        grad_worker_t *wk = &workers[worker_id];
        for (;;) {
            int32_t f = __sync_fetch_and_add(&next_idx, 1);
            if (f >= F) break;
            double ll = 0.0;
            (void)process_family(
                tree, sp, profiles + (size_t)f * num_leaves,
                &fact, (const recount_rfactln_t * const *)rfacts,
                wk, tail_size, &ll,
                NULL, NULL,  /* aggregated stats only */
                1.0);        /* unweighted */
        }
    });

    /* Reduce */
    for (int v = 0; v < n; v++) {
        out->copies_node[v] = 0.0;
        out->copies_edge[v] = 0.0;
        out->families_present[v] = 0.0;
        out->num_families_active[v] = 0;
    }
    for (int w = 0; w < nworkers; w++) {
        for (int v = 0; v < n; v++) {
            out->copies_node[v] += workers[w].N_v[v];
            out->copies_edge[v] += workers[w].S_v[v];
            out->families_present[v] += workers[w].fp_v[v];
            out->num_families_active[v] += workers[w].n_active[v];
        }
    }
    /* gain_events = copies_node - copies_edge */
    for (int v = 0; v < n; v++) {
        out->gain_events[v] = out->copies_node[v] - out->copies_edge[v];
    }
    /* loss_events[v] = max(copies_node[parent] - copies_edge[v], 0) */
    for (int v = 0; v < n; v++) out->loss_events[v] = 0.0;
    for (int v = 0; v < n; v++) {
        if (v == root) continue;
        int parent = tree->parent[v];
        double l = out->copies_node[parent] - out->copies_edge[v];
        out->loss_events[v] = (l > 0.0) ? l : 0.0;
    }

    for (int w = 0; w < nworkers; w++) free_worker(&workers[w]);
    free(workers);
    free_rfacts(rfacts, n);
    recount_factln_free(&fact);
    return RECOUNT_OK;
}

/* ===================================================================
 * Public API: per-family per-node posteriors
 *
 * Same inside/outside pipeline as recount_branch_stats_batch but writes
 * the per-family per-node E[ξ_v|f] and P{ξ_v>0|f} matrices instead of
 * aggregating. Output buffers are F·N doubles each.
 * =================================================================== */
int recount_per_family_posteriors_batch(
    const recount_tree_t *tree, const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_copies_FxN,    /* [F * num_nodes] row-major */
    double *out_present_FxN)   /* [F * num_nodes] row-major */
{
    int n = tree->num_nodes;
    int num_leaves = tree->num_leaves;

    int max_sum = 0, max_count = 0;
    for (int32_t f = 0; f < F; f++) {
        int s = 0;
        const int32_t *prof = profiles + (size_t)f * num_leaves;
        for (int32_t v = 0; v < num_leaves; v++) {
            int32_t c = prof[v];
            if (c > 0) {
                s += c;
                if (c > max_count) max_count = c;
            }
        }
        if (s > max_sum) max_sum = s;
    }
    int max_n_fact = (max_count > max_sum ? max_count : max_sum) + 8;
    int tail_size = max_sum + 2;

    recount_factln_t fact;
    if (recount_factln_init(&fact, max_n_fact) != RECOUNT_OK) return RECOUNT_ENOMEM;
    recount_rfactln_t **rfacts = NULL;
    int rc = build_rfacts_shared(sp, max_n_fact, &rfacts);
    if (rc != RECOUNT_OK) { recount_factln_free(&fact); return rc; }

    int nworkers = (num_threads > 0) ? num_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nworkers < 1) nworkers = 1;
    if (nworkers > 32) nworkers = 32;

    grad_worker_t *workers = (grad_worker_t *)calloc((size_t)nworkers, sizeof(grad_worker_t));
    if (!workers) { free_rfacts(rfacts, n); recount_factln_free(&fact); return RECOUNT_ENOMEM; }
    int alloc_failed = 0;
    for (int wi = 0; wi < nworkers; wi++) {
        if (alloc_worker(&workers[wi], n, max_sum + 1, tail_size) != RECOUNT_OK) alloc_failed = 1;
    }
    if (alloc_failed) {
        for (int wi = 0; wi < nworkers; wi++) free_worker(&workers[wi]);
        free(workers); free_rfacts(rfacts, n); recount_factln_free(&fact);
        return RECOUNT_ENOMEM;
    }

    /* next_idx is shared across workers; needs __block on Apple for
     * the dispatch closure to see writes, plain shared on OpenMP. */
#ifdef __APPLE__
    __block int32_t next_idx = 0;
#else
    int32_t next_idx = 0;
#endif
    RECOUNT_PARALLEL_APPLY(nworkers, {
        grad_worker_t *wk = &workers[worker_id];
        for (;;) {
            int32_t f = __sync_fetch_and_add(&next_idx, 1);
            if (f >= F) break;
            double ll = 0.0;
            double *pf_copies  = out_copies_FxN  + (size_t)f * (size_t)n;
            double *pf_present = out_present_FxN + (size_t)f * (size_t)n;
            (void)process_family(
                tree, sp, profiles + (size_t)f * num_leaves,
                &fact, (const recount_rfactln_t * const *)rfacts,
                wk, tail_size, &ll, pf_copies, pf_present,
                1.0);  /* unweighted */
        }
    });

    for (int wi = 0; wi < nworkers; wi++) free_worker(&workers[wi]);
    free(workers); free_rfacts(rfacts, n); recount_factln_free(&fact);
    return RECOUNT_OK;
}

/* ===================================================================
 * Public API: gradient batch
 * =================================================================== */

/* Public API supporting both unweighted (family_weights == NULL) and
 * per-family-weighted gradient aggregation. When weights != NULL, each
 * family's contribution to the gradient accumulators (and the final Fd
 * regulariser) is multiplied by weights[f]. Use case: K>1 LogisticShift
 * mixture gradient, where weights = posterior responsibility γ_{f,k} of
 * category k for family f and the kernel is called once per category.
 *
 * Unweighted call: pass family_weights = NULL; behavior is bit-identical
 * to the previous version of this function.
 */
int recount_gradient_batch_full(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    int min_copies,
    const double *family_weights,  /* nullable [F] */
    double *out_lls,
    double *out_grad)
{
    int n = tree->num_nodes;
    int num_leaves = tree->num_leaves;
    int root = tree->root;

    /* Determine max profile sum + max count for factorial sizing. */
    int max_sum = 0, max_count = 0;
    for (int32_t f = 0; f < F; f++) {
        int s = 0;
        const int32_t *prof = profiles + (size_t)f * num_leaves;
        for (int32_t v = 0; v < num_leaves; v++) {
            int32_t c = prof[v];
            if (c > 0) {
                s += c;
                if (c > max_count) max_count = c;
            }
        }
        if (s > max_sum) max_sum = s;
    }
    int max_n_fact = (max_count > max_sum ? max_count : max_sum) + 8;
    int tail_size = max_sum + 2;  /* per-node tail array sizing */

    /* Shared factorial caches */
    recount_factln_t fact;
    if (recount_factln_init(&fact, max_n_fact) != RECOUNT_OK) return RECOUNT_ENOMEM;
    recount_rfactln_t **rfacts = NULL;
    int rc = build_rfacts_shared(sp, max_n_fact, &rfacts);
    if (rc != RECOUNT_OK) { recount_factln_free(&fact); return rc; }

    int nworkers = (num_threads > 0) ? num_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nworkers < 1) nworkers = 1;
    if (nworkers > 32) nworkers = 32;

    grad_worker_t *workers = (grad_worker_t *)calloc((size_t)nworkers, sizeof(grad_worker_t));
    if (!workers) { free_rfacts(rfacts, n); recount_factln_free(&fact); return RECOUNT_ENOMEM; }
    int alloc_failed = 0;
    for (int w = 0; w < nworkers; w++) {
        if (alloc_worker(&workers[w], n, max_sum + 1, tail_size) != RECOUNT_OK) {
            alloc_failed = 1;
        }
    }
    if (alloc_failed) {
        for (int w = 0; w < nworkers; w++) free_worker(&workers[w]);
        free(workers); free_rfacts(rfacts, n); recount_factln_free(&fact);
        return RECOUNT_ENOMEM;
    }

    /* Distribute families */
    /* next_idx is shared across workers; needs __block on Apple for
     * the dispatch closure to see writes, plain shared on OpenMP. */
#ifdef __APPLE__
    __block int32_t next_idx = 0;
#else
    int32_t next_idx = 0;
#endif
    RECOUNT_PARALLEL_APPLY(nworkers, {
        grad_worker_t *wk = &workers[worker_id];
        for (;;) {
            int32_t f = __sync_fetch_and_add(&next_idx, 1);
            if (f >= F) break;
            double ll = 0.0;
            double weight = (family_weights != NULL) ? family_weights[f] : 1.0;
            int rc1 = process_family(
                tree, sp, profiles + (size_t)f * num_leaves,
                &fact, (const recount_rfactln_t * const *)rfacts,
                wk, tail_size, &ll,
                NULL, NULL,  /* gradient path doesn't need per-family copies */
                weight);
            out_lls[f] = (rc1 == RECOUNT_OK) ? ll : NAN;
        }
    });

    /* Reduce thread-local accumulators */
    double *N_v = (double *)calloc((size_t)n, sizeof(double));
    double *S_v = (double *)calloc((size_t)n, sizeof(double));
    double *N_tail = (double *)calloc((size_t)n * (size_t)tail_size, sizeof(double));
    double *S_tail = (double *)calloc((size_t)n * (size_t)tail_size, sizeof(double));
    if (!N_v || !S_v || !N_tail || !S_tail) {
        free(N_v); free(S_v); free(N_tail); free(S_tail);
        for (int w = 0; w < nworkers; w++) free_worker(&workers[w]);
        free(workers); free_rfacts(rfacts, n); recount_factln_free(&fact);
        return RECOUNT_ENOMEM;
    }
    for (int w = 0; w < nworkers; w++) {
        for (int v = 0; v < n; v++) {
            N_v[v] += workers[w].N_v[v];
            S_v[v] += workers[w].S_v[v];
        }
        for (size_t i = 0; i < (size_t)n * (size_t)tail_size; i++) {
            N_tail[i] += workers[w].N_tail[i];
            S_tail[i] += workers[w].S_tail[i];
        }
    }

    /* Compute one_minus_L0 (closed form). */
    int use_correction = (min_copies >= 1);
    double one_minus_L0 = 1.0;
    if (use_correction) {
        if (min_copies > 1) {
            /* min_copies >= 2 needs an additional ∂(F·log(1-(L0+L1+...)))/∂θ
             * term that the native gradient kernel doesn't yet compute. The
             * Python orchestrator (corrected_log_likelihood_native) handles
             * min_copies=2 for the forward LL via recount.gld helpers, but
             * the analytical gradient currently only supports {0, 1}. */
            return RECOUNT_ENOTSUP;
        }
        double L0 = empty_log_likelihood(sp, root);
        one_minus_L0 = -expm1(L0);
    }

    /* Apply Csurös 2021 Corollary 9 — fill out_grad[3*v + axis] */
    memset(out_grad, 0, (size_t)(3 * n) * sizeof(double));
    /* Effective family count: sum of weights when weighted, plain F otherwise. */
    double Fd;
    if (family_weights != NULL) {
        Fd = 0.0;
        for (int32_t f = 0; f < F; f++) Fd += family_weights[f];
    } else {
        Fd = (double)F;
    }
    for (int v = 0; v < n; v++) {
        int is_polya = sp->is_polya[v];
        double kappa_or_r = sp->gain[v];
        double q = sp->q[v], q_c = sp->q_c[v];

        /* GAIN derivative */
        if (is_polya) {
            double log1_q = sp->log_q_c[v];
            double emp = use_correction ? (Fd * log1_q / one_minus_L0) : (Fd * log1_q);
            int len_tail = tail_size;
            for (int i = 0; i < len_tail; i++) {
                double n_t = N_tail[(size_t)v * tail_size + i];
                double s_t = S_tail[(size_t)v * tail_size + i];
                emp += (n_t - s_t) / (kappa_or_r + (double)i);
            }
            out_grad[3 * v + RECOUNT_GAIN] = emp;
        } else {
            if (kappa_or_r > 0.0) {
                if (use_correction) {
                    out_grad[3 * v + RECOUNT_GAIN] =
                        (N_v[v] - S_v[v]) / kappa_or_r - Fd / one_minus_L0;
                } else {
                    out_grad[3 * v + RECOUNT_GAIN] = (N_v[v] - S_v[v]) / kappa_or_r - Fd;
                }
            }
        }

        /* DUP derivative */
        if (is_polya) {
            double dup_grad;
            if (use_correction) {
                dup_grad = (N_v[v] - S_v[v]) / q
                         - (S_v[v] + Fd * kappa_or_r / one_minus_L0) / q_c;
            } else {
                dup_grad = (N_v[v] - S_v[v]) / q - (S_v[v] + Fd * kappa_or_r) / q_c;
            }
            out_grad[3 * v + RECOUNT_DUP] = dup_grad;
        }

        /* LOSS derivative (non-root only) */
        if (v == root) continue;
        double p = sp->p[v], p_c = sp->p_c[v];
        int parent = tree->parent[v];
        double eps_sib = 1.0;
        int cs = tree->first_child[parent], ce = tree->first_child[parent + 1];
        for (int ci = cs; ci < ce; ci++) {
            int c = tree->child_list[ci];
            if (c != v) eps_sib *= sp->p[c];
        }
        double one_minus_pe = 1.0 - p * eps_sib;
        out_grad[3 * v + RECOUNT_LOSS] = (
            (N_v[parent] - S_v[v]) / p
            - (1.0 - eps_sib) * S_v[v] / p_c
        ) / one_minus_pe;
    }

    free(N_v); free(S_v); free(N_tail); free(S_tail);
    for (int w = 0; w < nworkers; w++) free_worker(&workers[w]);
    free(workers);
    free_rfacts(rfacts, n);
    recount_factln_free(&fact);
    return RECOUNT_OK;
}

/* Stub matching the original header API. */
int recount_gradient_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_lls,
    double *out_grad)
{
    return recount_gradient_batch_full(tree, sp, profiles, F, num_threads,
                                       /*min_copies=*/1, /*family_weights=*/NULL,
                                       out_lls, out_grad);
}
