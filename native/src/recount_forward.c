/* Port of recount.gld._forward (gld.py:320-374).
 * Felsenstein peeling for a single profile (single family).
 *
 * Scratchpad layout (one big bump allocator):
 *   - C[v] array: per-node inside log-likelihoods, padded to width[v]+1.
 *   - K[v] array: per-node edge log-likelihoods, padded to K_len[v].
 *   - terms_scratch: temporary for logsumexp inside combine/edge.
 *   - C_work_buf: copy of C used for destructive in-place mutation.
 *   - C2_buf: output buffer for combine results.
 *   - widths[v]: per-node widths.
 *
 * For simplicity v1 allocates all of these inside the family pass; v2
 * uses a persistent thread-local arena (see recount_batch.c).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "recount_native.h"
#include "recount_internal.h"

extern void recount_compute_edge(
    const double *C, int C_len, double *K, int K_len,
    double log_q, double log_q_c, double log_gain, int is_polya,
    const recount_factln_t *fact, const recount_rfactln_t *rfact,
    double *terms_scratch);

extern void recount_compute_sibling(
    double *C, int C_len,
    const double *K, int K_len,
    double p_junior, double p_junior_c, double eps_sib,
    const recount_factln_t *fact,
    double *C2, int *out_combined,
    double *terms_scratch);

/* Compute per-node widths from a profile, mirroring gld._calc_widths.
 * widths[v] = max(surviving copies) + 1 at node v.
 * Leaves: max(0, count + 1) (negative count == ambiguous → 0).
 * Internal: 0 if all children ambiguous, else (sum of children's (w-1)) + 1.
 */
static void calc_widths(const recount_tree_t *t, const int32_t *profile, int32_t *widths)
{
    for (int32_t v = 0; v < t->num_nodes; v++) {
        if (t->is_leaf[v]) {
            int32_t c = profile[v];
            widths[v] = c < 0 ? 0 : c + 1;
        } else {
            int32_t cs = t->first_child[v], ce = t->first_child[v + 1];
            int32_t ambi = 0, total = 0, nc = 0;
            for (int32_t ci = cs; ci < ce; ci++) {
                int32_t c = t->child_list[ci];
                int32_t cn = widths[c] - 1;
                if (cn < 0) ambi++; else total += cn;
                nc++;
            }
            widths[v] = (ambi == nc) ? 0 : total + 1;
        }
    }
}

int recount_forward_family_with_scratch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profile,
    const recount_factln_t *fact,
    const recount_rfactln_t * const *rfacts,   /* per-node, may be NULL */
    int32_t *widths,                            /* [num_nodes] */
    double **C_slots, double **K_slots,         /* [num_nodes] each */
    double *bigpool, size_t bigpool_n,
    double *out_ll)
{
    int32_t n = tree->num_nodes;
    calc_widths(tree, profile, widths);

    /* Bump-allocate C[v], K[v] from bigpool */
    size_t off = 0;
    int max_w = 0;
    for (int32_t v = 0; v < n; v++) {
        int32_t w = widths[v];
        if (w > max_w) max_w = w;
        C_slots[v] = &bigpool[off];
        off += (size_t)w;
        /* K_len = w except root: 1 or 2 depending on ROOTLOSS */
        int32_t k_len;
        if (v == tree->root) {
            k_len = (sp->log_p_c[v] == NEG_INF) ? 1 : 2;
        } else {
            k_len = w;
        }
        K_slots[v] = &bigpool[off];
        off += (size_t)k_len;
    }
    /* terms_scratch: max(max_w, max(K_len)) */
    double *terms_scratch = &bigpool[off]; off += (size_t)(max_w + 4);
    /* combine work buffers: C_work (mutated copy), C2 (output) */
    double *C_work = &bigpool[off]; off += (size_t)(2 * max_w + 4);
    double *C2_buf = &bigpool[off]; off += (size_t)(2 * max_w + 4);
    if (off > bigpool_n) return RECOUNT_ENOMEM;

    /* Post-order bottom-up pass */
    for (int32_t v = 0; v < n; v++) {
        int32_t w = widths[v];
        double *Cv = C_slots[v];

        if (tree->is_leaf[v]) {
            for (int32_t i = 0; i < w; i++) Cv[i] = NEG_INF;
            if (w > 0) Cv[w - 1] = 0.0;
        } else {
            if (sp->eps[v] == 1.0) {
                /* Whole subtree extinct */
                double acc = 0.0;
                int32_t cs = tree->first_child[v], ce = tree->first_child[v + 1];
                for (int32_t ci = cs; ci < ce; ci++) {
                    int32_t c = tree->child_list[ci];
                    /* K[c][0] if it exists */
                    /* note widths/K_slots[c] already populated */
                    /* find K[c]'s allocated length: w_c if non-root else 1 or 2 */
                    if (widths[c] > 0) acc += K_slots[c][0];
                }
                if (w >= 1) {
                    for (int32_t i = 0; i < w; i++) Cv[i] = NEG_INF;
                    Cv[0] = acc;
                }
            } else {
                /* Chained binary combine */
                int32_t cs = tree->first_child[v], ce = tree->first_child[v + 1];
                int Cv_len = 0;
                double sib_extinct = 1.0;
                for (int32_t ci = cs; ci < ce; ci++) {
                    int32_t c = tree->child_list[ci];
                    int Kc_len = (c == tree->root) ? ((sp->log_p_c[c] == NEG_INF) ? 1 : 2)
                                                   : widths[c];
                    const double *Kc = K_slots[c];
                    int combined = 0;
                    if (Cv_len == 0) {
                        /* First sibling: result is K[c] directly */
                        for (int i = 0; i < Kc_len; i++) C2_buf[i] = Kc[i];
                        combined = Kc_len - 1;
                    } else {
                        /* Copy current C2_buf into C_work for destructive mutation */
                        for (int i = 0; i < Cv_len; i++) C_work[i] = C2_buf[i];
                        recount_compute_sibling(
                            C_work, Cv_len,
                            Kc, Kc_len,
                            sp->p[c], sp->p_c[c], sib_extinct,
                            fact,
                            C2_buf, &combined,
                            terms_scratch);
                    }
                    Cv_len = combined + 1;
                    sib_extinct *= sp->p[c];
                }
                /* Trim/pad to width[v] */
                int n_copy = Cv_len < w ? Cv_len : w;
                for (int i = 0; i < n_copy; i++) Cv[i] = C2_buf[i];
                for (int i = n_copy; i < w; i++) Cv[i] = NEG_INF;
            }
        }

        /* Edge step → K[v] */
        int32_t k_len;
        if (v == tree->root) {
            k_len = (sp->log_p_c[v] == NEG_INF) ? 1 : 2;
        } else {
            k_len = w;
        }
        recount_compute_edge(
            Cv, w, K_slots[v], k_len,
            sp->log_q[v], sp->log_q_c[v], sp->log_gain[v], sp->is_polya[v],
            fact, rfacts ? rfacts[v] : NULL, terms_scratch);
    }

    /* Root LL */
    int32_t root = tree->root;
    double *Kr = K_slots[root];
    double LL = Kr[0];
    double p_root = sp->p[root];
    int kr_len = (sp->log_p_c[root] == NEG_INF) ? 1 : 2;
    if (p_root != 1.0 && kr_len == 2) {
        LL = LL + log(p_root);
        LL = recount_logaddexp(LL, Kr[1] + log(sp->p_c[root]));
    }
    *out_ll = LL;
    return RECOUNT_OK;
}

/* Convenience: allocate caches on the heap and run one family. */
size_t recount_forward_scratch_size(int32_t num_nodes, int32_t max_width)
{
    /* per-node C + K (each up to max_width+2) + slot pointers + widths */
    /* + global terms_scratch + C_work + C2_buf (each ~2*max_width) */
    size_t per_node = (size_t)(2 * (max_width + 2));
    size_t pool_doubles = per_node * (size_t)num_nodes + (size_t)(6 * (max_width + 8));
    size_t slots = (size_t)num_nodes * (2 * sizeof(double *) + sizeof(int32_t));
    return pool_doubles * sizeof(double) + slots + 1024;
}

int recount_forward_family(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profile,
    void *scratch, size_t scratch_n,
    double *out_ll)
{
    /* Build factln & rfacts on the fly (small overhead for one family) */
    int max_count = 0;
    for (int32_t v = 0; v < tree->num_leaves; v++) {
        if (profile[v] > max_count) max_count = profile[v];
    }
    int total = 0;
    for (int32_t v = 0; v < tree->num_leaves; v++) {
        if (profile[v] > 0) total += profile[v];
    }
    int max_n = (max_count > total ? max_count : total) + 8;

    recount_factln_t fact;
    if (recount_factln_init(&fact, max_n) != RECOUNT_OK) return RECOUNT_ENOMEM;

    /* Per-node rfacts (only for polya nodes with gain > 0) */
    int n = tree->num_nodes;
    recount_rfactln_t **rfacts = (recount_rfactln_t **)calloc((size_t)n, sizeof(*rfacts));
    if (!rfacts) { recount_factln_free(&fact); return RECOUNT_ENOMEM; }
    for (int32_t v = 0; v < n; v++) {
        if (sp->is_polya[v] && sp->gain[v] > 0.0) {
            rfacts[v] = (recount_rfactln_t *)malloc(sizeof(recount_rfactln_t));
            if (!rfacts[v] || recount_rfactln_init(rfacts[v], sp->gain[v], total + 8) != RECOUNT_OK) {
                /* cleanup */
                for (int32_t u = 0; u <= v; u++) {
                    if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
                }
                free(rfacts); recount_factln_free(&fact);
                return RECOUNT_ENOMEM;
            }
        }
    }

    /* Lay out scratch */
    uint8_t *base = (uint8_t *)scratch;
    int32_t *widths = (int32_t *)base;        base += (size_t)n * sizeof(int32_t);
    double **C_slots = (double **)base;       base += (size_t)n * sizeof(double *);
    double **K_slots = (double **)base;       base += (size_t)n * sizeof(double *);
    /* align to 16 bytes */
    size_t used = (size_t)(base - (uint8_t *)scratch);
    used = (used + 15) & ~(size_t)15;
    base = (uint8_t *)scratch + used;
    double *bigpool = (double *)base;
    size_t bigpool_n = (scratch_n - used) / sizeof(double);

    int rc = recount_forward_family_with_scratch(
        tree, sp, profile, &fact, (const recount_rfactln_t * const *)rfacts,
        widths, C_slots, K_slots, bigpool, bigpool_n, out_ll);

    /* Cleanup */
    for (int32_t v = 0; v < n; v++) {
        if (rfacts[v]) { recount_rfactln_free(rfacts[v]); free(rfacts[v]); }
    }
    free(rfacts);
    recount_factln_free(&fact);
    return rc;
}
