/* Outside (down-pass) — port of recount.gld._compute_outside (gld.py:490-621).
 *
 * Given a family's forward state (C[v], K[v] per node), computes:
 *   B[v][ℓ] = log P(profile outside v's subtree | v has ℓ surviving copies)
 *   J[v][s] = log P(profile outside v's edge   | s copies enter v's edge)
 *
 * Pre-order descent: for each non-root v, combine the inside K's of v's
 * siblings (excluding v) into K_sib, then symmetric destructive update on
 * K2_work + a gather producing J[v]. Edge step turns J[v] into B[v].
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "recount_native.h"
#include "recount_internal.h"

extern void recount_compute_sibling(
    double *C, int C_len,
    const double *K, int K_len,
    double p_junior, double p_junior_c, double eps_sib,
    const recount_factln_t *fact,
    double *C2, int *out_combined,
    double *terms_scratch);

extern void recount_compute_edge(
    const double *C, int C_len, double *K, int K_len,
    double log_q, double log_q_c, double log_gain, int is_polya,
    const recount_factln_t *fact, const recount_rfactln_t *rfact,
    double *terms_scratch);

/* B[v] from J[v] via the edge-gain PMF (going outside→inside).
 *   B[v][ell] = LSE_s J[v][s] + gain_pmf(ell - s | s)
 * Same PMF as the forward edge step but the iteration direction is reversed. */
static void edge_outside_to_node(
    const double *Jv, int Jv_len,
    double *Bv, int Bv_len,
    double log_q, double log_q_c, double log_gain, int is_polya,
    const recount_factln_t *fact, const recount_rfactln_t *rfact,
    double *terms_scratch)
{
    if (Bv_len == 0) return;
    for (int i = 0; i < Bv_len; i++) Bv[i] = NEG_INF;

    if (!is_polya) {
        if (log_gain == NEG_INF) {
            int n = Bv_len < Jv_len ? Bv_len : Jv_len;
            for (int ell = 0; ell < n; ell++) Bv[ell] = Jv[ell];
            return;
        }
        double r = exp(log_gain);
        /* tterm[t] = t·log_gain - factln(t) - r, folding (t==0)?0.0. */
        double tterm_stack[RECOUNT_VVEXP_STACK_MAX];
        double *tterm = recount_width_scratch(
            tterm_stack, RECOUNT_VVEXP_STACK_MAX, Bv_len);
        if (!tterm) return;   /* OOM — Bv already NEG_INF-initialised */
        tterm[0] = -r;
        for (int t = 1; t < Bv_len; t++) {
            tterm[t] = (double)t * log_gain - recount_factln_get(fact, t) - r;
        }
        for (int ell = 0; ell < Bv_len; ell++) {
            int s_hi = ell + 1;
            if (s_hi > Jv_len) s_hi = Jv_len;
            int cnt = 0;
            for (int s = 0; s < s_hi; s++) {
                terms_scratch[cnt++] = Jv[s] + tterm[ell - s];
            }
            Bv[ell] = recount_logsumexp(terms_scratch, cnt);
        }
        recount_width_scratch_free(tterm, tterm_stack);
        return;
    }

    /* Pólya */
    if (log_gain == NEG_INF) {
        int n = Bv_len < Jv_len ? Bv_len : Jv_len;
        for (int ell = 0; ell < n; ell++) Bv[ell] = Jv[ell];
        return;
    }
    double loglog1_q = (log_q_c < 0.0) ? log(-log_q_c) : log_q;
    double kappa_log1_q = exp(log_gain + loglog1_q);

    /* Per-node precomputes — invariant across the (ell, s) gather:
     *   tterm[t]   = t·log_q - factln(t)            (folds (t==0)?0.0 guard)
     *   sterm[s]   = -rfact(s) + ks_log1_q          (s-only part of binom + ks term)
     * Then inner: Jv[s] + tterm[t] + sterm[s] + rfact(ell).
     * The rfact(ell) is invariant inside the s loop — pull out. */
    double obn_stack[2 * RECOUNT_VVEXP_STACK_MAX];
    double *obn_buf = recount_width_scratch(
        obn_stack, 2 * RECOUNT_VVEXP_STACK_MAX, Bv_len + Jv_len);
    if (!obn_buf) return;   /* OOM — Bv already NEG_INF-initialised */
    double *tterm = obn_buf;
    double *sterm = obn_buf + Bv_len;
    tterm[0] = 0.0;
    for (int t = 1; t < Bv_len; t++) {
        tterm[t] = (double)t * log_q - recount_factln_get(fact, t);
    }
    for (int s = 0; s < Jv_len; s++) {
        double ks_log1_q;
        if (log_q_c == NEG_INF) {
            ks_log1_q = (s != 0) ? NEG_INF : -kappa_log1_q;
        } else {
            ks_log1_q = (double)s * log_q_c - kappa_log1_q;
        }
        sterm[s] = ks_log1_q - recount_rfactln_get(rfact, s);
    }

    for (int ell = 0; ell < Bv_len; ell++) {
        int s_hi = ell + 1;
        if (s_hi > Jv_len) s_hi = Jv_len;
        double rfact_ell = recount_rfactln_get(rfact, ell);
        int cnt = 0;
        for (int s = 0; s < s_hi; s++) {
            terms_scratch[cnt++] = Jv[s] + tterm[ell - s] + sterm[s] + rfact_ell;
        }
        Bv[ell] = recount_logsumexp(terms_scratch, cnt);
    }
    recount_width_scratch_free(obn_buf, obn_stack);
}

/* Combine siblings of `parent` excluding `exclude_child`.
 * Result written to `K_sib_out`, length returned in `*K_sib_len`.
 *
 * For binary trees this is trivial (one other sibling) — we copy its K[]
 * directly. For multifurcations, we chain combines.
 *
 * Workspace: K_sib_out and K_sib_work each must hold up to max width tensors.
 */
static void combine_siblings_excluding(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int parent, int exclude_child,
    int *K_lens,            /* [num_nodes] */
    double **K_slots,       /* [num_nodes] */
    const recount_factln_t *fact,
    double *K_sib_out, int *K_sib_len,
    double *K_sib_work, double *terms_scratch)
{
    int cs = tree->first_child[parent];
    int ce = tree->first_child[parent + 1];

    /* Single-pass: walk siblings, chain combine */
    int cur_len = 0;
    double sib_extinct = 1.0;
    int first = 1;
    for (int ci = cs; ci < ce; ci++) {
        int c = tree->child_list[ci];
        if (c == exclude_child) continue;
        int Kc_len = K_lens[c];
        const double *Kc = K_slots[c];
        if (first) {
            /* First sibling: K2 starts as K[c] */
            for (int i = 0; i < Kc_len; i++) K_sib_out[i] = Kc[i];
            cur_len = Kc_len;
            first = 0;
        } else {
            /* Combine K_sib_out (= C) with Kc (= K_junior) */
            for (int i = 0; i < cur_len; i++) K_sib_work[i] = K_sib_out[i];
            int combined;
            recount_compute_sibling(
                K_sib_work, cur_len,
                Kc, Kc_len,
                sp->p[c], sp->p_c[c], sib_extinct,
                fact,
                K_sib_out, &combined, terms_scratch);
            cur_len = combined + 1;
        }
        sib_extinct *= sp->p[c];
    }
    if (first) { *K_sib_len = 0; return; }  /* no siblings — should not happen in binary */
    *K_sib_len = cur_len;
}

/* Compute outside arrays J[v], B[v] for one family.
 *
 * Inputs (from a prior forward pass): C[v], K[v] per node, plus widths
 * and K_lens per node.
 *
 * Outputs (caller-allocated B_slots[v] and J_slots[v]): per-node arrays
 * with the same length as the corresponding C[v] (for B) or K[v] (for J).
 */
int recount_compute_outside(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int32_t *widths,                          /* [num_nodes] from forward */
    int *K_lens,                              /* [num_nodes] from forward */
    double **C_slots, double **K_slots,       /* forward state */
    double **B_slots, double **J_slots,       /* outputs (in scratch) */
    const recount_factln_t *fact,
    const recount_rfactln_t * const *rfacts,
    double *scratch_work,                     /* extra scratch ≥ 6*max_K_len */
    int max_K_len)
{
    int n = tree->num_nodes;
    int root = tree->root;

    /* Layout scratch_work */
    double *K_sib_buf  = scratch_work;
    double *K_sib_work = scratch_work + max_K_len * 2;
    double *terms_buf  = scratch_work + max_K_len * 4;

    /* Initialize J[root] */
    if (sp->log_p_c[root] == NEG_INF) {
        J_slots[root][0] = 0.0;
    } else {
        J_slots[root][0] = sp->log_p[root];
        J_slots[root][1] = sp->log_p_c[root];
    }
    int Jroot_len = (sp->log_p_c[root] == NEG_INF) ? 1 : 2;

    /* B[root] = edge_outside_to_node(root, J[root]) */
    edge_outside_to_node(
        J_slots[root], Jroot_len,
        B_slots[root], widths[root],
        sp->log_q[root], sp->log_q_c[root], sp->log_gain[root], sp->is_polya[root],
        fact, rfacts ? rfacts[root] : NULL, terms_buf);

    /* Pre-order descent: iterate v from N-1 down to 0 */
    for (int v = n - 1; v >= 0; v--) {
        if (v == root) continue;
        int parent = tree->parent[v];
        const double *Bu = B_slots[parent];
        int Bu_len = widths[parent];

        int Jv_len = K_lens[v];
        double *Jv = J_slots[v];
        for (int i = 0; i < Jv_len; i++) Jv[i] = NEG_INF;

        if (Bu_len == 0) {
            /* parent's B vanished; J[v] stays -inf, B[v] same */
            edge_outside_to_node(
                Jv, Jv_len, B_slots[v], widths[v],
                sp->log_q[v], sp->log_q_c[v], sp->log_gain[v], sp->is_polya[v],
                fact, rfacts ? rfacts[v] : NULL, terms_buf);
            continue;
        }

        /* K_sib = combine of siblings excluding v */
        int K_sib_len;
        combine_siblings_excluding(
            tree, sp, parent, v, K_lens, K_slots, fact,
            K_sib_buf, &K_sib_len, K_sib_work, terms_buf);

        double p = sp->p[v], p_c = sp->p_c[v];
        double eps_sib;
        int K2_len;
        double *K2 = K_sib_buf;
        if (K_sib_len == 0) {
            /* No siblings — should not happen in binary trees */
            K2_len = 1;
            K_sib_work[0] = 0.0;
            K2 = K_sib_work;
            eps_sib = 0.0;
        } else {
            K2_len = K_sib_len;
            eps_sib = 1.0;
            int cs = tree->first_child[parent], ce = tree->first_child[parent + 1];
            for (int ci = cs; ci < ce; ci++) {
                int c = tree->child_list[ci];
                if (c != v) eps_sib *= sp->p[c];
            }
        }

        double log_e   = (eps_sib > 0.0) ? log(eps_sib) : NEG_INF;
        double log_e_c = (eps_sib < 1.0) ? log1p(-eps_sib) : NEG_INF;
        double log_a   = log1p(-p * eps_sib);
        double logp1   = log(p_c) - log_a;
        double logp2   = log(p) + log_e_c - log_a;

        /* Symmetric destructive update on K2_work, computed inline with the
         * gather. We need a writable copy of K2 since the update mutates it
         * across the s loop. */
        double *K2_work = K_sib_work;
        for (int i = 0; i < K2_len; i++) K2_work[i] = K2[i];

        /* Precompute per-axis tables — invariant across the s loop.
         *   tterm[t]          = t·logp2 - factln(t)        (folds (t==0) guard)
         *   Bu_plus_fact[ell] = Bu[ell] + factln(ell)      (factln offset folded)
         * Per-s invariant `s_inv = s·logp1 - factln(s)` computed inline.
         * One heap block backs both tables for very wide families. */
        double obl_stack[2 * RECOUNT_VVEXP_STACK_MAX];
        double *obl_buf = recount_width_scratch(
            obl_stack, 2 * RECOUNT_VVEXP_STACK_MAX, K2_len + Bu_len);
        if (!obl_buf) return RECOUNT_ENOMEM;
        double *tterm = obl_buf;
        double *Bu_plus_fact = obl_buf + K2_len;
        tterm[0] = 0.0;
        for (int t = 1; t < K2_len; t++) {
            tterm[t] = (double)t * logp2 - recount_factln_get(fact, t);
        }
        for (int ell = 0; ell < Bu_len; ell++) {
            Bu_plus_fact[ell] = Bu[ell] + recount_factln_get(fact, ell);
        }

        for (int s = 0; s < Jv_len; s++) {
            if (s > 0 && K2_len > 0) {
                /* Symmetric destructive update: t = K2_len-1 down to 0
                 * K2[t] = logadd(prev_y + log_e_c, K2[t] + log_e)  for t < top
                 * K2[top] = K2[top] + log_e */
                int t = K2_len - 1;
                double y = K2_work[t];
                K2_work[t] = K2_work[t] + log_e;
                while (t > 0) {
                    t -= 1;
                    double x = y + log_e_c;
                    double new_y = K2_work[t];
                    double z = new_y + log_e;
                    y = new_y;
                    K2_work[t] = recount_logaddexp(x, z);
                }
            }

            double s_inv = (s == 0) ? 0.0
                : ((double)s * logp1 - recount_factln_get(fact, s));
            int cnt = 0;
            int t = 0, ell = s;
            while (t < K2_len && ell < Bu_len) {
                terms_buf[cnt++] = Bu_plus_fact[ell] + K2_work[t] + tterm[t] + s_inv;
                t++; ell++;
            }
            Jv[s] = recount_logsumexp(terms_buf, cnt);
        }
        recount_width_scratch_free(obl_buf, obl_stack);

        /* B[v] = edge_outside_to_node(v, Jv) */
        edge_outside_to_node(
            Jv, Jv_len, B_slots[v], widths[v],
            sp->log_q[v], sp->log_q_c[v], sp->log_gain[v], sp->is_polya[v],
            fact, rfacts ? rfacts[v] : NULL, terms_buf);
    }

    return RECOUNT_OK;
}
