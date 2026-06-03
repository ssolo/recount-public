/*
 * recount_unobserved_grad.c — full L(0)-corrected gradient pipeline when
 * min_copies ≥ 2. Native C port of recount/unobserved_outside.py.
 *
 * Current contents (post C-port phase 6, commit c2659f1 / 7fd3425):
 *   ✓ recount_unobserved_pairing       (compute_pairing_likelihoods)
 *   ✓ recount_unobserved_outside       (compute_unobserved_outside)
 *   ✓ recount_unobserved_posteriors    (compute_unobserved_posteriors)
 *   ✓ recount_unobserved_bd_tails      (compute_birth_death_tails)
 *   ✓ recount_unobserved_logsurv_grad  (compute_log_survival_gradient_unobs)
 *   ✓ recount_chain_rule_logit_to_rates (rate-domain chain rule, phase 6)
 *   - recount_L0_gradient_native       (still a Python orchestrator in
 *                                       unobserved_outside.compute_L0_gradient_analytical;
 *                                       all leaf kernels are native C above)
 *
 * Numerical conventions match the numpy reference exactly:
 *   - All tensors are log-probabilities; NEG_INF (= -INFINITY) is the
 *     "log of zero" sentinel.
 *   - Layout: row-major, "small dimensions last" (see header).
 *   - log-add: y + log1p(exp(x - y)) with the larger argument outside.
 */

#include "recount_native.h"
#include "recount_internal.h"
#include "recount_unobserved_grad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NEG_INF
#define NEG_INF (-INFINITY)
#endif

/* Log-add: log(exp(a) + exp(b)), numerically stable. */
static inline double logadd(double a, double b)
{
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    if (a > b) return a + log1p(exp(b - a));
    return b + log1p(exp(a - b));
}

/*
 * Pairing likelihoods L̃_w,m[ell, t] per node w (skip root).
 * Caller has already populated Lw_all with NEG_INF.
 *
 * Direct port of compute_pairing_likelihoods() in
 * recount/unobserved_outside.py (lines 136-167).
 *
 * For each m ∈ [0, W):
 *   For each ell ∈ [0, W):
 *     Start t = min(ell, m); x = K_w,m[t] (if t≤m) else x = NEG_INF, t = m+1
 *     Loop down t = t-1 ... 0:
 *       x ← x + log_pw_c  (handle NEG_INF cases)
 *       if ell ≥ 1: x ← logadd(x, Lw[m, ell-1, t] + log_pw)
 *       Lw[m, ell, t] = x
 */
int recount_unobserved_pairing(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *K_all,                 /* [N, W, W] (w, m, s) */
    int W,
    double *Lw_all)                       /* [N, W, W, W] (w, m, ell, t) */
{
    int N    = tree->num_nodes;
    int root = tree->root;
    int stride_w_m_ell_t = W * W * W;     /* per-w stride */
    int stride_m_ell_t   = W * W;         /* per-(w,m) stride */
    int stride_ell_t     = W;             /* per-(w,m,ell) stride */

    int stride_K_w  = W * W;              /* per-w stride in K_all */
    int stride_K_m  = W;                  /* per-(w,m) stride */

    /* Initialise all to NEG_INF. */
    size_t total = (size_t)N * W * W * W;
    for (size_t i = 0; i < total; i++) Lw_all[i] = NEG_INF;

    for (int w = 0; w < N; w++) {
        if (w == root) continue;
        double log_pw   = sp->log_p[w];
        double log_pw_c = sp->log_p_c[w];

        double *Lw_base = Lw_all + (size_t)w * stride_w_m_ell_t;
        const double *Kw_base = K_all + (size_t)w * stride_K_w;

        for (int m = 0; m < W; m++) {
            const double *Kw_m = Kw_base + (size_t)m * stride_K_m;  /* [s ≤ m] */
            double *Lw_m = Lw_base + (size_t)m * stride_m_ell_t;     /* [ell, t] */

            for (int ell = 0; ell < W; ell++) {
                double *Lw_m_ell = Lw_m + (size_t)ell * stride_ell_t; /* [t] */

                int t_start;
                double x;
                if (ell <= m) {
                    /* Initialise from K_w,m[ell] then iterate t = ell-1 ... 0. */
                    x = Kw_m[ell];
                    Lw_m_ell[ell] = x;
                    t_start = ell;
                } else {
                    /* Start with x = NEG_INF, t_start = m+1 (Python sets t = m+1
                     * but doesn't store at that position — the while loop
                     * immediately decrements t). */
                    x = NEG_INF;
                    t_start = m + 1;
                }

                int t = t_start;
                while (t > 0) {
                    t--;
                    /* x ← x + log_pw_c  (handle NEG_INF cases): */
                    if (x == NEG_INF) {
                        /* x stays NEG_INF */
                    } else if (log_pw_c == NEG_INF) {
                        x = NEG_INF;
                    } else {
                        x += log_pw_c;
                    }
                    /* y = Lw[m, ell-1, t] + log_pw (with NEG_INF guards): */
                    if (ell >= 1) {
                        double Lw_prev = Lw_m[(size_t)(ell - 1) * stride_ell_t + t];
                        double y;
                        if (Lw_prev == NEG_INF || log_pw == NEG_INF) {
                            y = NEG_INF;
                        } else {
                            y = Lw_prev + log_pw;
                        }
                        x = logadd(x, y);
                    }
                    Lw_m_ell[t] = x;
                }
            }
        }
    }
    return RECOUNT_OK;
}

/*
 * edge_outside_to_node — per-(m, v) helper that turns the EDGE-outside slice
 *   J_v,m[s]  (probability that η̃_v = s and complement-sum = M − m)
 * into the joint NODE-outside slice
 *   Bns_v,m[n, s] = J_v,m[s] + log P{ξ̃_v = n | η̃_v = s}
 * and the marginal B_v,m[n] = logsumexp over s.
 *
 * The gain PMF is node v's own (Pólya or Poisson, offset by s starting copies):
 *   Pólya:    P{gain t | start s} = NegBin(t; κ+s, 1-q)
 *   Poisson:  P{gain t | start s} = Poisson(t; r)
 * with t = n - s.
 *
 * For Pólya we use the recount_rfactln-style precise rising factorial
 *   lgamma(κ+s+t) − lgamma(κ+s) = Σ_{i=0..t-1} log(κ + s + i)
 * which avoids the catastrophic cancellation lgamma() suffers when κ is
 * huge (publication-regime κ values up to 5×10^13 violate the Java
 * Logistic(33) cap — see GAIN_CONSTRAINT.md). The naïve lgamma form loses
 * up to 2 nats with `-ffast-math` enabled. This change makes the formula
 * bit-tolerant to compiler aggressiveness.
 *
 * Port of _edge_outside_to_node in recount/unobserved_outside.py (which
 * was updated to use the same direct-sum form for matching semantics).
 */
static void edge_outside_to_node(
    const double *Jv,                    /* [W] J_all[m, v, :] */
    const recount_survival_t *sp,
    int v, int W,
    const double *fact,                  /* [W+4] lgamma(i+1) */
    double *Bns_out,                     /* [W * W] flat */
    double *B_out)                        /* [W] */
{
    int is_polya     = sp->is_polya[v];
    double log_q     = sp->log_q[v];
    double log_q_c   = sp->log_q_c[v];
    double log_gain  = sp->log_gain[v];
    double kappa     = sp->gain[v];
    double r         = (!is_polya && log_gain != NEG_INF) ? exp(log_gain) : 0.0;

    for (int i = 0; i < W * W; i++) Bns_out[i] = NEG_INF;

    if (log_gain == NEG_INF) {
        /* No gain on this edge: only s = n survives (t = 0). */
        for (int n = 0; n < W; n++) {
            double Jv_n = Jv[n];
            if (Jv_n != NEG_INF) Bns_out[n * W + n] = Jv_n;
        }
    } else if (!is_polya) {
        /* Poisson gain. */
        for (int n = 0; n < W; n++) {
            for (int s = 0; s <= n; s++) {
                double Jv_s = Jv[s];
                if (Jv_s == NEG_INF) continue;
                int t = n - s;
                double tlogr = (t == 0) ? 0.0 : (double)t * log_gain;
                Bns_out[n * W + s] = Jv_s - r + tlogr - fact[t];
            }
        }
    } else {
        /* Pólya gain. Rising factorial via direct sum (cancellation-safe). */
        for (int n = 0; n < W; n++) {
            for (int s = 0; s <= n; s++) {
                double Jv_s = Jv[s];
                if (Jv_s == NEG_INF) continue;
                int t = n - s;
                double rising = 0.0;
                for (int i = 0; i < t; i++) {
                    rising += log(kappa + (double)(s + i));
                }
                double binom = rising - fact[t];
                double ks_l1q;
                if (log_q_c == NEG_INF) {
                    /* q≈1: (κ+s)·log(1-q) = -∞; the slice collapses. */
                    ks_l1q = NEG_INF;
                } else {
                    ks_l1q = (kappa + (double)s) * log_q_c;
                }
                double tlogq = (t == 0) ? 0.0 : (double)t * log_q;
                Bns_out[n * W + s] = Jv_s + binom + ks_l1q + tlogq;
            }
        }
    }

    /* B[n] = logsumexp over s of Bns[n, :n+1]. */
    for (int n = 0; n < W; n++) {
        double mx = NEG_INF;
        for (int s = 0; s <= n; s++) {
            double x = Bns_out[n * W + s];
            if (x > mx) mx = x;
        }
        if (mx == NEG_INF) { B_out[n] = NEG_INF; continue; }
        double sum = 0.0;
        for (int s = 0; s <= n; s++) {
            double x = Bns_out[n * W + s];
            if (x != NEG_INF) sum += exp(x - mx);
        }
        B_out[n] = mx + log(sum);
    }
}

/*
 * recount_unobserved_outside — outside pass building B/J/Bns/Jns per-size
 * tensors. Mirrors compute_unobserved_outside() in
 * recount/unobserved_outside.py exactly (root-init → pre-order
 * Bu·Lw convolution → edge_outside_to_node → cumulate-in-m).
 *
 * Layout (W = min_copies):
 *   B_all[m, v, n]      flat  [W, N, W]
 *   J_all[m, v, s]      flat  [W, N, W]
 *   Bns_all[m, v, n, s] flat  [W, N, W, W]
 *   Jns_all[m, v, n, s] flat  [W, N, W, W]
 *   Lw_all_out[w, m, ell, t] flat [N, W, W, W]  (also computed here)
 *
 * Restrictions (match Python reference):
 *   - root must have p̃ = 1 (log_p_c == NEG_INF); otherwise ROOTLOSS, not
 *     supported (Java FSL throws too).
 *   - binary tree only (every non-leaf has exactly 2 children).
 */
int recount_unobserved_outside(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *K_all,
    int min_copies,
    double *B_all, double *J_all,
    double *Bns_all, double *Jns_all,
    double *Lw_all_out)
{
    if (min_copies < 1) return RECOUNT_EINVAL;
    int W = min_copies;
    int N = tree->num_nodes;
    int R = tree->root;

    /* lgamma table for log-factorials up to 2*W+8. */
    int FW = 2 * W + 8;
    double *fact = (double *)malloc(sizeof(double) * (size_t)FW);
    if (!fact) return RECOUNT_ENOMEM;
    fact[0] = 0.0;
    for (int i = 1; i < FW; i++) fact[i] = fact[i - 1] + log((double)i);

    /* Compute pairing likelihoods Lw_all_out[w, m, ell, t]. */
    int ret = recount_unobserved_pairing(tree, sp, K_all, W, Lw_all_out);
    if (ret != RECOUNT_OK) { free(fact); return ret; }

    /* Strides. */
    size_t s_BJ_m  = (size_t)N * W;        /* [W, N, W] m-stride */
    size_t s_BJ_v  = (size_t)W;            /* per-v */
    size_t s_BnJn_m = (size_t)N * W * W;   /* [W, N, W, W] m-stride */
    size_t s_BnJn_v = (size_t)W * W;       /* per-v */
    size_t s_Lw_w  = (size_t)W * W * W;    /* [N, W, W, W] w-stride */

    /* Initialise B/J/Bns/Jns to NEG_INF. */
    size_t total_BJ = (size_t)W * N * W;
    for (size_t i = 0; i < total_BJ; i++) {
        B_all[i] = NEG_INF;
        J_all[i] = NEG_INF;
    }
    size_t total_BnJn = (size_t)W * N * W * W;
    for (size_t i = 0; i < total_BnJn; i++) {
        Bns_all[i] = NEG_INF;
        Jns_all[i] = NEG_INF;
    }

    /* Root init: requires p̃_root = 1 (log_p_c == -∞). */
    if (sp->log_p_c[R] != NEG_INF) {
        free(fact);
        return RECOUNT_ENOTSUP;
    }
    /* Jns[0, R, 0, 0] = J[0, R, 0] = 0 (delta at m=s=0). */
    Jns_all[(size_t)0 * s_BnJn_m + (size_t)R * s_BnJn_v + 0 * W + 0] = 0.0;
    J_all[(size_t)0 * s_BJ_m + (size_t)R * s_BJ_v + 0] = 0.0;

    /* B[root, m] from J[root, m] for every m. */
    for (int m = 0; m < W; m++) {
        const double *Jv = J_all + (size_t)m * s_BJ_m + (size_t)R * s_BJ_v;
        double *Bns_dst = Bns_all + (size_t)m * s_BnJn_m + (size_t)R * s_BnJn_v;
        double *B_dst   = B_all + (size_t)m * s_BJ_m + (size_t)R * s_BJ_v;
        edge_outside_to_node(Jv, sp, R, W, fact, Bns_dst, B_dst);
    }

    /* Pre-order from root's children. */
    int *order = (int *)malloc((size_t)N * sizeof(int));
    int *stack = (int *)malloc((size_t)N * sizeof(int));
    if (!order || !stack) {
        free(order); free(stack); free(fact);
        return RECOUNT_ENOMEM;
    }
    int order_n = 0;
    int top = 0;
    stack[top++] = R;
    while (top > 0) {
        int u = stack[--top];
        int cstart = tree->first_child[u];
        int cend   = tree->first_child[u + 1];
        for (int i = cstart; i < cend; i++) {
            int c = tree->child_list[i];
            order[order_n++] = c;
            stack[top++] = c;
        }
    }
    free(stack);

    /* Walk in pre-order, fill outside tensors at each v. */
    for (int oi = 0; oi < order_n; oi++) {
        int v = order[oi];
        int u = tree->parent[v];

        /* Find sibling w (binary only). */
        int cstart = tree->first_child[u];
        int cend   = tree->first_child[u + 1];
        if (cend - cstart != 2) {
            free(order); free(fact);
            return RECOUNT_ENOTSUP;
        }
        int w = (tree->child_list[cstart] == v)
            ? tree->child_list[cstart + 1] : tree->child_list[cstart];

        /* logit_p = (log p̃_v - log(1-p̃_v)) + log(1 - p̃_w). */
        double logit_pv = sp->log_p[v] - sp->log_p_c[v];
        double log1_e   = sp->log_p_c[w];
        double logit_p  = logit_pv + log1_e;

        /* Stable softplus: log_p, log1_p from logit_p. */
        double log_p, log1_p;
        if (logit_p >= 0.0) {
            double expm = exp(-logit_p);
            double l1pm = log1p(expm);
            log_p  = -l1pm;
            log1_p = -logit_p - l1pm;
        } else {
            double expp = exp(logit_p);
            double l1pp = log1p(expp);
            log_p  = logit_p - l1pp;
            log1_p = -l1pp;
        }

        const double *Lw_base = Lw_all_out + (size_t)w * s_Lw_w;
        /* Lw_base[mw, n, t] = Lw_base[mw * W*W + n * W + t] */

        for (int m = 0; m < W; m++) {
            double *Jns_v = Jns_all + (size_t)m * s_BnJn_m + (size_t)v * s_BnJn_v;

            /* Jns_all[m, v, n, s] = (Σ_{mw≤m} B_all[m-mw, u, n] · Lw[w, mw, n, t])
             *                     · C(n, s) · (1-p̃)^s · p̃^t,   t = n - s
             */
            for (int n = 0; n < W; n++) {
                double lognf = fact[n];
                for (int s = 0; s <= n; s++) {
                    int t = n - s;
                    double BuLw = NEG_INF;
                    for (int mw = 0; mw <= m; mw++) {
                        int Bu_size = m - mw;
                        /* Bu_size ∈ [0, m] ⊂ [0, W-1] — always valid. */
                        double Bu_n = B_all[(size_t)Bu_size * s_BJ_m
                                            + (size_t)u * s_BJ_v + n];
                        if (Bu_n == NEG_INF) continue;
                        if (t > mw) continue;
                        double Lw_t = Lw_base[(size_t)mw * (W * W)
                                              + (size_t)n * W + t];
                        if (Lw_t == NEG_INF) continue;
                        BuLw = logadd(BuLw, Bu_n + Lw_t);
                    }
                    double binom   = lognf - fact[s] - fact[t];
                    double slog1_p = (s == 0) ? 0.0 : (double)s * log1_p;
                    double tlog_p  = (t == 0) ? 0.0 : (double)t * log_p;
                    Jns_v[n * W + s] = BuLw + binom + slog1_p + tlog_p;
                }
            }

            /* J_all[m, v, s] = logsumexp over n≥s of Jns_v[n, s]. */
            double *J_vm = J_all + (size_t)m * s_BJ_m + (size_t)v * s_BJ_v;
            for (int s = 0; s < W; s++) {
                double acc = NEG_INF;
                for (int n = s; n < W; n++) {
                    double val = Jns_v[n * W + s];
                    if (val != NEG_INF) acc = logadd(acc, val);
                }
                J_vm[s] = acc;
            }

            /* B_all[m, v, n], Bns_all[m, v, n, s] from J_vm via gain PMF. */
            double *Bns_dst = Bns_all + (size_t)m * s_BnJn_m
                                       + (size_t)v * s_BnJn_v;
            double *B_dst = B_all + (size_t)m * s_BJ_m + (size_t)v * s_BJ_v;
            edge_outside_to_node(J_vm, sp, v, W, fact, Bns_dst, B_dst);
        }
    }
    free(order);

    /* Cumulate in m: T[m, v, ...] += T[m-1, v, ...] (logadd). */
    for (int m = 1; m < W; m++) {
        for (int v = 0; v < N; v++) {
            double *Bm  = B_all + (size_t)m * s_BJ_m + (size_t)v * s_BJ_v;
            double *Bm1 = B_all + (size_t)(m - 1) * s_BJ_m + (size_t)v * s_BJ_v;
            double *Jm  = J_all + (size_t)m * s_BJ_m + (size_t)v * s_BJ_v;
            double *Jm1 = J_all + (size_t)(m - 1) * s_BJ_m + (size_t)v * s_BJ_v;
            for (int n = 0; n < W; n++) Bm[n] = logadd(Bm[n], Bm1[n]);
            for (int s = 0; s < W; s++) Jm[s] = logadd(Jm[s], Jm1[s]);
            double *Bnsm  = Bns_all + (size_t)m * s_BnJn_m
                                     + (size_t)v * s_BnJn_v;
            double *Bnsm1 = Bns_all + (size_t)(m - 1) * s_BnJn_m
                                     + (size_t)v * s_BnJn_v;
            double *Jnsm  = Jns_all + (size_t)m * s_BnJn_m
                                     + (size_t)v * s_BnJn_v;
            double *Jnsm1 = Jns_all + (size_t)(m - 1) * s_BnJn_m
                                     + (size_t)v * s_BnJn_v;
            int WW = W * W;
            for (int i = 0; i < WW; i++) {
                Bnsm[i] = logadd(Bnsm[i], Bnsm1[i]);
                Jnsm[i] = logadd(Jnsm[i], Jnsm1[i]);
            }
        }
    }

    free(fact);
    return RECOUNT_OK;
}

/*
 * L(0) scalar from cached inside (K) and outside (J) tensors. Equivalent
 * to compute_unobserved_log_likelihood() in unobserved_outside.py.
 */
int recount_unobserved_logL0_from_tensors(
    int root, int min_copies,
    const double *K_all,
    const double *J_all,
    int num_nodes,
    double *out_logL0)
{
    int W = min_copies, M = W - 1, N = num_nodes, R = root;
    /* J_all: [W, N, W]. K_all: [N, W, W]. */
    size_t Jm = (size_t)N * W;
    size_t Km = (size_t)W;
    double LL = NEG_INF;
    for (int m = 0; m <= M; m++) {
        const double *Krm = K_all + (size_t)R * W * W + (size_t)m * Km;
        const double *Jrow = J_all + (size_t)(M - m) * Jm + (size_t)R * W;
        for (int s = 0; s <= m; s++) {
            double Js = Jrow[s];
            double Ks = Krm[s];
            if (Js != NEG_INF && Ks != NEG_INF) LL = logadd(LL, Js + Ks);
        }
    }
    *out_logL0 = LL;
    return RECOUNT_OK;
}

/*
 * Per-node marginal posteriors: node ξ_v and edge η_v.
 * Mirrors compute_unobserved_posteriors() exactly (incl. (M-m) < W guard).
 */
int recount_unobserved_posteriors(
    int num_nodes, int min_copies,
    const double *C_all, const double *K_all,
    const double *B_all, const double *J_all,
    double log_L0,
    double *out_node_post, double *out_edge_post)
{
    int W = min_copies, M = W - 1, N = num_nodes;
    size_t BJm = (size_t)N * W;            /* [W, N, W] m-stride */
    size_t CKv = (size_t)W * W;            /* [N, W, W] v-stride */

    for (int v = 0; v < N; v++) {
        /* Edge posteriors */
        for (int s = 0; s < W; s++) {
            double z = NEG_INF;
            for (int m = s; m <= M; m++) {
                /* (M - m) < W is always true since m >= 0 → M-m ≤ M = W-1. */
                double Js = J_all[(size_t)(M - m) * BJm + (size_t)v * W + s];
                double Ks = K_all[(size_t)v * CKv + (size_t)m * W + s];
                if (Js != NEG_INF && Ks != NEG_INF) z = logadd(z, Js + Ks);
            }
            out_edge_post[(size_t)v * W + s] = z - log_L0;
        }
        /* Node posteriors */
        for (int n = 0; n < W; n++) {
            double z = NEG_INF;
            for (int m = n; m <= M; m++) {
                double Bn = B_all[(size_t)(M - m) * BJm + (size_t)v * W + n];
                double Cn = C_all[(size_t)v * CKv + (size_t)m * W + n];
                if (Bn != NEG_INF && Cn != NEG_INF) z = logadd(z, Bn + Cn);
            }
            out_node_post[(size_t)v * W + n] = z - log_L0;
        }
    }
    return RECOUNT_OK;
}

/*
 * Joint transition posteriors P{ξ_v=n, η_v=s | unobs} (node) and
 * P{ξ_u=n, η_v=s | unobs} (across edge entering v).
 * Mirrors compute_transition_posteriors() exactly.
 */
int recount_unobserved_transitions(
    int num_nodes, int min_copies,
    const double *C_all, const double *K_all,
    const double *Bns_all, const double *Jns_all,
    double log_L0,
    double *out_node_trans, double *out_edge_trans)
{
    int W = min_copies, M = W - 1, N = num_nodes;
    size_t BnJn_m = (size_t)N * W * W;      /* [W, N, W, W] m-stride */
    size_t BnJn_v = (size_t)W * W;          /* per-v */
    size_t CK_v   = (size_t)W * W;          /* [N, W, W] v-stride */

    /* Initialise outputs to NEG_INF (s > n entries stay NEG_INF). */
    for (size_t i = 0; i < (size_t)N * W * W; i++) {
        out_node_trans[i] = NEG_INF;
        out_edge_trans[i] = NEG_INF;
    }

    for (int v = 0; v < N; v++) {
        for (int n = 0; n < W; n++) {
            for (int s = 0; s <= n; s++) {
                /* Node transitions */
                double zn = NEG_INF;
                for (int m = n; m <= M; m++) {
                    double Bnsm = Bns_all[(size_t)(M - m) * BnJn_m
                                          + (size_t)v * BnJn_v + n * W + s];
                    double Cn = C_all[(size_t)v * CK_v + (size_t)m * W + n];
                    if (Bnsm != NEG_INF && Cn != NEG_INF)
                        zn = logadd(zn, Bnsm + Cn);
                }
                out_node_trans[(size_t)v * (W * W) + n * W + s] = zn - log_L0;

                /* Edge transitions */
                double ze = NEG_INF;
                for (int m = s; m <= M; m++) {
                    double Jnsm = Jns_all[(size_t)(M - m) * BnJn_m
                                          + (size_t)v * BnJn_v + n * W + s];
                    double Ks = K_all[(size_t)v * CK_v + (size_t)m * W + s];
                    if (Jnsm != NEG_INF && Ks != NEG_INF)
                        ze = logadd(ze, Jnsm + Ks);
                }
                out_edge_trans[(size_t)v * (W * W) + n * W + s] = ze - log_L0;
            }
        }
    }
    return RECOUNT_OK;
}

/*
 * Per-node birth/death tail differences. For a [W, W] joint posterior
 * log_trans_v[n, s] (s ≤ n), produces N_S[ell] = log Σ_s Σ_{n > ell}
 * exp(log_trans_v[n, s]). Mirrors log_tail_difference() and the loop in
 * compute_birth_death_tails().
 */
static void log_tail_difference_node(
    const double *log_trans_v,            /* [W, W] (n, s) */
    int W,
    double *N_S_out)                       /* [W] */
{
    for (int ell = 0; ell < W; ell++) N_S_out[ell] = NEG_INF;
    for (int s = 0; s < W; s++) {
        double tail = NEG_INF;
        for (int ell = W - 1; ell >= s; ell--) {
            N_S_out[ell] = logadd(N_S_out[ell], tail);
            if (s <= ell) {
                tail = logadd(tail, log_trans_v[(size_t)ell * W + s]);
            }
        }
    }
}

int recount_unobserved_bd_tails(
    int num_nodes, int min_copies,
    const double *log_node_trans,
    const double *log_edge_trans,
    double *out_birth, double *out_death)
{
    int W = min_copies, N = num_nodes;
    for (int v = 0; v < N; v++) {
        log_tail_difference_node(
            log_node_trans + (size_t)v * (W * W), W,
            out_birth + (size_t)v * W);
        log_tail_difference_node(
            log_edge_trans + (size_t)v * (W * W), W,
            out_death + (size_t)v * W);
    }
    return RECOUNT_OK;
}

/* logsumexp of an array of W doubles (-inf inputs skipped). */
static inline double log_sum_arr(const double *xs, int W)
{
    double mx = NEG_INF;
    for (int i = 0; i < W; i++) {
        if (xs[i] > mx) mx = xs[i];
    }
    if (mx == NEG_INF) return NEG_INF;
    double sum = 0.0;
    for (int i = 0; i < W; i++) {
        if (xs[i] != NEG_INF) sum += exp(xs[i] - mx);
    }
    return mx + log(sum);
}

/* exp(x) treating non-finite as 0 (matches the np.isfinite gate in the
 * Python reference, where NaN/Inf positive/negative are all dropped). */
static inline double exp_or_zero(double x)
{
    return isfinite(x) ? exp(x) : 0.0;
}

/*
 * Log-survival gradient for the unobserved profile (compute_log_survival_
 * gradient_unobs in unobserved_outside.py). Per-node computation; no
 * inter-node recursion.
 */
int recount_unobserved_logsurv_grad(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    int min_copies,
    const double *log_edge_post,
    const double *log_birth_tails,
    const double *log_death_tails,
    double profile_count,
    double *out_d_logit_p,
    double *out_d_logit_q,
    double *out_d_log_kappa)
{
    int W = min_copies, N = tree->num_nodes, root = tree->root;
    double logF = (profile_count > 0.0) ? log(profile_count) : NEG_INF;

    /* Stack-allocated per-node tail buffer for E[η_v]. W is small (≤ 10
     * in practice). */
    double pSv[64];
    if (W > 64) return RECOUNT_EINVAL;

    for (int v = 0; v < N; v++) {
        /* log_edge_post[v, s] → pSv[j] then cumulative tail trick.
         * After the loop, pSv[j] = log P{η_v > j}. logSv = log E[η_v]. */
        const double *epv = log_edge_post + (size_t)v * W;
        for (int j = 0; j < W; j++) pSv[j] = epv[j];
        double log_tail = NEG_INF;
        for (int j = W - 1; j >= 0; j--) {
            double x = pSv[j];
            pSv[j] = log_tail;
            log_tail = logadd(log_tail, x);
        }
        double logSv = log_sum_arr(pSv, W);

        /* ---- Loss gradient d_logit_p[v] ---- */
        double logp_v  = sp->log_p[v];
        double log1_pv = sp->log_p_c[v];

        if (log1_pv == NEG_INF) {
            /* p̃ = 1 (root or extinct edge): no loss gradient */
            out_d_logit_p[v] = 0.0;
        } else {
            const double *tNu_Sv = log_death_tails + (size_t)v * W;
            double logNu_Sv = log_sum_arr(tNu_Sv, W);
            double dpos, dneg;
            if (v == root) {
                /* ROOTLOSS — kept for completeness; standard tree never
                 * enters this branch. */
                dpos = logNu_Sv + log1_pv;
                dneg = logSv + logp_v;
            } else {
                int u = tree->parent[v];
                int cstart = tree->first_child[u];
                int cend   = tree->first_child[u + 1];
                int num_kids = cend - cstart;
                double log1_e;
                if (num_kids == 2) {
                    int sib = (tree->child_list[cstart] == v)
                        ? tree->child_list[cstart + 1]
                        : tree->child_list[cstart];
                    log1_e = sp->log_p_c[sib];
                } else {
                    /* Multifurcation: eps_excl_v = eps_u / p̃_v. */
                    double eps_u = sp->eps[u];
                    double log_eps_u = (eps_u > 0.0) ? log(eps_u) : NEG_INF;
                    double diff = log_eps_u - logp_v;
                    if (diff >= 0.0) {
                        log1_e = NEG_INF;
                    } else if (diff < -1e-3) {
                        log1_e = log1p(-exp(diff));
                    } else {
                        log1_e = log(-expm1(diff));
                    }
                }
                /* log1_pe = logadd(log1_pv, logp_v + log1_e) */
                double inner = (log1_e == NEG_INF) ? NEG_INF : (logp_v + log1_e);
                double log1_pe = logadd(log1_pv, inner);
                dpos = logNu_Sv + log1_pv - log1_pe;
                dneg = logSv + logp_v + log1_e - log1_pe;
            }
            out_d_logit_p[v] = exp_or_zero(dpos) - exp_or_zero(dneg);
        }

        /* ---- Gain & dup gradients ---- */
        double log_q_v   = sp->log_q[v];
        double log1_q_v  = sp->log_q_c[v];
        double kappa_v   = sp->gain[v];
        double log_kappa_v = sp->log_gain[v];
        const double *tNv_Sv = log_birth_tails + (size_t)v * W;
        double logNv_Sv = log_sum_arr(tNv_Sv, W);

        if (log_q_v < -500.0) {
            /* Poisson (q ≈ 0) */
            if (log_kappa_v < -500.0) {
                out_d_log_kappa[v] = 0.0;
            } else {
                double dpos = logNv_Sv;
                double dneg = log_kappa_v + logF;
                out_d_log_kappa[v] = exp_or_zero(dpos) - exp_or_zero(dneg);
            }
            out_d_logit_q[v] = 0.0;
        } else {
            /* Pólya */
            if (log_kappa_v < -500.0) {
                out_d_log_kappa[v] = 0.0;
            } else {
                /* κ-grad with harmonic-sum term */
                double dpos = tNv_Sv[0];
                for (int i = 1; i < W; i++) {
                    double log_k_ki;
                    if ((double)i < kappa_v) {
                        log_k_ki = -log1p((double)i / kappa_v);
                    } else {
                        log_k_ki = log_kappa_v - log((double)i)
                                 - log1p(kappa_v / (double)i);
                    }
                    if (isfinite(tNv_Sv[i])) {
                        dpos = logadd(dpos, tNv_Sv[i] + log_k_ki);
                    }
                }
                /* loglog1_q with Java fallback */
                double loglog1_q;
                if (log1_q_v < 0.0) {
                    loglog1_q = log(-log1_q_v);
                    if (loglog1_q == NEG_INF) loglog1_q = log_q_v;
                } else {
                    loglog1_q = log_q_v;
                }
                double log_kappa_log1_q = log_kappa_v + loglog1_q;
                double dneg = log_kappa_log1_q + logF;
                out_d_log_kappa[v] = exp_or_zero(dpos) - exp_or_zero(dneg);
            }
            /* dup */
            double dpos_q = logNv_Sv + log1_q_v;
            double dneg_q = logadd(logSv, log_kappa_v + logF) + log_q_v;
            out_d_logit_q[v] = exp_or_zero(dpos_q) - exp_or_zero(dneg_q);
        }
    }
    return RECOUNT_OK;
}

/* Defined out-of-line in recount_rates.c. */
extern void recount_rate_to_pq(double mu, double lam, double t,
                               double *p, double *p_c, double *q, double *q_c);
extern void recount_rate_to_pq_jac(double mu, double lam, double t,
                                   double *dp_dmu, double *dp_dlam, double *dp_dt,
                                   double *dq_dmu, double *dq_dlam, double *dq_dt);

/* Safe divide returning 0 when the denominator is non-positive or the
 * numerator is non-finite. Mirrors the _safediv helper in Python. */
static inline double safediv(double num, double den)
{
    if (den > 0.0 && isfinite(num)) return num / den;
    return 0.0;
}

/*
 * Reverse-mode chain rule from (logit p̃, logit q̃, log κ) to (g, l, d, t).
 * Mirrors recount/chain_rule_unobs.py::chain_rule_logit_to_rates line-for-line.
 *
 * Forward pass (post-order): recompute per-edge p_raw, q_raw, q_raw_c
 *   (the sp struct already has eps, eps_c, p̃, q̃, p̃_c, q̃_c, log_gain).
 * Initial adjoints: adj_p_t = cdp / p̃, adj_p_t_c = -cdp / p̃_c, etc.
 * Reverse pass (reverse post-order, root → leaves):
 *   1. Reverse the survival recurrence to get adj on (p_raw, p_raw_c,
 *      q_raw, q_raw_c, eps, eps_c).
 *   2. Add gain adjoint contribution (Pólya: gain_t = gain; Poisson:
 *      gain_t = gain · eps_c).
 *   3. Collapse adj_eps_c into adj_eps via eps_c = 1 - eps.
 *   4. Propagate adj_eps to children's adj_p_t.
 *   5. Use rate_to_pq_jac to convert adj on (p_raw, q_raw) to adj on
 *      (loss, dup, length).
 */
int recount_chain_rule_logit_to_rates(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *gain, const double *loss,
    const double *dup, const double *length,
    const double *d_logit_p, const double *d_logit_q,
    const double *d_log_kappa,
    double *out_d_gain, double *out_d_loss,
    double *out_d_dup, double *out_d_length)
{
    int N = tree->num_nodes;

    /* Per-node scratch (p_raw, p_raw_c, q_raw, q_raw_c). */
    double *p_r  = (double *)malloc((size_t)N * sizeof(double));
    double *p_rc = (double *)malloc((size_t)N * sizeof(double));
    double *q_r  = (double *)malloc((size_t)N * sizeof(double));
    double *q_rc = (double *)malloc((size_t)N * sizeof(double));
    double *adj_p_t   = (double *)calloc((size_t)N, sizeof(double));
    double *adj_p_t_c = (double *)calloc((size_t)N, sizeof(double));
    double *adj_q_t   = (double *)calloc((size_t)N, sizeof(double));
    double *adj_q_t_c = (double *)calloc((size_t)N, sizeof(double));
    if (!p_r || !p_rc || !q_r || !q_rc ||
        !adj_p_t || !adj_p_t_c || !adj_q_t || !adj_q_t_c) {
        free(p_r); free(p_rc); free(q_r); free(q_rc);
        free(adj_p_t); free(adj_p_t_c); free(adj_q_t); free(adj_q_t_c);
        return RECOUNT_ENOMEM;
    }

    /* Forward: recompute raw (p, p_c, q, q_c) per node. */
    for (int v = 0; v < N; v++) {
        recount_rate_to_pq(loss[v], dup[v], length[v],
                           &p_r[v], &p_rc[v], &q_r[v], &q_rc[v]);
    }

    /* Initial adjoints from (d_logit_p, d_logit_q) on (p̃, q̃, p̃_c, q̃_c). */
    for (int v = 0; v < N; v++) {
        double cdp = isfinite(d_logit_p[v])  ? d_logit_p[v]  : 0.0;
        double cdq = isfinite(d_logit_q[v])  ? d_logit_q[v]  : 0.0;
        adj_p_t[v]   = safediv( cdp, sp->p[v]);
        adj_p_t_c[v] = safediv(-cdp, sp->p_c[v]);
        adj_q_t[v]   = safediv( cdq, sp->q[v]);
        adj_q_t_c[v] = safediv(-cdq, sp->q_c[v]);
    }

    for (int v = 0; v < N; v++) {
        out_d_gain[v]   = 0.0;
        out_d_loss[v]   = 0.0;
        out_d_dup[v]    = 0.0;
        out_d_length[v] = 0.0;
    }

    /* Reverse pass: root (v=N-1) → leaves (v=0). Convention: parent[v] > v. */
    for (int v = N - 1; v >= 0; v--) {
        double e = sp->eps[v], e_c = sp->eps_c[v];
        double pt = sp->p[v], ptc = sp->p_c[v];
        double qt = sp->q[v], qtc = sp->q_c[v];
        double prv = p_r[v], prcv = p_rc[v];
        double qrv = q_r[v], qrcv = q_rc[v];
        double a = qrcv * e + e_c;

        double ap_t   = adj_p_t[v];
        double aq_t   = adj_q_t[v];
        double ap_tc  = adj_p_t_c[v];
        double aq_tc  = adj_q_t_c[v];

        /* Per-output partials → per-input adjoints. */
        double inv_a   = (a > 0.0) ? 1.0 / a : 0.0;
        double inv_a2  = inv_a * inv_a;

        double adj_p_r  = ap_t * (e_c * inv_a);
        double adj_p_rc = ap_tc * (e_c * inv_a);
        double adj_q_r  = aq_t * (e_c * inv_a);
        double adj_q_rc = (
              ap_t  * (e * ptc * inv_a)
            + ap_tc * (-ptc * e * inv_a)
            + aq_t  * (-qt * e * inv_a)
            + aq_tc * (e_c * inv_a2)
        );

        double adj_e = (
              (qrcv * inv_a) * (ap_t * ptc - aq_t * qt - ap_tc * ptc)
            + aq_tc * (-qtc * qrcv * inv_a)
        );
        double adj_ec = inv_a * (
              ap_t  * (prv  - pt)
            + aq_t  * (qrv  - qt)
            + ap_tc * (prcv - ptc)
            - aq_tc * qtc
        );

        /* Gain. */
        double gain_t_v = sp->is_polya[v] ? gain[v] : gain[v] * e_c;
        double adj_gt = (gain_t_v > 0.0 && isfinite(d_log_kappa[v]))
                        ? d_log_kappa[v] / gain_t_v : 0.0;
        if (sp->is_polya[v]) {
            out_d_gain[v] = adj_gt;
        } else {
            out_d_gain[v] = adj_gt * e_c;
            adj_ec += adj_gt * gain[v];
        }

        /* Collapse adj_eps_c → adj_eps via eps_c = 1 - eps. */
        adj_e += -adj_ec;

        /* Propagate adj_eps to children's adj_p_t: eps[v] = ∏_c p_t[c]. */
        if (!tree->is_leaf[v]) {
            int cstart = tree->first_child[v];
            int cend   = tree->first_child[v + 1];
            for (int i = cstart; i < cend; i++) {
                int c = tree->child_list[i];
                double pt_c = sp->p[c];
                if (pt_c > 0.0) {
                    adj_p_t[c] += adj_e * (e / pt_c);
                } else {
                    /* Leave-one-out product (rare; p_t[c] = 0). */
                    double prod_sib = 1.0;
                    for (int j = cstart; j < cend; j++) {
                        int c2 = tree->child_list[j];
                        if (c2 != c) prod_sib *= sp->p[c2];
                    }
                    adj_p_t[c] += adj_e * prod_sib;
                }
            }
        }

        /* (p_raw, q_raw) adj → (loss, dup, length) adj. */
        double adj_p_eff = adj_p_r - adj_p_rc;
        double adj_q_eff = adj_q_r - adj_q_rc;
        double dp_dmu, dp_dlam, dp_dt, dq_dmu, dq_dlam, dq_dt;
        recount_rate_to_pq_jac(loss[v], dup[v], length[v],
                               &dp_dmu, &dp_dlam, &dp_dt,
                               &dq_dmu, &dq_dlam, &dq_dt);
        out_d_loss[v]   = adj_p_eff * dp_dmu  + adj_q_eff * dq_dmu;
        out_d_dup[v]    = adj_p_eff * dp_dlam + adj_q_eff * dq_dlam;
        out_d_length[v] = adj_p_eff * dp_dt   + adj_q_eff * dq_dt;
    }

    free(p_r); free(p_rc); free(q_r); free(q_rc);
    free(adj_p_t); free(adj_p_t_c); free(adj_q_t); free(adj_q_t_c);
    return RECOUNT_OK;
}

int recount_L0_gradient_native(
    const recount_tree_t *tree,
    const double *gain, const double *loss,
    const double *dup,  const double *length,
    int min_copies, int F,
    double *out_g, double *out_l, double *out_d, double *out_t,
    double *out_L0)
{
    /* Not yet implemented — see TODO. */
    (void)tree; (void)gain; (void)loss; (void)dup; (void)length;
    (void)min_copies; (void)F;
    (void)out_g; (void)out_l; (void)out_d; (void)out_t; (void)out_L0;
    return RECOUNT_ENOTSUP;
}
