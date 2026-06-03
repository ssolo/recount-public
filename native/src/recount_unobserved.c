/* recount_unobserved.c — Csurös 2026 Theorems 3-5 (arbitrary Ωmin correction).
 *
 * Implements algorithms Uinner/Uouter/unobservedNodePosteriors from
 * the SI of the 2026 PNAS paper (Figs S11-S13), enabling exact
 * observation-bias correction L(0) = P{Ω_R ≤ M} = P{profile sum < Ωmin}
 * for arbitrary Ωmin ≥ 1.
 *
 * The shipped CountXXV.jar caps Ωmin ≤ 2 (Gradient.java line 57:
 *   this.min_copies = Integer.min(2, table.minCopies())
 * ); recount native goes to arbitrary M via the SI algorithms.
 *
 * Per-node storage indexed by (m, n) ∈ [0, M] × [0, M]:
 *   C̃_u,m(n) — log P{Ω_u = m | ξ̃_u = n}
 *   K̃_u,m(s) — log P{Ω_u = m | η̃_u = s}, derived from C̃ via the edge step
 *
 * Per-edge tensor indexed by (n, t) ∈ [0, M] × [0, n]:
 *   L̃_v,m[n][t] — used for combining child contributions at the parent
 *
 * For Ωmin=4 (M=3), per node it's 4×4=16 doubles for C̃, K̃ each; per edge
 * 4×4×4=64 for L̃. Trivial memory.
 *
 * This file currently implements ONLY the L(0) computation (Uinner + a
 * root recurrence to extract Σ_m P{Ω_R = m}). The gradient terms
 * (B̃, J̃, δ⁻, δ⁺, σ) follow as separate function in a later commit.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "recount_native.h"
#include "recount_internal.h"

/* ladd(x, y) = log(exp(x) + exp(y)), safe for ±inf */
static inline double ladd(double x, double y) {
    return recount_logaddexp(x, y);
}

/* Multiply a log-space quantity by a non-negative integer, using Csurös'
 * convention `0 · (-∞) = 0` (the term carries zero weight in the
 * surrounding ladd). IEEE returns NaN for 0 · (-∞), so this needs an
 * explicit guard.
 */
static inline double mul_log_by_count(double log_x, int n) {
    if (n == 0) return 0.0;
    if (log_x == NEG_INF) return NEG_INF;
    return (double)n * log_x;
}

/* Forward-declared static helpers */
static void Uinner(
    const recount_tree_t *tree, const recount_survival_t *sp, int u,
    int M, const recount_factln_t *fact, const recount_rfactln_t * const *rfacts,
    double *Cmat,    /* [num_nodes * (M+1) * (M+1)] row-major */
    double *Kmat,    /* [num_nodes * (M+1) * (M+1)] row-major */
    double *Ltens);  /* [num_nodes * (M+1) * (M+1) * (M+1)] row-major */

/* Compute log L(0) = log P{Ω_R ≤ M} = log P{profile sum < Ωmin}.
 * Returns RECOUNT_OK on success; stores result in *out_logL0.
 *
 * For Ωmin == 0 (no correction): returns -∞.
 * For Ωmin == 1: equivalent to empty_log_likelihood (P{all zero}).
 * For Ωmin == 2: empty + singleton.
 * For Ωmin >= 3: SI Theorems 3-5 (general case).
 */
/* Forward declaration for the shared inside-pass implementation. */
static int unobserved_inside_impl(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies, double *out_Cmat, double *out_Kmat, double *out_logL0);

int recount_unobserved_logL0(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies, double *out_logL0)
{
    return unobserved_inside_impl(tree, sp, min_copies, NULL, NULL, out_logL0);
}

int recount_unobserved_inside_tensors(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies,
    double *out_Cmat, double *out_Kmat,
    double *out_logL0)
{
    if (out_Cmat == NULL || out_Kmat == NULL) return RECOUNT_EINVAL;
    double dummy;
    return unobserved_inside_impl(
        tree, sp, min_copies, out_Cmat, out_Kmat,
        out_logL0 ? out_logL0 : &dummy);
}

static int unobserved_inside_impl(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies, double *out_Cmat, double *out_Kmat, double *out_logL0)
{
    if (min_copies < 0) return RECOUNT_EINVAL;
    if (min_copies == 0) {
        if (out_logL0) *out_logL0 = NEG_INF;
        /* In the Ωmin=0 case the caller-provided buffers stay untouched —
         * the convention there is "no correction" and the tensors are
         * undefined. */
        return RECOUNT_OK;
    }

    int M = min_copies - 1;  /* threshold: profile sum can be 0..M */
    int N = tree->num_nodes;
    int W = M + 1;

    /* Allocate factorial caches (small: up to M ≤ a few) */
    recount_factln_t fact;
    if (recount_factln_init(&fact, M + 4) != RECOUNT_OK) return RECOUNT_ENOMEM;
    /* rfacts per Pólya node, sized for offset κ, max_n = M */
    recount_rfactln_t **rfacts = (recount_rfactln_t **)calloc((size_t)N, sizeof(*rfacts));
    if (!rfacts) { recount_factln_free(&fact); return RECOUNT_ENOMEM; }
    for (int v = 0; v < N; v++) {
        if (sp->is_polya[v] && sp->gain[v] > 0.0) {
            rfacts[v] = (recount_rfactln_t *)malloc(sizeof(recount_rfactln_t));
            if (!rfacts[v] || recount_rfactln_init(rfacts[v], sp->gain[v], M + 4) != RECOUNT_OK) {
                for (int u = 0; u <= v; u++) {
                    if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
                }
                free(rfacts); recount_factln_free(&fact);
                return RECOUNT_ENOMEM;
            }
        }
    }

    /* Allocate C̃, K̃ [N × W × W] and L̃ [N × W × W × W]; everything fits
     * in tens of KB for Ωmin up to ~8. */
    size_t Csize = (size_t)N * W * W;
    size_t Lsize = (size_t)N * W * W * W;
    double *Cmat = (double *)malloc(Csize * sizeof(double));
    double *Kmat = (double *)malloc(Csize * sizeof(double));
    double *Ltens = (double *)malloc(Lsize * sizeof(double));
    if (!Cmat || !Kmat || !Ltens) {
        free(Cmat); free(Kmat); free(Ltens);
        for (int u = 0; u < N; u++) {
            if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
        }
        free(rfacts); recount_factln_free(&fact);
        return RECOUNT_ENOMEM;
    }
    /* Init to -inf (= log 0) */
    for (size_t i = 0; i < Csize; i++) Cmat[i] = NEG_INF;
    for (size_t i = 0; i < Csize; i++) Kmat[i] = NEG_INF;
    for (size_t i = 0; i < Lsize; i++) Ltens[i] = NEG_INF;

    /* Postorder pass: u = 0, 1, ..., N-1 (recount convention has parent > v) */
    for (int u = 0; u < N; u++) {
        Uinner(tree, sp, u, M, &fact, (const recount_rfactln_t * const *)rfacts,
               Cmat, Kmat, Ltens);
    }

    /* At root: L(0) = Σ_m P{Ω_R = m} = Σ_m Σ_n P{Ω_R = m | ξ̃_R = n} × P{ξ̃_R = n}
     * where P{ξ̃_R = n} = the root prior for surviving copies n.
     *
     * Under the standard convention (root edge length = +∞ → p_root = 1),
     * the root prior is degenerate at n = 0 (all copies have already had
     * the root-edge BD process applied via the survival params; the prior
     * over root surviving copies is Poisson(r̃_R) or Pólya(κ_R, q̃_R) per
     * Theorem 2 with no parent edge contribution).
     *
     * Specifically: P{ξ̃_R = n | η̃_R = 0} from Theorem 2's edge-gain PMF
     * applied at the root with starting state 0. So:
     *   P{ξ̃_R = n} = exp(K̃_R,n(0))  if we interpret the root's K̃ at s=0
     *   ...
     *
     * Actually the root logic in the algorithms is simpler: K̃_R,m(0) is the
     * log probability that the root has ξ̃_R producing Ω_R = m total leaf
     * copies, started from 0 ancestral copies (since the root's parent edge
     * is implicit). So:
     *
     *   L(0) = Σ_{m=0..M} exp(K̃_R,m(0))
     */
    int R = tree->root;
    double logL0 = NEG_INF;
    for (int m = 0; m <= M; m++) {
        double Kval = Kmat[((size_t)R * W + m) * W + 0];  /* K̃_R,m(s=0) */
        logL0 = ladd(logL0, Kval);
    }
    *out_logL0 = logL0;

    /* If the caller asked for the per-size inside tensors, copy them out
     * before freeing. L̃ is intermediate scratch and is not exposed (it
     * can be recomputed from K̃ if ever needed by an outside-pass caller). */
    if (out_Cmat) memcpy(out_Cmat, Cmat, Csize * sizeof(double));
    if (out_Kmat) memcpy(out_Kmat, Kmat, Csize * sizeof(double));

    free(Cmat); free(Kmat); free(Ltens);
    for (int u = 0; u < N; u++) {
        if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
    }
    free(rfacts); recount_factln_free(&fact);
    return RECOUNT_OK;
}

/* Index helpers */
static inline size_t Cidx(int u, int m, int n, int W) {
    return ((size_t)u * W + m) * W + n;
}
static inline size_t Kidx(int u, int m, int s, int W) {
    return ((size_t)u * W + m) * W + s;
}
static inline size_t Lidx(int u, int m, int n, int t, int W) {
    return (((size_t)u * W + m) * W + n) * W + t;
}

/* logit_p_minus = logit(p̃_v) + log(1 - p̃_w)
 *               = log(p̃_v) - log(1 - p̃_v) + log(1 - p̃_w)
 *               = log(p̃_v · (1 - p̃_w)) - log(1 - p̃_v)
 *               = log(p̃_v · (1 - p̃_w) / (1 - p̃_v · p̃_w)) - log((1 - p̃_v · p̃_w)/(1 - p̃_v))
 * Per Csurös' formula. We need ln(p⁻_v) and ln(1 - p⁻_v).
 *   p⁻_v = p̃_v · (1 - p̃_w) / (1 - p̃_v · p̃_w)
 *   1 - p⁻_v = (1 - p̃_v) / (1 - p̃_v · p̃_w)
 *
 * Returns (z0, z1) = (ln p⁻_v, ln(1 - p⁻_v)).
 */
static void compute_p_minus_logs(double p_v, double pc_v, double p_w, double pc_w,
                                 double *z0, double *z1)
{
    /* Numerator and denominators in linear space.
     * 1 - p̃_v · p̃_w is the "1 - eps for this pair"; use the stable form
     * via (1 - p̃_v) + p̃_v · (1 - p̃_w). */
    double denom_complement = pc_v + p_v * pc_w;  /* = 1 - p_v·p_w, stable */
    if (denom_complement < 1e-300) denom_complement = 1e-300;
    double pminus = (p_v * pc_w) / denom_complement;
    double pminus_c = pc_v / denom_complement;
    *z0 = pminus > 0.0 ? log(pminus) : NEG_INF;
    *z1 = pminus_c > 0.0 ? log(pminus_c) : NEG_INF;
}

/* Implementation of Algorithm Uinner (Fig S11) for one node u.
 * Computes:
 *   C̃_u,m[n] for m, n ∈ [0, M]
 *   K̃_u,m[s] for m, s ∈ [0, M]
 *   L̃_v,m[n][t] for each child v (used by the PARENT's combine, but we
 *               compute it here because L̃ depends only on p̃_v and K̃_v)
 */
static void Uinner(
    const recount_tree_t *tree, const recount_survival_t *sp, int u,
    int M, const recount_factln_t *fact, const recount_rfactln_t * const *rfacts,
    double *Cmat, double *Kmat, double *Ltens)
{
    int W = M + 1;

    /* ---- Step 1: compute C̃_u,m[n] ---- */
    if (tree->is_leaf[u]) {
        /* Leaf: P{Ω_u = m | ξ̃_u = n} = 1 if n == m, 0 otherwise (Ω_leaf = ξ̃_leaf). */
        for (int m = 0; m <= M; m++) {
            for (int n = 0; n <= M; n++) {
                Cmat[Cidx(u, m, n, W)] = (n == m) ? 0.0 : NEG_INF;
            }
        }
    } else {
        /* Internal node with two children v, w. */
        int cs = tree->first_child[u], ce = tree->first_child[u + 1];
        int n_children = ce - cs;
        if (n_children != 2) {
            /* Multifurcations not yet supported in unobserved-profiles path */
            return;
        }
        int v = tree->child_list[cs];
        int w = tree->child_list[cs + 1];

        double p_v = sp->p[v], pc_v = sp->p_c[v];
        double p_w = sp->p[w], pc_w = sp->p_c[w];
        double z0, z1;
        compute_p_minus_logs(p_v, pc_v, p_w, pc_w, &z0, &z1);

        for (int m = 0; m <= M; m++) {
            for (int n = 0; n <= m; n++) {
                double Cval = NEG_INF;
                /* I8-I13: s walks from n down to 0; t walks 0 up to n */
                int s = n, t = 0;
                while (s >= 0) {
                    /* inner sum over k = t, t+1, ..., m-s */
                    double x = NEG_INF;
                    for (int k = t; k <= m - s; k++) {
                        double Kv  = Kmat[Kidx(v, m - k, s, W)];
                        double Lwk = Ltens[Lidx(w, k, n, t, W)];
                        x = ladd(x, Kv + Lwk);
                    }
                    /* binom(n, s) + z1 * s + z0 * t  — guard 0 · (-∞) */
                    double binom = recount_factln_get(fact, n)
                                 - recount_factln_get(fact, s)
                                 - recount_factln_get(fact, n - s);
                    Cval = ladd(Cval, x + binom
                                + mul_log_by_count(z1, s)
                                + mul_log_by_count(z0, t));
                    s--; t++;
                }
                Cmat[Cidx(u, m, n, W)] = Cval;
            }
        }
    }

    /* ---- Step 2: compute K̃_u,m[s] from C̃_u,m[n] via the edge step ----
     * K̃_u,m[s] = log P{Ω_u = m | η̃_u = s}, which is the convolution of
     * P{Ω_u = m | ξ̃_u = n} (= C̃) with P{ξ̃_u = n | η̃_u = s} (= the gain
     * PMF on the edge entering u).
     *
     * Range: at the root we only need K̃[R,m][0]; otherwise s ∈ [0, m].
     */
    int s_max = (u == tree->root) ? 0 : M;
    int is_polya = sp->is_polya[u];
    double log_q   = sp->log_q[u];
    double log_q_c = sp->log_q_c[u];
    double log_gain = sp->log_gain[u];
    double r_u = (!is_polya && log_gain != NEG_INF) ? exp(log_gain) : 0.0;
    /* For Pólya: precompute (s + κ) log(1 - q) where κ = sp->gain[u] */
    double kappa = sp->gain[u];

    for (int m = 0; m <= M; m++) {
        int s_lo = 0;
        int s_hi = m < s_max ? m : s_max;
        for (int s = s_lo; s <= s_hi; s++) {
            /* Sum over t = 0..m-s, n = s + t */
            double Kval = NEG_INF;
            int n = s;
            for (int t = 0; n <= m; t++, n++) {
                double Cv = Cmat[Cidx(u, m, n, W)];
                if (Cv == NEG_INF) continue;
                double logpmf;
                if (!is_polya) {
                    /* Poisson(r̃) on the edge: P{ξ̃ = n | η̃ = s} requires
                     * t = n - s extra copies via Poisson(r̃). */
                    if (log_gain == NEG_INF) {
                        logpmf = (t == 0) ? 0.0 : NEG_INF;  /* no gain → t must be 0 */
                    } else {
                        logpmf = -r_u + mul_log_by_count(log_gain, t)
                                 - recount_factln_get(fact, t);
                    }
                } else {
                    /* Pólya: NegBin(t; κ+s, 1-q) */
                    if (log_gain == NEG_INF) {
                        logpmf = (t == 0) ? 0.0 : NEG_INF;
                    } else {
                        const recount_rfactln_t *rf = rfacts[u];
                        double binom = (rf ? recount_rfactln_get(rf, s + t)
                                             - recount_rfactln_get(rf, s) : 0.0)
                                       - recount_factln_get(fact, t);
                        /* (s + κ) · log(1-q): guard for log(1-q) = -∞ (q≈1)
                         * paired with (s+κ) > 0. (s+κ) is always > 0 because
                         * κ > 0 here (we're in the is_polya && log_gain finite
                         * branch). So this only matters when log_q_c = -∞:
                         * the result is then -∞ (Pólya with q=1 puts all
                         * mass on infinite gain, so the L(0) contribution
                         * for any finite t collapses). */
                        double ks_log1q;
                        if (log_q_c == NEG_INF) {
                            ks_log1q = NEG_INF;
                        } else {
                            ks_log1q = ((double)s + kappa) * log_q_c;
                        }
                        double t_logq = mul_log_by_count(log_q, t);
                        logpmf = binom + ks_log1q + t_logq;
                    }
                }
                Kval = ladd(Kval, Cv + logpmf);
            }
            Kmat[Kidx(u, m, s, W)] = Kval;
        }
    }

    /* ---- Step 3: compute L̃_u,m[n][t] (used by parent later) ----
     * For non-root nodes only — the root has no parent that needs L̃_R.
     * Fig S11 lines I26-I33.
     */
    if (u != tree->root) {
        /* Use safelog_pair so log(p) and log(1-p) preserve ULP-level
         * precision near both boundaries (p ≈ 0 and p ≈ 1). */
        double log_p_u  = recount_safelog_pair(sp->p[u], sp->p_c[u]);
        double log_pc_u = recount_safelog_pair(sp->p_c[u], sp->p[u]);
        for (int m = 0; m <= M; m++) {
            for (int n = 0; n <= M; n++) {
                int t = n;
                /* Initialize: L̃[n][t=n] = K̃[u,m][t] */
                Ltens[Lidx(u, m, n, t, W)] = Kmat[Kidx(u, m, t, W)];
                /* Walk t down. Each step: x ← x + log(1-p̃); y ← L̃[n-1][t]
                 * + log(p̃); L̃[n][t] ← ladd(x, y). Guard for boundary
                 * cases p̃ = 0 or p̃ = 1 where one log is -∞. */
                double x = Ltens[Lidx(u, m, n, t, W)];
                while (t > 0) {
                    t--;
                    /* x carries log(survival in n-th lineage so far);
                     * adding log(1-p̃) = log probability the next lineage
                     * dies. If log(1-p̃) = -∞ (p̃ = 1, certain survival),
                     * x → -∞ (impossible event), which is the right answer. */
                    x = (log_pc_u == NEG_INF) ? NEG_INF : (x + log_pc_u);
                    double y;
                    if (n >= 1) {
                        double Lprev = Ltens[Lidx(u, m, n - 1, t, W)];
                        y = (log_p_u == NEG_INF) ? NEG_INF : (Lprev + log_p_u);
                    } else {
                        y = NEG_INF;
                    }
                    x = ladd(x, y);
                    Ltens[Lidx(u, m, n, t, W)] = x;
                }
            }
        }
    }
}
