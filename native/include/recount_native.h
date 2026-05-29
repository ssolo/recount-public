/* recount_native.h — public C API for the native CPU backend.
 *
 * Mirrors the algorithm in recount/gld.py (NumPy reference) but written
 * in scalar/NEON C with libdispatch threading and Apple Accelerate's
 * vForce for vectorized log-space math. Target: beat Java Count's
 * forward LL on Williams2017 (60 leaves, 5378 families, max W=668)
 * — Java is 325 ms on M2 Ultra; we want < 100 ms.
 *
 * All log-space arrays use IEEE -inf as the "no contribution" sentinel.
 * All functions return 0 on success, non-zero on error.
 */
#ifndef RECOUNT_NATIVE_H
#define RECOUNT_NATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- tree topology (caller-owned arrays) ---- */
typedef struct {
    int32_t num_nodes;
    int32_t num_leaves;
    int32_t root;
    const int32_t *parent;        /* [num_nodes], parent[v] > v, parent[root] = -1 */
    const uint8_t *is_leaf;       /* [num_nodes] */
    /* CSR children list (built by recount_tree_init from parent) */
    const int32_t *first_child;   /* [num_nodes + 1] */
    const int32_t *child_list;    /* [num_nodes - 1] */
} recount_tree_t;

/* Helper: derive first_child/child_list from parent. Caller allocates
 * out_first_child[num_nodes+1] and out_child_list[num_nodes-1]. */
void recount_tree_build_csr(
    int32_t num_nodes, int32_t root, const int32_t *parent,
    int32_t *out_first_child, int32_t *out_child_list);

/* ---- survival params (allocated by recount_survival_alloc, freed by ..._free) ---- */
typedef struct {
    int32_t num_nodes;
    double *p, *p_c, *q, *q_c;
    double *gain, *eps, *eps_c;
    double *log_p, *log_p_c, *log_q, *log_q_c;
    double *log_gain, *log_eps, *log_eps_c;
    uint8_t *is_polya;
} recount_survival_t;

int recount_survival_alloc(recount_survival_t *sp, int32_t num_nodes);
void recount_survival_free(recount_survival_t *sp);

/* Bottom-up: build (p̃, q̃, r̃/κ̃, ε) from raw rates per the same
 * recurrence as recount/gld.py:compute_survival_params (lines 85-150). */
int recount_compute_survival_params(
    const recount_tree_t *tree,
    const double *gain, const double *loss,
    const double *dup,  const double *length,
    recount_survival_t *sp);

/* ---- forward LL ---- */
/* Required scratch size (in bytes) for one family of width up to max_width. */
size_t recount_forward_scratch_size(int32_t num_nodes, int32_t max_width);

/* Single-family forward LL. profile[num_leaves] = observed counts (>= 0;
 * negative for ambiguous leaves). scratch must be at least
 * recount_forward_scratch_size(N, max(profile sum)+1). */
int recount_forward_family(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profile,
    void *scratch, size_t scratch_n,
    double *out_ll);

/* Threaded batch: forward LL for F families in parallel via libdispatch.
 * profiles is row-major [F, num_leaves]; out_lls is [F]. num_threads=0
 * means use libdispatch default (P-core count). */
int recount_forward_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_lls);

/* ---- analytical gradient (Csurös 2021 Corollary 9) ---- */
/* Returns ∂LL_raw/∂(p̃, q̃, r̃/κ̃) flat-indexed as 3*v + {GAIN=0, LOSS=1, DUP=2}.
 * For the corrected LL (min_copies=1), subtract F * d log(1-L(0)) / d(...)
 * separately (closed form) — the inside-outside part only needs the raw LL. */
int recount_gradient_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_lls,         /* [F] */
    double *out_grad);       /* [3 * num_nodes] — survival-parameterization */

/* ---- per-branch event counts ---- */
typedef struct {
    int32_t num_nodes;
    double *copies_node;            /* [N] Σ_f E[ξ̃_v | Ξ_f] */
    double *copies_edge;            /* [N] Σ_f E[η̃_v | Ξ_f] */
    double *gain_events;            /* [N] copies_node - copies_edge */
    double *loss_events;            /* [N] parent's copies - edge_copies */
    int32_t *num_families_active;   /* [N] # families with P(ξ_v > 0) > 0.5 */
    double *families_present;       /* [N] Σ_f (1 - P{ξ_v=0|Ξ_f}) — continuous
                                     *     analogue of num_families_active.
                                     *     Matches Java's families_present (sum of
                                     *     posterior presence probabilities) used
                                     *     in MixedRatePosteriors.reportPosteriors. */
} recount_branch_stats_t;

int recount_branch_stats_alloc(recount_branch_stats_t *bs, int32_t num_nodes);
void recount_branch_stats_free(recount_branch_stats_t *bs);

int recount_branch_stats_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double active_threshold,        /* default 0.5 */
    recount_branch_stats_t *out);

/* Per-family per-node posteriors: same pipeline as recount_branch_stats_batch,
 * but writes E[ξ_v|f] and P{ξ_v>0|f} as F·N matrices instead of aggregating.
 * Output buffers are row-major [F][N] doubles, caller-allocated. */
int recount_per_family_posteriors_batch(
    const recount_tree_t *tree, const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_copies_FxN,    /* [F * num_nodes] */
    double *out_present_FxN);  /* [F * num_nodes] */

/* ---- arbitrary-Ωmin observation-bias correction (Csurös 2026 Thms 3-5) ---- */
/* Computes log L(0) = log P{profile sum < min_copies} given the survival
 * params. For min_copies=0 returns -∞; for 1 matches empty_log_likelihood;
 * for 2 matches empty + singleton; for k ≥ 3 uses the SI Theorems 3-5
 * algorithms (NOT implemented in the shipped CountXXV.jar, which caps at 2).
 */
int recount_unobserved_logL0(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies, double *out_logL0);

/* Expose the per-size inside tensors (Algorithm Uinner) for downstream
 * outside/posteriors/gradient computation. Returns:
 *   out_Cmat: row-major [N * W * W], indexed (u, m, n) -> ((u*W)+m)*W+n
 *             C̃_u,m[n] = log P{Ω_u = m | ξ̃_u = n}
 *   out_Kmat: row-major [N * W * W], indexed (u, m, s) -> ((u*W)+m)*W+s
 *             K̃_u,m[s] = log P{Ω_u = m | η̃_u = s}
 *   out_logL0: optional (may be NULL) — convenience scalar L(0).
 * Caller allocates buffers each of size N*W*W doubles, where W = min_copies. */
int recount_unobserved_inside_tensors(
    const recount_tree_t *tree, const recount_survival_t *sp,
    int min_copies,
    double *out_Cmat, double *out_Kmat,
    double *out_logL0);

/* ---- tree-Brownian autocorrelated log-rate prior ----
 * Per-edge Gaussian on log_rate_v − log_rate_pa(v) (per axis: gain, dup, length).
 * v1 uses time-uniform variance (sigma^2 alone, no per-edge t_v weighting).
 * Reference Python: validation/_shared.py:_brownian_prior_and_grad.
 * Documentation:    docs/brownian_prior.tex.
 *
 * Packing matches validation/_shared.py:rates_to_x:
 *     x[0..N-1]       : log_gain[v] for v in [0, N), root included
 *     x[N..2*N-2]     : dup block (logit space when subcritical=True),
 *                       non-root v at position p = v if v<root else v-1
 *     x[2*N-1..3*N-3] : log_length, non-root v, same position map
 *
 * out_grad is zero-initialised inside. Total size = 3*N - 2 doubles.
 */
int recount_brownian_prior_and_grad(
    int32_t N,
    int32_t root,
    const int32_t *parent,          /* [N], parent[root] == -1 */
    const double *x,                /* [3*N - 2] packed */
    double sigma_brownian_gain,
    double sigma_brownian_dup,
    double sigma_brownian_length,
    double mu_root_gain,
    double mu_root_dup,
    double mu_root_length,
    double sigma_root,
    double *out_log_prior,
    double *out_grad);              /* [3*N - 2] */

/* ---- error codes ---- */
#define RECOUNT_OK            0
#define RECOUNT_ENOMEM        1
#define RECOUNT_EINVAL        2
#define RECOUNT_EOVERFLOW     3
#define RECOUNT_ENOTSUP       4

/* ---- version / build info ---- */
const char *recount_native_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RECOUNT_NATIVE_H */
