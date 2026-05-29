/*
 * recount_unobserved_grad.h — native C path for the L(0)-corrected
 * gradient when min_copies ≥ 2.
 *
 * Replaces the Python pipeline in recount/unobserved_outside.py:
 *   1. native inside tensors (already in C: recount_unobserved_inside_tensors)
 *   2. numpy outside pass               → C: recount_unobserved_outside
 *   3. numpy posteriors + transitions   → C: recount_unobserved_posteriors
 *   4. numpy birth/death tails          → C: recount_unobserved_bd_tails
 *   5. numpy survival-gradient          → C: recount_unobserved_logsurv_grad
 *   6. torch autograd chain rule        → C: recount_chain_rule_logit_to_rates
 *
 * Top-level entry:
 *   recount_L0_gradient_native(rates, profiles, min_copies, out_g, out_l,
 *                              out_d, out_t, out_L0)
 * does the full pipeline in C — caller never crosses the Python boundary
 * inside a single BFGS iteration.
 *
 * Layout for the per-size tensors (W = min_copies; arrays are flat with
 * row-major indexing):
 *   B_all[m, v, n]      → idx = m*N*W + v*W + n             (size W*N*W)
 *   J_all[m, v, s]      → idx = m*N*W + v*W + s             (size W*N*W)
 *   Bns_all[m, v, n, s] → idx = m*N*W*W + v*W*W + n*W + s   (size W*N*W*W)
 *   Jns_all[m, v, n, s] → idx = m*N*W*W + v*W*W + n*W + s   (size W*N*W*W)
 *   Lw_all[w, m, ell, t] → idx = w*W*W*W + m*W*W + ell*W + t (size N*W*W*W)
 */

#ifndef RECOUNT_UNOBSERVED_GRAD_H
#define RECOUNT_UNOBSERVED_GRAD_H

#include <stdint.h>
#include "recount_native.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pairing likelihoods L̃_w,m[ell, t], one tensor per node w (skip root).
 * Caller allocates Lw_all of size N*W*W*W and zeros (or NEG_INFs) it.
 * K_all is the inside tensor K̃_w,m[s] from recount_unobserved_inside_tensors.
 *
 * Mirrors compute_pairing_likelihoods() in recount/unobserved_outside.py.
 *
 * Returns 0 on success, nonzero on alloc / arg error.
 */
int recount_unobserved_pairing(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *K_all,                 /* [N, W, W] (m, s) */
    int W,                               /* = min_copies */
    double *Lw_all);                     /* out: [N, W, W, W] */

/*
 * Outside pass: B_all, J_all, Bns_all, Jns_all per-size tensors.
 * Pre-order traversal from root's children.
 * Mirrors compute_unobserved_outside() in recount/unobserved_outside.py.
 */
int recount_unobserved_outside(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *K_all,                 /* [N, W, W] from inside */
    int min_copies,                      /* W = min_copies */
    double *B_all,                       /* out: [W, N, W] */
    double *J_all,                       /* out: [W, N, W] */
    double *Bns_all,                     /* out: [W, N, W, W] */
    double *Jns_all,                     /* out: [W, N, W, W] */
    double *Lw_all_out);                 /* out: [N, W, W, W] (also returned) */

/*
 * L(0) scalar from outside+inside tensors.
 * LL = logsumexp over (m, s) of [J_all[M-m, root, s] + K_all[root, m, s]]
 */
int recount_unobserved_logL0_from_tensors(
    int root, int min_copies,
    const double *K_all,                 /* [N, W, W] */
    const double *J_all,                 /* [W, N, W] */
    int num_nodes,
    double *out_logL0);

/*
 * Marginal posteriors P{ξ_v = n | unobs} and P{η_v = s | unobs}
 * Mirrors compute_unobserved_posteriors() in recount/unobserved_outside.py.
 */
int recount_unobserved_posteriors(
    int num_nodes, int min_copies,
    const double *C_all,                 /* [N, W, W] */
    const double *K_all,                 /* [N, W, W] */
    const double *B_all,                 /* [W, N, W] */
    const double *J_all,                 /* [W, N, W] */
    double log_L0,
    double *out_node_post,               /* [N, W] */
    double *out_edge_post);              /* [N, W] */

/*
 * Joint transition posteriors:
 *   P{ξ_v=n, η_v=s | unobs}    (node)
 *   P{ξ_u=n, η_v=s | unobs}    (edge entering v)
 * Mirrors compute_transition_posteriors() in recount/unobserved_outside.py.
 */
int recount_unobserved_transitions(
    int num_nodes, int min_copies,
    const double *C_all,                 /* [N, W, W] */
    const double *K_all,                 /* [N, W, W] */
    const double *Bns_all,               /* [W, N, W, W] */
    const double *Jns_all,               /* [W, N, W, W] */
    double log_L0,
    double *out_node_trans,              /* [N, W, W] */
    double *out_edge_trans);             /* [N, W, W] */

/*
 * Birth/death tail differences from joint transition posteriors.
 * birth[v, ell] = log Σ_s Σ_{n>ell} exp(node_trans[v, n, s])
 * death[v, ell] = log Σ_s Σ_{n>ell} exp(edge_trans[v, n, s])
 * Mirrors compute_birth_death_tails() in recount/unobserved_outside.py.
 */
int recount_unobserved_bd_tails(
    int num_nodes, int min_copies,
    const double *log_node_trans,        /* [N, W, W] */
    const double *log_edge_trans,        /* [N, W, W] */
    double *out_birth,                   /* [N, W] */
    double *out_death);                  /* [N, W] */

/*
 * Per-node log-survival gradient w.r.t. the logit-parameterized survival
 * params:
 *   d_logit_p[v]    = ∂ log L(<min_copies) / ∂ logit p̃_v  (linear scale)
 *   d_logit_q[v]    = ∂ log L(<min_copies) / ∂ logit q̃_v
 *   d_log_kappa[v]  = ∂ log L(<min_copies) / ∂ log κ_v
 *
 * Mirrors compute_log_survival_gradient_unobs() in
 * recount/unobserved_outside.py (and Java
 * LogGradient.PosteriorStatistics.getLogSurvivalGradient).
 */
int recount_unobserved_logsurv_grad(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    int min_copies,
    const double *log_edge_post,         /* [N, W] */
    const double *log_birth_tails,       /* [N, W] */
    const double *log_death_tails,       /* [N, W] */
    double profile_count,                /* F in the rate-prior formula */
    double *out_d_logit_p,               /* [N] */
    double *out_d_logit_q,               /* [N] */
    double *out_d_log_kappa);            /* [N] */

/*
 * Reverse-mode chain rule from (logit p̃, logit q̃, log κ) adjoints to
 * (gain, loss, dup, length) gradients per node. Walks the survival
 * recurrence in reverse post-order and applies analytical rate_to_p/q
 * Jacobians. Replaces the torch-autograd step in
 * compute_L0_gradient_analytical.
 *
 * Outputs are pure gradients of the same scalar that d_logit_* are
 * adjoints of; i.e. for L = synthetic_dot, returns ∂L/∂(gain[v], ...).
 *
 * Validated against Python prototype to bit-equivalence (both implement
 * the same Yule-tolerance branching for μ ≈ λ).
 */
int recount_chain_rule_logit_to_rates(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const double *gain, const double *loss,
    const double *dup, const double *length,
    const double *d_logit_p, const double *d_logit_q,
    const double *d_log_kappa,
    double *out_d_gain, double *out_d_loss,
    double *out_d_dup, double *out_d_length);

/*
 * Top-level: full L(0)-corrected gradient in rate space.
 *
 *   gain[N], loss[N], dup[N], length[N]   — input rates
 *   profiles is unused for L(0) (only F is needed)
 *
 * Output:
 *   *out_L0        — L(0) scalar
 *   out_g, out_l, out_d, out_t — each [N], ∂L(0)/∂{g, l, d, t}
 *
 * Implementation status (2026-05-18):
 *   - inside tensors:  ✓ native
 *   - outside pass:     in progress (this header)
 *   - posteriors:       todo
 *   - BD tails:         todo
 *   - survival grad:    todo
 *   - rate chain rule:  todo
 *
 * Until the full pipeline is in C, this header documents the planned API.
 */
int recount_L0_gradient_native(
    const recount_tree_t *tree,
    const double *gain, const double *loss,
    const double *dup,  const double *length,
    int min_copies, int F,
    double *out_g, double *out_l, double *out_d, double *out_t,
    double *out_L0);

#ifdef __cplusplus
}
#endif

#endif /* RECOUNT_UNOBSERVED_GRAD_H */
