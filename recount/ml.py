"""Maximum-likelihood fitting of GLD rates.

Optimizes the per-edge gain, duplication, and (optionally) loss and length
parameters to maximize the corrected log-likelihood under the observed
profile table.

The forward / gradient stack is `recount.gld` (NumPy, analytical, matches
Count's Java to machine precision on simple cases).  The chain rule from
``∂LL/∂(p̃, q̃, r̃/κ̃)`` (what ``gradient_survival`` returns) to
``∂LL/∂(raw rate parameters)`` is built on the fly with `torch.autograd`
for convenience (the rate→survival transform is straightforward and
autograd does the right algebra for us — only the raw-rate Jacobian, not
the full per-family forward).

Public entry point:

    result = fit_rates(tree, profiles, *,
                       initial_rates=None,
                       fix_loss=True, fix_root_length=True,
                       min_copies=1, max_iter=200, tol=1e-6, verbose=False)

The optimizer is `scipy.optimize.minimize` with L-BFGS-B; parameters are
log-transformed so positivity is automatic.  ``fit_rates`` returns a
``FitResult`` dataclass with the optimized ``GLDRates``, the achieved
log-likelihood, and the scipy convergence info.
"""
from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from recount.gld import (
    GAIN, LOSS, DUP,
    corrected_log_likelihood,
    gradient_survival,
    compute_survival_params,
)
from recount.rates import GLDRates
from recount.tree import Tree


@dataclass
class FitResult:
    rates: GLDRates
    log_likelihood: float
    n_iter: int
    converged: bool
    message: str
    runtime_s: float
    history: list  # list of (iter, LL) tuples


@dataclass
class MixtureFitResult:
    """Output of :func:`fit_mixture` — base rates plus optimized LogisticShift."""
    rates: GLDRates
    shift: "LogisticShift"  # forward-ref; concrete type below
    log_likelihood: float
    n_iter: int
    converged: bool
    message: str
    runtime_s: float
    history: list


# ----------------------------------------------------------------------------
# default-rate construction
# ----------------------------------------------------------------------------


def default_initial_rates(tree: Tree, *, length: float = 1.0,
                          loss: float = 1.0, dup: float = 0.5,
                          gain: float = 0.1) -> GLDRates:
    """A neutral starting point: uniform per-node rates.

    WARNING: on large trees with the sub-critical Yule constraint
    (``dup <= 1``), this uniform-rate starting point produces a
    degenerate Hessian (many nodes simultaneously want to push
    ``dup`` past the cap) and scipy BFGS gets trapped at a poor
    local optimum (verified on arc269 where it stalls at
    LL=-1,247,428 vs the random-init basin at -1,137,500). Prefer
    ``random_initial_rates`` for any non-trivial ML fit.
    """
    n = tree.num_nodes
    return GLDRates(
        tree=tree,
        gain=np.full(n, gain),
        loss=np.full(n, loss),
        dup=np.full(n, dup),
        length=np.where(tree.parent == -1, np.inf, length),
    )


def random_initial_rates(tree: Tree, rng: np.random.Generator | None = None,
                          *, loss: float = 1.0) -> GLDRates:
    """Csurös-style random initialisation, matching his Java's
    ``TreeWithRates(tree, RND)`` branch with non-null RND
    (``count/model/TreeWithRates.java:511-533``).

    Per-node:
        ``mu = 1.0``
        ``lambda ~ Uniform(0, 0.5)``
        ``gamma ~ Uniform(0, 0.2)``
        ``t ~ Exponential(mean=1)``
        ``t[root] = +inf``

    Csurös 2026 SI Section B.1: "Optimization with Count was
    initialized with a random GLD model (with random seed 2025) and
    ran until the parameter values did not change anymore (within
    machine precision)." Using this init is essential to avoid the
    degenerate-Hessian trap that uniform inits hit at the dup<=1
    cap on large trees.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = tree.num_nodes
    gain = rng.uniform(0.0, 0.2, size=n)
    dup = rng.uniform(0.0, 0.5, size=n)
    length = rng.exponential(scale=1.0, size=n)
    # Root edge length is conventionally +inf
    length[tree.root] = np.inf
    return GLDRates(
        tree=tree,
        gain=gain,
        loss=np.full(n, loss),
        dup=dup,
        length=length,
    )


# ----------------------------------------------------------------------------
# parameterization
# ----------------------------------------------------------------------------


def _params_to_rates(theta: np.ndarray, rates_template: GLDRates,
                     fix_loss: bool, fix_root_length: bool) -> GLDRates:
    """Unpack the optimizer variable θ (log-space) into a GLDRates."""
    n = rates_template.tree.num_nodes
    root = rates_template.tree.root
    gain = np.zeros(n); loss = np.zeros(n); dup = np.zeros(n); length = np.zeros(n)

    idx = 0
    for v in range(n):
        gain[v] = np.exp(theta[idx]); idx += 1
        if not fix_loss:
            loss[v] = np.exp(theta[idx]); idx += 1
        else:
            loss[v] = rates_template.loss[v]
        dup[v] = np.exp(theta[idx]); idx += 1
        if v == root and fix_root_length:
            length[v] = np.inf
        else:
            length[v] = np.exp(theta[idx]); idx += 1
    assert idx == theta.size
    return GLDRates(tree=rates_template.tree, gain=gain, loss=loss, dup=dup, length=length)


def _rates_to_params(rates: GLDRates, fix_loss: bool, fix_root_length: bool) -> np.ndarray:
    """Pack a GLDRates into the optimizer variable θ (log of each free rate)."""
    tree = rates.tree
    root = tree.root
    parts = []
    for v in range(tree.num_nodes):
        parts.append(np.log(max(rates.gain[v], 1e-300)))
        if not fix_loss:
            parts.append(np.log(max(rates.loss[v], 1e-300)))
        parts.append(np.log(max(rates.dup[v], 1e-300)))
        if not (v == root and fix_root_length):
            parts.append(np.log(max(rates.length[v], 1e-300)))
    return np.array(parts, dtype=np.float64)


# ----------------------------------------------------------------------------
# chain rule survival → raw rates
# ----------------------------------------------------------------------------


def _gradient_raw(tree: Tree, rates: GLDRates, profiles: np.ndarray,
                  min_copies: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (LL, ∂LL/∂gain, ∂LL/∂loss, ∂LL/∂dup, ∂LL/∂length) — per-node.

    LL is the corrected log-likelihood.  Each gradient component has shape
    [num_nodes]. The chain rule from ``gradient_survival`` to the raw rates
    uses PyTorch autograd through ``compute_survival_params_t``.
    """
    import torch  # local import so recount.ml works without torch installed
    from recount.torch_fast import compute_survival_params_t

    # corrected LL (NumPy)
    LL = corrected_log_likelihood(tree, rates, profiles, min_copies=min_copies)
    # ∂LL/∂(p̃, q̃, r̃/κ̃)  per node — analytical NumPy gradient
    g_surv = gradient_survival(tree, rates, profiles, min_copies=min_copies)
    g_surv = g_surv.reshape(tree.num_nodes, 3)  # columns: GAIN=0, LOSS=1, DUP=2

    # rate → survival, autograd
    dt = torch.float64
    gain = torch.tensor(rates.gain, dtype=dt, requires_grad=True)
    loss = torch.tensor(rates.loss, dtype=dt, requires_grad=True)
    dup = torch.tensor(rates.dup, dtype=dt, requires_grad=True)
    length = torch.tensor(rates.length, dtype=dt, requires_grad=True)
    sp = compute_survival_params_t(tree, gain, loss, dup, length)
    # The "GAIN" axis of g_surv refers to r̃ (Poisson) or κ̃ (Pólya).
    # In sp, sp.gain stores that survival quantity.
    surv_outputs = torch.stack([sp.gain, sp.p, sp.q])  # shape (3, N) matching columns (GAIN,LOSS,DUP)
    surv_grad_out = torch.tensor(g_surv.T, dtype=dt)   # shape (3, N), matching surv_outputs
    grads = torch.autograd.grad(
        outputs=surv_outputs, inputs=(gain, loss, dup, length),
        grad_outputs=surv_grad_out, allow_unused=True,
    )
    g_gain = grads[0].detach().numpy() if grads[0] is not None else np.zeros(tree.num_nodes)
    g_loss = grads[1].detach().numpy() if grads[1] is not None else np.zeros(tree.num_nodes)
    g_dup  = grads[2].detach().numpy() if grads[2] is not None else np.zeros(tree.num_nodes)
    g_len  = grads[3].detach().numpy() if grads[3] is not None else np.zeros(tree.num_nodes)
    # Sanitize: root edge length is conventionally fixed (+inf), the gradient
    # there is mathematically undefined; zero it so the optimizer doesn't see noise.
    if np.isnan(g_len[tree.root]) or np.isinf(g_len[tree.root]):
        g_len[tree.root] = 0.0
    return float(LL), g_gain, g_loss, g_dup, g_len


def _gradient_raw_native(tree: Tree, rates: GLDRates, profiles_np,
                          min_copies: int):
    """Native CPU backend gradient — fully native C path as of 2026-05-18.

    Pipeline (no torch in the hot loop):
      1. gradient_native → corrected LL + ∂LL_corr/∂(κ, p̃, q̃)
         (Csurös 2021 Cor. 9 analytical formulas, ported to C)
      2. chain_rule_logit_to_rates_native → ∂LL_corr/∂(gain, loss, dup,
         length) via reverse-mode walk of the survival recurrence with
         analytical rate_to_p_jac / rate_to_q_jac

    For ``min_copies`` ≥ 2, the aux L(0) delta computed inside
    gradient_native (analytical via compute_L0_gradient_analytical) is
    added on top to upgrade the min_copies=1 corrected gradient to
    min_copies=k.
    """
    from recount.native_backend import (
        gradient_native, _GradientWithL0Aux, _compute_survival,
        chain_rule_logit_to_rates_native, _RecountSurvival, _lib,
    )
    import ctypes as _ctypes

    # Native: corrected LL + survival-param gradient ∂LL_corr/∂(κ, p̃, q̃)
    LL, g_surv_flat = gradient_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles_np, min_copies=min_copies, num_threads=0,
    )
    aux = None
    if isinstance(g_surv_flat, _GradientWithL0Aux):
        aux = g_surv_flat
        g_surv_flat = np.asarray(g_surv_flat)
    g_surv = g_surv_flat.reshape(tree.num_nodes, 3)  # cols (GAIN, LOSS, DUP)

    # Pull the survival params (p̃, q̃, p̃_c, q̃_c, κ_t) from native, then
    # convert (κ_t, p̃, q̃) adjoints to (log κ_t, logit p̃, logit q̃)
    # adjoints — the form chain_rule_logit_to_rates_native expects.
    #   adj_log_κ_t = adj_κ_t · κ_t
    #   adj_logit_p̃ = adj_p̃ · p̃ · p̃_c    (since logit_p̃ = log(p̃/(1-p̃)))
    #   adj_logit_q̃ = adj_q̃ · q̃ · q̃_c
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, rates.gain, rates.loss, rates.dup, rates.length)
    try:
        N = int(tree.num_nodes)
        def _g(ptr):
            return np.frombuffer(
                _ctypes.cast(ptr, _ctypes.POINTER(_ctypes.c_double * N))[0],
                dtype=np.float64).copy()
        p_t   = _g(sp_struct.p)
        p_t_c = _g(sp_struct.p_c)
        q_t   = _g(sp_struct.q)
        q_t_c = _g(sp_struct.q_c)
        kappa_t = _g(sp_struct.gain)
    finally:
        _lib.recount_survival_free(_ctypes.byref(sp_struct))

    adj_logit_p = g_surv[:, 1] * p_t * p_t_c
    adj_logit_q = g_surv[:, 2] * q_t * q_t_c
    adj_log_kappa = g_surv[:, 0] * kappa_t

    g_gain, g_loss, g_dup, g_len = chain_rule_logit_to_rates_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        adj_logit_p, adj_logit_q, adj_log_kappa,
    )
    # min_copies ≥ 2: add the analytical L(0) correction term to the raw-rate
    # gradient. native_backend.gradient_native bakes in the (min_copies=1)
    # correction already; we ADD the delta to upgrade to min_copies=k.
    # Eq 5 in the paper: ∂(F·log(1-L0))/∂θ = -F · L0/(1-L0) · ∂L0/∂θ
    if aux is not None:
        # aux.L0_grad_* come from compute_L0_gradient_analytical (fully
        # native C as of commit 7fd3425) via native_backend._gradient_raw_native;
        # they are the additive delta to apply to the min_copies=1 corrected
        # gradient to convert it to the min_copies=k corrected gradient.
        if getattr(aux, "delta_is_final", False):
            g_gain += aux.L0_grad_gain
            if aux.L0_grad_loss is not None:
                g_loss += aux.L0_grad_loss
            g_dup  += aux.L0_grad_dup
            g_len  += aux.L0_grad_length
        else:
            # Legacy pre-port format (the F · (ratio_k − ratio_1) · ∂L0/∂θ
            # form, kept for back-compat with any code that still produces
            # `aux` without `delta_is_final`; current native path always sets it).
            F = aux.F
            ratio_k = aux.ratio_k
            ratio_1 = aux.ratio_1
            g_gain += -F * (ratio_k - ratio_1) * aux.L0_grad_gain
            g_dup  += -F * (ratio_k - ratio_1) * aux.L0_grad_dup
            g_len  += -F * (ratio_k - ratio_1) * aux.L0_grad_length
    if np.isnan(g_len[tree.root]) or np.isinf(g_len[tree.root]):
        g_len[tree.root] = 0.0
    return float(LL), g_gain, g_loss, g_dup, g_len


def _gradient_raw_fast(tree: Tree, rates: GLDRates, profiles_t,
                        min_copies: int, W: int, chunk_F: int = 1024,
                        use_destructive: bool = False):
    """Same return shape as ``_gradient_raw`` but using full torch autograd
    through :func:`recount.torch_fast.corrected_log_likelihood_fast` instead
    of the per-family NumPy analytical gradient.

    On Williams 1.4k families this is ~15× faster than the NumPy path; on
    larger trees / datasets the gap grows further.  Returns finite gradients
    once the ``safe_logsumexp`` and ``safe_log_pair`` fixes in
    ``recount.torch_fast`` are in place.

    Set ``use_destructive=True`` to route through the O(W²) destructive
    combine in ``recount.destructive_combine`` (pure PyTorch — the Triton
    CUDA branch was removed for the M-series target).
    """
    import torch
    from recount.torch_fast import corrected_log_likelihood_fast
    dt = torch.float64
    gain = torch.tensor(rates.gain, dtype=dt, requires_grad=True)
    loss = torch.tensor(rates.loss, dtype=dt, requires_grad=True)
    dup = torch.tensor(rates.dup, dtype=dt, requires_grad=True)
    length = torch.tensor(rates.length, dtype=dt, requires_grad=True)
    LL = corrected_log_likelihood_fast(
        tree, gain, loss, dup, length, profiles_t,
        min_copies=min_copies, W=W, chunk_F=chunk_F,
        use_destructive=use_destructive,
    )
    LL.backward()
    g_gain = gain.grad.detach().cpu().numpy()
    g_loss = loss.grad.detach().cpu().numpy() if loss.grad is not None else np.zeros(tree.num_nodes)
    g_dup = dup.grad.detach().cpu().numpy()
    g_len = length.grad.detach().cpu().numpy()
    # Root edge length is fixed; zero its gradient.
    if np.isnan(g_len[tree.root]) or np.isinf(g_len[tree.root]):
        g_len[tree.root] = 0.0
    return float(LL.detach()), g_gain, g_loss, g_dup, g_len


# ----------------------------------------------------------------------------
# tree-Brownian autocorrelated log-rate prior (Thorne-Kishino-Painter)
# ----------------------------------------------------------------------------


def _brownian_objgrad(tree: Tree, rates: GLDRates, sigma_brownian: float,
                      mu_root_gain: float, mu_root_dup: float,
                      mu_root_length: float, sigma_root: float):
    """Tree-Brownian log-rate prior + its log-space gradient.

    A per-edge Gaussian N(0, σ²) on the parent→child increment of
    log(gain), log(dup) and log(length), plus a wide N(μ_root, σ_root²)
    anchor on the root's log(gain). See docs/brownian_prior.tex.

    Returns ``(log_prior, plg, pld, pll)`` where plg/pld/pll are per-node
    arrays of ∂log_prior/∂(log gain / log dup / log length) — already in
    the optimizer's log-rate space, so they add straight onto the
    log-likelihood gradient. The root entries of pld/pll are 0 (the root
    dup/length are not free Brownian parameters)."""
    from recount.native_backend import brownian_prior_native
    N = tree.num_nodes
    root = tree.root
    nonroot = np.ones(N, dtype=bool); nonroot[root] = False
    log_gain = np.log(np.clip(rates.gain, 1e-300, None))
    log_dup = np.log(np.clip(rates.dup, 1e-300, None))
    log_length = np.log(np.clip(rates.length, 1e-300, None))
    # x packing matches native recount_brownian_prior_and_grad:
    #   log_gain[N] | log_dup[non-root] | log_length[non-root]
    x = np.concatenate([log_gain, log_dup[nonroot], log_length[nonroot]])
    log_prior, gx = brownian_prior_native(
        parent=np.ascontiguousarray(tree.parent, dtype=np.int32),
        root=int(root), x=x,
        sigma_brownian_gain=sigma_brownian,
        sigma_brownian_dup=sigma_brownian,
        sigma_brownian_length=sigma_brownian,
        mu_root_gain=mu_root_gain, mu_root_dup=mu_root_dup,
        mu_root_length=mu_root_length, sigma_root=sigma_root,
    )
    plg = gx[:N]
    pld = np.zeros(N); pld[nonroot] = gx[N:2 * N - 1]
    pll = np.zeros(N); pll[nonroot] = gx[2 * N - 1:]
    return float(log_prior), plg, pld, pll


# ----------------------------------------------------------------------------
# loss function for scipy
# ----------------------------------------------------------------------------


def _make_objective(tree: Tree, profiles: np.ndarray,
                    rates_template: GLDRates,
                    fix_loss: bool, fix_root_length: bool,
                    min_copies: int, history: list, verbose: bool,
                    backend: str = "numpy", use_destructive: bool = False,
                    prior: str = "none", sigma_brownian: float = 1.0,
                    mu_root_gain: float = 0.0, mu_root_dup: float = 0.0,
                    mu_root_length: float = 0.0, sigma_root: float = 5.0):
    """Build the (objective, gradient) function for scipy.

    ``backend``:
      * ``"numpy"`` (default): analytical NumPy ``gradient_survival`` plus a
        tiny autograd chain for rate → survival.  Best for tiny trees / few
        families; slow for big trees because the inside/outside pass is a
        Python loop.
      * ``"torch"``: full autograd through the vectorized
        ``torch_fast.forward_fast``.  ~15× faster on Williams-scale data.
      * ``"native"``: native C analytical gradient via
        :func:`recount.native_backend.gradient_native` — matches Java's
        ``count.model.Gradient`` algorithm bit-for-bit, ~75× faster than
        Java's forward+gradient on Williams (and ~3000× faster than the
        NumPy backend). Requires the ``native/librecount.dylib`` build.
    """
    iters = {"k": 0}
    if backend == "torch":
        import torch
        W = int(profiles.sum(axis=1).max())
        profiles_t = torch.tensor(profiles, dtype=torch.long)
    profiles_np_i32 = None
    if backend == "native":
        profiles_np_i32 = np.ascontiguousarray(profiles, dtype=np.int32)

    def f_and_g(theta: np.ndarray) -> tuple[float, np.ndarray]:
        # Clamp log-params: prevents the optimizer from wandering to rates so
        # extreme they make the analytical gradient ill-defined (κ=0, q=0).
        theta = np.clip(theta, -40.0, 40.0)
        rates = _params_to_rates(theta, rates_template, fix_loss, fix_root_length)
        try:
            if backend == "torch":
                LL, g_gain, g_loss, g_dup, g_len = _gradient_raw_fast(
                    tree, rates, profiles_t, min_copies, W,
                    use_destructive=use_destructive,
                )
            elif backend == "native":
                LL, g_gain, g_loss, g_dup, g_len = _gradient_raw_native(
                    tree, rates, profiles_np_i32, min_copies,
                )
            else:
                LL, g_gain, g_loss, g_dup, g_len = _gradient_raw(
                    tree, rates, profiles, min_copies,
                )
        except Exception:
            LL, g_gain, g_loss, g_dup, g_len = (
                float("nan"), np.zeros(tree.num_nodes), np.zeros(tree.num_nodes),
                np.zeros(tree.num_nodes), np.zeros(tree.num_nodes),
            )
        # tree-Brownian log-rate prior → objective becomes the log-posterior.
        plg = pld = pll = None
        if prior == "brownian" and np.isfinite(LL):
            try:
                log_prior, plg, pld, pll = _brownian_objgrad(
                    tree, rates, sigma_brownian,
                    mu_root_gain, mu_root_dup, mu_root_length, sigma_root)
                LL = LL + log_prior
            except Exception:
                plg = pld = pll = None
        if not np.isfinite(LL):
            # Signal "infeasible" to L-BFGS-B without breaking it.
            LL = -1e12
        # Chain log: ∂LL/∂log(x) = x · ∂LL/∂x; the Brownian prior gradient
        # (plg/pld/pll) is already in log-rate space, so it adds straight on.
        grad_parts = []
        n = tree.num_nodes
        for v in range(n):
            gg = g_gain[v] * rates.gain[v]
            if plg is not None:
                gg += plg[v]
            grad_parts.append(gg)
            if not fix_loss:
                grad_parts.append(g_loss[v] * rates.loss[v])
            gd = g_dup[v] * rates.dup[v]
            if pld is not None:
                gd += pld[v]
            grad_parts.append(gd)
            if not (v == tree.root and fix_root_length):
                # NaN/inf-safe: zero the gradient for length at the root edge
                gl = g_len[v]
                if not np.isfinite(gl):
                    gl = 0.0
                gl_theta = gl * rates.length[v] if np.isfinite(rates.length[v]) else 0.0
                if pll is not None:
                    gl_theta += pll[v]
                grad_parts.append(gl_theta)
        grad = np.array(grad_parts, dtype=np.float64)
        # Replace residual NaN/Inf to keep L-BFGS-B happy
        if not np.all(np.isfinite(grad)):
            grad = np.where(np.isfinite(grad), grad, 0.0)
        iters["k"] += 1
        history.append((iters["k"], LL))
        if verbose and iters["k"] % 5 == 0:
            print(f"  iter {iters['k']:>4d}  LL = {LL:.6f}", file=sys.stderr, flush=True)
        return -LL, -grad  # scipy minimizes

    return f_and_g


# ----------------------------------------------------------------------------
# top-level entry point
# ----------------------------------------------------------------------------


def _build_bounds(rates_template: GLDRates, fix_loss: bool, fix_root_length: bool,
                  bound_dup_by_loss: bool, bound_gain_by_loss: bool):
    """Per-parameter (lo, hi) bounds for scipy L-BFGS-B.

    The optimizer variables are log(gain), [log(loss)], log(dup), [log(length)]
    in the order produced by ``_rates_to_params``.  The biological-meaning
    constraints from Csurös 2026 are ``λ_v ≤ 1`` (dup ≤ loss) and
    ``γ_v ≤ 1`` (gain ≤ loss, regularizing).  When ``fix_loss=True`` and
    ``loss[v]`` is fixed (Williams convention sets ``loss=1`` everywhere),
    these become simple upper bounds on ``log(gain[v])`` and ``log(dup[v])``.
    With ``fix_loss=False`` we leave them un-bounded to keep the bound list
    simple — the constraint can be re-imposed by re-running with a fixed
    loss after.
    """
    tree = rates_template.tree
    root = tree.root
    bounds = []
    for v in range(tree.num_nodes):
        log_loss_v = float(np.log(max(rates_template.loss[v], 1e-300)))
        gain_hi = log_loss_v if (bound_gain_by_loss and fix_loss) else None
        dup_hi = log_loss_v if (bound_dup_by_loss and fix_loss) else None
        bounds.append((None, gain_hi))                       # log(gain)
        if not fix_loss:
            bounds.append((None, None))                      # log(loss)
        bounds.append((None, dup_hi))                        # log(dup)
        if not (v == root and fix_root_length):
            bounds.append((None, None))                      # log(length)
    return bounds


def fit_rates(tree: Tree, profiles: np.ndarray, *,
              initial_rates: Optional[GLDRates] = None,
              fix_loss: bool = True, fix_root_length: bool = True,
              bound_dup_by_loss: bool = True, bound_gain_by_loss: bool = True,
              min_copies: int = 1, max_iter: int = 200, tol: float = 1e-6,
              backend: str = "numpy", use_destructive: bool = False,
              prior: str = "none", sigma_brownian: float = 1.0,
              verbose: bool = False) -> FitResult:
    """Maximum-likelihood fit of the GLD rates.

    ``min_copies`` ≥ 3 (arbitrary Ωmin per Csurös 2026 SI Theorems 3-5)
    requires ``backend="native"``; the shipped CountXXV.jar's Gradient
    class caps at 2 (line 90 throws).

    Parameters
    ----------
    tree, profiles
        The tree topology and a (num_families, num_leaves) integer table.
    initial_rates : optional GLDRates
        Starting point. Defaults to uniform rates from ``default_initial_rates``.
    fix_loss : bool
        If True (default) the loss rate ``μ`` is held at its initial value
        for every node (the Williams convention is ``μ=1`` everywhere).
    fix_root_length : bool
        If True (default) the root edge length stays at +∞.
    bound_dup_by_loss : bool
        Enforce the biological-meaning constraint ``λ_v ≤ 1`` (i.e.
        ``dup_v ≤ loss_v``).  Only applied when ``fix_loss=True``.
    bound_gain_by_loss : bool
        Enforce ``γ_v ≤ 1`` (i.e. ``gain_v ≤ loss_v``) — a numerical
        regularizer recommended in Csurös 2026 §Materials-and-Methods.
        Only applied when ``fix_loss=True``.
    min_copies : int >= 0
        Observation-bias correction (Ωmin). The numpy and torch backends
        cap at 2 (matching the shipped Java's Gradient class which throws
        at line 90 for higher); ``backend="native"`` supports arbitrary
        ``min_copies >= 0`` via the analytical L(0) gradient (Csurös 2026
        SI Theorems 3-5). Most callers pass 1 (no empty families) or 4
        (Csurös' subset convention).
    prior : {"none", "brownian"}
        Rate regulariser. ``"none"`` (default) is plain maximum
        likelihood; ``"brownian"`` adds a tree-Brownian autocorrelated
        log-rate prior (Thorne-Kishino-Painter), turning the fit into a
        MAP estimate that shares statistical strength across neighbouring
        branches.
    sigma_brownian : float
        Brownian prior std on each per-edge log-rate increment (σ=1
        default); larger σ → weaker shrinkage (σ→∞ recovers plain ML).
    max_iter, tol
        scipy L-BFGS-B options.
    verbose
        If True, print progress every 5 iterations.
    """
    if min_copies < 0:
        raise ValueError("min_copies must be >= 0")
    if min_copies > 2 and backend != "native":
        raise ValueError(
            f"min_copies={min_copies} requires backend='native' "
            f"(numpy/torch backends only support min_copies <= 2)"
        )
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError as e:
        raise ImportError("scipy is required for fit_rates; install via `pip install scipy`") from e

    if initial_rates is None:
        initial_rates = default_initial_rates(tree)

    if prior not in ("none", "brownian"):
        raise ValueError(f"prior must be 'none' or 'brownian', got {prior!r}")
    _root = tree.root
    mu_root_gain = float(np.log(max(initial_rates.gain[_root], 1e-300)))
    mu_root_dup = float(np.log(max(initial_rates.dup[_root], 1e-300)))
    mu_root_length = 0.0  # log(1) — reference length for the root's children
    if prior == "brownian":
        # fail loudly now if the native Brownian-prior kernel is unavailable
        from recount.native_backend import brownian_prior_native  # noqa: F401

    theta0 = _rates_to_params(initial_rates, fix_loss, fix_root_length)
    bounds = _build_bounds(initial_rates, fix_loss, fix_root_length,
                           bound_dup_by_loss, bound_gain_by_loss)
    # Clip the initial point into the feasible region so L-BFGS-B starts inside.
    for i, (_, hi) in enumerate(bounds):
        if hi is not None and theta0[i] > hi:
            theta0[i] = hi - 1e-6
    history: list = []
    f_and_g = _make_objective(tree, profiles, initial_rates,
                               fix_loss, fix_root_length, min_copies, history, verbose,
                               backend=backend, use_destructive=use_destructive,
                               prior=prior, sigma_brownian=sigma_brownian,
                               mu_root_gain=mu_root_gain, mu_root_dup=mu_root_dup,
                               mu_root_length=mu_root_length, sigma_root=5.0)

    if verbose:
        n_bound = sum(1 for _, h in bounds if h is not None)
        print(f"# starting optimization with {theta0.size} free parameters "
              f"({n_bound} with upper bounds)", file=sys.stderr, flush=True)
    t0 = time.time()
    res = minimize(
        f_and_g, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "gtol": tol, "disp": False},
    )
    runtime = time.time() - t0
    rates_out = _params_to_rates(res.x, initial_rates, fix_loss, fix_root_length)
    return FitResult(
        rates=rates_out,
        log_likelihood=-res.fun,
        n_iter=int(res.nit),
        converged=bool(res.success),
        message=str(res.message),
        runtime_s=runtime,
        history=history,
    )


# ----------------------------------------------------------------------------
# bounded-W → wider-W bootstrap (cheap warm start for heavy-tail data)
# ----------------------------------------------------------------------------


def fit_rates_bootstrap(tree: Tree, profiles: np.ndarray, *,
                        w_schedule: Optional[list] = None,
                        max_iter_each: Optional[list] = None,
                        initial_rates: Optional[GLDRates] = None,
                        fix_loss: bool = True, fix_root_length: bool = True,
                        bound_dup_by_loss: bool = True, bound_gain_by_loss: bool = True,
                        min_copies: int = 1, tol: float = 1e-6,
                        backend: str = "torch",
                        verbose: bool = False) -> FitResult:
    """Multi-phase fit on increasingly wider profile-sum subsets.

    The vectorized inside pass costs O(W³) per internal node where ``W`` is
    the max profile sum.  Williams2017 has ``max_sum = 668`` with a long
    heavy tail; a single full-data fit is prohibitive on CPU.  This helper
    runs a sequence of fits on growing subsets — each starts from the
    previous result's rates — so most of the iterations happen on cheap
    bounded-W data and only a handful of expensive iterations touch the
    heavy tail.

    Parameters
    ----------
    w_schedule
        Sequence of max profile-sum cutoffs.  Defaults to ``[8, 16, 32]``.
        Each phase uses the families with ``sum(profile) ≤ W``.
    max_iter_each
        Per-phase L-BFGS-B iteration cap.  Defaults to ``[40, 20, 10]``.
        Last phase usually only needs a few iterations to refine.
    backend
        Defaults to ``"torch"`` since the bootstrap targets large datasets.
    """
    if w_schedule is None:
        w_schedule = [8, 16, 32]
    if max_iter_each is None:
        max_iter_each = [40, 20, 10][: len(w_schedule)]
    if len(max_iter_each) != len(w_schedule):
        raise ValueError("len(max_iter_each) must match len(w_schedule)")

    rates = initial_rates
    combined_history: list = []
    total_runtime = 0.0
    last_fr: Optional[FitResult] = None
    sums = profiles.sum(axis=1)
    for phase_idx, (W, n_iter) in enumerate(zip(w_schedule, max_iter_each)):
        subset = profiles[sums <= W]
        if verbose:
            print(f"# phase {phase_idx + 1}/{len(w_schedule)}: W≤{W}, "
                  f"{subset.shape[0]} families", file=sys.stderr, flush=True)
        fr = fit_rates(
            tree, subset, initial_rates=rates,
            fix_loss=fix_loss, fix_root_length=fix_root_length,
            bound_dup_by_loss=bound_dup_by_loss,
            bound_gain_by_loss=bound_gain_by_loss,
            min_copies=min_copies, max_iter=n_iter, tol=tol,
            backend=backend, verbose=verbose,
        )
        if verbose:
            print(f"# phase {phase_idx + 1}: LL = {fr.log_likelihood:.4f}  "
                  f"({fr.n_iter} iters, {fr.runtime_s:.1f}s)",
                  file=sys.stderr, flush=True)
        rates = fr.rates
        combined_history.extend((phase_idx + 1, k, ll) for k, ll in fr.history)
        total_runtime += fr.runtime_s
        last_fr = fr

    return FitResult(
        rates=last_fr.rates,
        log_likelihood=last_fr.log_likelihood,
        n_iter=sum(p_iter for _, p_iter, _ in combined_history) if combined_history else last_fr.n_iter,
        converged=last_fr.converged,
        message=f"bootstrap done; last phase: {last_fr.message}",
        runtime_s=total_runtime,
        history=combined_history,
    )


# ----------------------------------------------------------------------------
# soft-landing — Phase-0 global-rate fit + annealed Brownian-MAP warm-starts
# ----------------------------------------------------------------------------


def fit_global_rates(tree: Tree, profiles: np.ndarray, *,
                     min_copies: int = 1,
                     init_gain: float = 0.1, init_dup: float = 0.5,
                     init_length: float = 1.0,
                     max_iter: int = 80,
                     verbose: bool = False) -> FitResult:
    """Fit a single global γ, single global λ, and per-branch lengths.

    Loss is fixed at 1; root edge length stays at +∞; λ is bounded above
    by 1 (sub-critical). This is the well-conditioned 2 + (N-1) parameter
    problem that opens the soft-landing pipeline — it never visits the
    runaway-γ degenerate corner that cold-start per-branch L-BFGS-B can
    fall into on annotations with extreme paralogy.

    Returned ``FitResult`` carries broadcast-global γ and λ across every
    node, plus the optimised per-branch lengths.
    """
    from scipy.optimize import minimize  # type: ignore

    N = tree.num_nodes
    root = tree.root
    nonroot = [v for v in range(N) if v != root]
    n_lens = len(nonroot)
    pi32 = np.ascontiguousarray(profiles, dtype=np.int32)

    def unpack(theta):
        log_G, log_D = theta[0], theta[1]
        log_lens = theta[2:]
        gain = np.full(N, np.exp(log_G))
        dup = np.full(N, np.exp(log_D))
        loss = np.full(N, 1.0)
        length = np.full(N, np.inf)
        for i, v in enumerate(nonroot):
            length[v] = np.exp(log_lens[i])
        return GLDRates(tree=tree, gain=gain, loss=loss, dup=dup, length=length)

    iters = {"k": 0}
    history: list = []

    def f_and_g(theta):
        theta = np.clip(theta, -40.0, 40.0)
        rates = unpack(theta)
        try:
            LL, g_gain, _g_loss, g_dup, g_len = _gradient_raw_native(
                tree, rates, pi32, min_copies)
        except Exception:
            return 1e12, np.zeros_like(theta)
        if not np.isfinite(LL):
            return 1e12, np.zeros_like(theta)
        G = float(np.exp(theta[0])); D = float(np.exp(theta[1]))
        d_log_G = G * float(np.sum(g_gain))
        d_log_D = D * float(np.sum(g_dup))
        d_log_lens = np.zeros(n_lens)
        for i, v in enumerate(nonroot):
            gl = g_len[v] if np.isfinite(g_len[v]) else 0.0
            L = rates.length[v]
            d_log_lens[i] = gl * L if np.isfinite(L) else 0.0
        grad = np.concatenate([[d_log_G, d_log_D], d_log_lens])
        grad = np.where(np.isfinite(grad), grad, 0.0)
        iters["k"] += 1
        history.append((iters["k"], LL))
        if verbose and iters["k"] % 5 == 0:
            print(f"    iter {iters['k']:>4d}  LL = {LL:.6f}  γ={G:.4g}  λ={D:.4g}",
                  file=sys.stderr, flush=True)
        return -LL, -grad

    theta0 = np.concatenate(
        [[np.log(init_gain), np.log(init_dup)],
         np.full(n_lens, np.log(init_length))])
    bounds = [(None, None)] * len(theta0)
    bounds[1] = (None, 0.0)  # log(λ) ≤ 0 ⇒ λ ≤ 1 (sub-critical)

    t0 = time.time()
    res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "gtol": 1e-6, "disp": False})
    return FitResult(
        rates=unpack(res.x),
        log_likelihood=-float(res.fun),
        n_iter=int(res.nit),
        converged=bool(res.success),
        message=str(res.message),
        runtime_s=time.time() - t0,
        history=history,
    )


def soft_landing(tree: Tree, profiles: np.ndarray, *,
                 min_copies: int = 1,
                 sigma_schedule=(0.05, 0.15, 0.3, 0.6, 1.0),
                 phase_max_iter: int = 60,
                 backend: str = "native",
                 verbose: bool = False):
    """Soft-landing fit: global-rate Phase 0 → annealed Brownian-MAP phases.

    Recommended for **large datasets (≥ several hundred taxa)** or
    annotations with extreme per-cell paralogy where a cold-start
    ``fit_rates(prior="brownian")`` lands in a degenerate basin — γ on the
    +∞ root edge → tens, λ → 0, root reconstruction collapses to ~0
    families. Soft-landing avoids that basin by starting from a well-
    conditioned globally-uniform rate fit, then gradually relaxing the
    per-branch Brownian prior from very stiff to its default σ = 1.

    Pipeline:

    * **Phase 0** — fit a single global γ, single global λ, per-branch
      lengths (well-conditioned, 2 + N−1 parameters).
    * **Phase 1** — Brownian-MAP at σ = ``sigma_schedule[0]`` (default
      0.05), warm-started from Phase 0 → rates stay tightly clustered
      around the global fit.
    * **Phases 2..N** — σ relaxed monotonically toward the final value
      (default 1.0, matching ``fit_rates(prior="brownian")``).

    Returns
    -------
    rates : GLDRates
        Final fitted per-branch rates.
    history : list[dict]
        Per-phase record (``phase``, ``kind``, ``LL``/``MAP_obj``,
        ``iters``, ``converged``, ``time_s``, ``sigma`` for the annealed
        phases).
    """
    if verbose:
        print(f"  Phase 0: fit global γ + λ + {tree.num_nodes - 1} branch lengths",
              file=sys.stderr, flush=True)
    fr0 = fit_global_rates(tree, profiles, min_copies=min_copies,
                           max_iter=phase_max_iter, verbose=verbose)
    rates = fr0.rates
    if verbose:
        print(f"  Phase 0 done: LL={fr0.log_likelihood:.4f}  iters={fr0.n_iter}  "
              f"converged={fr0.converged}  time={fr0.runtime_s:.1f}s",
              file=sys.stderr, flush=True)
        print(f"    γ* = {rates.gain[tree.root]:.6g}    λ* = {rates.dup[tree.root]:.6g}",
              file=sys.stderr, flush=True)

    history = [{"phase": 0, "kind": "global_GD_perbranch_L",
                "LL": fr0.log_likelihood, "iters": fr0.n_iter,
                "converged": fr0.converged, "time_s": fr0.runtime_s}]

    for i, sigma in enumerate(sigma_schedule):
        if verbose:
            print(f"\n  Phase {i+1}: Brownian-MAP, σ={sigma}, warm-start",
                  file=sys.stderr, flush=True)
        fr = fit_rates(
            tree, profiles, initial_rates=rates,
            fix_loss=True, fix_root_length=True,
            bound_dup_by_loss=True, bound_gain_by_loss=False,
            min_copies=min_copies,
            prior="brownian", sigma_brownian=float(sigma),
            max_iter=phase_max_iter, tol=1e-6,
            backend=backend, verbose=False,
        )
        rates = fr.rates
        if verbose:
            print(f"  Phase {i+1} done: MAP-obj={fr.log_likelihood:.4f}  "
                  f"iters={fr.n_iter}  converged={fr.converged}  "
                  f"time={fr.runtime_s:.1f}s", file=sys.stderr, flush=True)
            print(f"    γ range [{rates.gain.min():.4g}, {rates.gain.max():.4g}]    "
                  f"λ range [{rates.dup.min():.4g}, {rates.dup.max():.4g}]",
                  file=sys.stderr, flush=True)
        history.append({"phase": i + 1, "kind": "brownian_MAP",
                        "sigma": float(sigma), "MAP_obj": fr.log_likelihood,
                        "iters": fr.n_iter, "converged": fr.converged,
                        "time_s": fr.runtime_s})
    return rates, history


# ----------------------------------------------------------------------------
# K-category LogisticShift mixture fitting
# ----------------------------------------------------------------------------


def _pack_mixture(rates: GLDRates, shift, fix_loss: bool, fix_root_length: bool
                  ) -> np.ndarray:
    """Pack (rates, shift) into a single flat parameter vector for scipy.

    Gauge-fixing: ``mod_p[0]`` and ``mod_q[0]`` are held at 0 (category 0
    is the reference, all shifts are relative to it).  Without this the
    optimizer can compensate a base-rate change with an equal-and-opposite
    ``mod_p[0]`` change, wandering through a flat direction in the joint
    parameter space.  Weight logits are softmax-normalized internally; the
    extra dof is harmless to L-BFGS-B.
    """
    K = shift.weights.shape[0]
    parts = [_rates_to_params(rates, fix_loss, fix_root_length)]
    w = shift.weights.detach().cpu().numpy()
    parts.append(np.log(np.maximum(w, 1e-300)))
    parts.append(shift.mod_p[1:].detach().cpu().numpy())  # mod_p[0] fixed at 0
    parts.append(shift.mod_q[1:].detach().cpu().numpy())  # mod_q[0] fixed at 0
    return np.concatenate(parts)


def _unpack_mixture(theta: np.ndarray, rates_template: GLDRates, K: int,
                    fix_loss: bool, fix_root_length: bool):
    """Inverse of :func:`_pack_mixture`."""
    from recount.rate_variation import LogisticShift
    import torch
    n_free_shift = 2 * (K - 1)  # mod_p[1..K-1], mod_q[1..K-1]
    n_rate = theta.size - K - n_free_shift
    rates = _params_to_rates(theta[:n_rate], rates_template, fix_loss, fix_root_length)
    off = n_rate
    log_w = theta[off:off + K]; off += K
    mod_p_free = theta[off:off + (K - 1)]; off += (K - 1)
    mod_q_free = theta[off:off + (K - 1)]
    mod_p = np.concatenate([[0.0], mod_p_free])
    mod_q = np.concatenate([[0.0], mod_q_free])
    log_w_norm = log_w - logsumexp_np(log_w)
    weights = np.exp(log_w_norm)
    dtype = torch.float64
    shift = LogisticShift(
        weights=torch.tensor(weights, dtype=dtype),
        mod_p=torch.tensor(mod_p, dtype=dtype),
        mod_q=torch.tensor(mod_q, dtype=dtype),
    )
    return rates, shift


def logsumexp_np(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def fit_mixture(tree: Tree, profiles: np.ndarray, *,
                K: int = 2,
                initial_rates: Optional[GLDRates] = None,
                initial_shift=None,
                fix_loss: bool = True, fix_root_length: bool = True,
                bound_dup_by_loss: bool = True, bound_gain_by_loss: bool = True,
                min_copies: int = 1, max_iter: int = 200, tol: float = 1e-6,
                verbose: bool = False) -> "MixtureFitResult":
    """Maximum-likelihood fit of a K-category LogisticShift mixture.

    Free parameters
    ---------------
    * Per-node rates: ``gain``, ``dup`` (and optionally ``loss``, ``length``)
      via the same log-space packing as :func:`fit_rates`.
    * Per-category log-weights (softmax-normalized internally).
    * Per-category ``mod_p`` and ``mod_q`` shifts.

    Gradient comes from autograd through ``mixture_log_likelihood``, which
    uses the per-family Python loop in ``recount.torch_backend`` for
    numerical stability at boundary parameters.  ~60× slower per
    iteration than :func:`fit_rates`; expect ~minute-scale fits on
    small subsets, not seconds.

    Returns
    -------
    MixtureFitResult
    """
    import torch
    from scipy.optimize import minimize
    from recount.rate_variation import LogisticShift, mixture_log_likelihood

    if initial_rates is None:
        initial_rates = default_initial_rates(tree)
    if initial_shift is None:
        # Mild asymmetry breaks the degenerate mixture symmetry at init.
        initial_shift = LogisticShift(
            weights=torch.full((K,), 1.0 / K, dtype=torch.float64),
            mod_p=torch.linspace(-0.3, 0.3, K, dtype=torch.float64),
            mod_q=torch.zeros(K, dtype=torch.float64),
        )

    profiles_t = torch.tensor(profiles, dtype=torch.long)
    n_rate_params = _rates_to_params(initial_rates, fix_loss, fix_root_length).size
    theta0 = _pack_mixture(initial_rates, initial_shift, fix_loss, fix_root_length)
    rate_bounds = _build_bounds(initial_rates, fix_loss, fix_root_length,
                                bound_dup_by_loss, bound_gain_by_loss)
    # Weight logits and shift params are unbounded.
    bounds = list(rate_bounds) + [(None, None)] * (K + 2 * (K - 1))
    for i, (_, hi) in enumerate(bounds):
        if hi is not None and theta0[i] > hi:
            theta0[i] = hi - 1e-6
    history: list = []
    iters = {"k": 0}

    def f_and_g(theta: np.ndarray):
        theta = np.clip(theta, -40.0, 40.0)
        rates_np = _params_to_rates(theta[:n_rate_params], initial_rates,
                                    fix_loss, fix_root_length)
        off = n_rate_params
        log_w = theta[off:off + K]; off += K
        mod_p_free = theta[off:off + (K - 1)]; off += (K - 1)
        mod_q_free = theta[off:off + (K - 1)]
        log_w_t = torch.tensor(log_w, dtype=torch.float64, requires_grad=True)
        weights_t = torch.softmax(log_w_t, dim=0)
        mod_p_free_t = torch.tensor(mod_p_free, dtype=torch.float64, requires_grad=True)
        mod_q_free_t = torch.tensor(mod_q_free, dtype=torch.float64, requires_grad=True)
        # Gauge-fix category 0 at identity (mod_p[0] = mod_q[0] = 0): without
        # this the optimizer wanders along a flat direction where a base-rate
        # shift exactly cancels a mod_p[0] shift.
        mod_p_t = torch.cat([torch.zeros(1, dtype=torch.float64), mod_p_free_t])
        mod_q_t = torch.cat([torch.zeros(1, dtype=torch.float64), mod_q_free_t])
        gain_t = torch.tensor(rates_np.gain, dtype=torch.float64, requires_grad=True)
        loss_t = torch.tensor(rates_np.loss, dtype=torch.float64, requires_grad=True)
        dup_t = torch.tensor(rates_np.dup, dtype=torch.float64, requires_grad=True)
        length_t = torch.tensor(rates_np.length, dtype=torch.float64, requires_grad=True)
        shift = LogisticShift(weights=weights_t, mod_p=mod_p_t, mod_q=mod_q_t)
        try:
            LL = mixture_log_likelihood(tree, gain_t, loss_t, dup_t, length_t,
                                        shift, profiles_t, min_copies=min_copies)
            LL_val = float(LL.detach())
            if not np.isfinite(LL_val):
                raise ValueError(f"non-finite LL {LL_val}")
            LL.backward()
        except Exception as e:
            if verbose:
                print(f"  iter {iters['k']}: infeasible ({e}); signalling -1e12",
                      file=sys.stderr)
            iters["k"] += 1
            history.append((iters["k"], -1e12))
            return 1e12, np.zeros_like(theta)

        g_gain = gain_t.grad.detach().cpu().numpy()
        g_loss = loss_t.grad.detach().cpu().numpy()
        g_dup = dup_t.grad.detach().cpu().numpy()
        g_len = length_t.grad.detach().cpu().numpy()
        grad_rate_parts = []
        for v in range(tree.num_nodes):
            grad_rate_parts.append(g_gain[v] * rates_np.gain[v])
            if not fix_loss:
                grad_rate_parts.append(g_loss[v] * rates_np.loss[v])
            grad_rate_parts.append(g_dup[v] * rates_np.dup[v])
            if not (v == tree.root and fix_root_length):
                gl = g_len[v] if np.isfinite(g_len[v]) else 0.0
                grad_rate_parts.append(gl * rates_np.length[v]
                                       if np.isfinite(rates_np.length[v]) else 0.0)
        grad_rate = np.array(grad_rate_parts, dtype=np.float64)
        grad_w = log_w_t.grad.detach().cpu().numpy()
        grad_p_free = mod_p_free_t.grad.detach().cpu().numpy()
        grad_q_free = mod_q_free_t.grad.detach().cpu().numpy()
        grad = np.concatenate([grad_rate, grad_w, grad_p_free, grad_q_free])
        grad = np.where(np.isfinite(grad), grad, 0.0)

        iters["k"] += 1
        history.append((iters["k"], LL_val))
        if verbose and iters["k"] % 5 == 0:
            w_str = ", ".join(f"{w:.3f}" for w in weights_t.detach().tolist())
            print(f"  iter {iters['k']:>4d}  LL = {LL_val:.6f}  w=[{w_str}]",
                  file=sys.stderr, flush=True)
        return -LL_val, -grad

    if verbose:
        print(f"# mixture fit: {theta0.size} free params "
              f"({n_rate_params} rate + {K} log_w + 2·{K - 1} shift)",
              file=sys.stderr, flush=True)
    t0 = time.time()
    res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "gtol": tol, "disp": False})
    runtime = time.time() - t0
    rates_out, shift_out = _unpack_mixture(res.x, initial_rates, K,
                                            fix_loss, fix_root_length)
    return MixtureFitResult(
        rates=rates_out, shift=shift_out,
        log_likelihood=-res.fun, n_iter=int(res.nit),
        converged=bool(res.success), message=str(res.message),
        runtime_s=runtime, history=history,
    )
