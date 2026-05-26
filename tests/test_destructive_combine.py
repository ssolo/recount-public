"""Tests for the O(W²) destructive sibling-combine path.

Covers the pure-PyTorch implementation (auto-skips Triton tests unless CUDA
+ triton are available). All tests use fp64 to compare against the
closed-form reference at machine precision.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from recount.destructive_combine import (
    destructive_combine,
    destructive_combine_pytorch,
    triton_available,
)
from recount.torch_fast import (
    _build_log_split,
    _build_sibling_params,
    _log_fact_table,
    _vectorized_combine,
    corrected_log_likelihood_fast,
    gradient_fast,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _make_inputs(W: int, F: int, seed: int = 0, dtype=torch.float64):
    """Random k1, k2 + a consistent set of (log_split, sibling_params) for one node."""
    torch.manual_seed(seed)
    k1 = torch.randn(F, W + 1, dtype=dtype)
    k2 = torch.randn(F, W + 1, dtype=dtype)

    # Pick valid sibling probabilities in (0, 1)
    p_j1 = torch.tensor(0.35, dtype=dtype)
    p_c_j1 = torch.tensor(0.65, dtype=dtype)
    p_j2 = torch.tensor(0.42, dtype=dtype)
    p_c_j2 = torch.tensor(0.58, dtype=dtype)
    eps_c_parent = torch.tensor(1.0 - 0.35 * 0.42, dtype=dtype)

    log_fact = _log_fact_table(W, dtype, torch.device("cpu"))
    log_split = _build_log_split(p_j1, p_c_j1, p_j2, p_c_j2, eps_c_parent, W, log_fact)
    log_e, log_e_c, logp1, logp2 = _build_sibling_params(
        p_j1, p_c_j1, p_j2, p_c_j2, eps_c_parent,
    )
    return k1, k2, log_e, log_e_c, logp1, logp2, log_fact, log_split


# ----------------------------------------------------------------------------
# 1. PyTorch matches closed-form on random inputs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("W", [4, 8, 16])
def test_pytorch_matches_closed_form_random(W):
    F = 4
    k1, k2, log_e, log_e_c, logp1, logp2, log_fact, log_split = _make_inputs(W, F)
    C_closed = _vectorized_combine(k1, k2, log_split, chunk_F=F)
    C_destr = destructive_combine_pytorch(k1, k2, log_e, log_e_c, logp1, logp2, log_fact)
    assert C_closed.shape == C_destr.shape == (F, W + 1)
    max_abs = (C_closed - C_destr).abs().max().item()
    assert max_abs < 1e-10, f"max |diff| = {max_abs}"


# ----------------------------------------------------------------------------
# 2. Extinct-sibling edge case (small eps so log_e is large-negative)
# ----------------------------------------------------------------------------


def test_pytorch_matches_closed_form_near_extinct():
    """p_j1 very small → log_e ≈ −∞-ish; verify no NaN and still matches closed-form."""
    dtype = torch.float64
    W, F = 6, 3
    torch.manual_seed(1)
    k1 = torch.randn(F, W + 1, dtype=dtype)
    k2 = torch.randn(F, W + 1, dtype=dtype)
    p_j1 = torch.tensor(1e-12, dtype=dtype)
    p_c_j1 = torch.tensor(1.0 - 1e-12, dtype=dtype)
    p_j2 = torch.tensor(0.5, dtype=dtype)
    p_c_j2 = torch.tensor(0.5, dtype=dtype)
    eps_c_parent = torch.tensor(1.0 - 1e-12 * 0.5, dtype=dtype)
    log_fact = _log_fact_table(W, dtype, torch.device("cpu"))
    log_split = _build_log_split(p_j1, p_c_j1, p_j2, p_c_j2, eps_c_parent, W, log_fact)
    le, lec, lp1, lp2 = _build_sibling_params(p_j1, p_c_j1, p_j2, p_c_j2, eps_c_parent)

    C_closed = _vectorized_combine(k1, k2, log_split, chunk_F=F)
    C_destr = destructive_combine_pytorch(k1, k2, le, lec, lp1, lp2, log_fact)
    assert torch.isfinite(C_destr).all(), "destructive output has non-finite values"
    assert torch.allclose(C_closed, C_destr, atol=1e-10, rtol=1e-10)


# ----------------------------------------------------------------------------
# 3. Gradcheck on the PyTorch path
# ----------------------------------------------------------------------------


def test_gradcheck_pytorch():
    """torch.autograd.gradcheck on a tiny case (small W keeps finite-diff stable)."""
    dtype = torch.float64
    W, F = 3, 2
    k1, k2, le, lec, lp1, lp2, log_fact, _ = _make_inputs(W, F, seed=2, dtype=dtype)
    inputs = tuple(t.detach().clone().requires_grad_(True) for t in (k1, k2, le, lec, lp1, lp2))
    # log_fact is not differentiable input — pass as non-grad arg via closure
    fn = lambda *args: destructive_combine_pytorch(*args, log_fact)
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)


# ----------------------------------------------------------------------------
# 4–5. End-to-end on the 4-leaf fixture (compare to Java reference)
# ----------------------------------------------------------------------------


def test_end_to_end_4leaf_forward(tree, torch_rates_or_inline, torch_profiles_or_inline, java_reference):
    """Forward LL via destructive path matches Java to 1e-10."""
    dt = torch.float64
    g = torch.tensor(torch_rates_or_inline["gain"], dtype=dt)
    l = torch.tensor(torch_rates_or_inline["loss"], dtype=dt)
    d = torch.tensor(torch_rates_or_inline["dup"], dtype=dt)
    t = torch.tensor(torch_rates_or_inline["length"], dtype=dt)
    p = torch.tensor(torch_profiles_or_inline, dtype=torch.long)
    W = int(p.sum(dim=1).max().item()) + 1
    ll = corrected_log_likelihood_fast(
        tree, g, l, d, t, p, min_copies=1, W=W, use_destructive=True,
    )
    assert float(ll) == pytest.approx(java_reference["ll_corrected_min1"], abs=1e-10)


def test_end_to_end_4leaf_gradient(tree, torch_rates_or_inline, torch_profiles_or_inline, java_reference):
    """Gain gradient via destructive path matches Java distribution gradient."""
    dt = torch.float64
    g = torch.tensor(torch_rates_or_inline["gain"], dtype=dt)
    l = torch.tensor(torch_rates_or_inline["loss"], dtype=dt)
    d = torch.tensor(torch_rates_or_inline["dup"], dtype=dt)
    t = torch.tensor(torch_rates_or_inline["length"], dtype=dt)
    p = torch.tensor(torch_profiles_or_inline, dtype=torch.long)
    W = int(p.sum(dim=1).max().item()) + 1
    LL, grads = gradient_fast(
        tree, g, l, d, t, p, min_copies=1, W=W, use_destructive=True,
    )
    expected_gain = java_reference["gradient_distribution"][:, 0]
    diff = grads["gain"].numpy() - expected_gain
    assert float(np.max(np.abs(diff))) < 1e-10


def test_gradient_finite_with_padded_neg_inf():
    """Regression test: destructive must NOT produce NaN gradients when the
    inputs have -inf padding (the usual case when widths < W+1).

    Without ``_safe_logaddexp``, ``torch.logaddexp(-inf, -inf)`` returns
    -inf in forward but NaN in backward (``exp(-inf − -inf) = exp(NaN) = NaN``).
    The NaN poisons the whole gradient and silently breaks ML fitting.
    """
    dt = torch.float64
    W, F = 8, 4
    torch.manual_seed(123)
    # k1 has valid entries only at t = 0..3 (rest -inf, like a profile of width 4)
    k1 = torch.full((F, W + 1), float("-inf"), dtype=dt)
    k1[:, :4] = torch.randn(F, 4, dtype=dt)
    k2 = torch.full((F, W + 1), float("-inf"), dtype=dt)
    k2[:, :4] = torch.randn(F, 4, dtype=dt)
    k1 = k1.requires_grad_(True)
    k2 = k2.requires_grad_(True)
    le = torch.tensor(-0.7, dtype=dt, requires_grad=True)
    lec = torch.tensor(-0.7, dtype=dt, requires_grad=True)
    lp1 = torch.tensor(-0.5, dtype=dt, requires_grad=True)
    lp2 = torch.tensor(-0.3, dtype=dt, requires_grad=True)
    log_fact = torch.lgamma(torch.arange(W + 1, dtype=dt) + 1.0)
    C = destructive_combine_pytorch(k1, k2, le, lec, lp1, lp2, log_fact)
    # Sum only the valid (finite) entries so the loss doesn't trivially -inf out
    loss = torch.where(torch.isfinite(C), C, torch.zeros_like(C)).sum()
    loss.backward()
    for name, g_ in [("k1", k1.grad), ("k2", k2.grad),
                     ("log_e", le.grad), ("log_e_c", lec.grad),
                     ("logp1", lp1.grad), ("logp2", lp2.grad)]:
        assert torch.isfinite(g_).all(), f"{name} grad has NaN/Inf: {g_}"


# Inline fixtures so we don't depend on torch_rates / torch_profiles fixtures
# (they live in test_torch_backend.py and may not be visible here).
@pytest.fixture
def torch_rates_or_inline(rates):
    return dict(
        gain=rates.gain, loss=rates.loss, dup=rates.dup, length=rates.length,
    )


@pytest.fixture
def torch_profiles_or_inline(profiles):
    return profiles


# ----------------------------------------------------------------------------
# 6. Dispatcher picks PyTorch on CPU
# ----------------------------------------------------------------------------


def test_dispatcher_uses_pytorch_on_cpu():
    """On CPU we always end up in the PyTorch path (no triton on non-CUDA tensors)."""
    W, F = 4, 2
    k1, k2, le, lec, lp1, lp2, log_fact, _ = _make_inputs(W, F, seed=3)
    # CPU tensors → dispatcher should match the pytorch impl byte-for-byte.
    C_dispatch = destructive_combine(k1, k2, le, lec, lp1, lp2, log_fact)
    C_pytorch = destructive_combine_pytorch(k1, k2, le, lec, lp1, lp2, log_fact)
    assert torch.equal(C_dispatch, C_pytorch)


# ----------------------------------------------------------------------------
# 7. Triton kernel matches PyTorch (skipped unless CUDA + triton available)
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not (torch.cuda.is_available() and triton_available()),
    reason="requires CUDA and triton",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triton_kernel_matches_pytorch(dtype):
    from recount.destructive_combine import destructive_combine_triton
    W, F = 16, 8
    k1_cpu, k2_cpu, le, lec, lp1, lp2, log_fact, _ = _make_inputs(W, F, seed=4, dtype=dtype)
    C_pytorch = destructive_combine_pytorch(k1_cpu, k2_cpu, le, lec, lp1, lp2, log_fact)
    cuda = torch.device("cuda")
    C_triton = destructive_combine_triton(
        k1_cpu.to(cuda), k2_cpu.to(cuda),
        le.to(cuda), lec.to(cuda), lp1.to(cuda), lp2.to(cuda),
        log_fact.to(cuda),
    ).cpu()
    atol = 1e-10 if dtype == torch.float64 else 1e-5
    assert torch.allclose(C_pytorch, C_triton, atol=atol, rtol=atol)
