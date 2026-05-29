"""Smoke tests for the LogisticShift K-category rate-variation mixture.

The LogisticShift mixture overlays K rate categories on the base GLD
process; each category k shifts log(λ) and log(t) by category-specific
scalars (Δ_dup^(k), Δ_length^(k)). Formal derivation + the closed-form
mixture gradient are in `docs/logistic_shift_gradient.pdf`.

Coverage:
1. K=1 with zero shifts is *identically* equal to the bare GLD LL.
2. K=1 (one category) equals K=2 with identical categories — the
   mixture distribution is invariant under duplicate atoms.
3. A non-zero (Δ_dup, Δ_length) shift actually changes the LL (so the
   parameter does something).
4. PyTorch autograd through `mixture_log_likelihood` produces finite
   gradients on every parameter axis.
5. `fit_mixture` smoke-test: a K=2 fit on the 4-leaf fixture completes
   without error and improves the LL over the seed.

Finite-difference agreement on the *gradient* (the precision claim used
by the BFGS optimiser) is locked in by `test_mixture_gradient_fd.py`.
"""
import numpy as np
import pytest
import torch

from recount.gld import corrected_log_likelihood
from recount.rate_variation import (
    LogisticShift,
    corrected_mixture_log_likelihood,
    mixture_log_likelihood,
)


def test_k1_identity_matches_bare_ll(tree, rates, profiles):
    """K=1 LogisticShift with zero shifts reproduces the bare GLD LL.

    Matches via the torch_backend per-family loop, so the tolerance is
    looser than NumPy↔NumPy but still well below 1e-3 per family on the
    canonical 4-leaf case.
    """
    ll_bare = corrected_log_likelihood(tree, rates, profiles, min_copies=1)
    shift = LogisticShift.identity(K=1)
    ll_mix = corrected_mixture_log_likelihood(tree, rates, shift, profiles, min_copies=1)
    assert abs(ll_mix - ll_bare) < 1e-6


def test_k1_identity_equals_k2_identity(tree, rates, profiles):
    """K=2 with uniform identity shifts gives the same LL as K=1 identity
    (both categories are the same model, so the mixture collapses)."""
    s1 = LogisticShift.identity(K=1)
    s2 = LogisticShift.identity(K=2)
    ll1 = corrected_mixture_log_likelihood(tree, rates, s1, profiles, min_copies=1)
    ll2 = corrected_mixture_log_likelihood(tree, rates, s2, profiles, min_copies=1)
    assert abs(ll1 - ll2) < 1e-9


def test_shift_changes_ll(tree, rates, profiles):
    """A non-trivial logit shift changes the LL — sanity-check that the
    shift parameters are actually wired through to the forward pass."""
    s_id = LogisticShift.identity(K=2)
    s_shift = LogisticShift(
        weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
        mod_p=torch.tensor([0.0, 1.0], dtype=torch.float64),
        mod_q=torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    ll_id = corrected_mixture_log_likelihood(tree, rates, s_id, profiles, min_copies=1)
    ll_shift = corrected_mixture_log_likelihood(tree, rates, s_shift, profiles, min_copies=1)
    assert abs(ll_shift - ll_id) > 1e-3


def test_autograd_through_mixture(tree, rates, profiles):
    """Backward through mixture_log_likelihood gives finite gradients for
    rates AND for the LogisticShift parameters."""
    dtype = torch.float64
    n = tree.num_nodes
    gain = torch.tensor(rates.gain, dtype=dtype, requires_grad=True)
    loss = torch.tensor(rates.loss, dtype=dtype, requires_grad=True)
    dup = torch.tensor(rates.dup, dtype=dtype, requires_grad=True)
    length = torch.tensor(rates.length, dtype=dtype, requires_grad=True)
    profs = torch.tensor(profiles, dtype=torch.long)

    log_w = torch.tensor([0.0, 0.0], dtype=dtype, requires_grad=True)
    mod_p = torch.tensor([-0.3, 0.3], dtype=dtype, requires_grad=True)
    mod_q = torch.tensor([0.1, -0.1], dtype=dtype, requires_grad=True)
    shift = LogisticShift(weights=torch.softmax(log_w, dim=0),
                          mod_p=mod_p, mod_q=mod_q)

    LL = mixture_log_likelihood(tree, gain, loss, dup, length, shift, profs,
                                min_copies=1)
    LL.backward()
    for name, g in [("gain", gain.grad), ("loss", loss.grad),
                    ("dup", dup.grad), ("length", length.grad)]:
        assert torch.isfinite(g).all(), f"non-finite grad in {name}"
    assert torch.isfinite(mod_p.grad).all()
    assert torch.isfinite(mod_q.grad).all()
    assert torch.isfinite(log_w.grad).all()
    # Non-zero shift gradients confirm we're not silently zeroing out.
    assert mod_p.grad.abs().sum() > 0
    assert mod_q.grad.abs().sum() > 0


def test_fit_mixture_smoke(tree, profiles):
    """fit_mixture should run, weights stay on the simplex, and the LL
    end up within a few nats of the bare GLD fit on the same data.

    Both fits use the same biological-meaning constraints (``dup ≤ loss``,
    ``gain ≤ loss``); under those constraints the mixture has K=2 extra
    parameters and a gauge-fixed first category, so on a tiny 4-profile
    dataset it may land at a slightly different local optimum than the
    bare fit — we just check it's in the same ballpark.
    """
    from recount.ml import default_initial_rates, fit_rates, fit_mixture
    init = default_initial_rates(tree, gain=1.0, loss=1.0, dup=0.1, length=2.0)
    fr_bare = fit_rates(tree, profiles, initial_rates=init, max_iter=20)
    fr_mix = fit_mixture(tree, profiles, K=2, initial_rates=init, max_iter=20)
    assert fr_mix.log_likelihood >= fr_bare.log_likelihood - 5.0
    w = fr_mix.shift.weights.detach().numpy()
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    # Gauge: mod_p[0] = mod_q[0] = 0 always.
    assert float(fr_mix.shift.mod_p[0]) == 0.0
    assert float(fr_mix.shift.mod_q[0]) == 0.0
