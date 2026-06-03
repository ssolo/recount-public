"""PyTorch backend tests (`recount.torch_backend`).

The torch backend is kept primarily as an autograd-validated cross-
check for the analytical C gradient; it is no longer on the production
path (see `recount/torch_fast.md`).

Coverage:
1. Forward log-likelihood matches Java reference values on the 4-leaf
   fixture (within machine precision).
2. Corrected log-likelihood matches Java at Ωmin=1.
3. Reverse-mode autograd through `corrected_log_likelihood_t` gives a
   per-rate gradient that agrees with the Java analytical gradient on
   the gain axis to ~1e-12.
4. The autograd gradient has the expected per-axis shape on every
   {gain, loss, dup, length} block.
5. ∂LL/∂t_root = 0 — the root has no incoming edge; its length is
   fixed at +∞ by convention.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from recount import gradient_survival  # NumPy analytical for cross-check
from recount.torch_backend import (
    corrected_log_likelihood_t,
    gradient_autograd,
    log_likelihood_t,
)


@pytest.fixture
def torch_rates(rates):
    """Same rates as the NumPy fixture, but as float64 torch tensors."""
    dt = torch.float64
    return dict(
        gain=torch.tensor(rates.gain, dtype=dt),
        loss=torch.tensor(rates.loss, dtype=dt),
        dup=torch.tensor(rates.dup, dtype=dt),
        length=torch.tensor(rates.length, dtype=dt),
    )


@pytest.fixture
def torch_profiles(profiles):
    return torch.tensor(profiles, dtype=torch.long)


def test_log_likelihood_matches_java(tree, torch_rates, torch_profiles, java_reference):
    ll = log_likelihood_t(tree, **torch_rates, profiles=torch_profiles)
    assert float(ll) == pytest.approx(java_reference["ll_raw"], abs=1e-10)


def test_corrected_log_likelihood_matches_java(tree, torch_rates, torch_profiles, java_reference):
    cll = corrected_log_likelihood_t(
        tree, **torch_rates, profiles=torch_profiles, min_copies=1,
    )
    assert float(cll) == pytest.approx(java_reference["ll_corrected_min1"], abs=1e-10)


def test_autograd_gain_matches_java_distribution(tree, torch_rates, torch_profiles, java_reference):
    """PyTorch autograd is w.r.t. the RAW rate parameters (gain_raw = κ for
    Pólya, r for Poisson). For Pólya nodes κ̃=κ so the value matches both
    Java's survival- and distribution-gain gradients; for the Poisson root,
    r̃ = r·(1-ε), so autograd gives the distribution gradient — the value
    chain-ruled by (1-ε)."""
    grads = gradient_autograd(tree, **torch_rates, profiles=torch_profiles, min_copies=1)
    expected_gain = java_reference["gradient_distribution"][:, 0]
    diff = grads["gain"].numpy() - expected_gain
    assert float(np.max(np.abs(diff))) < 1e-12


def test_autograd_dimensions(tree, torch_rates, torch_profiles):
    grads = gradient_autograd(tree, **torch_rates, profiles=torch_profiles, min_copies=1)
    for k in ("gain", "loss", "dup", "length"):
        assert grads[k].shape == (tree.num_nodes,)


def test_autograd_root_length_grad_is_zero(tree, torch_rates, torch_profiles):
    """Root edge length is +∞ → its gradient must be 0 (the LL doesn't move
    when you change the root's "length")."""
    grads = gradient_autograd(tree, **torch_rates, profiles=torch_profiles, min_copies=1)
    assert float(grads["length"][tree.root]) == 0.0
