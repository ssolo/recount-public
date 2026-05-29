"""Native CPU backend tests — same fixtures as test_torch_backend, but the
target is bit-exact agreement with Java since the native code is a literal
C port of recount.gld (same algorithm + same summation order).
"""
import numpy as np
import pytest

try:
    from recount.native_backend import (
        corrected_log_likelihood_native,
        gradient_native,
        log_likelihood_native,
        native_version,
        per_branch_stats_native,
    )
    _NATIVE_AVAILABLE = True
except ImportError as e:
    _NATIVE_AVAILABLE = False
    _NATIVE_ERR = str(e)


pytestmark = pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason=f"native backend not built ({_NATIVE_ERR if not _NATIVE_AVAILABLE else ''})",
)


def test_native_version():
    v = native_version()
    assert v.startswith("recount-native")


def test_native_raw_ll_matches_java(tree, rates, profiles, java_reference):
    per_fam = log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32))
    expected = java_reference["ll_per_family"]
    diff = per_fam - expected
    assert float(np.max(np.abs(diff))) < 1e-10
    assert float(per_fam.sum()) == pytest.approx(java_reference["ll_raw"], abs=1e-10)


def test_native_corrected_ll_matches_java(tree, rates, profiles, java_reference):
    ll = corrected_log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), min_copies=1)
    assert ll == pytest.approx(java_reference["ll_corrected_min1"], abs=1e-10)


def test_native_threading_invariant(tree, rates, profiles):
    """Single-thread vs multi-thread must give bit-identical results."""
    per_fam_1 = log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), num_threads=1)
    per_fam_8 = log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), num_threads=8)
    assert np.array_equal(per_fam_1, per_fam_8), \
        f"threading changed result: {per_fam_1} vs {per_fam_8}"


def test_native_matches_numpy(tree, rates, profiles):
    """Native and NumPy share the same algorithm (destructive sibling combine)
    and the same summation order — they should agree to machine precision."""
    from recount.gld import corrected_log_likelihood
    ll_np = corrected_log_likelihood(tree, rates, profiles, min_copies=1)
    ll_native = corrected_log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), min_copies=1)
    assert ll_native == pytest.approx(ll_np, abs=1e-12)


def test_native_gradient_matches_java(tree, rates, profiles, java_reference):
    """Analytical gradient via native inside-outside matches Java's
    Gradient.getCorrectedGradient() to machine precision on the 4-leaf
    fixture. Java reference is the survival-parameterization gradient
    (∂LL_corr/∂(p̃, q̃, r̃/κ̃)) — see conftest.JAVA_GRADIENT_SURVIVAL."""
    LL, grad = gradient_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), min_copies=1)
    grad = grad.reshape(tree.num_nodes, 3)
    expected = java_reference["gradient_survival"]
    diff = np.abs(grad - expected)
    assert float(np.max(diff)) < 1e-10, (
        f"max gradient diff = {np.max(diff):.4e}"
    )
    assert LL == pytest.approx(java_reference["ll_corrected_min1"], abs=1e-10)


def test_native_gradient_matches_numpy(tree, rates, profiles):
    """Native gradient matches NumPy reference (gld.gradient_survival)
    to machine precision on the fixture (same algorithm + summation order)."""
    from recount.gld import gradient_survival
    grad_np = gradient_survival(tree, rates, profiles, min_copies=1).reshape(tree.num_nodes, 3)
    _, grad_native = gradient_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32), min_copies=1)
    grad_native = grad_native.reshape(tree.num_nodes, 3)
    assert np.allclose(grad_native, grad_np, atol=1e-10), \
        f"max |native - numpy| = {np.max(np.abs(grad_native - grad_np)):.4e}"


def test_native_unobserved_l0_matches_numpy(tree, rates, profiles):
    """Native log L(0) (Csurös SI Thms 3-5, arbitrary Ωmin) matches the
    NumPy empty+singleton ground truth for Ωmin ∈ {1, 2}, and extends to
    Ωmin ≥ 3 where the shipped Count CountXXV.jar throws.
    """
    from recount.gld import empty_log_likelihood, singleton_log_likelihood
    from recount.native_backend import unobserved_logL0_native

    # Ωmin=1: matches empty_log_likelihood exactly
    L0_1_numpy = float(empty_log_likelihood(tree, rates))
    L0_1_native = unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, 1)
    assert L0_1_native == pytest.approx(L0_1_numpy, abs=1e-12), \
        f"Ωmin=1: native={L0_1_native}, numpy_empty={L0_1_numpy}"

    # Ωmin=2: matches log(exp(empty)+exp(singleton)) — but NumPy singleton
    # accumulates ~3e-5 numerical error across the per-leaf forward passes,
    # so we use a looser tolerance and trust that native matches Java to
    # machine precision (separately verified).
    L0_empty = empty_log_likelihood(tree, rates)
    L0_single = singleton_log_likelihood(tree, rates)
    L0_2_numpy = float(np.logaddexp(L0_empty, L0_single))
    L0_2_native = unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, 2)
    assert L0_2_native == pytest.approx(L0_2_numpy, abs=1e-3), \
        f"Ωmin=2: native={L0_2_native}, numpy={L0_2_numpy}"

    # Ωmin=k for k ≥ 3 is novel — just sanity-check monotonicity:
    # log L(0) should INCREASE (toward log 1 = 0) as Ωmin grows.
    L0_prev = L0_2_native
    for k in (3, 4, 5):
        L0_k = unobserved_logL0_native(
            tree, rates.gain, rates.loss, rates.dup, rates.length, k)
        assert L0_k > L0_prev, (
            f"L(0) should be monotone increasing in Ωmin: got L0[{k}]={L0_k} "
            f"<= L0[{k-1}]={L0_prev}"
        )
        L0_prev = L0_k


def test_native_events_match_numpy(tree, rates, profiles):
    """Per-branch event counts via native match the NumPy reference
    (recount.events.per_branch_stats) on the 4-leaf fixture to ~1e-12."""
    from recount.events import per_branch_stats
    np_stats = per_branch_stats(tree, rates, profiles)
    stats = per_branch_stats_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles.astype(np.int32))
    for key in ("copies_node", "copies_edge", "gain_events", "loss_events"):
        diff = np.max(np.abs(stats[key] - getattr(np_stats, key)))
        assert diff < 1e-12, f"{key}: max |native - numpy| = {diff:.4e}"
    # num_families_active is an integer count; must match exactly.
    assert np.array_equal(
        stats["num_families_active"], np_stats.num_families_active
    ), "num_families_active disagreement"
