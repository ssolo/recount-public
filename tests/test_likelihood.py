"""Forward log-likelihood tests against the Java reference."""
import numpy as np
import pytest

from recount import (
    compute_survival_params,
    corrected_log_likelihood,
    empty_log_likelihood,
    log_likelihood,
    singleton_log_likelihood,
)
from recount.gld import _forward, _make_caches


def test_raw_log_likelihood(tree, rates, profiles, java_reference):
    ll = log_likelihood(tree, rates, profiles)
    assert ll == pytest.approx(java_reference["ll_raw"], abs=1e-10)


def test_corrected_log_likelihood_min1(tree, rates, profiles, java_reference):
    cll = corrected_log_likelihood(tree, rates, profiles, min_copies=1)
    assert cll == pytest.approx(java_reference["ll_corrected_min1"], abs=1e-10)


def test_empty_log_likelihood(tree, rates, java_reference):
    ll0 = empty_log_likelihood(tree, rates)
    assert ll0 == pytest.approx(java_reference["ll_empty"], abs=1e-10)


def test_singleton_log_likelihood(tree, rates, java_reference):
    ll1 = singleton_log_likelihood(tree, rates)
    assert ll1 == pytest.approx(java_reference["ll_singleton"], abs=1e-10)


def test_per_family_log_likelihood(tree, rates, profiles, java_reference):
    sp = compute_survival_params(tree, rates)
    fact, rfacts = _make_caches(tree, sp, profiles)
    for f in range(profiles.shape[0]):
        pc = _forward(tree, sp, profiles[f], fact, rfacts)
        assert pc.LL == pytest.approx(java_reference["ll_per_family"][f], abs=1e-10)


def test_corrected_equals_raw_with_min_copies_zero(tree, rates, profiles):
    """min_copies=0 disables the correction, returning the raw LL."""
    raw = log_likelihood(tree, rates, profiles)
    cll = corrected_log_likelihood(tree, rates, profiles, min_copies=0)
    assert cll == pytest.approx(raw, abs=1e-13)


def test_corrected_with_min_copies_2(tree, rates, profiles):
    """min_copies=2 corrects for both empty AND singleton profiles."""
    raw = log_likelihood(tree, rates, profiles)
    ll0 = empty_log_likelihood(tree, rates)
    ll1 = singleton_log_likelihood(tree, rates)
    F = profiles.shape[0]
    expected = raw - F * np.log(-np.expm1(np.logaddexp(ll0, ll1)))
    cll = corrected_log_likelihood(tree, rates, profiles, min_copies=2)
    assert cll == pytest.approx(expected, abs=1e-12)


def test_invalid_min_copies(tree, rates, profiles):
    with pytest.raises(ValueError):
        corrected_log_likelihood(tree, rates, profiles, min_copies=3)
