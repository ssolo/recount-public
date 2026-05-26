"""Smoke tests for the ML optimizer and per-branch events.

These tests do not lock in numerical values against an external
reference; they confirm that the optimizer + per-branch event
computation don't regress at the API level on the 4-leaf fixture:

- `default_initial_rates` produces a usable starting point with the
  expected shape and the root edge length set to +∞;
- `fit_rates` improves the corrected log-likelihood relative to its
  starting point on the 4-leaf fixture;
- `events` returns the expected dict of arrays (gain_events,
  loss_events, copies_node, families_present, …) with the right
  shapes and finite values.

Bit-perfect agreement against Java + finite-difference checks on the
analytical gradient are in `test_likelihood.py`, `test_gradient.py`,
and `test_native_backend.py`.
"""
import numpy as np
import pytest

from recount import (
    Tree, GLDRates,
    corrected_log_likelihood, log_likelihood,
)


def test_default_initial_rates_runs(tree):
    from recount.ml import default_initial_rates
    init = default_initial_rates(tree)
    assert init.gain.shape == (tree.num_nodes,)
    assert np.isinf(init.length[tree.root])


def test_ml_improves_LL(tree, profiles):
    """ML should improve the LL from a poor starting point."""
    from recount.ml import default_initial_rates, fit_rates

    # Start from clearly-suboptimal rates
    init = default_initial_rates(tree, gain=1.0, loss=2.0, dup=0.1, length=2.0)
    ll0 = corrected_log_likelihood(tree, init, profiles, min_copies=1)
    fr = fit_rates(tree, profiles, initial_rates=init, max_iter=50, tol=1e-5)
    assert fr.log_likelihood > ll0
    # No NaN, all rates positive
    assert np.all(np.isfinite(fr.rates.gain))
    assert np.all(fr.rates.gain > 0)
    assert np.all(fr.rates.loss > 0)


def test_events_basic(tree, rates, profiles):
    """Per-branch stats should have sane shapes and non-negative values."""
    from recount.events import per_branch_stats
    stats = per_branch_stats(tree, rates, profiles)
    n = tree.num_nodes
    for arr in (stats.copies_node, stats.copies_edge, stats.gain_events,
                stats.loss_events, stats.gain, stats.loss, stats.dup):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr[~np.isinf(arr)]))
    # # copies and active counts are non-negative
    assert (stats.copies_node >= 0).all()
    assert (stats.num_families_active >= 0).all()
    assert (stats.num_families_active <= profiles.shape[0]).all()
