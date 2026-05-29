"""Analytical gradient tests against the Java reference."""
import numpy as np
import pytest

from recount import GAIN, LOSS, DUP, gradient_survival


def test_gradient_survival_matches_java(tree, rates, profiles, java_reference):
    """The full inside-outside analytical gradient should match Java to ~1e-12
    on every entry, including the root (where p̃=1, so the LOSS and DUP
    entries are 0 by convention)."""
    g = gradient_survival(tree, rates, profiles, min_copies=1)
    expected = java_reference["gradient_survival"].flatten()
    diff = g - expected
    max_abs = float(np.max(np.abs(diff)))
    assert max_abs < 1e-11, f"max |Δ| = {max_abs:.2e}; per-entry diffs = {diff}"


def test_gradient_index_layout(tree, rates, profiles):
    """Gradient is flat with layout 3*v + {GAIN=0, LOSS=1, DUP=2}."""
    g = gradient_survival(tree, rates, profiles, min_copies=1)
    assert g.shape == (3 * tree.num_nodes,)
    assert GAIN == 0 and LOSS == 1 and DUP == 2


def test_gradient_root_loss_dup_zero(tree, rates, profiles):
    """Root has p̃=1 (length=∞) → LOSS and DUP gradients are 0 by convention."""
    g = gradient_survival(tree, rates, profiles, min_copies=1)
    root = tree.root
    assert g[3 * root + LOSS] == 0.0
    assert g[3 * root + DUP] == 0.0


def test_gradient_min_copies_zero_skips_correction(tree, rates, profiles):
    """min_copies=0 should give a gradient that differs from min_copies=1 in
    the GAIN/DUP terms (no empty-profile correction)."""
    g0 = gradient_survival(tree, rates, profiles, min_copies=0)
    g1 = gradient_survival(tree, rates, profiles, min_copies=1)
    assert not np.allclose(g0, g1)


def test_gradient_invalid_min_copies(tree, rates, profiles):
    with pytest.raises(ValueError):
        gradient_survival(tree, rates, profiles, min_copies=3)
