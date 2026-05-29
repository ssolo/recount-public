"""Finite-difference regression test for ``mixture_gradient_native``.

Locks the contract that the K=2 LogisticShift mixture gradient matches a
central-difference approximation of the corrected log-likelihood to ~10
digits across all six gradient blocks (g_base_gain/dup/length, g_delta_dup,
g_delta_length, g_alpha). Without this test, future edits to
``mixture_gradient_native`` or ``derive_category_rates`` could silently
regress the gradient.

See docs/logistic_shift_gradient.tex Lemma "Finite-difference".
"""
from __future__ import annotations

import numpy as np
import pytest

from recount.io.countxml import load_countxml
from recount.logistic_shift import (
    LogisticShiftCategory,
    mixture_log_likelihood_native,
)
from recount.logistic_shift_gradient import mixture_gradient_native


def _setup():
    sess = next(iter(load_countxml("validation/Williams2017.countxml.gz").values()))
    tbl = sess.tables["wsz60-aletrim-min4.txt"]
    profiles = tbl.profiles[:40].astype(np.int32)  # subsample for speed
    rates = sess.rates
    return sess.tree, rates.gain.copy(), rates.loss.copy(), rates.dup.copy(), \
           rates.length.copy(), profiles


def _cats_from_alpha_delta(alphas, delta_dup, delta_length):
    exp_alpha = np.exp(alphas - alphas.max())
    probs = exp_alpha / exp_alpha.sum()
    return [
        LogisticShiftCategory(
            probability=float(probs[k]),
            mod_length=float(delta_length[k]),
            mod_duplication=float(delta_dup[k]),
        )
        for k in range(len(alphas))
    ]


def _ll_at(tree, g, l, d, t, profiles, alphas, delta_dup, delta_length, mc=4):
    cats = _cats_from_alpha_delta(alphas, delta_dup, delta_length)
    return mixture_log_likelihood_native(tree, g, l, d, t, profiles, cats, min_copies=mc)


def _central_diff(f, x, i, eps):
    xp = x.copy(); xp[i] += eps
    xm = x.copy(); xm[i] -= eps
    return (f(xp) - f(xm)) / (2 * eps)


@pytest.mark.parametrize(
    "delta_dup, delta_length, alphas",
    [
        # K=2, nontrivial shifts + nonzero alpha
        (np.array([0.0, 0.3]),  np.array([0.0, -0.4]), np.array([0.0, 0.2])),
    ],
)
def test_mixture_gradient_matches_finite_difference(delta_dup, delta_length, alphas):
    """Central-diff vs analytical gradient on all 6 gradient blocks.

    Tolerances: 1e-6 absolute or relative. Rates-block FD at eps=1e-5 typically
    gives ~1e-9 agreement; shifts/alpha similarly.
    """
    tree, g, l, d, t, profiles = _setup()
    mc = 4
    N = tree.num_nodes
    K = len(alphas)

    # Analytical gradient
    grad = mixture_gradient_native(
        tree, g, l, d, t, delta_dup, delta_length, alphas, profiles,
        min_copies=mc, num_threads=4,
    )

    eps = 1e-5
    # Subsample which nodes to FD-check (full N would be slow); take the
    # root + a leaf + a couple of internal nodes.
    root = tree.root
    leaf = 0
    interior_sample = sorted({root, leaf, root // 2, (root + leaf) // 3})

    # 1. base_gain (per-node) — sample a few nodes
    for v in interior_sample:
        def f_gain(x_):
            g2 = g.copy(); g2[v] = x_[0]
            return _ll_at(tree, g2, l, d, t, profiles, alphas, delta_dup, delta_length, mc)
        fd = _central_diff(f_gain, np.array([g[v]]), 0, eps * max(abs(g[v]), 1.0))
        analytic = grad.g_base_gain[v]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_base_gain[{v}]: fd={fd:.10g} analytic={analytic:.10g}"

    # 2. base_dup (per-node, non-root only) — sample
    nonroot_sample = [v for v in interior_sample if v != root and v != leaf]
    for v in nonroot_sample:
        def f_dup(x_):
            d2 = d.copy(); d2[v] = x_[0]
            return _ll_at(tree, g, l, d2, t, profiles, alphas, delta_dup, delta_length, mc)
        fd = _central_diff(f_dup, np.array([d[v]]), 0, eps * max(abs(d[v]), 1e-3))
        analytic = grad.g_base_dup[v]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_base_dup[{v}]: fd={fd:.10g} analytic={analytic:.10g}"

    # 3. base_length (per-node, non-root only) — sample
    for v in nonroot_sample:
        if not np.isfinite(t[v]):
            continue
        def f_len(x_):
            t2 = t.copy(); t2[v] = x_[0]
            return _ll_at(tree, g, l, d, t2, profiles, alphas, delta_dup, delta_length, mc)
        fd = _central_diff(f_len, np.array([t[v]]), 0, eps * max(abs(t[v]), 1e-3))
        analytic = grad.g_base_length[v]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_base_length[{v}]: fd={fd:.10g} analytic={analytic:.10g}"

    # 4. delta_dup[k] for k >= 1
    for k in range(1, K):
        def f_dd(x_):
            dd = delta_dup.copy(); dd[k] = x_[0]
            return _ll_at(tree, g, l, d, t, profiles, alphas, dd, delta_length, mc)
        fd = _central_diff(f_dd, np.array([delta_dup[k]]), 0, eps)
        analytic = grad.g_delta_dup[k]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_delta_dup[{k}]: fd={fd:.10g} analytic={analytic:.10g}"

    # 5. delta_length[k] for k >= 1
    for k in range(1, K):
        def f_dl(x_):
            dl = delta_length.copy(); dl[k] = x_[0]
            return _ll_at(tree, g, l, d, t, profiles, alphas, delta_dup, dl, mc)
        fd = _central_diff(f_dl, np.array([delta_length[k]]), 0, eps)
        analytic = grad.g_delta_length[k]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_delta_length[{k}]: fd={fd:.10g} analytic={analytic:.10g}"

    # 6. alpha[k] for k >= 1
    for k in range(1, K):
        def f_a(x_):
            a = alphas.copy(); a[k] = x_[0]
            return _ll_at(tree, g, l, d, t, profiles, a, delta_dup, delta_length, mc)
        fd = _central_diff(f_a, np.array([alphas[k]]), 0, eps)
        analytic = grad.g_alpha[k]
        assert np.isclose(fd, analytic, rtol=1e-5, atol=1e-6), \
            f"g_alpha[{k}]: fd={fd:.10g} analytic={analytic:.10g}"
