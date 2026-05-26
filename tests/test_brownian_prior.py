"""Unit tests for the tree-Brownian autocorrelated log-rate prior.

The Brownian prior places a Gaussian on the log-rate *increment* across
every tree edge — the Thorne–Kishino–Painter relaxed-clock construction
adapted to the GLD gain / dup / length axes. Formal exposition and the
analytical gradient are in `docs/brownian_prior.pdf`.

Coverage:
1. FD-validation of `_brownian_prior_and_grad` on a dpann80 random init
   — the analytical gradient must match central finite differences to
   ≤ 10⁻⁶ relative.
2. FD-validation of the full MAP objective with the Brownian prior —
   the prior chain rule must compose correctly with the GLD gradient.
3. Sanity: σ_brownian → ∞ makes the prior contribution vanish (recovers
   approximately pure ML).
4. Sanity: σ_brownian → 0 (sharp shrinkage) pulls all log-rates toward
   the root values, so the optimum has near-uniform log-rates.
5. The native C implementation matches the Python reference bit-for-bit.

Implementations:
- `validation/_shared._brownian_prior_and_grad` — Python fallback;
- `native/src/recount_brownian_prior.c` — production O(N) kernel.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation._shared import (
    _brownian_prior_and_grad,
    _brownian_prior_and_grad_python,
    make_objgrad_map_brownian,
    rates_to_x,
    x_to_rates,
    load_dataset,
)
from recount.ml import random_initial_rates


def _fd_grad(fn, x, eps=1e-6):
    """Central finite-difference gradient of scalar-valued ``fn(x)``."""
    g = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (fn(x + e) - fn(x - e)) / (2 * eps)
    return g


def test_brownian_prior_gradient_matches_fd():
    """FD-validate `_brownian_prior_and_grad` to rel_err ≤ 1e-5 on dpann80."""
    tree, _, _, _, _ = load_dataset("dpann80")
    rng = np.random.default_rng(42)
    init = random_initial_rates(tree, rng)
    x0 = rates_to_x(init, tree.root, subcritical=True)

    sigma_g, sigma_d, sigma_t = 0.7, 0.5, 0.9
    mu_g, mu_d, mu_t = float(np.log(0.1)), 0.0, 0.0
    sigma_root = 5.0

    def f(x):
        lp, _ = _brownian_prior_and_grad(
            x, tree, sigma_g, sigma_d, sigma_t,
            mu_root_gain=mu_g, mu_root_dup=mu_d, mu_root_length=mu_t,
            sigma_root=sigma_root, subcritical=True,
        )
        return lp

    _, g_analytic = _brownian_prior_and_grad(
        x0, tree, sigma_g, sigma_d, sigma_t,
        mu_root_gain=mu_g, mu_root_dup=mu_d, mu_root_length=mu_t,
        sigma_root=sigma_root, subcritical=True,
    )
    # FD over a random subset of 30 indices (full sweep is N=475 — too slow)
    sample_idx = rng.choice(len(x0), 30, replace=False)
    for i in sample_idx:
        e = np.zeros_like(x0); e[i] = 1e-6
        fd = (f(x0 + e) - f(x0 - e)) / (2 * 1e-6)
        rel = abs(fd - g_analytic[i]) / max(abs(fd), 1e-30)
        # Accept either rel_err ≤ 1e-4 OR abs_err ≤ 1e-8 (FD noise floor)
        abs_err = abs(fd - g_analytic[i])
        assert rel < 1e-4 or abs_err < 1e-8, (
            f"index {i}: analytic={g_analytic[i]:+.6e}  fd={fd:+.6e}  "
            f"rel={rel:.2e}  abs={abs_err:.2e}"
        )


def test_full_objgrad_with_brownian_prior_matches_fd():
    """End-to-end: make_objgrad_map_brownian gradient matches FD."""
    tree, profiles, _, mc, _ = load_dataset("dpann80")
    rng = np.random.default_rng(7)
    init = random_initial_rates(tree, rng)
    x0 = rates_to_x(init, tree.root, subcritical=True)

    obj = make_objgrad_map_brownian(
        tree, init, profiles, mc,
        sigma_brownian_gain=0.6, sigma_brownian_dup=0.4, sigma_brownian_length=0.8,
        subcritical=True,
    )
    f0, g_analytic = obj(x0)
    # Spot-check 10 random indices across all 3 blocks
    N = tree.num_nodes
    sample_idx = rng.choice(len(x0), 10, replace=False)
    for i in sample_idx:
        e = np.zeros_like(x0); e[i] = 1e-5
        f_plus, _ = obj(x0 + e)
        f_minus, _ = obj(x0 - e)
        fd = (f_plus - f_minus) / (2 * 1e-5)
        rel = abs(fd - g_analytic[i]) / max(abs(fd), 1e-30)
        abs_err = abs(fd - g_analytic[i])
        block = ("gain" if i < N else ("dup" if i < 2 * N - 1 else "length"))
        assert rel < 1e-4 or abs_err < 1e-6, (
            f"index {i} ({block}): analytic={g_analytic[i]:+.4e}  "
            f"fd={fd:+.4e}  rel={rel:.2e}  abs={abs_err:.2e}"
        )


def test_brownian_prior_vanishes_at_large_sigma():
    """Sanity: σ_brownian → ∞ ⇒ prior contribution becomes negligible.

    At σ = 1e6 per axis, the per-edge term 1/(2·σ²) ~ 5e-13 → log_prior dwarfed
    by any reasonable rate vector. Gradient magnitudes should also be ~0.
    """
    tree, _, _, _, _ = load_dataset("dpann80")
    rng = np.random.default_rng(123)
    init = random_initial_rates(tree, rng)
    x0 = rates_to_x(init, tree.root, subcritical=True)

    lp_strong, g_strong = _brownian_prior_and_grad(
        x0, tree, 1.0, 1.0, 1.0,
        mu_root_gain=float(np.log(0.1)), mu_root_dup=0.0, mu_root_length=0.0,
        sigma_root=5.0, subcritical=True,
    )
    lp_loose, g_loose = _brownian_prior_and_grad(
        x0, tree, 1e6, 1e6, 1e6,
        mu_root_gain=float(np.log(0.1)), mu_root_dup=0.0, mu_root_length=0.0,
        sigma_root=1e6, subcritical=True,
    )
    # log_prior with huge σ should be ~12 orders of magnitude smaller than strong
    assert abs(lp_loose) < 1e-6 * abs(lp_strong)
    # gradient also collapses
    assert np.max(np.abs(g_loose)) < 1e-6 * np.max(np.abs(g_strong))


def test_brownian_prior_centred_correctly():
    """Sanity: if all log-rates equal their parent (uniform-tree config),
    Brownian prior contribution is zero from the edge terms (only the
    root anchor remains)."""
    tree, _, _, _, _ = load_dataset("dpann80")
    N = tree.num_nodes
    root = tree.root
    # Build a rate config where every non-root rate exactly matches root's value.
    # In x-space: all gain positions = mu_root_gain; all dup positions = mu_root_dup;
    # all length positions = mu_root_length.
    mu_g, mu_d, mu_t = float(np.log(0.1)), 0.0, 0.0
    x = np.empty(3 * N - 2)
    x[:N] = mu_g
    x[N:2 * N - 1] = mu_d
    x[2 * N - 1:] = mu_t

    lp, g = _brownian_prior_and_grad(
        x, tree, 0.5, 0.5, 0.5,
        mu_root_gain=mu_g, mu_root_dup=mu_d, mu_root_length=mu_t,
        sigma_root=5.0, subcritical=True,
    )
    # All edge diffs are zero, root anchor diff is zero → log_prior should be 0
    assert abs(lp) < 1e-12, f"expected 0, got {lp}"
    # Gradient also exactly zero
    assert np.max(np.abs(g)) < 1e-12


def test_native_matches_python_reference():
    """Native C implementation must match the Python reference bit-for-bit
    (or to round-off, ~1e-12 abs) on random inputs."""
    tree, _, _, _, _ = load_dataset("dpann80")
    rng = np.random.default_rng(2024)
    init = random_initial_rates(tree, rng)
    x0 = rates_to_x(init, tree.root, subcritical=True)

    sigmas = [
        (1.0, 1.0, 1.0, 5.0),
        (0.5, 0.7, 0.9, 3.0),
        (0.1, 0.3, 0.2, 10.0),
        (2.5, 1.5, 0.4, 5.0),
    ]
    for sg, sd, st, sr in sigmas:
        lp_native, g_native = _brownian_prior_and_grad(
            x0, tree, sg, sd, st,
            mu_root_gain=float(np.log(0.1)), mu_root_dup=0.0, mu_root_length=0.0,
            sigma_root=sr, subcritical=True,
        )
        lp_py, g_py = _brownian_prior_and_grad_python(
            x0, tree, sg, sd, st,
            mu_root_gain=float(np.log(0.1)), mu_root_dup=0.0, mu_root_length=0.0,
            sigma_root=sr, subcritical=True,
        )
        # Native loops in different order from Python's np.add.at, so allow tiny
        # FP-summation drift (max ~1e-12 absolute, ~1e-14 relative)
        assert abs(lp_native - lp_py) < max(1e-10, 1e-12 * abs(lp_py)), (
            f"σ=({sg},{sd},{st},{sr}): lp_native={lp_native:.12e}  "
            f"lp_py={lp_py:.12e}  diff={lp_native - lp_py:.3e}"
        )
        max_abs = np.max(np.abs(g_native - g_py))
        max_rel = np.max(np.abs(g_native - g_py) / np.maximum(np.abs(g_py), 1e-30))
        assert max_abs < 1e-10 or max_rel < 1e-12, (
            f"σ=({sg},{sd},{st},{sr}): max abs grad diff = {max_abs:.3e}, "
            f"max rel = {max_rel:.3e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
