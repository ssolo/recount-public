"""Tests for `recount.logistic_shift` — the K-category LogisticShift
mixture forward likelihood (production path used by
`validation/mixture_ml.py`).

Coverage:
1. `_sanity_check_k1` — K=1 with zero shifts equals the bare GLD
   likelihood (identity transform).
2. Explicit K=1 call through `mixture_log_likelihood_native` agrees
   with the bare GLD path on the same inputs (no numerical drift from
   the mixture machinery alone).
3. K=2 mixture LL is bounded below by Jensen's inequality applied to
   the per-family LL of each category — a basic sanity check on the
   mixture aggregation.
4. Invalid mixing weights (don't sum to 1) are rejected.
5. Empty category list is rejected.

The closed-form mixture *gradient* and the rate-domain vs
survival-domain parameterisations are documented in
`docs/logistic_shift_gradient.pdf`; the FD-precision contract on the
gradient lives in `test_mixture_gradient_fd.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from recount.io.countxml import load_countxml
from recount.logistic_shift import (
    LogisticShiftCategory,
    _sanity_check_k1,
    mixture_log_likelihood_native,
)
from recount.native_backend import corrected_log_likelihood_native


def _williams_session():
    return next(iter(load_countxml('validation/Williams2017.countxml.gz').values()))


def test_k1_identity():
    """K=1 with zero shifts must equal base-GLD LL bit-for-bit."""
    sess = _williams_session()
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    base, mix = _sanity_check_k1(sess.tree, g, l, d, t, profiles, min_copies=4)
    assert abs(base - mix) < 1e-12


def test_k1_explicit_call():
    """Same identity via the public ``mixture_log_likelihood_native`` API."""
    sess = _williams_session()
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    cats = [LogisticShiftCategory(probability=1.0, mod_length=0.0, mod_duplication=0.0)]
    mix = mixture_log_likelihood_native(sess.tree, g, l, d, t, profiles, cats, min_copies=4)
    base = corrected_log_likelihood_native(sess.tree, g, l, d, t, profiles, min_copies=4)
    assert abs(base - mix) < 1e-12


def test_k2_mixture_jensen_lower_bound():
    """K=2 mixture LL must satisfy Jensen's inequality lower bound:
        log Σ_k p_k · L_f_k ≥ Σ_k p_k · log L_f_k
    summing over families gives mix_LL ≥ Σ_k p_k · single_cat_LL_k.
    """
    sess = _williams_session()
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length

    cats = [
        LogisticShiftCategory(probability=0.6, mod_length=0.0, mod_duplication=0.0),
        LogisticShiftCategory(probability=0.4, mod_length=np.log(1.3), mod_duplication=np.log(0.8)),
    ]
    mix = mixture_log_likelihood_native(sess.tree, g, l, d, t, profiles, cats, min_copies=4)
    base = corrected_log_likelihood_native(sess.tree, g, l, d, t, profiles, min_copies=4)
    # Compute the "second category alone" LL
    g2, l2, d2, t2 = (g, l, d * 0.8, np.where(np.isinf(t), t, t * 1.3))
    other = corrected_log_likelihood_native(sess.tree, g2, l2, d2, t2, profiles, min_copies=4)
    lower_bound = 0.6 * base + 0.4 * other
    assert mix >= lower_bound - 1e-6, f"Jensen violated: mix={mix} < lb={lower_bound}"


def test_probabilities_must_sum_to_one():
    sess = _williams_session()
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    bad_cats = [
        LogisticShiftCategory(probability=0.5, mod_length=0.0, mod_duplication=0.0),
        LogisticShiftCategory(probability=0.4, mod_length=0.1, mod_duplication=0.0),  # sums to 0.9
    ]
    with pytest.raises(ValueError, match="sum to 1"):
        mixture_log_likelihood_native(sess.tree, g, l, d, t, profiles, bad_cats, min_copies=4)


def test_empty_categories_rejected():
    sess = _williams_session()
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    with pytest.raises(ValueError, match="at least one category"):
        mixture_log_likelihood_native(sess.tree, g, l, d, t, profiles, [], min_copies=4)
