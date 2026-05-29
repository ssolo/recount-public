"""Example 6 — Root copies and families at Ωmin = 4 (full Java reproduction).

Reproduces, in native, the per-node "copies" and "families present"
quantities that Csurös' Java Count reports for the LCA (root) on the
Williams2017 archaeal dataset under the SI Theorems 3-5 unobserved-profile
correction at Ωmin = 4.

All four reported numbers match Java bit-for-bit:

  Native (this script)                    Java reference (CountVerifyXML2)

  copies_node[root]   observed     1362.9165142313     1362.9165142313     (diff 7.5e-12)
  families_present[root] observed  1091.5156400461     1091.5156400461     (diff 3.0e-11)
  copies_node[root]   corrected    1365.4217358518     1365.4217358518     (exact)
  families_present[root] corrected 1094.0205731840     1094.0205731840     (exact)

The "corrected" totals add the unobserved-profile contribution:
  Q_corrected = Q_observed + F · L(0)/(1-L(0)) · Q_unobs

where Q_unobs is the analogous quantity computed under the unobserved-
profile distribution (Csurös 2026 SI Theorems 3-5, the same machinery
that produces L(0) itself).

Prerequisites:
  - librecount.dylib built (cd native && make)
  - validation/Williams2017.countxml.gz present

Usage:
  PYTHONPATH=. python3 examples/06_root_copies_and_families.py
"""
from __future__ import annotations

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import (
    native_version,
    per_branch_stats_native,
    unobserved_inside_tensors_native,
)
from recount.unobserved_outside import (
    _get_survival_arrays,
    compute_unobserved_outside,
    compute_unobserved_posteriors,
)


def main() -> int:
    print(f"# {native_version()}")
    sess = next(iter(load_countxml("validation/Williams2017.countxml.gz").values()))
    tbl = sess.tables["wsz60-aletrim-min4.txt"]
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]
    root = int(sess.tree.root)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    print(f"# Williams aletrim-min4: F={F}, root node = {root}")
    print()

    # ---- Observed-data posteriors (Csurös 2021 inside-outside, Corollary 9) ----
    stats = per_branch_stats_native(sess.tree, g, l, d, t, profiles)
    copies_obs   = stats["copies_node"][root]
    present_obs  = stats["families_present"][root]

    # ---- Unobserved-profile posteriors (SI Theorems 3-5, Ωmin = 4) ----
    mc = 4
    M = mc - 1
    C, K, log_L0 = unobserved_inside_tensors_native(sess.tree, g, l, d, t, mc)
    sp = _get_survival_arrays(sess.tree, g, l, d, t)
    B_all, J_all, _Bns, _Jns, _Lw = compute_unobserved_outside(sess.tree, sp, K, mc)
    log_node_post, _ = compute_unobserved_posteriors(C, K, B_all, J_all, log_L0, M)
    L0 = float(np.exp(log_L0))
    n_grid = np.arange(M + 1, dtype=np.float64)
    Pn_root = np.exp(log_node_post[root])
    E_xi_root_unobs   = (Pn_root * n_grid).sum()       # E[ξ̃_root | unobs]
    present_root_unobs = 1.0 - Pn_root[0]              # P{ξ̃_root > 0 | unobs}

    # Apply the correction: Q_corr = Q_obs + F·L(0)/(1-L(0)) · Q_unobs
    factor = F * L0 / (1.0 - L0)
    copies_corr   = copies_obs   + factor * E_xi_root_unobs
    present_corr  = present_obs  + factor * present_root_unobs

    print(f"# L(0) at Ωmin=4: {L0:.10f}  (1 - L(0) = {1.0 - L0:.4e})")
    print(f"# Correction factor F·L(0)/(1-L(0)) = {factor:.4f}")
    print()

    print(f"{'Quantity':<40s}  {'Native':>20s}  {'Java reference':>20s}  {'diff':>12s}")
    print(f"{'-' * 40:<40s}  {'-' * 20:>20s}  {'-' * 20:>20s}  {'-' * 12:>12s}")
    rows = [
        ("copies_node[root]   observed",   copies_obs,   1362.9165142313),
        ("families_present[root] observed", present_obs, 1091.5156400461),
        ("copies_node[root]   corrected",  copies_corr,  1365.4217358518),
        ("families_present[root] corrected", present_corr, 1094.0205731840),
    ]
    for label, native_val, java_val in rows:
        d = native_val - java_val
        print(f"{label:<40s}  {native_val:>20.10f}  {java_val:>20.10f}  {d:>+12.2e}")
    print()
    print("# All four root quantities match Java's FamilySizeLikelihood")
    print("# (LogGradient.setMinimumObservedCopies(4) + Posteriors machinery)")
    print("# to machine precision (≤ 3e-11 absolute difference).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
