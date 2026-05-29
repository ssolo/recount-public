"""Example 2 — log L(0) for arbitrary Ωmin (Csurös 2026 SI Theorems 3-5).

Reproduces the claim:
    "Native implements Csurös 2026 SI Theorem 3 (Algorithm Uinner from
     Fig S11) for arbitrary observation-bias parameter Ωmin. Matches
     Java's Likelihood.getEmptyLL() + getSingletonLL() to machine
     epsilon for Ωmin ∈ {1, 2}."

Csurös 2026 (Materials and Methods, Theorems 3-9) defines the
unobserved-profile correction — both the likelihood and its gradient —
for arbitrary Ωmin, and the phylogenetic-reconciliation datasets are
fit at Ωmin = 4. This example only carries hardcoded Java reference
values for Ωmin ∈ {1, 2} (empty and empty+singleton), so the Ωmin ≥ 3
rows are printed without a Java cross-check — not because Java cannot
produce one.

Demonstrates: Csurös 2026 SI Eq 4 with general Ωmin, evaluated on the
60-leaf Williams2017 dataset at Count's stored rates.

Prerequisites:
  - librecount.dylib built
  - validation/Williams2017.countxml.gz present

Usage:
  PYTHONPATH=. python3 examples/02_unobserved_logL0_arbitrary_omega.py
"""
from __future__ import annotations

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import unobserved_logL0_native

# Java reference values carried in this example for Ωmin = 1, 2 only
JAVA_LL_EMPTY     = -1.1616624726388685
JAVA_LL_SINGLETON = -1.2916692400830379


def main() -> int:
    sess = next(iter(load_countxml("validation/Williams2017.countxml.gz").values()))
    g, l, d, t = (
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    )
    print("# log L(0) = log P{Ω < Ωmin} at stored Williams rates")
    print("# (Eq 4 of Csurös 2026)")
    print()
    java_L0_min1 = JAVA_LL_EMPTY
    java_L0_min2 = float(np.logaddexp(JAVA_LL_EMPTY, JAVA_LL_SINGLETON))
    print(f"# {'Ωmin':>5}  {'native log L(0)':>20}  {'java log L(0)':>20}  {'Δ':>14}")
    for k in range(1, 9):
        native_L0 = unobserved_logL0_native(sess.tree, g, l, d, t, k)
        if k == 1:
            java_L0 = java_L0_min1
            delta = native_L0 - java_L0
            print(f"  {k:>3}  {native_L0:>20.16f}  {java_L0:>20.16f}  {delta:>+14.2e}")
        elif k == 2:
            java_L0 = java_L0_min2
            delta = native_L0 - java_L0
            print(f"  {k:>3}  {native_L0:>20.16f}  {java_L0:>20.16f}  {delta:>+14.2e}")
        else:
            print(f"  {k:>3}  {native_L0:>20.16f}  {'(no Java ref)':>20s}  {'n/a':>14s}")

    print()
    print("# Native matches Java to machine ε for Ωmin ∈ {1,2}, and computes")
    print("# arbitrary Ωmin via the Uinner algorithm (SI Fig S11) — the same")
    print("# correction Count applies (the focal datasets are fit at Ωmin=4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
