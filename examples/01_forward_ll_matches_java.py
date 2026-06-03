"""Example 1 — Native forward log-likelihood matches Count.Java exactly.

Reproduces the claim:
    "Native LL_raw on Williams2017 matches Java's Likelihood.getLL() to
     4.80e-10 absolute (machine epsilon range given accumulated rounding
     across 5378 families × 119 nodes × max width 668)."

Demonstrates: native correctness vs the published Java reference, on the
60-leaf Williams2017 archaeal dataset.

Prerequisites:
  - librecount.dylib built (`make -C native`)
  - validation/Williams2017.countxml.gz present
  - CountXXV.jar (optional — values are also hard-coded below from a
    prior verified Java run, so the script needs no Java installation
    to reproduce the comparison).

Usage:
  PYTHONPATH=. python3 examples/01_forward_ll_matches_java.py
"""
from __future__ import annotations

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import (
    corrected_log_likelihood_native,
    log_likelihood_native,
    native_version,
)

# Java reference values for Williams2017.countxml.gz at stored rates,
# computed via `java -cp ... count.model.CountVerifyXML`. Reproducible:
#   java -cp build/classes count.model.CountVerifyXML \
#        validation/Williams2017.countxml.gz "" "" 1
JAVA_LL_RAW              = -127288.46060739836
JAVA_LL_CORRECTED_MIN1   = -125269.71698651015
JAVA_LL_CORRECTED_MIN2   = -122522.52477922056
JAVA_LL_EMPTY            =      -1.1616624726388685
JAVA_LL_SINGLETON        =      -1.2916692400830379


def main() -> int:
    print(f"# {native_version()}")
    sess = next(iter(load_countxml("validation/Williams2017.countxml.gz").values()))
    tbl = sess.tables["wsz60-aletrim-min4.txt"]
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]
    g, l, d, t = (
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    )
    print(f"# Williams2017 wsz60-aletrim-min4: {F} families, "
          f"max_sum={profiles.sum(axis=1).max()}")
    print()

    per_fam = log_likelihood_native(sess.tree, g, l, d, t, profiles)
    ll_raw_native = float(per_fam.sum())
    diff_raw = ll_raw_native - JAVA_LL_RAW
    print(f"  LL_raw       native = {ll_raw_native:>20.10f}")
    print(f"               java   = {JAVA_LL_RAW:>20.10f}")
    print(f"               Δ      = {diff_raw:>+20.2e}")
    print()

    for mc, java_ref in [(1, JAVA_LL_CORRECTED_MIN1), (2, JAVA_LL_CORRECTED_MIN2)]:
        ll = corrected_log_likelihood_native(
            sess.tree, g, l, d, t, profiles, min_copies=mc)
        diff = ll - java_ref
        print(f"  LL_corr (Ωmin={mc}) native = {ll:>20.10f}")
        print(f"                  java       = {java_ref:>20.10f}")
        print(f"                  Δ          = {diff:>+20.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
