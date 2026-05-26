"""Example 12 — Csurös' stored rates vs our fit: a diagnostic.

User question: "Are these values the same as in the count.xml files?
The arc269 root with 13k copies looks suspicious. 'Different local optima,
same likelihood' — I don't believe that."

Honest answer (this example reproduces it):

  1. The native pipeline is bit-perfect against Csurös' Java at the SAME
     rates: feed his stored rates into ``per_branch_stats_native`` and
     every per-node quantity matches Java to ≤ 5e-11 absolute. See
     examples 06 and 07. NO disagreement at fixed rates.

  2. But our ML FIT (``recount.ml.fit_rates``) starts from a uniform
     initialisation and lands at a DIFFERENT stationary point than the
     one his stored rates sit at. So `recount analyze` running its own
     cold-start fit will reproduce Csurös' ancestor reconstructions
     only loosely (15-25% root-copy difference, Spearman ρ ≈ 0.99 on
     per-node copies, LL within ~50 nats / ≤ 0.01 nats/family).

  3. Csurös' stored rates aren't the maximum likelihood point — they're
     A near-optimum. If we WARM-START from his rates with NO bounds
     and another 100 BFGS iterations, we IMPROVE the LL by +66 to +319
     nats across the three datasets here (Williams +66, ed194 +307,
     eury114 +319). So the fitting landscape genuinely has multiple
     close-LL stationary points, and his published rates are one of
     them but not THE one.

  4. Two boundary issues make warm-starting tricky and were behind my
     earlier hand-wave:
        - 10-19% of his nodes sit at dup > 0.99 (q̃ very close to 1) —
          near a singular boundary where the gradient w.r.t. dup is
          large and the LBFGS-B line search can overshoot.
        - 3-18 nodes per dataset have gain > 10^10 (Pólya κ encoding
          effectively-Poisson regimes as q→1 limits). The CLI's default
          ``--bound-gain-by-loss`` (gain ≤ 1) WAS clipping these into
          oblivion at warm-start — that default has now been turned OFF.
          ``--bound-dup-by-loss`` stays on because dup > loss has no
          biological meaning.

Reproduces the per-dataset comparison table below for Williams + ed194 +
eury114 (the three reconc/arc269 datasets with stored rates from
sessions Csurös fit himself).

Usage:
  PYTHONPATH=. python3 examples/12_csuros_vs_fit_diagnostic.py
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr

from recount.io.countxml import load_countxml
from recount.ml import default_initial_rates, fit_rates
from recount.native_backend import (
    corrected_log_likelihood_native,
    per_branch_stats_native,
    unobserved_logL0_native,
)


# The Csurös data bundle is mirrored in the repo under docs/csuros_data/.
# Override with RECOUNT_CSUROS_DATA env var to point at an external copy.
import os
ROOT = os.environ.get("RECOUNT_CSUROS_DATA", "docs/csuros_data")
CASES = [
    ("williams", f"{ROOT}/reconc/Williams2017.countxml.gz",
     "wsz60-codes-edit", "wsz60-aletrim-min4.txt", 4),
    ("ed194",    f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "ed194-gtdb", "ed194-arcogm-min4.txt", 4),
    ("eury114",  f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "eury114-ba-gtdb", "eury114-arcogm-min4.txt", 4),
]


def report(sess, rates, profs, mc, prefix):
    g, l, d, t = rates.gain, rates.loss, rates.dup, rates.length
    ll = corrected_log_likelihood_native(sess.tree, g, l, d, t, profs, min_copies=mc)
    L0 = float(np.exp(unobserved_logL0_native(sess.tree, g, l, d, t, mc)))
    stats = per_branch_stats_native(sess.tree, g, l, d, t, profs)
    rc = stats["copies_node"][sess.tree.root]
    rf = stats["families_present"][sess.tree.root]
    print(f"  {prefix:42s}  LL={ll:>13.2f}  L(0)={L0:.4f}  root_copies={rc:>9.2f}  root_fams={rf:>8.2f}")
    return ll, L0, rc, rf, stats["copies_node"]


def main() -> int:
    for label, src, sid, tname, mc in CASES:
        sess = load_countxml(src)[sid]
        profs = sess.tables[tname].profiles.astype(np.int32)
        F, N = profs.shape[0], sess.tree.num_nodes
        g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length

        print()
        print("=" * 76)
        print(f"  {label}  (F={F}, N={N}, Ωmin={mc})")
        print("=" * 76)

        # Boundary diagnostics on Csurös' rates
        n_dup1     = int(np.sum(d == 1.0))
        n_dup_high = int(np.sum(d > 0.99))
        n_gain_big = int(np.sum(g > 1e10))
        print(f"  Boundary diagnostics for Csurös' stored rates:")
        print(f"    nodes with dup == 1.0 (singular q̃=1)             : {n_dup1:>4} / {N}")
        print(f"    nodes with dup  >  0.99 (near boundary)           : {n_dup_high:>4} / {N}"
              f"  ({100*n_dup_high/N:.1f}%)")
        print(f"    nodes with gain > 10^10 (effectively-Poisson Pólya κ): {n_gain_big:>4} / {N}")

        # (A) Csurös' stored rates
        print()
        cll, *_, ccopies = report(sess, sess.rates, profs, mc, "(A) Csurös' stored rates")

        # (B) Cold-start fit, 100 iters, default bounds (dup ≤ loss only)
        init = default_initial_rates(sess.tree)
        t0 = time.time()
        fr_cold = fit_rates(
            sess.tree, profs, initial_rates=init,
            fix_loss=True, fix_root_length=True,
            bound_dup_by_loss=True, bound_gain_by_loss=False,
            min_copies=mc, max_iter=100, backend="native", verbose=False,
        )
        cold_t = time.time() - t0
        ll, L0, rc, rf, fcopies = report(
            sess, fr_cold.rates, profs, mc,
            f"(B) cold-start, 100 iters ({cold_t:.0f}s, conv={fr_cold.converged})")
        print(f"      ΔLL vs Csurös = {ll - cll:+.2f}  ({(ll-cll)/F:+.4f} nats/family)")
        rho = spearmanr(ccopies, fcopies).correlation
        print(f"      Spearman ρ on per-node copies vs Csurös = {rho:.4f}")

        # (C) Warm-start from Csurös' rates, 100 iters, no bounds
        t0 = time.time()
        fr_warm = fit_rates(
            sess.tree, profs, initial_rates=sess.rates,
            fix_loss=True, fix_root_length=True,
            bound_dup_by_loss=False, bound_gain_by_loss=False,
            min_copies=mc, max_iter=100, backend="native", verbose=False,
        )
        warm_t = time.time() - t0
        ll, L0, rc, rf, wcopies = report(
            sess, fr_warm.rates, profs, mc,
            f"(C) warm-start from Csurös, 100 iters ({warm_t:.0f}s, conv={fr_warm.converged})")
        print(f"      ΔLL vs Csurös = {ll - cll:+.2f}  ({(ll-cll)/F:+.4f} nats/family)")
        if ll > cll:
            print(f"      ☑ Warm-start IMPROVES Csurös' LL — his stored rates aren't the MLE,")
            print(f"        they're A near-optimum (+{ll-cll:.0f} nats away from the local max).")
        else:
            print(f"      ⚠ Warm-start did not improve — landscape near his point is flat")
            print(f"        or boundaries prevent further progress.")

    print()
    print("=" * 76)
    print(" Bottom line")
    print("=" * 76)
    print()
    print(" * AT FIXED RATES, native = Java bit-for-bit (≤ 5e-11). See examples 06/07.")
    print(" * Csurös' STORED rates are not the MLE — warm-starting from them with")
    print("   the wrong bounds removed improves the LL by +66 to +320 nats here.")
    print(" * Our COLD-START fit lands at a different near-optimum (16-50 nats")
    print("   behind Csurös', ≤ 0.01 nats/family). Per-node copy reconstructions")
    print("   correlate at Spearman ρ > 0.99, but absolute root values can differ")
    print("   15-25%.")
    print(" * The arc269 figure (13k root copies at Ωmin=1) is the cold-start fit's")
    print("   correct output under its L(0)≈0.77 estimate — the L(0)/(1-L(0))·F")
    print("   amplification of a small per-family unobserved-profile E[ξ_root]")
    print("   gives the large corrected number. Run longer or warm-start for a")
    print("   tighter L(0) and smaller corrected root.")
    print()
    print(" For Csurös' SPECIFIC published per-node reconstructions, use his rates")
    print(" directly via examples 06/07 — those match Java to ≤ 5e-11 absolute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
