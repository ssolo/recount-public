"""Example 14 — arc269 LACA reconstruction from the canonical 15-start MAP rates.

Loads the pre-fitted MAP σ=1 15-start global rates from
``validation/outputs_no_dup_cap/profile_likelihood_arc269/``
(the canonical LACA quote — see PROGRESS.md and the README's
"Demonstration of reproduction" section), computes the full arc269
reconstruction in a few seconds, and prints the headline numbers.

The fit itself (15-start MAP σ=1 multistart BFGS on the full 90,243
families × 269 leaves) takes ~50 minutes on M4 Max. Loading the
saved rates and computing the reconstruction takes ~5 seconds.

Expected output:
    Root copies         : 5,426
    Root families       : 3,070
    Copies / family     : 1.77
  (bootstrap 95% CI: cp 5,182–5,687; fm 2,982–3,165; cpf 1.73–1.83;
   see VALIDATION.md and PROGRESS.md.)

Usage:
  PYTHONPATH=. python3 examples/14_arc269_sota_reconstruction.py

To re-run the underlying fit:
    PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \\
        --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2027 \\
        --no-subcritical \\
        --out-dir validation/outputs_no_dup_cap/profile_likelihood_arc269
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from recount.io.newick import read_newick
from recount.io.table import read_profile_table
from recount.native_backend import (
    corrected_log_likelihood_native,
    per_branch_stats_native,
    unobserved_logL0_native,
)
from recount.rates import GLDRates


def main() -> int:
    t0 = time.time()

    # Tree + profile table are bundled with the repo under examples/data/.
    tree, _, _ = read_newick("examples/data/arc269_tree.nwk")
    _, profiles = read_profile_table("examples/data/arc269_table.csv.gz",
                                     list(tree.leaf_names))
    profiles = profiles.astype(np.int32)
    F = profiles.shape[0]
    print(f"Loaded arc269: F={F:,}, N={tree.num_nodes}, "
          f"leaves={tree.num_leaves}  ({time.time()-t0:.1f}s)")

    # The canonical LACA quote comes from the 15-start MAP σ=1 fit, saved as
    # the recommended baseline in PROGRESS.md. The bounded variant has not
    # been re-run; the unconstrained 15-start global is interior
    # (max_dup=28, max_gain=0.53), bit-perfect against Java at the same rates,
    # and produces the bootstrap-CI'd cp=5,426 / fm=3,070 / cpf=1.77 quote.
    rates_path = Path("validation/outputs_no_dup_cap/profile_likelihood_arc269") / \
                 "arc269_sigma1.0_final_rates.npz"
    if not rates_path.exists():
        print(f"ERROR: {rates_path} not found.")
        print("Re-run the 15-start MAP fit; see this script's docstring for the recipe.")
        return 1
    npz = np.load(rates_path)
    rates = GLDRates(tree=tree, gain=npz["gain"], loss=npz["loss"],
                     dup=npz["dup"], length=npz["length"])
    print(f"Loaded MAP σ=1 15-start global rates from {rates_path.name}")

    # Forward (corrected) log-likelihood at Ωmin = 1 — the conditioning the
    # arc269 dataset itself uses (every family is at least one copy somewhere).
    t1 = time.time()
    ll = corrected_log_likelihood_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length,
        profiles, min_copies=1)
    print(f"  corrected LL    = {ll:.4f}  ({time.time()-t1:.2f}s)")

    # L(0) at Ωmin = 1 = P(Ω(Ξ) = 0 | fitted rates). For arc269 at this fit
    # L(0) ≈ 0.527, so the F/(1−L(0)) amplification is ~2.1× over the
    # visible 90,243 families.
    t1 = time.time()
    L0 = float(np.exp(unobserved_logL0_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, 1)))
    print(f"  L(0)            = {L0:.6f}  ({time.time()-t1:.2f}s)")

    # Per-branch posterior counts. stats["copies_node"][v] = Σ_f E[ξ_v | f];
    # stats["families_present"][v] = Σ_f P{ξ_v ≥ 1 | f}.
    t1 = time.time()
    stats = per_branch_stats_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, profiles)
    print(f"  per-branch stats computed  ({time.time()-t1:.2f}s)")

    rc = stats["copies_node"][tree.root]
    rf = stats["families_present"][tree.root]
    print()
    print("===== arc269 LACA reconstruction (MAP σ=1, 15-start global) =====")
    print(f"  Root copies         : {rc:,.0f}")
    print(f"  Root families       : {rf:,.0f}")
    print(f"  Copies / family     : {rc/rf:.2f}")
    print(f"  LL / family         : {ll/F:.4f}")
    print()
    print(f"Expected: ~5,426 copies / 3,070 families / 1.77 cp/fm")
    print(f"  (bootstrap 95% CI: cp 5,182–5,687; fm 2,982–3,165; cpf 1.73–1.83)")
    print()
    print(f"Total wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
