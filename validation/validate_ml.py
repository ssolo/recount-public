"""Validate the ML optimizer against Count's published rates.

Runs two ML fits on the same data and compares:

  1. **warm-start**: initialize from the rates already in the countxml.
     If those rates are an MLE for our model, the optimizer should make
     only a tiny improvement.

  2. **cold-start**: initialize from neutral defaults and re-fit.
     Should reach a LL comparable to (or better than) Count's, unless
     Count's full model includes structure we don't (e.g. LogisticShift
     rate variation, which is identity in Williams2017 but baked into
     Count's optimization trajectory).

Outputs JSON + a console summary.  Picks a profile-sum subset by default
to keep the run tractable (the M2 Ultra optimizer needs ~1s/iter per
500 families per max_iter, plus there's a heavy tail in Williams2017).

Usage:
    python validation/validate_ml.py validation/Williams2017.countxml.gz \
        --max-sum 8 --max-families 500 --max-iter 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from recount.gld import corrected_log_likelihood
from recount.io.countxml import load_countxml
from recount.ml import default_initial_rates, fit_rates


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xml", type=Path)
    ap.add_argument("--max-sum", type=int, default=8,
                    help="profile-sum filter (default: 8, ~57% of Williams families)")
    ap.add_argument("--max-families", type=int, default=500,
                    help="cap families used (default: 500)")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--min-copies", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "validation" / "results" / "validate_ml.json")
    args = ap.parse_args()

    sess = next(iter(load_countxml(args.xml).values()))
    tbl = next(iter(sess.tables.values()))
    profiles = tbl.profiles
    if args.max_sum is not None:
        profiles = profiles[profiles.sum(axis=1) <= args.max_sum]
    if args.max_families is not None:
        profiles = profiles[:args.max_families]
    F = profiles.shape[0]
    print(f"# {F} families (max sum = {profiles.sum(axis=1).max()})")

    ll_count = corrected_log_likelihood(sess.tree, sess.rates, profiles, min_copies=args.min_copies)
    print(f"# LL at Count's rates (this subset): {ll_count:.6f}")

    print("# --- warm-start (init = Count) ---")
    t0 = time.time()
    fw = fit_rates(sess.tree, profiles, initial_rates=sess.rates,
                    fix_loss=True, fix_root_length=True,
                    min_copies=args.min_copies, max_iter=args.max_iter, verbose=False)
    print(f"# warm-start: {fw.n_iter} iters, LL={fw.log_likelihood:.6f}, "
          f"Δ from Count = {fw.log_likelihood - ll_count:+.6f}  [{time.time()-t0:.1f}s]")

    print("# --- cold-start (init = defaults) ---")
    init = default_initial_rates(sess.tree)
    ll_init = corrected_log_likelihood(sess.tree, init, profiles, min_copies=args.min_copies)
    print(f"#   LL at defaults: {ll_init:.6f}")
    t0 = time.time()
    fc = fit_rates(sess.tree, profiles, initial_rates=init,
                    fix_loss=True, fix_root_length=True,
                    min_copies=args.min_copies, max_iter=args.max_iter, verbose=False)
    print(f"# cold-start: {fc.n_iter} iters, LL={fc.log_likelihood:.6f}, "
          f"Δ from Count = {fc.log_likelihood - ll_count:+.6f}  [{time.time()-t0:.1f}s]")

    summary = {
        "xml": str(args.xml),
        "num_families": F, "max_sum": int(profiles.sum(axis=1).max()),
        "ll_at_count_rates": ll_count,
        "warm_start": {
            "n_iter": fw.n_iter, "log_likelihood": fw.log_likelihood,
            "delta_vs_count": fw.log_likelihood - ll_count,
            "converged": fw.converged,
        },
        "cold_start": {
            "n_iter": fc.n_iter, "log_likelihood": fc.log_likelihood,
            "delta_vs_count": fc.log_likelihood - ll_count,
            "converged": fc.converged,
        },
    }
    print()
    print("# === SUMMARY ===")
    delta_w = fw.log_likelihood - ll_count
    delta_c = fc.log_likelihood - ll_count
    print(f"# warm-start moved Count's rates by Δ LL = {delta_w:+.4f}")
    print(f"# cold-start re-fit landed Δ LL = {delta_c:+.4f}")
    if abs(delta_w) < 0.5 * F * 1e-3:  # generous: 0.5e-3 per family
        print("# ✓ Count's rates are near-MLE for our model on this subset.")
    else:
        print("# ✗ Count's rates are NOT at our MLE — model details (e.g. LogisticShift) differ.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
