"""Validate the LogisticShift mixture ML against Count's published rates.

Runs three fits on the same subset of Williams2017 and compares:

  1. **No rate variation, warm-start from Count's rates** — same as
     ``validate_ml.py``.  Establishes the baseline GLD-only MLE.

  2. **LogisticShift K=1, identity init** — adds a rate-variation
     wrapper around the same model but starts the wrapper at identity
     (zero shifts, weight 1).  Should land near (1) within optimizer
     noise; if it lands meaningfully higher, the extra parameter is
     compensating for something in the optimization.

  3. **LogisticShift K=2 from defaults** — gives the optimizer two
     mixture components.  On Williams the second category typically
     captures a small fraction of "fast-loss" families and the LL
     improves by tens of nats.

Outputs JSON + a console summary.

Usage:
    python validation/validate_mixture.py validation/Williams2017.countxml.gz \\
        --max-sum 4 --max-families 100 --max-iter 30
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
from recount.ml import default_initial_rates, fit_rates, fit_mixture
from recount.rate_variation import LogisticShift

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xml", type=Path)
    ap.add_argument("--max-sum", type=int, default=4)
    ap.add_argument("--max-families", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--min-copies", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--k", "-K", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "validation" / "results" / "validate_mixture.json")
    args = ap.parse_args()

    sess = next(iter(load_countxml(args.xml).values()))
    tbl = next(iter(sess.tables.values()))
    profiles = tbl.profiles
    profiles = profiles[profiles.sum(axis=1) <= args.max_sum]
    profiles = profiles[:args.max_families]
    F = profiles.shape[0]
    print(f"# {F} families (max sum = {profiles.sum(axis=1).max()})")

    ll_count = corrected_log_likelihood(sess.tree, sess.rates, profiles,
                                        min_copies=args.min_copies)
    print(f"# LL at Count's rates (this subset): {ll_count:.6f}")

    # 1) Bare GLD warm-start
    print("# --- bare GLD, warm-start ---")
    t0 = time.time()
    f1 = fit_rates(sess.tree, profiles, initial_rates=sess.rates,
                   fix_loss=True, fix_root_length=True,
                   min_copies=args.min_copies, max_iter=args.max_iter)
    print(f"# bare GLD: {f1.n_iter} iters, LL={f1.log_likelihood:.6f}, "
          f"Δ vs Count = {f1.log_likelihood - ll_count:+.4f}  [{time.time()-t0:.1f}s]")

    # 2) K=1 LogisticShift identity init
    print("# --- LogisticShift K=1 identity ---")
    init_s1 = LogisticShift.identity(K=1)
    t0 = time.time()
    f2 = fit_mixture(sess.tree, profiles, K=1, initial_rates=sess.rates,
                     initial_shift=init_s1,
                     fix_loss=True, fix_root_length=True,
                     min_copies=args.min_copies, max_iter=args.max_iter)
    print(f"# K=1 ident: {f2.n_iter} iters, LL={f2.log_likelihood:.6f}, "
          f"Δ vs Count = {f2.log_likelihood - ll_count:+.4f}  [{time.time()-t0:.1f}s]")

    # 3) K=N warm-started from the bare GLD MLE
    #    (fairest comparison: shift starts at identity, rates already at MLE,
    #     so the optimizer only has to discover the mixture component)
    print(f"# --- LogisticShift K={args.k} warm-start from bare GLD MLE ---")
    init_shift = LogisticShift(
        weights=torch.full((args.k,), 1.0 / args.k, dtype=torch.float64),
        mod_p=torch.linspace(-0.5, 0.5, args.k, dtype=torch.float64),
        mod_q=torch.zeros(args.k, dtype=torch.float64),
    )
    t0 = time.time()
    f3 = fit_mixture(sess.tree, profiles, K=args.k,
                     initial_rates=f1.rates,         # warm-start rates
                     initial_shift=init_shift,        # mild asymmetry
                     fix_loss=True, fix_root_length=True,
                     min_copies=args.min_copies, max_iter=args.max_iter,
                     verbose=False)
    print(f"# K={args.k} warm: {f3.n_iter} iters, LL={f3.log_likelihood:.6f}, "
          f"Δ vs bare GLD = {f3.log_likelihood - f1.log_likelihood:+.4f}, "
          f"Δ vs Count = {f3.log_likelihood - ll_count:+.4f}  [{time.time()-t0:.1f}s]")
    print(f"#   weights: {[f'{w:.4f}' for w in f3.shift.weights.tolist()]}")
    print(f"#   mod_p  : {[f'{w:+.4f}' for w in f3.shift.mod_p.tolist()]}")
    print(f"#   mod_q  : {[f'{w:+.4f}' for w in f3.shift.mod_q.tolist()]}")

    summary = {
        "xml": str(args.xml),
        "num_families": F,
        "max_sum": int(profiles.sum(axis=1).max()),
        "ll_at_count_rates": ll_count,
        "bare_gld": {
            "n_iter": f1.n_iter, "log_likelihood": f1.log_likelihood,
            "delta_vs_count": f1.log_likelihood - ll_count,
        },
        "logistic_shift_k1_identity": {
            "n_iter": f2.n_iter, "log_likelihood": f2.log_likelihood,
            "delta_vs_count": f2.log_likelihood - ll_count,
        },
        f"logistic_shift_k{args.k}_warm": {
            "n_iter": f3.n_iter, "log_likelihood": f3.log_likelihood,
            "delta_vs_bare_gld": f3.log_likelihood - f1.log_likelihood,
            "delta_vs_count": f3.log_likelihood - ll_count,
            "weights": f3.shift.weights.tolist(),
            "mod_p": f3.shift.mod_p.tolist(),
            "mod_q": f3.shift.mod_q.tolist(),
        },
    }
    print()
    print("# === SUMMARY ===")
    print(f"# bare GLD vs Count       : Δ LL = {f1.log_likelihood - ll_count:+.4f}")
    print(f"# K=1 LogisticShift ident : Δ LL = {f2.log_likelihood - ll_count:+.4f}")
    print(f"# K={args.k} LogisticShift warm : Δ LL = {f3.log_likelihood - ll_count:+.4f}")
    if f3.log_likelihood > f1.log_likelihood + 1.0:
        print(f"# ✓ K={args.k} mixture improves over bare GLD by "
              f"{f3.log_likelihood - f1.log_likelihood:+.2f} nats.")
    else:
        print(f"# K={args.k} mixture vs bare GLD: "
              f"{f3.log_likelihood - f1.log_likelihood:+.2f} nats (within noise).")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
