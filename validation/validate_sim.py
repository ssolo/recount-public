"""Validate ``recount`` ML inference against simulated archaeal datasets
from Csurös 2026 (PNAS) supplementary file
``datasets-sims-ED194-E114-D80-P75.countxml.gz``.

Each session in that file contains:

  * the per-edge GLD rates Count fit to the *observed* data of that subset
    (these are stored in the ``<model>`` block);
  * a ``Sim.*`` table — a fresh family-by-leaf profile matrix forward-
    simulated *from those rates*; this is our "truth"-known dataset.

Validation flow per session:

  1. Evaluate the corrected LL of the Sim table at the stored rates.
     This is the LL the inference should be able to recover.
  2. Re-fit with ``recount.ml.fit_rates`` from default cold-start rates.
     A good optimizer should converge to LL close to (1).
  3. Re-fit warm-started from the stored rates.  Both fits should
     produce a similar ranking of per-edge rates (Spearman ~ 1) on a
     log scale.

By default we restrict to profiles with sum ≤ 8 and a cap of 500
families to keep the analytical-NumPy ``fit_rates`` runtime in the
minute range.  Pass ``--max-sum`` / ``--max-families`` to widen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from recount.gld import corrected_log_likelihood
from recount.io.countxml import load_countxml
from recount.ml import default_initial_rates, fit_rates


REPO_ROOT = Path(__file__).resolve().parent.parent


def _log_safe(x):
    return np.log(np.maximum(x, 1e-300))


def validate_session(sess, sim_table, *, max_sum: int, max_families: int,
                     max_iter: int) -> dict:
    prof = sim_table.profiles
    keep = prof.sum(axis=1) <= max_sum
    prof = prof[keep][:max_families]
    F = prof.shape[0]
    print(f"  using {F} families (max_sum={max_sum}, max_W={prof.sum(axis=1).max()})", flush=True)

    ll_truth = corrected_log_likelihood(sess.tree, sess.rates, prof, min_copies=1)
    print(f"  LL at truth (stored rates): {ll_truth:.4f}", flush=True)

    t0 = time.time()
    fr_cold = fit_rates(sess.tree, prof, max_iter=max_iter, backend="torch",
                         fix_loss=True, fix_root_length=True,
                         bound_dup_by_loss=True, bound_gain_by_loss=True)
    t_cold = time.time() - t0
    print(f"  cold-start fit: LL = {fr_cold.log_likelihood:.4f}  "
          f"(Δ vs truth = {fr_cold.log_likelihood - ll_truth:+.4f}, "
          f"{fr_cold.n_iter} iters, {t_cold:.1f}s)", flush=True)

    t0 = time.time()
    fr_warm = fit_rates(sess.tree, prof, initial_rates=sess.rates,
                         max_iter=max_iter, backend="torch",
                         fix_loss=True, fix_root_length=True,
                         bound_dup_by_loss=True, bound_gain_by_loss=True)
    t_warm = time.time() - t0
    print(f"  warm-start fit: LL = {fr_warm.log_likelihood:.4f}  "
          f"(Δ vs truth = {fr_warm.log_likelihood - ll_truth:+.4f}, "
          f"{fr_warm.n_iter} iters, {t_warm:.1f}s)", flush=True)

    # Rate correlations on log scale
    finite = ~np.isinf(fr_warm.rates.length)
    rho_gain = float(spearmanr(_log_safe(fr_warm.rates.gain),
                               _log_safe(sess.rates.gain)).statistic)
    rho_dup = float(spearmanr(_log_safe(fr_warm.rates.dup),
                              _log_safe(sess.rates.dup)).statistic)
    rho_len = float(spearmanr(_log_safe(fr_warm.rates.length[finite]),
                              _log_safe(sess.rates.length[finite])).statistic)
    print(f"  rate Spearman (warm vs truth): gain ρ={rho_gain:.3f}, "
          f"dup ρ={rho_dup:.3f}, length ρ={rho_len:.3f}", flush=True)

    return {
        "num_families": F,
        "max_W": int(prof.sum(axis=1).max()),
        "ll_truth": ll_truth,
        "cold": {
            "ll": fr_cold.log_likelihood,
            "delta_vs_truth": fr_cold.log_likelihood - ll_truth,
            "n_iter": fr_cold.n_iter,
            "runtime_s": t_cold,
        },
        "warm": {
            "ll": fr_warm.log_likelihood,
            "delta_vs_truth": fr_warm.log_likelihood - ll_truth,
            "n_iter": fr_warm.n_iter,
            "runtime_s": t_warm,
            "rho_gain": rho_gain,
            "rho_dup": rho_dup,
            "rho_length": rho_len,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xml", type=Path)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="restrict to these session IDs (default: all sessions with rates+Sim)")
    ap.add_argument("--max-sum", type=int, default=8)
    ap.add_argument("--max-families", type=int, default=500)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "validation" / "results" / "validate_sim.json")
    args = ap.parse_args()

    sessions = load_countxml(args.xml)
    print(f"# loaded {len(sessions)} sessions from {args.xml.name}", flush=True)

    results = {}
    for sid, sess in sessions.items():
        if args.sessions is not None and sid not in args.sessions:
            continue
        if sess.rates is None:
            continue
        sim = next((t for n, t in sess.tables.items() if n.startswith("Sim.")), None)
        if sim is None:
            continue
        print(f"# === session: {sid} ({sess.tree.num_leaves} leaves) ===", flush=True)
        results[sid] = validate_session(sess, sim,
                                        max_sum=args.max_sum,
                                        max_families=args.max_families,
                                        max_iter=args.max_iter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
