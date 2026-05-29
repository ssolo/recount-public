"""Multistart-polish ML fit: full BFGS + warm-restart cycling + multistart + polish.

Two ML configurations are documented in our papers:

- **Cold-start ML** — single warm-restart BFGS cycling from the default
  uniform init. Invoked as:
      ml.py --num-starts 1 --polish-sigmas ""
- **Multistart-polish ML** — multistart from N perturbed inits + lognormal
  polish around the best basin. Invoked as:
      ml.py --num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2

For any of the 5 datasets:
  1. Multistart from `num_starts` initializations (uniform default + perturbed).
  2. Each start: warm-restart BFGS cycling (up to `max_cycles` × `cycle_iters`).
  3. Polish: re-launch from the best-so-far rates with small lognormal
     perturbations to escape any final stagnation.
  4. Save final rates, full trajectory, JSON summary.
  5. For subsets only: compare against Csurös' published ML reconstruction.

Compute budget:
  - subsets (D80/P75/E114/ED194): ~5-15 min each on M4 Max.
  - arc269: ~18 min for multistart, ~50 min for multistart + polish.

For comparison, Csurös' SI B.1 reports his ED194 fit took 200 vCPU-hr on
9 threads (MacBook Pro). Extrapolating to arc269 would be ~4,000 vCPU-hr.

Convention: dup at the root is held at the default 0.5 to avoid the
"ROOTLOSS not supported" NotImplementedError; see validation/_shared.py.

Usage:
  PYTHONPATH=. python3 validation/ml.py --dataset dpann80
  PYTHONPATH=. python3 validation/ml.py --dataset arc269 \\
      --num-starts 5 --max-cycles 15 --polish-sigmas 0.05,0.10 --num-polish 3
  PYTHONPATH=. python3 validation/ml.py --dataset all --out-dir /tmp/all_ml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recount.ml import default_initial_rates, random_initial_rates
from recount.rates import GLDRates
from validation._shared import (
    DATASETS, fit_bfgs_cycles, load_dataset, make_objgrad_ml,
    perturb_init, reconstruct, write_outputs,
)


def fit_dataset(label: str, out_dir: Path, num_starts: int, cycle_iters: int,
                max_cycles: int, polish_sigmas: list[float], num_polish: int,
                seed: int, perturb_start_sigma: float = 0.3,
                optimizer: str = "BFGS",
                min_copies_override: int | None = None,
                subcritical: bool = True,
                init_mode: str = "random") -> dict:
    print(f"\n========== {label} (ML fit) ==========")
    tree, profiles, csuros_rates, mc, biology = load_dataset(label, min_copies_override=min_copies_override)
    print(f"  F={profiles.shape[0]:,}, N={tree.num_nodes}, leaves={tree.num_leaves}, Ωmin={mc}")
    print(f"  Biology: {biology}")

    # Baseline reconstruction (skip for arc269 which has no published rates)
    baseline = None
    if csuros_rates is not None:
        baseline = reconstruct(tree, csuros_rates, profiles, mc)
        print(f"  Csurös baseline: LL={baseline['ll']:.2f}  L(0)={baseline['L0']:.3f}  "
              f"root copies={baseline['root_copies_corr']:.0f}  "
              f"families={baseline['root_families_corr']:.0f}")

    rng = np.random.default_rng(seed)
    overall_best_obj = np.inf  # minimised objective (= -LL for ML)
    overall_best_rates = None
    full_trajectory = []

    def make_init(s: int) -> GLDRates:
        """Per-start initialisation. ``init_mode='random'`` matches Csurös'
        Java (per-node Uniform/Exp) and avoids the degenerate-Hessian trap
        that the legacy uniform init hits on arc269. ``init_mode='uniform'``
        keeps the old behaviour for back-compat."""
        if init_mode == "random":
            # Every start gets its own random draw (different draws per start
            # → multistart explores different basins).
            return random_initial_rates(tree, rng)
        # Legacy uniform init: start 1 is the fixed default, starts 2+ are
        # lognormal-perturbed around it.
        return (default_initial_rates(tree) if s == 0
                else perturb_init(default_initial_rates(tree), perturb_start_sigma, rng,
                                  subcritical=subcritical))

    # 1) MULTISTART
    print(f"  ----- multistart ({num_starts} starts, init={init_mode}) -----")
    for s in range(num_starts):
        init = make_init(s)
        rates, obj, traj = fit_bfgs_cycles(
            lambda r: make_objgrad_ml(tree, r, profiles, mc, subcritical=subcritical),
            tree, init, cycle_iters=cycle_iters, max_cycles=max_cycles,
            label=f"{label}-start{s+1}", optimizer=optimizer,
            subcritical=subcritical,
        )
        print(f"  start {s+1}: LL = {-obj:.4f}")
        full_trajectory.extend([{"phase": "multistart", "start": s+1, **t} for t in traj])
        if obj < overall_best_obj:
            overall_best_obj = obj; overall_best_rates = rates
            print(f"  ★ new best LL = {-obj:.4f}")

    # 2) POLISH (lognormal perturbations around best so far)
    if polish_sigmas and num_polish > 0:
        print(f"  ----- polish (sigmas={polish_sigmas}, n={num_polish} per sigma) -----")
        for sigma in polish_sigmas:
            for i in range(num_polish):
                init = perturb_init(overall_best_rates, sigma, rng, subcritical=subcritical)
                rates, obj, traj = fit_bfgs_cycles(
                    lambda r: make_objgrad_ml(tree, r, profiles, mc, subcritical=subcritical),
                    tree, init, cycle_iters=cycle_iters, max_cycles=max(8, max_cycles // 2),
                    label=f"{label}-polish-s{sigma:.2f}-p{i+1}", optimizer=optimizer,
                    subcritical=subcritical,
                )
                full_trajectory.extend([{"phase": "polish", "sigma": sigma, "p": i+1, **t} for t in traj])
                if obj < overall_best_obj:
                    print(f"  ★ polish improved: LL = {-obj:.4f} (Δ +{overall_best_obj - obj:.2f})")
                    overall_best_obj = obj; overall_best_rates = rates

    # 3) Final reconstruction
    rates = overall_best_rates
    final = reconstruct(tree, rates, profiles, mc)
    print(f"\n  Final ML reconstruction:")
    print(f"    LL                = {final['ll']:.4f}")
    print(f"    L(0)              = {final['L0']:.6f}")
    print(f"    root copies (corr)= {final['root_copies_corr']:.2f}")
    print(f"    root families     = {final['root_families_corr']:.2f}")
    print(f"    copies / family   = {final['copies_per_family']:.3f}")
    if baseline is not None:
        d_ll = final["ll"] - baseline["ll"]
        d_c = final["root_copies_corr"] - baseline["root_copies_corr"]
        d_f = final["root_families_corr"] - baseline["root_families_corr"]
        print(f"    Δ vs Csurös: ΔLL={d_ll:+.2f}  Δcopies={d_c:+.0f}  Δfamilies={d_f:+.0f}")

    # Save outputs: countxml + branches.csv + rates.npz + summary JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(label, tree, rates, profiles, mc, out_dir / f"{label}_ml")
    np.savez(out_dir / f"{label}_final_rates.npz",
             gain=rates.gain, loss=rates.loss, dup=rates.dup, length=rates.length)
    with open(out_dir / f"{label}_summary.json", "w") as fh:
        json.dump({
            "label": label, "F": int(profiles.shape[0]), "N": int(tree.num_nodes),
            "leaves": int(tree.num_leaves), "min_copies": int(mc),
            "final": final, "csuros_baseline": baseline,
            "fit_settings": {"num_starts": num_starts, "cycle_iters": cycle_iters,
                             "max_cycles": max_cycles, "polish_sigmas": polish_sigmas,
                             "num_polish": num_polish, "seed": seed,
                             "subcritical": subcritical, "init_mode": init_mode},
        }, fh, indent=2)
    with open(out_dir / f"{label}_trajectory.json", "w") as fh:
        json.dump(full_trajectory, fh, indent=2)
    print(f"  Wrote {out_dir}/{label}_{{final_rates.npz,summary.json,trajectory.json}}")
    return {"label": label, "final": final, "csuros_baseline": baseline}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    p.add_argument("--num-starts", type=int, default=3,
                   help="Multistart count (start 1 = default uniform; others = perturbed)")
    p.add_argument("--cycle-iters", type=int, default=100,
                   help="BFGS iters per warm-restart cycle")
    p.add_argument("--max-cycles", type=int, default=12,
                   help="Max warm-restart cycles per start")
    p.add_argument("--polish-sigmas", type=str, default="0.05,0.10",
                   help="Comma-separated sigmas for polish perturbations (empty = skip polish)")
    p.add_argument("--num-polish", type=int, default=2, help="Polish runs per sigma")
    p.add_argument("--seed", type=int, default=2031)
    p.add_argument("--optimizer",
                   choices=["BFGS", "trust-constr", "csuros_dfpmin", "native_bfgs"],
                   default="native_bfgs",
                   help="Per-cycle optimizer. 'native_bfgs' (default) = bit-faithful C "
                        "port of Csurös' Java NR dfpmin (recount_bfgs.c) — Csurös-matching "
                        "constraints + ~10x faster Hessian update than scipy. "
                        "'BFGS' = scipy BFGS + Wolfe line search. 'csuros_dfpmin' = "
                        "pure-Python port of the same algorithm (slower; for trajectory "
                        "debugging). 'trust-constr' = scipy trust-region (slow; ruled out).")
    p.add_argument("--out-dir", type=Path, default=Path("validation/sota_ml_results"))
    p.add_argument("--min-copies-override", type=int, default=None,
                   help="Override the dataset's default Ωmin (use e.g. 4 to "
                        "fit arc269 with Csurös' subset convention). Affects "
                        "only the L(0) correction, not which profiles are loaded.")
    p.add_argument("--no-subcritical", action="store_true",
                   help="Disable the sub-critical Yule constraint (dup ≤ loss = 1). "
                        "Default is to enforce it via L-BFGS-B bounds, matching "
                        "Csurös' Java `is_duprate_bounded=true`.")
    p.add_argument("--num-threads", type=int, default=0,
                   help="Native gradient thread count. 0 (default) uses "
                        "RECOUNT_NUM_THREADS env var if set, otherwise all "
                        "cores. When running N validation jobs in parallel, "
                        "set this to ncpu//N (or set RECOUNT_NUM_THREADS) to "
                        "avoid the gradient threads from all jobs fighting "
                        "for the same cores. E.g. on a 24-core M2 Ultra "
                        "with 4 parallel fits: --num-threads 6.")
    p.add_argument("--init", choices=["random", "uniform"], default="random",
                   help="Per-start initialisation. 'random' (default) matches "
                        "Csurös' Java (SI Section B.1: random GLD model with "
                        "seed 2025): per-node λ~U(0,0.5), γ~U(0,0.2), t~Exp(1). "
                        "'uniform' uses the constant (γ=0.1, λ=0.5, t=1) starting "
                        "point — DEPRECATED for non-trivial fits because the "
                        "Hessian is degenerate at this point under the dup<=1 "
                        "constraint and BFGS gets trapped at a local optimum "
                        "(verified on arc269).")
    args = p.parse_args(argv)

    # Thread autoscale: explicit CLI value overrides env var; env var
    # `RECOUNT_NUM_THREADS` is the default. The native gradient reads this
    # at call time via `recount.native_backend._resolve_num_threads`.
    import os as _os
    if args.num_threads > 0:
        _os.environ["RECOUNT_NUM_THREADS"] = str(args.num_threads)
    nt_eff = _os.environ.get("RECOUNT_NUM_THREADS", "0 (= all cores)")

    polish_sigmas = ([float(s) for s in args.polish_sigmas.split(",") if s.strip()]
                     if args.polish_sigmas else [])
    labels = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    t0 = time.time()
    subcritical = not args.no_subcritical
    print(f"[ML] subcritical (dup ≤ 1) = {subcritical}; init = {args.init}; "
          f"num_threads = {nt_eff}")
    for label in labels:
        results.append(fit_dataset(label, args.out_dir, args.num_starts, args.cycle_iters,
                                   args.max_cycles, polish_sigmas, args.num_polish, args.seed,
                                   optimizer=args.optimizer,
                                   min_copies_override=args.min_copies_override,
                                   subcritical=subcritical,
                                   init_mode=args.init))
    print(f"\nTotal wall: {time.time()-t0:.0f}s")

    if len(results) > 1:
        print("\n========== SUMMARY (ML fit vs Csurös where available) ==========")
        print(f"{'dataset':<10} {'F':>8} | {'Csurös LL':>13} {'Multi-ML LL':>13} {'ΔLL':>9} | "
              f"{'Csurös cpy':>11} {'Multistart-polish ML cpy':>10} {'Δcpy':>8} | "
              f"{'Csurös fam':>10} {'Multi-ML fam':>9} {'Δfam':>7}")
        for r in results:
            c = r["csuros_baseline"]; f = r["final"]
            cb = c if c is not None else {"ll": float("nan"), "root_copies_corr": float("nan"),
                                         "root_families_corr": float("nan")}
            dl = f["ll"] - cb["ll"] if c is not None else float("nan")
            dc = f["root_copies_corr"] - cb["root_copies_corr"] if c is not None else float("nan")
            df = f["root_families_corr"] - cb["root_families_corr"] if c is not None else float("nan")
            print(f"{r['label']:<10} {f['F']:>8,} | "
                  f"{cb['ll']:>13.1f} {f['ll']:>13.1f} {dl:>+9.1f} | "
                  f"{cb['root_copies_corr']:>11,.0f} {f['root_copies_corr']:>10,.0f} {dc:>+8,.0f} | "
                  f"{cb['root_families_corr']:>10,.0f} {f['root_families_corr']:>9,.0f} {df:>+7,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
