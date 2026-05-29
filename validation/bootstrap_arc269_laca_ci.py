"""Non-parametric bootstrap for arc269 LACA confidence intervals.

For each of N bootstrap iterations:
  1. Resample F=90,243 family indices with replacement.
  2. Re-fit MAP σ=1 on the bootstrap sample, warm-started from the
     known 15-start global rates (so we converge fast to the same basin
     unless the bootstrap fluctuation is large).
  3. Record (cp, fm, cpf, max_dup, LL) of the re-fit.

Output: a per-bootstrap summary JSON + an aggregate CI report.

Approximate cost: ~10 minutes per bootstrap on M4 Max with warm-start
(vs ~50 minutes per fresh 15-start MAP). 20 bootstraps ≈ 3-4 hours.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recount.rates import GLDRates
from validation._shared import (
    fit_bfgs_cycles, load_dataset, make_objgrad_map, reconstruct,
)


def fit_one_bootstrap(tree, rng_indices, base_rates, profiles, mc, sigma,
                      cycle_iters: int = 100, max_cycles: int = 12) -> dict:
    """Fit MAP σ=1 on a bootstrap sample, warm-started from base_rates."""
    boot_profiles = profiles[rng_indices]
    rates, neglp, traj = fit_bfgs_cycles(
        lambda r: make_objgrad_map(tree, r, boot_profiles, mc,
                                    mu_gain=float(np.log(0.1)),
                                    mu_dup=float(np.log(0.5)),
                                    mu_length=0.0,
                                    sigma=sigma),
        tree, base_rates, cycle_iters=cycle_iters, max_cycles=max_cycles,
        label="boot",
    )
    rec = reconstruct(tree, rates, boot_profiles, mc)
    return {
        "neglp": float(neglp), "LL_raw": float(rec["ll"]),
        "cp": float(rec["root_copies_corr"]),
        "fm": float(rec["root_families_corr"]),
        "cpf": float(rec["copies_per_family"]),
        "L0": float(rec["L0"]),
        "max_gain": float(rates.gain.max()),
        "max_dup": float(rates.dup.max()),
    }, rates


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--cycle-iters", type=int, default=100)
    p.add_argument("--max-cycles", type=int, default=8,
                   help="Per-bootstrap warm-restart cycles (8 should suffice for warm start)")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--global-rates", type=Path,
                   default=Path("validation/outputs_no_dup_cap/profile_likelihood_arc269/"
                                "arc269_sigma1.0_final_rates.npz"),
                   help="Path to .npz with the global-MAP rates (warm-start init). "
                        "Default points at the canonical 15-start LACA quote "
                        "(unbounded; cp=5,426 / fm=3,070 / cpf=1.77). The bounded "
                        "15-start has not been re-run; if one exists at "
                        "validation/outputs/profile_likelihood_arc269/, point at "
                        "that file explicitly.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("validation/outputs_no_dup_cap/bootstrap_arc269"),
                   help="Default targets the existing unbounded bootstrap "
                        "(matches the 95%% CI cited in PROGRESS.md / VALIDATION.md).")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading arc269 + global rates from {args.global_rates}", flush=True)
    tree, profiles, _, mc, biology = load_dataset("arc269")
    F = profiles.shape[0]
    rd = np.load(args.global_rates)
    base_rates = GLDRates(tree=tree, gain=rd["gain"], loss=rd["loss"],
                          dup=rd["dup"], length=rd["length"])
    print(f"F={F:,}, N={tree.num_nodes}, σ={args.sigma}", flush=True)

    rng = np.random.default_rng(args.seed)
    results = []
    t0 = time.time()
    for b in range(args.n_bootstrap):
        bt0 = time.time()
        indices = rng.integers(0, F, size=F)
        try:
            res, rates = fit_one_bootstrap(
                tree, indices, base_rates, profiles, mc, args.sigma,
                cycle_iters=args.cycle_iters, max_cycles=args.max_cycles)
        except Exception as e:
            print(f"  boot {b+1}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        res["wall_s"] = time.time() - bt0
        res["b"] = b
        results.append(res)
        print(f"  boot {b+1:2d}/{args.n_bootstrap}: cp={res['cp']:.0f}  fm={res['fm']:.0f}  "
              f"cpf={res['cpf']:.3f}  max_dup={res['max_dup']:.2g}  "
              f"-logP={res['neglp']:.1f}  wall={res['wall_s']:.0f}s", flush=True)
        # Save each bootstrap incrementally
        np.savez(args.out_dir / f"boot{b:03d}_rates.npz",
                 gain=rates.gain, loss=rates.loss, dup=rates.dup, length=rates.length,
                 indices=indices)

    if not results:
        print("No successful bootstraps. Exiting.")
        return 1

    print(f"\nTotal wall: {time.time()-t0:.0f}s ({len(results)} ok / {args.n_bootstrap} attempted)")

    # Aggregate
    arrs = {k: np.array([r[k] for r in results]) for k in
            ["cp", "fm", "cpf", "max_dup", "max_gain", "L0", "LL_raw", "neglp"]}
    print("\n========== Bootstrap CI ==========")
    print(f"{'param':<10} {'mean':>10} {'sd':>10} {'2.5%':>10} {'50%':>10} {'97.5%':>10}")
    summary = {}
    for k, v in arrs.items():
        mean = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        q025, q50, q975 = (float(np.quantile(v, q)) for q in [0.025, 0.5, 0.975])
        print(f"{k:<10} {mean:>10.4g} {sd:>10.4g} {q025:>10.4g} {q50:>10.4g} {q975:>10.4g}")
        summary[k] = {"mean": mean, "sd": sd, "q025": q025, "q50": q50, "q975": q975,
                      "values": v.tolist()}

    out_json = args.out_dir / "bootstrap_summary.json"
    out_json.write_text(json.dumps({
        "n_bootstrap": args.n_bootstrap, "n_ok": len(results),
        "sigma": args.sigma, "F": int(F),
        "stats": summary,
        "per_boot": results,
    }, indent=2))
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
