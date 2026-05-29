"""MAP fit: weakly-informative log-Normal prior to regularize the boundary.

The unconstrained ML objective for GLD models has an unbounded supremum
along the Pólya-κ asymptote (dup → ∞ collapses to a Poisson). A weak
log-Normal prior on each log(rate) makes the boundary climb finite-cost,
so the MAP is a well-defined unique point.

Prior layout:
    log(gain_v)   ~ N(mu_gain,   sigma)
    log(length_v) ~ N(mu_length, sigma)
    dup_v: see "Coordinate of the dup prior" below.

Defaults (CLI --mu-{gain,dup,length} are always in LOG-RATE units):
        mu_gain   = log(0.1)
        mu_dup    = log(0.5)
        mu_length = log(1.0)
        sigma     = 1.0   ("tight"; ~99 % mass within [mu/22, mu*22])
        sigma     = 3.0   ("almost ML"; only tames the boundary)

Coordinate of the dup prior. With --subcritical (the default, matching
Csurös' Java is_duprate_bounded=true), the optimizer parameterises dup
via logit(dup): dup_v = MAX_PROB_NOT1 · sigmoid(theta_v). The prior
acts on theta directly (theta_v ~ N(mu_dup_opt, sigma²)), so the CLI's
--mu-dup is translated from log-rate to logit-coord internally via
_dup_to_logit. Without that translation, log(0.5) = −0.693 would land
in logit coords and anchor the prior on dup ≈ 0.333, not the
documented dup = 0.5. Each summary JSON records both the user-passed
mu_dup (log-rate) and the actual optimizer-coord mu_dup_in_optimizer_coord.

Same warm-restart cycling + multistart + polish machinery as ml.py.

Convention: dup at the root is held at default 0.5 (see _shared.py).

Usage:
  PYTHONPATH=. python3 validation/map.py --dataset dpann80 --sigma 1.0
  PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \\
      --num-starts 5 --max-cycles 15
  PYTHONPATH=. python3 validation/map.py --dataset all --sigma 1.0
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
    DATASETS, MAX_PROB_NOT1, _dup_to_logit,
    fit_bfgs_cycles, load_dataset, make_objgrad_map,
    make_objgrad_map_brownian,
    perturb_init, reconstruct, write_outputs,
)


def _mu_dup_for_optimizer(mu_dup_log: float, subcritical: bool) -> float:
    """Translate the CLI's --mu-dup (always in log-rate space) into the
    coordinate the optimizer actually uses.

    When subcritical=True the dup block sits in logit(dup) coordinates
    (per the MAX_PROB_NOT1 logit transform), so the prior anchor must
    also be expressed in logit coords. Without this translation the
    prior gets centred on logit(dup)=−0.693 → dup ≈ 0.333 instead of
    the user-intended dup=0.5.
    """
    if not subcritical:
        return mu_dup_log
    dup_value = float(np.exp(mu_dup_log))
    if dup_value >= MAX_PROB_NOT1:
        raise ValueError(
            f"--mu-dup={mu_dup_log} implies dup={dup_value} ≥ MAX_PROB_NOT1; "
            "out of the bounded logit domain.")
    return float(_dup_to_logit(np.array([dup_value]))[0])


def fit_dataset(label: str, out_dir: Path, sigma: float, mu_gain: float,
                mu_dup: float, mu_length: float, num_starts: int,
                cycle_iters: int, max_cycles: int, seed: int,
                perturb_start_sigma: float = 0.3,
                min_copies_override: int | None = None,
                subcritical: bool = True,
                init_mode: str = "random",
                optimizer: str = "native_bfgs",
                prior: str = "independent",
                sigma_brownian_gain: float = 1.0,
                sigma_brownian_dup: float = 1.0,
                sigma_brownian_length: float = 1.0,
                sigma_root: float = 5.0) -> dict:
    # CLI --mu-dup is in log-rate space (user-friendly). Both make_objgrad_map
    # (independent prior) and make_objgrad_map_brownian (brownian prior)
    # expect this in the same coordinate as the optimizer vector — logit
    # space when subcritical=True. Translate once at the boundary so the
    # rest of the function (printing + objective + reporting) sees a
    # single value in the optimizer's space.
    mu_dup_opt = _mu_dup_for_optimizer(mu_dup, subcritical)
    print(f"\n========== {label} (MAP, prior={prior}, sigma={sigma}) ==========")
    tree, profiles, csuros_rates, mc, biology = load_dataset(label, min_copies_override=min_copies_override)
    print(f"  F={profiles.shape[0]:,}, N={tree.num_nodes}, leaves={tree.num_leaves}, Ωmin={mc}")
    if prior == "brownian":
        print(f"  Prior: tree-Brownian autocorrelated log-rate")
        print(f"    σ_brownian (gain, dup, length) = "
              f"({sigma_brownian_gain}, {sigma_brownian_dup}, {sigma_brownian_length})")
        print(f"    root anchor mu_gain={mu_gain:.3f} (log), "
              f"mu_dup={mu_dup:.3f} (log) → {mu_dup_opt:+.3f} ({'logit' if subcritical else 'log'}), "
              f"mu_length={mu_length:.3f} (log), sigma_root={sigma_root}")
    else:
        print(f"  Prior: independent log-Normal per node")
        print(f"    log(gain)~N({mu_gain:.3f},{sigma}), "
              f"{'logit' if subcritical else 'log'}(dup)~N({mu_dup_opt:+.3f},{sigma}) "
              f"[user mu_dup={mu_dup:.3f} in log-rate space], "
              f"log(length)~N({mu_length:.3f},{sigma})")

    baseline = None
    if csuros_rates is not None:
        baseline = reconstruct(tree, csuros_rates, profiles, mc)
        print(f"  Csurös baseline: LL={baseline['ll']:.2f}  L(0)={baseline['L0']:.3f}  "
              f"copies={baseline['root_copies_corr']:.0f}  "
              f"families={baseline['root_families_corr']:.0f}")

    rng = np.random.default_rng(seed)
    overall_best_neglp = np.inf
    overall_best_rates = None
    full_trajectory = []

    print(f"  ----- multistart ({num_starts} starts, init={init_mode}) -----")
    for s in range(num_starts):
        if init_mode == "random":
            init = random_initial_rates(tree, rng)
        else:
            init = (default_initial_rates(tree) if s == 0
                    else perturb_init(default_initial_rates(tree), perturb_start_sigma, rng,
                                      subcritical=subcritical))
        if prior == "brownian":
            def _make_obj(r):
                return make_objgrad_map_brownian(
                    tree, r, profiles, mc,
                    sigma_brownian_gain=sigma_brownian_gain,
                    sigma_brownian_dup=sigma_brownian_dup,
                    sigma_brownian_length=sigma_brownian_length,
                    mu_root_gain=mu_gain,
                    mu_root_dup=mu_dup_opt,
                    mu_root_length=mu_length,
                    sigma_root=sigma_root,
                    subcritical=subcritical,
                )
        else:
            def _make_obj(r):
                return make_objgrad_map(tree, r, profiles, mc, mu_gain, mu_dup_opt, mu_length, sigma,
                                        subcritical=subcritical)
        rates, neglp, traj = fit_bfgs_cycles(
            _make_obj,
            tree, init, cycle_iters=cycle_iters, max_cycles=max_cycles,
            label=f"{label}-start{s+1}",
            subcritical=subcritical,
            optimizer=optimizer,
        )
        print(f"  start {s+1}: -log P = {neglp:.4f}")
        full_trajectory.extend([{"start": s+1, **t} for t in traj])
        if neglp < overall_best_neglp:
            overall_best_neglp = neglp; overall_best_rates = rates
            print(f"  ★ new best -log P = {neglp:.4f}")

    # Final reconstruction (using pure LL, not the regularized objective)
    rates = overall_best_rates
    final = reconstruct(tree, rates, profiles, mc)
    # Recompute log_prior at the best for reporting
    N = tree.num_nodes; root = tree.root
    nonroot_mask = np.arange(N) != root
    if prior == "brownian":
        # Brownian log-prior: per-edge increment of log_rate vs parent. Use the
        # OPTIMIZER vector to compute (so dup is in logit space when subcritical=True).
        from validation._shared import rates_to_x, _brownian_prior_and_grad
        x_best = rates_to_x(rates, root, subcritical=subcritical)
        log_prior, _ = _brownian_prior_and_grad(
            x_best, tree, sigma_brownian_gain, sigma_brownian_dup, sigma_brownian_length,
            mu_root_gain=mu_gain, mu_root_dup=mu_dup_opt, mu_root_length=mu_length,
            sigma_root=sigma_root, subcritical=subcritical,
        )
    else:
        # Independent log-Normal prior. Match the optimizer's coordinate:
        # gain + length in log-space; dup in logit-space when subcritical=True
        # so the prior centre lines up with what the optimizer actually used.
        from validation._shared import _dup_to_logit as _d2l
        sigma2 = sigma * sigma
        log_g = np.log(np.clip(rates.gain, 1e-300, np.inf))
        log_t = np.log(np.clip(rates.length[nonroot_mask], 1e-300, np.inf))
        if subcritical:
            theta_d = _d2l(rates.dup[nonroot_mask])
            dev_d = theta_d - mu_dup_opt
        else:
            log_d = np.log(np.clip(rates.dup[nonroot_mask], 1e-300, np.inf))
            dev_d = log_d - mu_dup_opt
        log_prior = -1.0 / (2*sigma2) * (
            np.sum((log_g - mu_gain)**2) +
            np.sum(dev_d * dev_d) +
            np.sum((log_t - mu_length)**2)
        )

    print(f"\n  Final MAP reconstruction:")
    print(f"    -log P             = {overall_best_neglp:.4f}")
    print(f"    Pure LL            = {final['ll']:.4f}  (vs -log P - prior = {-(overall_best_neglp + log_prior):.4f})")
    print(f"    Log prior          = {log_prior:.4f}")
    print(f"    L(0)               = {final['L0']:.6f}")
    print(f"    root copies (corr) = {final['root_copies_corr']:.2f}")
    print(f"    root families      = {final['root_families_corr']:.2f}")
    print(f"    copies / family    = {final['copies_per_family']:.3f}")
    print(f"    max gain rate      = {rates.gain.max():.2g}  (nodes>100: {(rates.gain>100).sum()})")
    print(f"    max dup  rate      = {rates.dup.max():.2g}  (nodes>1000: {(rates.dup>1000).sum()})")
    if baseline is not None:
        d_ll = final["ll"] - baseline["ll"]
        d_c = final["root_copies_corr"] - baseline["root_copies_corr"]
        d_f = final["root_families_corr"] - baseline["root_families_corr"]
        print(f"    Δ vs Csurös: ΔLL={d_ll:+.2f}  Δcopies={d_c:+.0f}  Δfamilies={d_f:+.0f}")

    # Save outputs: countxml + branches.csv + rates.npz + summary JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(label, tree, rates, profiles, mc,
                  out_dir / f"{label}_map_sigma{sigma}")
    np.savez(out_dir / f"{label}_sigma{sigma}_final_rates.npz",
             gain=rates.gain, loss=rates.loss, dup=rates.dup, length=rates.length)
    with open(out_dir / f"{label}_sigma{sigma}_summary.json", "w") as fh:
        prior_block = {
            "kind": prior,
            "mu_gain": mu_gain, "mu_dup": mu_dup, "mu_length": mu_length,
            "mu_dup_in_optimizer_coord": mu_dup_opt,
            "dup_coord": "logit" if subcritical else "log",
        }
        if prior == "brownian":
            prior_block.update({
                "sigma_brownian_gain": sigma_brownian_gain,
                "sigma_brownian_dup": sigma_brownian_dup,
                "sigma_brownian_length": sigma_brownian_length,
                "sigma_root": sigma_root,
            })
        else:
            prior_block["sigma"] = sigma
        json.dump({
            "label": label, "F": int(profiles.shape[0]), "N": int(tree.num_nodes),
            "leaves": int(tree.num_leaves), "min_copies": int(mc),
            "prior": prior_block,
            "final": final, "log_prior_at_best": float(log_prior),
            "neg_logp_at_best": float(overall_best_neglp),
            "csuros_baseline": baseline,
            "fit_settings": {"num_starts": num_starts, "cycle_iters": cycle_iters,
                             "max_cycles": max_cycles, "seed": seed,
                             "subcritical": subcritical, "init_mode": init_mode,
                             "prior": prior},
        }, fh, indent=2)
    return {"label": label, "final": final, "csuros_baseline": baseline,
            "log_prior": float(log_prior), "neg_logp": float(overall_best_neglp)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Prior sigma in log-space (default 1.0; 3.0 is weak)")
    p.add_argument("--mu-gain", type=float, default=float(np.log(0.1)))
    p.add_argument("--mu-dup", type=float, default=float(np.log(0.5)))
    p.add_argument("--mu-length", type=float, default=float(np.log(1.0)))
    p.add_argument("--num-starts", type=int, default=3)
    p.add_argument("--cycle-iters", type=int, default=100)
    p.add_argument("--max-cycles", type=int, default=15)
    p.add_argument("--seed", type=int, default=2032)
    p.add_argument("--out-dir", type=Path, default=Path("validation/map_results"))
    p.add_argument("--min-copies-override", type=int, default=None,
                   help="Override the dataset's default Ωmin (use e.g. 4 to "
                        "fit arc269 with Csurös' subset convention).")
    p.add_argument("--no-subcritical", action="store_true",
                   help="Disable the sub-critical Yule constraint (dup ≤ 1). "
                        "Default is to enforce it via L-BFGS-B bounds, matching "
                        "Csurös' Java `is_duprate_bounded=true`.")
    p.add_argument("--init", choices=["random", "uniform"], default="random",
                   help="Per-start initialisation. 'random' (default) matches "
                        "Csurös' Java (per-node Uniform/Exp). 'uniform' is "
                        "deprecated for non-trivial fits (degenerate Hessian).")
    p.add_argument("--optimizer",
                   choices=["BFGS", "trust-constr", "csuros_dfpmin", "native_bfgs"],
                   default="native_bfgs",
                   help="Per-cycle optimizer. 'native_bfgs' (default) = bit-faithful C "
                        "port of Csurös 2026's Java NR dfpmin from "
                        "https://github.com/miklosc/Count (recount_bfgs.c). 'BFGS' = "
                        "scipy BFGS. 'csuros_dfpmin' = pure-Python port. 'trust-constr' = slow.")
    p.add_argument("--num-threads", type=int, default=0,
                   help="Native gradient thread count. 0 (default) uses "
                        "RECOUNT_NUM_THREADS env var if set, otherwise all "
                        "cores. For N parallel fits, set to ncpu//N.")
    p.add_argument("--prior", choices=["independent", "brownian"], default="independent",
                   help="Prior family. 'independent' (default, back-compat) puts a per-node "
                        "log-Normal(mu, sigma) on each rate. 'brownian' replaces with a "
                        "tree-Brownian autocorrelated prior: log_rate_v − log_rate_pa(v) "
                        "~ N(0, σ²_brownian_axis). Per-edge shrinkage; matches Thorne-"
                        "Kishino-Painter 1998 on log-scale. See plan in "
                        "/Users/ssolo/.claude/plans/read-this-repo-and-humming-flute.md")
    p.add_argument("--sigma-brownian-gain", type=float, default=1.0,
                   help="Brownian prior std for log-gain increments (only with --prior brownian)")
    p.add_argument("--sigma-brownian-dup", type=float, default=1.0,
                   help="Brownian prior std for logit-dup (or log-dup) increments")
    p.add_argument("--sigma-brownian-length", type=float, default=1.0,
                   help="Brownian prior std for log-length increments")
    p.add_argument("--sigma-root", type=float, default=5.0,
                   help="Brownian prior: std of independent root-anchor Gaussian "
                        "(centred at mu_gain) — wide enough not to dominate.")
    args = p.parse_args(argv)
    import os as _os
    if args.num_threads > 0:
        _os.environ["RECOUNT_NUM_THREADS"] = str(args.num_threads)

    subcritical = not args.no_subcritical
    print(f"[MAP] subcritical (dup ≤ 1) = {subcritical}; init = {args.init}")
    labels = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    t0 = time.time()
    for label in labels:
        results.append(fit_dataset(
            label, args.out_dir, args.sigma, args.mu_gain, args.mu_dup, args.mu_length,
            args.num_starts, args.cycle_iters, args.max_cycles, args.seed,
            min_copies_override=args.min_copies_override,
            subcritical=subcritical,
            init_mode=args.init,
            optimizer=args.optimizer,
            prior=args.prior,
            sigma_brownian_gain=args.sigma_brownian_gain,
            sigma_brownian_dup=args.sigma_brownian_dup,
            sigma_brownian_length=args.sigma_brownian_length,
            sigma_root=args.sigma_root))
    print(f"\nTotal wall: {time.time()-t0:.0f}s")

    if len(results) > 1:
        print(f"\n========== SUMMARY (MAP sigma={args.sigma} vs Csurös where available) ==========")
        print(f"{'dataset':<10} {'F':>8} | {'Csurös LL':>13} {'MAP LL':>13} {'ΔLL':>9} | "
              f"{'Csurös cpy':>11} {'MAP cpy':>10} {'Δcpy':>8} | "
              f"{'Csurös fam':>10} {'MAP fam':>9} {'Δfam':>7}")
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
