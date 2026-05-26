"""Brownian-extend ML — release the Brownian prior after MAP convergence.

Idea: the tree-Brownian prior smooths log-rate variation between adjacent
branches and reliably finds a *biologically tight* basin (cpf 1.1-1.3 on
the subsets vs 1.7+ for cold MAP), but the prior itself biases the rates
away from the data-likelihood peak. After Brownian MAP converges, drop
the prior and let pure ML fine-tune for a few cycles — the warm start
keeps the optimizer in the same basin while the rates relax toward
their maximum-likelihood values.

For each dataset:
  1. Load brownian_<ds>/<ds>_final_rates.npz (must exist; produced by
     validation/map.py --prior brownian).
  2. Construct ML objective via validation/_shared.make_objgrad_ml.
  3. Run native_bfgs cycles from the Brownian-MAP rates as initial x.
  4. Save outputs to brownian_extend_<ds>/<ds>_ml.{branches.csv,
     countxml.gz}, <ds>_final_rates.npz, <ds>_summary.json.

The short cycle budget (default 5 cycles × 100 iters) is enough for the
local fine-tuning step; if the rates were already at the ML peak the
optimizer hits early-stop with iters=0 within 1-2 cycles.

Usage:
    PYTHONPATH=. python3 validation/brownian_extend_ml.py --dataset dpann80
    PYTHONPATH=. python3 validation/brownian_extend_ml.py --dataset all
    PYTHONPATH=. python3 validation/brownian_extend_ml.py \\
        --dataset coleman --num-threads 16 --max-cycles 8

Both the dataset's Brownian dir and the optimizer setup mirror the
canonical bounded-fit recipe used by validation/queue_full_subset_run.sh.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recount.gld import GLDRates
from validation._shared import (
    DATASETS, load_dataset, fit_bfgs_cycles, make_objgrad_ml,
    reconstruct, write_outputs,
)


def run_one(ds: str, num_threads: int, cycle_iters: int, max_cycles: int,
            seed: int, base_out: Path,
            min_copies_override: int | None = None,
            rates_from: Path | None = None,
            out_subdir: str | None = None,
            label_tag: str | None = None) -> int:
    print(f"\n===== {ds}{('  Ωmin='+str(min_copies_override)) if min_copies_override else ''} =====", flush=True)
    tree, profiles, _, mc, biology = load_dataset(ds, min_copies_override=min_copies_override)
    print(f"  F={profiles.shape[0]:,}  N={tree.num_nodes}  Ωmin={mc}  | {biology}")

    if rates_from is not None:
        brown_rates = Path(rates_from)
    else:
        brown_rates = Path(f"validation/outputs/brownian_{ds}/{ds}_sigma1.0_final_rates.npz")
        if not brown_rates.exists():
            brown_rates = Path(f"validation/outputs/brownian_{ds}/{ds}_final_rates.npz")
    if not brown_rates.exists():
        print(f"  SKIP {ds}: no Brownian-MAP rates at {brown_rates}")
        return 1

    print(f"  loading Brownian rates from {brown_rates}")
    d = np.load(brown_rates)
    init = GLDRates(
        tree=tree,
        gain=d["gain"].copy(),
        loss=d["loss"].copy(),
        dup=d["dup"].copy(),
        length=d["length"].copy(),
    )
    print(f"  init rates: gain median={np.median(init.gain):.4f} "
          f"dup median={np.median(init.dup):.4f} "
          f"length median={np.median(init.length[np.isfinite(init.length)]):.4f}")

    out_dir = base_out / (out_subdir if out_subdir else f"brownian_extend_{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lbl = label_tag if label_tag else f"{ds}-brown-extend-ml"
    best_rates, best_obj, traj = fit_bfgs_cycles(
        objgrad_factory=lambda r: make_objgrad_ml(tree, r, profiles,
                                                  min_copies=mc, subcritical=True),
        tree=tree, init_rates=init, label=lbl,
        cycle_iters=cycle_iters, max_cycles=max_cycles,
        optimizer="native_bfgs", gtol=1e-7, subcritical=True,
        early_stop_iters_zero=2,
    )
    wall = time.time() - t0
    print(f"  brown-extend ML done in {wall:.1f}s  obj={best_obj:.4f}", flush=True)

    res = reconstruct(tree, best_rates, profiles, mc)
    print(f"  root: cp={res['root_copies_corr']:.0f}  fm={res['root_families_corr']:.0f}  "
          f"cpf={res['copies_per_family']:.3f}  L0={res['L0']:.4f}")

    out_prefix = out_dir / f"{ds}_ml"
    write_outputs(ds, tree, best_rates, profiles, mc, out_prefix)
    np.savez(out_dir / f"{ds}_final_rates.npz",
             gain=best_rates.gain, loss=best_rates.loss,
             dup=best_rates.dup,   length=best_rates.length)
    summary = {
        "label": ds, "init": f"brownian_{ds}",
        "F": int(profiles.shape[0]), "N": int(tree.num_nodes),
        "leaves": int(tree.num_leaves), "min_copies": int(mc),
        "final": {
            "ll": -float(best_obj), "L0": res["L0"],
            "root_copies_corr": res["root_copies_corr"],
            "root_families_corr": res["root_families_corr"],
            "copies_per_family": res["copies_per_family"],
        },
        "wall_s": wall, "trajectory": traj,
        "fit_settings": {
            "init_mode": "brownian_extend",
            "init_rates_path": str(brown_rates),
            "cycle_iters": cycle_iters, "max_cycles": max_cycles,
            "seed": seed, "subcritical": True,
            "optimizer": "native_bfgs", "num_threads": num_threads,
            "polish_sigmas": [], "num_polish": 0, "num_starts": 1,
            "prior": "none",
        },
    }
    with open(out_dir / f"{ds}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {out_prefix}.branches.csv + {out_dir}/{ds}_summary.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True,
                    choices=sorted(DATASETS) + ["all"],
                    help="Dataset label or 'all' for every non-arc269 dataset that has a Brownian fit.")
    ap.add_argument("--cycle-iters", type=int, default=100)
    ap.add_argument("--max-cycles", type=int, default=5,
                    help="Short by design — Brownian-warm start usually needs 2-4 cycles.")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("validation/outputs"),
                    help="Base dir; per-dataset subdir brownian_extend_<ds>/ is created here.")
    ap.add_argument("--min-copies-override", type=int, default=None,
                    help="If set, override the dataset's default Ωmin (e.g. 4 for arc269 to match the subset convention).")
    ap.add_argument("--rates-from", type=Path, default=None,
                    help="Path to a Brownian-MAP rates NPZ to warm-start from. Defaults to validation/outputs/brownian_<ds>/<ds>_sigma1.0_final_rates.npz.")
    ap.add_argument("--out-subdir", type=str, default=None,
                    help="Override the per-dataset subdir name. Default: brownian_extend_<ds>/.")
    args = ap.parse_args(argv)

    if args.dataset == "all":
        targets = [ds for ds in ("dpann80", "proteo75", "eury114", "ed194",
                                 "williams", "coleman") if ds in DATASETS]
    else:
        targets = [args.dataset]

    rc = 0
    for ds in targets:
        rc |= run_one(ds, args.num_threads, args.cycle_iters, args.max_cycles,
                      args.seed, args.out_dir,
                      min_copies_override=args.min_copies_override,
                      rates_from=args.rates_from,
                      out_subdir=args.out_subdir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
