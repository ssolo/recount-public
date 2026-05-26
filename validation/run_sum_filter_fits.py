"""Sum-filter sweep runner.

Run MAP Brownian sigma=1 fits at sum>=N for each (dataset, N) pair on
the dataset's profile filtered to families with sum>=N over that
dataset's own leaves. Each fit warm-starts from the next-looser fit
(N-1) to stay in the same basin as the existing N=1..7 sweep.

Usage:
  PYTHONPATH=. python3 validation/run_sum_filter_fits.py
"""
import json, sys, time
import numpy as np
from pathlib import Path
from validation._shared import (
    load_dataset, make_objgrad_map_brownian, fit_bfgs_cycles, rates_to_x,
    write_outputs, reconstruct, GLDRates, _brownian_prior_and_grad,
)
from validation.map import _mu_dup_for_optimizer

ROOT = Path("/Users/ssolo/src/recount")
OUT = ROOT / "validation/outputs"
SIGMA = 1.0
SUBCRITICAL = True
MU_GAIN = float(np.log(0.1))
MU_DUP = float(np.log(0.5))
MU_LENGTH = float(np.log(1.0))
SIGMA_ROOT = 5.0
SIGMA_BROWN = 1.0


def fit_dir(dataset, N):
    """Filesystem location of the existing/expected sum>=N fit."""
    if dataset == "arc269":
        if N == 1: return OUT / "brownian_arc269"
        return OUT / f"brownian_arc269_omin4_sum{N}"
    else:
        if N == 1: return OUT / f"brownian_{dataset}_full"
        if N == 4: return OUT / f"brownian_{dataset}"
        return OUT / f"brownian_{dataset}_sum{N}"


def load_warm_rates(dataset, N_from):
    d = fit_dir(dataset, N_from)
    if d is None: return None
    rates_path = d / f"{dataset}_sigma1.0_final_rates.npz"
    if not rates_path.exists():
        return None
    z = np.load(rates_path)
    return dict(gain=z["gain"], loss=z["loss"], dup=z["dup"], length=z["length"])


def run_one(dataset, N, warm_N):
    out_subdir = fit_dir(dataset, N)
    out_subdir.mkdir(parents=True, exist_ok=True)
    summary_path = out_subdir / f"{dataset}_sigma{SIGMA}_summary.json"
    if summary_path.exists():
        print(f"  SKIP: {summary_path.name} already exists")
        return
    t_start = time.time()
    print(f"\n=========== {dataset} sum>={N}  (warm from sum>={warm_N}) ===========")
    tree, profiles_full, _, _, _ = load_dataset(dataset)
    sums = profiles_full.sum(axis=1)
    mask = sums >= N
    profiles = np.ascontiguousarray(profiles_full[mask])
    F_orig = profiles_full.shape[0]
    F_kept = profiles.shape[0]
    print(f"  filter sum>={N}: F {F_orig:,} -> {F_kept:,}  (N nodes={tree.num_nodes}, leaves={tree.num_leaves})")
    if F_kept < 50:
        print(f"  SKIP: too few families")
        return

    rates_dict = load_warm_rates(dataset, warm_N)
    if rates_dict is None:
        raise RuntimeError(f"no warm-start rates for {dataset} sum>={warm_N} "
                           f"(expected at {fit_dir(dataset, warm_N)})")
    init_rates = GLDRates(tree=tree, gain=rates_dict["gain"], loss=rates_dict["loss"],
                          dup=rates_dict["dup"], length=rates_dict["length"])

    mc = 1   # match existing sum-sweep fits' min_copies_override=1
    mu_dup_opt = _mu_dup_for_optimizer(MU_DUP, SUBCRITICAL)

    def _make_obj(r):
        return make_objgrad_map_brownian(
            tree, r, profiles, mc,
            sigma_brownian_gain=SIGMA_BROWN, sigma_brownian_dup=SIGMA_BROWN,
            sigma_brownian_length=SIGMA_BROWN,
            mu_root_gain=MU_GAIN, mu_root_dup=mu_dup_opt, mu_root_length=MU_LENGTH,
            sigma_root=SIGMA_ROOT, subcritical=SUBCRITICAL,
        )
    rates, neglp, traj = fit_bfgs_cycles(
        _make_obj, tree, init_rates,
        cycle_iters=100, max_cycles=12,
        label=f"{dataset}-sum{N}",
        subcritical=SUBCRITICAL, optimizer="native_bfgs",
    )
    final = reconstruct(tree, rates, profiles, mc)
    wall = time.time() - t_start
    print(f"  done in {wall:.1f}s  -log P = {neglp:.4f}  LL = {final['ll']:.4f}  "
          f"L(0) = {final['L0']:.4f}  cp = {final['root_copies_corr']:.2f}  "
          f"fm = {final['root_families_corr']:.2f}  cpf = {final['copies_per_family']:.3f}")

    out_stem = f"{dataset}_map_sigma{SIGMA}"
    write_outputs(dataset, tree, rates, profiles, mc, out_subdir / out_stem)
    np.savez(out_subdir / f"{dataset}_sigma{SIGMA}_final_rates.npz",
             gain=rates.gain, loss=rates.loss, dup=rates.dup, length=rates.length)
    with open(summary_path, "w") as fh:
        json.dump({
            "F_original": int(F_orig), "F_kept": int(F_kept), "F": int(F_kept),
            "min_copies": int(mc),
            "filter": f"sum>={N}",
            "init": f"warm_start_from_{fit_dir(dataset, warm_N).name}",
            "final": final,
            "wall_s": wall,
            "fit_settings": {
                "cycle_iters": 100, "max_cycles": 12, "subcritical": True,
                "optimizer": "native_bfgs", "num_threads": 16, "prior": "brownian",
                "sigma_brownian_gain": SIGMA_BROWN, "sigma_brownian_dup": SIGMA_BROWN,
                "sigma_brownian_length": SIGMA_BROWN, "num_starts": 1,
                "min_copies_override": 1, "profile_filter": f"sum>={N}",
                "init_mode": f"warm_start_from_{fit_dir(dataset, warm_N).name}",
            }
        }, fh, indent=2)


def main():
    # Datasets and their warm-start N=7 baseline
    DATASETS = ["dpann80", "proteo75", "eury114", "ed194", "arc269"]
    # For each dataset, chain N=8 from N=7, N=9 from N=8, N=10 from N=9
    for ds in DATASETS:
        for n in (8, 9, 10):
            run_one(ds, n, warm_N=n - 1)


if __name__ == "__main__":
    main()
