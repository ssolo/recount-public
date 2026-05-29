"""Compute per-family per-node presence posteriors for a saved fit.

For each family f and node v, computes:
    P_present[f, v] = P{ξ_v >= 1 | family f's leaf observations, fitted rates}
    E_copies[f, v]  = E[ξ_v | family f's leaf observations, fitted rates]

Reads rates from <fit-dir>/<dataset>_(final_rates|sigma1.0_final_rates).npz.
Writes a compressed F×N npz to <fit-dir>/<dataset>_per_family_FxN.npz
plus a per-family summary CSV.

Usage — single dataset:
    PYTHONPATH=. python3 validation/per_family_presence.py --dataset dpann80 \\
        --fit-dir validation/outputs/map_sigma1

Usage — every dataset whose rates are present in the directory:
    PYTHONPATH=. python3 validation/per_family_presence.py --all \\
        --fit-dir validation/outputs/map_sigma1

The --all mode auto-detects which of {dpann80, proteo75, eury114, ed194,
williams, arc269} have saved rates in --fit-dir and runs them all in
sequence. Use this after any ml.py / map.py / mixture_ml.py fit to
generate the per-family tables alongside the standard summary outputs.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from recount.native_backend import per_family_posteriors_native
from validation._shared import DATASETS, load_dataset


def _find_rates_npz(fit_dir: Path, dataset: str) -> Path:
    """Try the two known suffixes for saved rate files."""
    for suffix in ("_sigma1.0_final_rates.npz", "_final_rates.npz"):
        p = fit_dir / f"{dataset}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No saved rates for {dataset} in {fit_dir}: tried "
        f"{dataset}_sigma1.0_final_rates.npz and {dataset}_final_rates.npz"
    )


def _run_one(dataset: str, fit_dir: Path, out_dir: Path,
             threshold: float, num_threads: int) -> int:
    print(f"\n===== {dataset} =====", flush=True)
    print(f"Loading {dataset}...", flush=True)
    tree, profiles, _, mc, biology = load_dataset(dataset)
    F = int(profiles.shape[0])
    N = int(tree.num_nodes)
    print(f"  F={F:,}  N={N}  leaves={tree.num_leaves}  Ωmin={mc}")
    print(f"  Biology: {biology}")

    rates_path = _find_rates_npz(fit_dir, dataset)
    print(f"Loading rates from {rates_path}")
    d = np.load(rates_path)
    gain, loss, dup, length = d["gain"], d["loss"], d["dup"], d["length"]

    print(f"Computing per-family posteriors via native backend "
          f"(F×N = {F:,}×{N} = {F*N:,} entries per matrix)...", flush=True)
    t0 = time.time()
    copies, present = per_family_posteriors_native(
        tree, gain, loss, dup, length, profiles, num_threads=num_threads,
    )
    wall = time.time() - t0
    print(f"  done in {wall:.1f}s")

    mb = copies.nbytes / 1e6
    print(f"  copies array: {copies.shape}, {mb:.1f} MB per matrix")

    # Write compressed NPZ — both matrices in one file
    presence_path = out_dir / f"{dataset}_per_family_FxN.npz"
    print(f"Saving F×N posteriors to {presence_path}")
    np.savez_compressed(presence_path, present=present, copies=copies,
                        F=F, N=N, dataset=dataset, threshold=threshold)

    # Quick CSV index. Convention in this codebase: post-order, root has
    # the HIGHEST node index (tree.root == N - 1 usually, but the actual
    # root index is tree.root). "Deepest ancestor" = the most root-ward
    # node where the family is present = the HIGHEST node index v in
    # range(N) with binary[f, v] == True. We also report the shallowest
    # (closest-to-leaves) present node for symmetry.
    print(f"Writing per-family summary CSV (presence threshold {threshold})...")
    summary_path = out_dir / f"{dataset}_per_family_summary.csv"
    binary = present >= threshold          # (F, N) bool
    n_present_per_fam = binary.sum(axis=1) # (F,) — how many nodes each family is "present" at
    has_any = binary.any(axis=1)
    # Deepest (root-most) = highest column index where True. Use a masked argmax:
    cols = np.arange(N, dtype=np.int64)
    deepest_node = np.where(has_any,
                            np.where(binary, cols[np.newaxis, :], -1).max(axis=1),
                            -1)
    # Shallowest (leaf-most) = lowest column index where True
    shallowest_node = np.where(has_any, binary.argmax(axis=1), -1)
    present_at_root = binary[:, tree.root].astype(np.int8)
    # Empirical leaf totals from profile (for cross-check)
    leaf_count_per_fam = profiles.sum(axis=1)
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_idx", "n_nodes_present",
                    "deepest_node_idx",    # most root-ward present node (HIGHEST idx in post-order)
                    "shallowest_node_idx", # most leaf-ward present node (LOWEST idx)
                    "present_at_root",     # 0/1 — is the family present at the actual root?
                    "leaf_count_observed"])
        for f in range(F):
            w.writerow([f, int(n_present_per_fam[f]),
                        int(deepest_node[f]),
                        int(shallowest_node[f]),
                        int(present_at_root[f]),
                        int(leaf_count_per_fam[f])])
    print(f"  wrote {summary_path}")
    return 0


_ALL_DATASETS = ["dpann80", "proteo75", "eury114", "ed194", "williams", "arc269"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(DATASETS),
                   help="Single dataset to process.")
    g.add_argument("--all", action="store_true",
                   help=("Auto-detect every dataset whose rates exist in "
                         "--fit-dir and process them in sequence."))
    ap.add_argument("--fit-dir", type=Path, required=True,
                    help="Directory containing <dataset>_final_rates.npz "
                         "or <dataset>_sigma1.0_final_rates.npz.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Presence threshold for the binary index CSV (default 0.5)")
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write outputs (default: same as --fit-dir)")
    args = ap.parse_args(argv)

    out_dir = args.out_dir if args.out_dir is not None else args.fit_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Discover which datasets have rates in fit-dir
        targets = []
        for ds in _ALL_DATASETS:
            for suffix in ("_sigma1.0_final_rates.npz", "_final_rates.npz"):
                if (args.fit_dir / f"{ds}{suffix}").exists():
                    targets.append(ds)
                    break
        if not targets:
            print(f"No rates found in {args.fit_dir} for any of {_ALL_DATASETS}")
            return 1
        print(f"--all mode: found rates for {targets} in {args.fit_dir}")
    else:
        targets = [args.dataset]

    for ds in targets:
        _run_one(ds, args.fit_dir, out_dir, args.threshold, args.num_threads)

    print(f"\nDONE. To load and query in Python:")
    print(f"  import numpy as np")
    print(f"  d = np.load('{out_dir}/<dataset>_per_family_FxN.npz')")
    print(f"  present = d['present']        # shape (F, N), P{{ξ_v>0|f}}")
    print(f"  copies = d['copies']          # shape (F, N), E[ξ_v|f]")
    print(f"  # binary presence at θ=0.5:")
    print(f"  is_present = present >= 0.5   # bool (F, N)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
