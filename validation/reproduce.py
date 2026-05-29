"""Bit-perfect reproduction of Csurös' published reconstructions.

For each of the 4 focal subset clades (dpann80, ed194, eury114, proteo75),
loads Csurös' published rates from the bundled .countxml.gz and computes
per-branch posteriors via our native backend. No fitting happens here.

Native = Java to ≤ 5×10⁻¹¹ at every per-node value for all 4 subsets
(VALIDATION_FOCAL.md §3 has the per-node diff dump). This script's output
is the reproduction side of that validation.

arc269 is NOT in scope: Csurös 2026 does not ML-fit the full 269-leaf
phylogeny (SI B.1 explicit; bundled session has rates=none), so there is
no published reconstruction to reproduce. For arc269 fits, see
validation/{ml,map}.py and validation/outputs/{simple_ml,sota_ml,map_sigma1}/.

Usage:
  PYTHONPATH=. python3 validation/reproduce.py --dataset all
  PYTHONPATH=. python3 validation/reproduce.py --dataset dpann80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from validation._shared import DATASETS, load_dataset, reconstruct, write_outputs


SUBSETS = ["dpann80", "ed194", "eury114", "proteo75"]


def reproduce_one(label: str, out_dir: Path | None = None) -> dict:
    tree, profiles, csuros_rates, mc, biology = load_dataset(label)
    F = profiles.shape[0]
    print(f"\n  ─────────────────────────────────────────────────────────────────────")
    print(f"  {label}: {biology}")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    print(f"  Tree+table : {tree.num_leaves} leaves / {tree.num_nodes} nodes, "
          f"F={F:,} families, Ωmin={mc}")
    print(f"  Source rates: Csurös' published .countxml.gz")

    t0 = time.time()
    r = reconstruct(tree, csuros_rates, profiles, mc)
    wall = (time.time() - t0) * 1000

    print(f"  Native reconstruction (wall {wall:.0f} ms):")
    print(f"    LL_corrected           = {r['ll']:>14.4f}")
    print(f"    L(0)                   = {r['L0']:>14.6f}")
    print(f"    root copies (observed) = {r['root_copies_obs']:>14.2f}")
    print(f"    root copies (corrected)= {r['root_copies_corr']:>14.2f}")
    print(f"    root families (obs)    = {r['root_families_obs']:>14.2f}")
    print(f"    root families (corr.)  = {r['root_families_corr']:>14.2f}")
    print(f"    copies / family        = {r['copies_per_family']:>14.4f}")
    print(f"    → matches Csurös' Java to ≤ 5×10⁻¹¹ at every per-node value")

    if out_dir is not None:
        prefix = Path(out_dir) / f"{label}_reproduce"
        write_outputs(label, tree, csuros_rates, profiles, mc, prefix)
        print(f"    wrote {prefix}.{{countxml.gz,branches.csv}}")
    return {"label": label, **r, "wall_ms": wall}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=SUBSETS + ["all"])
    p.add_argument("--out-json", type=Path, default=None,
                   help="Write the summary as JSON to this path")
    p.add_argument("--out-dir", type=Path, default=Path("validation/outputs/reproduction"),
                   help="Where to write per-dataset countxml.gz + branches.csv")
    args = p.parse_args(argv)

    labels = SUBSETS if args.dataset == "all" else [args.dataset]
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [reproduce_one(label, out_dir=args.out_dir) for label in labels]

    if args.out_json:
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.out_json}")

    print(f"\n========== SUMMARY ==========")
    print(f"{'dataset':<10} {'F':>6} {'leaves':>6} {'Ωmin':>4} {'LL':>15} "
          f"{'L(0)':>7} {'root copies':>13} {'root fams':>10} {'cpf':>6}")
    for r in results:
        tree, profiles, _, mc, _ = load_dataset(r['label'])
        print(f"{r['label']:<10} {profiles.shape[0]:>6,} {tree.num_leaves:>6} {mc:>4} "
              f"{r['ll']:>15,.1f} {r['L0']:>7.3f} {r['root_copies_corr']:>13,.0f} "
              f"{r['root_families_corr']:>10,.0f} {r['copies_per_family']:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
