"""ML on arc269 starting from the subset-rate seed (validation/outputs/arc269_seed_from_subsets_v2.npz).

Compares the resulting basin against:
  - cold ML from random init (already running, separately)
  - MAP σ=1 cold basin (LL=-1,128,902)
  - MAP σ=1 15-start global basin (LL=-1,096,512)

Outputs:
  validation/outputs/arc269_seeded_ml/arc269_ml.branches.csv
  validation/outputs/arc269_seeded_ml/arc269_summary.json
  validation/outputs/arc269_seeded_ml/arc269_final_rates.npz
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np

# Resolve paths relative to the repo root regardless of working dir;
# imports work via `PYTHONPATH=. python3 validation/arc269_ml_from_subset_seed.py`.
from validation._shared import (
    load_dataset, make_objgrad_ml, fit_bfgs_cycles, reconstruct, write_outputs,
)
from recount.gld import GLDRates

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "validation/outputs/arc269_seed_from_subsets_v2.npz"
OUT_DIR = ROOT / "validation/outputs/arc269_seeded_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading arc269...")
tree, profiles, _, mc, biology = load_dataset("arc269")
N = tree.num_nodes
print(f"  N={N}, F={profiles.shape[0]}, mc={mc}")

print(f"Loading seed rates from {SEED_PATH}")
d = np.load(SEED_PATH)
init = GLDRates(
    tree=tree,
    gain=d["gain"].copy(), loss=d["loss"].copy(),
    dup=d["dup"].copy(),  length=d["length"].copy(),
)
print(f"  seed LL (saved): {float(d['ll_at_seed']):,.0f}")

# Build objgrad factory and run BFGS cycling
print("Building ML objective + native BFGS cycling (subcritical, 8 cycles × 100 iters)")
objgrad = make_objgrad_ml(tree, init, profiles, min_copies=mc, subcritical=True)
t0 = time.time()
best_rates, best_obj, traj = fit_bfgs_cycles(
    objgrad_factory=lambda r: make_objgrad_ml(tree, r, profiles, min_copies=mc, subcritical=True),
    tree=tree, init_rates=init,
    label="arc269-seeded",
    cycle_iters=100, max_cycles=8,
    optimizer="native_bfgs",
    gtol=1e-7,
    subcritical=True,
    early_stop_iters_zero=2,
)
wall = time.time() - t0
print(f"\n  total wall: {wall:.0f}s = {wall/60:.1f} min")
print(f"  best obj (-LL): {best_obj:,.4f}    LL = {-best_obj:,.4f}")
print(f"  cold MAP basin: LL = -1,128,902")
print(f"  15-start global: LL = -1,096,512")
print(f"  gap vs 15-start: {(-best_obj) - (-1096512):+,.0f}")

# Reconstruct + write outputs — reconstruct() returns root_copies_corr, root_families_corr
# (not cp/fm shorthand)
res = reconstruct(tree, best_rates, profiles, mc)
print(f"\n  root copies: {res['root_copies_corr']:.0f}   "
      f"root families: {res['root_families_corr']:.0f}   "
      f"cp/fm: {res['copies_per_family']:.3f}")

# Save
out_prefix = OUT_DIR / "arc269_ml"
write_outputs("arc269", tree, best_rates, profiles, mc, out_prefix)
np.savez(OUT_DIR / "arc269_final_rates.npz",
         gain=best_rates.gain, loss=best_rates.loss,
         dup=best_rates.dup,   length=best_rates.length)
summary = {
    "label": "arc269",
    "init": "subset_seed_v2",
    "ll_at_init": float(d["ll_at_seed"]),
    "ll_final": -float(best_obj),
    "wall_s": wall,
    "trajectory": traj,
    "root_copies_corr": res["root_copies_corr"],
    "root_families_corr": res["root_families_corr"],
    "copies_per_family": res["copies_per_family"],
    "fit_settings": {"max_cycles": 8, "cycle_iters": 100, "subcritical": True,
                     "optimizer": "native_bfgs"},
}
with open(OUT_DIR / "arc269_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nWrote outputs to {OUT_DIR}/")
