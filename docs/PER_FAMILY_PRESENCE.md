# Per-family per-node presence posteriors

For any saved fit (cold ML, multistart-polish ML, MAP σ=1, or the
Brownian-prior MAP), you can produce two F × N matrices that tell you,
for each gene family f and each tree node v:

| matrix | element | meaning |
|---|---|---|
| `present[f, v]` | `P{ξ_v ≥ 1 | data_f, fitted rates}` | posterior probability family *f* is present at node *v* |
| `copies[f, v]`  | `E[ξ_v | data_f, fitted rates]` | posterior expected copy number for *f* at *v* |

These are NOT written by the standard fit pipelines (the F × N arrays can
be hundreds of MB) — they're computed on demand by a small wrapper around
the native C kernel `recount_per_family_posteriors_batch`.

## Compute on demand

**Single dataset** (writes two files alongside the fit):
```sh
PYTHONPATH=. python3 validation/per_family_presence.py --dataset dpann80 \
    --fit-dir validation/outputs/map_sigma1
```

**Every dataset in a fit directory** (auto-discovers which `*_final_rates.npz`
or `*_sigma1.0_final_rates.npz` files are present):
```sh
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs/map_sigma1
```

Examples — one-line invocations for each commonly-used fit directory:
```sh
# Sub-critical + gain-capped Csurös-exact bounded fits:
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs/bounded_csuros

# Independent MAP σ=1:
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs/map_sigma1

# Pre-cap unbounded snapshot (already populated):
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs_no_dup_cap/map_sigma1

# Cold-start ML (Csurös-matching 1-start):
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs/csuros_match

# Multistart-polish ML:
PYTHONPATH=. python3 validation/per_family_presence.py --all \
    --fit-dir validation/outputs/sota_ml
```

**Options:**
- `--threshold T`  (default 0.5): cut-off used for the binary index CSV
- `--num-threads N`  (default 0 = all cores): thread count for the native backend
- `--out-dir DIR`  (default = `--fit-dir`): write outputs to a different place

## Output files (per dataset)

| file | shape | content |
|---|---|---|
| `<ds>_per_family_FxN.npz` | compressed | keys `present`, `copies` (both float64 F × N), plus `F`, `N`, `dataset`, `threshold` metadata |
| `<ds>_per_family_summary.csv` | F rows | `family_idx, n_nodes_present, deepest_node_idx, leaf_count_observed` — quick browsable index, one row per family |

The summary CSV uses `threshold` (default 0.5) to define "present"; the
full posterior is in the NPZ so you can re-threshold without recomputing.

## Compute time (M4 Max, native backend)

| dataset | F | N | wall (4 threads) | NPZ size (compressed) |
|---|---:|---:|---:|---:|
| dpann80 | 3,034 | 159 | 0.1 s | 1.7 MB |
| proteo75 | 5,179 | 149 | 0.1 s | 2.4 MB |
| eury114 | 7,335 | 227 | 0.3 s | 4.5 MB |
| ed194 | 8,855 | 387 | 0.6 s | 7.1 MB |
| arc269 | 90,243 | 537 | 3.6 s | 21 MB |

A full `--all` rerun across the five datasets takes **~5 s** on M4 Max.

## Query patterns

```python
import numpy as np

# Load arc269 per-family table
d = np.load('validation/outputs_no_dup_cap/map_sigma1/arc269_per_family_FxN.npz')
present = d['present']          # (90243, 537)  P{ξ_v ≥ 1 | f}
copies  = d['copies']           # (90243, 537)  E[ξ_v | f]

# Q1: which nodes does family 42 occupy with P >= 0.5?
np.where(present[42] >= 0.5)[0]

# Q2: which families are present at node 347 (DPANN ancestor inside arc269)?
fams_at_dpann = np.where(present[:, 347] >= 0.5)[0]

# Q3: how many "core" families (present at root with high confidence)?
from validation._shared import load_dataset
tree, _, _, _, _ = load_dataset('arc269')
root = tree.root
for thr in [0.5, 0.95, 0.99, 0.999]:
    print(f"P{{root}} >= {thr}: {(present[:, root] >= thr).sum():,}")

# Q4: for each family, the deepest ancestor where it's "present" (posterior >= 0.5)
binary = present >= 0.5
deepest = np.where(binary.any(axis=1), binary.argmax(axis=1), -1)
# deepest[f] = index of the deepest node v with P{ξ_v >= 1 | f} >= 0.5,
# or -1 if no node passes the threshold.

# Q5: expected copy histograms at the LACA root
import matplotlib.pyplot as plt
plt.hist(copies[:, root], bins=50, log=True)
plt.xlabel("E[copies at root | family]"); plt.ylabel("# families (log scale)")
plt.show()
```

## Where it lives in the codebase

| component | path |
|---|---|
| CLI / dispatcher | [validation/per_family_presence.py](../validation/per_family_presence.py) |
| Python wrapper for native | `recount.native_backend.per_family_posteriors_native` ([recount/native_backend.py](../recount/native_backend.py)) |
| Native C implementation | `recount_per_family_posteriors_batch` in [native/src/recount_batch.c](../native/src/recount_batch.c) (declared in [native/include/recount_native.h](../native/include/recount_native.h)) |

## Wiring into the fit pipeline (optional)

The per-family computation is intentionally **NOT** wired into
`validation/ml.py` / `validation/map.py` / `validation/mixture_ml.py`
automatically — these arrays add ~5 s wall + tens of MB on every fit
and not everyone needs them. To re-generate after every fit, append the
`--all` command above to your launcher script (e.g., in
[tools/pool_fits.sh](../tools/pool_fits.sh) or
[tools/chain_after_pool.sh](../tools/chain_after_pool.sh)).
