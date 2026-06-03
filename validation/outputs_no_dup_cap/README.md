# `validation/outputs/` — fit results for all 5 datasets × 4 modes

Per-dataset, per-fit-mode artifacts produced by the consolidated driver
scripts in [`validation/`](../). The five datasets are:

| label | clade | F (families) | leaves | nodes | Ωmin |
|---|---|---:|---:|---:|---:|
| `arc269` | full Archaea (LACA) | 90,243 | 269 | 537 | 1 |
| `dpann80` | DPANN superphylum (Nanobdellati) | 3,034 | 80 | 159 | 4 |
| `proteo75` | Proteoarchaea (TACK) | 5,179 | 75 | 149 | 4 |
| `eury114` | Methanobacteriati / basal Eury. | 7,335 | 114 | 227 | 4 |
| `ed194` | Euryarchaeota (LECA-ish) | 8,855 | 194 | 387 | 4 |

The four fit modes:

| dir | what it is | script | clip | prior |
|---|---|---|---|---|
| `reproduction/` | Csurös' published rates re-loaded and re-evaluated by our native engine (subsets only — arc269 uses our own basin since Csurös never fit it). | [`validation/reproduce.py`](../reproduce.py) | — | — |
| `simple_ml/` | Single warm-restart BFGS from default uniform init. Cheap (≈ 25 min on M4 Max for all 5). | [`validation/ml.py`](../ml.py) `--num-starts 1 --polish-sigmas ""` | log-rate ∈ [−50, 50] | — |
| `sota_ml/` | Multistart (3 starts: default + 2 perturbed) + lognormal polish at σ ∈ {0.05, 0.10}. ≈ 80 min for all 5. | [`validation/ml.py`](../ml.py) `--num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2` | log-rate ∈ [−50, 50] | — |
| `map_sigma1/` | Weakly-informative MAP with log-Normal prior log(rate) ~ N(μ, 1) (centred at biological typicals: gain=0.1, dup=0.5, length=1). | [`validation/map.py`](../map.py) `--sigma 1.0` | log-rate ∈ [−50, 50] | log-Normal(σ=1) on every log-rate |

For each `{mode}/{dataset}` the standard artifacts are:

- `{dataset}_{tag}.countxml.gz` — Csurös' XML format, loadable in the Count Java GUI for visual inspection.
- `{dataset}_{tag}.branches.csv` — per-node summary: gain/dup/length rates, posterior copy counts (observed and corrected), posterior family probabilities.
- `{dataset}_summary.json` — headline numbers: LL, L(0), root copies/families, cp/fam.
- `{dataset}_final_rates.npz` — raw rates (gain, loss, dup, length arrays).
- `{dataset}_trajectory.json` — (ml.py / sota_ml only) per-cycle BFGS state (obj, |g|, iters, wall).

## Why four modes (and which to quote for LACA)

The arc269 LACA reconstruction is the headline result; it depends on
which fitting mode you trust:

| mode | arc269 LL | root copies | root families | cp/fam | interior optimum? |
|---|---:|---:|---:|---:|:---:|
| simple_ml | −1,129,653 | 8,664 | 3,533 | 2.45 | **No** — boundary basin (one node at dup ≈ 7.6 × 10⁹) |
| sota_ml | −1,109,266 | 8,963 | 3,608 | 2.48 | **No** — same boundary basin, optimizer climbed further (5.6 × 10⁹) |
| map_sigma1 (cold start) | −1,117,480 | 5,906 | 3,108 | 1.90 | Yes (local) — max(dup) = 43 |
| profile_likelihood_arc269 (MAP σ=1, 15-start global) | **−1,096,512** | **5,426** | **3,070** | **1.77** | **Yes (global)** — max(dup) = 28 |

The simple_ml and sota_ml rows are not biologically interpretable: both
are sequences along the Pólya-κ → ∞ asymptote at a single internal node.
See [`docs/boundary_identifiability.pdf`](../../docs/boundary_identifiability.pdf)
for the full proof and quantitative analysis.

**The recommended LACA quote is the 15-start global MAP row (from
`profile_likelihood_arc269/`):**
**5,426 root copies / 3,070 root families / cp/fam ≈ 1.77.**
The cold-start MAP at 5,906/3,108/1.90 in `map_sigma1/` is a *local*
optimum 21 k nat above the global; multistart is essential.

## Reproduction (subsets)

For all 4 subsets (D80, P75, E114, ED194), our `reproduction/` outputs
match Csurös' Java to ≤ 5 × 10⁻¹¹ at every per-node value when loaded
with our native pipeline. See [`VALIDATION_FOCAL.md`](../../VALIDATION_FOCAL.md).

## How to re-generate

```sh
# Reproduction (≈ 30 s total):
PYTHONPATH=. python3 validation/reproduce.py --dataset all \
    --out-dir validation/outputs/reproduction

# Cold-start ML (≈ 25 min):
PYTHONPATH=. python3 validation/ml.py --dataset all \
    --num-starts 1 --polish-sigmas "" --cycle-iters 100 --max-cycles 15 \
    --out-dir validation/outputs/simple_ml

# Multistart-polish ML (≈ 80 min):
PYTHONPATH=. python3 validation/ml.py --dataset all \
    --num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2 \
    --cycle-iters 100 --max-cycles 12 \
    --out-dir validation/outputs/sota_ml

# MAP σ=1 (≈ 25 min):
PYTHONPATH=. python3 validation/map.py --dataset all --sigma 1.0 \
    --num-starts 5 --cycle-iters 100 --max-cycles 15 \
    --out-dir validation/outputs/map_sigma1
```

