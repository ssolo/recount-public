# Reproducing the full bounded-fit campaign

Single canonical BFGS recipe — matches the latest arc269 setup — applied
to every dataset (`dpann80`, `proteo75`, `eury114`, `ed194`, `williams`,
`coleman`, `arc269`) across four fit modes. arc269 modes 1, 3, 4 are
already on disk with canonical settings; the queue will auto-skip those
and only run arc269's missing mode 2 (Brownian-extend), saving ~3 h.

```
native_bfgs   --cycle-iters 100   --max-cycles 12   --seed 2025
sub-critical Yule cap   |log_rate| ≤ 33   prior centred at (γ=0.1, λ=0.5, t=1)
```

Each mode lands in a canonical subdir of `validation/outputs/` so the
plot script (`subclade_ancestor_comparison.py`) picks it up
automatically.

| mode | script | flags | output dir |
|---|---|---|---|
| 1 cold ML BOUNDED        | `validation/ml.py`                  | `--num-starts 1 --polish-sigmas ""`                                                                                                | `bounded_csuros/`           |
| 2 Brownian-extend ML     | `validation/brownian_extend_ml.py`  | `--max-cycles 5`  (warm-starts from mode 4's Brownian rates, then drops the prior for a short pure-ML fine-tune)                  | `brownian_extend_<ds>/`     |
| 3 MAP σ=1 cold           | `validation/map.py`                 | `--sigma 1.0 --num-starts 1`                                                                                                        | `map_sigma1/`               |
| 4 MAP Brownian σ=1       | `validation/map.py`                 | `--prior brownian --sigma-brownian-{gain,dup,length} 1.0 --num-starts 1`                                                            | `brownian_<ds>/`            |

Modes 1, 3, 4 use `--cycle-iters 100 --max-cycles 12 --seed 2025
--optimizer native_bfgs`. Mode 2 uses `--cycle-iters 100 --max-cycles 5
--seed 2025` (short by design — Brownian-warm start usually converges
in 2-4 cycles). The default queue order is **1 → 3 → 4 → 2** so each
dataset's mode 4 lands before mode 2 needs it.

The 15-start global MAP (`--num-starts 15`, writes to
`profile_likelihood_<ds>/`) is **not** in this canonical queue — too slow
for routine sweeps. Run it on demand for a single dataset:

```sh
PYTHONPATH=. python3 validation/map.py --dataset <ds> --sigma 1.0 \
    --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2025 \
    --optimizer native_bfgs --num-threads 16 \
    --out-dir validation/outputs/profile_likelihood_<ds>
```

## One-liner — launch all 6 datasets × all 4 modes

```sh
# ~5 h serial on M4 Max @ 16 threads (~3.5 h dominated by ed194 + coleman).
# Idempotent: skips any (dataset, mode) whose summary.json was already produced
# with cycle_iters=100, max_cycles=12.
NUM_THREADS=16 bash validation/queue_full_subset_run.sh
```

Background it and watch the log:

```sh
nohup NUM_THREADS=16 bash validation/queue_full_subset_run.sh > /tmp/full_queue.out 2>&1 &
tail -f validation/logs/queue_full_subset_*.log
```

## Common variants

**Just one dataset**:

```sh
NUM_THREADS=16 bash validation/queue_full_subset_run.sh coleman
```

**Just selected modes** (e.g. only cold ML + MAP cold + Brownian):

```sh
MODES="1 3 4" bash validation/queue_full_subset_run.sh
```

**Several datasets** (positional args, smallest-first order is fine):

```sh
NUM_THREADS=16 bash validation/queue_full_subset_run.sh williams dpann80 proteo75
```

## Expected per-run wall on M4 Max @ 16 threads (post-2026-05-19 fast lib)

| dataset | F | leaves | cold ML (1×12 cyc) | Brown-extend ML (1×≤5 cyc) | MAP cold | MAP Brownian |
|---|---:|---:|---:|---:|---:|---:|
| williams | 5,378  | 60  | ~2 min  | ~30 s   | ~2 min  | ~2 min  |
| dpann80  | 3,034  | 80  | ~1.5 min| ~30 s   | ~1.5 min| ~1.5 min|
| proteo75 | 5,179  | 75  | ~2 min  | ~45 s   | ~2 min  | ~2 min  |
| eury114  | 7,335  | 114 | ~4 min  | ~1.5 min| ~4 min  | ~4 min  |
| ed194    | 8,855  | 194 | ~10 min | ~3 min  | ~10 min | ~10 min |
| coleman  | 11,272 | 265 | ~17 min | ~5 min  | ~17 min | ~17 min |
| arc269   | 90,243 | 269 | ~30 min | ~12 min | ~30 min | ~30 min |

For a **fresh** run including arc269 from scratch: ~3.5-4 h at 16 threads
(arc269 dominates at ~100 min for all 4 modes).

For the **current state** (arc269 modes 1, 3, 4 already on disk with
canonical settings; only mode 2 needs to land): subsets+williams+coleman
~ 2 h + arc269 mode 2 ~ 12 min = **~2 h 15 min** at 16 threads.

Per-mode totals across all 7 datasets (16 threads): cold ML ≈ 65 min,
Brown-extend ≈ 22 min, MAP cold ≈ 65 min, MAP Brownian ≈ 65 min.
**Fresh grand total ≈ 3.5-4 h. With arc269 1/3/4 already done ≈ 2 h.**

Smaller datasets (williams / dpann80 / proteo75) don't fully saturate 16
threads; their per-cycle wall is dominated by setup overhead. Going
8→16 threads gives ~1.4-1.6× speedup on ed194/coleman, near-flat on the small ones.

## After the queue finishes

The queue script auto-runs:

```sh
# Per-family per-node posterior tables for every populated fit dir:
PYTHONPATH=. python3 validation/per_family_presence.py --all --fit-dir <subdir>

# Plots:
PYTHONPATH=. python3 validation/subclade_ancestor_comparison.py --variant bounded
```

To rerun the plots manually after partial completion:

```sh
PYTHONPATH=. python3 validation/subclade_ancestor_comparison.py --variant bounded
```

## What each fit mode tells you

- **cold ML (mode 1)** — what the BFGS basin looks like with NO prior and a single random init. Tends to over-estimate gain at boundary nodes (super-critical Yule).
- **Brownian-extend ML (mode 2)** — uses the Brownian-MAP (mode 4) rates as initial values, then drops the prior and runs pure ML for ≤ 5 cycles. The Brownian prior anchored the basin (cpf ≈ 1.1-1.3); releasing it lets the rates relax toward the ML peak without diverging away from the basin. Expected: small LL improvement (~100 nats on dpann80), root cp/fm stays close to the Brownian value but moves slightly toward the data fit.
- **MAP σ=1 cold (mode 3)** — weakly-informative prior log(rate) ~ N(μ, 1) regularises the boundary blow-ups but still finds local optima.
- **MAP Brownian (mode 4)** — per-edge Gaussian on log-rate increments (TKP autocorrelated rates); smooths adjacent-branch rate variation along the tree. Subset Brownian fits land at cpf 1.1-1.3 (much tighter than cold MAP's ~1.7) — clearest single-mode signal. **Prerequisite for mode 2.**

For the 15-start global MAP (deeper basin search at ~3-5 h per dataset), launch it on demand per the snippet above the table — it lives outside the canonical queue.

## Adding a new dataset

Edit [validation/_shared.py](_shared.py) `DATASETS` dict — see the
`coleman` entry for the bare-Newick + CSV-table format, or any subset
entry for the Csurös-countxml format. Then re-run the queue with the
new label as positional arg.
