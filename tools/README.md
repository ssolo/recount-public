# tools/ — orchestrators for parallel recount fits

Schedulers + helpers for running many `validation/{ml,map,mixture_ml}.py`
fits without oversubscribing the CPU. All bash + small Python helpers;
no extra dependencies beyond what `recount` already needs.

| Script | Pattern | When to use |
|--------|---------|-------------|
| [`pool_fits.sh`](pool_fits.sh) | Keep N fits active continuously | **Default for a queue.** CPU stays at 100%; a slot is refilled the moment one finishes. |
| [`launch_safe.sh`](launch_safe.sh) | Single fit, full-machine threads | **Default for one big job** (e.g. arc269 cold ML at 16 threads). Refuses to launch a duplicate or overwrite a fresh summary. |
| [`run_waves.sh`](run_waves.sh) | Sequential waves of N fits | Easier to reason about for short benchmark sweeps; idles cores while waiting for the slowest fit in each wave. |
| [`hourly_checkup.sh`](hourly_checkup.sh) | Background daemon | Re-runs README table refresh + plot regen + git push every hour. Survives terminal close. |
| [`refresh_readme_tables.py`](refresh_readme_tables.py) | One-shot | Rewrites the README result tables from fresh saved fits. Called by `hourly_checkup.sh`. |
| [`audit_and_cleanup.py`](audit_and_cleanup.py) | One-shot | Stale-data audit + repo-wide cleanup + README cross-check + push. |

## One-time setup (fresh machine)

```bash
# Clone the work
git clone git@github.com:ssolo/recount.git
cd recount

# Build the native backend (Apple Accelerate + libdispatch on macOS,
# OpenMP on Linux; the Makefile auto-detects):
cd native && make && cd ..

# Sanity-check the all-C path — should reproduce Csurös' published
# dpann80 LL to ≤ 0.06 nats:
PYTHONPATH=. python3 -c "
from validation._shared import load_dataset, reconstruct
tree, profiles, csuros_rates, mc, _ = load_dataset('dpann80')
print('dpann80 baseline LL:', reconstruct(tree, csuros_rates, profiles, mc)['ll'])
# Expected: -94146.94...
"
```

## Pull fresh state

```bash
git fetch origin && git pull --ff-only origin main
```

If you have local in-flight fits writing into `validation/outputs/*`,
the pull will rebase against the remote's hourly commits. Conflicts are
rare (the hourly cron only touches `README.md` + `PROGRESS.md` +
`RESUME_CONTEXT.md` + plots); resolve by keeping your local fit's
`*_summary.json` / `*_final_rates.npz` / `*_trajectory.json`.

## Mode 1 — pool of fits (`pool_fits.sh`)

The default for any multi-fit workload. Keeps `SLOTS` fits active in
parallel at `THREADS` threads each. Slots are refilled instantly.

```bash
# Default: 4 slots × 4 threads on a 16-core Apple Silicon.
nohup tools/pool_fits.sh 4 4 tools/queues/all_six_sota_map.txt myrun \
  > /dev/null 2>&1 &
disown

# Watch:
tail -f validation/logs/pool_myrun_*.log
```

Master log goes to `validation/logs/pool_<tag>_<TS>.log`; per-fit logs
to `validation/logs/<ds>_<kind>_<tag>_<TS>.log`. Fit outputs land in
the usual places under `validation/outputs/`.

### Pre-launch idempotency guards

`pool_fits.sh` refuses to launch a fit if either:

1. The same `(dataset, out-dir)` is already running.
2. A fresh summary JSON already exists on disk (mtime > the post-port
   cutoff at 2026-05-18 16:30 JST).

Override with `POOL_FORCE=1`. Both guards prevent the duplicate-launch
chaos that happens when two pool masters run on the same queue.

### Sizing for your machine

| Cores | Recommended SLOTS × THREADS |
|-------|-----------------------------|
| 8     | 2 × 4 or 4 × 2 |
| 16    | **4 × 4** |
| 24    | 4 × 6 or 6 × 4 |
| 32    | 4 × 8 or 8 × 4 |

`THREADS` becomes each fit's `RECOUNT_NUM_THREADS`. Use SLOTS × THREADS
= physical cores; ignore SMT/hyperthreads (Apple Silicon doesn't have
them and recount is CPU-bound, so logical-only threads hurt).

### Queue file format

One `<dataset> <kind>` per line; blank lines and `#` comments ok:

```
dpann80 csuros_match
proteo75 sota_ml
arc269 map_sigma1_15start
```

`<dataset>` is anything from `validation/_shared.py:DATASET_NAMES`
(arc269, dpann80, proteo75, eury114, ed194, williams, plus the new
coleman dataset).

`<kind>` table:

| kind | wraps |
|------|-------|
| `csuros_match` | `validation/ml.py --num-starts 1 --max-cycles 40` (cold ML targeting Csurös per-subset LL) |
| `sota_ml` | `validation/ml.py --num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2` (multistart-polish ML) |
| `sota_ml_1start` | same as `sota_ml` but `--num-starts 1` (for big datasets where 3 starts is overkill) |
| `sota_ml_seed2024` | same as `sota_ml` but seed=2024 (escape route for seed-dependent bad basins) |
| `map_sigma1` | `validation/map.py --sigma 1.0 --num-starts 3` (MAP σ=1, 3-start) |
| `map_sigma1_15start` | `validation/map.py --sigma 1.0 --num-starts 15` → `profile_likelihood_arc269/` |
| `mixture_K2` | `validation/mixture_ml.py --bounded --K 2 --num-starts 3` (K=2 LogisticShift mixture) |

All recipes use `--optimizer native_bfgs` and seed=2025 by default. To
add or change a recipe, edit the `case "$kind"` block in
[`pool_fits.sh`](pool_fits.sh) (and the matching block in
[`launch_safe.sh`](launch_safe.sh) for solo mode).

## Mode 2 — solo full-machine fit (`launch_safe.sh`)

When the only remaining work is one big fit (typically arc269 — F=90k)
and you want to drain it in hours rather than days, run a single fit
at all 16 threads instead of 4 threads in a pool slot. Per-cycle wall
time scales close to linearly with threads, so 16 threads is ~4× faster
than 4 threads.

```bash
# Cold ML — the headline LL number for arc269
tools/launch_safe.sh arc269 csuros_match

# MAP σ=1 (3-start)
tools/launch_safe.sh arc269 map_sigma1

# Profile likelihood (15-start MAP σ=1) — tightens bootstrap CI
tools/launch_safe.sh arc269 map_sigma1_15start

# Multistart-polish ML, 1-start
tools/launch_safe.sh arc269 sota_ml_1start

# K=2 LogisticShift mixture, bounded
tools/launch_safe.sh dpann80 mixture_K2
```

Default thread count is 16; override with a third positional arg:

```bash
tools/launch_safe.sh arc269 csuros_match 8
```

`launch_safe.sh` runs the same three pre-flight checks as `pool_fits.sh`
plus an oversubscription warning if other Python fits are already
active. Override any guard with `LAUNCH_FORCE=1`.

**Run solo fits serially**, not in parallel — each saturates all 16
cores; two in parallel oversubscribe and slow both. The script warns
but doesn't block (clusters legitimately do this sometimes).

## Mode 3 — hybrid (2 × 8)

For medium datasets where you want some parallelism but more cores per
fit:

```bash
nohup tools/pool_fits.sh 2 8 tools/queues/<your_queue>.txt mytag \
  > /dev/null 2>&1 &
disown
```

Keeps 2 fits active at 8 threads each — twice the per-fit speed of
4 × 4 at the cost of half the parallelism. Right for eury114/ed194-scale
runs.

## Mode 4 — sequential waves (`run_waves.sh`)

```bash
tools/run_waves.sh 4 tools/queues/all_six_sota_map.waves allC
```

Waves file format — lines = waves; fits within a wave separated by `,`:

```
# Wave 1 — fast 4
williams sota_ml, williams map_sigma1, dpann80 sota_ml, dpann80 map_sigma1
# Wave 2 — medium 4
proteo75 sota_ml, proteo75 map_sigma1, eury114 sota_ml, eury114 map_sigma1
# Wave 3 — heavy 4
ed194 sota_ml, ed194 map_sigma1, arc269 sota_ml, arc269 map_sigma1
```

Idles cores while waiting for the slowest fit in each wave — usually
better to use `pool_fits.sh` instead.

## Hourly auto-checkup

```bash
nohup tools/hourly_checkup.sh > validation/logs/hourly_checkup.log 2>&1 &
disown
```

Survives terminal close. Pushes to `origin/main` every hour after
refreshing README tables / plots from any fit that landed since the
last tick. Stops with `pkill -f hourly_checkup.sh`.

## What to expect — arc269 fit timings on M4 Max

Realistic per-cycle wall times extrapolated from in-flight measurements
(4× linear scaling to 16 threads):

| Fit | Per-cycle wall | Total wall (16 threads solo) |
|-----|----------------|------------------------------|
| arc269 csuros_match (1×40) | ~6 min | **~1 h** |
| arc269 map_sigma1 (3×25) | ~10 s avg | **~20–30 min** |
| arc269 map_sigma1_15start (15×25) | ~10 s avg | **~1.5–2 h** |
| arc269 sota_ml_1start (1×25 + polish) | ~6 min initial, ~3 min once \|g\| settles | **~3–4 h** |

**All four serially: ~6–8 h total.** Compare to ~22–30 h on a 4×4
pool slot. MAP runs converge in 1–2 iters per cycle after the first
few cycles, so they're much faster than ML which keeps doing 200
iters/cycle until `|g|` drops.

The constrained regime is **not slower** than unconstrained — the
logit transform on `dup` is one sigmoid + one Jacobian per gradient
call (statistical noise vs the per-cycle BFGS cost). Cycle counts are
also similar (the constraint barely changes basin geometry for fits
where `dup` doesn't saturate, which is most of them).

## Output layout (where fits land)

```
validation/outputs/
├── csuros_match/                  # 1-start cold ML (Csurös methodology) — subsets
├── sota_ml/                       # 3-start polish ML (deeper basin search) — subsets
├── map_sigma1/                    # MAP σ=1 cold start — all datasets
├── mixture_K2/                    # K=2 LogisticShift mixture (bounded, the canonical variant)
├── reproduction/                  # Csurös' published rates re-applied (validation)
├── bootstrap_arc269_bounded/      # 20-iter non-parametric bootstrap on arc269 LACA
├── bounded_csuros/                # canonical bounded cold ML — all datasets (the LL table)
├── brownian_*/                    # tree-Brownian-prior MAP fits (mode 4 of the canonical campaign)
├── brownian_extend_*/             # Brownian-warm-start ML fine-tune (mode 2)
├── arc269_seeded_ml/              # arc269 ML warm-started from subset rates
└── arc269_shift_only_K3/          # K=3 LogisticShift shift-only mixture on arc269

# pre-cap (unconstrained) outputs preserved under ../outputs_no_dup_cap/:
#   simple_ml/, sota_ml/, map_sigma1/, profile_likelihood_arc269/,
#   bootstrap_arc269/, map_sigma_sweep/, mixture_K2/, mixture_K3/,
#   arc269_omin4/. The 15-start MAP global LACA quote
#   (cp=5,426 / fm=3,070 / cpf=1.77) lives in
#   ../outputs_no_dup_cap/profile_likelihood_arc269/.
```

Each fit writes 5 files:

- `<ds>_summary.json` — LL, fm/cp at root, max(dup)/max(gain), final rates
- `<ds>_final_rates.npz` — per-node `(gain, loss, dup, length)` arrays
- `<ds>_ml.branches.csv` (or `<ds>_map_sigma1.0.branches.csv`) — per-node posterior cp / fm
- `<ds>_ml.countxml.gz` — Csurös-format XML (loadable by their Java)
- `<ds>_trajectory.json` — per-cycle LL + |g| history (for the cycle plots)

## Monitoring + stopping

```bash
# Active Python fits + their CPU
ps aux | grep -E 'validation/(ml|map|mixture_ml)\.py' | grep -v grep

# Pool scheduler activity
tail -f validation/logs/pool_<tag>_*.log

# Specific fit progress (cycle-by-cycle LL + |g|)
tail -f validation/logs/<ds>_<kind>_*.log

# Per-fit best LL so far across the pool
for f in validation/logs/*_<tag>_*.log; do
  base=$(basename "$f" .log)
  echo -n "$base: "
  grep -E '^\s*\[' "$f" | tail -1 | awk -F'obj=' '{print $2}' | head -c 80
  echo
done

# Stop a pool scheduler (in-flight fits keep running)
pkill -f 'pool_fits.sh.*<tag>'

# Stop a specific fit
pkill -f 'validation/ml.py --dataset arc269'

# Nuclear — stop everything
pkill -f 'validation/(ml|map|mixture_ml)\.py'
pkill -f 'pool_fits.sh' ; pkill -f 'launch_safe.sh'
pkill -f 'hourly_checkup.sh'
```

## Bundled queue files

| File | Workload |
|------|----------|
| [`queues/all_six_sota_map.txt`](queues/all_six_sota_map.txt) | sota_ml + MAP σ=1 for all 6 datasets (12 fits) |
| [`queues/all_six_sota_map.waves`](queues/all_six_sota_map.waves) | Same workload, 3 sequential waves of 4 |
| [`queues/csuros_match_4subsets.txt`](queues/csuros_match_4subsets.txt) | 1-start cold ML on the 4 small subsets (4 fits) |
| [`queues/cold_ml_remaining.txt`](queues/cold_ml_remaining.txt) | Cold ML for datasets missing fresh fits + K=2 mixture for all 5 subsets |
| [`queues/post_current_pool.txt`](queues/post_current_pool.txt) | Continuation queue (used by chain_after_pool.sh) |
