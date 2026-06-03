# Resume context

> **Historical reference.** This file is the post-hoc resume note for
> the bounded-fit campaign and the gradient C-port. The hourly
> auto-checkup cron that used to drive it has expired, and the
> "in-flight" tables it used to carry are no longer meaningful. What
> remains here — the campaign goal, the gradient C-port summary, the
> code-surface reference, and the threading note — is still valid and
> useful for orienting a new session.

## Starting a new session

```bash
git clone -b main git@github.com:ssolo/recount.git
cd recount

# Build the native backend (first import auto-builds; or force):
cd native && make && cd ..

# Sanity-check the all-C path — reproduces Csurös' published dpann80 LL
# to ≤ 0.06 nats:
PYTHONPATH=. python3 -c "
from validation._shared import load_dataset, reconstruct
tree, profiles, csuros_rates, mc, _ = load_dataset('dpann80')
print('Csurös baseline LL on dpann80:',
      reconstruct(tree, csuros_rates, profiles, mc)['ll'])
# Expect: -94146.94... (matches Java to ≤ 0.06 nats)
"
```

The campaign queue file ([validation/QUEUE_INSTRUCTIONS.md](validation/QUEUE_INSTRUCTIONS.md))
is the canonical recipe for the 4-mode × 6-dataset bounded sweep.
The pool scheduler ([tools/pool_fits.sh](tools/pool_fits.sh), docs
in [tools/README.md](tools/README.md)) handles running multiple fits
without oversubscribing the CPU.

## Campaign goal

Reproduce (and where possible improve on) Csurös 2026's published
per-subset ML log-likelihoods on the four arc269 subsets (D80, P75,
E114, ED194) and quote a LACA reconstruction on arc269, using a
bit-faithful port of the Java optimiser pipeline from
[miklosc/Count](https://github.com/miklosc/Count) (Numerical
Recipes `dfpmin` BFGS + Armijo lnsrch) under:

- the duplication constraint the Java actually enforces
  (`dup ≤ MAX_PROB_NOT1` via logit), and
- the gain regime the publication runs actually use (no upper cap;
  `MAX_GAIN_RATE = 33` is declared in the source but the bundled
  rates exceed it by 12+ orders of magnitude — see
  [GAIN_CONSTRAINT.md](GAIN_CONSTRAINT.md)).

Williams2017 (a separate paper, 60 leaves, Ωmin = 4) is also fit and
kept as a self-contained validation preamble; its numbers are not
mixed into the arc269-subset tables.

The campaign result tables live in [README.md](README.md);
the canonical recipe + queue in [validation/QUEUE_INSTRUCTIONS.md](validation/QUEUE_INSTRUCTIONS.md);
the historical bounded vs unbounded comparison in
[NO_DUPLICATION_CONSTRAINT.md](NO_DUPLICATION_CONSTRAINT.md);
the boundary-identifiability proof and the σ-sweep on arc269 in
[MODEL_COMPLEXITY.md](MODEL_COMPLEXITY.md);
the validation accounting in [VALIDATION.md](VALIDATION.md);
the LACA quote with bootstrap CI summary in [PROGRESS.md](PROGRESS.md).

## L(0)-gradient C-port (commits)

The entire L(0)-corrected gradient pipeline is native C
(`native/src/recount_unobserved_grad.c` + `recount_rates.c`).
Per-call timing on dpann80 (mc=4): `gradient_native` 47.6 ms (9.4×
faster than the pre-port path); `compute_L0_gradient_analytical`
4.6 ms (1000× faster than the original Python+torch reference).

| step | commit  | result |
|------|---------|--------|
| 1. pairing likelihoods           | `16008dc` | 44× faster, bit-identical to numpy |
| 2. outside (B/J/Bns/Jns)         | `e9dde4e` | 236–299× faster; Pólya direct-sum dodges `lgamma` cancellation under `-ffast-math` |
| 3. posteriors + transitions + BD tails | `d419bce` | machine-precision, sub-ms |
| 4. log-survival gradient         | `aa3b027` | bit-identical to numpy |
| 5. hybrid orchestrator (C steps 1–4 + torch chain rule) | `117ac4d` | intermediate landing |
| 6. analytical Jacobians + chain rule in C | `000a73f` / `fe6e5a4` / `665d59a` / `4099f81` / `c2659f1` | replaces torch autograd; matches Python prototype bit-for-bit; ~1e-5 rel vs torch at Yule-near corner nodes |
| 7. retire torch from `_gradient_raw_native` (second torch holdout in `recount/ml.py`) | `7fd3425` | full C path end-to-end |

Per-phase max |Δ| vs the reference implementation is reported in the
README's "C-port orthogonality" notes; the math is identical, only
the wall time changed.

## Code-surface reference

| File | What it owns |
|---|---|
| `tools/pool_fits.sh`                          | 4-slot pool scheduler for parallel fits |
| `tools/launch_safe.sh`                        | solo idempotent runner |
| `tools/run_waves.sh`                          | sequential-waves alternative |
| `tools/README.md`                             | usage + queue file format |
| `validation/_shared.py`                       | optimiser plumbing: `rates_to_x` / `x_to_rates`, `make_objgrad_ml/_map/_map_brownian`, `fit_bfgs_cycles`, the bounded-dup logit transform |
| `validation/ml.py`                            | ML CLI; `--optimizer native_bfgs` is the default |
| `validation/map.py`                           | MAP CLI; supports `--prior {independent, brownian}` |
| `validation/brownian_extend_ml.py`            | warm-start ML fine-tune on top of Brownian-MAP rates |
| `validation/mixture_ml.py`                    | K-category LogisticShift mixture CLI; `--bounded` uses the same logit transform as `ml.py` / `map.py` |
| `validation/reproduce.py`                     | load Csurös' published rates → bit-perfect per-node check |
| `validation/subset_only_path_plot.py`         | per-subset path plots (no arc269 dependency) |
| `validation/subclade_ancestor_comparison.py`  | arc269-overlay per-subset plots + bar chart |
| `validation/per_family_presence.py`           | per-family per-node `P{ξ_v ≥ 1}` and `E[ξ_v]` tables, on demand |
| `validation/williams_preamble_plot.py`        | Williams2017 preamble plot |
| `native/src/recount_bfgs.c`                   | Native C BFGS (bit-faithful port of Java `dfpmin`) |
| `native/src/recount_unobserved_grad.c`        | Native L(0)-gradient pipeline (steps 1–6 above) |
| `native/src/recount_rates.c`                  | `rate_to_pq` + analytical Jacobians for the chain rule |
| `native/src/recount_brownian_prior.c`         | tree-Brownian autocorrelated log-rate prior + gradient |
| `recount/native_backend.py`                   | ctypes bindings |
| `recount/ml.py`                               | `_gradient_raw_native` (full C); `random_initial_rates` |
| `recount/chain_rule_unobs.py`                 | Python prototype of the chain rule (kept as a reference for the C port) |
| `docs/references/`                            | Csurös' main paper PDF, SI PDF, arXiv preprint |
| `docs/csuros_data/`                           | raw arc269 + reconc data + Count Java jar |

## Threading

Set `RECOUNT_NUM_THREADS=N` or pass `--num-threads N`. On a 16-core
M4 Max with the pool, 4 slots × 4 threads = no oversubscription.
Default 0 = all cores (right for solo runs). On Linux the libdispatch
parallelism is replaced by OpenMP (`libgomp`); numerics are identical.

## What's NOT in scope

- Bounded MAP σ=1 15-start global on arc269. The unconstrained
  15-start global at `validation/outputs_no_dup_cap/profile_likelihood_arc269/`
  (cp = 5,426 / fm = 3,070 / cpf = 1.77, LL = −1,096,512, max_dup = 28
  — all firmly interior) remains the recommended LACA quote; a
  bounded re-run would only tighten the bootstrap CI slightly.
- Per-node unit test against Csurös' Java `getLogGainRate` confirming
  `gain[v] == κ` in the Pólya case (currently inferred from the LL
  bit-match).
- Native BFGS testing on Linux/x86 (the dylib has only been exercised
  on Darwin/arm64; the OpenMP path compiles but has not been
  benchmarked).
