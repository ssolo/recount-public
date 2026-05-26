# Campaign record: the bounded-fit sweep + gradient C-port

> **Historical reference.** This file is the post-hoc record of the
> bounded-fit campaign and the L(0)-gradient C-port that produced
> the canonical fits cited in [README.md](README.md). It is no
> longer live: the in-flight pool, the hourly auto-checkup cron, and
> the ETAs that used to live here have all expired. The README's
> tables are the source of truth for current numbers; this file
> remains useful as a compact map of the campaign sequencing, the
> commit log of the C-port, and the constraint regime in force.

## What this campaign produced

A canonical 4-mode fit on each of six datasets, plus arc269.
Identical optimiser settings across modes and datasets (the canonical
recipe in [validation/QUEUE_INSTRUCTIONS.md](validation/QUEUE_INSTRUCTIONS.md)):

```
optimizer       native_bfgs              # bit-faithful port of Java dfpmin
cycle iters     100
max cycles      12        (5 for the Brownian-extend ML pass)
seed            2025
clip            |log_rate| ≤ 33          (Csurös' MAX_GAIN_RATE)
sub-critical    dup_v ≤ MAX_PROB_NOT1 = 1 − 2⁻³⁰   (logit transform)
```

Four fit modes per dataset, each landing in its canonical subdir of
`validation/outputs/`:

| mode | description | script | output dir |
|---|---|---|---|
| 1 | cold ML BOUNDED                       | `validation/ml.py`                  | `bounded_csuros/`       |
| 2 | Brownian-extend ML (warm-start ML on top of mode 4 rates) | `validation/brownian_extend_ml.py` | `brownian_extend_<ds>/` |
| 3 | MAP σ=1 cold                          | `validation/map.py`                 | `map_sigma1/`           |
| 4 | MAP Brownian σ=1                      | `validation/map.py --prior brownian` | `brownian_<ds>/`        |

The README's "Demonstration of reproduction" tables (root families,
copies-per-family, log-likelihood, boundary diagnostics) are populated
from these directories. Modes 2 and 4 — the recommended baselines —
match Csurös' published log-likelihood to within a few nat on every
focal subset.

## The arc269 LACA quote

The recommended LACA quote is the MAP σ=1 15-start global from
`validation/outputs_no_dup_cap/profile_likelihood_arc269/`
(produced in the pre-cap regime, never re-run in bounded form because
the cold-start bounded MAP already lands at a clean interior optimum;
see [MODEL_COMPLEXITY.md](MODEL_COMPLEXITY.md)):

| param | value | 95% CI (20-iter non-parametric bootstrap) |
|---|---:|---|
| LL                  | −1,096,512        | — |
| root copies         | 5,426             | 5,182 – 5,687 |
| root families       | 3,070             | 2,982 – 3,165 |
| copies per family   | 1.77              | 1.73 – 1.83 |
| max(dup) at MAP     | 28                | — |
| max(gain) at MAP    | 0.53              | — |

`max(dup)` and `max(gain)` are *exactly constant* across all 20
bootstrap iterations — the basin is stable under family resampling;
only the per-family aggregation contributing to cp / fm varies.

## Gradient C-port — all 6 phases landed

The L(0)-corrected gradient pipeline is end-to-end native C
(`native/src/recount_unobserved_grad.c`, `native/src/recount_rates.c`).
Per-call timing on dpann80 (mc=4, M4 Max single fit):

| call                              | pre-port  | after port |
|-----------------------------------|----------:|-----------:|
| `compute_L0_gradient_analytical`  | ~4,300 ms | 4.6 ms (1000×) |
| `gradient_native`                 | ~446 ms   | 47.6 ms (9.4×) |

Commit log:

| commit  | step |
|---------|------|
| `16008dc` | `recount_unobserved_pairing` (44× faster, bit-identical to numpy) |
| `e9dde4e` | `recount_unobserved_outside` (236–299× faster; Pólya direct-sum dodges `lgamma` cancellation under `-ffast-math`) |
| `d419bce` | posteriors + transitions + BD tails (machine-precision, sub-ms) |
| `aa3b027` | `recount_unobserved_logsurv_grad` (bit-identical to numpy) |
| `117ac4d` | hybrid orchestrator: C steps 1–4 + torch chain rule (intermediate landing) |
| `000a73f` | `rate_to_p_jac` / `rate_to_q_jac` analytical Jacobians |
| `4099f81` | chain rule Python prototype |
| `fe6e5a4` | Yule limit + t=inf fixes for the Jacobians |
| `665d59a` | Yule-tolerance branching (`|μ−λ|/max ≤ 1e-7`) |
| `c2659f1` | chain rule in native C, torch retired from L(0) path |
| `7fd3425` | retire torch from `_gradient_raw_native` too (full C path end-to-end) |

For ed194 (F = 12k): `gradient_native` ≈ 425 ms, dominated by the
observed-grad batch over all families; the L(0) part is ≈ 8 ms.

## K=2 LogisticShift mixture (commit `bae92b8`)

`validation/mixture_ml.py` was migrated from `LOG_RATE_CLIP` to the
bounded logit transform on `base_dup`, sharing the
`_shared._dup_to_logit` / `_logit_to_dup` / `_dup_jacobian` /
`MAX_PROB_NOT1` machinery with `ml.py` and `map.py`. `delta_dup`
stays in log-space as a multiplicative shift; the MAP prior on the dup
block now sits in logit coords (`mu_dup = logit(0.5)`).

CLI gained `--num-starts`, `--optimizer {BFGS, native_bfgs}`, and
`--num-threads` so K=2 composes with `tools/pool_fits.sh`.

Smoke-test (dpann80, K=2, `--bounded`, 5 cycles, 51 s):

- LL_corr_mix = −90,511 (single-component reference: −94,177)
- max(base_dup) = 0.682 (well below cap 1 − 2⁻³⁰)

Pre-migration (unconstrained) K=2 results are preserved in
[NO_DUPLICATION_CONSTRAINT.md](NO_DUPLICATION_CONSTRAINT.md). The
bounded K=2 fits live in `validation/outputs/mixture_K2/`.

## Constraint regime

| axis | constraint | enforced in |
|---|---|---|
| duplication | `dup ≤ MAX_PROB_NOT1 ≈ 1 − 2⁻³⁰` via logit | `validation/_shared.py:_logit_to_dup` (matches Java `is_duprate_bounded = true`) |
| gain        | `|log γ| ≤ 33` clip                       | `make_objgrad_*` in `validation/_shared.py` (matches Java `MAX_GAIN_RATE`) |
| length      | same `|log t| ≤ 33` clip                  | same |
| init        | `random_initial_rates` per `recount/ml.py`, seed=2025 | matches Java `TreeWithRates(tree, RND)` per-node Uniform/Exp draws |
| optimizer   | `native_bfgs` (`native/src/recount_bfgs.c`) | bit-faithful port of Java `FunctionMinimization.dfpmin` from [miklosc/Count](https://github.com/miklosc/Count) |

The gain cap is documented in [GAIN_CONSTRAINT.md](GAIN_CONSTRAINT.md):
the SI/Java-declared `Logistic(GainParameter, 33)` cap is NOT enforced
by the publication runs (Csurös' bundled κ values reach
4.9 × 10¹³), so we keep the gain block uncapped on the upper side to
match the publication regime. The sub-critical duplication cap IS
respected by the publication rates (`max dup = 1.000` exactly) and is
on by default in the bounded code path.

## How to regenerate after a fit lands

```bash
PYTHONPATH=. python3 validation/subclade_ancestor_comparison.py --variant bounded
```

Reads from
`validation/outputs/{bounded_csuros, csuros_match, sota_ml, map_sigma1, brownian_*, brownian_extend_*, mixture_K2, reproduction}/`
and writes the per-subclade path plots + bar charts that the README
embeds. ~30 s total. The full plot pipeline (per-family presence
tables + every Set 1/1.5/2/3 path + the rate-scatter panels) is in
[validation/QUEUE_INSTRUCTIONS.md](validation/QUEUE_INSTRUCTIONS.md)
under "After the queue finishes".
