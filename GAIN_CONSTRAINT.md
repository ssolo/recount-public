# Gain-parameter constraint: why it's left OFF

> **TL;DR**: Csurös 2026 (PNAS main text + SI A.4) claims `γ ≤ 1`. The
> Java implementation in
> [miklosc/Count](https://github.com/miklosc/Count) declares
> `Logistic(GainParameter, MAX_GAIN_RATE=33)`. The bundled published
> rates exceed that cap by 12+ orders of magnitude (κ up to 4.9×10¹³ on
> dpann80 node 89; 34 of 158 non-root nodes violate the cap).
> Empirically, no upper cap is in force in the runs that produced the
> publication countxml. To match the publication regime, the gain block
> is left unbounded on the upper side in our `subcritical=True` regime;
> the duplication constraint (`dup ≤ MAX_PROB_NOT1 ≈ 1`) — which IS
> empirically enforced — is kept.

## Paper claims vs implementation claims vs published rates

### Csurös 2026 PNAS, main text

> "the biologically inspired restriction that **duplication and gain
> rates are bounded by the loss rate** (λ_v ≤ 1, γ_v ≤ 1)"

### Csurös 2026 PNAS, SI Section A.4

> "In our implementation we impose **bounded gain rates γ ≤ 1** and use
> the logit-scaled rate as the optimizable parameter."

### Java declaration

[`Count/src/count/model/MLDistribution.java:71`](https://github.com/miklosc/Count/blob/master/src/count/model/MLDistribution.java#L71)
in [miklosc/Count](https://github.com/miklosc/Count):
```java
public static final double MAX_GAIN_RATE = 33.0;
// larger value is better for numerical optimization;
// with small values one may get stuck on the boundary
```

[`MLDistribution.java:306`](https://github.com/miklosc/Count/blob/master/src/count/model/MLDistribution.java#L306):
```java
distribution_params.add(newLogistic(new GainParameter(node), MAX_GAIN_RATE));
```

The Logistic wrapper maps θ ∈ ℝ to κ ∈ [0, 33] (`Logistic.set` in
[`ML.java:477-540`](https://github.com/miklosc/Count/blob/master/src/count/model/ML.java#L477)
clips `t = sigmoid(x)` to `[EPS, 1-EPS]` before scaling by max_value).
After this path runs, `κ ∈ [33·EPS, 33·(1-EPS)] ≈ [7×10⁻¹⁵, 33]`.

### Empirical: bundled published countxml rates

Loading the bundled `.countxml.gz` files and extracting the gain values
(stored as κ in `gain_rates[node]` per
[`TreeWithRates.java:346`](https://github.com/miklosc/Count/blob/master/src/count/model/TreeWithRates.java#L346),
where `setGainRate` writes `κ = exp(log_γ + log_p − log_q)` to the
field — the field name is misleading: it holds κ, not γ):

| Dataset (countxml gain field) | max κ (Pólya shape) | # nodes with κ > 33 |
|---|---:|---:|
| dpann80 (D80) | **4.896 × 10¹³** | 34 / 158 |
| proteo75 (P75) | **5.2 × 10¹⁴** | 3 / 148 |
| eury114 (E114) | **5.8 × 10¹²** | 23 / 226 |
| ed194 (ED194) | **1.3 × 10¹⁴** | 42 / 386 |

All four publication fits violate the stated cap, many by 12+ orders of
magnitude.

### Cost of enforcing the cap

Applying `Logistic(33)` to the bundled dpann80 rates (clipping every
κ > 33 down to ≈ 33 via the `Logistic.set` path) and recomputing the LL
through this repo's native backend:

| Configuration | LL |
|---|---:|
| Bundled published rates, as-loaded | **−94,146.94** (matches Java to ≤0.06 nats) |
| Same rates after Logistic(33) roundtrip | **−95,831.75** |
| Δ | **−1,684.81 nats worse** |

So the Java-declared `MAX_GAIN_RATE = 33` constraint, applied to the
publication rates themselves, costs 1685 nats of LL on dpann80 alone.
The publication code path therefore cannot have been running through
the `Logistic(33)` wrapper. Either:

1. An older version of Count without the `Logistic(33)` wrapping
   produced the published rates, OR
2. A different `MAX_GAIN_RATE` was set for the publication runs (the
   comment "larger value is better for numerical optimization" suggests
   the constant is treated as configurable, even though it is `final`
   in the current `master` branch source).

We cannot recover the exact publication-time configuration. The
observable behaviour is "no effective upper bound on κ".

## What this repo implements

In `validation/_shared.py` with `subcritical=True` (default):

| Block | Constraint | Optimiser sees |
|---|---|---|
| `loss[v]` | fixed at 1 (gauge) | — |
| `dup[v]` (non-root) | `dup ∈ (0, MAX_PROB_NOT1 ≈ 1)` via logit | θ ∈ ℝ |
| `gain[v]` | `log γ ∈ [−LOG_RATE_CLIP, +LOG_RATE_CLIP]` (no upper Logistic) | log γ ∈ ℝ |
| `length[v]` (non-root) | `log t ∈ ±LOG_RATE_CLIP` | log t ∈ ℝ |

The outer `±LOG_RATE_CLIP = ±50` is a floating-point overflow guard,
not a model-level constraint. With `LOG_RATE_CLIP = 50`, γ ∈ [e⁻⁵⁰,
e⁵⁰] ≈ [2×10⁻²², 5×10²¹] — wide enough to encompass the largest
publication κ (5×10¹⁴) with five orders of magnitude of margin on each
side.

The duplication constraint is kept because:
- It IS enforced in the publication rates: `max dup = 1.000` exactly on
  every one of D80 / P75 / E114 / ED194.
- It is biologically meaningful (`λ > 1` ⇒ super-critical Yule with no
  stationary distribution).
- The Java `is_duprate_bounded = true` path uses
  `Logistic(DuplicationRate, MAX_PROB_NOT1 · loss_rate)` and the
  publication rates respect it to floating-point precision.

## How to enable the cap

To enforce `Logistic(GainParameter, MAX_GAIN_PARAMETER)` literally
(which gives worse LL on every dataset compared to the publication
rates):

- `validation/_shared.py` keeps the helper functions `_logit_to_gain`,
  `_gain_to_logit`, `_gain_jacobian` and the constant
  `MAX_GAIN_PARAMETER` (currently 33.0). They are unused with the
  default settings; replacing the gain block in `rates_to_x` and
  `x_to_rates` (and `_gain_jacobian` in `make_objgrad_ml` /
  `make_objgrad_map`) re-enables the constraint. Setting
  `MAX_GAIN_PARAMETER` to a much larger value (e.g. `1e15`) bounds
  γ above without clipping the publication regime.

## References

- Csurös 2026 PNAS — paper claims (`docs/references/Csuros2026_main.pdf`
  and `docs/references/Csuros2026_SI.pdf` in this repo)
- [`miklosc/Count`](https://github.com/miklosc/Count) on GitHub —
  Java implementation
- `validation/outputs/reproduction/{ds}_reproduce.branches.csv` —
  per-node bit-perfect recompute of the publication values from the
  loaded countxml rates (matches Java's posterior values to ≤ 5×10⁻¹¹)
- [`NO_DUPLICATION_CONSTRAINT.md`](NO_DUPLICATION_CONSTRAINT.md) —
  extended discussion of the duplication constraint, including the
  boundary-pathology evidence on arc269
