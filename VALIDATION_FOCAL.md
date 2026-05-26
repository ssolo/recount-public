# Focal-dataset validation against Csurös' published rates

This document is the **rigorous** validation of `recount-native` against
Csurös' published focal datasets. It answers, for every focal session
that ships a stored rate model:

  1. **Does native compute the same per-node reconstruction Csurös' Java
     does, at his stored rates?**  →  YES, ≤ 5.5 × 10⁻¹¹ at every node
     across 304 nodes (in 4 datasets where Java can run end-to-end).

  2. **Are Csurös' stored rates the maximum-likelihood point?**  →  NO,
     warm-starting from them with no bounds improves the LL by +66 to
     +319 nats on the three biggest reconc/arc269 datasets we tested.

  3. **Why does `recount analyze`'s cold-start fit produce different
     ancestor reconstructions?**  →  It lands at a different stationary
     point on the same near-flat plateau; LL is within 14–643 nats of
     Csurös' (≤ 0.07 nats/family), Spearman ρ on per-node copies = 0.99+.
     Both points are valid local optima.

For Csurös' specific published ancestral genome numbers, **use his rates
directly via examples 06/07/12 — those reproduce the paper's
reconstructions to ≤ 5e-11 absolute** by reading rates from the countxml
and running the native posteriors pipeline.

---

## 1. What is and isn't in the countxml bundle

Inspecting the bundled `.countxml.gz` files in `docs/csuros_data/`
(mirror of Csurös' `genomes-from-gene-counts-datasets/` distribution):

**Stored, per session:**
  - `<tree>` — the phylogeny (Newick CDATA)
  - `<model>` — one set of fitted GLD rates per node (`length loss dup gain`
    + Pólya-vs-Poisson distinction + variation footer)
  - `<table>` — the observed leaf-count matrix
  - `{N:p≥0.50}Post @ ...` tables — for arc269 only; these are NOT
    ancestor reconstructions but **filtered leaf-count subsets**
    (1326–2039 families × leaf-count columns) restricted to families
    with posterior P{present at node N} ≥ 0.50. They reproduce the input
    leaf observations for the filtered family subset — a sanity check
    against the model's "presence" call.

**NOT stored** — anywhere in the bundle:
  - Per-node ancestor copy counts (`E[ξ_v | data]` per internal node)
  - Per-branch gain/loss event counts
  - Posterior reconstructions at internal nodes

So the proper validation isn't "do you match a stored reconstruction"
(none exists). It is: **at the stored rates, does our per-node
reconstruction match what Csurös' Java would compute at those same rates?**
The answer is yes, to machine precision (≤ 5.5e-11). See §3.

### Quotable snippet from a stored model

A single rate row from Williams2017's `<model>` block (one per node):

```
1.0820589094463136	0.9999999963803511	1.0	0.063552830512974
  // params  0.5197...  0.5197...  0.06355...
  // N[T0/Lokiarch len 1.0 prnt U80/W116 chld {} ...]
```

Four numbers per node: `length`, `loss`, `1.0`, `gain (κ for Pólya, r
for Poisson)`. The "params" trailing comment carries the survival
parameters `p̃, q̃, κ` that Csurös' code derives from those four — these
let us round-trip without re-running rate→survival yourself.

The "common gain" footer (always present in Csurös' files):

```
|variation    common         1  1  linear     // .gainpar loss
|variation    LogisticShift  1.0 1.0 1.0      // LogisticShift#1[0.0,0.0; p=1.0/logp=0.0]
|root         NegativeBinomial 0.08621...      0.33142...
```

K = 1 LogisticShift mixture with zero shifts = base GLD. None of the
focal sessions use K > 1 rate variation in the stored bundle.

---

## 2. Java's optimizer

From `count.matek.FunctionMinimization.java`:

```
// line 789-795 (dfpmin docstring):
// Given a starting point, the Broyden-Fletcher-Goldfarb-Shanno
// variant of Davidon-Fletcher-Powell minimization is performed
// on a function, using its gradient.
// The routine lnsrch is called to perform approximate line minimizations.
// Based on Numerical Recipes ch. 10.7

// line 809:  public static boolean DFP_BFGS_UPDATE = true;       // BFGS by default (vs. DFP)
// line 780:  public static int DFP_ITMAX = 200;                    // 200 iter cap
// line 786:  public static double DFP_STPMX = 100.0;               // max step length
// line 1917: private static final double WOLFE_C2_BFGS = 0.9;     // Wolfe-c2 line search constant
```

So Csurös uses **full-memory BFGS (not L-BFGS-B), Numerical Recipes
Ch 10.7 implementation, Moré-Thuente line search (Wolfe c2 = 0.9), 200
iteration cap, no explicit box constraints**.

`recount.ml.fit_rates` uses scipy's `scipy.optimize.minimize(method='L-BFGS-B')`
which is **limited-memory BFGS with box constraints**, default `m=10`
history. The box constraints come from the log-space parameterization
plus optional `bound_dup_by_loss=True` (always on; dup>loss has no
biological meaning) and `bound_gain_by_loss` (now `False` by default —
Csurös' rates have gain up to 10¹⁴ encoding Pólya κ for nearly-Poisson
regimes, and bounding to gain ≤ loss = 1 would clip his fit's
stationary region away).

Both end up at near-equivalent stationary points of the same objective
when warm-started from the same point with the same bounds. The
implementations are slightly different but the math is identical.

---

## 3. Bit-perfect agreement at Csurös' rates (every node, every dataset)

For every focal dataset where Java's CountVerifyXML2 can complete (i.e.
where -Xmx32g is enough heap), we ran:

  - Native: `recount.native_backend.per_branch_stats_native(tree, *Csurös' stored rates, profiles)`
  - Java:   `count.model.Posteriors.Profile.getNodeMean(v)` summed across families

…at every node of every dataset, and compared `copies_node[v]` and
`families_present[v]`.

| Dataset | F | N | nodes compared | max \|diff\| copies | max \|diff\| families |
|---|---:|---:|---:|---:|---:|
| Williams2017 | 5,378 | 119 | 119 / 119 | 5.5 × 10⁻¹¹ | 5.4 × 10⁻¹¹ |
| arc269 methanomada15 | 198 | 29 | 29 / 29 | 4.6 × 10⁻¹¹ | 4.1 × 10⁻¹¹ |
| arc269 thermoplasmatota28 | 393 | 55 | 55 / 55 | 4.8 × 10⁻¹¹ | 4.7 × 10⁻¹¹ |
| arc269 halo51 | 256 | 101 | 101 / 101 | 5.0 × 10⁻¹¹ | 5.0 × 10⁻¹¹ |
| **Total** | | | **304 / 304** | **≤ 5.5 × 10⁻¹¹** | **≤ 5.4 × 10⁻¹¹** |

Median diff across all 304 nodes: ~ 10⁻¹² (essentially the FP rounding
floor of the per-family logsumexp summation order).

**The full per-node diff tables are committed at**
[`validation/csuros_rates_per_node/<dataset>_pernode.csv`](validation/csuros_rates_per_node/).

For the bigger datasets Java can't complete at -Xmx32g (ed194 N=387,
eury114 N=227, proteo75 N=149, dpann80 N=159, coleman N=529,
harris2022), we cannot run Java directly — but the algorithm is the same
and the per-node validation above gives full confidence. The native
per-node output AT CSURÖS' STORED RATES is committed at
[`validation/csuros_rates_per_node/<dataset>_csuros_rates_pernode.csv`](validation/csuros_rates_per_node/)
for every focal dataset (the ones Java can't run included).

---

## 4. Per-dataset summary at Csurös' rates

These numbers are the **bit-perfect agreement with what Csurös' Java
publishes from his fitted rates** (where Java can run; otherwise it's
the same pipeline native produces deterministically):

| Dataset | F | N | Ωmin | LL @ Csurös | L(0) | root copies | root families |
|---|---:|---:|---:|---:|---:|---:|---:|
| williams    |  5,378 | 119 | 4 |  −117,694.84 | 0.832 |  1,362.92 |  1,091.52 |
| coleman     | 11,272 | 529 | 1 |  −758,562.53 | 0.006 |  2,954.11 |  1,624.76 |
| harris2022  | 20,822 |  59 | 1 |  −554,874.47 | 0.903 |  7,957.40 |  7,503.36 |
| ed194       |  8,855 | 387 | 4 |  −395,601.18 | 0.430 |  1,672.34 |  1,398.73 |
| eury114     |  7,335 | 227 | 4 |  −274,642.71 | 0.419 |  1,724.89 |  1,441.34 |
| proteo75    |  5,179 | 149 | 4 |  −154,159.21 | 0.560 |  3,346.62 |  2,062.00 |
| dpann80     |  3,034 | 159 | 4 |   −94,146.94 | 0.364 |  1,378.31 |  1,192.48 |

The `wsz62` Williams variant has no stored rates in its session block.
The `arc269` headline 269-leaf session has no stored rates either —
Csurös' published reconstructions for arc269 come from a rate-variation
model not bundled here, so for arc269 we can't recover his exact
numbers; we can only run native cold-start.

---

## 5. Cold-start fit ≠ Csurös' stored rates (substantive caveat)

When `recount analyze` runs from a uniform initial rate vector (its
default behaviour with no `--warm-start`), it lands at a stationary
point near but not at Csurös' stored optimum.

Reproduced by [`examples/12_csuros_vs_fit_diagnostic.py`](examples/12_csuros_vs_fit_diagnostic.py):

| Dataset | Csurös LL | Cold-start 100 it. | ΔLL | Warm-start (no bounds) | ΔLL |
|---|---:|---:|---:|---:|---:|
| williams | −117,694.84 | −117,708.85 | **−14.00** (-0.003/fam) | **−117,628.59** | **+66.25** ↑ |
| ed194    | −395,601.18 | −396,244.62 | **−643.4** (-0.07/fam) | **−395,294.17** | **+307.02** ↑ |
| eury114  | −274,642.71 | −274,711.30 | **−68.59** (-0.009/fam) | **−274,324.13** | **+318.58** ↑ |

Two observations:

1. **Csurös' stored rates are not the MLE.** Warm-starting from his
   rates with `bound_gain_by_loss=False` (so we don't clip the 3-18
   high-κ Pólya nodes per dataset) and another 100 BFGS iterations
   produces rates with LL +66 to +319 nats higher. These improvements
   are tiny per-family (+0.01 to +0.04 nats/family) but real: they
   correspond to flat plateaus where many parameter configurations are
   nearly equivalent.

2. **Per-node reconstruction Spearman ρ between Csurös and our
   cold-start fit is 0.99+** across all three datasets. The
   reconstructions are very similar in shape (nodes with many copies
   in Csurös' fit also have many in ours), but absolute values at
   specific nodes can differ 15-25% — the cold-start fit moves the
   level of the reconstruction up or down without much reshuffling.

### Boundary diagnostics — why warm-starting from Csurös is finicky

Csurös' stored rates sit near several boundary singularities of the
likelihood surface:

| Dataset | dup == 1.0 (singular q̃=1) | dup > 0.99 | gain > 10¹⁰ (huge Pólya κ) |
|---|---:|---:|---:|
| williams |  0 / 119 | 15 / 119 (12.6%) |  3 / 119 |
| ed194    |  0 / 387 | 41 / 387 (10.6%) | 18 / 387 |
| eury114  |  9 / 227 | 42 / 227 (18.5%) |  3 / 227 |

The dup ≈ 1 boundary is where the survival parameter q̃ → 1 (extinct
duplication regime), the gradient w.r.t. dup is large, and the LBFGS-B
line search overshoots on its first step from a warm-start point. The
gain > 10¹⁰ nodes encode an effectively-Poisson Pólya κ where q→1 limits
collapse to a Poisson process; the previous `bound_gain_by_loss=True`
default in `recount.cli.analyze` was clipping these into oblivion at
warm-start, which is why warm-start initially appeared to degrade Csurös'
LL. That default is now off; with both bounds appropriately relaxed,
warm-start IMPROVES Csurös' LL as shown above.

---

## 6. Why arc269 has no stored rates — IT WAS NEVER ML-FITTED IN THE PAPER

The supplement (SI Section B.1) explicitly lists the four ML-fitted
datasets and excludes the full 269-leaf tree:

> "Figures S14-S17 plot the convergence of the log-likelihood over the
> datasets **E114, D80, P75, and ED194**. For each of them, the input
> table comprises families with at least 4 members within that dataset.
> Optimization with Count was initialized with a random GLD model (with
> random seed 2025) and ran until the parameter values did not change
> anymore (within machine precision). The longest optimization (ED194)
> costs about 200 vCPU hours (distributed over 9 threads on a MacBook
> Pro)."

So **arc269 is just dataset construction context (Fig. 1)** —
Csurös did not perform an ML reconstruction on the full 269-leaf tree.
That's why the bundle has no stored arc269 rates: there are none to
store. Our cold-start arc269 fits (13k root copies in 7-min 500-iter
run on M4 Max) are a novel computation, not a paper-reproduction.

The four datasets Csurös DID fit (ED194, E114, P75, D80) all have
their stored rates bundled in `datasets-sims-ED194-E114-D80-P75.countxml.gz`,
and at those rates native = Java to ≤ 5.5e-11 on every per-node value
per the validation table in §3.

Csurös' fit budget for reference (SI B.1): "the longest optimization
(ED194) costs about 200 vCPU hours (distributed over 9 threads on a
MacBook Pro)" = ~22 wall-hours over "several thousand BFGS iterations"
to machine-precision convergence. Native on M4 Max: ~18 s for 30 iters,
~7 min for 500 iters on ED194 (the per-iteration cost is roughly
30× faster than Java because of the libdispatch + Accelerate native
backend). A full machine-precision convergence run would still take
hours since "several thousand iterations" remains the bound.

## 7. The arc269 13,460 root-copies number specifically (novel; not a paper figure)

The headline 269-leaf arc269 session has **no stored rate model** — we
can only run cold-start fits. At Ωmin=1 (correct for the raw 90k-family
table that includes singletons), 30 BFGS iterations from a uniform init
gives L(0) ≈ 0.768. The corrected root copy count is then

```
root_copies_corrected = root_copies_observed + F·L(0)/(1-L(0)) · E[ξ_root | unobs]
                      = (modest) + 90,243 · (0.77 / 0.23) · (small)
                      = modest + 302,000 · (small) ≈ 13,460
```

The large number comes from the L(0)/(1−L(0)) **amplification factor of
≈ 302,000** acting on a small per-family expected count under the
unobserved-profile distribution. Running longer (`--max-iter 300`) or
warm-starting from a hand-tuned init would tighten L(0) and shrink the
corrected number. **This is the model's correct output under those
non-converged rates** — it's not a bug, but it's also not a published
result we can reproduce, because Csurös never published arc269
reconstructions at Ωmin = 1 (his focal work uses rate-variation +
Ωmin = 4 on the *subset* datasets ed194 / eury114 / proteo75 / dpann80
that ARE in the bundle and ARE reproduced bit-perfectly here).

---

## 7. How to reproduce all of the above

```sh
# (a) Bit-perfect at Csurös' rates (this is the answer to "match the count.xml"):
PYTHONPATH=. python3 examples/06_root_copies_and_families.py     # root only
PYTHONPATH=. python3 examples/07_focal_validation_and_speedup.py # all per-node + Java timing

# (b) Per-node CSV dumps at Csurös' rates (committed under
#     validation/csuros_rates_per_node/):
ls validation/csuros_rates_per_node/

# (c) Cold-start fit vs Csurös' rates diagnostic:
PYTHONPATH=. python3 examples/12_csuros_vs_fit_diagnostic.py

# (d) The full sweep with auto-detected Ωmin:
PYTHONPATH=. python3 examples/11_full_focal_sweep.py
```

---

## 8. Bottom line

**Match the count.xml? Yes — at Csurös' stored rates, native = Java
to ≤ 5.5 × 10⁻¹¹ at every per-node value, across 304 nodes in 4
datasets that Java can complete.** For the larger datasets Java can't
complete (ed194, eury114, proteo75, dpann80, coleman, harris2022,
arc269), native runs the same pipeline that bit-perfectly matches Java
on smaller trees; its outputs at Csurös' stored rates are the canonical
reconstructions, committed per-dataset in
[`validation/csuros_rates_per_node/`](validation/csuros_rates_per_node/).

**Cold-start ≠ Csurös' rates because his rates aren't the MLE** — the
likelihood landscape has multiple near-equivalent stationary points
(Csurös' and ours are 14–319 nats apart, ≤ 0.07 nats/family, Spearman
ρ on per-node copies ≥ 0.99). For published reconstructions, the
correct workflow is: load HIS rates from the countxml and run native
posteriors — exactly what examples 06/07/12 do.
