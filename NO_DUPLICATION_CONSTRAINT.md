# recount — Results WITHOUT the sub-critical Yule constraint (λ ≤ 1)

> This document mirrors the structure of [README.md](README.md) but
> reports fits done **without** Csurös' default `is_duprate_bounded = true`
> + `MAX_GAIN_RATE = 33` constraints. These are the historical pre-
> constraint snapshot — preserved here for comparison with the now-default
> bounded results in [README.md](README.md).
>
> Raw fit artifacts: `validation/outputs_no_dup_cap/`.

## Context: why this constraint matters

**Csurös 2026 PNAS Methods**:
> "the biologically inspired restriction that **duplication and gain rates
> are bounded by the loss rate** (λ_v ≤ 1, γ_v ≤ 1)"

**SI Section A.4**:
> "In our implementation we impose **bounded gain rates γ ≤ 1** and use
> the logit-scaled rate as the optimizable parameter"

**Java implementation** ([MLDistribution.java:157](https://github.com/miklosc/Count/blob/master/src/count/model/MLDistribution.java#L157)):
```java
private boolean is_duprate_bounded = true; // if bounded, <1.0 is enforced
```
Enforced at [MLDistribution.java:329](https://github.com/miklosc/Count/blob/master/src/count/model/MLDistribution.java#L329) via logit transform: `dup ≤ MAX_PROB_NOT1 · loss_rate`.

**But: the gain cap in code is much looser than what the paper/SI claim.**
[MLDistribution.java:71](https://github.com/miklosc/Count/blob/master/src/count/model/MLDistribution.java#L71): `MAX_GAIN_RATE = 33.0` is the cap on the **log-gain logit parameter**, not on the gain rate directly. The `Logistic(θ, max_value=33)` transform allows the back-transformed gain rate to reach `e³³ ≈ 2.14×10¹⁴`. Empirically Csurös' fits all show max gain near this cap: 4.9×10¹³, **5.2×10¹⁴** (slightly above the soft cap), 5.8×10¹², 1.3×10¹⁴ on the four subsets. So the SI's stated `γ ≤ 1` is **inaccurate**; the actually-enforced constraint is `log γ ≤ ~33` ⇒ `γ ≤ ~2×10¹⁴`.

**Biological meaning**: `λ > 1` = super-critical Yule process; per-family
gene-count grows unboundedly with positive probability and has no
stationary distribution. While `recount`'s Pólya math still gives a
finite likelihood at `λ > 1` (the Pólya q stays in [0, 1)), the
underlying BD process is biologically implausible.

**Our discovery (this session, pre-constraint code)**:
* Csurös' published .countxml fits saturate the bound: 0/20/9/0 of
  159/149/227/387 non-root nodes (D80/P75/E114/ED194) hit
  `dup = 1.0` exactly — not coincidence, the optimizer is bumping the cap.
* Our (now-historical) MAP σ=1 15-start global on arc269 has **15/536
  non-root branches with `dup > 1`** (max `dup = 28`).
  Our `validation/_shared.py:fit_bfgs_cycles` historically used only
  `LOG_RATE_CLIP = ±50` (no sub-critical constraint), unlike Csurös' Java
  which caps `log(dup) ≤ log(1) = 0` and `log(gain) ≤ log(33)`.

The session that produced this comparison **added** both constraints
(dup logit + gain log-cap) to the optimizer — those are now the README's
default. This doc preserves the pre-constraint comparison.

## Headline findings (unconstrained — no dup cap, no gain cap)

### Validation against Csurös' Java

Same as [README.md → Validation](README.md#validation-against-csurs-java). The
validation tests do not depend on the constraint regime — they test the
forward LL / gradient / per-branch posterior pipeline.

### Reproduction accuracy on the focal subset clades

Same as [README.md → Reproduction accuracy](README.md#reproduction-accuracy-on-the-focal-subset-clades). Loading Csurös' bundled
rates and recomputing is independent of any constraint we add to the
optimizer.

### Root reconstructions across all 5 datasets and three fitting modes

This is the comprehensive comparison **without** the sub-critical bounded
regime: Csurös' published per-subclade ML results, our independent ML
re-fit (cold-start single-run BFGS from default init), the multistart-
polish ML variant (3 random starts + 2× σ-perturbed polish), and our MAP
fit with the same log-Normal σ=1 prior. **Williams2017 is absent from
this snapshot** — it was added to the dataset suite after the constraint
was already enforced; see [README.md](README.md#root-families-across-4-fit-modes) for the bounded williams numbers.

Root *families*:

| Dataset | Csurös ML | Cold ML | Multistart-polish ML | MAP σ=1 (cold) | MAP σ=1 (15-start global) |
|---|---:|---:|---:|---:|---:|
| dpann80 (D80, 80 leaves) | 1,210 | 1,007 | 996 | 1,301 | — |
| proteo75 (P75, 75 leaves) | 2,096 | 2,095 | 1,820 | 2,130 | — |
| eury114 (E114, 114 leaves) | 1,453 | 1,259 | 1,249 | 1,428 | — |
| ed194 (ED194, 194 leaves) | 1,419 | 1,663 | 1,511 | 1,489 | — |
| arc269 LACA (269 leaves) | *— not fit —* | 3,533 (asymptote) | 3,608 (asymptote) | 3,108 | 3,070 |

Root *copies*:

| Dataset | Csurös ML | Cold ML | Multistart-polish ML | MAP σ=1 (cold) | MAP σ=1 global | Csurös cp/fm | MAP global cp/fm |
|---|---:|---:|---:|---:|---:|---:|---:|
| dpann80 | 1,396 | 1,310 | 1,247 | 1,713 | — | 1.15 | — |
| proteo75 | 3,381 | 3,171 | 2,509 | 3,183 | — | 1.61 | — |
| eury114 | 1,736 | 1,597 | 1,559 | 1,763 | — | 1.20 | — |
| ed194 | 1,692 | 2,262 | 1,970 | 1,936 | — | 1.19 | — |
| arc269 LACA | *— not fit —* | 8,664 (asymp.) | 8,963 (asymp.) | 5,906 | 5,426 | — | 1.77 |

LLs and basin diagnostics (ours vs Csurös, with **max_dup** and **max_gain**
showing the boundary escape that the now-default constraint prevents):

| Dataset | Csurös LL | Cold ML LL (Δ) | Cold max dup | Cold max gain | MS-polish LL (Δ) | MS-polish max dup | MS-polish max gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| dpann80 | -94,147 | -93,857 (+290) | 2.2 | 42.9 | -93,821 (+326) | 2.2 | (high) |
| proteo75 | -154,159 | -154,344 (-185) | 4.3 | 5,468 | -154,058 (+101) | 1.1×10⁴ | (high) |
| eury114 | -274,643 | -274,644 (-1) | **9.3×10⁶** | **4.8×10⁴** | -274,518 (+124) | 8.8×10⁶ | (high) |
| ed194 | -395,601 | -396,078 (-477) | **8.4×10⁴** | **3.2×10¹⁴** | -395,323 (+278) | 4.4×10³ | (high) |
| arc269 | *— not fit —* | -1,129,653 | **7.6×10⁹** | 1,538 | -1,109,266 | 5.6×10⁹ | 1,756 |

MAP σ=1 (with the log-Normal prior) tames the boundary escape:

| Dataset | MAP cold LL | MAP cold max dup | MAP cold max gain |
|---|---:|---:|---:|
| dpann80 | -94,227 | 1.2 | 1.89 |
| proteo75 | -154,076 | 3.7 | 1.02 |
| eury114 | -274,379 | 3.1 | 1.66 |
| ed194 | -395,540 | 3.0 | 1.77 |
| arc269 (cold) | -1,117,480 | 43 | 0.12 |
| **arc269 (15-start global)** | -1,096,512 | 28 | 0.53 |

The 15-start MAP σ=1 multistart found a basin +20,968 nat deeper than
the cold MAP, with rates still interior (max dup = 28, max gain = 0.53).
This is the historical LACA quote referenced as "global MAP".

The multistart-polish ML deltas above mostly come from finding deeper
boundary basins than Csurös' Java found. Only the dpann80 multistart-polish
ML lands cleanly interior (max dup = 2.2 — biological scale). Csurös'
published rates also have boundary structure (max gain reaching 10¹⁴ on
some subset nodes) — so all of {Csurös ML, our cold ML, our
multistart-polish ML} on arc269 are climbing the same Pólya-shape → ∞
asymptote at different depths. The only quote that is unambiguously
biologically interpretable is MAP σ=1 15-start global (max dup = 28 on
arc269).

For arc269 the simple-ML row is not a biologically meaningful LACA
estimate — it sits on the Pólya-shape → ∞ asymptote at a single
internal node (max dup ≈ 7.6 × 10⁹ at the cold-ML basin, 5.6 × 10⁹ at the
multistart-polish basin). The full mathematical characterisation,
including a proof that `sup_Θ L(Θ)` is not attained on the open parameter
set, is in
[docs/boundary_identifiability.pdf](docs/boundary_identifiability.pdf).

### K=2 LogisticShift mixture fits — all interior, consistent improvement (historical)

Adding a second rate category (K=2 LogisticShift mixture; see
[docs/logistic_shift_gradient.tex](docs/logistic_shift_gradient.tex))
improves the corrected log-likelihood substantially on every dataset
without falling into a boundary basin:

| Dataset | K=1 cold ML LL | K=2 LL | Δ (K=2 − K=1) | p_mix (cat 1, 2) | δ_dup, δ_length (cat 2) | K=2 max_dup |
|---|---:|---:|---:|:---:|:---:|---:|
| dpann80 | -93,857 | -89,670 | **+4,186** | 0.50 / 0.50 | -0.64 / -1.59 | 6.1 |
| proteo75 | -154,344 | -147,818 | **+6,526** | 0.34 / 0.66 | -0.86 / -1.25 | 19.9 |
| eury114 | -274,644 | -255,596 | **+19,048** | 0.36 / 0.64 | -0.14 / -1.98 | 16.8 |
| ed194 | -396,078 | -371,284 | **+24,794** | 0.42 / 0.58 | -0.23 / -1.82 | 14.2 |
| arc269 | -1,129,653 | -1,089,203 | **+40,450** | 0.63 / 0.37 | -0.20 / -1.78 | 42 |

All K=2 fits have max dup in [6, 42] — interior without needing a MAP
prior. The mixture structure naturally regularises the boundary
asymptote: instead of one node pushing dup → ∞ to fit a near-Poisson
clade, the optimizer assigns those families to a second category with
shorter branches (length × ~0.17–0.29) and slightly less duplication
(× ~0.42–0.82).

arc269 K=2 LL = −1,089,203 sits +7,309 nat below the MAP σ=1 15-start
global, also with an interior basin — K=2 mixture was the deepest
interior fit to date on arc269 in the unconstrained regime.

> K=2 with the new bounded code is out of scope for this comparison;
> the K=2 path uses `LOG_RATE_CLIP = ±50` rather than the logit
> transform (see [validation/mixture_ml.py:57](validation/mixture_ml.py)),
> so re-running it would not be cleanly bounded. The numbers above are
> the unconstrained K=2.

#### K=2 ancestral reconstructions (p_mix-weighted across categories)

| node | cat 1 cp / fm | cat 2 cp / fm | p-weighted cp / fm / cpf |
|---|---:|---:|---:|
| dpann80 root | 1,484 / 896 | 932 / 829 | (0.50, 0.50) → 1,210 / 862 / 1.40 |
| proteo75 root | 2,240 / 1,428 | 2,170 / 1,652 | (0.34, 0.66) → 2,194 / 1,575 / 1.39 |
| eury114 root | 2,938 / 1,543 | 1,335 / 1,194 | (0.36, 0.64) → 1,918 / 1,321 / 1.45 |
| ed194 root | 3,207 / 1,649 | 1,408 / 1,256 | (0.42, 0.58) → 2,171 / 1,423 / 1.53 |
| arc269 LACA (root) | 9,734 / 4,364 | 2,806 / 2,462 | (0.63, 0.37) → **7,192 / 3,666 / 1.96** |
| arc269 DPANN ancestor (node 347) | 8,537 / 3,593 | 2,261 / 2,049 | → 6,234 / 3,026 / 2.06 |
| arc269 Proteoarchaea (node 535) | 8,862 / 4,161 | 2,848 / 2,505 | → 6,655 / 3,553 / 1.87 |
| arc269 Euryarchaeota (node 461) | 18,008 / 4,654 | 2,994 / 2,555 | → 12,498 / 3,884 / 3.22 |

The K=2 LACA quote is **cp=7,192 / fm=3,666 / cpf=1.96** — larger than
the K=1 MAP σ=1 15-start global (cp=5,426 / fm=3,070 / cpf=1.77) by ~33%
in cp and ~19% in fm.

### Subclade ancestors: arc269 full-tree fits vs subset-only fits (unconstrained)

![Subclade-ancestor copies + families (unconstrained, no MAP polish)](validation/outputs/subclade_ancestor_comparison_unconstrained.png)

With the MAP polish (15-start global) line added:

![Subclade-ancestor copies + families (unconstrained, with MAP polish)](validation/outputs/subclade_ancestor_comparison_unconstrained_with_map_polish.png)

Per-subset path plots (arc269 root → median-fm leaf within each subset
clade; left = posterior copies log-scale; right = posterior families
linear). Set 1 (no MAP polish):

| DPANN (D80) | Proteoarchaea (P75) |
|---|---|
| ![dpann80 unconstrained path](validation/outputs/arc269_path_dpann80_unconstrained.png) | ![proteo75 unconstrained path](validation/outputs/arc269_path_proteo75_unconstrained.png) |
| **Euryarchaeota (ED194)** | **Methanobacteriati (E114)** |
| ![ed194 unconstrained path](validation/outputs/arc269_path_ed194_unconstrained.png) | ![eury114 unconstrained path](validation/outputs/arc269_path_eury114_unconstrained.png) |

Set 2 (with MAP polish = 15-start global):

| DPANN (D80) | Proteoarchaea (P75) |
|---|---|
| ![dpann80 unconstrained path w/ MAP polish](validation/outputs/arc269_path_dpann80_unconstrained_with_map_polish.png) | ![proteo75 unconstrained path w/ MAP polish](validation/outputs/arc269_path_proteo75_unconstrained_with_map_polish.png) |
| **Euryarchaeota (ED194)** | **Methanobacteriati (E114)** |
| ![ed194 unconstrained path w/ MAP polish](validation/outputs/arc269_path_ed194_unconstrained_with_map_polish.png) | ![eury114 unconstrained path w/ MAP polish](validation/outputs/arc269_path_eury114_unconstrained_with_map_polish.png) |

**Reading the path plots**: the gold star at the rightmost x-value marks
the **true observed leaf totals** for the median-fm leaf in each subset
(median selected with a random tiebreak, seed=2026). The vertical dotted
line marks the subclade-ancestor depth. The left panel uses log scale
(copies vary widely along the path); the right panel uses linear scale
(family counts in a tighter range, denser tick labels).

The boundary-basin cold/multistart-polish ML fits (red/orange lines)
overshoot at the subclade ancestor — sometimes by an order of magnitude
relative to the subset-only Csurös fit. The MAP σ=1 lines stay closer
to the subset-only baseline.

**Subclade-ancestor reconstruction table** (read off the same arc269
internal nodes that correspond to each subset's root):

| Subclade ancestor (arc269 node) | Csurös subset-only ML cp / fm | arc269 cold ML cp / fm | arc269 MS-polish ML cp / fm | arc269 MAP σ=1 cold cp / fm | arc269 MAP σ=1 global cp / fm |
|---|---:|---:|---:|---:|---:|
| DPANN (node 347, 80 leaves) | 1,396 / 1,210 | 11,186 / 3,236 | 8,020 / 2,610 | 7,239 / 2,787 | **3,009 / 1,999** |
| Proteoarchaea (node 535, 75 leaves) | 3,381 / 2,096 | 7,549 / 3,230 | 7,789 / 3,286 | 5,817 / 3,042 | **5,118 / 2,947** |
| Euryarchaeota+Methanobacteriati (node 461, 194 leaves) | 1,692 / 1,419 (ED194) and 1,736 / 1,453 (E114) | 9,519 / 3,908 | 9,095 / 3,731 | 6,557 / 3,460 | **5,372 / 3,087** |

**Finding**: every arc269 full-tree fit *overestimates* the subclade
ancestor cp/fm relative to the subset-only fits — by factors of:
- DPANN: 2.2×–8.0× cp, 1.7×–2.7× fm
- Proteoarchaea: 1.5×–2.3× cp, 1.4×–1.6× fm
- Euryarchaeota: 3.2×–5.6× cp, 2.2×–2.7× fm

Even the "best" (most interior) MAP σ=1 global gives ~2-3× higher
subclade-ancestor counts than the subset-only fits on the same data.

### Non-parametric bootstrap CI on the K=1 MAP σ=1 LACA quote (unconstrained)

20 bootstrap iterations, each resamples 90,243 family indices with
replacement and refits MAP σ=1 warm-started from the **unconstrained**
15-start global rates. All 20 fits converged to the same basin
(max dup=28.01 and max gain=0.529 exactly across all 20).

| param | mean | SD | 2.5% | 50% | 97.5% |
|---|---:|---:|---:|---:|---:|
| **cp** | 5,436 | 139 | 5,182 | 5,423 | 5,687 |
| **fm** | 3,074 | 56 | 2,982 | 3,065 | 3,165 |
| **cpf** | 1.768 | 0.029 | 1.731 | 1.761 | 1.827 |
| LL_raw | -1,097k | 9,154 | -1,114k | -1,098k | -1,081k |

**Historical LACA quote with confidence (unconstrained MAP σ=1 15-start
global): cp = 5,436 (95% CI: 5,182 – 5,687); fm = 3,074 (95% CI: 2,982 –
3,165); cpf = 1.77 (95% CI: 1.73 – 1.83)**.

### Headline observation: LACA is ~2-3× any subclade ancestor (same as bounded)

The qualitative finding — LACA cp/fm is ~2-3× any subclade ancestor's
cp/fm under either fit regime — is robust to whether the sub-critical
constraint is enforced. The constraint changes the **per-node maximum
dup/gain rates** (capped vs unbounded) without much affecting the root
posterior expectations under MAP. See
[README.md → Headline findings](README.md#headline-findings)
for the bounded version of this finding.

## Csurös' published rates are also near the boundary

To put our boundary observations in context: Csurös' own published
maximum-likelihood rates for the 4 focal subset clades (extracted
directly from the bundled `.countxml.gz` files) also have many nodes
near the Pólya-shape → ∞ asymptote on the gain parameter:

| Dataset (Csurös' published ML) | F families | leaves | max gain | max dup | max length | # nodes with gain > 10³ |
|---|---:|---:|---:|---:|---:|---:|
| dpann80  (D80) | 3,034 | 80 | **4.9 × 10¹³** | 1.0 | 1.56 | 33 |
| proteo75 (P75) | 5,179 | 75 | **5.2 × 10¹⁴** | 1.0 | 1.69 | 3 |
| eury114  (E114) | 7,335 | 114 | **5.8 × 10¹²** | 1.0 | 1.41 | 23 |
| ed194    (ED194) | 8,855 | 194 | **1.3 × 10¹⁴** | 1.0 | 2.35 | 41 |

- `max gain` on the order of 10¹²–10¹⁴ at one or more nodes per subset
  — same Pólya-shape boundary the arc269 cold/multistart-polish ML hits,
  just on the `gain` axis instead of the `dup` axis. Csurös' Java caps
  `log γ ≤ 33` (= `γ ≤ e³³ ≈ 2.14×10¹⁴`), so these are sitting at the
  Java-implementation cap (the now-bounded code in this repo uses the
  same cap by default).
- `max dup = 1` exactly on all 4 subsets — Csurös' fits cap dup at 1.
  The unconstrained snapshot here removed that cap; **we found dup > 1
  routinely in the interior MAP basins** (max=43 on arc269).
- "# nodes with gain > 10³": **3–41 nodes per subset** are at the
  boundary on the published rates from
  [miklosc/Count](https://github.com/miklosc/Count). The phenomenon
  isn't unique to arc269; it's a property of the GLD likelihood at any
  sufficient depth.

We reproduce Csurös' published per-node posterior reconstructions
bit-perfectly even with these boundary rates (≤ 5 × 10⁻¹¹), which
means the numerical inflation of the gain rate is consistent with the
data — it's the price of the model not capping it.

## Note on the "asymptote" label in the arc269 row

In the tables above, the arc269 cold ML (cp=8,664 / fm=3,533 / cpf=2.45)
and multistart-polish ML (cp=8,963 / fm=3,608 / cpf=2.48) rows are tagged
"asymptote". These are not biologically interpretable LACA estimates —
they are sequences along the **Pólya-shape → Poisson asymptote** of the
GLD likelihood at a single internal node, where one node's duplication
rate is pushed to ~5.6×10⁹ (vs the second-largest node's value of 124,
eight orders of magnitude lower). For the formal statement and proof
that $\sup_{\Theta \in \Theta^\circ} \mathcal{L}(\Theta)$ is **not
attained on the open parameter set**, see
[docs/boundary_identifiability.tex](docs/boundary_identifiability.tex) /
[docs/boundary_identifiability.pdf](docs/boundary_identifiability.pdf)
— Theorem 1 (non-attainment), Lemma 1 (Pólya → Poisson via PGF / Lévy
continuity), Proposition 2 (quantitative gap budget), and Theorem 2
(coercivity of the MAP objective under a log-Normal prior).

The Pólya parameterisation is from Csűrös & Miklós 2009 ("Streamlining
and large ancestral genomes in archaea inferred with a phylogenetic
birth-and-death model", *Molecular Biology and Evolution* 26(9):2087–
2095, [doi:10.1093/molbev/msp123](https://doi.org/10.1093/molbev/msp123));
the arc269 dataset, the four focal subset clades (D80, P75, E114, ED194),
and the $\Omega_{\min}$ family-conditioning correction are from Csűrös
2026 PNAS (cited in [README.md → Citing](README.md#citing)). The
L(0)-correction proof and the analytical gradient recursions our native
backend implements are from Csűrös 2021 arXiv:2107.11440.

Two different optimizer runs (cold ML vs multistart-polish ML) stop at
different points along the same asymptote — the gap of ~20,000 nat
between them is *how far the optimizer climbed before line-search
precision ran out*, not a difference in biology. Only the MAP σ=1
15-start global column on arc269 (cp=5,426, fm=3,070, cpf=1.77, max
dup=28) is a true interior optimum and the defensible LACA quote in
the unconstrained regime.

Under the **bounded** regime (now the default in [README.md](README.md)),
the constraint prevents the boundary escape entirely — the cold ML fits
still land at the cap (max dup ≈ 1 exactly) but don't go super-critical,
and the MAP σ=1 fits behave qualitatively the same as in the
unconstrained snapshot above.

## Saved unconstrained results

| Subdirectory under `validation/outputs_no_dup_cap/` | Contents |
|---|---|
| `reproduction/` | Csurös' published Java fits, bit-perfect (≤ 5×10⁻¹¹) reproduction |
| `simple_ml/` | Cold-start ML, single-start, 4 subsets + arc269 (no williams) |
| `sota_ml/` | Multistart-polish ML, num-starts=3 + polish-sigmas 0.05,0.10 |
| `map_sigma1/` | MAP σ=1 cold-start, single seed |
| `profile_likelihood_arc269/` | MAP σ=1 15-start global (the historical LACA quote) |
| `map_sigma_sweep/` | σ ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0} |
| `mixture_K2/` | K=2 LogisticShift mixture fits |
| `mixture_K3/` | K=3 LogisticShift (partial) |
| `arc269_omin4/` | Ωmin=4 sensitivity fit on arc269 |
| `bootstrap_arc269/` | 20-iteration non-parametric bootstrap CI |
| `subclade_ancestor_comparison.{png,json}` | Comparison plots (pre-refactor; new versions at the repo's `validation/outputs/*_unconstrained*.png`) |
| `arc269_root_to_leaf_path.png` | Original single-leaf path plot (pre-refactor) |
| `k2_reconstruction.json` | K=2 per-category posteriors |

Original commit reference for the pre-constraint snapshot: c2eb61b
(claude/confident-fermi-ec9e2c).
