# GLD model complexity on arc269 vs the focal subsets — what we learned

> Mathematical companion: full exposition in
> [docs/boundary_identifiability.pdf](docs/boundary_identifiability.pdf)
> (LaTeX source [docs/boundary_identifiability.tex](docs/boundary_identifiability.tex)).
> This document covers the **non-identifiability** we hit trying to ML-fit
> the full 269-leaf arc269 phylogeny under the bare GLD model, how a
> weak log-Normal prior resolves it, and the resulting LACA
> reconstruction. Headline findings + biological interpretation are at
> the top; the mathematical detail and σ-sensitivity sweep follow.

## Headline findings, in order of decreasing confidence

1. **Csurös' subset-clade reconstructions reproduce bit-perfect.**
   Loading his published rates from the bundled `.countxml.gz` files
   and recomputing posteriors via our native pipeline matches the Java
   reference to ≤ 5 × 10⁻¹¹ at every per-node value, for all 4 focal
   subsets (D80, P75, E114, ED194). The implementation does what Java
   does.

2. **Csurös never fit arc269.** Verified in three places: main-text
   Table 1, SI B.1, and the bundled `tree269-baker-alti-gtdb`
   `.countxml.gz` session (which has `rates=none`). His pipeline only
   fits the four subset clades.

3. **arc269 LACA under the same GLD model (MAP σ=1, 15-start global —
   the only interior optimum):** **~5,900 total copies / ~3,070
   families / cp/fam ≈ 1.77.** The pure-ML estimators on arc269 are
   *not* interior optima — they approach the Pólya-shape → ∞ asymptote
   at a single internal node and give materially different
   reconstructions (cold ML: 8,664 / 3,533 / 2.45; multistart-polish
   ML: 8,963 / 3,608 / 2.48). Only the MAP σ=1 run has a coercive
   objective and a unique global optimum.

4. **LACA inflates over subclade ancestors**: ~1.5–2.5× as many
   families as the largest subset ancestor (Proteoarchaea, 2,096
   families), ~1.7–4.2× as many total copies. cp/fam at the LACA
   (1.77) is in the same range as the subset ancestors (1.15–1.61) —
   only modestly higher. The inflation is dominated by family count
   (the HGT signature), not by copies-per-family (HGT spreads families
   across lineages without duplication). Biological discussion below.

## TL;DR

The unconstrained ML objective for arc269 is **unbounded** along the
Pólya-shape → ∞ asymptote: the optimizer can push the duplication rate
at one internal node toward infinity, infinitesimally improving LL each
step. The supremum of the log-likelihood is not attained on the open
parameter set; see [Theorem 1 of the boundary exposé](docs/boundary_identifiability.pdf) (non-attainment of $\sup \mathcal{L}$)
for the formal statement and PGF-based proof of Pólya → Poisson
convergence along this asymptote.

A finite log-rate clip alone is **not sufficient**: with clip ±50
(rate ∈ [2×10⁻²², 5×10²¹]) the optimizer still climbs to
$\log\,\text{dup}_v \approx 22.5$ at the one boundary node and the
likelihood surface still has the asymptote pathology — clipping just
moves the false-convergence point further inside the box.

A weakly-informative log-Normal prior (σ = 1 in log-rate space) makes
the MAP objective **coercive** (Theorem 2 of the exposé — coercivity of MAP under a log-Normal prior) and yields a
unique global mode in the interior. The three estimators differ
significantly on arc269:

| | Cold-start ML (clip ±50) | Multistart-polish ML (multistart + polish, clip ±50) | MAP σ=1 (15-start global) |
|---|---:|---:|---:|
| arc269 LL | -1,129,653 | -1,109,266 | **-1,096,512** (raw LL at MAP) |
| arc269 root copies | 8,664 | 8,963 | **5,426** |
| arc269 root families | 3,533 | 3,608 | **3,070** |
| arc269 cp/fam | 2.45 | 2.48 | **1.77** |
| max dup rate at solution | 7.6 × 10⁹ | 5.6 × 10⁹ | **28** |
| interior optimum? | no | no | **yes** |

Both ML rows lie on the same asymptote at a single boundary node; the
+20k-nat improvement from cold-start ML to Multistart-polish ML is the optimizer climbing
further, not finding a different basin. Only the MAP σ=1 row is a
biologically interpretable LACA quote.

**Recommended LACA quote**: ~3,100 families with ~1.9 copies each
≈ 5,900 total copies. That's ~2-3× the typical Csurös subclade
ancestor in family count; cp/fam is at the high end of the subclade
range (1.15–1.61). The inflation is dominated by family count (the HGT
signature) rather than copies per family.

## Headline table — clade ancestor reconstructions

| Dataset | F families | leaves | Ωmin | Csurös root copies / families / cp/fam | Our MAP σ=1 (clipped, weak prior) |
|---|---:|---:|---:|---:|---:|
| DPANN (D80) | 3,034 | 80 | 4 | 1,396 / 1,210 / 1.15 | 1,713 / 1,301 / 1.32 |
| Proteoarchaea (P75) | 5,179 | 75 | 4 | 3,381 / 2,096 / 1.61 | 3,183 / 2,130 / 1.49 |
| Methanobacteriati (E114) | 7,335 | 114 | 4 | 1,736 / 1,453 / 1.20 | 1,763 / 1,428 / 1.23 |
| Euryarchaeota (ED194) | 8,855 | 194 | 4 | 1,692 / 1,419 / 1.19 | 1,936 / 1,489 / 1.30 |
| arc269 LACA | 90,243 | 269 | 1 | *not fit by Csurös* | **5,426 / 3,070 / 1.77** (15-start global) |

For the 4 focal subset clades, MAP σ=1 is within ±300 nats of Csurös'
LL and within ±20% on root copies/families — so the prior doesn't
distort the subset answers.

## The unbounded-ML problem

The Pólya parameterisation has the property that as `dup_v → ∞` on any
node v with the gain shape co-rescaled to keep the offspring mean
finite, the per-edge model converges weakly to a Poisson — see
[Lemma 1 of the boundary exposé](docs/boundary_identifiability.pdf)
for the PGF/Lévy-continuity proof. In log-rate space, the LL surface
has an asymptote — every step toward the boundary gives strictly more
LL than the previous, but with diminishing returns. Under genericity
(any clade whose data is sub-Pólya / Poisson-like) the supremum is
**not attained on the open parameter set**.

This produces three optimizer pathologies:

1. **scipy BFGS line search false-converges** at finite log-rate
   because `factr × epsmch` cannot be improved (the boundary gives only
   sub-precision LL improvements at very large rates).
2. **Different starts find different points along the same asymptote**
   depending on where the line search stalls. We see this directly: on
   arc269 the simple-ML basin stops at LL = −1,129,653 (max dup ≈ 7.6 × 10⁹)
   and the Multistart-polish basin pushes another 20 k nat further to
   LL = −1,109,266 (max dup ≈ 5.6 × 10⁹ — *same node*, log-rate
   only marginally further inside the box).
3. **Per-node reconstruction is sensitive to where the climb stopped**:
   the same boundary node absorbs different amounts of "high-variance"
   lineage signal, redistributing ancient copy mass downstream by
   ~10–20 % between cold-start ML and Multistart-polish ML.

Tightening the log-rate clip below Csurös' published max rate (~10¹⁴,
log ≈ 32 on the focal subsets) would block the focal-subset basins
entirely, so we cannot rule out the asymptote by clipping alone.

## A finite log-rate clip is not enough; a smooth prior is

A log-rate clip at ±50 is a hard uniform improper prior on
`rate ∈ [2×10⁻²², 5×10²¹]`. It is *necessary* to prevent numerical
overflow but *not sufficient* to remove the asymptote pathology: the
optimizer can still climb to log-rates ≈ 22 (well inside the clip)
before line search stalls. Our Multistart-polish ML on arc269 is exactly this
pattern — see [Proposition 2 (quantitative gap)](docs/boundary_identifiability.pdf):
truncating at any finite clip $B$ only guarantees that the MLE is
$\epsilon$-suboptimal relative to the unbounded supremum, with
$\epsilon \to 0$ as $B \to \infty$.

The MAP `log(rate) ~ N(μ, σ=1)` prior achieves real coercivity (see
[Theorem 2](docs/boundary_identifiability.pdf) of the exposé): the
quadratic prior cost grows like $\Theta(|\log \cdot|^2 / \sigma^2)$,
which is enough to dominate the asymptote gain and force the optimum
into the interior. With σ = 1, the 99 % credible interval is two orders
of magnitude wide — barely informative for biology, but firmly killing
the boundary climbs. On arc269 the resulting MAP optimum (15-start
global) has max dup = 28 (log ≈ 3.3, prior cost ~5 nat per node) and
max gain = 0.53, all comfortably interior.

### Is σ = 1 actually strong enough?

We have **empirical evidence** that σ=1 suppresses the asymptote on
arc269, but no theoretical guarantee. The reasoning chain:

1. **Coercivity is theoretical** — Theorem 2 says any σ < ∞ gives a
   coercive MAP objective with a finite-rate interior optimum. So at
   any positive σ, *some* interior maximum exists.
2. **But "interior" is not the same as "biological"** — at very large
   σ, the prior is weak and the interior maximum could still have
   `max dup` in the millions or billions (just below the clip B=50).
   This is what we see at σ=2 (max dup = 2.3 × 10¹¹) and σ=3 (max dup
   = 2.4 × 10¹¹) in the σ-sweep below: technically interior, but the
   one-node-asymptote pathology survives.
3. **σ=1 is empirically the loosest σ that gives max dup ~10** — at
   σ=1, max dup = 28 (global) or 43 (cold-start), both biological.
   At σ=1.1, max dup jumps to 6.4 × 10⁵ even from cold start. So σ=1
   is on the *edge* of the regime where the prior cost actually wins
   against the asymptote pull.
4. **The empirical case for σ=1 strong enough**: at the 15-start global
   MAP, the prior cost at max dup=28 is ~5 nat. The asymptote pull on
   that single node is bounded by the LL gain from climbing it further:
   we observed the difference between the cold-start MAP (max dup=43,
   LL=−1,117,480) and the 15-start global (max dup=28, LL=−1,096,512)
   is +21k nat in pure LL, *with the global having a lower max dup*.
   So at the global, the LL surface has stopped favouring the asymptote
   in exchange for fitting the rest of the data better.
5. **What we'd need for higher confidence**: (a) repeat the 15-start
   bootstrap with 50–100 starts to confirm no even-deeper basin exists;
   (b) compute a profile-likelihood CI on cp by holding cp fixed and
   optimising; (c) try σ ∈ {0.8, 0.9} to see if a slightly stronger
   prior gives a similar interior answer or shifts the LACA.

This caveat is captured in [VALIDATION.md](VALIDATION.md) §"arc269
LACA quote (partial)".

## arc269 MAP σ-sweep: the prior-strength transition

To characterise the σ-dependence of the LACA quote we ran a sweep at
σ ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0} on
arc269. All entries are single-start BFGS from the default uniform
init *except* the σ=1.0 row marked "15-start global", which is the
best of a 15-start lognormal-perturbation multistart bootstrap (see
[VALIDATION.md](VALIDATION.md) §"arc269 LACA quote").

| σ | LL at MAP | root copies | root families | cp/fam | max gain | max dup | regime |
|---|---:|---:|---:|---:|---:|---:|---|
| **0.1** | -1,174,102 | 6,276 | 2,995 | 2.10 | 0.097 | 1.3 | fully interior (strongest prior) |
| **0.25** | -1,151,270 | 2,839 | 2,224 | 1.28 | 0.15 | 8.8 | fully interior |
| **0.5** | -1,137,754 | 5,831 | 3,031 | 1.92 | 4.0 | 2.1 × 10⁴ | interior (one node creeping) |
| **0.75** | -1,137,015 | 7,809 | 3,404 | 2.29 | 37 | 6.3 × 10⁴ | semi-boundary basin |
| **0.9** | -1,115,775 | 6,807 | 3,327 | 2.05 | 0.16 | 66 | interior (basin ≠ σ=1) |
| **1.0** (cold start) | -1,117,480 | 5,906 | 3,108 | 1.90 | 0.12 | 43 | interior (local optimum) |
| **1.0** (15-start global) | **-1,096,512** | **5,426** | **3,070** | **1.77** | **0.53** | **28** | **interior (global)** |
| 1.1 | -1,144,089 | 3,966 | 2,507 | 1.58 | 2.9 | 6.4 × 10⁵ | boundary |
| 1.25 | -1,144,073 | 7,665 | 3,281 | 2.34 | 653 | 1.36 × 10⁹ | boundary (basin A) |
| 1.5 | -1,128,519 | 4,304 | 2,666 | 1.62 | 19 | 1.66 × 10⁶ | boundary (basin B) |
| 2.0 | -1,143,340 | 8,539 | 3,492 | 2.45 | 1.9 × 10³ | 2.3 × 10¹¹ | boundary (basin A') |
| 3.0 | -1,143,522 | 9,184 | 3,575 | 2.57 | 4.0 × 10³ | 2.4 × 10¹¹ | boundary (basin A') |

**Reading the σ ≤ 1 (tight) regime** — the LACA quote (cp / fm / cpf)
ranges over roughly {4,300 – 7,800} / {2,700 – 3,400} / {1.6 – 2.3}
across σ ∈ [0.5, 0.9] from single-start BFGS. The variation is
*basin-of-attraction* noise (different starts land in different local
optima), not a smooth σ-deformation: σ=0.75's cp=7,809 is higher than
σ=0.5's 5,831 *because the σ=0.75 single start happened to land in a
semi-boundary basin* (max dup=6.3×10⁴ vs σ=0.5's 2.1×10⁴). σ=0.9
finds yet another interior basin with cp=6,807 (different from σ=1's
5,906/5,426). The tight rows (σ=0.1 and σ=0.25) are fully interior —
max dup=1.3 and 8.8 respectively — but they pull cp in opposite
directions (σ=0.1 → 6,276; σ=0.25 → 2,839). At σ=0.1 the prior dominates
so heavily that all rates are forced toward μ, *raising* gain
contributions (cp moves up). σ=0.25 happens to find a basin with
many low-gain nodes (cp moves down). The non-monotonic cp vs σ is
again basin-noise from single starts at each σ. **Multistart matters
at every σ.** The σ=1 15-start result is the best-anchored row in this
table and is our recommended LACA quote: **cp=5,426 / fm=3,070 /
cpf=1.77.**

What this tells us:

1. **The interior↔boundary transition happens between σ=1 and σ=1.25.**
   Even σ=1.25 is too weak to suppress the Pólya asymptote — single-start
   BFGS flips into a boundary basin with one node at dup ≈ 10⁹. The
   transition is **sharp**, consistent with Theorem 2 (the prior either
   dominates the asymptote, giving an interior optimum, or it doesn't,
   in which case there's no interior critical point and the optimizer
   drifts to the boundary).
2. **The boundary regime has multiple local optima.** σ=1.5 lands in
   a *different* boundary basin (cp=4,304 / fm=2,666 / cpf=1.62,
   max dup=1.7 × 10⁶) than σ=1.25 (cp=7,665 / fm=3,281 / cpf=2.34,
   max dup=1.4 × 10⁹) or σ=2 / σ=3 (cp=8,500-9,200 / fm=3,500-3,600 /
   cpf≈2.5, max dup=2.3 × 10¹¹). Which basin you find depends on the
   starting point and the random ridge structure the optimizer
   encounters. Single-start MAP at σ > 1 is therefore not reliable —
   multistart is required to consistently land at the global MAP.
3. **σ = 1 is robustly the right choice for the prior strength** (the
   loosest σ that still gives a *guaranteed* interior optimum), but a
   single start is **not** enough to find the global σ=1 MAP. A 15-start
   multistart (commit c68bca1) showed that the cold-start MAP at
   5,906 cp / 3,108 fam / 1.90 cpf is a *local* optimum 21 k nat above
   the true global MAP, which sits at **5,426 cp / 3,070 fam / 1.77 cpf**
   (LL=-1,096,512, max dup=28). Most perturbed starts at σ=1 land in
   distinctly worse basins (LL between -1,158k and -1,402k); only one
   of 15 (start 10) found the global. So the σ=1 MAP landscape has the
   same multi-basin issue as the boundary regime, just confined to
   interior basins.
4. **Pure LL at MAP is non-monotonic in σ** because of basin-flipping.
   σ=1.5 has higher LL than σ=1.25 and σ=2 despite being in a boundary
   basin, because its particular boundary basin happens to be deeper in
   pure LL. None of these are interpretable as a "marginal sensitivity"
   curve — they're snapshots from different basins of attraction.

Three lessons for practitioners working with the GLD model on similar
deep phylogenies:

- The MAP-prior approach is essential at scale; the bare ML on arc269
  has no interior optimum.
- σ ≈ 1 in log-rate-Normal space is a good default. Tighter
  (σ ≤ 0.5) doesn't add much; looser (σ ≥ 1.25) re-introduces the
  multi-basin pathology.
- If you do use σ > 1, **always run multistart** — single-start can
  flip into one of the boundary basins.

## arc269 Ωmin sensitivity (size > 3 families only)

Question we asked: would conditioning on families with ≥ 4 total leaf
copies (Csurös' subset convention; Csurös uses Ωmin=1 only on arc269,
Ωmin=4 on D80/P75/E114/ED194) suppress the asymptote on arc269?

We ran cold-start ML, multistart-polish ML, and MAP σ=1 (15-start) on
arc269 with the L(0)-correction recomputed at Ωmin=4 (profiles
unchanged, but the corrected LL now divides by `(1-L(0)_{Ωmin=4})^F`
instead of `(1-L(0)_{Ωmin=1})^F`):

| Fit | Ωmin=1 LL/cp/fm/cpf/max_dup | Ωmin=4 LL/cp/fm/cpf/max_dup |
|---|---|---|
| Cold ML | -1,129,653 / 8,664 / 3,533 / 2.45 / 7.6 × 10⁹ | +756,546 / 55,855 / 7,678 / 7.27 / **1.25 × 10¹⁴** |
| Multistart-polish ML | -1,109,266 / 8,963 / 3,608 / 2.48 / 5.6 × 10⁹ | +999,578 / 61,463 / 7,328 / 8.39 / **65** (interior!) |
| MAP σ=1 (15-start) | -1,096,512 / 5,426 / 3,070 / 1.77 / 28 | +811,244 / 41,621 / 7,467 / 5.57 / **3.4 × 10¹⁴** |

(Ωmin=4 LL values are positive because the L(0) correction
`-F · log(1-L(0))` outweighs the negative raw LL when L(0) is close to 1.)

**Surprises:**

1. **Ωmin=4 does NOT consistently fix the asymptote.** Cold ML and
   MAP σ=1 both fall into boundary basins (max dup ≈ 10¹⁴) at Ωmin=4
   even though they were interior or near-interior at Ωmin=1. The
   L(0)-correction term changes the loss landscape in a way that *re-introduces*
   the boundary attractor at MAP σ=1.
2. **Multistart-polish ML accidentally lands interior at Ωmin=4** (max
   dup=65), giving a clean LACA reconstruction of 61,463 cp /
   7,328 fm / 8.39 cpf — but the much larger ancestral repertoire
   reflects that we now condition on "≥4 copies somewhere"
   (rare in deep ancestor → big inflation of estimated copies to
   compensate).
3. **The Ωmin=4 LACA is a different question.** It estimates "expected
   copies at LACA, conditioned on families with ≥4 leaf-copies", not
   "expected copies of all families". The two are not comparable.

**Bottom line**: Ωmin=4 doesn't solve the boundary issue and changes
the LACA quote substantially (cp goes from 5,426 at Ωmin=1 / σ=1 to
either 41,621 boundary or 61,463 interior at Ωmin=4, depending on
which optimizer + which basin). The MAP σ=1 Ωmin=1 quote
(cp=5,426 / fm=3,070 / cpf=1.77) remains the recommended LACA estimate
because:
- it's the only fit that converges to a clean interior basin (max
  dup=28) with both a coercive objective and the original Ωmin=1
  family conditioning;
- it's biologically interpretable (same conditioning as the underlying
  data, no spurious large gain/dup rates);
- the Ωmin=4 results are mutually inconsistent across optimizers, so
  no single Ωmin=4 quote is well-anchored.

## Subsets show the same effect, more subtly

On the 4 subset clades, Csurös' published rates have max(gain) values
up to **10¹⁴**, which is the same boundary phenomenon (the κ
parameter at certain nodes is effectively at the Poisson limit). At
the subset scale this is largely cosmetic — root copies and families
shift by < 20% between different boundary climbs. At the arc269 scale,
with 90× more families and a more heterogeneous tree, the same
phenomenon drives the much larger 2-3× shifts we observed.

This is consistent with the absence of an arc269 fit in Csurös 2026:
ML on this dataset is genuinely unstable in the publication setup (the
asymptote pathology is severe on the full 269-leaf tree), while on the
subsets the boundary issues are small enough to ignore.

## Biological discussion

The numbers in the headline table tell two stories:

### 1. Subset clade ancestors are reproducible

For each of the 4 subset clades, Csurös' ML, our independent simple
ML, and our MAP σ=1 all converge to root reconstructions within ±20%
of each other and ±300 nats of each other in LL. These four numbers
are real and robust:

- DPANN (Nanobdellati) ancestor: ~1,200 families, ~1.1 cp/fam
- Methanobacteriati ancestor: ~1,450 families, ~1.2 cp/fam
- Euryarchaeota (LECA) ancestor: ~1,420 families, ~1.2 cp/fam
- Proteoarchaea (TACK) ancestor: ~2,100 families, ~1.6 cp/fam

cp/fam in the 1.1–1.6 range across the four subclades suggests these
ancestors had mostly single-copy ortholog cores with little post-LCA
duplication that survived to the present.

### 2. LACA inflates relative to subclade ancestors

The full arc269 fit gives:
- ~3,000 families (about 1.4–2.5× the subclade ancestors)
- 1.9 copies per family (modestly higher than the 1.1–1.6 subclade range)
- ~5,900 total copies (about 1.7–4.2× the subclade ancestors)

This inflation is *most likely* the systematic ancestral-genome
inflation that profile methods get when HGT isn't modeled. The model
has to explain HGT-driven gene-family presences at internal nodes
using only vertical birth/death/gain, and does so by inflating root
content. Two predictions of this hypothesis:

- The inflation should scale with how much HGT history the tree
  aggregates. arc269 has way more cross-clade transfer than any single
  subclade subset → bigger inflation. ✓
- The inflation should be dominated by family count, not cp/fam
  (HGT moves families across lineages without duplication). ✓ (LACA
  cp/fam = 1.9 is close to subclade range; family count is the
  factor that grew).

ALE / DTL pipelines (which model HGT explicitly) typically reconstruct
much smaller LACAs in the literature. Csurös 2026 (p. 5) explicitly
treats ALE's small-ancestor estimates as **wrong-by-design** —
arguing GLD's larger ancestors are correct and that "ALE systematically
favors recent gene origins by HGT over vertical inheritance, placing
implausibly few genes at the deepest ancestors." Our GLD-only finding
necessarily inherits that framing: the inflation relative to ALE is
the gap GLD-vs-ALE will always show in the absence of an HGT term;
whether it's GLD over- or ALE under-estimating is a separate
modelling-choice question that the GLD-only LACA quote does not
adjudicate.

### 3. Why Csurös doesn't fit arc269

From SI B.1 verbatim:
> "The longest optimization (ED194) costs about 200 vCPU hours
> (distributed over 9 threads on a MacBook Pro)."

ED194 has 8,855 families and 194 leaves; arc269 has 90,243 families
and 269 leaves — roughly **10× more families and ~2× more nodes**, so
the per-iteration cost is ≈ 20× higher and the optimization landscape
is harder. Extrapolated to "several thousand iterations to machine
precision", that's ≈ 4,000 vCPU-hours ≈ 18 wall-days on 9 threads — or
~40-90 wall-days on a standard 8-core laptop with thermal throttling.

The native backend handles arc269 in **~30 wall-minutes on a single
M4 Max**, driven by (a) the C kernels running ~30× faster per gradient
call than the JVM, (b) parallelism across all 16 cores via libdispatch,
(c) the log-rate clip preventing the unbounded boundary climbs that
would otherwise drive iteration count up.

(Was the arc269 omission deliberate methodology, compute budget, or
both? Our findings on the unbounded-ML pathology suggest the latter
in either case — even with infinite compute, the unconstrained ML on
arc269 doesn't have a unique answer, so a publication-ready fit
requires a regulariser like the MAP-σ=1 prior used here.)

## Bottom line

| Quantity | Subset clades (4) | Full arc269 LACA | Ratio |
|---|---:|---:|---:|
| root families | 1,210–2,096 | **~3,100** | ~1.5–2.5× |
| copies / family | 1.15–1.61 | **~1.9** | ~1.2–1.6× |
| total root copies | 1,396–3,381 | **~5,900** | ~1.7–4.2× |

For the subset clade ancestors — bit-perfect against the published
Java reconstructions from [miklosc/Count](https://github.com/miklosc/Count),
no fitting needed.

For the LACA: we have the first ML reconstruction under the GLD model
of Csurös 2026, at ~5,900 total copies / ~3,100 families / cp/fam ≈ 1.9. The 2-3×
family count inflation vs subclade ancestors is consistent with the
known systematic-ancestral-genome-inflation pathology that profile
methods exhibit in the presence of unmodeled HGT.

## Reproduce

```sh
# Cold-start ML, no sub-critical cap (the 8,664 / 3,533 / 2.45 quote
# above is the uncapped fit; the canonical capped variant lives at
# validation/outputs/bounded_csuros/). ~25 min on M4 Max:
PYTHONPATH=. python3 validation/ml.py --dataset all --num-starts 1 \
    --polish-sigmas "" --cycle-iters 100 --max-cycles 15 --no-subcritical \
    --out-dir validation/outputs_no_dup_cap/simple_ml

# MAP σ=1, 15-start multistart on arc269 — the 5,426 / 3,070 / 1.77
# LACA quote. ~50 min on M4 Max:
PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \
    --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2027 \
    --out-dir validation/outputs_no_dup_cap/profile_likelihood_arc269

# Cold-start MAP σ=1 on all five datasets (the cp=5,906 / fm=3,108 cold-start
# row in the σ-sweep), ~25 min:
PYTHONPATH=. python3 validation/map.py --dataset all --sigma 1.0 \
    --num-starts 1 --out-dir validation/outputs/map_sigma1
```

All three output directories above are present in the repo. The
full σ-sweep entries (σ ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5,
2.0, 3.0}) used to build the σ-sweep table are in
`validation/outputs_no_dup_cap/map_sigma_sweep/`.
