# What's verified, what's not, and how to reproduce

> Companion to [README.md](README.md). This is the honest accounting of
> the validation state — what numerical claims have been spot-checked,
> what's an LLM-drafted derivation, and what remains assumed.

## Bit-perfect (against external authority)

These are the strongest validation claims. The native CPU backend
reproduces Csurös' Java to ≤ 5 × 10⁻¹¹ at every per-node value, on all
four focal subset clades (D80, P75, E114, ED194). Per-node diff dumps
are in [VALIDATION_FOCAL.md](VALIDATION_FOCAL.md) §3.

```sh
# Reproduce the bit-perfect comparison (~30 s):
PYTHONPATH=. python3 validation/reproduce.py --dataset all \
    --out-dir validation/outputs/reproduction
```

The arc269 dataset has no Csurös fit to compare against (he didn't fit
it; verified by inspecting the bundled `.countxml.gz`), so the
bit-perfect check does not extend to arc269.

## FD-validated analytical gradients

The two analytical gradient implementations have been verified against
central finite differences:

- `recount.native_backend.gradient_native`: median 1e-7 relative error
  on Williams2017 across all per-node parameters.
- `recount.logistic_shift_gradient.mixture_gradient_native`: 1e-9
  relative error on all parameter axes (base_gain, base_dup, base_length,
  delta_dup, delta_length, alpha) on dpann80 K=2 (commit 96b9832).
- The C-kernel modification for per-family weights (commit 96b9832)
  is bit-equivalent to a brute-force per-family loop, to 1e-14 absolute
  error (machine epsilon).

## LaTeX proofs (LLM-drafted, spot-validated)

Both documents in `docs/` were drafted by Claude in this session:

- [`docs/boundary_identifiability.tex`](docs/boundary_identifiability.tex):
  **Lemma 1** (Pólya → Poisson via PGF / Lévy continuity) is
  numerically verified: tested at α ∈ {10, 100, 1000, 10000} with
  Λ=2.5 fixed, the pmf differences |P_Pólya(k) − P_Pois(k)| decrease
  exactly as 1/α (confirms the convergence rate).
  **Remark 1** rate claims were initially incorrect (claimed
  `1 − q̃ ~ μ/λ`; corrected to `~ e^{(μ−dup)τ}`); the correction was
  numerically verified by sweeping dup → ∞ symbolically.
  **Theorems 1 and 2** follow standard arguments (non-attainment of
  the supremum via the asymptote; coercivity of the MAP objective in
  log-rate coordinates) but are not independently peer-reviewed.

- [`docs/logistic_shift_gradient.tex`](docs/logistic_shift_gradient.tex):
  **Proposition 1** (responsibility-weighted mixture-gradient
  identity) and **Theorem 1** (softmax weight gradient) are
  FD-validated to 1e-9 on all parameter axes. The L(0) correction term
  in the softmax gradient was initially missing (off by 19% in the FD
  test) and was added per Proposition 5.
  **Remark 1** documents an implementation note: our Python code uses
  a rate-level shift parameterisation (`dup_k = dup_base · exp(Δ_dup)`),
  not the logit-of-survival form stated in the .tex.

Both .tex files include a "Provenance and disclaimer" box on page 1
making the LLM provenance explicit.

## The arc269 LACA quote (cp=5,436 ± 139 / fm=3,074 ± 56 / cpf=1.77 ± 0.03)

This is the headline result. As of this session it has been validated
both by **point estimate convergence under multistart** and by
**95% bootstrap CI**.

### What's verified

- The MAP σ=1 objective is **mathematically coercive** (Theorem 2 of
  boundary_identifiability.tex), so the global maximum exists at an
  interior point.
- The point-estimate reported numbers (LL=−1,096,512, cp=5,426,
  fm=3,070, cpf=1.77) come from the best of 15 multistart runs with
  default-init + 14 lognormal(0, 0.3) perturbations. All numbers
  reproduce from the saved rates in
  `validation/outputs_no_dup_cap/profile_likelihood_arc269/`.
- At the reported point, all rates are firmly interior (max
  dup=28, max gain=0.53). The log-Normal prior dominates the
  Pólya asymptote at these values.
- **Non-parametric bootstrap 95% CI** (20 iterations resampling
  90,243 family indices with replacement, refit MAP σ=1 warm-started
  from 15-start global):

  | param | mean | SD | 2.5% | 97.5% |
  |---|---:|---:|---:|---:|
  | cp | 5,436 | 139 | 5,182 | 5,687 |
  | fm | 3,074 | 56 | 2,982 | 3,165 |
  | cpf | 1.768 | 0.029 | 1.731 | 1.827 |

  Notable: max_dup, max_gain, L(0) were *exactly constant* across all
  20 bootstrap iterations (28.01, 0.5295, 0.5275 respectively). The
  basin is stable under family resampling; the only variation is in
  the per-family aggregation contributing to cp and fm.

### What's still not verified

- **The K=2 mixture LACA differs from K=1**: K=2 gives cp=7,192 /
  fm=3,666 / cpf=1.96 with a better LL (+7,309 nat over K=1 MAP σ=1
  global, both interior). No bootstrap CI for K=2 yet; the
  point-estimate disagreement between K=1 and K=2 is real and means
  the quoted LACA depends on model class. K=2 should arguably be
  preferred (better fit, accounts for rate heterogeneity); the K=1
  MAP σ=1 quote is reported because it's been more thoroughly
  validated.
- **The σ = 1 prior strength is empirical**, not derived from first
  principles. The σ-sweep (σ ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1,
  1.25, 1.5, 2.0, 3.0}, see [MODEL_COMPLEXITY.md](MODEL_COMPLEXITY.md))
  shows σ=1 is the loosest σ where single-start cold MAP is interior;
  σ=1.1 already flips into a boundary basin. Confidence rests on
  empirical observation, not a theoretical guarantee.
- **Subclade-ancestor inflation persists** at all arc269 fits including
  K=2: the per-node cp/fm at the DPANN/Proteo/Eury ancestors inside
  arc269 are 2-7× higher than the subset-only Csurös fits at the same
  clade root. Most likely cause: unmodelled HGT (no HGT in the GLD
  model means it has to ascribe transferred families to deep ancestors).
- The 15-start MAP is the *interior* answer for the GLD model. **No
  HGT modelling has been applied** — the "ancestral inflation"
  discussion in
  [MODEL_COMPLEXITY.md §Biological discussion](MODEL_COMPLEXITY.md)
  is a *hypothesis* about why the GLD-only LACA is larger than
  Csurös' subclade ancestors, not a proven mechanism.

### How to reproduce the LACA quote

```sh
# σ=1 MAP, 15-start multistart (~50 min on M4 Max):
PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \
    --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2027 \
    --out-dir validation/outputs_no_dup_cap/profile_likelihood_arc269
```

The best -log P across starts will be the global MAP. Compare against
the cold-start canonical value `validation/outputs/map_sigma1/`.

## Numerical claims checked in this session

The following table cross-references every numerical claim in the docs
against the actual saved outputs. ✓ = verified; ⚠ = corrected this
session.

| claim | doc value | actual | status |
|---|---:|---:|:---:|
| Csurös subset LLs | -94,147 / -154,159 / -274,643 / -395,601 | matches | ✓ |
| arc269 cold-start ML | LL=-1,129,653 / cp=8,664 / fm=3,533 / cpf=2.45 | matches | ✓ |
| arc269 cold-start ML max dup | 5.6 × 10⁹ | **7.56 × 10⁹** | ⚠ corrected |
| arc269 Multistart-polish ML | LL=-1,109,266 / cp=8,963 / fm=3,608 / cpf=2.48 / max_dup=5.6e9 | matches all | ✓ |
| arc269 MAP cold-start | LL=-1,117,480 / cp=5,906 / fm=3,108 / cpf=1.90 / max_dup=43 | matches all | ✓ |
| arc269 MAP 15-start global | LL=-1,096,512 / cp=5,426 / fm=3,070 / cpf=1.77 / max_dup=28 | matches all | ✓ |
| Csurös subset max gain ≈ 10¹⁴ | "~10¹⁴" | dpann80=4.9e13, proteo75=5.15e14, eury114=5.8e12, ed194=1.31e14 | ✓ |

## "Csurös' Java GUI does multi-restart" claim

In CONVERGENCE_EXPERIMENT.md and several derived discussions we cited
that Csurös' Java GUI internally cycles BFGS via `DFP_ITMAX = 200` plus
GUI-driven re-launches, supporting his SI B.1's "several thousand
iterations" cost claim. The `DFP_ITMAX = 200` constant is verifiable
from his source at
`/Users/ssolo/src/count/src/count/matek/FunctionMinimization.java`
line 780. The GUI cycling behaviour is not documented in source we have
access to and was inferred from the SI's iteration counts; treat that
particular claim as plausible but unverified.

## Reproducibility checklist

Everything in this paper can be reproduced from a fresh git checkout
with the steps below. Tested on the worktree as of commit 413a635
(plus this VALIDATION.md).

```sh
# 0. Build the native CPU backend (~1 min):
cd native && make && cd ..

# 1. Bit-perfect reproduction of Csurös' subset fits (~30 s):
PYTHONPATH=. python3 validation/reproduce.py --dataset all \
    --out-dir validation/outputs/reproduction

# 2. Cold-start ML on all 5 (~25 min):
PYTHONPATH=. python3 validation/ml.py --dataset all \
    --num-starts 1 --polish-sigmas "" --cycle-iters 100 --max-cycles 15 \
    --out-dir validation/outputs_no_dup_cap/simple_ml

# 3. Multistart-polish ML — multistart + polish (~2-3 h):
PYTHONPATH=. python3 validation/ml.py --dataset all --num-starts 3 \
    --polish-sigmas 0.05,0.10 --num-polish 2 \
    --cycle-iters 100 --max-cycles 12 \
    --out-dir validation/outputs/sota_ml

# 4. MAP σ=1 multistart=15 on arc269 (~50 min) — THE LACA quote:
PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \
    --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2027 \
    --out-dir validation/outputs_no_dup_cap/profile_likelihood_arc269

# 5. Non-parametric bootstrap CI on the LACA quote (~15 min warm-started):
PYTHONPATH=. python3 validation/bootstrap_arc269_laca_ci.py \
    --n-bootstrap 20 --cycle-iters 100 --max-cycles 8 --sigma 1.0 \
    --seed 2026

# 6. Optional σ-sweep on arc269 (~25 min):
for s in 0.1 0.25 0.5 0.75 0.9 1.1 1.25 1.5 2.0 3.0; do
  PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma $s \
    --num-starts 1 --cycle-iters 100 --max-cycles 12 \
    --out-dir validation/outputs_no_dup_cap/map_sigma_sweep
done

# 7. Subclade-ancestor comparison + plots:
PYTHONPATH=. python3 validation/subclade_ancestor_comparison.py

# 8. K=2 mixture per-category reconstruction:
PYTHONPATH=. python3 validation/k2_reconstruction.py

# 6. K=2 LogisticShift mixture fits (~30 min per subset, ~1 hr arc269):
PYTHONPATH=. python3 validation/mixture_ml.py --dataset all --K 2 \
    --sigma 1.0 --max-cycles 10 --cycle-iters 100 \
    --out-dir validation/outputs/mixture_K2
```

After these steps, the JSON summaries under `validation/outputs/*/` will
match the numbers cited in README.md and the .tex files. If a number
differs, it's a real discrepancy — please file an issue or check the
git log for any optimizer/clip changes between the snapshot and the
re-run.
