# `validation/outputs/` — BOUNDED fits (production)

All fits in this directory are produced with the **sub-critical Yule
constraint** (`dup ≤ MAX_PROB_NOT1 = 1 − 2⁻³⁰`, equivalent to Csurös'
Java logit parameterization) plus the hard log-cap `log γ ≤ 33` on the
gain rate. These are the **production results** used for every
post-2026-05-18 plot, CSV export and downstream paper claim.

Pre-cap (UNBOUNDED) snapshots are preserved in a sibling tree at
[`../outputs_no_dup_cap/`](../outputs_no_dup_cap/) for the historical
comparison and the basin-identifiability discussion.

## The six datasets

| label | clade | F (families) | leaves | nodes | Ωmin |
|---|---|---:|---:|---:|---:|
| `arc269`   | full Archaea (LACA)              | 90,243 | 269 | 537 | 1 |
| `dpann80`  | DPANN superphylum (Nanobdellati)  |  3,034 |  80 | 159 | 4 |
| `proteo75` | Proteoarchaea (TACK)              |  5,179 |  75 | 149 | 4 |
| `eury114`  | Methanobacteriati / basal Eury    |  7,335 | 114 | 227 | 4 |
| `ed194`    | Euryarchaeota (LECA-ish)          |  8,855 | 194 | 387 | 4 |
| `williams` | Williams 2017 archaeal-bacterial ALE-trimmed (`wsz60-aletrim-min4.txt`) |  5,378 |  60 | 119 | 4 |

## What lives where — each subdir maps to one producing script

Naming convention for every artifact: `{dataset}_{tag}.{branches.csv,
countxml.gz}` plus `{dataset}_summary.json` and
`{dataset}_final_rates.npz`. The `_per_family_FxN.npz` /
`_per_family_summary.csv` companions are produced post-hoc by
[`validation/per_family_presence.py --all --fit-dir <subdir>`](../per_family_presence.py).

### ML (no prior)

| subdir | producing script | command flags | datasets covered |
|---|---|---|---|
| `bounded_csuros/`   | [`validation/ml.py`](../ml.py)   | `--num-starts 1 --polish-sigmas "" --cycle-iters 100 --max-cycles 12 --optimizer native_bfgs` | arc269 (queue) |
| `csuros_match/`     | [`validation/ml.py`](../ml.py)   | same Csurös recipe, all 5 subsets | dpann80, proteo75, eury114, ed194, williams |
| `sota_ml/`          | [`validation/ml.py`](../ml.py)   | `--num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2 --cycle-iters 100 --max-cycles 12` | subsets (arc269 sits in `bounded_csuros/` instead) |
| `arc269_seeded_ml/` | [`validation/arc269_ml_from_subset_seed.py`](../arc269_ml_from_subset_seed.py) | arc269 ML cold-restarted from cross-subset median rates seed (see `arc269_seed_from_subsets_v2.npz`); lands in cold basin (~LL=-1,138k) | arc269 |

### MAP (with prior)

| subdir | producing script | command flags | datasets covered |
|---|---|---|---|
| `map_sigma1/`                | [`validation/map.py`](../map.py) | `--sigma 1.0 --num-starts 1` | all subsets + arc269 (cold start) |
| `brownian_*/`                | [`validation/map.py`](../map.py) | `--prior brownian --sigma-brownian-gain 1.0 --sigma-brownian-dup 1.0 --sigma-brownian-length 1.0` | one dir per dataset — tree-Brownian autocorrelated log-rate prior, see [`docs/brownian_prior.tex`](../../docs/brownian_prior.tex) |

The 15-start global MAP for arc269 (the headline LACA quote
5,426 cp / 3,070 fm / cpf=1.77) lives in
[`../outputs_no_dup_cap/profile_likelihood_arc269/`](../outputs_no_dup_cap/profile_likelihood_arc269/)
— it was produced before the sub-critical Yule cap was made the default.
The bounded 15-start has not been re-run; the cold-start MAP in
`map_sigma1/arc269_*` is the bounded analogue available here.

### Mixture (K-component LogisticShift, per-family categories)

| subdir | producing script | command flags |
|---|---|---|
| `mixture_K2/` | [`validation/mixture_ml.py`](../mixture_ml.py) | `--K 2` |

The K=3 mixture fits live in
[`../outputs_no_dup_cap/mixture_K3/`](../outputs_no_dup_cap/mixture_K3/)
(unconstrained variant); the bounded K=3 has not been re-run. A recent
arc269 K=3 shift-only fit (LogisticShift mixture without re-fitting
base rates) lives in
[`arc269_shift_only_K3/`](arc269_shift_only_K3/).

### Reference / non-fit

| subdir | producing script | what it is |
|---|---|---|
| `reproduction/`             | [`validation/reproduce.py`](../reproduce.py) | Csurös' published rates re-loaded and re-evaluated by our native engine. Subsets only (Csurös never fit arc269). Bit-perfect agreement at every per-node value. |
| `bootstrap_arc269_bounded/` | [`validation/bootstrap_arc269_laca_ci.py`](../bootstrap_arc269_laca_ci.py) | Parametric bootstrap of arc269 LACA reconstruction at σ=1 MAP, sub-critical cap respected. 20 replicates → `bootstrap_summary.json` with cp/fm/cpf 95% CIs. |

## Cross-cutting outputs (top-level files)

These are produced by single scripts that read from multiple subdirs:

| file | producing script | what it shows |
|---|---|---|
| `subclade_ancestor_comparison.json`    | [`validation/subclade_ancestor_comparison.py --variant bounded`](../subclade_ancestor_comparison.py) | machine-readable per-subclade summary (cp/fm at each subset ancestor across all arc269 fits + Csurös subset-only). |
| `arc269_seed_from_subsets_v2.npz`      | [`validation/diagnose_arc269_vs_subsets.py`](../diagnose_arc269_vs_subsets.py) | gain/loss/dup arrays for arc269 built from per-subset MAP rates (median across subsets that map to each arc269 node via LCA). Used as init for `arc269_seeded_ml/`. |

## Standard per-fit artifacts (every subdir)

- `{dataset}_{tag}.countxml.gz` — Csurös' XML session format, loadable in the Count Java GUI.
- `{dataset}_{tag}.branches.csv` — per-node summary: gain/loss/dup rates, branch length, posterior copies (observed and L(0)-corrected), gain/loss events, posterior families present.
- `{dataset}_summary.json` — headline: LL, L(0), root copies/families (corrected), cp/fam, and `fit_settings` block recording every CLI parameter.
- `{dataset}_final_rates.npz` — raw rates (`gain`, `loss`, `dup`, `length` arrays).
- `{dataset}_trajectory.json` — per-cycle BFGS state (obj, |g|, iters, wall).
- `{dataset}_per_family_FxN.npz` + `{dataset}_per_family_summary.csv` — per-family per-node `P{ξ_v≥1|f}` and `E[ξ_v|f]` tables, written by `per_family_presence.py` after each fit.

## Sub-critical Yule + log-gain cap — what changed

| constraint | enforced where | value | reason |
|---|---|---|---|
| `dup ≤ 1 − 2⁻³⁰` | `_logit_to_dup` in [`validation/_shared.py`](../_shared.py) | `MAX_PROB_NOT1 ≈ 0.99999999907` | Csurös' Pólya κ asymptote: dup > 1 makes the Yule process super-critical → divergent posteriors. The bounded fits use a logit-of-MAX parameterization so the optimizer can approach but never cross the boundary. |
| `\|log_rate\| ≤ 33` | clip on `x = log_*` in `make_objgrad_*` | `LOG_RATE_CLIP = 33` | Csurös' `MAX_GAIN_RATE`. Bounds the gradient when a rate is being driven to inf. |
| Same on `length`, `loss` | same `LOG_RATE_CLIP` | `\|log\| ≤ 33` | uniform cap on all rate axes |

The historical pre-cap snapshots in `../outputs_no_dup_cap/` were
produced by the same scripts with `--no-subcritical` (and
`LOG_RATE_CLIP = 50` at the time). They are kept only for the
historical comparison plots and the basin-identifiability discussion in
[`../../docs/boundary_identifiability.pdf`](../../docs/boundary_identifiability.pdf).

## Cleanup history

The bounded campaign consolidated this directory in 2026-05-19. The
following pre-cap (unbounded) artifacts were moved out and are
preserved in [`../outputs_no_dup_cap/`](../outputs_no_dup_cap/):

- `arc269_omin4/`     — pre-cap arc269 Ωmin=4 explorations
- `bootstrap_arc269/` — pre-cap bootstrap (the bounded counterpart is
  `bootstrap_arc269_bounded/`)
- `map_sigma_sweep/`  — pre-cap σ-sweep
- `profile_likelihood_arc269/` — pre-cap 15-start MAP global

Stale debug snapshots (`map_sigma1/*.stale`, `sota_ml/arc269_*.stale`,
`profile_likelihood_arc269/*.unbounded_stale`) have been deleted.

## Plot pipeline (after any new fit)

```sh
# 1. Per-family per-node posterior tables for every fresh fit dir:
PYTHONPATH=. python3 validation/per_family_presence.py --all --fit-dir <subdir>

# 2. Per-subclade ancestor comparison summary (bounded variant):
PYTHONPATH=. python3 validation/subclade_ancestor_comparison.py --variant bounded
```
