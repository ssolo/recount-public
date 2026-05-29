# recount

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20388465.svg)](https://doi.org/10.5281/zenodo.20388465)

A C / Python implementation of the gain–loss–duplication (GLD)
phylogenetic likelihood for ancestral gene content reconstruction
(Csűrös, *PNAS* 2026; [doi:10.1073/pnas.2537812123](https://doi.org/10.1073/pnas.2537812123)).
This archive accompanies the Szöllősi & Williams (2026) PNAS Letter
to the Editor.

## Install

```bash
git clone https://github.com/ssolo/recount-public.git
cd recount-public
pip install -e .
```

The native C backend (`recount.native_backend`) builds automatically
on first import via `make -C native`. macOS uses Accelerate +
libdispatch; Linux uses OpenMP. Set `RECOUNT_NO_AUTOBUILD=1` to
disable.

## CLI

Two subcommands. Both default to the Brownian-MAP (tree-autocorrelated
log-rate) prior and the native backend.

### `recount fit` — fit GLD rates by MAP / ML on a Count-format input

```bash
recount fit data.countxml.gz --out-json fit.json
```

Useful flags:

| flag | meaning |
|---|---|
| `--prior {brownian,none}`        | `brownian` = MAP fit (default); `none` = plain ML |
| `--sigma-brownian σ`             | prior strength; smaller = stronger shrinkage (default 1.0) |
| `--soft-landing`                 | annealed Brownian pipeline for large/noisy data (≥ several hundred taxa) |
| `--sigma-schedule σ₁,σ₂,…`       | σ schedule when `--soft-landing` is set (default `0.05,0.15,0.3,0.6,1.0`) |
| `--min-copies N`                 | observation threshold Ωmin (default 1) |
| `--backend {native,torch,numpy}` | gradient backend (default `native`) |
| `--num-threads N`                | thread count for the native backend |

### `recount analyze` — one-shot fit + per-branch posteriors

```bash
recount analyze tree.nwk table.csv \
    --out-prefix /tmp/run --min-copies 4
```

Writes `<prefix>.countxml.gz`, `<prefix>.branches.csv`, and
`<prefix>.families.csv`. Accepts the same prior / backend / soft-landing
flags as `fit`. `--min-copies -1` auto-detects.

`recount fit --help` and `recount analyze --help` list all flags.

## Reproducing the PNAS Letter figure

The figure-generation script reads canonical-campaign outputs and
the archaeal input data shipped under `docs/csuros_data/arc269/`:

```bash
python validation/pnas_letter_v2_figure_alt.py
```

## Layout

```
recount/            Python library (CLI, ml.py, gld.py, native bindings)
native/             C backend (Accelerate on macOS / OpenMP on Linux)
tests/              pytest test suite
validation/         reproduction scripts incl. pnas_letter_v2_figure_alt.py
examples/           small example scripts + data
tools/              utility scripts
docs/csuros_data/   Csűrös' published archaeal + reconciliation inputs
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff). DOI: [10.5281/zenodo.20388465](https://doi.org/10.5281/zenodo.20388465).
