"""Example 11 — full reconc + arc269 sweep, every native count run.

Runs ``recount analyze`` on every primary count table from Csurös'
public ``genomes-from-gene-counts-datasets`` distribution: the four
``reconc/`` datasets (real biological data from prior published Count
analyses) plus the five ``arc269/`` datasets from the Csurös 2026
PNAS focal analysis, including the headline 269-leaf / 90,243-family
arc269 table itself.

Per dataset we print:
  • a one-line biological description,
  • the command used,
  • a phase-by-phase timing breakdown (Python import, tree+table load,
    GLD ML fit, per-branch + per-family posteriors, CSV/CountXML write),
  • the fitted log-likelihood, L(0), and the reconstructed root genome.

The three output files per dataset (``<prefix>.countxml.gz``,
``<prefix>.branches.csv``, ``<prefix>.families.csv``) land in
``/tmp/recount_focal_sweep/`` and use the same conventions every
``recount analyze`` invocation produces — see SUMMARY.md for the
column dictionary.

Defaults used here:
  --min-copies AUTO         (auto-detect per dataset from the table's
                              minimum profile sum: e.g. tables filtered
                              to ≥4 copies use Ωmin=4, the arc269 raw
                              table with singletons uses Ωmin=1. Most
                              focal tables here are *-min4 → Ωmin=4
                              automatically. Pass an explicit integer
                              to override uniformly.)
  --max-iter 30             (L-BFGS-B step cap — `converged=False` in
                              the output just means we hit this cap and
                              are still improving; bump for production)
  --posterior-threshold 0.5 (suppress per-family CSV rows where
                              E[copies] AND P(present) are both < 0.5;
                              keeps families.csv tractable on the
                              90k-family arc269 — drop to 1e-3 for the
                              full per-family posterior matrix)

Wall-clock measured on M4 Max (12 P + 4 E = 16 cores; libdispatch
across all online cores; native CPU backend with vForce SIMD):

  ┌────────────┬────────┬─────┬──────┬───────────┐
  │ Dataset    │     F  │  N  │ maxW │ Wall (s)  │
  ├────────────┼────────┼─────┼──────┼───────────┤
  │ williams   │  5,378 │ 119 │  668 │      3.2  │
  │ wsz62      │  5,498 │ 123 │  669 │      3.3  │
  │ coleman    │ 11,272 │ 529 │ 1097 │     18.4  │
  │ harris2022 │ 20,822 │  59 │  479 │     12.6  │
  │ arc269     │ 90,243 │ 537 │ 1044 │     48.7  │ ← HEADLINE
  │ ed194      │  8,855 │ 387 │  764 │     10.2  │
  │ eury114    │  7,335 │ 227 │  712 │      6.8  │
  │ proteo75   │  5,179 │ 149 │  464 │      3.8  │
  │ dpann80    │  3,034 │ 159 │  201 │      3.0  │
  └────────────┴────────┴─────┴──────┴───────────┘
  Total: 153,816 families across 9 datasets in ~ 110 s wall-clock.

A note on the reconstructed root genome sizes:

  The numbers below are from COLD-START fits (uniform initial rates,
  --max-iter 30) at the auto-detected Ωmin. They land near a local
  optimum of the corrected log-likelihood, but the GLD inverse problem
  has multiple near-equivalent local optima, so the ancestral genome
  sizes can differ ~10-30 % from Csurös' published values even when
  the LLs agree to ~20 nats (i.e. < 0.005 nats/family).

  For bit-perfect agreement with Csurös' Java Count using HIS already-
  fitted rates, see example 07 (focal validation) and example 06 (root
  copies/families): every quantity matches to ≤ 5e-11 absolute when we
  feed his stored rates into the native pipeline.

  This sweep is about runnability and per-dataset timing, not about
  reproducing the paper's specific ancestral-genome point estimates —
  for that, warm-start the fit from his rates or run to full
  convergence with a much larger --max-iter.

Usage:
  PYTHONPATH=. python3 examples/11_full_focal_sweep.py
  PYTHONPATH=. python3 examples/11_full_focal_sweep.py --datasets arc269
  PYTHONPATH=. python3 examples/11_full_focal_sweep.py --min-copies 4
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


# (label, one-line biological description)
DATASETS = [
    ("williams",   "Williams 2017 — 60-leaf archaeal phylogeny"
                   " (5,378 families; LCA = LACA)"),
    ("wsz62",      "Williams 2017 — 62-leaf variant of the same dataset"),
    ("coleman",    "Coleman 2021 — 265-leaf bacterial phylogeny"
                   " (11,272 families; LCA = LBCA)"),
    ("harris2022", "Harris 2022 — 30-leaf bryophyte phylogeny"
                   " (20,822 plant gene families)"),
    ("arc269",     "Csurös 2026 PNAS focal — 269-leaf full archaeal phylogeny"
                   " (90,243 families; headline reconstruction)"),
    ("ed194",      "Csurös 2026 focal — 194-leaf Euryarchaeota subset"),
    ("eury114",    "Csurös 2026 focal — 114-leaf basal-archaea subset"),
    ("proteo75",   "Csurös 2026 focal —  75-leaf proteo-archaea subset"),
    ("dpann80",    "Csurös 2026 focal —  80-leaf DPANN superphylum"
                   " (root = Nanobdellati ancestor)"),
]

WORK_DIR = Path("/tmp/recount_focal_sweep")
DATA_DIR = Path("examples/data")


def _parse_stderr(text):
    """Pull per-phase timings + key numbers out of the CLI's stderr."""
    metrics = {"fit_s": None, "post_s": None, "ll": None,
               "l0": None, "root_copies": None, "root_families": None,
               "topo": None, "F": None, "maxW": None,
               "branches_rows": None, "families_rows": None,
               "countxml_bytes": None}
    for line in text.splitlines():
        m = re.match(r"#\s+tree:\s+(\d+)\s+leaves,\s+(\d+)\s+nodes", line)
        if m:
            metrics["topo"] = f"{m.group(1)} leaves / {m.group(2)} nodes"
        m = re.match(r"#\s+table:\s+(\d+)\s+families,\s+(?:max profile sum = (\d+)"
                     r"|profile sum range \[(\d+),\s*(\d+)\])", line)
        if m:
            metrics["F"] = int(m.group(1))
            metrics["maxW"] = int(m.group(2) or m.group(4))
            if m.group(3):
                metrics["minW"] = int(m.group(3))
        m = re.search(r"auto-detected --min-copies = (\d+)", line)
        if m:
            metrics["mc_auto"] = int(m.group(1))
        m = re.search(r"fitting at .min=(\d+),", line)
        if m:
            metrics["mc"] = int(m.group(1))
        m = re.search(r"fit done:.*time=([\d.]+)s", line)
        if m:
            metrics["fit_s"] = float(m.group(1))
        m = re.search(r"posteriors done in ([\d.]+)s", line)
        if m:
            metrics["post_s"] = float(m.group(1))
        m = re.search(r"Fitted LL.*?:\s*(-?\d+\.\d+)", line)
        if m:
            metrics["ll"] = float(m.group(1))
        m = re.search(r"L\(0\) = (\d+\.\d+)", line)
        if m:
            metrics["l0"] = float(m.group(1))
        m = re.search(r"copies = ([\d.]+), families_present = ([\d.]+)", line)
        if m:
            metrics["root_copies"] = float(m.group(1))
            metrics["root_families"] = float(m.group(2))
    return metrics


def _count_rows(path):
    try:
        with open(path) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except FileNotFoundError:
        return 0


def run_one(label, biology, args):
    tree = DATA_DIR / f"{label}_tree.nwk"
    table = DATA_DIR / f"{label}_table.csv.gz"
    out_prefix = WORK_DIR / label
    cmd = [
        sys.executable, "-m", "recount.cli", "analyze",
        str(tree), str(table),
        "--out-prefix", str(out_prefix),
        "--max-iter", str(args.max_iter),
        "--posterior-threshold", str(args.posterior_threshold),
    ]
    if args.min_copies >= 0:
        cmd += ["--min-copies", str(args.min_copies)]
    # else: CLI auto-detects --min-copies from the table's min profile sum.
    print()
    print("=" * 72)
    print(f"  {label}: {biology}")
    print("=" * 72)
    print(f"  Command:")
    mc_part = f"--min-copies {args.min_copies} " if args.min_copies >= 0 else "(auto-detect --min-copies) "
    print(f"    python3 -m recount.cli analyze \\")
    print(f"        examples/data/{label}_tree.nwk \\")
    print(f"        examples/data/{label}_table.csv.gz \\")
    print(f"        --out-prefix /tmp/recount_focal_sweep/{label} \\")
    print(f"        {mc_part}--max-iter {args.max_iter} "
          f"--posterior-threshold {args.posterior_threshold}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    m = _parse_stderr(r.stderr + "\n" + r.stdout)
    m["wall_s"] = wall
    m["label"] = label

    # Phase breakdown
    fit_s = m["fit_s"] or 0.0
    post_s = m["post_s"] or 0.0
    other_s = max(0.0, wall - fit_s - post_s)  # load + write + Python startup

    # File sizes
    branches_p = out_prefix.with_suffix(".branches.csv")
    families_p = out_prefix.with_suffix(".families.csv")
    countxml_p = Path(f"{out_prefix}.countxml.gz")
    m["branches_rows"] = _count_rows(branches_p) if branches_p.exists() else 0
    m["families_rows"] = _count_rows(families_p) if families_p.exists() else 0
    m["countxml_bytes"] = countxml_p.stat().st_size if countxml_p.exists() else 0

    mc_chosen = m.get("mc") or m.get("mc_auto") or args.min_copies
    auto_note = " (auto-detected)" if m.get("mc_auto") else ""
    print(f"\n  Tree + table: {m['topo']}, F={m['F']}, maxW={m['maxW']}")
    print(f"  Conditioning: Ωmin = {mc_chosen}{auto_note}")
    m["mc"] = mc_chosen
    if r.returncode != 0:
        print(f"  ⚠ FAILED (rc={r.returncode}):")
        for ln in (r.stderr.splitlines()[-5:]):
            print(f"    {ln}")
        return m

    print(f"\n  Timing (wall {wall:.1f} s):")
    print(f"    Python startup + tree/table load + write : {other_s:>6.1f} s")
    print(f"    L-BFGS-B fit (max_iter={args.max_iter})            : {fit_s:>6.1f} s")
    print(f"    per-branch + per-family posteriors        : {post_s:>6.1f} s")

    print(f"\n  Fit result:")
    if m["ll"] is not None:
        print(f"    Corrected log-likelihood  : {m['ll']:>14.2f}")
    if m["l0"] is not None:
        print(f"    L(0) (unobserved-profile) : {m['l0']:>14.4f}")
    if m["root_copies"] is not None:
        print(f"    Root copies (corrected)   : {m['root_copies']:>14.1f}")
        print(f"    Root families present     : {m['root_families']:>14.1f}")

    print(f"\n  Output files (in /tmp/recount_focal_sweep/):")
    print(f"    {label}.countxml.gz   {m['countxml_bytes']/1024:>6.0f} KB"
          " — Java-readable session XML")
    print(f"    {label}.branches.csv  {m['branches_rows']:>6} rows"
          " — per-node summary (rates, copies, events, families)")
    print(f"    {label}.families.csv  {m['families_rows']:>6} rows"
          " — per-family per-node E[copies] + P(present),"
          f" threshold={args.posterior_threshold}")
    return m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of dataset labels (default: all 9)")
    parser.add_argument("--min-copies", type=int, default=-1,
                        help="Ωmin conditioning. Default -1 = auto-detect per dataset "
                             "from the table's minimum profile sum (the right setting; "
                             "most focal tables are pre-filtered to min sum 4). Pass an "
                             "explicit integer to override every dataset uniformly.")
    parser.add_argument("--max-iter", type=int, default=30,
                        help="L-BFGS-B step cap per fit (default 30)")
    parser.add_argument("--posterior-threshold", type=float, default=0.5,
                        help="suppress per-family rows where E[copies] AND P(present)"
                             " < threshold (default 0.5; set 1e-3 for the full per-family"
                             " posterior matrix)")
    args = parser.parse_args(argv)

    WORK_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print(" recount analyze — full sweep across reconc/ + arc269/ datasets")
    print("=" * 72)
    print()
    print(f" Defaults:")
    print(f"   --min-copies          {args.min_copies}")
    print(f"   --max-iter            {args.max_iter}")
    print(f"   --posterior-threshold {args.posterior_threshold}")
    print()
    print(f" Each fit runs the analytical native pipeline:")
    print(f"   1. Tree + count-table load (recount.io.newick + recount.io.table)")
    print(f"   2. Survival-parameter compute + GLD ML fit (L-BFGS-B over the")
    print(f"      threaded native C analytical gradient, libdispatch across cores)")
    print(f"   3. Per-branch posteriors + per-family per-node posteriors via the")
    print(f"      same inside/outside pipeline")
    print(f"   4. Write the CountXML + two CSVs")

    if args.datasets:
        wanted = set(args.datasets.split(","))
        cases = [(l, b) for l, b in DATASETS if l in wanted]
    else:
        cases = list(DATASETS)

    t_total = time.time()
    summaries = []
    for label, biology in cases:
        s = run_one(label, biology, args)
        summaries.append(s)
    total_wall = time.time() - t_total

    print()
    print("=" * 72)
    print(f" SUMMARY — {len(summaries)} datasets in {total_wall:.1f} s total wall-clock")
    print("=" * 72)
    print(f"{'dataset':<12} {'F':>6} {'maxW':>5} {'Ωmin':>5} {'wall (s)':>9}"
          f"  {'fit LL':>15} {'L(0)':>6} {'root copies':>12}  {'root fams':>10}")
    print("-" * 105)
    sum_F = 0
    for s in summaries:
        sum_F += s.get("F", 0) or 0
        F_s   = f"{s.get('F'):>6}"   if s.get('F') is not None else "     ?"
        mw_s  = f"{s.get('maxW'):>5}" if s.get('maxW') is not None else "    ?"
        mc_s  = f"{s.get('mc'):>5}"   if s.get('mc') is not None else "    ?"
        ll_s  = f"{s['ll']:>15.1f}"            if s["ll"]            is not None else " " * 15
        l0_s  = f"{s['l0']:>6.4f}"             if s["l0"]            is not None else " " * 6
        rc_s  = f"{s['root_copies']:>12.1f}"   if s["root_copies"]   is not None else " " * 12
        rf_s  = f"{s['root_families']:>10.1f}" if s["root_families"] is not None else " " * 10
        print(f"{s['label']:<12} {F_s} {mw_s} {mc_s} {s['wall_s']:>9.1f}  "
              f"{ll_s} {l0_s} {rc_s}  {rf_s}")
    print("-" * 100)
    print(f"{'TOTAL':<12} {sum_F:>6} families across {len(summaries)} datasets,"
          f" {total_wall:.1f} s wall.")

    print()
    print(" Reading the outputs in pandas:")
    print("   import pandas as pd")
    print("   b = pd.read_csv('/tmp/recount_focal_sweep/arc269.branches.csv')")
    print("   b.query('not is_leaf').nlargest(10, 'copies_node_corrected')")
    print("   # → top-10 ancestral nodes by reconstructed genome size across")
    print("   #   the full 269-leaf archaeal phylogeny.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
