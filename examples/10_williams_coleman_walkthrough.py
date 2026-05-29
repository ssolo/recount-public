"""Example 10 — Williams2017 + Coleman2021 end-to-end walkthrough.

A didactic tour of the ``recount analyze`` one-shot CLI using two of the
focal datasets from the Csurös 2026 PNAS paper: Williams2017 (60-leaf
archaeal tree) and Coleman2021 (265-leaf bacterial tree). For each
dataset we (1) fit GLD rates, (2) compute per-branch posterior events
+ ancestral genome size, and (3) compute per-family per-node posteriors,
all from a bare Newick tree + a CSV count table.

Input data (extracted once, committed to examples/data/):
  examples/data/williams_tree.nwk      Newick, 60 leaves / 119 nodes
  examples/data/williams_table.csv.gz  5,378 families × 60 leaves
  examples/data/coleman_tree.nwk       Newick, 265 leaves / 529 nodes
  examples/data/coleman_table.csv.gz   11,272 families × 265 leaves

Both were extracted from the focal countxml files Csurös distributes in
``genomes-from-gene-counts-datasets/reconc/{Williams2017,Coleman2021}.countxml.gz``
using the helper at the bottom of this file (``extract_from_countxml``).
The extraction strips the rate model + format metadata from the bundle
and writes plain Newick + CSV — the same input format anyone with a
species tree + family count table can produce. CSV here is just
tab-to-comma rewriting; the loader in ``recount.io.table`` autodetects
either delimiter.

What the analyze command does, in order, for one dataset:

  1. Loads the tree and the count table (with leaf-name reordering).
  2. Fits GLD rates by L-BFGS-B over the analytical native gradient
     (gain, loss=fixed-at-1, dup, length per edge), with the biological
     dup ≤ loss constraint, the root edge held at +∞, and
     ``--min-copies 1`` conditioning (i.e. condition on family-absent —
     the empty profile is excluded from the likelihood, matching the
     typical "we only see families with at least one copy somewhere"
     observation model).
  3. Computes per-branch posteriors at the fitted rates via the threaded
     inside/outside pipeline (Apple libdispatch across all P+E cores).
  4. Computes per-family per-node posteriors via the same pipeline,
     emitting E[copies at node | family] and P(family present at node)
     for every (family, node) pair.
  5. Writes three output files:
        <prefix>.countxml.gz   — Java-compatible session XML (loads
                                  directly into Csurös' CountXXV.jar
                                  and round-trips through
                                  recount.io.countxml.load_countxml).
        <prefix>.branches.csv  — per-node summary (one row per node,
                                  including leaves and root).
        <prefix>.families.csv  — per-family per-node E[copies] and
                                  P(present), thresholded by
                                  --posterior-threshold (default 1e-3)
                                  to keep file sizes tractable.

Reproducible measured timing on M4 Max (12 P + 4 E = 16 cores):
  Williams2017  fit 7.6 s + posteriors 0.0 s ≈ 8 s wall (50 BFGS iters)
  Coleman2021   fit 14.4 s + posteriors 0.6 s ≈ 18 s wall (30 BFGS iters)

Both fits are intentionally capped to a small iteration budget here so
the walkthrough runs in well under a minute; bump --max-iter for a
production fit.

----------------------------------------------------------------------
Reading the output CSVs in Python / pandas:

    import pandas as pd
    branches = pd.read_csv("results.branches.csv")
    branches.query("not is_leaf").nlargest(10, "copies_node_corrected")
    # → the 10 ancestral nodes with the largest reconstructed genome

    families = pd.read_csv("results.families.csv")
    families.query("family_name == 'fam001' and P_present > 0.5")
    # → all nodes (incl. internal ancestors) where family fam001 is
    #   inferred present with > 50% posterior probability

The branches.csv "*_corrected" columns add the unobserved-profile
contribution: at Ωmin = 1 the empty profile contributes E[copies] = 0
at every node so observed = corrected; at Ωmin ≥ 2 they differ.
----------------------------------------------------------------------

Usage:
  PYTHONPATH=. python3 examples/10_williams_coleman_walkthrough.py

Equivalent direct invocations:
  PYTHONPATH=. python3 -m recount.cli analyze \\
      examples/data/williams_tree.nwk examples/data/williams_table.csv.gz \\
      --out-prefix /tmp/williams --min-copies 1 --max-iter 50

  PYTHONPATH=. python3 -m recount.cli analyze \\
      examples/data/coleman_tree.nwk examples/data/coleman_table.csv.gz \\
      --out-prefix /tmp/coleman --min-copies 1 --max-iter 30
"""
from __future__ import annotations

import gzip
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


WORK_DIR = Path("/tmp/walkthrough_out")
DATA_DIR = Path("examples/data")
CASES = [
    {"label": "Williams2017", "key": "williams", "max_iter": 50,
     "biology": "60-leaf archaeal tree; LCA = LACA (Last Archaeal Common Ancestor)"},
    {"label": "Coleman2021",  "key": "coleman",  "max_iter": 30,
     "biology": "265-leaf bacterial tree; LCA = LBCA (Last Bacterial Common Ancestor)"},
]


def run_analyze(case):
    """Invoke `recount analyze` as a subprocess and capture the timing."""
    out_prefix = WORK_DIR / case["key"]
    tree = DATA_DIR / f"{case['key']}_tree.nwk"
    table = DATA_DIR / f"{case['key']}_table.csv.gz"
    cmd = [
        sys.executable, "-m", "recount.cli", "analyze",
        str(tree), str(table),
        "--out-prefix", str(out_prefix),
        "--min-copies", "1",
        "--max-iter", str(case["max_iter"]),
    ]
    print(f"\n#### {case['label']}: {case['biology']}")
    print(f"#### {' '.join(cmd)}")
    import time
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"#### wall-clock: {time.time() - t0:.1f} s")
    return out_prefix


def summarize(out_prefix, label):
    """Show the human-readable bits of the output for the walkthrough."""
    branches_path = out_prefix.with_suffix(".branches.csv")
    families_path = out_prefix.with_suffix(".families.csv")

    branches = branches_path.read_text().splitlines()
    header = branches[0].split(",")
    body = [ln.split(",") for ln in branches[1:]]

    print(f"\n  ── {branches_path.name} — {len(body)} rows × {len(header)} cols ──")
    # Top-3 leaves by observed copies (= leaf-column sums from the input)
    leaf_rows = [r for r in body if r[2] == "True"]
    leaf_rows.sort(key=lambda r: -float(r[9]))
    print(f"    Top-3 leaves by observed copy count:")
    for r in leaf_rows[:3]:
        print(f"      {r[1]:<14s}  copies={float(r[9]):>8.0f}  families_present={float(r[14]):>6.0f}")

    # Root row
    root_row = next((r for r in body if r[0] == str(int(np.max([int(r[0]) for r in body])))), None)
    if root_row:
        print(f"    Root (node {root_row[0]}, '{root_row[1]}'):")
        print(f"      gain_rate           = {float(root_row[6]):.4f}")
        print(f"      dup_rate            = {float(root_row[8]):.4f}")
        print(f"      copies_node_corr    = {float(root_row[10]):>8.1f}  (= reconstructed ancestral genome size)")
        print(f"      families_present_corr = {float(root_row[15]):>8.1f}")

    # Top-3 internal nodes by reconstructed genome size
    int_rows = [r for r in body if r[2] == "False"]
    int_rows.sort(key=lambda r: -float(r[10]))
    print(f"    Top-3 internal nodes by ancestral genome size (corrected copies):")
    for r in int_rows[:3]:
        print(f"      {r[1]:<14s}  copies_corr={float(r[10]):>8.1f}  gains={float(r[12]):>7.1f}  losses={float(r[13]):>7.1f}")

    # families.csv stats
    with open(families_path) as fh:
        n_rows = sum(1 for _ in fh) - 1
    print(f"\n  ── {families_path.name} — {n_rows} rows (filtered by --posterior-threshold) ──")
    with open(families_path) as fh:
        head = [next(fh) for _ in range(5)]
    print("    " + head[0].rstrip())
    for ln in head[1:]:
        print("      " + ln.rstrip())


def extract_from_countxml(
    countxml_path, session_id, table_name, out_prefix
):
    """One-shot helper: pull a bare Newick + CSV out of a Csurös
    session-bundle countxml file.

    Used once to populate examples/data/. Kept in this file as a record
    of how those files were produced.
    """
    d = gzip.open(countxml_path).read().decode()
    m_sess = re.search(rf'<session id="{re.escape(session_id)}"[^>]*>(.*?)</session>',
                       d, re.DOTALL)
    body = m_sess.group(1)
    nw = re.search(r'<tree[^>]*>\s*<!\[CDATA\[\s*(.*?)\s*\]\]>', body, re.DOTALL).group(1).strip()
    if not nw.endswith(";"):
        nw += ";"
    tbl = re.search(rf'<table[^>]*name="{re.escape(table_name)}"[^>]*>\s*<!\[CDATA\[\s*(.*?)\s*\]\]>',
                    body, re.DOTALL).group(1).strip()
    csv_text = "\n".join(",".join(ln.split("\t")) for ln in tbl.splitlines())
    Path(out_prefix + "_tree.nwk").write_text(nw + "\n")
    with gzip.open(out_prefix + "_table.csv.gz", "wt") as fh:
        fh.write(csv_text + "\n")


def main() -> int:
    WORK_DIR.mkdir(exist_ok=True)
    print("================================================================")
    print(" recount analyze — Williams2017 + Coleman2021 walkthrough")
    print("================================================================")
    print()
    print("Defaults used for both fits:")
    print("  * loss rate fixed at 1.0 (Williams / arc269 / Csurös 2026 convention)")
    print("  * root edge length fixed at +∞")
    print("  * dup ≤ loss biological bound (rate-variation off)")
    print("  * --min-copies 1 (condition on family absent)")
    print("  * native CPU backend (libdispatch across all P+E cores)")
    for case in CASES:
        out_prefix = run_analyze(case)
        summarize(out_prefix, case["label"])

    print("\n================================================================")
    print(" Files produced (per dataset, replace 'X' with williams or coleman):")
    print("================================================================")
    print(f"  {WORK_DIR}/X.countxml.gz   — Java-compatible session XML")
    print(f"  {WORK_DIR}/X.branches.csv  — per-node summary (rates + events + copies + families)")
    print(f"  {WORK_DIR}/X.families.csv  — per-family per-node E[copies] + P(present)")
    print()
    print("How to interpret the output:")
    print("  branches.csv:")
    print("    `copies_node_observed`   = Σ_f E[ξ_v | family f]  (genes at node v summed over data)")
    print("    `copies_node_corrected`  = observed + F·L(0)/(1-L(0)) · E[ξ_v | unobserved profile]")
    print("    `gain_events`            = E[ξ_v − η_v | data]    (gains on branch entering v)")
    print("    `loss_events`            = E[ξ_u − η_v | data]    (losses on branch entering v from parent u)")
    print("    `families_present_*`     = Σ_f (1 − P{ξ_v = 0 | f})  (number of families with ≥1 copy at v)")
    print("    `families_active_thresh50` = count of families with P{ξ_v > 0} > 0.5 (binary presence)")
    print()
    print("  families.csv (long format — one row per (family, node) above posterior threshold):")
    print("    `E_copies`   = posterior expected copy count at that node for that family")
    print("    `P_present`  = posterior probability of at least one copy at that node for that family")
    print()
    print("Reading the CSVs back with pandas:")
    print('    import pandas as pd')
    print('    b = pd.read_csv("results.branches.csv")')
    print('    b.query("not is_leaf").nlargest(10, "copies_node_corrected")')
    print('    # → 10 ancestral nodes with the largest reconstructed genomes')
    print('    f = pd.read_csv("results.families.csv")')
    print('    f.query("P_present > 0.9").groupby("node_name").size().nlargest(20)')
    print('    # → top 20 ancestors by number of "confidently present" families')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
