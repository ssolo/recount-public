"""Example 9 — end-to-end ``recount analyze`` CLI on Williams2017.

Demonstrates the single-command workflow:

  1. Extract a Newick tree + TSV count table from the Williams2017
     countxml fixture (one-time setup for the demo).
  2. Run ``recount analyze`` which: fits the GLD rates, computes
     per-branch posteriors, computes per-family per-node posteriors,
     and writes three output files keyed by ``--out-prefix``:
        <prefix>.countxml.gz    — Java-compatible session XML
        <prefix>.branches.csv   — per-node summary (rates, copies,
                                  events, family-presence — observed
                                  and unobserved-corrected)
        <prefix>.families.csv   — per-family per-node E[copies] and
                                  P(present)
  3. Round-trip the CountXML back through ``load_countxml`` to verify
     it's Java-compatible.

The defaults match Csurös' canonical workflow:
  - fix loss rate at 1.0 (Williams / arc269 convention)
  - root edge length at +∞
  - duplication rate bounded by loss (biological constraint)
  - Ωmin = 1: condition on family-absent (empty profile excluded)

Usage:
  PYTHONPATH=. python3 examples/09_analyze_cli.py
"""
from __future__ import annotations

import gzip
import re
import subprocess
import sys
from pathlib import Path


WORK_DIR = Path("/tmp/recount_analyze_demo")
WORK_DIR.mkdir(exist_ok=True)
TREE_PATH = WORK_DIR / "williams_tree.tre"
TABLE_PATH = WORK_DIR / "williams_table.tsv"
OUT_PREFIX = WORK_DIR / "williams_out"


def extract_inputs():
    """Pull tree + table out of the bundled Williams2017 countxml fixture."""
    d = gzip.open("validation/Williams2017.countxml.gz").read().decode()
    # Tree CDATA — first <tree>...<![CDATA[ ... ]]>
    m = re.search(r"<tree[^>]*>\s*<!\[CDATA\[\s*(.*?)\s*\]\]>", d, re.DOTALL)
    newick = m.group(1).strip().rstrip(";") + ";"
    TREE_PATH.write_text(newick + "\n")
    # Table CDATA — find the aletrim-min4 table
    i = d.find("wsz60-aletrim-min4.txt")
    i_cdata = d.find("<![CDATA[", i)
    i_end = d.find("]]>", i_cdata)
    table = d[i_cdata + len("<![CDATA[") : i_end].strip()
    TABLE_PATH.write_text(table + "\n")
    print(f"# wrote {TREE_PATH} (Newick, {TREE_PATH.stat().st_size} bytes)")
    print(f"# wrote {TABLE_PATH} (TSV,    {TABLE_PATH.stat().st_size} bytes)")


def run_analyze():
    """Invoke the CLI as a subprocess (so the example is realistic)."""
    cmd = [
        sys.executable, "-m", "recount.cli", "analyze",
        str(TREE_PATH), str(TABLE_PATH),
        "--out-prefix", str(OUT_PREFIX),
        "--min-copies", "1",
        "--max-iter", "30",
    ]
    print(f"# running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def report_outputs():
    """Show the first few rows of each output file."""
    print()
    print("---- branches.csv (first 4 rows, including root + a leaf) ----")
    lines = (OUT_PREFIX.with_suffix(".branches.csv").read_text().splitlines())
    # header + first leaf + root
    print(lines[0][:160])
    for ln in lines[1:3]:
        print(ln[:160])
    root_line = [ln for ln in lines if ln.startswith("118,")]
    if root_line:
        print(root_line[0][:160])
    print(f"  ({len(lines) - 1} non-header rows total)")

    print()
    print("---- families.csv (first 5 rows) ----")
    fam_lines = (OUT_PREFIX.with_suffix(".families.csv").read_text().splitlines())
    print(fam_lines[0])
    for ln in fam_lines[1:5]:
        print(ln)
    print(f"  ({len(fam_lines) - 1} non-header rows total)")

    print()
    print("---- countxml.gz round-trip ----")
    from recount.io.countxml import load_countxml
    s = load_countxml(str(OUT_PREFIX) + ".countxml.gz")
    for sid, sess in s.items():
        print(f"  session={sid}: N={sess.tree.num_nodes}, has_rates={sess.rates is not None}, "
              f"tables={list(sess.tables)}")


def main() -> int:
    extract_inputs()
    run_analyze()
    report_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
