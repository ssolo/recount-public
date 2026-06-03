"""Example 8 — native threading scaling on Apple silicon.

Sweeps ``num_threads`` from 1 to (2 × ncpu) on Williams + the big eury114
focal session to show how the native dispatch_apply parallelisation
scales on this machine, and what the default (num_threads=0, ie
sysconf(_SC_NPROCESSORS_ONLN)) settles on.

Quick takeaway from M4 Max (12P + 4E = 16 cores):
  • Forward LL on eury114 (F=7335, N=227, maxW=712): 480 ms → 40 ms
    at 16 threads (12× speedup).
  • Branch stats on eury114: 1087 ms → 89 ms at 16 threads (12× speedup).
  • The default num_threads=0 picks 16 (=ncpu) and matches the explicit
    16-thread setting.
  • P-only (12 threads) is ~10% slower than P+E (16) on big workloads;
    the E-cores meaningfully contribute even though they're slower.

For M3 Ultra (24P + 8E = 32 cores) the same dispatch_apply path will
pick up all 32 cores by default; the big focal datasets should scale
near-linearly past 24 threads given they're already at 12× on 12 cores
here.

Usage:
  PYTHONPATH=. python3 examples/08_threading_scaling.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# Csurös data mirrored under docs/csuros_data/; override with
# RECOUNT_CSUROS_DATA env var for an external copy.
_DATA_ROOT = Path(os.environ.get("RECOUNT_CSUROS_DATA", "docs/csuros_data"))

from recount.io.countxml import load_countxml
from recount.native_backend import (
    cpu_topology,
    log_likelihood_native,
    native_version,
    per_branch_stats_native,
)


CASES = [
    ("Williams2017",
     "validation/Williams2017.countxml.gz",
     "wsz60-codes-edit", "wsz60-aletrim-min4.txt"),
    ("arc269 eury114",
     str(_DATA_ROOT / "arc269" / "cryptic-e114-min4.countxml.gz"),
     "eury114-ba-gtdb", "table-e114-arcogm-mlrvl-min4-annot.txt"),
]


def main() -> int:
    print(f"# {native_version()}")
    topo = cpu_topology()
    print(f"# CPU: {topo.get('brand')}")
    print(f"# Topology: {topo.get('perf_cores')} P-cores + {topo.get('eff_cores')} E-cores"
          f" = {topo.get('ncpu')} online")
    print()

    P = topo.get("perf_cores", 0) or 0
    E = topo.get("eff_cores", 0) or 0
    total = (P or 0) + (E or 0)
    # Sweep 1, 2, 4, 8, P-only, P+E, oversubscribed
    threads_to_try = sorted({1, 2, 4, 8, max(P, 1), total, 2 * total, 0})
    threads_to_try = [t for t in threads_to_try if t >= 0]

    for label, path, sid, table in CASES:
        try:
            sess = load_countxml(path)[sid]
        except (FileNotFoundError, KeyError) as e:
            print(f"---- {label} (skipped: {e}) ----")
            continue
        tbl = sess.tables[table]
        profiles = tbl.profiles.astype(np.int32)
        F = profiles.shape[0]
        N = sess.tree.num_nodes
        max_w = int(profiles.sum(axis=1).max())
        g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
        print(f"---- {label}: F={F}, N={N}, maxW={max_w} ----")
        print(f"{'num_threads':>14s} {'forward LL (ms)':>18s} {'branch stats (ms)':>20s}")
        for nthr in threads_to_try:
            # Warmup
            _ = log_likelihood_native(sess.tree, g, l, d, t, profiles, num_threads=nthr)
            ll_times = []
            for _ in range(7):
                t0 = time.time()
                _ = log_likelihood_native(sess.tree, g, l, d, t, profiles, num_threads=nthr)
                ll_times.append(time.time() - t0)
            bs_times = []
            _ = per_branch_stats_native(sess.tree, g, l, d, t, profiles, num_threads=nthr)
            for _ in range(7):
                t0 = time.time()
                _ = per_branch_stats_native(sess.tree, g, l, d, t, profiles, num_threads=nthr)
                bs_times.append(time.time() - t0)
            label_t = "auto (0)" if nthr == 0 else str(nthr)
            print(f"{label_t:>14s} {min(ll_times)*1000:>18.1f} {min(bs_times)*1000:>20.1f}")
        print()
    print("# `num_threads=0` (auto) uses sysconf(_SC_NPROCESSORS_ONLN) = "
          f"all {total} online cores.")
    print("# Native scales near-linearly per added P-core; E-cores add ~10–15% more.")
    print("# On M3 Ultra (32 cores) the eury114 numbers should roughly halve again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
