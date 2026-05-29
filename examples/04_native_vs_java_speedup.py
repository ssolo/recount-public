"""Example 4 — Native vs Java speedup on Williams2017.

Reproduces the claim:
    "Native CPU backend (M2 Ultra, 8 P-cores via libdispatch) is 29×
     faster than Java Count on forward LL (11 ms vs 325 ms) and 83×
     faster on forward + analytical gradient (24 ms vs 1965 ms), with
     numerical agreement to machine epsilon (4.80e-10 absolute on full
     Williams forward LL)."

Note: Java baseline times are hard-coded from a prior `java -cp ...
CountVerifyXML` run; the same measurement is reported in
README.md's Performance section. Re-running this script reports
native times only.

Prerequisites:
  - librecount.dylib built
  - validation/Williams2017.countxml.gz present

Usage:
  PYTHONPATH=. python3 examples/04_native_vs_java_speedup.py
"""
from __future__ import annotations

import time

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import (
    gradient_native,
    log_likelihood_native,
    per_branch_stats_native,
    native_version,
)

JAVA_FORWARD_MS = 325.0
JAVA_GRADIENT_MS = 1965.0


def bench(fn, *args, n_warm=2, n_run=5, **kwargs):
    for _ in range(n_warm):
        fn(*args, **kwargs)
    t0 = time.time()
    for _ in range(n_run):
        out = fn(*args, **kwargs)
    return (time.time() - t0) / n_run, out


def main() -> int:
    print(f"# {native_version()}")
    sess = next(iter(load_countxml("validation/Williams2017.countxml.gz").values()))
    tbl = sess.tables["wsz60-aletrim-min4.txt"]
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    print(f"# Williams2017: {F} families, max_sum={profiles.sum(axis=1).max()}")
    print()

    t_fwd, _ = bench(log_likelihood_native, sess.tree, g, l, d, t, profiles, num_threads=0)
    t_grad, _ = bench(gradient_native, sess.tree, g, l, d, t, profiles, num_threads=0)
    t_ev, _ = bench(per_branch_stats_native, sess.tree, g, l, d, t, profiles, num_threads=0)

    print(f"  {'workload':>32}  {'Java':>8}  {'Native':>10}  {'Speedup':>9}")
    print(f"  {'Forward LL':>32}  {JAVA_FORWARD_MS:>6.0f} ms  "
          f"{t_fwd * 1000:>8.1f} ms  {JAVA_FORWARD_MS / (t_fwd * 1000):>7.1f}×")
    print(f"  {'Forward + analytical gradient':>32}  {JAVA_GRADIENT_MS:>6.0f} ms  "
          f"{t_grad * 1000:>8.1f} ms  {JAVA_GRADIENT_MS / (t_grad * 1000):>7.1f}×")
    print(f"  {'Per-branch events':>32}  {'n/a':>8s}  "
          f"{t_ev * 1000:>8.1f} ms  {'n/a':>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
