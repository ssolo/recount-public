"""Example 7 — Focal-dataset validation + speedup vs Java (Phase C).

Runs the full native pipeline (forward LL, L(0), per-branch posterior
stats, root copies & families both observed and unobserved-corrected) on
the Csurös 2026 focal arc269 datasets at Ωmin = 4 and compares each
quantity against Java's CountXXV.jar (via the validation/CountVerifyXML2
driver). Also reports wall-clock timings for both backends.

Results from a fresh run on M2 Ultra (recount-native 0.1.0, JDK 21,
-Xmx32g):

  Dataset                       F     N  maxW   LL diff    Forward-LL  Branch-stats
                                                           speedup     speedup
  ---------------------------- ----- ----- ----- --------- ----------- ------------
  Williams2017                 5378   119   668  4.8e-10   20× (11 ms) 41× (24 ms)
  arc269 methanomada15          198    29    31  1.4e-12   16× (0.1 ms) 60× (0.2 ms)
  arc269 thermoplasmatota28     393    55    84  8.2e-12   32× (0.3 ms) 72× (0.7 ms)
  arc269 halo51                 256   101   179  7.3e-12   29× (1.3 ms) 70× (2.7 ms)

For every completed case, native matches Java at the root to ≤ 5e-11
absolute for all four reported quantities (observed/corrected copies
and observed/corrected families_present at the LCA).

The two big focal sessions — tmhh94 (N=187, F=7335, maxW=654) and
eury114 (N=227, F=7335, maxW=712) — Java cannot complete within a
10-minute wall-clock budget even at -Xmx32g: per-profile heap allocation
scales with maxW and the JVM stalls in GC. Native runs them in
~30-95 ms each (forward LL) and ~70-95 ms (branch stats).

LogisticShift K-category mixture-likelihood (recount.logistic_shift):
   - K=1 with zero shifts is exactly equivalent to the base GLD path
     (validated by ``_sanity_check_k1`` below — diff 0.0).
   - K>1 implementation requires inverse-survival-map machinery not
     yet ported. Deferred to Phase D; no stored Williams/arc269 dataset
     actually uses K>1 mixtures (all use K=1 with mod_length=mod_dup=0
     = base GLD), so this is feature-completeness work rather than a
     reproduction blocker.

Prerequisites:
  - librecount.dylib built (cd native && make)
  - validation/Williams2017.countxml.gz present
  - docs/csuros_data/arc269/cryptic-t94-min4.countxml.gz present
    (bundled with the repo)
  - Java 17+ available on PATH (override with the JAVA env var)
  - Csurös' CountXXV.jar bundled at docs/csuros_data/arc269/CountXXV.jar
    or override the location with the COUNT_JAR env var
  - Java helper compiled: javac -d /tmp/recount_java_build \\
      -classpath docs/csuros_data/arc269/CountXXV.jar \\
      validation/CountVerifyXML2.java

Usage:
  PYTHONPATH=. python3 examples/07_focal_validation_and_speedup.py

Cases referencing data files not present on disk are skipped with a
warning rather than crashing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from recount.io.countxml import load_countxml
from recount.logistic_shift import (
    LogisticShiftCategory,
    _sanity_check_k1,
    mixture_log_likelihood_native,
)
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


# Java binary: respect the JAVA env var, otherwise look on PATH.
JAVA = os.environ.get("JAVA") or shutil.which("java") or "java"
# CountXXV.jar location: the repo bundles a copy at
# docs/csuros_data/arc269/CountXXV.jar; the COUNT_JAR env var overrides.
_DEFAULT_JAR = "docs/csuros_data/arc269/CountXXV.jar"
COUNT_JAR = os.environ.get("COUNT_JAR", _DEFAULT_JAR)
# Helper bytecode is compiled into /tmp/recount_java_build by the
# javac line in the docstring above; if it's missing we'll skip the
# Java half of the comparison gracefully.
JAVA_BUILD = os.environ.get("RECOUNT_JAVA_BUILD", "/tmp/recount_java_build")
JAVA_CP = f"{JAVA_BUILD}:{COUNT_JAR}"
JAR_HEAP = "-Xmx32g"
MIN_COPIES = 4
JAVA_TIMEOUT = 600  # seconds

# arc269 focal countxml files are bundled at docs/csuros_data/arc269/.
_ARC269 = "docs/csuros_data/arc269"
CASES = [
    ("Williams2017",
     "validation/Williams2017.countxml.gz",
     "wsz60-codes-edit", "wsz60-aletrim-min4.txt"),
    ("arc269 methanomada15",
     f"{_ARC269}/cryptic-t94-min4.countxml.gz",
     "methanomada15-gtdb", "cryptic-t94-m15-min4.txt"),
    ("arc269 thermoplasmatota28",
     f"{_ARC269}/cryptic-t94-min4.countxml.gz",
     "thermoplasmatota28-gtdb", "cryptic-t94-t28-min4.txt"),
    ("arc269 halo51",
     f"{_ARC269}/cryptic-t94-min4.countxml.gz",
     "halo51-gtdb", "cryptic-t94-h51-min4.txt"),
    ("arc269 tmhh94 (big)",
     f"{_ARC269}/cryptic-t94-min4.countxml.gz",
     "tmhh94-gtdb", "table-t94-arcogm-mlrvl-min4-annot.txt"),
    ("arc269 eury114 (big)",
     f"{_ARC269}/cryptic-e114-min4.countxml.gz",
     "eury114-ba-gtdb", "table-e114-arcogm-mlrvl-min4-annot.txt"),
]


def run_java(countxml_path, session_id, table_name, mc):
    p = subprocess.run(
        [JAVA, JAR_HEAP, "-classpath", JAVA_CP, "count.model.CountVerifyXML2",
         countxml_path, session_id, table_name, str(mc)],
        capture_output=True, text=True, timeout=JAVA_TIMEOUT,
    )
    metrics = {}
    for line in p.stdout.splitlines():
        if line.startswith("LL_corrected_min"):
            metrics["LL_corr"] = float(line.split("=")[1])
        elif line.startswith("L0="):
            metrics["L0"] = float(line.split("=")[1])
        elif "java LL timed:" in line:
            metrics["java_LL_ms"] = float(line.split(":")[1].split()[0])
        elif "java branch_stats" in line:
            metrics["java_branch_ms"] = float(line.split(":")[1].split()[0])
        elif line.startswith("NODE_STATS") and "node" not in line:
            parts = line.split()
            try:
                int(parts[1])
            except ValueError:
                continue
            metrics["root_obs_copies"]    = float(parts[2])
            metrics["root_corr_copies"]   = float(parts[3])
            metrics["root_obs_present"]   = float(parts[4])
            metrics["root_corr_present"]  = float(parts[5])
    return metrics


def run_native(countxml_path, session_id, table_name, mc, n_iters=5):
    sess = load_countxml(countxml_path)[session_id]
    tbl = sess.tables[table_name]
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]
    root = sess.tree.root
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length

    # warmup
    _ = corrected_log_likelihood_native(sess.tree, g, l, d, t, profiles, min_copies=mc)
    times_ll = []
    for _ in range(n_iters):
        t0 = time.time()
        ll = corrected_log_likelihood_native(sess.tree, g, l, d, t, profiles, min_copies=mc)
        times_ll.append(time.time() - t0)
    t_ll = min(times_ll) * 1000
    times_bs = []
    for _ in range(n_iters):
        t0 = time.time()
        stats = per_branch_stats_native(sess.tree, g, l, d, t, profiles)
        times_bs.append(time.time() - t0)
    t_bs = min(times_bs) * 1000
    L0 = unobserved_logL0_native(sess.tree, g, l, d, t, mc)

    C, K, _ = unobserved_inside_tensors_native(sess.tree, g, l, d, t, mc)
    sp = _get_survival_arrays(sess.tree, g, l, d, t)
    B_all, J_all, _, _, _ = compute_unobserved_outside(sess.tree, sp, K, mc)
    log_node_post, _ = compute_unobserved_posteriors(C, K, B_all, J_all, L0, mc - 1)
    n_grid = np.arange(mc, dtype=np.float64)
    Pn_root = np.exp(log_node_post[root])
    L0_lin = np.exp(L0)
    factor = F * L0_lin / (1.0 - L0_lin)
    return {
        "F": F, "N": sess.tree.num_nodes, "root": root,
        "max_W": int(profiles.sum(axis=1).max()),
        "LL_corr": ll, "L0": L0_lin,
        "root_obs_copies":   stats["copies_node"][root],
        "root_corr_copies":  stats["copies_node"][root] + factor * (Pn_root * n_grid).sum(),
        "root_obs_present":  stats["families_present"][root],
        "root_corr_present": stats["families_present"][root] + factor * (1.0 - Pn_root[0]),
        "native_LL_ms": t_ll, "native_branch_ms": t_bs,
    }


def main() -> int:
    print(f"# {native_version()}")
    print(f"# Phase C focal-dataset validation + speedup")
    print(f"# native (M2 Ultra) vs Java (CountXXV.jar, JDK 21, {JAR_HEAP})")
    print(f"# All values at Ωmin = {MIN_COPIES}")
    print()

    # Pre-flight: is the Java side wired up? Only matters if the helper
    # bytecode + the jar are both reachable; otherwise we skip Java and
    # still report the native pipeline.
    # NB: `shutil.which(JAVA)` correctly resolves both bare command names
    # (e.g. "java") and absolute paths, unlike `Path(JAVA).exists()` which
    # only handles the absolute-path case.
    have_java = (Path(JAVA_BUILD, "count", "model", "CountVerifyXML2.class").exists()
                 and Path(COUNT_JAR).exists()
                 and shutil.which(JAVA) is not None)
    if not have_java:
        print(f"# Java side not configured (compiled helper at {JAVA_BUILD}/"
              f"count/model/CountVerifyXML2.class or jar at {COUNT_JAR} or"
              f" java at {JAVA} missing); reporting native pipeline only.")

    rows = []
    for label, path, sid, table in CASES:
        print(f"---- {label} ({sid}, {table}) ----")
        if not Path(path).exists():
            print(f"  SKIP: countxml not found at {path}")
            print()
            continue
        try:
            nat = run_native(path, sid, table, MIN_COPIES)
            jav = run_java(path, sid, table, MIN_COPIES) if have_java else None
        except subprocess.TimeoutExpired:
            jav = None
        print(f"  Tree N={nat['N']}, leaves={(nat['N']+1)//2}, F={nat['F']}, max_W={nat['max_W']}")
        if jav is None or "root_obs_copies" not in jav:
            reason = ("Java side not configured" if not have_java
                      else f"Java timed out / OOM at {JAR_HEAP}")
            print(f"  {reason}; native completes:")
            print(f"    LL_corr_min{MIN_COPIES} = {nat['LL_corr']:.6f}")
            print(f"    L(0)                  = {nat['L0']:.10f}")
            print(f"    copies_node[root] obs = {nat['root_obs_copies']:.4f}")
            print(f"    families_present[root] obs = {nat['root_obs_present']:.4f}")
            print(f"    Timing: forward LL {nat['native_LL_ms']:.1f} ms, branch_stats {nat['native_branch_ms']:.1f} ms")
            rows.append((label, nat, None))
            print()
            continue
        rows.append((label, nat, jav))
        print(f"  LL_corr_min{MIN_COPIES}:    native {nat['LL_corr']:.6f}  Java {jav['LL_corr']:.6f}  diff {nat['LL_corr']-jav['LL_corr']:+.2e}")
        print(f"  L(0):              native {nat['L0']:.10f}  Java {jav['L0']:.10f}")
        print(f"  copies[root]  obs: native {nat['root_obs_copies']:>12.4f}  Java {jav['root_obs_copies']:>12.4f}  diff {nat['root_obs_copies']-jav['root_obs_copies']:+.2e}")
        print(f"  copies[root]  corr:native {nat['root_corr_copies']:>12.4f}  Java {jav['root_corr_copies']:>12.4f}  diff {nat['root_corr_copies']-jav['root_corr_copies']:+.2e}")
        print(f"  families[root] obs: native {nat['root_obs_present']:>12.4f}  Java {jav['root_obs_present']:>12.4f}  diff {nat['root_obs_present']-jav['root_obs_present']:+.2e}")
        print(f"  families[root] corr:native {nat['root_corr_present']:>12.4f}  Java {jav['root_corr_present']:>12.4f}  diff {nat['root_corr_present']-jav['root_corr_present']:+.2e}")
        sp_ll = jav["java_LL_ms"] / max(nat["native_LL_ms"], 1e-9)
        sp_bs = jav["java_branch_ms"] / max(nat["native_branch_ms"], 1e-9)
        print(f"  Timing — forward LL:    native {nat['native_LL_ms']:>7.1f} ms  Java {jav['java_LL_ms']:>7.1f} ms  speedup {sp_ll:>4.1f}x")
        print(f"  Timing — branch stats:  native {nat['native_branch_ms']:>7.1f} ms  Java {jav['java_branch_ms']:>7.1f} ms  speedup {sp_bs:>4.1f}x")
        print()

    # LogisticShift K=1 sanity check (mixture path = base path bit-for-bit)
    print("---- LogisticShift K=1 sanity check ----")
    sess = next(iter(load_countxml('validation/Williams2017.countxml.gz').values()))
    tbl = sess.tables['wsz60-aletrim-min4.txt']
    profiles = tbl.profiles.astype(np.int32)
    g, l, d, t = sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
    base, mix = _sanity_check_k1(sess.tree, g, l, d, t, profiles, min_copies=MIN_COPIES)
    print(f"  base LL = {base:.10f}")
    print(f"  K=1 LogisticShift mixture LL = {mix:.10f}")
    print(f"  diff = {base - mix:+.3e} (must be 0 — K=1 with shift=(0,0) is identity)")
    print()

    print("=" * 92)
    print(f"SUMMARY — native vs Java at Ωmin = {MIN_COPIES}")
    print("=" * 92)
    print(f"{'Dataset':<30s} {'F':>5s} {'N':>4s} {'maxW':>5s} {'LL diff':>12s} {'LL ms n':>8s} {'LL ms j':>8s} {'LL ×':>6s} {'bs ms n':>8s} {'bs ms j':>8s} {'bs ×':>6s}")
    for label, nat, jav in rows:
        if jav is None:
            print(f"{label:<30s} {nat['F']:>5d} {nat['N']:>4d} {nat['max_W']:>5d} {'(Java skipped)':>12s} {nat['native_LL_ms']:>8.1f} {'--':>8s} {'--':>6s} {nat['native_branch_ms']:>8.1f} {'--':>8s} {'--':>6s}")
        else:
            sp_ll = jav["java_LL_ms"] / max(nat["native_LL_ms"], 1e-9)
            sp_bs = jav["java_branch_ms"] / max(nat["native_branch_ms"], 1e-9)
            diff = nat["LL_corr"] - jav["LL_corr"]
            print(f"{label:<30s} {nat['F']:>5d} {nat['N']:>4d} {nat['max_W']:>5d} {diff:>+12.2e} "
                  f"{nat['native_LL_ms']:>8.1f} {jav['java_LL_ms']:>8.1f} {sp_ll:>5.1f}× "
                  f"{nat['native_branch_ms']:>8.1f} {jav['java_branch_ms']:>8.1f} {sp_bs:>5.1f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
