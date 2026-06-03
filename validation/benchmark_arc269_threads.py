"""Direct measurement: arc269 LL + gradient at 1, 4, 16 threads.

Captures wall times at canonical rates (γ=0.5, λ=0.5, μ=1, t=1) for
production_speedup_estimate computation.
"""
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/Users/ssolo/src/recount")
import numpy as np
from validation._shared import load_dataset
from recount.native_backend import (
    corrected_log_likelihood_native, gradient_native,
    per_branch_stats_native, native_version,
)

print("native_version =", native_version())

# arc269
print("Loading arc269...")
tree, profiles, _, mc, _ = load_dataset("arc269")
print(f"  F={profiles.shape[0]:,} N={tree.num_nodes} mc={mc}")
N = tree.num_nodes
gain = np.full(N, 0.5, dtype=np.float64)
loss = np.full(N, 1.0, dtype=np.float64)
dup  = np.full(N, 0.5, dtype=np.float64)
length = np.full(N, 1.0, dtype=np.float64)

results = {"native_version": native_version()}
for nt in (1, 4, 16):
    print(f"\n--- arc269 @ {nt} threads ---")
    # warm
    _ = corrected_log_likelihood_native(tree, gain, loss, dup, length, profiles, min_copies=mc, num_threads=nt)
    # LL
    N_RUN = 3 if nt > 1 else 1
    t0 = time.time()
    for _ in range(N_RUN):
        ll = corrected_log_likelihood_native(tree, gain, loss, dup, length, profiles, min_copies=mc, num_threads=nt)
    t_ll = (time.time() - t0) / N_RUN * 1000
    print(f"  LL only:  {t_ll:.1f} ms")
    # gradient
    t0 = time.time()
    for _ in range(N_RUN):
        ll2, g = gradient_native(tree, gain, loss, dup, length, profiles, min_copies=mc, num_threads=nt)
    t_g = (time.time() - t0) / N_RUN * 1000
    print(f"  gradient: {t_g:.1f} ms")
    # branch posteriors (for the Java comparison)
    t0 = time.time()
    for _ in range(N_RUN):
        bs = per_branch_stats_native(tree, gain, loss, dup, length, profiles, num_threads=nt)
    t_bs = (time.time() - t0) / N_RUN * 1000
    print(f"  branch_stats: {t_bs:.1f} ms")
    results[f"arc269_{nt}t_ll_ms"] = t_ll
    results[f"arc269_{nt}t_grad_ms"] = t_g
    results[f"arc269_{nt}t_bs_ms"] = t_bs

# williams for the Java comparison
print("\nLoading williams...")
tree, profiles, _, mc, _ = load_dataset("williams")
print(f"  F={profiles.shape[0]:,} N={tree.num_nodes} mc={mc}")
N = tree.num_nodes
gain = np.full(N, 0.5, dtype=np.float64)
loss = np.full(N, 1.0, dtype=np.float64)
dup  = np.full(N, 0.5, dtype=np.float64)
length = np.full(N, 1.0, dtype=np.float64)

for nt in (16,):
    print(f"\n--- williams @ {nt} threads ---")
    _ = corrected_log_likelihood_native(tree, gain, loss, dup, length, profiles, min_copies=mc, num_threads=nt)
    N_RUN = 10
    t0 = time.time()
    for _ in range(N_RUN):
        ll = corrected_log_likelihood_native(tree, gain, loss, dup, length, profiles, min_copies=mc, num_threads=nt)
    t_ll = (time.time() - t0) / N_RUN * 1000
    print(f"  LL only:  {t_ll:.2f} ms")
    t0 = time.time()
    for _ in range(N_RUN):
        bs = per_branch_stats_native(tree, gain, loss, dup, length, profiles, num_threads=nt)
    t_bs = (time.time() - t0) / N_RUN * 1000
    print(f"  branch_stats: {t_bs:.2f} ms")
    results[f"williams_{nt}t_ll_ms"] = t_ll
    results[f"williams_{nt}t_bs_ms"] = t_bs

# Computed speedups
JAVA_WILLIAMS_LL_MS = 227.0
JAVA_WILLIAMS_BS_MS = 934.0
results["java_williams_ll_ms"] = JAVA_WILLIAMS_LL_MS
results["java_williams_bs_ms"] = JAVA_WILLIAMS_BS_MS
results["williams_speedup_ll"] = JAVA_WILLIAMS_LL_MS / results["williams_16t_ll_ms"]
results["williams_speedup_bs"] = JAVA_WILLIAMS_BS_MS / results["williams_16t_bs_ms"]

print("\n=== summary ===")
print(json.dumps(results, indent=2))

with open("/tmp/benchmark_arc269_threads.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote /tmp/benchmark_arc269_threads.json")
