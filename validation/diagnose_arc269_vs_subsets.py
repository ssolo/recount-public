"""v2: seed arc269 rates from subset fits, KEEP arc269 native branch lengths.

The subset's branch_length at node u is the t value on the *subset's*
edge into u. That subset-edge often corresponds to MULTIPLE arc269 edges
collapsed (the arc269 tree has internal nodes that the subset tree skips).
So transplanting subset branch_length into arc269 is wrong — it would
make arc269 think a multi-edge path has the length of one collapsed
edge.

Per-unit rates (gain, dup) ARE transplantable: they are intensities per
unit length and don't depend on how the edges happen to be carved up.
loss_rate is 1.0 by convention so transplant is identity.

Also: try the seed against arc269's bundled rates from the countxml as
a SANITY CHECK that the cold fit's branch lengths are sensible.
"""
from pathlib import Path
import csv
import numpy as np

# Imports work via `PYTHONPATH=. python3 validation/diagnose_arc269_vs_subsets.py`.
from validation._shared import load_dataset
from recount.native_backend import corrected_log_likelihood_native, gradient_native

# Resolve relative to the repo root regardless of working directory.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation/outputs"

SUBSETS = ["dpann80", "proteo75", "eury114", "ed194"]

def load_branches(path):
    out = {}
    if not path.exists(): return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try: out[int(r["node_index"])] = r
            except: pass
    return out

def map_subset_to_arc269(subtree, tree269):
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    children = [[] for _ in range(subtree.num_nodes)]
    for v in range(subtree.num_nodes):
        if subtree.parent[v] >= 0:
            children[subtree.parent[v]].append(v)
    leaves_below = [None] * subtree.num_nodes
    for u in range(subtree.num_nodes):
        if u < subtree.num_leaves:
            leaves_below[u] = [u]
        else:
            leaves_below[u] = []
            for c in children[u]: leaves_below[u].extend(leaves_below[c])
    mapping = {}
    for u in range(subtree.num_nodes):
        names = [subtree.leaf_names[l] for l in leaves_below[u]]
        idxs = [name_to_idx[n] for n in names if n in name_to_idx]
        if not idxs: continue
        ancs = []
        for li in idxs:
            s = set(); v = li
            while v >= 0:
                s.add(int(v)); v = tree269.parent[v]
            ancs.append(s)
        mapping[u] = min(set.intersection(*ancs))
    return mapping

tree269, profiles269, _, mc269, _ = load_dataset("arc269")
N = tree269.num_nodes
# arc269 cold-MAP fit's branch lengths — the basin baseline we'll evaluate against
arc269_cold_rates = np.load(OUT / "map_sigma1/arc269_sigma1.0_final_rates.npz")
arc269_15_rates  = np.load(OUT / "profile_likelihood_arc269/arc269_sigma1.0_final_rates.npz")
cold_length = arc269_cold_rates["length"]
print(f"arc269 cold-MAP branch lengths: median={np.median(cold_length):.4f} "
      f"min={cold_length.min():.4f} max={cold_length.max():.4f}")
print(f"arc269 15-start global branch lengths: median={np.median(arc269_15_rates['length']):.4f} "
      f"min={arc269_15_rates['length'].min():.4f} max={arc269_15_rates['length'].max():.4f}")

# Collect subset rates per arc269 node (per-unit-rate, transplantable)
per_node = {v: {"gain": [], "loss": [], "dup": []} for v in range(N)}
for ds in SUBSETS:
    subtree, _, _, _, _ = load_dataset(ds)
    sub2arc = map_subset_to_arc269(subtree, tree269)
    # Prefer "ourMAP" (matches σ=1 prior) but fall back to csuros
    for fit_dir in (OUT / "map_sigma1", OUT / "reproduction"):
        for fname in (f"{ds}_map_sigma1.0.branches.csv", f"{ds}_reproduce.branches.csv"):
            p = fit_dir / fname
            if not p.exists(): continue
            rows = load_branches(p)
            for u, av in sub2arc.items():
                if u not in rows: continue
                try:
                    g = float(rows[u]["gain_rate"]); d = float(rows[u]["dup_rate"])
                    l = float(rows[u]["loss_rate"])
                except (ValueError, KeyError):
                    continue
                # Filter boundary blow-up values from subset fits at the
                # sub-critical / gain-unbounded edge — they would poison
                # the seed if transplanted. Keep only sensible interior
                # values; the cross-subset median picks up the rest.
                if not (np.isfinite(g) and 1e-4 < g < 50.0): continue
                if not (np.isfinite(d) and 1e-4 < d < 0.99): continue
                if not (np.isfinite(l) and 1e-4 < l < 50.0): continue
                per_node[av]["gain"].append(g); per_node[av]["loss"].append(l); per_node[av]["dup"].append(d)
            break  # one fit per subset is enough

# Median per node; for unmapped nodes use the cross-mapped median.
seed_gain = np.full(N, np.nan)
seed_loss = np.full(N, np.nan)
seed_dup  = np.full(N, np.nan)
for v in range(N):
    if per_node[v]["gain"]:
        seed_gain[v] = np.median(per_node[v]["gain"])
        seed_loss[v] = np.median(per_node[v]["loss"])
        seed_dup[v]  = np.median(per_node[v]["dup"])

def fill(arr, default):
    mask = np.isfinite(arr)
    med = float(np.median(arr[mask])) if mask.any() else default
    arr[~mask] = med
    return arr
seed_gain = fill(seed_gain, 0.1)
seed_loss = fill(seed_loss, 1.0)
seed_dup  = fill(seed_dup,  0.5)

# Final safety clamp on the assembled seed
seed_gain = np.clip(seed_gain, 1e-3, 20.0)
seed_dup  = np.clip(seed_dup,  1e-3, 0.95)
seed_loss = np.clip(seed_loss, 1e-3, 20.0)

print(f"\nSeed rates:")
print(f"  gain: median={np.median(seed_gain):.4f} min={seed_gain.min():.4f} max={seed_gain.max():.4f}")
print(f"  loss: median={np.median(seed_loss):.4f} min={seed_loss.min():.4f} max={seed_loss.max():.4f}")
print(f"  dup:  median={np.median(seed_dup):.4f}  min={seed_dup.min():.4f}  max={seed_dup.max():.4f}")
print(f"  length: NOT transplanted; tested with two arc269-native length vectors below")

# Reference LLs
print("\n--- arc269 reference LLs ---")
LLc = corrected_log_likelihood_native(tree269,
    arc269_cold_rates["gain"], arc269_cold_rates["loss"],
    arc269_cold_rates["dup"], arc269_cold_rates["length"],
    profiles269, min_copies=mc269, num_threads=8)
print(f"  arc269 cold MAP (rates + length, basin baseline):  LL = {LLc:>12,.0f}")
LL15 = corrected_log_likelihood_native(tree269,
    arc269_15_rates["gain"], arc269_15_rates["loss"],
    arc269_15_rates["dup"], arc269_15_rates["length"],
    profiles269, min_copies=mc269, num_threads=8)
print(f"  arc269 15-start global (rates + length):           LL = {LL15:>12,.0f}")

# What if we swap rates and length between fits?
LL_15g_cl = corrected_log_likelihood_native(tree269,
    arc269_15_rates["gain"], arc269_15_rates["loss"], arc269_15_rates["dup"],
    cold_length, profiles269, min_copies=mc269, num_threads=8)
print(f"  15-start rates + cold length (length-swap test):  LL = {LL_15g_cl:>12,.0f}")
LL_cg_15l = corrected_log_likelihood_native(tree269,
    arc269_cold_rates["gain"], arc269_cold_rates["loss"], arc269_cold_rates["dup"],
    arc269_15_rates["length"], profiles269, min_copies=mc269, num_threads=8)
print(f"  cold rates + 15-start length (length-swap test):  LL = {LL_cg_15l:>12,.0f}")

print("\n--- subset-rates seed ---")
LL_cl = corrected_log_likelihood_native(tree269,
    seed_gain, seed_loss, seed_dup, cold_length,
    profiles269, min_copies=mc269, num_threads=8)
print(f"  subset rates + cold length:                       LL = {LL_cl:>12,.0f}")
LL_15l = corrected_log_likelihood_native(tree269,
    seed_gain, seed_loss, seed_dup, arc269_15_rates["length"],
    profiles269, min_copies=mc269, num_threads=8)
print(f"  subset rates + 15-start length:                   LL = {LL_15l:>12,.0f}")
LL_unit = corrected_log_likelihood_native(tree269,
    seed_gain, seed_loss, seed_dup, np.ones(N),
    profiles269, min_copies=mc269, num_threads=8)
print(f"  subset rates + length=1 everywhere:               LL = {LL_unit:>12,.0f}")

# Save the BEST seed (subset rates + best-known length)
best_LL = max(LL_cl, LL_15l, LL_unit)
if best_LL == LL_15l:
    best_len = arc269_15_rates["length"]
    best_label = "subset rates + 15-start length"
elif best_LL == LL_cl:
    best_len = cold_length
    best_label = "subset rates + cold length"
else:
    best_len = np.ones(N)
    best_label = "subset rates + length=1"

np.savez(OUT / "arc269_seed_from_subsets_v2.npz",
         gain=seed_gain, loss=seed_loss, dup=seed_dup, length=best_len,
         ll_at_seed=best_LL,
         n_mapped=int(sum(1 for v in per_node if per_node[v]["gain"])))
print(f"\nBest seed: '{best_label}'  LL={best_LL:,.0f}  gap_vs_cold={best_LL-LLc:+,.0f}  gap_vs_15st={best_LL-LL15:+,.0f}")
print(f"Saved to {OUT}/arc269_seed_from_subsets_v2.npz")
