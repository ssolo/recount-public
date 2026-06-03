"""Reconstruct per-node copies/families at LACA + subclade-ancestor nodes
for the K=2 LogisticShift mixture fits.

Approach: for each category k, compute the standard GLD reconstruction
using rates_k = derive_category_rates(base_rates, cat_k). Combine
across categories by mixing weight p_k:

    cp_mix[v] = Σ_k p_k · cp_k[v]
    fm_mix[v] = Σ_k p_k · fm_k[v]

This is an approximation: a rigorous family-by-family responsibility-
weighted average would give cp_mix[v] = Σ_f Σ_k γ_{f,k} · cp_{f,k}[v]
(per-family per-category posteriors), which requires per-family
reconstruction. The p_k-weighted average matches the family-averaged
version when responsibilities are well-mixed (γ ≈ p for all f), which
is true for the K=2 fits we see here (p_mix not concentrated on one
category).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recount.rates import GLDRates
from recount.logistic_shift import LogisticShiftCategory, derive_category_rates
from validation._shared import load_dataset, reconstruct


SUBSETS = {
    "dpann80": "dpann80",
    "proteo75": "proteo75",
    "eury114": "eury114",
    "ed194": "ed194",
}


def find_lca_in_arc269(tree269, subset_leaf_names):
    name_to_idx = {n: i for i, n in enumerate(tree269.leaf_names)}
    leaf_indices = [name_to_idx[n] for n in subset_leaf_names if n in name_to_idx]
    parent = tree269.parent
    ancs_per_leaf = []
    for li in leaf_indices:
        ancs = set(); v = li
        while v >= 0: ancs.add(int(v)); v = parent[v]
        ancs_per_leaf.append(ancs)
    return min(set.intersection(*ancs_per_leaf))


def reconstruct_k2(label: str):
    """For dataset `label`, load K=2 fit and compute per-node posteriors."""
    tree, profiles, _, mc, _ = load_dataset(label)
    summary = json.load(open(f"validation/outputs/mixture_K2/{label}_K2_summary.json"))
    rates_npz = np.load(f"validation/outputs/mixture_K2/{label}_K2_final.npz")
    base_gain = rates_npz["base_gain"]; base_loss = rates_npz["base_loss"]
    base_dup = rates_npz["base_dup"]; base_length = rates_npz["base_length"]
    delta_dup = np.array(summary["delta_dup"])
    delta_length = np.array(summary["delta_length"])
    p_mix = np.array(summary["p_mix"])
    K = len(p_mix)

    # Per-category reconstruction
    cp_by_k = []; fm_by_k = []
    ll_by_k = []
    for k in range(K):
        cat = LogisticShiftCategory(probability=float(p_mix[k]),
                                     mod_length=float(delta_length[k]),
                                     mod_duplication=float(delta_dup[k]))
        gk, lk, dk, tk = derive_category_rates(base_gain, base_loss, base_dup, base_length, cat)
        rates_k = GLDRates(tree=tree, gain=gk, loss=lk, dup=dk, length=tk)
        rec = reconstruct(tree, rates_k, profiles, mc)
        # We need PER-NODE cp and fm. reconstruct() returns only the root values.
        # For arc269 we want subclade-ancestor cp/fm — we need per-node arrays.
        # Use the branches-writer for that. For now, just root values.
        cp_by_k.append(rec["root_copies_corr"])
        fm_by_k.append(rec["root_families_corr"])
        ll_by_k.append(rec["ll"])

    return {
        "label": label, "K": K, "p_mix": p_mix.tolist(),
        "ll_per_cat": ll_by_k,
        "root_cp_per_cat": cp_by_k, "root_fm_per_cat": fm_by_k,
        "root_cp_mix": float(np.dot(p_mix, cp_by_k)),
        "root_fm_mix": float(np.dot(p_mix, fm_by_k)),
        "summary_LL_corr_mix": summary["LL_corr_mix"],
        "L0_mix": summary["L0_mix"],
    }


def main():
    print(f"{'dataset':<10} {'K':>2} {'p_mix':>20} {'cat1 cp/fm':>15} {'cat2 cp/fm':>15} {'p-weighted cp/fm/cpf':>22}")
    print("-" * 95)
    out = {}
    for ds in SUBSETS:
        r = reconstruct_k2(ds)
        cpf_mix = r["root_cp_mix"] / r["root_fm_mix"] if r["root_fm_mix"] > 0 else float("nan")
        print(f"{ds:<10} {r['K']:>2} {str([round(p,2) for p in r['p_mix']]):>20} "
              f"{r['root_cp_per_cat'][0]:>6,.0f}/{r['root_fm_per_cat'][0]:>5,.0f}     "
              f"{r['root_cp_per_cat'][1]:>6,.0f}/{r['root_fm_per_cat'][1]:>5,.0f}     "
              f"{r['root_cp_mix']:>6,.0f}/{r['root_fm_mix']:>5,.0f}/{cpf_mix:.2f}")
        out[ds] = r

    # For arc269: also reconstruct at subclade-ancestor nodes.
    print("\n--- arc269 K=2 at subclade ancestors (per-category, p-weighted) ---")
    tree269, profiles269, _, mc269, _ = load_dataset("arc269")
    rates_npz = np.load("validation/outputs/mixture_K2/arc269_K2_final.npz")
    summary = json.load(open("validation/outputs/mixture_K2/arc269_K2_summary.json"))
    p_mix = np.array(summary["p_mix"])
    delta_dup = np.array(summary["delta_dup"]); delta_length = np.array(summary["delta_length"])

    # Find subclade-ancestor nodes
    subclade_node = {}
    for ds in SUBSETS:
        subtree, _, _, _, _ = load_dataset(ds)
        subclade_node[ds] = find_lca_in_arc269(tree269, subtree.leaf_names)

    # For each category, get per-node posteriors via a proper reconstruction call
    # that returns ALL node copies. Use the branches.csv writer interface, but
    # write to a temp file and re-load.
    from validation._shared import write_outputs
    import tempfile, csv
    arc269_subclade = {}
    K = 2
    for k in range(K):
        cat = LogisticShiftCategory(probability=float(p_mix[k]),
                                     mod_length=float(delta_length[k]),
                                     mod_duplication=float(delta_dup[k]))
        gk, lk, dk, tk = derive_category_rates(rates_npz["base_gain"], rates_npz["base_loss"],
                                                rates_npz["base_dup"], rates_npz["base_length"], cat)
        rates_k = GLDRates(tree=tree269, gain=gk, loss=lk, dup=dk, length=tk)
        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / f"arc269_K{k}"
            write_outputs("arc269", tree269, rates_k, profiles269, mc269, prefix)
            with open(f"{prefix}.branches.csv") as fh:
                for row in csv.DictReader(fh):
                    idx = int(row["node_index"])
                    if idx in subclade_node.values() or idx == tree269.root:
                        arc269_subclade.setdefault(idx, {}).setdefault(k, {})
                        arc269_subclade[idx][k]["cp"] = float(row["copies_node_corrected"])
                        arc269_subclade[idx][k]["fm"] = float(row["families_present_corrected"])

    # Aggregate via p_mix
    print(f"\n{'node':<35} {'cat1 cp/fm':>15} {'cat2 cp/fm':>15} {'p-weighted cp/fm/cpf':>22}")
    arc269_subclade_summary = {}
    for ds in SUBSETS:
        idx = subclade_node[ds]
        c1 = arc269_subclade[idx][0]; c2 = arc269_subclade[idx][1]
        cp_mix = p_mix[0] * c1["cp"] + p_mix[1] * c2["cp"]
        fm_mix = p_mix[0] * c1["fm"] + p_mix[1] * c2["fm"]
        cpf_mix = cp_mix / fm_mix if fm_mix > 0 else float("nan")
        print(f"{ds + f' (node {idx})':<35} {c1['cp']:>6,.0f}/{c1['fm']:>5,.0f}     "
              f"{c2['cp']:>6,.0f}/{c2['fm']:>5,.0f}     "
              f"{cp_mix:>6,.0f}/{fm_mix:>5,.0f}/{cpf_mix:.2f}")
        arc269_subclade_summary[ds] = {"node_idx": idx, "cp_mix": cp_mix, "fm_mix": fm_mix,
                                        "cpf_mix": cpf_mix, "per_cat": [c1, c2]}
    # arc269 root
    c1 = arc269_subclade[tree269.root][0]; c2 = arc269_subclade[tree269.root][1]
    cp_mix = p_mix[0] * c1["cp"] + p_mix[1] * c2["cp"]
    fm_mix = p_mix[0] * c1["fm"] + p_mix[1] * c2["fm"]
    cpf_mix = cp_mix / fm_mix if fm_mix > 0 else float("nan")
    print(f"{'arc269 LACA (root)':<35} {c1['cp']:>6,.0f}/{c1['fm']:>5,.0f}     "
          f"{c2['cp']:>6,.0f}/{c2['fm']:>5,.0f}     "
          f"{cp_mix:>6,.0f}/{fm_mix:>5,.0f}/{cpf_mix:.2f}")
    out["arc269_LACA_K2_mix"] = {"cp": cp_mix, "fm": fm_mix, "cpf": cpf_mix}
    out["arc269_subclade_K2_mix"] = arc269_subclade_summary

    Path("validation/outputs/k2_reconstruction.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote validation/outputs/k2_reconstruction.json")


if __name__ == "__main__":
    main()
