"""Empirical sum-bin histogram vs model-predicted under each arc269 fit.

For each fit with saved rates, re-evaluate L(0, N) under the fitted rates at
N = 1..11 via Csurös SI Theorems 3-5 (`unobserved_logL0_native`), convert to
predicted bin probabilities P(Omega=k) = L(0,k+1) - L(0,k), and compare
F* * P(Omega=k) to the empirical count of arc269 families with sum=k.

Three conditioning modes per fit, side by side:

  - UNCONDITIONAL: predicted F* * P(Omega=k) vs empirical E(k) for all k>=1
  - EXCLUDE SINGLETONS (cond on Omega>=2): test the rate process's fit to the
    NON-SINGLETON tail. Singletons are widely suspected to include annotation
    noise + single-genome de novo events the GLD birth-death process does not
    aim to capture.
  - EXCLUDE 1+2 COPY FAMILIES (cond on Omega>=3): a stricter version that
    additionally drops doubletons.

Under conditioning Omega >= k_min the model's bin probability is
renormalized to P(Omega=k | Omega>=k_min) = P(Omega=k) / (1 - L(0,k_min)),
and the predicted count at bin k (for k>=k_min) is rescaled to match the
empirical Omega>=k_min total: E^{>=k_min} * P(Omega=k | Omega>=k_min).
Under this rescaling, the ratio test is purely a SHAPE comparison on the
bins both pipelines agree exist.
"""
import numpy as np
from pathlib import Path
from validation._shared import load_dataset
from recount.native_backend import unobserved_logL0_native

OUT = Path("validation/outputs")
tree, profiles, _, _, _ = load_dataset("arc269")
F_all = profiles.shape[0]
sums = profiles.sum(axis=1)
emp_bins = np.bincount(sums, minlength=12)
emp_ge_8 = int((sums >= 8).sum())

# Fit registry: (label, dir, filter Omega_min, F_visible)
FITS = [
    ("arc269 mc=1 no filter",    "brownian_arc269",                    1, F_all),
    ("arc269 sum >= 2",          "brownian_arc269_omin4_sum2",         2, 24_904),
    ("arc269 sum >= 3",          "brownian_arc269_omin4_sum3",         3, 15_413),
    ("arc269 sum >= 4 (Csuros)", "brownian_arc269_omin4_sum4",         4, 11_554),
    ("arc269 sum >= 5",          "brownian_arc269_omin4_sum5",         5,  9_492),
    ("arc269 sum >= 6",          "brownian_arc269_omin4_sum6",         6,  8_316),
    ("arc269 sum >= 7",          "brownian_arc269_omin4_sum7",         7,  7_465),
    ("arc269 sum >= 8",          "brownian_arc269_omin4_sum8",         8,  6_821),
    ("arc269 sum >= 9",          "brownian_arc269_omin4_sum9",         9,  6_315),
    ("arc269 sum >= 10",         "brownian_arc269_omin4_sum10",       10,  5_879),
    ("arc269 union-min4",        "brownian_arc269_union_subsets_min4", 4, 11_148),
]

emp_totals = {k: int((sums >= k).sum()) for k in (1, 2, 3)}

print(f"arc269: F={F_all:,}, max sum per family = {int(sums.max())}")
print(f"Empirical bins: sum=1..7 + sum>=8:")
for k in range(1, 11):
    print(f"  sum = {k:>2}: {int(emp_bins[k]):>8,}")
print(f"  sum >=11: {int((sums >= 11).sum()):>8,}")
print(f"Empirical Omega>=1: {emp_totals[1]:,};  "
      f">=2: {emp_totals[2]:,};  >=3: {emp_totals[3]:,}")
print()


def ratio(pred, emp):
    if emp <= 0:
        return float("inf") if pred > 0 else 1.0
    return pred / emp


def analyse_fit(label, dirname, Nf, F_vis):
    path = OUT / dirname / "arc269_sigma1.0_final_rates.npz"
    if not path.exists():
        return None
    rates = np.load(path)
    g, l, d, t = rates["gain"], rates["loss"], rates["dup"], rates["length"]
    L0s = [float(np.exp(unobserved_logL0_native(tree, g, l, d, t, min_copies=N)))
           for N in range(1, 12)]
    L0_fit = L0s[Nf - 1]
    Fstar = F_vis / (1 - L0_fit)
    P = [L0s[0]] + [L0s[k] - L0s[k - 1] for k in range(1, 11)]
    return dict(label=label, Nf=Nf, F_vis=F_vis, L0_fit=L0_fit, Fstar=Fstar,
                L0s=L0s, P=P)


def print_row(fit, cond, fits_summary_rows):
    """One row of pred/emp ratios at conditioning Omega>=cond.

    cond == 1: unconditional — predicted count is F* * P(Omega=k); compared
               to empirical E(k).
    cond  > 1: rescaled — predicted count is E^{>=cond} * P(Omega=k|Omega>=cond)
               with P(Omega=k|Omega>=cond) = P(Omega=k) / (1 - L(0,cond));
               compared to empirical E(k) (for k >= cond).
    """
    L0s = fit["L0s"]
    if cond == 1:
        pred_total = fit["Fstar"]
        mass_cond = 1.0
    else:
        pred_total = emp_totals[cond]
        mass_cond = max(1e-300, 1.0 - L0s[cond - 1])
    ratios = []
    for k in (1, 2, 3, 4, 5, 6, 7):
        if k < cond:
            ratios.append("    --")
            continue
        pk_cond = fit["P"][k] / mass_cond
        pred = pred_total * pk_cond
        emp = int(emp_bins[k])
        r = ratio(pred, emp)
        ratios.append(f"{r:>6.3f}")
    P_tail = (1.0 - L0s[7]) / mass_cond
    pred_tail = pred_total * P_tail
    ratio_tail = ratio(pred_tail, emp_ge_8)
    ratios.append(f"{ratio_tail:>6.3f}")
    name = f"{fit['label']} (L0={fit['L0_fit']:.3f})"
    print(f"  {name:<40} | " + " | ".join(ratios))
    fits_summary_rows.append((name, ratios))


def main():
    fits = []
    for entry in FITS:
        r = analyse_fit(*entry)
        if r is not None:
            fits.append(r)
        else:
            print(f"  MISSING fit: {entry[0]}  ({entry[1]})")

    for cond, cond_label in ((1, "UNCONDITIONAL (all bins, includes singletons)"),
                             (2, "CONDITIONED ON Omega>=2 (singletons excluded)"),
                             (3, "CONDITIONED ON Omega>=3 (1+2 copy families excluded)")):
        print()
        print("=" * 120)
        print(f"{cond_label}")
        if cond > 1:
            print(f"  empirical Omega>=" + f"{cond}" + f" total = {emp_totals[cond]:,};  "
                  f"predicted total rescaled to match.")
        print("=" * 120)
        header = f"  {'fit':<40} | " + " | ".join(
            f"k={k:>2}" for k in (1, 2, 3, 4, 5, 6, 7)) + " | sum>=8"
        print(header)
        print("  " + "-" * 110)
        rows = []
        for r in fits:
            print_row(r, cond, rows)
    print()


if __name__ == "__main__":
    main()
