"""Comprehensive native-backend validation on the 4-leaf fixture and
Williams2017. Run as:

    PYTHONPATH=. python3 validation/validate_native.py

Outputs:
    validation/results/validate_native.json — all metrics in machine-readable form
    stdout                                  — human-readable summary
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from recount import Tree, GLDRates
from recount.events import per_branch_stats
from recount.gld import (
    corrected_log_likelihood, gradient_survival, log_likelihood,
)
from recount.io.countxml import load_countxml
from recount.ml import default_initial_rates, fit_rates
from recount.native_backend import (
    corrected_log_likelihood_native,
    gradient_native,
    log_likelihood_native,
    native_version,
    per_branch_stats_native,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "validation" / "results" / "validate_native.json"
JAVA_REF_PATH = REPO_ROOT / "validation" / "results" / "Williams2017_min1.json"
WILLIAMS = REPO_ROOT / "validation" / "Williams2017.countxml.gz"


# ---- Java reference values for the 4-leaf fixture (from tests/conftest.py) ----
PARENT4 = np.array([4, 4, 5, 5, 6, 6, -1])
LEAF_NAMES4 = ["A", "B", "C", "D"]
GAIN4   = np.array([0.10, 0.20, 0.15, 0.25, 0.30, 0.18, 0.50])
LOSS4   = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
DUP4    = np.array([0.40, 0.30, 0.50, 0.20, 0.45, 0.35, 0.00])
LENGTH4 = np.array([0.50, 0.30, 0.40, 0.60, 0.20, 0.10, np.inf])
PROF4   = np.array([
    [1, 0, 1, 1],
    [2, 0, 0, 1],
    [1, 1, 1, 1],
    [3, 2, 1, 0],
], dtype=np.int64)
JAVA_LL_RAW_4   = -22.828057245637
JAVA_LL_CORR_4  = -19.549998418545
JAVA_LL_PER_FAM_4 = [
    -4.718514262887, -7.076803097912, -3.678838805376, -7.353901079461,
]
JAVA_GRAD_SURV_4 = np.array([
    [ 6.474801393571e-01, -6.039058620592e+00,  6.327173214408e+00],
    [-5.832741090132e-01,  7.657675543676e+00, -3.442377009903e+00],
    [-1.235754528008e+00,  1.156682901058e+00, -4.833024979152e+00],
    [-4.693230084236e-01, -1.215266469122e+00, -4.648020383458e+00],
    [-5.109813500281e-02, -3.995406790690e+00,  3.104705710827e+00],
    [-1.989540042975e-01, -6.280286100697e-01, -4.801469013416e+00],
    [ 7.444742630037e-01,  0.000000000000e+00,  0.000000000000e+00],
])


def part1_four_leaf():
    """4-leaf fixture: native vs Java reference to machine precision."""
    print()
    print("## PART 1: 4-leaf fixture (vs Java reference)")
    tree = Tree(parent=PARENT4, leaf_names=LEAF_NAMES4)
    rates = GLDRates(tree=tree, gain=GAIN4, loss=LOSS4, dup=DUP4, length=LENGTH4)
    prof32 = PROF4.astype(np.int32)

    # Forward LL (raw + corrected, per-family)
    per_fam = log_likelihood_native(tree, GAIN4, LOSS4, DUP4, LENGTH4, prof32)
    ll_corr = corrected_log_likelihood_native(
        tree, GAIN4, LOSS4, DUP4, LENGTH4, prof32, min_copies=1)
    diff_perfam = float(np.max(np.abs(per_fam - JAVA_LL_PER_FAM_4)))
    diff_corr = abs(float(ll_corr) - JAVA_LL_CORR_4)
    print(f"  forward LL per-family: max|native - java| = {diff_perfam:.2e}")
    print(f"  forward LL corrected:  |native - java|     = {diff_corr:.2e}")

    # Analytical gradient (survival parameterization)
    LL_grad, grad_flat = gradient_native(
        tree, GAIN4, LOSS4, DUP4, LENGTH4, prof32, min_copies=1)
    grad = grad_flat.reshape(tree.num_nodes, 3)
    diff_grad = float(np.max(np.abs(grad - JAVA_GRAD_SURV_4)))
    print(f"  gradient (survival):   max|native - java| = {diff_grad:.2e}")

    # Per-branch events: native vs NumPy (no Java reference for events)
    np_stats = per_branch_stats(tree, rates, PROF4)
    n_stats = per_branch_stats_native(tree, GAIN4, LOSS4, DUP4, LENGTH4, prof32)
    diffs_ev = {
        k: float(np.max(np.abs(n_stats[k] - getattr(np_stats, k))))
        for k in ("copies_node", "copies_edge", "gain_events", "loss_events")
    }
    diffs_ev["num_families_active"] = int(np.max(np.abs(
        n_stats["num_families_active"].astype(np.int64) -
        np_stats.num_families_active.astype(np.int64))))
    print("  per-branch events (native vs NumPy):")
    for k, v in diffs_ev.items():
        print(f"    {k:20s}: max|diff| = {v}")

    return {
        "ll_perfamily_native_vs_java_max": diff_perfam,
        "ll_corrected_native_vs_java": diff_corr,
        "gradient_native_vs_java_max": diff_grad,
        "events_native_vs_numpy": diffs_ev,
    }


def part2_williams_forward():
    """Williams forward LL: native vs cached Java reference."""
    print()
    print("## PART 2: Williams2017 forward LL (vs cached Java)")
    sess = next(iter(load_countxml(WILLIAMS).values()))
    tbl = next(iter(sess.tables.values()))
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]
    print(f"  F={F} families, max_sum={profiles.sum(axis=1).max()}")
    ref = json.loads(JAVA_REF_PATH.read_text())

    g, l, d, t = (sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length)
    # Warm up + time multi-thread
    for _ in range(2):
        log_likelihood_native(sess.tree, g, l, d, t, profiles, num_threads=0)
    t0 = time.time()
    for _ in range(5):
        per_fam = log_likelihood_native(sess.tree, g, l, d, t, profiles, num_threads=0)
    t_mt = (time.time() - t0) / 5
    ll_raw_native = float(per_fam.sum())
    ll_corr_native = corrected_log_likelihood_native(
        sess.tree, g, l, d, t, profiles, min_copies=1, num_threads=0)
    diff_java = ll_raw_native - ref["java"]["LL_raw"]
    diff_numpy = ll_raw_native - ref["python"]["LL_raw"]
    print(f"  native time (MT, 8 cores):     {t_mt*1000:.1f} ms")
    print(f"  native LL_raw:                 {ll_raw_native:.6f}")
    print(f"  java   LL_raw:                 {ref['java']['LL_raw']:.6f}  (Δ {diff_java:+.2e})")
    print(f"  numpy  LL_raw:                 {ref['python']['LL_raw']:.6f}  (Δ {diff_numpy:+.2e})")
    print(f"  java vs numpy stored Δ:        {ref['diff']['LL_raw']['diff']:.2f} nats (1.1e-5 rel)")
    print(f"  speedup vs Java 325 ms:        {325/t_mt/1000:.1f}x")

    return {
        "F": F,
        "max_sum": int(profiles.sum(axis=1).max()),
        "ll_raw_native": ll_raw_native,
        "ll_corrected_native": float(ll_corr_native),
        "ll_raw_java": ref["java"]["LL_raw"],
        "ll_raw_numpy": ref["python"]["LL_raw"],
        "delta_native_vs_java": diff_java,
        "delta_native_vs_numpy": diff_numpy,
        "wall_clock_ms_native_mt": t_mt * 1000,
    }, sess, tbl


def part3_ml_fit_all_backends(sess, tbl):
    """ML fit on a Williams subset: all three backends should converge identically."""
    print()
    print("## PART 3: Williams ML fit, three backends (500 fams W≤8, 30 iters)")
    subset = tbl.profiles[tbl.profiles.sum(axis=1) <= 8][:500]
    init = default_initial_rates(sess.tree)
    results = {}
    for be in ("numpy", "torch", "native"):
        t0 = time.time()
        fr = fit_rates(
            sess.tree, subset, initial_rates=init,
            fix_loss=True, fix_root_length=True, min_copies=1, max_iter=30,
            backend=be,
        )
        wall = time.time() - t0
        results[be] = {"fr": fr, "wall_s": wall}
        print(f"  {be:>6}: LL={fr.log_likelihood:.6f}  iters={fr.n_iter}  time={wall:.2f}s")

    # Pairwise rate agreement
    pair_diffs = {}
    for b1, b2 in [("numpy", "torch"), ("numpy", "native"), ("torch", "native")]:
        r1 = results[b1]["fr"].rates
        r2 = results[b2]["fr"].rates
        diffs = {
            "gain": float(np.max(np.abs(r1.gain - r2.gain))),
            "dup":  float(np.max(np.abs(r1.dup - r2.dup))),
        }
        pair_diffs[f"{b1}_vs_{b2}"] = diffs
        print(f"  rate diff {b1} vs {b2}: max|Δgain|={diffs['gain']:.2e}  max|Δdup|={diffs['dup']:.2e}")
    return {
        "ll_native": results["native"]["fr"].log_likelihood,
        "ll_numpy":  results["numpy"]["fr"].log_likelihood,
        "ll_torch":  results["torch"]["fr"].log_likelihood,
        "wall_native_s": results["native"]["wall_s"],
        "wall_numpy_s":  results["numpy"]["wall_s"],
        "wall_torch_s":  results["torch"]["wall_s"],
        "pairwise_rate_diffs": pair_diffs,
    }, results["native"]["fr"].rates


def part4_events_subset_vs_numpy(sess, tbl):
    """Events: native vs NumPy on a Williams subset (NumPy too slow for full)."""
    print()
    print("## PART 4: Williams events — native vs NumPy (1000 fams W≤16)")
    sub = tbl.profiles[tbl.profiles.sum(axis=1) <= 16][:1000]
    sub32 = sub.astype(np.int32)
    g, l, d, t = (sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length)

    for _ in range(2):
        per_branch_stats_native(sess.tree, g, l, d, t, sub32, num_threads=0)
    t0 = time.time()
    for _ in range(5):
        stats_native = per_branch_stats_native(sess.tree, g, l, d, t, sub32, num_threads=0)
    t_n = (time.time() - t0) / 5

    t0 = time.time()
    stats_np = per_branch_stats(sess.tree, sess.rates, sub)
    t_p = time.time() - t0
    speedup = t_p / t_n
    print(f"  native: {t_n*1000:.1f} ms  (avg of 5)")
    print(f"  numpy:  {t_p*1000:.0f} ms")
    print(f"  speedup: {speedup:.0f}x")

    diffs = {}
    for key in ("copies_node", "copies_edge", "gain_events", "loss_events"):
        d_ = float(np.max(np.abs(stats_native[key] - getattr(stats_np, key))))
        diffs[key] = d_
        print(f"  max|Δ {key:18s}|: {d_:.2e}")
    nfa_diff = int(np.max(np.abs(
        stats_native["num_families_active"].astype(np.int64) -
        stats_np.num_families_active.astype(np.int64))))
    diffs["num_families_active"] = nfa_diff
    print(f"  max|Δ num_families_active|: {nfa_diff} (int)")
    return {
        "wall_native_ms": t_n * 1000,
        "wall_numpy_ms":  t_p * 1000,
        "speedup": speedup,
        "max_diffs_native_vs_numpy": diffs,
    }


def part5_full_williams_events(sess, tbl, fitted_rates):
    """Full Williams events at both stored and MLE-fitted rates (native only)."""
    print()
    print("## PART 5: Full Williams events (5378 fams) — native only")
    profiles = tbl.profiles.astype(np.int32)
    F = profiles.shape[0]

    # @ Count stored rates
    g, l, d, t = (sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length)
    for _ in range(2):
        per_branch_stats_native(sess.tree, g, l, d, t, profiles, num_threads=0)
    t0 = time.time()
    for _ in range(5):
        ev_count = per_branch_stats_native(sess.tree, g, l, d, t, profiles, num_threads=0)
    t_count = (time.time() - t0) / 5

    # @ MLE fitted rates
    rg, rl, rd, rt = (fitted_rates.gain, fitted_rates.loss, fitted_rates.dup, fitted_rates.length)
    t0 = time.time()
    for _ in range(5):
        ev_fit = per_branch_stats_native(sess.tree, rg, rl, rd, rt, profiles, num_threads=0)
    t_fit = (time.time() - t0) / 5

    print(f"  events @ Count stored rates: {t_count*1000:.1f} ms")
    print(f"  events @ MLE fitted rates:   {t_fit*1000:.1f} ms")
    print()
    print(f"  {'metric':>25s}  {'@ stored':>14s}  {'@ MLE fit':>14s}  ratio")
    fields = ["copies_node", "copies_edge", "gain_events", "loss_events", "num_families_active"]
    rows = {}
    for k in fields:
        s = float(ev_count[k].sum())
        f_ = float(ev_fit[k].sum())
        ratio = f_ / s if s != 0 else float("nan")
        rows[k] = {"at_count": s, "at_mle": f_, "ratio": ratio}
        print(f"  {('Σ ' + k):>25s}  {s:>14.2f}  {f_:>14.2f}  {ratio:>5.2f}x")

    # Top 10 branches by gain events (under MLE)
    print()
    print("  Top 10 branches by gain_events @ MLE fit:")
    order = np.argsort(-ev_fit["gain_events"])[:10]
    leaves = list(sess.tree.leaf_names) if sess.tree.leaf_names else []
    top10 = []
    for v_ in order:
        v = int(v_)
        typ = "leaf" if sess.tree.is_leaf[v] else "node"
        name = leaves[v] if (typ == "leaf" and v < len(leaves)) else f"node{v}"
        gain = float(ev_fit["gain_events"][v])
        loss = float(ev_fit["loss_events"][v])
        cn = float(ev_fit["copies_node"][v])
        nf = int(ev_fit["num_families_active"][v])
        print(f"    v={v:>3d} {typ:>4s} {name[:14]:>14s}  gain={gain:>8.2f}  loss={loss:>8.2f}  "
              f"copies@v={cn:>9.2f}  #fams={nf}")
        top10.append({"v": v, "type": typ, "name": name, "gain": gain,
                      "loss": loss, "copies_node": cn, "num_families_active": nf})

    return {
        "F": F,
        "wall_count_ms": t_count * 1000,
        "wall_mle_ms": t_fit * 1000,
        "totals": rows,
        "top10_by_gain_events_at_mle": top10,
        "events_at_count_stored": {
            k: ev_count[k].tolist() for k in fields
        },
        "events_at_mle_fitted": {
            k: ev_fit[k].tolist() for k in fields
        },
    }


def main() -> int:
    print(f"# {native_version()}")

    p1 = part1_four_leaf()
    p2, sess, tbl = part2_williams_forward()
    p3, fitted_rates = part3_ml_fit_all_backends(sess, tbl)
    p4 = part4_events_subset_vs_numpy(sess, tbl)
    p5 = part5_full_williams_events(sess, tbl, fitted_rates)

    summary = {
        "native_version": native_version(),
        "part1_4leaf_fixture": p1,
        "part2_williams_forward": p2,
        "part3_ml_fit_three_backends": p3,
        "part4_events_subset_vs_numpy": p4,
        "part5_full_williams_events": p5,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  forward LL (4-leaf):   max|native - java|         = {p1['ll_perfamily_native_vs_java_max']:.2e}")
    print(f"  gradient (4-leaf):     max|native - java|         = {p1['gradient_native_vs_java_max']:.2e}")
    print(f"  forward LL (Williams): Δ native vs Java           = {p2['delta_native_vs_java']:+.2e}")
    print(f"  ML fit (3 backends):   LL identical to            = {abs(p3['ll_native'] - p3['ll_numpy']):.2e}")
    print(f"  events (subset):       max|native - numpy| counts = "
          f"{max(p4['max_diffs_native_vs_numpy'][k] for k in ('copies_node','copies_edge','gain_events','loss_events')):.2e}")
    print(f"  full Williams events @ MLE rates: completed in {p5['wall_mle_ms']:.0f} ms")
    print()
    print(f"  wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
