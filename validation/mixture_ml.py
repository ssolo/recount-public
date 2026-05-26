"""K>1 LogisticShift mixture ML / MAP fitting driver.

Uses the analytical mixture gradient from recount.logistic_shift_gradient
to optimize the (base_rates, delta_dup, delta_length, alphas) joint
parameter vector. Per-category category rate ``rates_k`` derived from
base by ``derive_category_rates``; mixing weights via softmax on alphas.

Usage:
  PYTHONPATH=. python3 validation/mixture_ml.py --dataset dpann80 --K 2
  PYTHONPATH=. python3 validation/mixture_ml.py --dataset all --K 2 --out-dir validation/outputs/mixture_K2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from recount.ml import default_initial_rates
from recount.rates import GLDRates
from recount.logistic_shift_gradient import mixture_gradient_native
from validation._shared import (
    DATASETS, LOG_RATE_CLIP, MAX_PROB_NOT1,
    _dup_jacobian, _dup_to_logit, _logit_to_dup,
    load_dataset, write_outputs,
)


def pack_params(base_gain, base_dup, base_length, delta_dup, delta_length, alphas,
                root: int, subcritical: bool = False):
    """Pack free parameters into a flat vector.

    Layout:
      [0:N]                     log(base_gain) per node
      [N:2N-1]                  log(base_dup) per non-root node OR
                                logit(base_dup) when ``subcritical=True``
      [2N-1:3N-2]               log(base_length) per non-root node
      [3N-2:3N-2+(K-1)]         delta_dup[1:K]   (delta_dup[0]=0 by id.)
      [3N-2+(K-1):3N-2+2(K-1)]  delta_length[1:K]
      [3N-2+2(K-1):3N-2+3(K-1)] alphas[1:K]

    With ``subcritical=True`` the ``base_dup`` block uses Csurös' Java
    Logistic transform (``dup = MAX_PROB_NOT1 · sigmoid(θ)``) so that
    ``dup ≤ MAX_PROB_NOT1`` is enforced unconstrained. Matches the
    single-component constraint in ``_shared.rates_to_x``.
    """
    N = base_gain.shape[0]; K = len(delta_dup)
    mask = np.ones(N, dtype=bool); mask[root] = False
    if subcritical:
        dup_block = _dup_to_logit(base_dup[mask])
    else:
        dup_block = np.log(np.clip(base_dup[mask], 1e-300, np.inf))
    parts = [
        np.log(np.clip(base_gain, 1e-300, np.inf)),
        dup_block,
        np.log(np.clip(base_length[mask], 1e-300, np.inf)),
        delta_dup[1:].astype(np.float64),
        delta_length[1:].astype(np.float64),
        alphas[1:].astype(np.float64),
    ]
    return np.concatenate(parts)


def unpack_params(x, N, K, base_template, root, subcritical: bool = False):
    """Inverse of pack_params. Returns (rates, delta_dup, delta_length, alphas).

    With ``subcritical=True`` the ``base_dup`` block is in logit space.
    """
    x_safe = np.clip(x, -LOG_RATE_CLIP, LOG_RATE_CLIP)
    mask = np.ones(N, dtype=bool); mask[root] = False
    base_gain = np.exp(x_safe[:N])
    base_dup = base_template.dup.copy()
    if subcritical:
        base_dup[mask] = _logit_to_dup(x[N:2*N - 1])
    else:
        base_dup[mask] = np.exp(x_safe[N:2*N - 1])
    base_length = base_template.length.copy()
    base_length[mask] = np.exp(x_safe[2*N - 1:3*N - 2])
    off = 3 * N - 2
    delta_dup = np.concatenate([[0.0], x[off:off + (K-1)]])
    off += (K-1)
    delta_length = np.concatenate([[0.0], x[off:off + (K-1)]])
    off += (K-1)
    alphas = np.concatenate([[0.0], x[off:off + (K-1)]])
    return base_gain, base_dup, base_length, delta_dup, delta_length, alphas


def make_objgrad_mixture(tree, base_template, profiles, min_copies, K,
                         mu_gain: float, mu_dup: float, mu_length: float,
                         sigma: float | None,
                         subcritical: bool = False,
                         num_threads: int = 0):
    """Negative log-posterior (or -LL if sigma is None) + analytical gradient.

    Returns (f, grad) where grad has the packed layout of pack_params.

    Prior (when sigma is not None): log(base_rate) ~ N(mu, sigma) on gain
    and length blocks. On the dup block: prior is on ``log(base_dup)`` in
    the unbounded regime; in ``subcritical=True`` regime the prior shifts
    to ``θ_dup ~ N(mu_dup, sigma)`` (logit-space), matching
    ``_shared.make_objgrad_map``. ``delta_dup[k] ~ N(0, sigma_delta=1.0)``,
    ``delta_length[k] ~ N(0, sigma_delta=1.0)``, ``alphas[k] ~ N(0,
    sigma_alpha=5.0)``. Mild priors on the shift/weight parameters to
    discourage extreme category rates.
    """
    N = tree.num_nodes; root = tree.root
    mask = np.ones(N, dtype=bool); mask[root] = False
    use_prior = sigma is not None and sigma > 0
    inv_sig2 = 1.0 / (sigma * sigma) if use_prior else 0.0
    inv2sig2 = 0.5 * inv_sig2
    # Mild prior on shifts/weights to break invariance.
    SIGMA_SHIFT = 1.0
    SIGMA_ALPHA = 5.0
    inv_sig2_shift = 1.0 / (SIGMA_SHIFT ** 2)
    inv_sig2_alpha = 1.0 / (SIGMA_ALPHA ** 2)

    def f_and_g(x):
        bg, bd, bl, dd, dl, al = unpack_params(x, N, K, base_template, root,
                                               subcritical=subcritical)
        mg = mixture_gradient_native(
            tree, bg, base_template.loss, bd, bl, dd, dl, al, profiles,
            min_copies=min_copies, num_threads=num_threads)
        # Objective = -LL_corr_mix (- log prior if use_prior)
        f = -mg.LL_corr_mix

        log_g = x[:N]
        # Dup block is θ (logit) in subcritical regime, else log(dup).
        dup_block = x[N:2*N - 1]
        log_t = x[2*N - 1:3*N - 2]
        off = 3 * N - 2
        d_shifts_dup = x[off:off + (K-1)]
        off += (K-1)
        d_shifts_len = x[off:off + (K-1)]
        off += (K-1)
        alphas_free = x[off:off + (K-1)]

        if use_prior:
            dev_g = log_g - mu_gain
            dev_d = dup_block - mu_dup
            dev_t = log_t - mu_length
            log_prior = -inv2sig2 * (dev_g @ dev_g + dev_d @ dev_d + dev_t @ dev_t)
            log_prior += -0.5 * inv_sig2_shift * (d_shifts_dup @ d_shifts_dup + d_shifts_len @ d_shifts_len)
            log_prior += -0.5 * inv_sig2_alpha * (alphas_free @ alphas_free)
            f -= log_prior

        # Gradients: pack analytical mg derivatives into the layout.
        out = np.empty_like(x)
        # base_gain (log space): ∂(-LL)/∂log_gain[v] = -(gain[v] * ∂LL/∂gain[v])
        out[:N] = -(bg * mg.g_base_gain)
        if subcritical:
            # θ → dup via Java Logistic: ∂dup/∂θ = dup·(1 - dup/MAX_PROB_NOT1).
            dup_jac = _dup_jacobian(bd[mask])
            out[N:2*N - 1] = -(dup_jac * mg.g_base_dup[mask])
        else:
            out[N:2*N - 1] = -(bd[mask] * mg.g_base_dup[mask])
        out[2*N - 1:3*N - 2] = -(bl[mask] * mg.g_base_length[mask])
        off = 3 * N - 2
        # delta_dup[1:]: gradient is already direct (not log-space).
        out[off:off + (K-1)] = -mg.g_delta_dup[1:]
        off += (K-1)
        out[off:off + (K-1)] = -mg.g_delta_length[1:]
        off += (K-1)
        out[off:off + (K-1)] = -mg.g_alpha[1:]

        if use_prior:
            # Prior gradients: ∂(-log_prior)/∂x[v] = inv_sig2 * (x[v] - mu).
            # In subcritical mode the dup block lives in logit space, so the
            # prior is on θ directly (no extra chain-rule factor).
            out[:N]            += inv_sig2 * (log_g - mu_gain)
            out[N:2*N - 1]     += inv_sig2 * (dup_block - mu_dup)
            out[2*N - 1:3*N - 2] += inv_sig2 * (log_t - mu_length)
            out[3*N - 2:3*N - 2 + (K-1)]                 += inv_sig2_shift * d_shifts_dup
            out[3*N - 2 + (K-1):3*N - 2 + 2*(K-1)]       += inv_sig2_shift * d_shifts_len
            out[3*N - 2 + 2*(K-1):3*N - 2 + 3*(K-1)]     += inv_sig2_alpha * alphas_free

        return f, out
    return f_and_g


def _run_one_start(objgrad, x0, max_cycles, cycle_iters, optimizer, label):
    """Warm-restart cycling for a single init. Returns (best_obj, best_x)."""
    t0 = time.time()
    best_obj = np.inf; best_x = x0.copy()
    x = x0.copy()
    for cyc in range(max_cycles):
        if optimizer == "native_bfgs":
            from recount.native_backend import bfgs_native
            x_work, fun, nit, _status = bfgs_native(
                x, objgrad, gtol=1e-7, max_iters=cycle_iters)
            x_final = x_work
            ginf = float(np.max(np.abs(objgrad(x_final)[1])))
        else:
            r = minimize(objgrad, x, jac=True, method="BFGS",
                         options={"maxiter": cycle_iters, "gtol": 1e-7, "disp": False})
            fun, x_final, nit = float(r.fun), r.x.copy(), int(r.nit)
            try:
                ginf = float(np.max(np.abs(r.jac))) if r.jac is not None else float("nan")
            except Exception:
                ginf = float("nan")
        wall = time.time() - t0
        print(f"    [{label}] cyc {cyc+1:2d}: obj={fun:.4f}  iters={nit}  "
              f"|g|={ginf:.2e}  wall={wall:.0f}s", flush=True)
        if fun < best_obj:
            best_obj = float(fun); best_x = x_final.copy()
        if nit == 0:
            break
        # Restart from the best-so-far so non-monotone overshoots near the
        # logit boundary don't degrade the warm-restart sequence.
        x = best_x.copy()
    return best_obj, best_x


def fit_one(label: str, K: int, out_dir: Path, sigma: float | None,
            max_cycles: int, cycle_iters: int, seed: int,
            num_starts: int = 1,
            subcritical: bool = False,
            optimizer: str = "BFGS",
            num_threads: int = 0) -> dict:
    print(f"\n========== {label} K={K} (sigma={sigma}, bounded={subcritical}) ==========")
    tree, profiles, csuros_rates, mc, biology = load_dataset(label)
    print(f"  F={profiles.shape[0]:,}, N={tree.num_nodes}, K={K}")
    init = default_initial_rates(tree)

    rng = np.random.default_rng(seed)
    objgrad = make_objgrad_mixture(
        tree, init, profiles, mc, K,
        mu_gain=float(np.log(0.1)),
        mu_dup=(float(_dup_to_logit(np.array([0.5]))[0]) if subcritical else float(np.log(0.5))),
        mu_length=0.0,
        sigma=sigma,
        subcritical=subcritical,
        num_threads=num_threads)

    overall_best_obj = np.inf
    overall_best_x = None
    for s in range(num_starts):
        # Per-start shifts: small perturbation to break symmetry.
        delta_dup0 = np.zeros(K); delta_length0 = np.zeros(K); alphas0 = np.zeros(K)
        for k in range(1, K):
            delta_dup0[k] = float(rng.normal(0, 0.3))
            delta_length0[k] = float(rng.normal(0, 0.3))
            alphas0[k] = 0.0
        if s == 0:
            base_for_init = init
        else:
            # Per-start lognormal perturbation around the default init; clip
            # dup ≤ 1 so the subcritical pack stays inside the logit's domain.
            from validation._shared import perturb_init
            base_for_init = perturb_init(init, 0.3, rng, subcritical=subcritical)
        x0 = pack_params(base_for_init.gain, base_for_init.dup, base_for_init.length,
                         delta_dup0, delta_length0, alphas0,
                         tree.root, subcritical=subcritical)
        start_label = f"{label}-K{K}-s{s+1}"
        obj, x_best = _run_one_start(objgrad, x0, max_cycles, cycle_iters,
                                     optimizer, start_label)
        if obj < overall_best_obj:
            overall_best_obj = obj; overall_best_x = x_best
            print(f"  ★ new best obj = {obj:.4f}")

    # Recover params at best point
    bg, bd, bl, dd, dl, al = unpack_params(overall_best_x, tree.num_nodes, K,
                                           init, tree.root, subcritical=subcritical)
    rates = GLDRates(tree=tree, gain=bg, loss=init.loss.copy(), dup=bd, length=bl)
    mg = mixture_gradient_native(tree, bg, init.loss, bd, bl, dd, dl, al, profiles,
                                 min_copies=mc, num_threads=num_threads)
    p_mix = np.exp(al - np.max(al)); p_mix /= p_mix.sum()

    print(f"\n  Final K={K} mixture ML/MAP:")
    print(f"    LL_corr_mix = {mg.LL_corr_mix:.4f}")
    print(f"    L(0)_mix    = {mg.L0_mix:.6f}")
    print(f"    p_mix       = {p_mix}")
    print(f"    delta_dup    = {dd}")
    print(f"    delta_length = {dl}")
    print(f"    max(base_dup)= {float(np.max(bd)):.12f}  (cap = {MAX_PROB_NOT1:.12f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{label}_K{K}_final.npz",
             base_gain=bg, base_loss=init.loss, base_dup=bd, base_length=bl,
             delta_dup=dd, delta_length=dl, alphas=al, p_mix=p_mix)
    with open(out_dir / f"{label}_K{K}_summary.json", "w") as fh:
        json.dump({
            "label": label, "K": K, "F": int(profiles.shape[0]),
            "N": int(tree.num_nodes), "leaves": int(tree.num_leaves), "min_copies": int(mc),
            "final": {"ll": float(mg.LL_corr_mix), "L0": float(mg.L0_mix),
                      "max_base_dup": float(np.max(bd)),
                      "max_base_gain": float(np.max(bg))},
            "rates": {"max_dup": float(np.max(bd)), "max_gain": float(np.max(bg)),
                      "max_length": float(np.max(bl[np.isfinite(bl)]))},
            "LL_corr_mix": float(mg.LL_corr_mix), "L0_mix": float(mg.L0_mix),
            "p_mix": p_mix.tolist(),
            "delta_dup": dd.tolist(), "delta_length": dl.tolist(),
            "alphas": al.tolist(),
            "sigma_prior": sigma,
            "fit_settings": {"max_cycles": max_cycles, "cycle_iters": cycle_iters,
                             "seed": seed, "num_starts": num_starts,
                             "subcritical": subcritical, "optimizer": optimizer,
                             "num_threads": num_threads},
        }, fh, indent=2)
    return {"label": label, "K": K, "LL_corr_mix": float(mg.LL_corr_mix),
            "L0_mix": float(mg.L0_mix), "p_mix": p_mix.tolist(),
            "max_base_dup": float(np.max(bd))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--sigma", type=float, default=1.0,
                   help="MAP prior σ on log(base_rate); 0 or negative = pure ML.")
    p.add_argument("--cycle-iters", type=int, default=100)
    p.add_argument("--max-cycles", type=int, default=10)
    p.add_argument("--num-starts", type=int, default=1)
    p.add_argument("--seed", type=int, default=2031)
    p.add_argument("--bounded", action="store_true",
                   help="Enforce dup ≤ MAX_PROB_NOT1 = 1 − 2⁻³⁰ on the base_dup "
                        "block via Csurös' Java Logistic transform (matches the "
                        "single-component ML / MAP bounded regime in _shared.py). "
                        "delta_dup remains in log-space.")
    p.add_argument("--optimizer", choices=["BFGS", "native_bfgs"], default="BFGS",
                   help="'native_bfgs' = bit-faithful C port of Csurös' Java dfpmin; "
                        "'BFGS' = scipy BFGS + Wolfe line search.")
    p.add_argument("--num-threads", type=int, default=0,
                   help="Native gradient thread count (0 = env var or all cores).")
    p.add_argument("--out-dir", type=Path, default=Path("validation/outputs/mixture"))
    args = p.parse_args(argv)
    sigma = args.sigma if args.sigma > 0 else None
    import os as _os
    if args.num_threads > 0:
        _os.environ["RECOUNT_NUM_THREADS"] = str(args.num_threads)
    labels = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    t0 = time.time()
    for label in labels:
        fit_one(label, args.K, args.out_dir, sigma, args.max_cycles, args.cycle_iters,
                args.seed, num_starts=args.num_starts,
                subcritical=args.bounded, optimizer=args.optimizer,
                num_threads=args.num_threads)
    print(f"\nTotal wall: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
