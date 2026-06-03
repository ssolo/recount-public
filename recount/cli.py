"""Command-line interface to recount.

Subcommands:

    recount analyze  — END-TO-END: tree + count table → rates + per-branch
                       gain/loss events + per-family per-node posteriors.
                       Writes a CountXML (Java-readable) + two CSVs.
    recount ll       — compute the (corrected) log-likelihood of a fitted model
    recount fit      — fit GLD rates by maximum likelihood
    recount events   — output per-branch copy and gain/loss posterior expectations

The ``ll``, ``fit``, and ``events`` subcommands take a Count session XML
(``.countxml`` or ``.countxml.gz``) as input. ``analyze`` is the one-shot
pipeline that takes a Newick tree and a TSV count table — defaults match
the typical "fit base GLD, condition on family-absent" workflow.

Run ``recount <subcommand> -h`` for the full option list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from recount.io.countxml import load_countxml


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _load_session(path: Path, session_id: Optional[str] = None):
    sessions = load_countxml(path)
    if session_id is not None:
        if session_id not in sessions:
            raise SystemExit(f"session {session_id!r} not in {path}; available: {list(sessions)}")
        return sessions[session_id]
    # default: the first session that has a tree+model
    for sid, s in sessions.items():
        if s.rates is not None:
            return s
    # no session has rates; just return the first one (rates will be None)
    return next(iter(sessions.values()))


def _pick_table(session, table_name: Optional[str]):
    if not session.tables:
        raise SystemExit("no <table> found in this session")
    if table_name is None:
        return next(iter(session.tables.values()))
    if table_name not in session.tables:
        raise SystemExit(
            f"table {table_name!r} not in this session; available: {list(session.tables)}"
        )
    return session.tables[table_name]


def _maybe_subset(profiles: np.ndarray, max_sum: Optional[int], max_families: Optional[int]) -> np.ndarray:
    if max_sum is not None:
        profiles = profiles[profiles.sum(axis=1) <= max_sum]
    if max_families is not None and max_families < profiles.shape[0]:
        profiles = profiles[:max_families]
    return profiles


def _logdiffexp(log_a: float, log_b: float) -> float:
    """Return log(exp(log_a) - exp(log_b)) for log_a >= log_b."""
    if log_a < log_b:
        if np.isclose(log_a, log_b, rtol=1e-12, atol=1e-12):
            return float(-np.inf)
        raise ValueError(f"logdiffexp requires log_a >= log_b, got {log_a} < {log_b}")
    if log_a == log_b:
        return float(-np.inf)
    return float(log_a + np.log1p(-np.exp(log_b - log_a)))


def _require_binary_for_correction(tree, min_copies: int) -> None:
    """The Ωmin >= 2 sampling-bias correction (Csurös SI Theorems 3-5, in
    recount_unobserved.c) only supports fully binary trees. A multifurcation
    — most often the trifurcating root of an unrooted tree, e.g. a raw
    IQ-TREE .treefile — makes the L(0) inside pass bail out and silently
    return log L(0) = -inf, which degenerates the corrected likelihood.
    Fail loudly here rather than emit a silently wrong result."""
    if min_copies < 2:
        return
    bad = [v for v in range(tree.num_nodes)
           if len(tree.children[v]) not in (0, 2)]
    if not bad:
        return
    v = bad[0]
    nc = len(tree.children[v])
    raise SystemExit(
        f"error: tree node {v} has {nc} child(ren) — the Ωmin >= {min_copies} "
        f"sampling-bias correction needs a fully binary rooted tree "
        f"({len(bad)} non-binary internal node(s) found).\n"
        f"  A node with 3+ children is usually an unrooted tree (a raw "
        f"IQ-TREE .treefile has a trifurcating root) — root the tree first. "
        f"Or pass --min-copies 1 to skip the Ωmin >= 2 correction."
    )


def _add_common_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path,
                   help="Count session XML (.countxml or .countxml.gz)")
    p.add_argument("--session", default=None,
                   help="pick a specific session by id (default: first with a rate model)")
    p.add_argument("--table", default=None,
                   help="pick a specific table by name (default: first table in the session)")
    p.add_argument("--max-sum", type=int, default=None,
                   help="filter to families with total copy count ≤ MAX_SUM")
    p.add_argument("--max-families", type=int, default=None,
                   help="cap on number of families used (after sum filter)")
    p.add_argument("--min-copies", type=int, default=1,
                   help="observation-bias correction Ωmin (0=raw, 1=exclude empty; "
                        "native paths support arbitrary nonnegative integers)")


def _add_native_thread_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-threads", type=int, default=0,
                   help="native worker threads (0=auto; RECOUNT_NUM_THREADS also works)")


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------


def cmd_ll(args) -> int:
    from recount.native_backend import (
        corrected_log_likelihood_native,
        log_likelihood_native,
        unobserved_logL0_native,
    )

    sess = _load_session(args.input, args.session)
    if sess.rates is None:
        raise SystemExit("the chosen session has no rate model; pass --session or fit one first")
    if args.min_copies < 0:
        raise SystemExit("--min-copies must be >= 0")
    if args.num_threads < 0:
        raise SystemExit("--num-threads must be >= 0")
    tbl = _pick_table(sess, args.table)
    profiles = _maybe_subset(tbl.profiles, args.max_sum, args.max_families)
    print(f"# tree: {sess.tree.num_leaves} leaves / {sess.tree.num_nodes} nodes")
    print(f"# table: {tbl.name}  ({profiles.shape[0]} of {tbl.profiles.shape[0]} families)")
    print(f"# min_copies: {args.min_copies}")
    _require_binary_for_correction(sess.tree, args.min_copies)
    ll_raw = float(log_likelihood_native(
        sess.tree,
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length,
        profiles,
        num_threads=args.num_threads,
    ).sum())
    print(f"LL_raw            = {ll_raw:.10f}")
    ll_corr = corrected_log_likelihood_native(
        sess.tree,
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length,
        profiles,
        min_copies=args.min_copies,
        num_threads=args.num_threads,
    )
    print(f"LL_corrected_min{args.min_copies} = {ll_corr:.10f}")
    ll_empty = unobserved_logL0_native(
        sess.tree,
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length,
        1,
    )
    ll_empty_or_singleton = unobserved_logL0_native(
        sess.tree,
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length,
        2,
    )
    print(f"LL_empty          = {ll_empty:.10f}")
    print(f"LL_singleton      = {_logdiffexp(ll_empty_or_singleton, ll_empty):.10f}")
    return 0


def cmd_fit(args) -> int:
    from recount.ml import default_initial_rates, fit_rates, fit_mixture, fit_rates_bootstrap

    sess = _load_session(args.input, args.session)
    tbl = _pick_table(sess, args.table)
    profiles = _maybe_subset(tbl.profiles, args.max_sum, args.max_families)
    print(f"# tree: {sess.tree.num_leaves} leaves / {sess.tree.num_nodes} nodes", file=sys.stderr)
    print(f"# table: {tbl.name}  ({profiles.shape[0]} of {tbl.profiles.shape[0]} families)", file=sys.stderr)
    print(f"# min_copies: {args.min_copies}", file=sys.stderr)
    _require_binary_for_correction(sess.tree, args.min_copies)
    if args.soft_landing:
        scheme = (f"Brownian-MAP via soft-landing — Phase-0 global γ,λ + "
                  f"per-branch lengths, then σ-annealed Brownian phases "
                  f"[{args.sigma_schedule}]")
    elif args.rate_variation == "logistic-shift":
        scheme = f"LogisticShift K={args.k} mixture (cold ML)"
    elif args.w_bootstrap is not None:
        scheme = "bounded-W bootstrap (cold ML)"
    elif args.prior == "brownian":
        scheme = (f"Brownian-MAP (tree-autocorrelated log-rate prior, "
                  f"σ={args.sigma_brownian:g})")
    else:
        scheme = "cold maximum-likelihood"
    print(f"# optimizer: scipy L-BFGS-B  ·  {args.backend} gradient backend"
          f"  ·  {scheme}", file=sys.stderr)

    if args.warm_start and sess.rates is not None:
        init = sess.rates
        print("# starting from the rates in the input file", file=sys.stderr)
    else:
        init = default_initial_rates(
            sess.tree,
            gain=args.init_gain, loss=args.init_loss,
            dup=args.init_dup, length=args.init_length,
        )
        print(f"# starting from defaults (gain={args.init_gain}, loss={args.init_loss}, "
              f"dup={args.init_dup}, length={args.init_length})", file=sys.stderr)

    if args.soft_landing:
        from recount.ml import soft_landing as _soft_landing, FitResult as _FitResult
        sigma_schedule = tuple(float(s) for s in args.sigma_schedule.split(","))
        print(f"# soft-landing schedule: σ ∈ {list(sigma_schedule)}", file=sys.stderr)
        rates_out, sl_history = _soft_landing(
            sess.tree, profiles, min_copies=args.min_copies,
            sigma_schedule=sigma_schedule,
            phase_max_iter=args.phase_max_iter,
            backend=args.backend, verbose=True,
        )
        fr = _FitResult(
            rates=rates_out,
            log_likelihood=float(sl_history[-1].get('MAP_obj', sl_history[-1].get('LL', 0.0))),
            n_iter=sum(int(h.get('iters', 0)) for h in sl_history),
            converged=bool(sl_history[-1].get('converged', False)),
            message=f"soft-landing, {len(sl_history)} phases",
            runtime_s=sum(float(h.get('time_s', 0.0)) for h in sl_history),
            history=sl_history,
        )
        shift_out = None
    elif args.rate_variation == "logistic-shift":
        print(f"# rate variation: LogisticShift K={args.k}", file=sys.stderr)
        fr = fit_mixture(
            sess.tree, profiles,
            K=args.k,
            initial_rates=init,
            fix_loss=args.fix_loss,
            fix_root_length=not args.unfix_root_length,
            bound_dup_by_loss=args.bound_dup_by_loss,
            bound_gain_by_loss=args.bound_gain_by_loss,
            min_copies=args.min_copies,
            max_iter=args.max_iter, tol=args.tol,
            verbose=True,
        )
        shift_out = fr.shift
    elif args.w_bootstrap is not None:
        w_schedule = [int(w) for w in args.w_bootstrap.split(",")]
        iters_each = (
            [int(n) for n in args.w_bootstrap_iters.split(",")]
            if args.w_bootstrap_iters else None
        )
        print(f"# w-bootstrap schedule: {w_schedule}", file=sys.stderr)
        fr = fit_rates_bootstrap(
            sess.tree, profiles,
            w_schedule=w_schedule, max_iter_each=iters_each,
            initial_rates=init,
            fix_loss=args.fix_loss,
            fix_root_length=not args.unfix_root_length,
            bound_dup_by_loss=args.bound_dup_by_loss,
            bound_gain_by_loss=args.bound_gain_by_loss,
            min_copies=args.min_copies, tol=args.tol,
            backend=args.backend, verbose=True,
        )
        shift_out = None
    else:
        fr = fit_rates(
            sess.tree, profiles,
            initial_rates=init,
            fix_loss=args.fix_loss,
            fix_root_length=not args.unfix_root_length,
            bound_dup_by_loss=args.bound_dup_by_loss,
            bound_gain_by_loss=args.bound_gain_by_loss,
            min_copies=args.min_copies,
            prior=args.prior, sigma_brownian=args.sigma_brownian,
            max_iter=args.max_iter, tol=args.tol,
            backend=args.backend, verbose=True,
        )
        shift_out = None

    print(f"# converged={fr.converged}  iters={fr.n_iter}  runtime={fr.runtime_s:.1f}s", file=sys.stderr)
    print(f"# final LL = {fr.log_likelihood:.6f}", file=sys.stderr)
    if shift_out is not None:
        print(f"# weights : {[f'{w:.4f}' for w in shift_out.weights.tolist()]}", file=sys.stderr)
        print(f"# mod_p   : {[f'{w:+.4f}' for w in shift_out.mod_p.tolist()]}", file=sys.stderr)
        print(f"# mod_q   : {[f'{w:+.4f}' for w in shift_out.mod_q.tolist()]}", file=sys.stderr)

    # Output rates as a tab-delimited table on stdout
    leaf_names = list(sess.tree.leaf_names) if sess.tree.leaf_names else []
    print("node\ttype\tname\tlength\tgain\tloss\tdup")
    for v in range(sess.tree.num_nodes):
        is_leaf = sess.tree.is_leaf[v]
        nm = leaf_names[v] if is_leaf and v < len(leaf_names) else f"node{v}"
        t = fr.rates.length[v]
        print(f"{v}\t{'leaf' if is_leaf else 'node'}\t{nm}\t"
              f"{('inf' if np.isinf(t) else f'{t:.6f}')}\t"
              f"{fr.rates.gain[v]:.6e}\t{fr.rates.loss[v]:.6e}\t{fr.rates.dup[v]:.6e}")

    if args.out_json is not None:
        payload = {
            "converged": fr.converged,
            "n_iter": fr.n_iter,
            "log_likelihood": fr.log_likelihood,
            "runtime_s": fr.runtime_s,
            "history": fr.history,
            "rates": {
                "gain": fr.rates.gain.tolist(),
                "loss": fr.rates.loss.tolist(),
                "dup": fr.rates.dup.tolist(),
                "length": [
                    (None if np.isinf(x) else float(x)) for x in fr.rates.length
                ],
            },
        }
        if shift_out is not None:
            payload["rate_variation"] = {
                "kind": "logistic-shift",
                "weights": shift_out.weights.tolist(),
                "mod_p": shift_out.mod_p.tolist(),
                "mod_q": shift_out.mod_q.tolist(),
            }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"# wrote {args.out_json}", file=sys.stderr)
    return 0


def cmd_analyze(args) -> int:
    """End-to-end: tree + count table → rates + branch events + per-family posteriors.

    Output files (all under --out-prefix):
        <prefix>.countxml.gz   — Java-compatible session XML
        <prefix>.branches.csv  — per-node summary (rates, copies, events, families)
        <prefix>.families.csv  — per-family per-node copies + presence probability
    """
    import time
    from recount.io.newick import read_newick
    from recount.io.table import read_profile_table
    from recount.io.countxml_writer import write_countxml
    from recount.ml import default_initial_rates, fit_rates
    from recount.native_backend import (
        corrected_log_likelihood_native,
        per_branch_stats_native,
        per_family_posteriors_native,
        unobserved_logL0_native,
    )
    from recount.rates import GLDRates

    out_prefix = args.out_prefix
    print(f"# recount analyze — tree={args.tree}, table={args.table}", file=sys.stderr)

    # 1. Read tree + table
    tree, init_lengths, internal_names = read_newick(args.tree)
    leaf_names = list(tree.leaf_names)
    print(f"#   tree: {tree.num_leaves} leaves, {tree.num_nodes} nodes (root={tree.root})", file=sys.stderr)
    _root_kids = len(tree.children[tree.root])
    if _root_kids > 2:
        raise SystemExit(
            f"error: this tree is unrooted — its root has {_root_kids} "
            f"children. recount analyze reconstructs the root as the "
            f"ancestor, so it needs a ROOTED tree.\n"
            f"  Root it (on an outgroup, or by midpoint) and retry — a raw "
            f"IQ-TREE .treefile is unrooted by default.")
    family_names, profiles = read_profile_table(args.table, leaf_names)
    F = profiles.shape[0]
    sums = profiles.sum(axis=1)
    min_sum_in_table = int(sums.min())
    print(f"#   table: {F} families, profile sum range [{min_sum_in_table}, {sums.max()}]", file=sys.stderr)

    # Auto-detect --min-copies if the user didn't override (= -1):
    #   the right conditioning matches the table's minimum profile sum.
    #   E.g. a file named "*-min4.txt" typically has min sum = 4, which
    #   means singletons/doublets/triplets are already excluded from the
    #   observed data and the unobserved-profile correction must integrate
    #   over them (Ωmin = 4). Fitting with --min-copies 1 on such a table
    #   misrepresents the conditioning and yields a degenerate L(0).
    if args.min_copies < 0:
        args.min_copies = max(1, min_sum_in_table)
        print(f"#   auto-detected --min-copies = {args.min_copies} (table min profile sum)", file=sys.stderr)
    elif args.min_copies < min_sum_in_table:
        print(f"#   WARNING: --min-copies {args.min_copies} < table min profile sum "
              f"{min_sum_in_table} — model assumes empty/singleton/etc are *observed* "
              f"but they aren't in this filtered table. Expect a degenerate L(0).",
              file=sys.stderr)

    # Ωmin >= 2 needs a fully binary tree (see _require_binary_for_correction).
    _require_binary_for_correction(tree, args.min_copies)
    if args.backend != "native" and args.min_copies > 2:
        raise SystemExit(
            f"error: --min-copies {args.min_copies} (> 2) requires --backend "
            f"native; the numpy/torch backends only support Ωmin <= 2.")

    # 2. Fit rates (default: base GLD, fix loss=1, root_length=+inf, min_copies=1)
    init = default_initial_rates(
        tree,
        gain=args.init_gain, loss=args.init_loss,
        dup=args.init_dup, length=args.init_length,
    )
    if args.soft_landing:
        scheme_desc = (f"Brownian-MAP via soft-landing — Phase-0 global γ,λ + "
                       f"per-branch lengths, then σ-annealed Brownian phases "
                       f"[{args.sigma_schedule}]")
    elif args.prior == "brownian":
        scheme_desc = (f"Brownian-MAP — tree-autocorrelated log-rate prior "
                       f"(σ={args.sigma_brownian:g}), no rate variation")
    else:
        scheme_desc = "cold maximum-likelihood (no prior, no rate variation)"
    print(f"#   optimizer: scipy L-BFGS-B  ·  {args.backend} gradient backend  ·  "
          f"{scheme_desc}", file=sys.stderr)
    print(f"#   constraints: loss fixed at {args.init_loss}, root edge length +inf, "
          f"dup <= loss, gain unbounded", file=sys.stderr)
    print(f"#   fitting at Ωmin={args.min_copies}, max_iter={args.max_iter}, "
          f"tol={args.tol} ...", file=sys.stderr)
    t0 = time.time()
    # `bound_gain_by_loss` is OFF by default: Csurös' own published rates
    # have gain up to 10^14 (Pólya κ encoding effectively-Poisson regimes
    # as q→1), so clipping gain ≤ loss=1 clips off his fit's stationary
    # region. `bound_dup_by_loss` stays ON because dup>loss has no
    # biological interpretation and Csurös' rates respect it.
    if args.soft_landing:
        from recount.ml import soft_landing as _soft_landing, FitResult as _FitResult
        sigma_schedule = tuple(float(s) for s in args.sigma_schedule.split(","))
        rates_out, sl_history = _soft_landing(
            tree, profiles, min_copies=args.min_copies,
            sigma_schedule=sigma_schedule,
            phase_max_iter=args.phase_max_iter,
            backend=args.backend, verbose=True,
        )
        fr = _FitResult(
            rates=rates_out,
            log_likelihood=float(sl_history[-1].get('MAP_obj', sl_history[-1].get('LL', 0.0))),
            n_iter=sum(int(h.get('iters', 0)) for h in sl_history),
            converged=bool(sl_history[-1].get('converged', False)),
            message=f"soft-landing, {len(sl_history)} phases",
            runtime_s=sum(float(h.get('time_s', 0.0)) for h in sl_history),
            history=sl_history,
        )
    else:
        fr = fit_rates(
            tree, profiles,
            initial_rates=init,
            fix_loss=True, fix_root_length=True,
            bound_dup_by_loss=True, bound_gain_by_loss=False,
            min_copies=args.min_copies,
            prior=args.prior, sigma_brownian=args.sigma_brownian,
            max_iter=args.max_iter, tol=args.tol,
            backend=args.backend, verbose=False,
        )
    print(f"#   fit done: LL={fr.log_likelihood:.4f}, converged={fr.converged}, "
          f"iters={fr.n_iter}, time={time.time()-t0:.1f}s", file=sys.stderr)
    rates = fr.rates

    # 3. Compute per-branch + per-family posteriors at the fitted rates
    print(f"#   computing per-branch + per-family posteriors...", file=sys.stderr)
    t0 = time.time()
    stats = per_branch_stats_native(tree, rates.gain, rates.loss, rates.dup, rates.length, profiles)
    pf_copies, pf_present = per_family_posteriors_native(
        tree, rates.gain, rates.loss, rates.dup, rates.length, profiles)
    log_L0 = unobserved_logL0_native(tree, rates.gain, rates.loss, rates.dup, rates.length, args.min_copies)
    print(f"#   posteriors done in {time.time()-t0:.1f}s", file=sys.stderr)

    # 4. Apply L(0) correction to copies + families_present per node
    L0 = float(np.exp(log_L0))
    if args.min_copies >= 1 and L0 > 0:
        from recount.native_backend import unobserved_inside_tensors_native
        from recount.unobserved_outside import (
            compute_unobserved_outside, compute_unobserved_posteriors,
            _get_survival_arrays,
        )
        C, K, _ = unobserved_inside_tensors_native(
            tree, rates.gain, rates.loss, rates.dup, rates.length, args.min_copies)
        sp = _get_survival_arrays(tree, rates.gain, rates.loss, rates.dup, rates.length)
        B_all, J_all, _, _, _ = compute_unobserved_outside(tree, sp, K, args.min_copies)
        log_node_post, _ = compute_unobserved_posteriors(C, K, B_all, J_all, log_L0, args.min_copies - 1)
        unobs_factor = F * L0 / (1.0 - L0) if L0 < 1.0 else 0.0
        n_grid = np.arange(args.min_copies, dtype=np.float64)
        Pn = np.exp(log_node_post)
        E_xi_unobs = (Pn * n_grid[None, :]).sum(axis=1)
        present_unobs = 1.0 - Pn[:, 0]
        copies_corr = stats["copies_node"] + unobs_factor * E_xi_unobs
        present_corr = stats["families_present"] + unobs_factor * present_unobs
    else:
        copies_corr = stats["copies_node"].copy()
        present_corr = stats["families_present"].copy()

    # 5. Output CountXML
    countxml_path = Path(f"{out_prefix}.countxml.gz")
    write_countxml(
        countxml_path, tree, rates, family_names, profiles,
        internal_names=internal_names, session_id="recount-analyze",
        table_name=Path(args.table).name,
    )
    print(f"#   wrote {countxml_path}", file=sys.stderr)

    # 6. Output branches.csv
    branches_path = Path(f"{out_prefix}.branches.csv")
    with open(branches_path, "w") as fh:
        fh.write("node_index,node_name,is_leaf,parent_index,parent_name,branch_length,"
                 "gain_rate,loss_rate,dup_rate,"
                 "copies_node_observed,copies_node_corrected,copies_edge_observed,"
                 "gain_events,loss_events,"
                 "families_present_observed,families_present_corrected,families_active_thresh50\n")
        for v in range(tree.num_nodes):
            is_leaf = bool(tree.is_leaf[v])
            p = int(tree.parent[v])
            nm = internal_names[v] if internal_names and internal_names[v] else (
                leaf_names[v] if is_leaf and v < len(leaf_names) else f"node{v}"
            )
            pnm = internal_names[p] if (p >= 0 and internal_names and internal_names[p]) else (
                f"node{p}" if p >= 0 else ""
            )
            length = rates.length[v]
            length_s = "inf" if np.isinf(length) else f"{length:.6g}"
            fh.write(f"{v},{nm},{is_leaf},{p if p >= 0 else ''},{pnm},{length_s},"
                     f"{rates.gain[v]:.6g},{rates.loss[v]:.6g},{rates.dup[v]:.6g},"
                     f"{stats['copies_node'][v]:.6f},{copies_corr[v]:.6f},{stats['copies_edge'][v]:.6f},"
                     f"{stats['gain_events'][v]:.6f},{stats['loss_events'][v]:.6f},"
                     f"{stats['families_present'][v]:.6f},{present_corr[v]:.6f},{int(stats['num_families_active'][v])}\n")
    print(f"#   wrote {branches_path}", file=sys.stderr)

    # 7. Output families.csv  (long-format: family_name, node_index, ...)
    families_path = Path(f"{out_prefix}.families.csv")
    with open(families_path, "w") as fh:
        fh.write("family_name,node_index,node_name,is_leaf,E_copies,P_present\n")
        # For compactness only write entries with E_copies > 1e-6 OR P_present > 1e-6
        # — for F=5378, N=119 that's still up to 640k rows.
        thresh = args.posterior_threshold
        for f_idx in range(F):
            for v in range(tree.num_nodes):
                c = pf_copies[f_idx, v]
                p_pres = pf_present[f_idx, v]
                if c < thresh and p_pres < thresh:
                    continue
                is_leaf = bool(tree.is_leaf[v])
                nm = internal_names[v] if internal_names and internal_names[v] else (
                    leaf_names[v] if is_leaf and v < len(leaf_names) else f"node{v}"
                )
                fh.write(f"{family_names[f_idx]},{v},{nm},{is_leaf},{c:.6f},{p_pres:.6f}\n")
    print(f"#   wrote {families_path}", file=sys.stderr)

    # 8. Summary on stdout
    print(f"# Optimizer: {scheme_desc}  ·  {args.backend} backend")
    if args.prior == "brownian":
        print(f"# Fitted MAP objective (corrected LL + Brownian prior, "
              f"Ωmin={args.min_copies}): {fr.log_likelihood:.4f}")
    else:
        print(f"# Fitted LL (Ωmin={args.min_copies}): {fr.log_likelihood:.4f}")
    print(f"# L(0) = {L0:.6f}  (1 - L(0) = {1 - L0:.4f})")
    print(f"# Root (node {tree.root}, name {internal_names[tree.root]}):"
          f" copies = {copies_corr[tree.root]:.2f}, families_present = {present_corr[tree.root]:.2f}")
    return 0


def cmd_events(args) -> int:
    from recount.events import BranchStats, format_branch_table
    from recount.native_backend import per_branch_stats_native

    sess = _load_session(args.input, args.session)
    if sess.rates is None:
        raise SystemExit("the chosen session has no rate model; fit one with `recount fit` first")
    if args.num_threads < 0:
        raise SystemExit("--num-threads must be >= 0")
    tbl = _pick_table(sess, args.table)
    profiles = _maybe_subset(tbl.profiles, args.max_sum, args.max_families)
    print(f"# tree: {sess.tree.num_leaves} leaves / {sess.tree.num_nodes} nodes", file=sys.stderr)
    print(f"# table: {tbl.name}  ({profiles.shape[0]} of {tbl.profiles.shape[0]} families)", file=sys.stderr)

    native_stats = per_branch_stats_native(
        sess.tree,
        sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length,
        profiles,
        num_threads=args.num_threads,
    )
    stats = BranchStats(
        copies_node=native_stats["copies_node"],
        copies_edge=native_stats["copies_edge"],
        gain_events=native_stats["gain_events"],
        loss_events=native_stats["loss_events"],
        num_families_active=native_stats["num_families_active"],
        gain=sess.rates.gain.copy(),
        loss=sess.rates.loss.copy(),
        dup=sess.rates.dup.copy(),
        length=sess.rates.length.copy(),
    )
    print(format_branch_table(sess.tree, stats))

    if args.out_json is not None:
        payload = {
            "copies_node": stats.copies_node.tolist(),
            "copies_edge": stats.copies_edge.tolist(),
            "gain_events": stats.gain_events.tolist(),
            "loss_events": stats.loss_events.tolist(),
            "families_present": native_stats["families_present"].tolist(),
            "families_active": stats.num_families_active.tolist(),
            "gain_rate": stats.gain.tolist(),
            "loss_rate": stats.loss.tolist(),
            "dup_rate": stats.dup.tolist(),
            "length": [None if np.isinf(x) else float(x) for x in stats.length],
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"# wrote {args.out_json}", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------
# argparse plumbing
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recount", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # ll
    p_ll = sub.add_parser("ll", help="compute the corrected log-likelihood of a fitted model")
    _add_common_input_args(p_ll)
    _add_native_thread_arg(p_ll)
    p_ll.set_defaults(func=cmd_ll)

    # fit
    p_fit = sub.add_parser("fit", help="fit GLD rates by maximum likelihood")
    _add_common_input_args(p_fit)
    p_fit.add_argument("--warm-start", action="store_true",
                       help="start from the rates already in the input file (default: defaults)")
    p_fit.add_argument("--init-gain", type=float, default=0.1)
    p_fit.add_argument("--init-loss", type=float, default=1.0)
    p_fit.add_argument("--init-dup", type=float, default=0.5)
    p_fit.add_argument("--init-length", type=float, default=1.0)
    p_fit.add_argument("--fix-loss", action="store_true", default=True,
                       help="hold loss rate fixed at its initial value (default: True, Williams convention)")
    p_fit.add_argument("--no-fix-loss", dest="fix_loss", action="store_false",
                       help="also optimize the loss rate")
    p_fit.add_argument("--unfix-root-length", action="store_true",
                       help="also optimize the root edge length (default: held at +inf)")
    p_fit.add_argument("--max-iter", type=int, default=200)
    p_fit.add_argument("--tol", type=float, default=1e-6)
    p_fit.add_argument("--backend", choices=["numpy", "torch", "native"], default="native",
                       help="gradient backend (default: native — the only backend that "
                            "supports min_copies ≥ 2 correctly, and the fastest. The "
                            "torch backend silently produced a wrong correction at "
                            "min_copies ≥ 3 in pre-2026-05-19 versions; it now raises "
                            "NotImplementedError there. The numpy backend is a pure-"
                            "Python reference, slow but useful for testing.)")
    p_fit.add_argument("--no-bound-dup", dest="bound_dup_by_loss",
                       action="store_false", default=True,
                       help="disable the biological dup ≤ loss bound")
    p_fit.add_argument("--no-bound-gain", dest="bound_gain_by_loss",
                       action="store_false", default=True,
                       help="disable the biological gain ≤ loss bound")
    p_fit.add_argument("--prior", choices=["none", "brownian"], default="brownian",
                       help="rate regulariser (default: brownian). 'brownian' adds a "
                            "tree-Brownian autocorrelated log-rate prior "
                            "(Thorne-Kishino-Painter) coupling each branch's rates to "
                            "its parent's — a MAP fit that shares statistical strength "
                            "across neighbouring branches; 'none' is plain maximum "
                            "likelihood. Ignored when --rate-variation or --w-bootstrap "
                            "is set (those paths are cold-ML only).")
    p_fit.add_argument("--sigma-brownian", type=float, default=1.0,
                       help="Brownian prior std on each per-edge log-rate increment "
                            "(default 1.0); larger → weaker shrinkage, σ→∞ recovers "
                            "plain ML.")
    p_fit.add_argument("--rate-variation", choices=["none", "logistic-shift"], default="none",
                       help="rate-variation model layered on top of per-edge GLD")
    p_fit.add_argument("--k", "-K", type=int, default=2,
                       help="number of categories when --rate-variation logistic-shift (default: 2)")
    p_fit.add_argument("--w-bootstrap", type=str, default=None,
                       help="bounded-W bootstrap schedule, e.g. '8,16,32' — fit on cumulative "
                            "subsets of increasing max profile sum; only applies to rate fits "
                            "(no rate-variation)")
    p_fit.add_argument("--w-bootstrap-iters", type=str, default=None,
                       help="per-phase iter counts for --w-bootstrap, e.g. '40,20,10'")
    p_fit.add_argument("--soft-landing", action="store_true",
                       help="use the annealed Brownian soft-landing optimizer "
                            "(Phase-0 global γ,λ + per-branch lengths, then a "
                            "series of Brownian-MAP phases with σ annealed "
                            "through --sigma-schedule). Recommended for "
                            "large datasets (≥ several hundred taxa) where "
                            "the cold-start basin tends to collapse to "
                            "root=0; overrides --prior / --rate-variation.")
    p_fit.add_argument("--sigma-schedule", default="0.05,0.15,0.3,0.6,1.0",
                       help="comma-separated σ values for the Brownian phases "
                            "(only relevant with --soft-landing).")
    p_fit.add_argument("--phase-max-iter", type=int, default=60,
                       help="L-BFGS-B max_iter per soft-landing phase (default 60).")
    p_fit.add_argument("--out-json", type=Path, default=None,
                       help="if set, also write the fitted rates + history to this JSON file")
    p_fit.set_defaults(func=cmd_fit)

    # analyze (end-to-end pipeline)
    p_an = sub.add_parser(
        "analyze",
        help="end-to-end: tree+TSV → rates + per-branch gain/loss events + per-family per-node posteriors")
    p_an.add_argument("tree", type=Path,
                      help="Newick tree file (or - for stdin)")
    p_an.add_argument("table", type=Path,
                      help="TSV count table: 'family_name<TAB>leaf_1<TAB>...<TAB>leaf_N'"
                           " — first column is the family name, remaining columns must match leaf labels")
    p_an.add_argument("--out-prefix", "-o", type=Path, required=True,
                      help="output filename prefix; writes <prefix>.countxml.gz, "
                           "<prefix>.branches.csv, <prefix>.families.csv")
    p_an.add_argument("--min-copies", type=int, default=-1,
                      help="observation-bias correction. Default: auto-detect from the "
                           "table's minimum profile sum (e.g. files named *-min4.txt "
                           "will use 4). Pass an explicit integer to override; a warning "
                           "fires if the override is smaller than the table's actual min sum.")
    p_an.add_argument("--init-gain", type=float, default=0.1)
    p_an.add_argument("--init-loss", type=float, default=1.0)
    p_an.add_argument("--init-dup", type=float, default=0.5)
    p_an.add_argument("--init-length", type=float, default=1.0)
    p_an.add_argument("--max-iter", type=int, default=200)
    p_an.add_argument("--tol", type=float, default=1e-6)
    p_an.add_argument("--backend", choices=["numpy", "torch", "native"],
                      default="native",
                      help="gradient backend for the fit (default: native — "
                           "the only backend that supports Ωmin >= 3)")
    p_an.add_argument("--prior", choices=["none", "brownian"], default="brownian",
                      help="rate regulariser (default: brownian). 'brownian' adds a "
                           "tree-Brownian autocorrelated log-rate prior "
                           "(Thorne-Kishino-Painter) coupling each branch's rates to "
                           "its parent's — a MAP fit that shares statistical strength "
                           "across neighbouring branches; 'none' is plain maximum "
                           "likelihood.")
    p_an.add_argument("--sigma-brownian", type=float, default=1.0,
                      help="Brownian prior std on each per-edge log-rate increment "
                           "(default 1.0); larger → weaker shrinkage, σ→∞ recovers "
                           "plain ML.")
    p_an.add_argument("--soft-landing", action="store_true",
                      help="use the annealed Brownian-MAP pipeline (Phase-0 fits "
                           "a single global γ, λ + per-branch lengths; then σ is "
                           "annealed through --sigma-schedule). Recommended for "
                           "large datasets (≥ several hundred taxa) or annotations "
                           "with extreme per-cell paralogy, where a cold-start "
                           "Brownian fit collapses to a degenerate root.")
    p_an.add_argument("--sigma-schedule", default="0.05,0.15,0.3,0.6,1.0",
                      help="comma-separated σ values for the annealed phases "
                           "(only relevant with --soft-landing).")
    p_an.add_argument("--phase-max-iter", type=int, default=60,
                      help="L-BFGS-B max_iter per soft-landing phase (default 60).")
    p_an.add_argument("--posterior-threshold", type=float, default=1e-3,
                      help="suppress per-family rows where E[copies] and P(present) "
                           "are both below this threshold (default 1e-3 — keeps the CSV "
                           "compact; set 0 to write every node for every family)")
    p_an.set_defaults(func=cmd_analyze)

    # events
    p_ev = sub.add_parser(
        "events",
        help="per-branch copy and gain/loss posterior expectations under a fitted model")
    _add_common_input_args(p_ev)
    _add_native_thread_arg(p_ev)
    p_ev.add_argument("--out-json", type=Path, default=None,
                      help="if set, also write the per-branch stats to this JSON file")
    p_ev.set_defaults(func=cmd_events)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
