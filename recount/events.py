"""Per-branch posterior expectations.

Once a model is fit, you can ask: under the posterior given the data, how
many copies pass through each node? How many gain / loss / duplication
events happen on each branch? How many families have at least one such
event there?

This module collects those statistics by running the inside-outside pass
(``recount.gld._forward`` + ``_compute_outside``) family-by-family and
accumulating the per-node and per-edge posterior expectations.

The reported per-edge quantities (matching the ``BranchStats``
dataclass below, which mirrors a subset of count.model.Posteriors):

  * ``copies_node``   = Σ_f E[ξ̃_v | Ξ_f]            # surviving copies at v after gain
  * ``copies_edge``   = Σ_f E[η̃_v | Ξ_f]            # incoming copies from parent before gain
  * ``gain_events``   = copies_node − copies_edge   # net gain on this edge (expected)
  * ``loss_events``   = Σ_f (E[ξ̃_x] − E[η̃_v])     # parent-copies that didn't make it
                          where x = parent of v
  * ``num_families_active``  = # families with P(ξ̃_v > 0 | Ξ_f) > 0.5

A separate per-edge ``dup_events`` (E[duplications]) is NOT currently
reported — the bookkeeping for the joint posterior of (parent count,
child count, # gain, # dup) isn't implemented. Add only if a downstream
analysis actually requires it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from recount.gld import (
    _compute_outside,
    _forward,
    _make_caches,
    compute_survival_params,
)
from recount.rates import GLDRates
from recount.tree import Tree


@dataclass
class BranchStats:
    """Per-node summary statistics."""
    # per-node arrays of length num_nodes
    copies_node: np.ndarray   # Σ_f E[S_v | Ξ_f]
    copies_edge: np.ndarray   # Σ_f E[S_v_from_parent | Ξ_f]
    gain_events: np.ndarray   # copies_node - copies_edge
    loss_events: np.ndarray   # parent's copies that didn't make it through this edge
    num_families_active: np.ndarray  # # families with P(S_v > 0 | Ξ_f) > 0.5
    # per-edge rate parameters (lifted from the fit for the report)
    gain: np.ndarray
    loss: np.ndarray
    dup: np.ndarray
    length: np.ndarray


def per_branch_stats(tree: Tree, rates: GLDRates, profiles: np.ndarray,
                     active_prob_threshold: float = 0.5) -> BranchStats:
    """Compute per-branch posterior statistics under the given fitted model.

    Iterates families one-by-one (inside + outside) and accumulates the
    posterior expectations described in the module docstring. Linear in
    F · N · W² overall.
    """
    sp = compute_survival_params(tree, rates)
    fact, rfacts = _make_caches(tree, sp, profiles)
    n = tree.num_nodes
    F = profiles.shape[0]

    copies_node = np.zeros(n)
    copies_edge = np.zeros(n)
    n_active = np.zeros(n, dtype=np.int64)

    for f in range(F):
        pc = _forward(tree, sp, profiles[f], fact, rfacts)
        J, B = _compute_outside(tree, sp, pc.C, pc.K, fact, rfacts)
        LL_f = pc.LL
        for v in range(n):
            # Marginal posterior over S_v
            Bv = B[v]; Cv = pc.C[v]
            L = min(Bv.size, Cv.size)
            if L == 0:
                continue
            lp = Bv[:L] + Cv[:L] - LL_f
            finite = np.isfinite(lp)
            if not finite.any():
                continue
            post = np.where(finite, np.exp(lp), 0.0)
            mean_node = float(np.sum(np.arange(L) * post))
            copies_node[v] += mean_node
            # P(S_v > 0 | Ξ) = 1 - post[0]
            if L > 0 and float(post[0]) < 1.0 - active_prob_threshold:
                n_active[v] += 1
            # Marginal posterior over S_v_from_parent (the edge into v)
            Jv = J[v]; Kv = pc.K[v]
            L2 = min(Jv.size, Kv.size)
            if L2 > 0:
                lp2 = Jv[:L2] + Kv[:L2] - LL_f
                finite2 = np.isfinite(lp2)
                if finite2.any():
                    post2 = np.where(finite2, np.exp(lp2), 0.0)
                    copies_edge[v] += float(np.sum(np.arange(L2) * post2))

    # net gain on each branch = expected difference between node and edge means
    gain_events = copies_node - copies_edge
    # loss_events: per node v, parent_copies - edge_copies (for non-root)
    loss_events = np.zeros(n)
    for v in range(n):
        if v == tree.root:
            continue
        p = int(tree.parent[v])
        loss_events[v] = max(copies_node[p] - copies_edge[v], 0.0)

    return BranchStats(
        copies_node=copies_node,
        copies_edge=copies_edge,
        gain_events=gain_events,
        loss_events=loss_events,
        num_families_active=n_active,
        gain=rates.gain.copy(),
        loss=rates.loss.copy(),
        dup=rates.dup.copy(),
        length=rates.length.copy(),
    )


def format_branch_table(tree: Tree, stats: BranchStats) -> str:
    """Pretty-print branch statistics as a tab-aligned text table."""
    leaf_names = list(tree.leaf_names) if tree.leaf_names else []
    cols = [
        "node", "type", "name",
        "len", "gain_rate", "loss_rate", "dup_rate",
        "copies_at_node", "copies_into_edge",
        "gain_events", "loss_events", "families_active",
    ]
    fmt = "{:>4}  {:>5}  {:>14}  {:>8}  {:>9}  {:>9}  {:>9}  {:>13}  {:>15}  {:>11}  {:>11}  {:>14}"
    out = [fmt.format(*cols)]
    for v in range(tree.num_nodes):
        is_leaf = tree.is_leaf[v]
        name = (leaf_names[v] if (is_leaf and v < len(leaf_names)) else
                f"node{v}")
        type_ = "leaf" if is_leaf else "node"
        if np.isinf(stats.length[v]):
            len_s = "inf"
        else:
            len_s = f"{stats.length[v]:.4f}"
        out.append(fmt.format(
            v, type_, name[:14], len_s,
            f"{stats.gain[v]:.4g}", f"{stats.loss[v]:.4g}", f"{stats.dup[v]:.4g}",
            f"{stats.copies_node[v]:.2f}", f"{stats.copies_edge[v]:.2f}",
            f"{stats.gain_events[v]:.2f}", f"{stats.loss_events[v]:.2f}",
            int(stats.num_families_active[v]),
        ))
    return "\n".join(out)
