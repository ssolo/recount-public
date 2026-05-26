"""Bucketed driver for the fast PyTorch backend.

The vectorized recursion in ``recount.torch_fast`` is fastest when every
family shares one max-width ``W``, but real datasets like Williams2017 have
a heavy-tailed distribution of profile sums (median 6, max 668). Forcing
``W = max`` would waste ≈10000× memory on the majority of families.

This module sharps that tail by:

* sorting families by total copy count,
* splitting them into contiguous buckets of similar size,
* running each bucket at its own ``W`` (a tight cover of that bucket's max),
* summing the per-family log-likelihoods.

The resulting ``LL_total`` is autograd-tracked end-to-end, so a single
``LL_total.backward()`` recovers gradients w.r.t. the rate tensors across
all families.

Public API:

    from recount.torch_fast_bucket import corrected_log_likelihood_bucketed
    LL = corrected_log_likelihood_bucketed(tree, gain, loss, dup, length,
                                            profiles, min_copies=1, buckets=…)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from recount.torch_fast import (
    compute_survival_params_t,
    empty_log_likelihood_t,
    forward_fast,
)
from recount.tree import Tree


def make_buckets(
    profiles_t: Tensor, edges: Optional[Sequence[int]] = None,
) -> List[dict]:
    """Split families into buckets by profile sum.

    Parameters
    ----------
    profiles_t : LongTensor [F, num_leaves]
        Per-family observed counts. Should be on CPU; we just compute sums
        and pad sizes here.
    edges : sequence of int, optional
        Right edges of each bucket (inclusive). Default geometric: 4, 8,
        16, 32, 64, 128, 256, 512, 1024.

    Returns
    -------
    list of dict with keys:
        "indices" — LongTensor of family indices in this bucket
        "profiles" — LongTensor [B, num_leaves] of the bucket's profiles
        "W" — int width to use for the forward pass
        "max_sum" — int max profile sum in this bucket
    """
    sums = profiles_t.sum(dim=1).cpu().numpy()
    if edges is None:
        edges = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
    edges = sorted(set(int(e) for e in edges))
    buckets: List[dict] = []
    lo = 0
    for hi in edges:
        mask = (sums > lo) & (sums <= hi)
        idx = np.flatnonzero(mask)
        if idx.size > 0:
            indices = torch.from_numpy(idx).long()
            sub = profiles_t[indices]
            buckets.append({
                "indices": indices,
                "profiles": sub,
                "W": int(hi) + 2,
                "max_sum": int(sub.sum(dim=1).max()),
            })
        lo = hi
    # the largest bucket (sums > last edge)
    idx = np.flatnonzero(sums > lo)
    if idx.size > 0:
        indices = torch.from_numpy(idx).long()
        sub = profiles_t[indices]
        max_sum = int(sub.sum(dim=1).max())
        buckets.append({
            "indices": indices,
            "profiles": sub,
            "W": max_sum + 2,
            "max_sum": max_sum,
        })
    return buckets


def corrected_log_likelihood_bucketed(
    tree: Tree,
    gain: Tensor, loss: Tensor, dup: Tensor, length: Tensor,
    profiles: Tensor,
    *, min_copies: int = 1, bucket_edges: Optional[Sequence[int]] = None,
    chunk_F: int = 1024,
    log_every: int = 0,
    use_destructive: bool = False,
) -> Tensor:
    """Corrected log-likelihood with family bucketing.

    Behaves like ``corrected_log_likelihood_fast`` but tolerates a wide
    spread of profile sums by running multiple forward passes — one per
    bucket — each at its own ``W``. ``use_destructive=True`` switches each
    bucket's combine to the O(W²) destructive path (Triton on CUDA,
    PyTorch elsewhere) instead of the closed-form O(W³) combine.
    """
    sp = compute_survival_params_t(tree, gain, loss, dup, length)
    buckets = make_buckets(profiles, bucket_edges)
    # We only need the sum across families, so just accumulate per-bucket sums.
    # This avoids ``index_copy`` (unsupported on MPS) and keeps autograd intact.
    LL = torch.zeros((), dtype=gain.dtype, device=gain.device)
    for bi, bk in enumerate(buckets):
        if log_every and (bi % log_every == 0 or bi + 1 == len(buckets)):
            print(f"  bucket {bi+1}/{len(buckets)}: {bk['profiles'].shape[0]:>5d} families  W={bk['W']}", flush=True)
        bk_profiles = bk["profiles"].to(gain.device)
        ll_bucket = forward_fast(
            tree, sp, bk_profiles, W=bk["W"], chunk_F=chunk_F,
            use_destructive=use_destructive,
        )
        LL = LL + ll_bucket.sum()
    if min_copies == 0:
        return LL
    L0 = empty_log_likelihood_t(sp)
    if min_copies == 2:
        raise NotImplementedError("min_copies=2 (singleton correction) not supported here")
    F = profiles.shape[0]
    p_obs = -torch.expm1(L0)
    return LL - F * torch.log(p_obs)
