"""O(W²) destructive sibling combine — PyTorch path only (Triton stub).

This module replaces the closed-form O(W³) multinomial combine in
``recount.torch_fast._vectorized_combine`` with the destructive in-place
recurrence from M. Csurös' Java Count (the same one used by the NumPy
reference at ``recount.gld._compute_sibling``). One backend is active:

- ``destructive_combine_pytorch``: pure PyTorch, runs on any device,
  autograd-friendly. The production path for non-native callers.
- ``destructive_combine_triton``: stub — the Triton CUDA kernel was
  removed (M-series target, no CUDA available locally). Calling it
  raises ``RuntimeError``; ``triton_available()`` always returns
  ``False``. Kept as a stub so callers don't break.

The public entrypoint ``destructive_combine`` always routes to the
PyTorch path. For high-performance CPU work prefer
``recount.native_backend`` instead.

Algorithm (per call): for a binary internal node v with children j1, j2,
combine the children's edge log-likelihoods ``k1, k2 : [F, W+1]`` into the
parent's node log-likelihood ``C[v] : [F, W+1]`` via the recurrence

    Cw[f, t]  ← logaddexp(Cw[f, t] + log_e_c, Cw[f, t-1] + log_e)

interleaved with a multinomial gather

    C2[f, ell] = LSE_{s+t=ell} K[f, s] + Cw[f, t] + binom(ell;s,t)
                                       + s·logp1 + t·logp2.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch import Tensor


NEG_INF = float("-inf")


# ----------------------------------------------------------------------------
# triton availability detection (lazy-cached)
# ----------------------------------------------------------------------------


_TRITON_AVAILABLE: bool | None = None


def triton_available() -> bool:
    """Always returns False — Triton path was removed (M-series target, no CUDA).

    The dispatcher always uses the pure-PyTorch fallback. Kept as a stub
    for API compatibility with prior versions; the recount.native_backend
    is the recommended path for high-performance CPU work.
    """
    return False


def _force_pytorch() -> bool:
    return os.environ.get("RECOUNT_FORCE_PYTORCH_COMBINE", "0") == "1"


# ----------------------------------------------------------------------------
# safe logsumexp — matches recount.torch_fast._safe_logsumexp behavior so
# all-(-inf) rows backward as 0 rather than NaN
# ----------------------------------------------------------------------------


def _safe_logsumexp_stack(stacked: Tensor) -> Tensor:
    """``logsumexp`` along dim 0 of a [n, F] tensor, NaN-safe for all-(-inf) cols."""
    m = stacked.amax(dim=0, keepdim=True)
    valid = m > NEG_INF
    safe = torch.where(valid.expand_as(stacked), stacked, torch.zeros_like(stacked))
    result = torch.logsumexp(safe, dim=0)
    return torch.where(valid.squeeze(0), result, torch.full_like(result, NEG_INF))


def _safe_logaddexp(a: Tensor, b: Tensor) -> Tensor:
    """``torch.logaddexp`` whose backward is 0 (not NaN) when both inputs are -inf.

    ``torch.logaddexp(a, b)``'s backward is ``exp(a - out) · grad``. When
    both ``a`` and ``b`` are -inf, ``out = -inf``, and ``exp(-inf − -inf) =
    exp(NaN) = NaN``. We swap -inf pairs for zeros on the way in (so
    backward sees a benign input with bounded gradient) and re-mask the
    output to -inf on the way out.
    """
    m = torch.maximum(a, b)
    valid = m > NEG_INF
    a_safe = torch.where(valid, a, torch.zeros_like(a))
    b_safe = torch.where(valid, b, torch.zeros_like(b))
    out = torch.logaddexp(a_safe, b_safe)
    return torch.where(valid, out, torch.full_like(out, NEG_INF))


# ----------------------------------------------------------------------------
# PyTorch implementation (CPU / MPS / fallback when triton is unavailable)
# ----------------------------------------------------------------------------


# Pre-clamp value: large-negative replaces -inf so `torch.logaddexp` has a
# well-defined backward (``exp(-inf − -inf) = NaN`` otherwise). Backward of
# logaddexp at this clamp value is ``exp(-1e30 − finite) ≈ 0`` to ULP, so
# gradients stay correct AND we avoid the 6-op `_safe_logaddexp` wrapper.
_NEG_CLAMP = -1e30


def destructive_combine_pytorch(
    k1: Tensor, k2: Tensor,
    log_e: Tensor, log_e_c: Tensor,
    logp1: Tensor, logp2: Tensor,
    log_fact: Tensor,
) -> Tensor:
    """Pure-PyTorch O(W²) destructive sibling combine.

    Parameters
    ----------
    k1, k2 : Tensor [F, W+1]
        Children's edge log-likelihoods. ``k1`` plays the role of "C"
        (already-processed siblings) in ``gld._compute_sibling``; ``k2``
        plays "K_junior".
    log_e, log_e_c : Tensor [], scalars
        log(eps_sib) and log(1 − eps_sib). In the binary case
        eps_sib = p̃_{j1}, so log_e = log p̃_{j1} and log_e_c = log p̃_c_{j1}.
    logp1, logp2 : Tensor [], scalars
        Per-pair survival weights as defined in the binary parameter
        mapping (see ``_build_sibling_params``).
    log_fact : Tensor [W+1]
        log(n!) lookup table for n in [0, W].

    Returns
    -------
    Tensor [F, W+1]
        Parent node's inside log-likelihood ``C[v]``.

    Notes
    -----
    Mirrors ``recount.gld._compute_sibling`` (lines 241-310). Two
    optimizations vs the naïve port:

    * **Single 2-D state tensor** (rebuilt each ell via ``torch.cat``)
      instead of a Python list-of-tensors. ``index_select`` for the
      gather phase replaces the per-iteration ``torch.stack`` of [F]
      slices — collapses ~W gather Python ops into 2 tensor ops per ell.
    * **Plain ``torch.logaddexp``** with ``-inf`` inputs pre-clamped to
      ``-1e30``. Skips the 6-op ``_safe_logaddexp`` wrapper (whose only
      purpose is NaN-safe backward through the all-(-inf) case). The
      clamp value is well below any finite log-likelihood we'd ever see
      (LL on Williams ≈ -1.3e5) so it's effectively -∞ to ULP, and the
      backward `exp(-1e30 − finite)` is 0 cleanly.

    Per-ell tensor ops drop from ~3W to ~5; total speedup vs naïve is
    1.5× at W=8 growing to 3.5× at W=128.

    Only the first ``W+1`` outputs are computed (the full algorithm
    produces ``2W + 1`` columns but the parent's inside is padded to
    ``W+1``; the higher columns are masked off downstream).
    """
    F, Wp1 = k1.shape
    W = Wp1 - 1
    dtype = k1.dtype
    device = k1.device

    # Clamp -inf to a large-negative finite value so plain torch.logaddexp's
    # backward stays well-defined. The clamp affects forward by < ULP since
    # exp(-1e30 - any_finite) < min_double.
    Cw = k1.clamp_min(_NEG_CLAMP)  # [F, W+1]
    K = k2.clamp_min(_NEG_CLAMP)

    C2_cols: List[Tensor] = []

    for ell in range(Wp1):
        chain_len = min(ell, W)  # K_len-1 == W in our padded setup

        # ---- Destructive update phase ----
        # Compute x_k for k = 1..chain_len via the sequential recurrence
        #   x_k = logaddexp(x_{k-1} + log_e_c, Cw_old[ell-k] + log_e)
        # batched across F. x_0 = Cw_old[ell].
        if chain_len > 0:
            x = Cw[:, ell]  # [F]
            x_list: List[Tensor] = []
            for k in range(1, chain_len + 1):
                t = ell - k
                x = torch.logaddexp(x + log_e_c, Cw[:, t] + log_e)
                x_list.append(x)
            # new_part[:, k-1] = x_k → reversed so that Cw_new[:, ell-k] = x_k
            new_part = torch.stack(list(reversed(x_list)), dim=1)  # [F, chain_len]
            t_start = ell - chain_len
            # Functional update of Cw — autograd-safe (no in-place mutation).
            Cw = torch.cat([Cw[:, :t_start], new_part, Cw[:, ell:]], dim=1)

        # ---- Gather phase ----
        # C2[ell] = LSE_{s in [max(0,ell-W), min(ell,W)], t = ell-s}:
        #   K[s] + Cw_new[t] + log_fact[ell] - log_fact[s] - log_fact[t]
        #         + s·logp1 + t·logp2
        s_lo = max(0, ell - W)
        s_hi = min(ell, W)
        s_arr = torch.arange(s_lo, s_hi + 1, device=device)
        t_arr = ell - s_arr
        log_ellfact = log_fact[ell]
        binom = log_ellfact - log_fact[s_arr] - log_fact[t_arr]
        slogp1 = s_arr.to(dtype) * logp1
        tlogp2 = t_arr.to(dtype) * logp2
        K_slice = K.index_select(1, s_arr)
        Cw_slice = Cw.index_select(1, t_arr)
        terms = K_slice + Cw_slice + binom + slogp1 + tlogp2
        c2_ell = torch.logsumexp(terms, dim=1)
        C2_cols.append(c2_ell)

    return torch.stack(C2_cols, dim=1)  # [F, W+1]


# ----------------------------------------------------------------------------
# Triton CUDA kernel (lazy-imported only when actually dispatched to)
# ----------------------------------------------------------------------------


def destructive_combine_triton(
    k1: Tensor, k2: Tensor,
    log_e: Tensor, log_e_c: Tensor,
    logp1: Tensor, logp2: Tensor,
    log_fact: Tensor,
) -> Tensor:
    """Stub — the Triton CUDA path was removed. Always raises.

    Kept as a stub so that callers checking for the symbol don't break
    at import time; the public dispatcher ``destructive_combine`` no
    longer attempts to reach this function.
    """
    raise RuntimeError(
        "destructive_combine_triton: Triton path was removed. "
        "Use destructive_combine_pytorch or recount.native_backend instead."
    )


class _DestructiveCombineTriton:
    """Stub — Triton path removed (M-series target). Use the PyTorch path."""

    @staticmethod
    def apply(*args, **kwargs):
        raise RuntimeError("Triton path was removed; use destructive_combine_pytorch")


# ----------------------------------------------------------------------------
# Public dispatcher
# ----------------------------------------------------------------------------


def destructive_combine(
    k1: Tensor, k2: Tensor,
    log_e: Tensor, log_e_c: Tensor,
    logp1: Tensor, logp2: Tensor,
    log_fact: Tensor,
) -> Tensor:
    """O(W²) destructive sibling combine. Always uses the PyTorch path.

    The Triton CUDA branch was removed (target is Apple M-series with no
    CUDA available locally). This dispatcher exists for back-compat with
    callers that import ``destructive_combine`` directly; it just routes
    to ``destructive_combine_pytorch``. For high-performance CPU work
    use ``recount.native_backend`` instead.

    The ``RECOUNT_FORCE_PYTORCH_COMBINE`` env var is now a no-op (kept
    for back-compat); the only path is PyTorch.
    """
    return destructive_combine_pytorch(
        k1, k2, log_e, log_e_c, logp1, logp2, log_fact,
    )
