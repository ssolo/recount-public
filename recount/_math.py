"""Internal log-space arithmetic and combinatorial helpers."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

NEG_INF = -np.inf


def logadd(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) tolerant of -inf."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def logsumexp(xs: np.ndarray) -> float:
    """Stable log-sum-exp over a 1-D array (-inf tolerant)."""
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return NEG_INF
    m = xs.max()
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(xs - m))))


def safelog_pair(x: np.ndarray, x_complement: np.ndarray) -> np.ndarray:
    """log(x) using whichever of (x, 1-x_complement) is more accurate."""
    out = np.empty_like(x)
    for i in range(len(x)):
        if x[i] <= 0.0:
            out[i] = NEG_INF
        elif x[i] < x_complement[i]:
            out[i] = np.log(x[i])
        else:
            out[i] = np.log1p(-x_complement[i])
    return out


class LogFactorial:
    """Cache log(n!) for n up to some max, computed via gammaln on demand."""

    def __init__(self, max_n: int = 64) -> None:
        self._cap = max_n
        self._cache = gammaln(np.arange(max_n + 1, dtype=np.float64) + 1.0)

    def factln(self, n: int) -> float:
        if n < 0:
            return NEG_INF
        if n > self._cap:
            self._cap = 2 * max(self._cap, n)
            self._cache = gammaln(np.arange(self._cap + 1, dtype=np.float64) + 1.0)
        return float(self._cache[n])


class LogRisingFactorial:
    """log Γ(κ+n)/Γ(κ) for n = 0, 1, ..., — i.e. log((κ)·(κ+1)·...·(κ+n-1))."""

    def __init__(self, kappa: float, max_n: int = 64) -> None:
        self._kappa = float(kappa)
        self._cap = max_n
        idx = np.arange(max_n + 1, dtype=np.float64)
        self._cache = gammaln(self._kappa + idx) - gammaln(self._kappa)

    def factln(self, n: int) -> float:
        if n < 0:
            return NEG_INF
        if n > self._cap:
            self._cap = 2 * max(self._cap, n)
            idx = np.arange(self._cap + 1, dtype=np.float64)
            self._cache = gammaln(self._kappa + idx) - gammaln(self._kappa)
        return float(self._cache[n])
