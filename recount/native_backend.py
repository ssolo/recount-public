"""Native CPU backend — ctypes binding to ``native/librecount.{dylib,so}``.

Drop-in equivalent of ``recount.gld``, written in C with platform-tuned
threading and SIMD:

  Apple Silicon  : libdispatch threading + Accelerate vForce SIMD (default)
  Intel Mac      : libdispatch threading + Accelerate vForce SIMD
  Linux (any)    : OpenMP threading + compiler-auto-vectorized scalar math

Numerical results are identical on all paths; the portability layer is
in ``native/include/recount_parallel.h`` (the RECOUNT_PARALLEL_APPLY
macro picks libdispatch or OpenMP at compile time).

Build:
    cd native && make            # auto-detects platform
    cd native && make native     # opt-in: -mcpu=native (this machine only)

Use:
    from recount.native_backend import corrected_log_likelihood_native
    LL = corrected_log_likelihood_native(tree, gain, loss, dup, length,
                                         profiles, min_copies=1)
"""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import numpy as np

from recount.tree import Tree


def _resolve_num_threads(num_threads: int) -> int:
    """Resolve effective thread count for native gradient/forward calls.

    Precedence:
      1. ``num_threads > 0`` — used as-is (explicit override).
      2. ``num_threads == 0`` AND env var ``RECOUNT_NUM_THREADS`` set —
         parse env var (must be ≥ 0; 0 falls through to default).
      3. Default — return 0 (libdispatch / OpenMP picks ``nproc``).

    Use case: running N validation jobs in parallel, set
    ``RECOUNT_NUM_THREADS=$(($(sysctl -n hw.ncpu) / N))`` (or
    ``--num-threads`` in ``validation/ml.py`` / ``map.py``) so each job
    grabs only its fair share of cores instead of all of them.
    """
    if num_threads > 0:
        return num_threads
    env = os.environ.get("RECOUNT_NUM_THREADS")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    return 0


# ----------------------------------------------------------------------------
# Locate and dlopen the native library (.dylib on macOS, .so on Linux)
# ----------------------------------------------------------------------------


_HERE = Path(__file__).resolve().parent
_LIB_NAME = "librecount.dylib" if platform.system() == "Darwin" else "librecount.so"
_DYLIB_CANDIDATES = [
    _HERE / _LIB_NAME,
    _HERE.parent / "native" / _LIB_NAME,
    # Accept either extension — useful for cross-platform development.
    _HERE / "librecount.dylib",
    _HERE.parent / "native" / "librecount.dylib",
    _HERE / "librecount.so",
    _HERE.parent / "native" / "librecount.so",
    Path(os.environ.get("RECOUNT_NATIVE_DYLIB", "")),
]


def _load_dylib() -> ctypes.CDLL:
    last_err = None
    for path in _DYLIB_CANDIDATES:
        if not path or not path.exists():
            continue
        try:
            return ctypes.CDLL(str(path))
        except OSError as e:
            last_err = e

    # Nothing usable on disk — auto-build once from native/ if a Makefile is
    # there. Opt out by setting RECOUNT_NO_AUTOBUILD=1. We print to stderr so
    # users see a clear "I'm compiling" message (the first import then takes
    # ~3-5s on a modern Mac instead of failing outright).
    native_dir = _HERE.parent / "native"
    if (native_dir / "Makefile").exists() and not os.environ.get("RECOUNT_NO_AUTOBUILD"):
        import subprocess
        import sys
        print(
            f"recount: {_LIB_NAME} not found; building via `make -C {native_dir}` "
            "(set RECOUNT_NO_AUTOBUILD=1 to skip)…",
            file=sys.stderr,
            flush=True,
        )
        try:
            subprocess.run(
                ["make", "-C", str(native_dir)],
                check=True, capture_output=True, text=True,
            )
            built = native_dir / _LIB_NAME
            if built.exists():
                return ctypes.CDLL(str(built))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_err = e
            if isinstance(e, subprocess.CalledProcessError):
                last_err = OSError(
                    f"`make -C {native_dir}` failed (exit {e.returncode}):\n"
                    f"{(e.stderr or e.stdout or '').strip()}"
                )

    msg = (
        f"Could not load {_LIB_NAME}. Build it via `make -C native` "
        "or set RECOUNT_NATIVE_DYLIB to its path."
    )
    if last_err is not None:
        msg += f"\nLast error: {last_err}"
    raise ImportError(msg)


_lib = _load_dylib()


# ----------------------------------------------------------------------------
# ctypes signatures
# ----------------------------------------------------------------------------


class _RecountTree(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_int32),
        ("num_leaves", ctypes.c_int32),
        ("root", ctypes.c_int32),
        ("parent", ctypes.POINTER(ctypes.c_int32)),
        ("is_leaf", ctypes.POINTER(ctypes.c_uint8)),
        ("first_child", ctypes.POINTER(ctypes.c_int32)),
        ("child_list", ctypes.POINTER(ctypes.c_int32)),
    ]


class _RecountSurvival(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_int32),
        ("p", ctypes.POINTER(ctypes.c_double)),
        ("p_c", ctypes.POINTER(ctypes.c_double)),
        ("q", ctypes.POINTER(ctypes.c_double)),
        ("q_c", ctypes.POINTER(ctypes.c_double)),
        ("gain", ctypes.POINTER(ctypes.c_double)),
        ("eps", ctypes.POINTER(ctypes.c_double)),
        ("eps_c", ctypes.POINTER(ctypes.c_double)),
        ("log_p", ctypes.POINTER(ctypes.c_double)),
        ("log_p_c", ctypes.POINTER(ctypes.c_double)),
        ("log_q", ctypes.POINTER(ctypes.c_double)),
        ("log_q_c", ctypes.POINTER(ctypes.c_double)),
        ("log_gain", ctypes.POINTER(ctypes.c_double)),
        ("log_eps", ctypes.POINTER(ctypes.c_double)),
        ("log_eps_c", ctypes.POINTER(ctypes.c_double)),
        ("is_polya", ctypes.POINTER(ctypes.c_uint8)),
    ]


_lib.recount_tree_build_csr.argtypes = [
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.recount_tree_build_csr.restype = None

_lib.recount_survival_alloc.argtypes = [
    ctypes.POINTER(_RecountSurvival), ctypes.c_int32,
]
_lib.recount_survival_alloc.restype = ctypes.c_int

_lib.recount_survival_free.argtypes = [ctypes.POINTER(_RecountSurvival)]
_lib.recount_survival_free.restype = None

_lib.recount_compute_survival_params.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(ctypes.c_double),  # gain
    ctypes.POINTER(ctypes.c_double),  # loss
    ctypes.POINTER(ctypes.c_double),  # dup
    ctypes.POINTER(ctypes.c_double),  # length
    ctypes.POINTER(_RecountSurvival),
]
_lib.recount_compute_survival_params.restype = ctypes.c_int

_lib.recount_forward_scratch_size.argtypes = [ctypes.c_int32, ctypes.c_int32]
_lib.recount_forward_scratch_size.restype = ctypes.c_size_t

_lib.recount_forward_family.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_int32),    # profile
    ctypes.c_void_p,                   # scratch
    ctypes.c_size_t,                   # scratch_n
    ctypes.POINTER(ctypes.c_double),   # out_ll
]
_lib.recount_forward_family.restype = ctypes.c_int

_lib.recount_forward_batch.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_int32),    # profiles row-major
    ctypes.c_int32,                    # F
    ctypes.c_int32,                    # num_threads (0 = auto)
    ctypes.POINTER(ctypes.c_double),   # out_lls
]
_lib.recount_forward_batch.restype = ctypes.c_int

_lib.recount_gradient_batch_full.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int,                      # min_copies
    ctypes.POINTER(ctypes.c_double),   # family_weights [F] (nullable)
    ctypes.POINTER(ctypes.c_double),   # out_lls [F]
    ctypes.POINTER(ctypes.c_double),   # out_grad [3*N]
]
_lib.recount_gradient_batch_full.restype = ctypes.c_int


# tree-Brownian autocorrelated log-rate prior; see docs/brownian_prior.tex
_lib.recount_brownian_prior_and_grad.argtypes = [
    ctypes.c_int32,                    # N
    ctypes.c_int32,                    # root
    ctypes.POINTER(ctypes.c_int32),    # parent [N]
    ctypes.POINTER(ctypes.c_double),   # x [3*N - 2]
    ctypes.c_double,                   # sigma_brownian_gain
    ctypes.c_double,                   # sigma_brownian_dup
    ctypes.c_double,                   # sigma_brownian_length
    ctypes.c_double,                   # mu_root_gain
    ctypes.c_double,                   # mu_root_dup
    ctypes.c_double,                   # mu_root_length
    ctypes.c_double,                   # sigma_root
    ctypes.POINTER(ctypes.c_double),   # out_log_prior [1]
    ctypes.POINTER(ctypes.c_double),   # out_grad [3*N - 2]
]
_lib.recount_brownian_prior_and_grad.restype = ctypes.c_int


def brownian_prior_native(
    parent: np.ndarray,
    root: int,
    x: np.ndarray,
    sigma_brownian_gain: float,
    sigma_brownian_dup: float,
    sigma_brownian_length: float,
    mu_root_gain: float,
    mu_root_dup: float,
    mu_root_length: float,
    sigma_root: float,
) -> tuple[float, np.ndarray]:
    """Native C O(N) tree-Brownian prior + gradient.

    Bit-for-bit equivalent to validation/_shared.py:_brownian_prior_and_grad
    (Python reference). See docs/brownian_prior.tex.

    Parameters
    ----------
    parent : int32[N]
        parent[root] must be -1; parent[v] > v not required.
    root : int
    x : float64[3*N - 2]
        packed: log_gain[N] | dup_block[N-1] | log_length[N-1]
    sigma_brownian_* : float
        Per-axis Brownian std (positive).
    mu_root_*, sigma_root : float
        Root anchor parameters.

    Returns
    -------
    (log_prior : float, grad : float64[3*N - 2])
    """
    N = int(parent.shape[0])
    parent_c = np.ascontiguousarray(parent, dtype=np.int32)
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    if x_c.shape[0] != 3 * N - 2:
        raise ValueError(f"x must have length 3*N - 2 = {3*N - 2}, got {x_c.shape[0]}")
    out_grad = np.zeros(3 * N - 2, dtype=np.float64)
    out_lp = np.zeros(1, dtype=np.float64)
    status = _lib.recount_brownian_prior_and_grad(
        N, int(root),
        parent_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(sigma_brownian_gain),
        float(sigma_brownian_dup),
        float(sigma_brownian_length),
        float(mu_root_gain),
        float(mu_root_dup),
        float(mu_root_length),
        float(sigma_root),
        out_lp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"recount_brownian_prior_and_grad returned status {status}")
    return float(out_lp[0]), out_grad


class _RecountBranchStats(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_int32),
        ("copies_node", ctypes.POINTER(ctypes.c_double)),
        ("copies_edge", ctypes.POINTER(ctypes.c_double)),
        ("gain_events", ctypes.POINTER(ctypes.c_double)),
        ("loss_events", ctypes.POINTER(ctypes.c_double)),
        ("num_families_active", ctypes.POINTER(ctypes.c_int32)),
        ("families_present", ctypes.POINTER(ctypes.c_double)),
    ]


_lib.recount_branch_stats_alloc.argtypes = [ctypes.POINTER(_RecountBranchStats), ctypes.c_int32]
_lib.recount_branch_stats_alloc.restype = ctypes.c_int
_lib.recount_branch_stats_free.argtypes = [ctypes.POINTER(_RecountBranchStats)]
_lib.recount_branch_stats_free.restype = None
_lib.recount_branch_stats_batch.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,                   # active_threshold
    ctypes.POINTER(_RecountBranchStats),
]
_lib.recount_branch_stats_batch.restype = ctypes.c_int

_lib.recount_native_version.argtypes = []
_lib.recount_native_version.restype = ctypes.c_char_p

_lib.recount_unobserved_logL0.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.c_int,                       # min_copies
    ctypes.POINTER(ctypes.c_double),    # out_logL0
]
_lib.recount_unobserved_logL0.restype = ctypes.c_int

_lib.recount_unobserved_inside_tensors.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.c_int,                       # min_copies
    ctypes.POINTER(ctypes.c_double),    # out_Cmat
    ctypes.POINTER(ctypes.c_double),    # out_Kmat
    ctypes.POINTER(ctypes.c_double),    # out_logL0 (nullable)
]
_lib.recount_unobserved_inside_tensors.restype = ctypes.c_int


def unobserved_logL0_native(tree: Tree, gain, loss, dup, length, min_copies: int) -> float:
    """log L(0) = log P{profile sum < min_copies} via Csurös SI Thms 3-5.

    Native supports arbitrary min_copies (≥ 0); the shipped CountXXV.jar
    enforces min_copies ≤ 2 (Integer.min(2, table.minCopies()) in
    Gradient.java line 57).
    """
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        out = ctypes.c_double(0.0)
        rc = _lib.recount_unobserved_logL0(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            ctypes.c_int(int(min_copies)), ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(f"recount_unobserved_logL0 failed (code {rc})")
        return float(out.value)
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))


def unobserved_inside_tensors_native(
    tree: Tree, gain, loss, dup, length, min_copies: int,
):
    """Return (Cmat, Kmat, logL0) for the SI Theorems 3-5 inside pass.

    Cmat[u, m, n] = log P{Ω_u = m | ξ̃_u = n}  (per-size node inside)
    Kmat[u, m, s] = log P{Ω_u = m | η̃_u = s}  (per-size edge inside)

    Shape: (num_nodes, W, W) with W = min_copies. Matches Java's
    ``FamilySizeLikelihood.sizeProfiles[m].node_likelihoods[u]`` /
    ``edge_likelihoods[u]`` transposed.
    """
    if min_copies <= 0:
        raise ValueError("min_copies must be ≥ 1 for inside tensors")
    N = int(tree.num_nodes)
    W = int(min_copies)
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        Cmat = np.full((N, W, W), -np.inf, dtype=np.float64)
        Kmat = np.full((N, W, W), -np.inf, dtype=np.float64)
        out_l = ctypes.c_double(0.0)
        rc = _lib.recount_unobserved_inside_tensors(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            ctypes.c_int(int(min_copies)),
            Cmat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Kmat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(out_l),
        )
        if rc != 0:
            raise RuntimeError(f"recount_unobserved_inside_tensors failed (code {rc})")
        return Cmat, Kmat, float(out_l.value)
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))


def native_version() -> str:
    return _lib.recount_native_version().decode("utf-8")


# ----------------------------------------------------------------------------
# Python helpers
# ----------------------------------------------------------------------------


def _make_tree_struct(tree: Tree):
    """Build (tree_struct, holders) — holders keep numpy arrays alive."""
    n = int(tree.num_nodes)
    parent = np.ascontiguousarray(tree.parent, dtype=np.int32)
    is_leaf = np.ascontiguousarray(tree.is_leaf, dtype=np.uint8)
    first_child = np.zeros(n + 1, dtype=np.int32)
    child_list = np.zeros(max(0, n - 1), dtype=np.int32)
    _lib.recount_tree_build_csr(
        n, int(tree.root), parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        first_child.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        child_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    tree_struct = _RecountTree(
        num_nodes=n,
        num_leaves=int(tree.num_leaves),
        root=int(tree.root),
        parent=parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        is_leaf=is_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        first_child=first_child.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        child_list=child_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    holders = (parent, is_leaf, first_child, child_list)
    return tree_struct, holders


def _compute_survival(tree: Tree, gain, loss, dup, length):
    """Build a populated _RecountSurvival; caller must call recount_survival_free."""
    tree_struct, tree_holders = _make_tree_struct(tree)
    gain = np.ascontiguousarray(gain, dtype=np.float64)
    loss = np.ascontiguousarray(loss, dtype=np.float64)
    dup  = np.ascontiguousarray(dup,  dtype=np.float64)
    length = np.ascontiguousarray(length, dtype=np.float64)

    sp_struct = _RecountSurvival(num_nodes=int(tree.num_nodes))
    rc = _lib.recount_survival_alloc(ctypes.byref(sp_struct), int(tree.num_nodes))
    if rc != 0:
        raise MemoryError(f"recount_survival_alloc failed (code {rc})")
    rc = _lib.recount_compute_survival_params(
        ctypes.byref(tree_struct),
        gain.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dup.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        length.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(sp_struct),
    )
    if rc != 0:
        _lib.recount_survival_free(ctypes.byref(sp_struct))
        raise RuntimeError(f"recount_compute_survival_params failed (code {rc})")
    return tree_struct, tree_holders, sp_struct, (gain, loss, dup, length)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def log_likelihood_native(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    *, num_threads: int = 0,
) -> np.ndarray:
    """Per-family forward log-likelihoods via the native backend.

    Returns ``[F]`` array of LLs (raw, no min_copies correction).
    Uses libdispatch to parallelize across families.
    """
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        profiles = np.ascontiguousarray(profiles, dtype=np.int32)
        F = profiles.shape[0]
        out = np.zeros(F, dtype=np.float64)
        nt = _resolve_num_threads(int(num_threads))
        rc = _lib.recount_forward_batch(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            profiles.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(F), ctypes.c_int32(nt),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_forward_batch failed (code {rc})")
        return out
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))


def corrected_log_likelihood_native(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    *, min_copies: int = 1, num_threads: int = 0,
) -> float:
    """Corrected log-likelihood (scalar, summed over families).

    Supports arbitrary ``min_copies`` ≥ 0 via the native SI-Thms-3-5
    L(0) calculation (``recount.unobserved.unobserved_logL0_native``).
    The shipped CountXXV.jar caps ``min_copies`` at 2 (Gradient.java
    line 90 throws); native goes higher.
    """
    if min_copies < 0:
        raise ValueError("min_copies must be >= 0")
    per_family = log_likelihood_native(
        tree, gain, loss, dup, length, profiles, num_threads=num_threads)
    LL = float(per_family.sum())
    if min_copies == 0:
        return LL
    L0 = unobserved_logL0_native(tree, gain, loss, dup, length, min_copies)
    F = profiles.shape[0]
    p_obs = -np.expm1(L0)
    return LL - F * float(np.log(p_obs))


def gradient_native_weighted(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    family_weights: np.ndarray,
    *, num_threads: int = 0,
):
    """Raw-LL gradient with per-family weighting (no L(0) correction).

    Returns ``(weighted_LL_sum, weighted_grad_surv_flat)`` where:
      - ``weighted_LL_sum = Σ_f weight[f] * log L_f`` (raw, uncorrected).
      - ``weighted_grad_surv_flat`` shape ``[3 * num_nodes]``, indexed
        ``3*v + {GAIN=0, LOSS=1, DUP=2}``, equals
        ``Σ_f weight[f] * ∂log L_f / ∂(p̃, q̃, r̃/κ̃)``.

    This is the kernel needed for K>1 LogisticShift mixture gradients
    via the responsibility-weighted form (docs/logistic_shift_gradient.tex
    Prop 1): the mixture-LL gradient w.r.t. any parameter is
    ``Σ_k Σ_f γ_{f,k} ∂log L_{f,k} / ∂φ`` where ``γ_{f,k} = p_k L_{f,k} / L_f``.
    Call once per category ``k`` with ``family_weights = γ[:, k]``.

    Caller adds the L(0) correction separately if needed (mixture L(0) =
    Σ_k p_k L(0)_k and the correction is F * ∂L(0)_mix/∂φ / (1 - L(0)_mix)).
    """
    F = profiles.shape[0]
    N = int(tree.num_nodes)
    if family_weights.shape != (F,):
        raise ValueError(f"family_weights must have shape ({F},), got {family_weights.shape}")
    weights_f64 = np.ascontiguousarray(family_weights, dtype=np.float64)

    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        profiles_i32 = np.ascontiguousarray(profiles, dtype=np.int32)
        per_fam = np.zeros(F, dtype=np.float64)
        grad = np.zeros(3 * N, dtype=np.float64)
        nt = _resolve_num_threads(int(num_threads))
        rc = _lib.recount_gradient_batch_full(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            profiles_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(F), ctypes.c_int32(nt),
            ctypes.c_int(0),  # min_copies=0 — no L(0) correction (caller handles)
            weights_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            per_fam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_gradient_batch_full (weighted) failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))

    weighted_LL = float(np.dot(per_fam, weights_f64))
    return weighted_LL, grad


def gradient_native(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    *, min_copies: int = 1, num_threads: int = 0,
):
    """Forward LL + analytical gradient (survival parameterization).

    Returns ``(LL_corrected_scalar, grad_flat)`` where ``grad_flat`` has
    shape ``[3 * num_nodes]`` indexed as ``3*v + {GAIN=0, LOSS=1, DUP=2}``,
    matching Java's ``Gradient.getCorrectedGradient()`` orientation.

    For ``min_copies`` ∈ {0, 1}: fully analytical native gradient (Csurös
    2021 Corollary 9, ported to C).
    For ``min_copies`` ≥ 2: analytical gradient on the raw-LL part (same
    native C path) plus the analytical L(0) correction
    (``recount.unobserved_outside.compute_L0_gradient_analytical`` —
    Phase B port of Java FamilySizeLikelihood + LogGradient's
    PosteriorStatistics.getLogSurvivalGradient, validated against FD to
    median 1e-7 rel error on Williams2017).
    """
    if min_copies < 0:
        raise ValueError("min_copies must be >= 0")
    F = profiles.shape[0]
    N = int(tree.num_nodes)

    # For min_copies ∈ {0, 1}, native C handles everything analytically.
    # For min_copies ≥ 2, native C does only the raw-LL gradient; the L(0)
    # gradient term is added separately via compute_L0_gradient_analytical
    # (fully analytical, fully native C as of commit 7fd3425 — no finite
    # differences anywhere on this path).
    native_min_copies_for_grad = 1 if min_copies >= 1 else 0

    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        profiles_i32 = np.ascontiguousarray(profiles, dtype=np.int32)
        per_fam = np.zeros(F, dtype=np.float64)
        grad = np.zeros(3 * N, dtype=np.float64)
        nt = _resolve_num_threads(int(num_threads))
        rc = _lib.recount_gradient_batch_full(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            profiles_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(F), ctypes.c_int32(nt),
            ctypes.c_int(int(native_min_copies_for_grad)),
            None,  # family_weights = NULL (unweighted)
            per_fam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_gradient_batch failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))

    grad = grad.reshape(N, 3)  # [N, (GAIN, LOSS, DUP)]
    LL_raw = float(per_fam.sum())

    if min_copies == 0:
        return LL_raw, grad.reshape(-1)

    # If min_copies ≥ 2, need to (a) overwrite the LL correction to use the
    # full L(0), and (b) correct the gradient (the native gradient_batch used
    # min_copies=1 == only empty correction; for min_copies≥2 the L(0)
    # gradient term is different).
    L0_native = unobserved_logL0_native(tree, gain, loss, dup, length, min_copies)
    p_obs = -np.expm1(L0_native)
    LL_corr = LL_raw - F * float(np.log(p_obs))

    if min_copies >= 2:
        # Native gradient computed with min_copies=1 had a correction of
        # +F · L0_empty / (1 - L0_empty) · ∂L0_empty/∂θ baked in via Csurös
        # Cor. 9 formula. For min_copies≥2 we want
        # +F · L0_k / (1 - L0_k) · ∂L0_k/∂θ instead. Simplest: subtract the
        # baked-in min_copies=1 contribution and add the min_copies=k one.
        from recount.gld import empty_log_likelihood
        from recount.rates import GLDRates
        rates = GLDRates(tree=tree,
                         gain=np.asarray(gain, dtype=np.float64),
                         loss=np.asarray(loss, dtype=np.float64),
                         dup=np.asarray(dup, dtype=np.float64),
                         length=np.asarray(length, dtype=np.float64))
        L0_empty = float(empty_log_likelihood(tree, rates))

        # Compute ∂L(0)/∂(gain, loss, dup, length) analytically — see
        # recount.unobserved_outside.compute_L0_gradient_analytical (Phase B
        # port of Java FamilySizeLikelihood + LogGradient.getLogSurvivalGradient,
        # validated against FD to median 1e-7 rel error on Williams2017).
        from recount.unobserved_outside import compute_L0_gradient_analytical
        _, gL0_g_k, gL0_l_k, gL0_d_k, gL0_t_k = compute_L0_gradient_analytical(
            tree, gain, loss, dup, length, min_copies)
        _, gL0_g_1, gL0_l_1, gL0_d_1, gL0_t_1 = compute_L0_gradient_analytical(
            tree, gain, loss, dup, length, 1)
        # Native min_copies=1 grad already has:
        #   g_native = ∂(raw LL)/∂θ + F · ∂L(0)_1/∂θ / (1 - L(0)_1)
        # Target gradient (min_copies=k):
        #   g_target = ∂(raw LL)/∂θ + F · ∂L(0)_k/∂θ / (1 - L(0)_k)
        # so delta_grad = F · [∂L(0)_k/∂θ / (1 - L(0)_k)
        #                      − ∂L(0)_1/∂θ / (1 - L(0)_1)].
        # We pre-compute the delta here (in raw-rate space, NOT survival).
        L0_k_lin = float(np.exp(L0_native))
        L0_1_lin = float(np.exp(L0_empty))
        one_minus_L0_k = max(1.0 - L0_k_lin, 1e-300)
        one_minus_L0_1 = max(1.0 - L0_1_lin, 1e-300)
        delta_g_gain   = F * (gL0_g_k / one_minus_L0_k - gL0_g_1 / one_minus_L0_1)
        delta_g_loss   = F * (gL0_l_k / one_minus_L0_k - gL0_l_1 / one_minus_L0_1)
        delta_g_dup    = F * (gL0_d_k / one_minus_L0_k - gL0_d_1 / one_minus_L0_1)
        delta_g_length = F * (gL0_t_k / one_minus_L0_k - gL0_t_1 / one_minus_L0_1)
        grad_flat = grad.reshape(-1)
        # Store the pre-computed delta directly. The aux struct's L0_grad_*
        # fields here CARRY THE FINAL DELTA TO ADD (the legacy struct field
        # name is preserved for backward compat with ml.py; semantics changed
        # from "FD log-derivative" to "analytical ΔL(0)/ΔL(0)_1 contribution"
        # already-summed across rates).
        grad_flat = _GradientWithL0Aux(
            arr=grad_flat,
            L0_grad_gain  =delta_g_gain,
            L0_grad_loss  =delta_g_loss,
            L0_grad_dup   =delta_g_dup,
            L0_grad_length=delta_g_length,
            # Set ratio_k=ratio_1=0 so the ml.py wrapper's old formula
            # "F · (ratio_k - ratio_1) · L0_grad" becomes 0 and we can
            # instead add the delta directly. Keep these for any callers
            # that still read them.
            ratio_k=L0_k_lin / one_minus_L0_k,
            ratio_1=L0_1_lin / one_minus_L0_1,
            F=F,
        )
        # Mark struct so ml.py knows to apply the delta directly.
        grad_flat.delta_is_final = True
        return LL_corr, grad_flat

    return LL_corr, grad.reshape(-1)


class _GradientWithL0Aux(np.ndarray):
    """ndarray subclass that carries the additive Δ-gradient (analytical
    L(0)-correction) for min_copies≥2 fits. ml.py's native backend wrapper
    adds these deltas to the min_copies=1 corrected gradient to upgrade it
    to the correct min_copies=k gradient.

    Subclassed so any caller that treats the return as a plain flat
    ndarray still works — they get the min_copies=1 corrected gradient,
    which is the right shape but missing the L(0)≠empty-profile delta.
    """
    def __new__(cls, arr, L0_grad_gain, L0_grad_dup, L0_grad_length,
                ratio_k, ratio_1, F, L0_grad_loss=None):
        obj = np.asarray(arr).view(cls)
        obj.L0_grad_gain = L0_grad_gain
        obj.L0_grad_loss = L0_grad_loss
        obj.L0_grad_dup = L0_grad_dup
        obj.L0_grad_length = L0_grad_length
        obj.ratio_k = ratio_k
        obj.ratio_1 = ratio_1
        obj.F = F
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.L0_grad_gain = getattr(obj, 'L0_grad_gain', None)
        self.L0_grad_loss = getattr(obj, 'L0_grad_loss', None)
        self.L0_grad_dup = getattr(obj, 'L0_grad_dup', None)
        self.L0_grad_length = getattr(obj, 'L0_grad_length', None)
        self.ratio_k = getattr(obj, 'ratio_k', None)
        self.ratio_1 = getattr(obj, 'ratio_1', None)
        self.F = getattr(obj, 'F', None)


def per_branch_stats_native(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    *, active_threshold: float = 0.5, num_threads: int = 0,
):
    """Per-branch posterior event counts via the native backend.

    Returns a dict with the same fields as ``recount.events.BranchStats``:
    ``copies_node``, ``copies_edge``, ``gain_events``, ``loss_events``,
    ``num_families_active`` — all numpy arrays of length ``num_nodes``.

    Computed via the same forward + outside pipeline as ``gradient_native``,
    threaded across families. ~25 s in NumPy on Williams → expected <100 ms here.
    """
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    bs_struct = _RecountBranchStats()
    try:
        N = int(tree.num_nodes)
        rc = _lib.recount_branch_stats_alloc(ctypes.byref(bs_struct), N)
        if rc != 0:
            raise MemoryError(f"branch_stats alloc failed (code {rc})")
        profiles = np.ascontiguousarray(profiles, dtype=np.int32)
        F = profiles.shape[0]
        nt = _resolve_num_threads(int(num_threads))
        rc = _lib.recount_branch_stats_batch(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            profiles.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(F), ctypes.c_int32(nt),
            ctypes.c_double(float(active_threshold)),
            ctypes.byref(bs_struct),
        )
        if rc != 0:
            raise RuntimeError(f"branch_stats_batch failed (code {rc})")
        # Copy to numpy
        def _to_np(ptr, n, dtype=np.float64):
            arr = np.ctypeslib.as_array(ptr, shape=(n,)).copy()
            return arr.astype(dtype) if arr.dtype != dtype else arr
        return {
            "copies_node": _to_np(bs_struct.copies_node, N),
            "copies_edge": _to_np(bs_struct.copies_edge, N),
            "gain_events": _to_np(bs_struct.gain_events, N),
            "loss_events": _to_np(bs_struct.loss_events, N),
            "num_families_active": _to_np(bs_struct.num_families_active, N, np.int64),
            "families_present": _to_np(bs_struct.families_present, N),
        }
    finally:
        _lib.recount_branch_stats_free(ctypes.byref(bs_struct))
        _lib.recount_survival_free(ctypes.byref(sp_struct))


_lib.recount_per_family_posteriors_batch.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_int32),    # profiles
    ctypes.c_int32,                    # F
    ctypes.c_int32,                    # num_threads
    ctypes.POINTER(ctypes.c_double),   # out_copies [F * N]
    ctypes.POINTER(ctypes.c_double),   # out_present [F * N]
]
_lib.recount_per_family_posteriors_batch.restype = ctypes.c_int


def per_family_posteriors_native(
    tree: Tree, gain, loss, dup, length, profiles: np.ndarray,
    *, num_threads: int = 0,
):
    """Per-family per-node posteriors via the native backend.

    Returns (copies, present) — both shape (F, num_nodes):
      copies[f, v]  = E[ξ̃_v | family f]
      present[f, v] = P{ξ̃_v > 0 | family f}

    Uses the same threaded inside/outside pipeline as
    ``per_branch_stats_native`` (libdispatch across all online cores),
    just emitting per-family arrays instead of summing across f.
    """
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        N = int(tree.num_nodes)
        profiles = np.ascontiguousarray(profiles, dtype=np.int32)
        F = profiles.shape[0]
        copies = np.zeros((F, N), dtype=np.float64)
        present = np.zeros((F, N), dtype=np.float64)
        nt = _resolve_num_threads(int(num_threads))
        rc = _lib.recount_per_family_posteriors_batch(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            profiles.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(F), ctypes.c_int32(nt),
            copies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            present.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"per_family_posteriors_batch failed (code {rc})")
        return copies, present
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))


def cpu_topology() -> dict:
    """Apple-silicon CPU topology — Performance / Efficiency core counts.

    Native threading via ``libdispatch`` defaults to ``sysconf(_SC_NPROCESSORS_ONLN)``
    (= ``hw.ncpu`` = P + E cores). On every Apple-silicon machine we measured,
    P + E gives slightly better wall-clock than P-only on the big focal
    workloads (the E-cores contribute ~10–15% even though they're individually
    slower), so the default num_threads=0 is the right choice on M3 Ultra
    (24P + 8E = 32 cores) just as on M4 Max (12P + 4E = 16).

    Returns a dict with keys ``ncpu``, ``physicalcpu``, ``perf_cores``,
    ``eff_cores``, ``brand``. Reads ``sysctl`` values directly; works on
    macOS only. On non-Apple platforms returns ``{"ncpu": <count>}`` only.
    """
    import platform
    if platform.system() != "Darwin":
        import os
        return {"ncpu": os.cpu_count() or 1, "brand": platform.machine()}
    import subprocess

    def _sysctl(key, cast=int):
        try:
            out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            return cast(out)
        except Exception:
            return None

    info = {
        "ncpu":        _sysctl("hw.ncpu"),
        "physicalcpu": _sysctl("hw.physicalcpu"),
        "perf_cores":  _sysctl("hw.perflevel0.physicalcpu"),
        "eff_cores":   _sysctl("hw.perflevel1.physicalcpu"),
        "brand":       _sysctl("machdep.cpu.brand_string", str),
    }
    return info


# ----------------------------------------------------------------------------
# Native BFGS optimizer — bit-faithful port of Csuros' Java dfpmin.
# See native/include/recount_bfgs.h for the C signature.
# ----------------------------------------------------------------------------

# callback: f(x) -> double, writes grad[n].  void* user data passed opaquely.
_RecountObjgradFn = ctypes.CFUNCTYPE(
    ctypes.c_double,                     # return: f(x)
    ctypes.POINTER(ctypes.c_double),     # x[n]
    ctypes.c_int,                        # n
    ctypes.POINTER(ctypes.c_double),     # grad[n] (out)
    ctypes.c_void_p,                     # user
)

_lib.recount_bfgs.argtypes = [
    ctypes.POINTER(ctypes.c_double),     # x[n] (in/out)
    ctypes.c_int,                        # n
    _RecountObjgradFn,                   # callback
    ctypes.c_void_p,                     # user
    ctypes.c_double,                     # gtol
    ctypes.c_int,                        # max_iters
    ctypes.POINTER(ctypes.c_double),     # final_obj (out)
    ctypes.POINTER(ctypes.c_int),        # iters_out
    ctypes.POINTER(ctypes.c_int),        # status_out
]
_lib.recount_bfgs.restype = ctypes.c_int


def bfgs_native(x: np.ndarray, objgrad_callable, gtol: float = 1e-7,
                max_iters: int = 200) -> tuple[np.ndarray, float, int, int]:
    """Run Csuros-style BFGS in native C.

    objgrad_callable(x: np.ndarray) -> (f: float, grad: np.ndarray)

    Returns (x_best, f_best, iters, status).
    Status codes: 0=converged on grad, 1=on Δx, 2=hit max_iters,
                  3=line search failed, 4=alloc failed.
    """
    n = x.shape[0]
    x_arr = np.ascontiguousarray(x, dtype=np.float64).copy()
    grad_buf = np.empty(n, dtype=np.float64)

    # Wrap the Python callable in a C function pointer. Per-call overhead is
    # ~1µs for the marshaling; the actual work (LL+grad) calls back into native
    # libdispatch via _gradient_raw_native.
    def _trampoline(x_ptr, _n, grad_ptr, _user):
        # View x_ptr as numpy array (read-only) — no copy.
        x_np = np.ctypeslib.as_array(x_ptr, shape=(n,))
        f, g = objgrad_callable(x_np)
        # Copy gradient into the C-provided buffer.
        np.ctypeslib.as_array(grad_ptr, shape=(n,))[:] = g
        return float(f)

    cb = _RecountObjgradFn(_trampoline)
    final_obj = ctypes.c_double(0.0)
    iters_out = ctypes.c_int(0)
    status_out = ctypes.c_int(0)
    rc = _lib.recount_bfgs(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n, cb, None,
        ctypes.c_double(gtol),
        ctypes.c_int(max_iters),
        ctypes.byref(final_obj),
        ctypes.byref(iters_out),
        ctypes.byref(status_out),
    )
    if rc != 0:
        raise RuntimeError(f"recount_bfgs failed (status={status_out.value})")
    return x_arr, float(final_obj.value), int(iters_out.value), int(status_out.value)


# ----------------------------------------------------------------------------
# Native L(0)-corrected gradient (in progress — see native/src/recount_unobserved_grad.c)
# ----------------------------------------------------------------------------

_lib.recount_unobserved_pairing.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_double),     # K_all [N, W, W]
    ctypes.c_int,                        # W
    ctypes.POINTER(ctypes.c_double),     # Lw_all [N, W, W, W]
]
_lib.recount_unobserved_pairing.restype = ctypes.c_int


def unobserved_pairing_native(tree: Tree, gain, loss, dup, length,
                              K_all: np.ndarray, min_copies: int) -> np.ndarray:
    """Pairing likelihoods L̃_w,m[ell, t] via native C.

    Bit-faithful port of compute_pairing_likelihoods() in
    recount/unobserved_outside.py — useful for incremental validation of the
    in-progress full L(0) gradient C port.
    """
    if min_copies <= 0:
        raise ValueError("min_copies must be ≥ 1")
    N = int(tree.num_nodes)
    W = int(min_copies)
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        K = np.ascontiguousarray(K_all, dtype=np.float64)
        Lw = np.full((N, W, W, W), -np.inf, dtype=np.float64)
        rc = _lib.recount_unobserved_pairing(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(W),
            Lw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_unobserved_pairing failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))
    return Lw


_lib.recount_unobserved_outside.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_double),     # K_all [N, W, W]
    ctypes.c_int,                        # min_copies (= W)
    ctypes.POINTER(ctypes.c_double),     # B_all  [W, N, W]
    ctypes.POINTER(ctypes.c_double),     # J_all  [W, N, W]
    ctypes.POINTER(ctypes.c_double),     # Bns_all [W, N, W, W]
    ctypes.POINTER(ctypes.c_double),     # Jns_all [W, N, W, W]
    ctypes.POINTER(ctypes.c_double),     # Lw_all  [N, W, W, W]
]
_lib.recount_unobserved_outside.restype = ctypes.c_int


def unobserved_outside_native(
    tree: Tree, gain, loss, dup, length,
    K_all: np.ndarray, min_copies: int,
):
    """Outside pass (B_all, J_all, Bns_all, Jns_all, Lw_all) via native C.

    Bit-faithful port of compute_unobserved_outside() in
    recount/unobserved_outside.py — feeds the posteriors / BD-tails /
    survival-gradient steps of the L(0)-corrected gradient pipeline.
    """
    if min_copies <= 0:
        raise ValueError("min_copies must be ≥ 1")
    N = int(tree.num_nodes)
    W = int(min_copies)
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        K = np.ascontiguousarray(K_all, dtype=np.float64)
        B_all   = np.full((W, N, W),   -np.inf, dtype=np.float64)
        J_all   = np.full((W, N, W),   -np.inf, dtype=np.float64)
        Bns_all = np.full((W, N, W, W), -np.inf, dtype=np.float64)
        Jns_all = np.full((W, N, W, W), -np.inf, dtype=np.float64)
        Lw_all  = np.full((N, W, W, W), -np.inf, dtype=np.float64)
        rc = _lib.recount_unobserved_outside(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(W),
            B_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            J_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Bns_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Jns_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Lw_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_unobserved_outside failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))
    return B_all, J_all, Bns_all, Jns_all, Lw_all


_lib.recount_unobserved_logL0_from_tensors.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
_lib.recount_unobserved_logL0_from_tensors.restype = ctypes.c_int

_lib.recount_unobserved_posteriors.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.recount_unobserved_posteriors.restype = ctypes.c_int

_lib.recount_unobserved_transitions.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.recount_unobserved_transitions.restype = ctypes.c_int

_lib.recount_unobserved_bd_tails.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
_lib.recount_unobserved_bd_tails.restype = ctypes.c_int

_lib.recount_unobserved_logsurv_grad.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.recount_unobserved_logsurv_grad.restype = ctypes.c_int

_lib.recount_chain_rule_logit_to_rates.argtypes = [
    ctypes.POINTER(_RecountTree),
    ctypes.POINTER(_RecountSurvival),
    ctypes.POINTER(ctypes.c_double),  # gain
    ctypes.POINTER(ctypes.c_double),  # loss
    ctypes.POINTER(ctypes.c_double),  # dup
    ctypes.POINTER(ctypes.c_double),  # length
    ctypes.POINTER(ctypes.c_double),  # d_logit_p
    ctypes.POINTER(ctypes.c_double),  # d_logit_q
    ctypes.POINTER(ctypes.c_double),  # d_log_kappa
    ctypes.POINTER(ctypes.c_double),  # out_d_gain
    ctypes.POINTER(ctypes.c_double),  # out_d_loss
    ctypes.POINTER(ctypes.c_double),  # out_d_dup
    ctypes.POINTER(ctypes.c_double),  # out_d_length
]
_lib.recount_chain_rule_logit_to_rates.restype = ctypes.c_int


def unobserved_posteriors_native(
    C_all: np.ndarray, K_all: np.ndarray,
    B_all: np.ndarray, J_all: np.ndarray,
    log_L0: float, min_copies: int,
):
    """Marginal posteriors P{ξ_v=n | unobs}, P{η_v=s | unobs}."""
    W = int(min_copies)
    N = int(K_all.shape[0])
    Cc = np.ascontiguousarray(C_all, dtype=np.float64)
    Kc = np.ascontiguousarray(K_all, dtype=np.float64)
    Bc = np.ascontiguousarray(B_all, dtype=np.float64)
    Jc = np.ascontiguousarray(J_all, dtype=np.float64)
    np_post = np.full((N, W), -np.inf, dtype=np.float64)
    ed_post = np.full((N, W), -np.inf, dtype=np.float64)
    rc = _lib.recount_unobserved_posteriors(
        ctypes.c_int(N), ctypes.c_int(W),
        Cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Kc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Jc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(log_L0),
        np_post.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ed_post.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise RuntimeError(f"recount_unobserved_posteriors failed (code {rc})")
    return np_post, ed_post


def unobserved_transitions_native(
    C_all: np.ndarray, K_all: np.ndarray,
    Bns_all: np.ndarray, Jns_all: np.ndarray,
    log_L0: float, min_copies: int,
):
    """Joint transition posteriors at the node and across edge entering v."""
    W = int(min_copies)
    N = int(K_all.shape[0])
    Cc = np.ascontiguousarray(C_all, dtype=np.float64)
    Kc = np.ascontiguousarray(K_all, dtype=np.float64)
    Bnsc = np.ascontiguousarray(Bns_all, dtype=np.float64)
    Jnsc = np.ascontiguousarray(Jns_all, dtype=np.float64)
    node_trans = np.full((N, W, W), -np.inf, dtype=np.float64)
    edge_trans = np.full((N, W, W), -np.inf, dtype=np.float64)
    rc = _lib.recount_unobserved_transitions(
        ctypes.c_int(N), ctypes.c_int(W),
        Cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Kc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Bnsc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Jnsc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(log_L0),
        node_trans.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edge_trans.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise RuntimeError(f"recount_unobserved_transitions failed (code {rc})")
    return node_trans, edge_trans


def unobserved_bd_tails_native(
    log_node_trans: np.ndarray, log_edge_trans: np.ndarray, min_copies: int,
):
    """Per-node birth/death tail differences from joint posteriors."""
    W = int(min_copies)
    N = int(log_node_trans.shape[0])
    nt = np.ascontiguousarray(log_node_trans, dtype=np.float64)
    et = np.ascontiguousarray(log_edge_trans, dtype=np.float64)
    birth = np.full((N, W), -np.inf, dtype=np.float64)
    death = np.full((N, W), -np.inf, dtype=np.float64)
    rc = _lib.recount_unobserved_bd_tails(
        ctypes.c_int(N), ctypes.c_int(W),
        nt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        et.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        birth.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        death.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise RuntimeError(f"recount_unobserved_bd_tails failed (code {rc})")
    return birth, death


def chain_rule_logit_to_rates_native(
    tree: Tree, gain, loss, dup, length,
    d_logit_p: np.ndarray, d_logit_q: np.ndarray, d_log_kappa: np.ndarray,
):
    """Reverse-mode chain rule: (logit_p, logit_q, log_κ) adjoints →
    (gain, loss, dup, length) gradients per node. Native C — replaces
    torch autograd through compute_survival_params_t in
    compute_L0_gradient_analytical.
    """
    N = int(tree.num_nodes)
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        g_arr = np.ascontiguousarray(gain,   dtype=np.float64)
        l_arr = np.ascontiguousarray(loss,   dtype=np.float64)
        d_arr = np.ascontiguousarray(dup,    dtype=np.float64)
        t_arr = np.ascontiguousarray(length, dtype=np.float64)
        dp = np.ascontiguousarray(d_logit_p, dtype=np.float64)
        dq = np.ascontiguousarray(d_logit_q, dtype=np.float64)
        dk = np.ascontiguousarray(d_log_kappa, dtype=np.float64)
        d_gain   = np.zeros(N, dtype=np.float64)
        d_loss   = np.zeros(N, dtype=np.float64)
        d_dup    = np.zeros(N, dtype=np.float64)
        d_length = np.zeros(N, dtype=np.float64)
        rc = _lib.recount_chain_rule_logit_to_rates(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            l_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_gain.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_dup.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_length.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_chain_rule_logit_to_rates failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))
    return d_gain, d_loss, d_dup, d_length


def unobserved_logsurv_grad_native(
    tree: Tree, gain, loss, dup, length,
    log_edge_post: np.ndarray, log_birth_tails: np.ndarray,
    log_death_tails: np.ndarray, profile_count: float, min_copies: int,
):
    """Per-node log-survival gradient (logit p̃, logit q̃, log κ)."""
    N = int(tree.num_nodes)
    W = int(min_copies)
    tree_struct, _h_tree, sp_struct, _h_rates = _compute_survival(
        tree, gain, loss, dup, length)
    try:
        ep = np.ascontiguousarray(log_edge_post, dtype=np.float64)
        bt = np.ascontiguousarray(log_birth_tails, dtype=np.float64)
        dt = np.ascontiguousarray(log_death_tails, dtype=np.float64)
        d_logit_p   = np.zeros(N, dtype=np.float64)
        d_logit_q   = np.zeros(N, dtype=np.float64)
        d_log_kappa = np.zeros(N, dtype=np.float64)
        rc = _lib.recount_unobserved_logsurv_grad(
            ctypes.byref(tree_struct), ctypes.byref(sp_struct),
            ctypes.c_int(W),
            ep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(float(profile_count)),
            d_logit_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_logit_q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_log_kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"recount_unobserved_logsurv_grad failed (code {rc})")
    finally:
        _lib.recount_survival_free(ctypes.byref(sp_struct))
    return d_logit_p, d_logit_q, d_log_kappa


__all__ = [
    "log_likelihood_native",
    "corrected_log_likelihood_native",
    "gradient_native",
    "per_branch_stats_native",
    "per_family_posteriors_native",
    "bfgs_native",
    "unobserved_pairing_native",
    "unobserved_outside_native",
    "unobserved_posteriors_native",
    "unobserved_transitions_native",
    "unobserved_bd_tails_native",
    "unobserved_logsurv_grad_native",
    "chain_rule_logit_to_rates_native",
    "native_version",
    "cpu_topology",
]
